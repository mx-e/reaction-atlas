"""CRN exploration worker — runs on Cloud Batch GPU instances.

Each worker independently pulls work from PostgreSQL and writes results back.
This replaces the SLURM-based explore_hydra.py with DB-backed coordination.
"""

import os
import random
import signal
import time
import uuid
from functools import partial
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from loguru import logger

from lib.compound import Compound, get_smiles_from_conformer
from lib.db import DB
from lib.energy import create_energy_fn
from lib.exploration import ExplorationContext, create_single_mol_context
from lib.fragment_mols import load_fragment_conformers, populate_context_fragments, relax_conformer
from lib.md_et_calculator import get_md_et_calculator
from lib.pes_explorer import ExploreConfig
from lib.reaction_graph import ReactionGraph
from lib.ts_model import get_ts_model
from lib.ts_pipeline import explore_and_process_batch
from lib.types import Conformer


class SpotPreemptionError(Exception):
    """Raised by SIGTERM handler to immediately unwind the call stack on spot preemption."""
    pass


def _handle_sigterm(signum, frame):
    """Handle SIGTERM from GCP spot preemption (30s warning).

    Raises SpotPreemptionError to immediately cancel the current GPU operation
    and unwind to the main loop, where claimed work is released back to pending.
    """
    logger.warning("SIGTERM received — spot preemption. Cancelling current work.")
    raise SpotPreemptionError()


def get_config():
    """Load exploration config from environment variables."""
    # EXPERIMENT is required: workers MUST declare which experiment they're
    # writing into. No default — a typo / missing env var must fail loudly
    # rather than silently default to 'main' and cross-contaminate scopes.
    experiment = os.environ.get("EXPERIMENT")
    if not experiment:
        raise RuntimeError(
            "EXPERIMENT env var is required (e.g. 'main' or 'formose-drilldown'). "
            "Refusing to start a worker without an explicit scope."
        )
    from packages.db.experiments import EXPERIMENTS
    if experiment not in EXPERIMENTS:
        raise RuntimeError(
            f"Unknown EXPERIMENT '{experiment}'. Known: {sorted(EXPERIMENTS)}. "
            f"Add it to packages/db/experiments.py if it's intentional."
        )
    return {
        "database_url": os.environ["DATABASE_URL"],
        "experiment": experiment,
        # Closed-subgraph mode: drop reactions whose endpoints aren't already
        # tagged with this experiment. formose-drilldown sets this to true so
        # the worker only finds new TSs between the user-curated nodes.
        "restrict_to_existing_compounds": os.environ.get(
            "RESTRICT_TO_EXISTING_COMPOUNDS", "false"
        ).lower() == "true",
        "max_valid_nodes": int(os.environ.get("MAX_VALID_NODES", "1000")),
        "ts_batch_size": int(os.environ.get("TS_BATCH_SIZE", "32")),
        "max_denoising_steps": int(os.environ.get("MAX_DENOISING_STEPS", "1000")),
        "noise_min": int(os.environ.get("NOISE_MIN", "100")),
        "noise_max": int(os.environ.get("NOISE_MAX", "650")),
        "relaxation_fmax": float(os.environ.get("RELAXATION_FMAX", "0.005")),
        "pes_backlog_tolerance": float(os.environ.get("PES_BACKLOG_TOLERANCE", "0.5")),
        "pes_temperature": float(os.environ.get("PES_TEMPERATURE", "300")),
        "pes_md_steps": int(os.environ.get("PES_MD_STEPS", "500")),
        "pes_max_iterations": int(os.environ.get("PES_MAX_ITERATIONS", "10")),
        # Default True per upstream explore_hydra.py — accepts valid TSs
        # where IRC endpoints don't match the original input reactants but
        # still describe a valid reaction. Without this, ~85% of valid
        # diffusion-generated TSs get rejected by _validate_endpoints.
        "greedy_merge": os.environ.get("GREEDY_MERGE", "true").lower() == "true",
        "greedy_single": os.environ.get("GREEDY_SINGLE", "true").lower() == "true",
        "ts_model_path": os.environ.get("TS_MODEL_PATH", "/app/models"),
        "model_type": os.environ.get("MODEL_TYPE", "MoreRedJT"),
        "forces_model_path": os.environ.get("FORCES_MODEL_PATH", "/app/models"),
        "energy_model_path": os.environ.get("ENERGY_MODEL_PATH", "/app/models"),
        "start_xyz_path": os.environ.get("START_XYZ_PATH", "/app/data/start.xyz"),
        "fragment_path": os.environ.get("FRAGMENT_PATH", "/app/data/fragments"),
        "buffer_fragment_path": os.environ.get("BUFFER_FRAGMENT_PATH", "/app/data/buffer_fragments"),
        "noise_schedule": os.environ.get("NOISE_SCHEDULE", "uniform"),  # "uniform" or "decreasing"
        "compound_carbon_cap": int(os.environ.get("COMPOUND_CARBON_CAP", "5")),
        "oversample_factor": int(os.environ.get("OVERSAMPLE_FACTOR", "50")),
        "max_carbon_count": int(os.environ["MAX_CARBON_COUNT"]) if os.environ.get("MAX_CARBON_COUNT") else None,
        # PES exploration gating: wait + concentration thresholds.
        # Disabled (0.0) by default for back-compat; set > 0 to enable.
        "pes_min_compound_age_s": float(os.environ.get("PES_WAIT_FOR_KINETICS_S", "0.0")),
        "pes_min_conc_m": float(os.environ.get("PES_MIN_CONC_M", "0.0")),
        # PES_DEDUP_ONLY=1 → only claim job_kind='dedup', skip the concentration
        # gate (irrelevant for dedup), and skip the generative TS batch step
        # (this worker exists purely to drain the dedup queue).
        "pes_dedup_only": os.environ.get("PES_DEDUP_ONLY", "").lower() in ("1", "true", "yes"),
        "kinetic_sampling_enabled": os.environ.get("KINETIC_SAMPLING_ENABLED", "true").lower() == "true",
    }


def explore_single_minimum(
    compound: Compound,
    conformer_id: int,
    forces_model_calculator,
    explore_config: ExploreConfig,
) -> tuple[Compound, bool, list[dict], int, dict]:
    """Explore PES from a single minimum, returning escaped reactions.

    Args:
        compound: Compound with full PES graph (for dedup context).
        conformer_id: PES graph local ID of the minimum to explore.

    Returns:
        (compound, success, escaped_validations, n_new_ts, timings)
    """
    forces_model_calculator.reset()
    logger.info(f"Exploring minimum {conformer_id} of {compound.formula}")
    n_new_ts, escaped, timings = compound.explore_pes(
        conformer_id=conformer_id,
        calc=forces_model_calculator,
        config=explore_config,
        verbose=True,
    )
    logger.info(
        f"Found {n_new_ts} new intramolecular TSs from minimum {conformer_id}"
        + (f", {len(escaped)} escaped reactions" if escaped else "")
    )
    return compound, True, escaped, n_new_ts, timings


def _process_escaped_reactions(
    escaped_validations: list[dict],
    source_compound: Compound,
    energy_model,
    device: str,
    graph: ReactionGraph,
    calc=None,
) -> int:
    """Process escaped reactions from PES exploration.

    Imported from chem_graph_explorer but adapted for DB-backed graph.
    """
    from lib.compound import (
        get_smiles_from_structure,
        get_charge_from_smiles,
        canonicalize_structure,
    )

    if not escaped_validations:
        return 0

    atomic_numbers = np.array(source_compound.sorted_atomic_numbers)
    source_charge = get_charge_from_smiles(source_compound.smiles)
    valid_contexts = []
    energy_fn = create_energy_fn(energy_model)

    for vi, validation in enumerate(escaped_validations):
      try:
        fwd_smiles = get_smiles_from_structure(atomic_numbers, validation["fwd_positions"])
        bwd_smiles = get_smiles_from_structure(atomic_numbers, validation["bwd_positions"])

        if fwd_smiles is None or bwd_smiles is None:
            logger.info(
                f"Escaped reaction: SMILES determination failed "
                f"(fwd={fwd_smiles}, bwd={bwd_smiles}). Skipping."
            )
            continue

        if fwd_smiles == source_compound.smiles:
            reactant_key, product_key = "fwd", "bwd"
        elif bwd_smiles == source_compound.smiles:
            reactant_key, product_key = "bwd", "fwd"
        else:
            logger.warning(
                f"Escaped reaction: neither endpoint matches source compound "
                f"({source_compound.smiles}). fwd={fwd_smiles}, bwd={bwd_smiles}. Skipping."
            )
            continue

        # Warn when fwd/bwd energies are nearly equal
        energy_diff = abs(validation["fwd_energy"] - validation["bwd_energy"])
        if energy_diff < 1e-4:
            logger.warning(
                f"Escaped reaction: fwd/bwd energy difference is very small "
                f"({energy_diff:.2e} eV) — reference minimum choice is nearly arbitrary"
            )

        _, _, sort_indices = canonicalize_structure(
            atomic_numbers, validation[f"{reactant_key}_positions"]
        )
        canonical_anum = atomic_numbers[sort_indices]

        reactant_pos = validation[f"{reactant_key}_positions"][sort_indices]
        product_pos = validation[f"{product_key}_positions"][sort_indices]
        ts_pos = validation["ts_positions"][sort_indices]

        reactant_positions = torch.tensor(reactant_pos, dtype=torch.float64)
        product_positions = torch.tensor(product_pos, dtype=torch.float64)
        ts_positions = torch.tensor(ts_pos, dtype=torch.float64)
        anum_tensor = torch.tensor(canonical_anum, dtype=torch.long)

        start_conformer = Conformer(
            positions=reactant_positions,
            atomic_numbers=anum_tensor,
            charge=source_charge,
        )
        start_conformer.energy = energy_fn(start_conformer)

        ctx = create_single_mol_context(
            start_conformer=start_conformer,
            source_compound_smiles=source_compound.smiles,
        )
        ctx.discovery_method = "pes_exploration"
        ctx.discovery_timestamp = time.time()

        ts_conformer = Conformer(
            positions=ts_positions,
            atomic_numbers=anum_tensor,
            charge=source_charge,
        )
        ctx.ts_conformer = ts_conformer
        ctx.ts_id = ExplorationContext.compute_id(ts_conformer)
        ctx.ts_energy = energy_fn(ts_conformer)

        product_conformer = Conformer(
            positions=product_positions,
            atomic_numbers=anum_tensor,
            charge=source_charge,
        )
        ctx.min_conformer = product_conformer
        ctx.min_id = ExplorationContext.compute_id(product_conformer)
        ctx.min_energy = energy_fn(product_conformer)

        # Trajectories
        from lib.pes_explorer.pes_graph import RelaxationTrajectory

        def _permute_trajectory(traj, sort_idx):
            if traj is None:
                return None
            idx_3n = np.concatenate([[3 * i, 3 * i + 1, 3 * i + 2] for i in sort_idx])

            def _perm_h(h):
                return h[np.ix_(idx_3n, idx_3n)] if h is not None else None

            return RelaxationTrajectory(
                positions=[pos[sort_idx] for pos in traj.positions],
                energies=traj.energies,
                forces=[f[sort_idx] for f in traj.forces],
                hessians=[_perm_h(h) for h in traj.hessians],
            )

        ctx.reactant_trajectory = _permute_trajectory(
            validation.get(f"{reactant_key}_trajectory"), sort_indices
        )
        ctx.product_trajectory = _permute_trajectory(
            validation.get(f"{product_key}_trajectory"), sort_indices
        )

        # Validate barriers
        is_valid_fwd, reason_fwd, stat_key_fwd = ctx.validate_forward_barrier(graph.energy_threshold)
        if not is_valid_fwd:
            logger.debug(f"Escaped reaction rejected (forward): {reason_fwd}")
            graph.energy_validation_stats[stat_key_fwd] = (
                graph.energy_validation_stats.get(stat_key_fwd, 0) + 1
            )
            continue

        success = populate_context_fragments(
            ctx=ctx, energy_fn=energy_fn, calc=calc,
            stats_tracker=graph.decomposition_stats,
        )
        if not success:
            logger.debug("Escaped reaction rejected: invalid fragmentation")
            continue

        is_valid_bwd, reason_bwd, stat_key_bwd = ctx.validate_backward_barrier(graph.energy_threshold)
        if not is_valid_bwd:
            logger.debug(f"Escaped reaction rejected (backward): {reason_bwd}")
            graph.energy_validation_stats[stat_key_bwd] = (
                graph.energy_validation_stats.get(stat_key_bwd, 0) + 1
            )
            continue

        valid_contexts.append(ctx)
        logger.info(
            f"Valid escaped reaction from {source_compound.smiles}: "
            f"barrier_fwd={ctx.get_ts_barrier_forward():.3f} eV"
        )
      except Exception as e:
        logger.warning(
            f"Escaped reaction {vi+1}/{len(escaped_validations)} failed: "
            f"{type(e).__name__}: {e}"
        )
        continue

    if valid_contexts:
        n_committed = graph.add_contexts_to_graph(valid_contexts, calc=calc)
        return n_committed

    return 0


def _seed_initial_compounds(
    db: DB, ts_model, energy_model, forces_calculator,
    device: str, start_xyz_path: str, fragment_path: str,
    buffer_fragment_path: str,
    relaxation_fmax: float = 0.005,
):
    """Seed starting compounds into the DB. Uses a PostgreSQL advisory lock
    so only the first worker to arrive does the seeding; others wait and proceed.

    Also seeds the manual buffer-equilibrium reactions (water autoionization,
    CO2 hydration, etc.) — required for the kinetic ODE simulation to converge
    to physical concentrations. The diffusion model can't discover acid-base
    chemistry, so we layer these in with literature rate constants. See
    packages/worker/lib/seed_equilibria.py for the constants.
    """
    from sqlalchemy import text
    from lib.energy import create_energy_fn
    from lib.seed_equilibria import seed_buffer_equilibria

    with db.session() as session:
        # pg_advisory_lock(42) — blocks other workers until released at commit
        session.execute(text("SELECT pg_advisory_lock(42)"))
        try:
            # Check if already seeded
            n = db.get_compound_count(session)
            if n > 0:
                logger.info(f"DB already has {n} compounds, skipping start molecule + fragment seed")
                # Buffer equilibria are still idempotent — re-run in case a
                # previous worker died mid-seed before getting to this step.
                # Also migrates any zero-position placeholder Minima from
                # earlier seed runs that didn't load XYZ geometries.
                seed_buffer_equilibria(session, Path(buffer_fragment_path))
                return

            logger.info("First worker — seeding initial compounds")

            # Load starting molecule
            from ase.io import read as ase_read
            start_atoms = ase_read(start_xyz_path)
            start_mol = ts_model.model.prepare_inputs([start_atoms])
            start_conformer = Conformer.from_batch(start_mol)

            energy_fn = create_energy_fn(energy_model)

            # Load fragments
            fragment_conformers = load_fragment_conformers(
                fragment_path=Path(fragment_path),
                prepare_inputs_fn=ts_model.model.prepare_inputs,
                energy_fn=energy_fn,
                device=device,
            )
            logger.info(f"Loaded {len(fragment_conformers)} fragment conformers")

            # Relax and register all initial conformers
            all_conformers = [start_conformer] + fragment_conformers
            from lib.reaction_graph import ReactionGraph
            graph = ReactionGraph(db=db)

            for conformer in all_conformers:
                relaxed = relax_conformer(conformer, forces_calculator, fmax=relaxation_fmax)
                if relaxed is not None:
                    conformer = relaxed
                conformer.energy = energy_fn(conformer)

                result = graph.register_compound(session, conformer, conformer.energy, is_seed=True, calc=forces_calculator)
                if result is not None:
                    info, _, is_new = result
                    if is_new:
                        logger.info(f"Seeded compound: {info.formula} ({info.smiles})")

            # Now seed the manual buffer equilibria (water autoionization etc.).
            # MUST happen after the fragment seeding above so that the natural
            # buffer compounds (water, [HH]) are present in the DB first.
            seed_buffer_equilibria(session, Path(buffer_fragment_path))

            # Compute ML energies for all seed compounds that still have the
            # placeholder energy=0.0. Buffer compounds are seeded from raw XYZ
            # files without an energy model — fix that now so the E-diagram and
            # analysis views show real energies.
            from packages.db.models import Compound as CompoundRow, Minimum as MRow
            from packages.db.serialization import deserialize_ndarray as _deser, serialize_ndarray as _ser
            energy_fn = create_energy_fn(energy_model)
            seed_minima = (
                session.query(MRow)
                .join(CompoundRow, MRow.compound_id == CompoundRow.id)
                .filter(CompoundRow.is_seed == True, MRow.energy == 0.0)
                .all()
            )
            for m in seed_minima:
                compound = session.query(CompoundRow).filter(CompoundRow.id == m.compound_id).first()
                if compound is None:
                    continue
                try:
                    anum = _deser(compound.sorted_atomic_numbers).flatten()
                    pos = _deser(m.positions)
                    conf = Conformer(
                        positions=torch.tensor(pos, dtype=torch.float32),
                        atomic_numbers=torch.tensor(anum, dtype=torch.long),
                    )
                    e = float(energy_fn(conf))
                    m.energy = e
                    # Also update compound-level energy cache
                    if compound.energy_pbe0 is None:
                        # No PBE0 yet — the ML energy is the best we have
                        pass
                    logger.info(f"Computed ML energy for {compound.smiles}: {e:.4f} eV")
                except Exception as ex:
                    logger.warning(f"Failed to compute energy for compound {m.compound_id}: {ex}")

            logger.info(f"Seeding complete, {db.get_compound_count(session)} compounds")
        finally:
            session.execute(text("SELECT pg_advisory_unlock(42)"))


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)

    worker_id = f"worker-{uuid.uuid4().hex[:8]}"
    config = get_config()
    logger.info(
        f"Starting worker {worker_id} (experiment={config['experiment']}, "
        f"restrict_to_existing_compounds={config['restrict_to_existing_compounds']})"
    )

    # Initialize DB scoped to this worker's experiment.
    db = DB(
        database_url=config["database_url"],
        experiment=config["experiment"],
    )
    db.init_db()

    # Initialize models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    ts_model = get_ts_model(
        config["ts_model_path"], config["model_type"],
        dict(
            time_key="t",
            cutoff=5.0,
            recompute_neighbors=True,
            save_progress=True,
            progress_stride=1,
            results_on_cpu=True,
        ),
        T=1000, s=1e-5, dtype=torch.float64, variance_type="lower_bound",
    )
    forces_calculator = get_md_et_calculator(
        Path(config["forces_model_path"]), device, type="standard", filter_forces=True,
        hessian_batch_size=8,
    )
    pes_calculator = get_md_et_calculator(
        Path(config["forces_model_path"]), device, type="standard", filter_forces=True,
        hessian_batch_size=8,
    )
    from lib.energy import get_energy_model
    energy_model = get_energy_model(config["energy_model_path"])

    # Initialize graph
    graph = ReactionGraph(
        db=db,
        rmsd_ts_threshold=0.3,
        compound_carbon_cap=config["compound_carbon_cap"],
        kinetic_sampling_enabled=config["kinetic_sampling_enabled"],
        oversample_factor=config["oversample_factor"],
        restrict_to_existing_compounds=config["restrict_to_existing_compounds"],
    )

    explore_config = ExploreConfig(
        md_steps=config["pes_md_steps"],
        temperature=config["pes_temperature"],
        max_iterations=config["pes_max_iterations"],
    )

    batch_explorer = partial(
        explore_and_process_batch,
        max_denoising_steps=config["max_denoising_steps"],
        ts_model=ts_model,
        energy_model=energy_model,
        forces_model_calculator=forces_calculator,
        graph=graph,
        device=device,
        greedy_merge=config["greedy_merge"],
        greedy_single=config["greedy_single"],
        relaxation_fmax=config["relaxation_fmax"],
    )

    noise_schedule = config.get("noise_schedule", "uniform")
    if noise_schedule == "uniform":
        noise_fn = lambda: random.randint(config["noise_min"], config["noise_max"])
    elif noise_schedule == "decreasing":
        # Linearly decrease noise range as exploration progresses
        def noise_fn():
            with db.session() as s:
                n = db.get_compound_count(s)
            progress = min(1.0, n / config["max_valid_nodes"])
            hi = int(config["noise_max"] - progress * (config["noise_max"] - config["noise_min"]) * 0.5)
            return random.randint(config["noise_min"], hi)
    else:
        noise_fn = lambda: random.randint(config["noise_min"], config["noise_max"])
    batch_count = 0
    explored_contexts = 0

    # Seed starting compounds (first worker only, uses advisory lock).
    # Skipped under closed-subgraph mode: the experiment's compound set is
    # already populated (e.g. by Migration B for formose-drilldown), and
    # seeding would inject the canonical start molecule + fragments which
    # are not necessarily in scope for this experiment.
    if config["restrict_to_existing_compounds"]:
        logger.info(
            f"Skipping seed loop — restrict_to_existing_compounds=True "
            f"(experiment={config['experiment']})"
        )
    else:
        _seed_initial_compounds(
            db=db,
            ts_model=ts_model,
            energy_model=energy_model,
            forces_calculator=forces_calculator,
            device=device,
            start_xyz_path=config["start_xyz_path"],
            fragment_path=config["fragment_path"],
            buffer_fragment_path=config["buffer_fragment_path"],
            relaxation_fmax=config["relaxation_fmax"],
        )

    logger.info("Entering main exploration loop")

    pes_count = 0
    total_wall = 0.0

    def _heartbeat(status="idle", task=None):
        with db.session() as s:
            db.heartbeat(s, worker_id, "exploration", status, task,
                         batches_completed=batch_count, pes_completed=pes_count,
                         total_wall_time_s=total_wall)

    _heartbeat("idle")

    dedup_only = config["pes_dedup_only"]

    while not graph.is_exploration_complete(config["max_valid_nodes"]):
        # 1. Try PES exploration (per-minimum, compound-locked).
        # In dedup-only mode the concentration gate is bypassed (it exists to
        # avoid spending exploration GPU on low-conc compounds; dedup is cheap
        # and we want to resolve temp minima regardless).
        work = graph.get_compound_to_postprocess(
            config["pes_backlog_tolerance"],
            min_compound_age_s=0.0 if dedup_only else config["pes_min_compound_age_s"],
            min_conc=0.0 if dedup_only else config["pes_min_conc_m"],
            job_kind_filter="dedup" if dedup_only else None,
        )
        if work is not None:
            job_kind = work.get("job_kind", "explore")
            logger.info(
                f"Claimed PES work ({job_kind}): compound_id={work['compound_id']} "
                f"minimum_id={work['minimum_id']}"
            )

            # Dedup jobs: resolve a temp (negative-local_id) minimum against
            # the compound's PES graph using proper RMSD + NEB dedup.
            if job_kind == "dedup":
                _heartbeat("pes", f"dedup compound {work['compound_id']}")
                try:
                    graph.handle_dedup_job(
                        work["compound_id"], work["minimum_id"],
                        work["work_id"], calc=pes_calculator,
                    )
                    _heartbeat("idle")
                except SpotPreemptionError:
                    graph.release_compound_postprocessing(work["work_id"])
                    return
                except Exception as e:
                    logger.error(f"Dedup job failed: {e}")
                    graph.complete_compound_postprocessing(work["work_id"], failed=True)
                continue

            _heartbeat("pes", f"compound {work['compound_id']}")
            pes_start = time.perf_counter()

            result = graph.load_compound_for_pes(work["compound_id"], work["minimum_id"])
            if result is None:
                graph.complete_compound_postprocessing(work["work_id"], failed=True)
                continue

            compound, minimum_db_id, conformer_id = result

            # Already explored (e.g. by a checkpoint from a previous attempt)
            if compound.pes_graph.minima[conformer_id].explored:
                logger.info(f"Minimum {work['minimum_id']} already explored, skipping")
                graph.complete_compound_postprocessing(work["work_id"])
                continue

            try:
                updated, success, escaped, n_ts, timings = explore_single_minimum(
                    compound, conformer_id, pes_calculator, explore_config,
                )
                # Save results (persists new minima/TSs, marks this minimum explored)
                graph.save_pes_results(work["compound_id"], updated, minimum_db_id)
                graph.complete_compound_postprocessing(work["work_id"], failed=not success)

                # Process escaped reactions
                n_escaped_valid = 0
                if escaped:
                    n_escaped_valid = _process_escaped_reactions(
                        escaped, updated, energy_model, device, graph, calc=forces_calculator,
                    )

                pes_wall = time.perf_counter() - pes_start
                total_wall += pes_wall
                pes_count += 1
                graph.record_pes_exploration_stats(
                    n_intramol_ts=n_ts,
                    n_escaped=len(escaped),
                    n_escaped_valid=n_escaped_valid,
                    wall_time_s=pes_wall,
                    step_timings=timings,
                )
                _heartbeat("idle")
            except SpotPreemptionError:
                logger.warning(f"Preempted during PES work {work['work_id']} — discarding partial results, releasing")
                graph.release_compound_postprocessing(work["work_id"])
                logger.info("Work released. Shutting down.")
                return
            except Exception as e:
                logger.error(f"PES exploration failed: {e}")
                graph.complete_compound_postprocessing(work["work_id"], failed=True)
            continue

        # 2. Generative TS batch exploration.
        # Dedup-only workers exist purely to drain the dedup queue — if there
        # was no dedup work, idle-poll instead of starting a generative round.
        if dedup_only:
            time.sleep(10)
            continue

        try:
            batch = graph.sample_exploration_batch(
                max_contexts=config["ts_batch_size"],
                max_carbon_count=config["max_carbon_count"],
            )
            if not batch:
                logger.info("No contexts to explore, waiting...")
                time.sleep(10)
                continue

            logger.info(f"Round {batch_count}: exploring {len(batch)} contexts")
            batch_count += 1
            explored_contexts += len(batch)

            t = noise_fn()
            logger.info(f"Using noise level t={t}")
            gen_start = time.perf_counter()
            _heartbeat("generative", f"round {batch_count - 1}, t={t}")

            batch_summary = batch_explorer(contexts=batch, t=t)
            total_wall += time.perf_counter() - gen_start
            if batch_summary is not None:
                batch_summary["batch_idx"] = batch_count - 1
                batch_summary["worker_id"] = worker_id
                # Per-batch kinetic sampling stats (Phase 5) — attached by
                # ReactionGraph._sample_batch on the last call.
                if getattr(graph, "_last_sampling_stats", None):
                    batch_summary["sampling"] = graph._last_sampling_stats
                graph.record_batch_stats(batch_summary)
            _heartbeat("idle")
        except SpotPreemptionError:
            # TS batch has no claimed work to release — just exit
            logger.warning("Preempted during TS batch — no work to release. Shutting down.")
            with db.session() as s:
                db.remove_heartbeat(s, worker_id)
            return

    # Clean up heartbeat on normal exit
    with db.session() as s:
        db.remove_heartbeat(s, worker_id)

    # ── Post-exploration analysis ──
    logger.info(
        f"Exploration complete: {batch_count} batches, "
        f"{explored_contexts} contexts explored"
    )

    from lib.graph_analysis import (
        analyze_decompositions,
        print_decomposition_statistics,
        print_edge_energies,
        print_energy_validation_statistics,
        print_exploration_statistics,
    )
    print_edge_energies(graph)
    analyze_decompositions(graph)
    print_decomposition_statistics(graph)
    print_energy_validation_statistics(graph)
    print_exploration_statistics(graph)

    with db.session() as session:
        from sqlalchemy import func as sa_func
        from packages.db.models import Compound as CRow

        n_compounds = session.query(sa_func.count(CRow.id)).scalar()
        logger.info(f"Final compound count: {n_compounds}")

        # Verify starting compounds are still in graph
        seed_compounds = session.query(CRow).filter(CRow.is_seed == True).all()
        for sc in seed_compounds:
            logger.info(f"Seed compound {sc.smiles} ({sc.formula}) present in DB: OK")
        if not seed_compounds:
            logger.error("WARNING: No seed compounds found in DB!")


if __name__ == "__main__":
    main()
