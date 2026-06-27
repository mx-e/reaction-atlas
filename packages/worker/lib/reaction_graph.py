"""
ReactionGraph: DB-backed graph of chemical reactions between compounds.

Refactored from the pickle-based version to use PostgreSQL via lib/db.py.
The core validation logic (add_reaction_from_context, two-phase validate-then-commit)
is preserved; only the persistence layer changes.
"""

import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
import torch
from ase.calculators.calculator import Calculator
from loguru import logger

from lib.compound import (
    Compound,
    get_smiles_from_conformer,
    get_smiles_and_charge_from_conformer,
    get_sorted_atomic_numbers,
    get_chemical_formula,
    canonicalize_structure,
)
from lib.types import Conformer
from lib.exploration import (
    ExplorationContext,
    create_single_mol_context,
)
from lib.utils import rmsd_between_conformers
from lib.merge_mols import merge_conformers_to_context
from lib.naming import NameGenerator
from lib.pes_explorer import ExploreConfig, PESGraph
from lib.pes_explorer.pes_graph import RelaxationTrajectory, Minimum
from lib.constants import ENERGY_THRESHOLD_EV
from lib.db import DB
from packages.db.models import Minimum as MinimumRow
from packages.db.serialization import (
    serialize_ndarray,
    deserialize_ndarray,
    deserialize_ndarray_optional,
)


@dataclass(frozen=True)
class ReactantEntry:
    """A single reactant in a reaction."""
    smiles: str
    conformer_id: Optional[int] = None


@dataclass(frozen=True)
class ProductEntry:
    """A single product in a reaction."""
    smiles: str
    conformer_id: int
    energy: float


class ReactionGraph:
    """
    DB-backed graph of chemical reactions between compounds.

    Replaces the pickle + filelock version. Workers call methods that
    open DB transactions internally — no external locking needed.
    """

    def __init__(
        self,
        db: DB,
        rmsd_ts_threshold: float = 0.3,
        energy_threshold: float = ENERGY_THRESHOLD_EV,
        fragment_threshold: int = 3,
        compound_carbon_cap: Optional[int] = None,
        kinetic_sampling_enabled: bool = False,
        # Merge-pair sampling: draw `oversample_factor * n_needed`
        # candidate pairs up front, filter them by composition limit +
        # per-pair reaction cap in one pass, take the first n_needed
        # survivors.  Replaces the old nested retry loops which were
        # wasting cycles reconstructing conformers for pairs that always
        # failed check_tetrose_limit (~78% of C4-heavy draws).
        oversample_factor: int = 50,
        # When True, register_compound rejects any compound that isn't
        # already tagged with this worker's experiment — drops the entire
        # reaction rather than expanding the experiment's compound set.
        # Used by formose-drilldown to keep the subgraph closed: discover
        # new reactions between selected nodes, but never add nodes.
        restrict_to_existing_compounds: bool = False,
    ):
        self.db = db
        self.rmsd_ts_threshold = rmsd_ts_threshold
        self.energy_threshold = energy_threshold
        self.fragment_threshold = fragment_threshold
        self.compound_carbon_cap = compound_carbon_cap
        self.oversample_factor = oversample_factor
        self.restrict_to_existing_compounds = restrict_to_existing_compounds
        # When True, _sample_batch fetches the latest KineticsSnapshot from
        # the DB once per batch and uses entropy-weighted decade + log-conc
        # pair sampling for merge contexts. Falls back to uniform sampling
        # if no snapshot is available. Set via env KINETIC_SAMPLING_ENABLED
        # in worker.py:get_config().
        self.kinetic_sampling_enabled = kinetic_sampling_enabled
        self._last_sampling_stats: dict = {}
        self._namer = NameGenerator()
        self._merge_namer = NameGenerator()
        self._min_namer = NameGenerator()

        # In-memory stats that get flushed to DB periodically
        self.decomposition_stats = {
            "attempted": 0,
            "successful": 0,
            "rejected_invalid_fragments": 0,
            "single_component": 0,
        }
        self.energy_validation_stats = {}

    # =========================================================================
    # Compound Registration (DB-backed)
    # =========================================================================

    def register_compound(
        self, session, conformer: Conformer, energy: float, is_seed: bool = False,
        calc=None,
    ) -> Optional[tuple["CompoundInfo", int, bool]]:
        """Register a conformer as a compound in the DB.

        Returns (CompoundInfo, conformer_local_id, is_new_compound) or None.

        When `self.restrict_to_existing_compounds=True`, compounds outside
        this experiment's existing explorable set are registered as
        "frontier": tagged with the experiment so the graph view shows
        them, plus added to `frontier_in` so the sampling query skips
        them. The worker doesn't widen its sampling pool but still records
        the boundary chemistry it discovers (a partial reaction landing on
        a non-formose compound is still a real reaction, worth keeping).
        """
        result = get_smiles_and_charge_from_conformer(conformer)
        if result is None:
            logger.warning("Rejecting conformer: SMILES determination failed")
            return None
        smiles, charge = result
        conformer.charge = charge

        conf = conformer.to_numpy()
        sorted_anum = get_sorted_atomic_numbers(conf.atomic_numbers)

        if self.compound_carbon_cap is not None:
            n_carbons = sum(1 for z in sorted_anum if z == 6)
            if n_carbons > self.compound_carbon_cap:
                logger.warning(
                    f"Rejecting conformer: {n_carbons} carbons exceeds cap of {self.compound_carbon_cap}"
                )
                return None

        # Closed-subgraph mode: probe whether the compound is already in
        # this experiment's explorable pool. If not, mark as frontier and
        # let the rest of registration run normally — get_or_create_compound
        # will tag both `experiments` and `frontier_in`, suppressing CREST
        # and (downstream) PES enqueueing.
        is_frontier = False
        if self.restrict_to_existing_compounds:
            from packages.db.models import Compound as _CompoundRow
            existing = (
                session.query(_CompoundRow)
                .filter(_CompoundRow.smiles == smiles)
                .first()
            )
            if existing is None:
                is_frontier = True
                logger.info(
                    f"closed-subgraph: registering frontier compound {smiles} "
                    f"(experiment={self.db.experiment})"
                )
            elif self.db.experiment not in (existing.experiments or []):
                is_frontier = True
                logger.info(
                    f"closed-subgraph: registering existing compound {smiles} "
                    f"as frontier in '{self.db.experiment}' "
                    f"(prior tags={existing.experiments})"
                )

        formula = get_chemical_formula(sorted_anum)
        _, canonicalized_positions, _ = canonicalize_structure(
            conf.atomic_numbers, conf.positions
        )

        compound_id, is_new = self.db.get_or_create_compound(
            session=session,
            smiles=smiles,
            formula=formula,
            charge=charge,
            n_atoms=len(sorted_anum),
            sorted_atomic_numbers=np.array(sorted_anum, dtype=np.int32),
            is_seed=is_seed,
            frontier=is_frontier,
        )

        min_name = self._min_namer.generate("min")
        if is_new:
            # First conformer of a new compound — use positive local_id.
            # Frontier minima skip PES enqueueing (the compound is terminal
            # in this experiment, not worth further exploration).
            min_db_id, min_is_new = self.db.add_minimum(
                session=session,
                compound_id=compound_id,
                local_id=None,
                positions=canonicalized_positions,
                energy=energy,
                discovery_timestamp=time.time(),
                name=min_name,
                enqueue_pes=not is_frontier,
            )
            min_row = session.query(MinimumRow).filter(MinimumRow.id == min_db_id).first()
            next_id = min_row.local_id if min_row else 0
        else:
            # Existing compound — use temp (negative) local_id. A dedup PES
            # job is normally enqueued to resolve this against the
            # compound's existing PES graph (RMSD + NEB dedup). Skip the
            # dedup queue for frontier compounds (they aren't going to be
            # sampled, so dedup is wasted GPU).
            min_db_id, next_id = self.db.add_temp_minimum(
                session=session,
                compound_id=compound_id,
                positions=canonicalized_positions,
                energy=energy,
                discovery_timestamp=time.time(),
                name=min_name,
                enqueue_dedup_pes=not is_frontier,
            )

        info = CompoundInfo(
            compound_id=compound_id,
            smiles=smiles,
            formula=formula,
            charge=charge,
            n_atoms=len(sorted_anum),
            sorted_atomic_numbers=sorted_anum,
            is_seed=is_seed,
        )
        return info, next_id, is_new

    # =========================================================================
    # Adding Reactions (preserves two-phase validate-then-commit)
    # =========================================================================

    def add_contexts_to_graph(self, contexts: list[ExplorationContext], calc=None) -> int:
        """Add validated ExplorationContexts to the reaction graph.

        Each context runs in its own DB session. If _add_reaction_from_context
        returns False (early-abort path) we explicitly roll back the session
        BEFORE the context manager exits — otherwise any compound rows we
        registered during the validation phase would orphan-commit without a
        parent reaction.

        Returns the number of successfully committed reactions.
        """
        n_committed = 0
        for ctx in contexts:
            try:
                session = self.db._session_factory()
                try:
                    committed = self._add_reaction_from_context(session, ctx, calc=calc)
                    if committed:
                        session.commit()
                        n_committed += 1
                    else:
                        logger.info(f"Context {ctx.ts_id} rejected by _add_reaction_from_context (rollback)")
                        session.rollback()
                except Exception:
                    session.rollback()
                    raise
                finally:
                    session.close()
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                logger.error(f"Failed to add context {ctx.ts_id} to graph: {e}\n{tb}")
        return n_committed

    def _find_or_reuse_ts_name(self, session, ts_conformer: Conformer, ts_id: int) -> str:
        """Find an existing reaction with a similar TS geometry (RMSD < threshold).

        If found, reuse its graph name so the viewer groups them as one TS node.
        Otherwise, generate a new name.
        """
        from packages.db.models import Reaction as ReactionRow
        from sqlalchemy.orm import defer

        # defer the two trajectory blobs — we only read the TS conformer
        # positions/atomic_numbers here.  Each trajectory is multi-MB and
        # we have 10k+ reactions — loading them accidentally per
        # dedup-check was ~GB of wasted memory.
        existing_reactions = (
            session.query(ReactionRow)
            .options(
                defer(ReactionRow.reactant_trajectory),
                defer(ReactionRow.product_trajectory),
            )
            .all()
        )
        for rxn_row in existing_reactions:
            try:
                existing_positions = deserialize_ndarray(rxn_row.ts_conformer_positions)
                existing_anum = deserialize_ndarray(rxn_row.ts_conformer_atomic_numbers)
                new_conf_np = ts_conformer.to_numpy()

                # Only compare if same composition
                if existing_anum.shape != new_conf_np.atomic_numbers.shape:
                    continue
                if not np.array_equal(np.sort(existing_anum.flatten()), np.sort(new_conf_np.atomic_numbers.flatten())):
                    continue

                existing_conf = Conformer(
                    positions=torch.tensor(existing_positions, dtype=torch.float32),
                    atomic_numbers=torch.tensor(existing_anum.flatten(), dtype=torch.long),
                )
                rmsd = rmsd_between_conformers(existing_conf, ts_conformer)
                if rmsd < self.rmsd_ts_threshold:
                    return rxn_row.name or f"TS_{rxn_row.ts_id}"
            except Exception:
                continue

        # Generate a unique name. Per-worker NameGenerators have a random
        # offset, but with many parallel workers collisions still happen.
        # Retry until we find a name that isn't already in the DB so the
        # frontend doesn't crash with duplicate cytoscape node ids.
        for _ in range(50):
            candidate = self._namer.generate("rxn")
            existing = (
                session.query(ReactionRow.id)
                .filter(ReactionRow.name == candidate)
                .first()
            )
            if existing is None:
                return candidate
        # Last resort: append the ts_id hash for guaranteed uniqueness.
        # Caller never lands here unless 50 consecutive collisions occur.
        return f"{self._namer.generate('rxn')}-{ts_id}"

    def _add_reaction_from_context(self, session, ctx: ExplorationContext, calc=None) -> bool:
        """Add a complete reaction from an ExplorationContext.

        Returns True if a reaction was successfully created (caller should
        commit), False if the context was rejected mid-validation (caller
        should roll back so we don't orphan-commit the product compound rows
        registered during the validation phase).
        """
        if ctx.ts_conformer is None or ctx.ts_energy is None:
            raise ValueError("Context must have TS conformer and energy")

        products = self._get_products_from_context(ctx)
        is_fragmentation = ctx.has_fragments and len(products) > 1

        # Phase 1: Validate — register compounds
        registered_products = []
        for product_conf, product_id, product_energy in products:
            registered = self._register_product(session, product_conf, product_energy, ctx, calc=calc)
            registered_products.append((registered, product_id, product_energy))

        if is_fragmentation and any(r[0] is None for r in registered_products):
            logger.info(f"Reaction {ctx.ts_id}: rejected — fragmentation product registration failed")
            return False

        reactant_info = self._resolve_reactant_info(session, ctx)
        if reactant_info is None:
            logger.info(f"Reaction {ctx.ts_id}: rejected — reactant info resolution failed")
            return False

        reactant_smiles, reactant_compound_ids_map, barrier_forward = reactant_info
        valid_products = [(r, pid, pe) for r, pid, pe in registered_products if r is not None]
        if not valid_products:
            logger.info(f"Reaction {ctx.ts_id}: rejected — no valid products (all filtered by RMSD<1.0 or registration failure)")
            return False

        # Phase 2: Commit — add reaction, graph edges
        # Check for similar existing TS (RMSD-based deduplication)
        ts_id = ctx.ts_id
        ts_conf = ctx.ts_conformer.to_numpy()
        ts_name = self._find_or_reuse_ts_name(session, ctx.ts_conformer, ts_id)

        # Build reactant list for DB
        reactant_db_entries = []
        for smi in reactant_smiles:
            compound_row = self.db.get_compound_by_smiles(session, smi)
            if compound_row:
                conf_local_id = reactant_compound_ids_map.get(smi)
                reactant_db_entries.append((compound_row.id, conf_local_id))

        # Build product list for DB
        product_db_entries = []
        for (registered, product_id, product_energy) in valid_products:
            info, conformer_local_id = registered
            product_db_entries.append((info.compound_id, conformer_local_id, product_energy))

        barrier_backward = ctx.get_ts_barrier_backward()

        # Compute the principled barrier variants alongside the in-box ones.
        # Separated barriers (TS - sum of reference-conformer energies) bypass
        # ML long-distance artifacts on the IRC endpoints. IRC-extremum barriers
        # (TS - min(E along trajectory side)) are a weaker fix for the same issue.
        bf_sep, bb_sep = self._compute_separated_barriers(
            session,
            ctx.ts_energy,
            [cid for cid, _ in reactant_db_entries],
            [cid for cid, _, _ in product_db_entries],
        )
        bf_ex, bb_ex = self._compute_ex_barriers(ctx)

        # Coerce numpy scalars (np.float64 from upstream's force-integration
        # code) to plain Python floats. psycopg2 stringifies np.float64 as
        # "np.float64(...)" which SQL parses as schema.column → SchemaName error.
        def _f(x):
            return float(x) if x is not None else None

        reaction_db_id, is_new = self.db.create_reaction(
            session=session,
            ts_id=ts_id,
            ts_conformer_positions=ts_conf.positions,
            ts_conformer_atomic_numbers=ts_conf.atomic_numbers,
            ts_conformer_charge=ctx.ts_conformer.charge or 0,
            ts_energy=_f(ctx.ts_energy),
            barrier_forward=_f(barrier_forward),
            barrier_backward=_f(barrier_backward),
            barrier_forward_separated=_f(bf_sep),
            barrier_backward_separated=_f(bb_sep),
            barrier_forward_ex=_f(bf_ex),
            barrier_backward_ex=_f(bb_ex),
            reactant_compound_ids=reactant_db_entries,
            product_compound_ids=product_db_entries,
            reactant_trajectory=ctx.reactant_trajectory,
            product_trajectory=ctx.product_trajectory,
            discovery_method=ctx.discovery_method,
            discovery_noise_level=ctx.discovery_noise_level,
            discovery_timestamp=ctx.discovery_timestamp,
            name=ts_name,
        )

        if is_new:
            # Add graph edges for viewer
            # Reactant edges
            if ctx.was_merged:
                merge_id = self._merge_namer.generate("merge")
                for smi, count in Counter(reactant_smiles).items():
                    self.db.add_graph_edge(
                        session, smi, merge_id, "compound", "merge",
                        direction="merge", stoichiometry=count,
                    )
                self.db.add_graph_edge(
                    session, merge_id, ts_name, "merge", "ts",
                    direction="up", energy_diff=ctx.ts_energy,
                )
            elif len(reactant_smiles) == 1:
                self.db.add_graph_edge(
                    session, reactant_smiles[0], ts_name, "compound", "ts",
                    direction="up",
                    energy_diff=ctx.ts_energy - (ctx.start_energy or 0),
                )

            # Product edges — count multiplicity first, then add one edge per compound
            product_counts = Counter(r[0][0].smiles for r in valid_products)
            product_energies = {}
            for registered, _, product_energy in valid_products:
                info = registered[0]
                product_energies.setdefault(info.smiles, product_energy)
            for smi, count in product_counts.items():
                self.db.add_graph_edge(
                    session, ts_name, smi, "ts", "compound",
                    direction="down",
                    energy_diff=product_energies[smi] - ctx.ts_energy,
                    stoichiometry=count,
                    reaction_id=reaction_db_id,
                )

            sep_note = f", bf_sep: {bf_sep:.3f}" if bf_sep is not None else ""
            logger.info(
                f"Added reaction {ts_name}: "
                f"{' + '.join(reactant_smiles)} -> "
                f"{' + '.join(r[0][0].smiles for r in valid_products)} "
                f"(barrier: {barrier_forward:.3f} eV{sep_note})"
            )

        # Reached create_reaction (whether is_new or merged into existing).
        # Caller commits the session.
        return True

    def _compute_separated_barriers(
        self,
        session,
        ts_energy: float,
        reactant_compound_ids: list[int],
        product_compound_ids: list[int],
    ) -> tuple[Optional[float], Optional[float]]:
        """Separated barrier = TS energy - sum of each compound's lowest-E
        minimum energy. The principled choice for kinetics: bypasses ML
        long-distance artifacts at the IRC endpoints by using fully-relaxed
        reference geometries instead.

        Returns (forward, backward). Returns (None, None) if any participating
        compound has no minima yet (defensive — shouldn't happen since reactants
        and products are registered before this is called).
        """
        def _ref_energy_sum(compound_ids: list[int]) -> Optional[float]:
            total = 0.0
            for cid in compound_ids:
                lowest = (
                    session.query(MinimumRow.energy)
                    .filter(MinimumRow.compound_id == cid, MinimumRow.local_id >= 0)
                    .order_by(MinimumRow.energy.asc())
                    .first()
                )
                if lowest is None or lowest[0] is None:
                    return None
                total += lowest[0]
            return total

        reactant_ref = _ref_energy_sum(reactant_compound_ids)
        product_ref = _ref_energy_sum(product_compound_ids)
        bf = ts_energy - reactant_ref if reactant_ref is not None else None
        bb = ts_energy - product_ref if product_ref is not None else None
        return bf, bb

    def _compute_ex_barriers(
        self, ctx: ExplorationContext
    ) -> tuple[Optional[float], Optional[float]]:
        """IRC-extremum barrier = TS energy - min(E along IRC trajectory side).

        A weaker fix for the same long-distance ML artifact issue: instead of
        using the final trajectory frame's energy (which can be high if the ML
        model overshoots into a high-energy region), use the lowest energy seen
        anywhere along the trajectory side. Stored for the dataset; not used
        by the kinetics solver.
        """
        bf = None
        bb = None
        if ctx.reactant_trajectory is not None and ctx.reactant_trajectory.energies:
            bf = ctx.ts_energy - min(ctx.reactant_trajectory.energies)
        if ctx.product_trajectory is not None and ctx.product_trajectory.energies:
            bb = ctx.ts_energy - min(ctx.product_trajectory.energies)
        return bf, bb

    def _get_products_from_context(self, ctx: ExplorationContext):
        if ctx.has_fragments:
            return list(zip(ctx.min_fragments, ctx.min_fragment_ids, ctx.min_fragment_energies))
        elif ctx.min_conformer is not None:
            return [(ctx.min_conformer, ctx.min_id, ctx.min_energy)]
        else:
            raise ValueError("Context must have MIN conformer or fragments")

    def _register_product(self, session, product_conf, product_energy, ctx, calc=None):
        if not ctx.was_merged and ctx.start_conformer is not None:
            rmsd = rmsd_between_conformers(ctx.start_conformer, product_conf)
            if rmsd < 1.0:
                logger.debug(f"Product rejected: RMSD={rmsd:.3f} < 1.0 to reactant (identity reaction)")
                return None
        result = self.register_compound(session, product_conf, product_energy, calc=calc)
        if result is None:
            logger.info(f"Product rejected: register_compound returned None (SMILES determination failed)")
            return None
        info, conformer_id, is_new = result
        if is_new:
            logger.info(f"Discovered new compound: {info.formula} ({info.smiles})")
        return info, conformer_id

    def _resolve_reactant_info(self, session, ctx):
        if ctx.was_merged:
            comp1, comp2 = ctx.merge_component_conformers
            smiles_1 = get_smiles_from_conformer(comp1)
            smiles_2 = get_smiles_from_conformer(comp2)
            if smiles_1 is None or smiles_2 is None:
                return None
            canonical_smiles = []
            for comp, energy in zip(
                [comp1, comp2], ctx.merge_component_energies or [None, None]
            ):
                if energy is not None:
                    result = self.register_compound(session, comp, energy)
                    if result:
                        canonical_smiles.append(result[0].smiles)
                    else:
                        canonical_smiles.append(get_smiles_from_conformer(comp))
                else:
                    canonical_smiles.append(get_smiles_from_conformer(comp))
            return tuple(sorted(s for s in canonical_smiles if s)), {}, ctx.get_ts_barrier_forward()

        if ctx.start_conformer is None:
            return None

        single_smiles = ctx.source_compound_smiles or get_smiles_from_conformer(ctx.start_conformer)
        if single_smiles is None:
            return None

        reactant_conformer_ids = {}
        result = self.register_compound(session, ctx.start_conformer, ctx.start_energy)
        if result is not None:
            info, conf_id, is_new = result
            single_smiles = info.smiles
            reactant_conformer_ids[single_smiles] = conf_id
        return (single_smiles,), reactant_conformer_ids, ctx.get_ts_barrier_forward()

    # =========================================================================
    # Sampling (DB-backed)
    # =========================================================================

    def sample_exploration_batch(
        self,
        max_contexts: int,
        # 50:50 split single-vs-merge.  Early in exploration the graph has
        # only small compounds and merges are the only way to grow —
        # 30:70 made sense then.  Now with ~1000+ compounds of varying
        # size, fragmentations (single contexts) are equally valuable for
        # breaking down larger intermediates, and they reject far less in
        # sampling (no pair-count cap).
        single_context_ratio: float = 0.5,
        max_carbon_count: Optional[int] = None,
    ) -> list[ExplorationContext]:
        """Sample a batch of ExplorationContexts for TS generation."""
        with self.db.session() as session:
            return self._sample_batch(session, max_contexts, single_context_ratio, max_carbon_count=max_carbon_count)

    def _load_latest_kinetics_snapshot(self, session):
        """Fetch the latest KineticsSnapshot row, with instance-level caching.

        The kinetics payload is multi-MB JSONB (plot data + per-decade distributions
        + etc.), and this method is called every worker loop iter. Fetching the
        full payload each time cost ~35 ms per call → ~175% DB CPU across 50
        workers. We cache the decoded object by snapshot id and only reload when
        a cheap ID-only query shows a newer row.

        Returns a packages.kinetics.snapshot.KineticsSnapshot instance or None
        if no snapshot is available yet (kinetics solver hasn't run, or DB is
        empty). Caller should silently fall back to uniform sampling.
        """
        from packages.db.models import KineticsSnapshot as KineticsSnapshotRow
        try:
            from packages.kinetics.snapshot import KineticsSnapshot
        except ImportError:
            logger.error("packages.kinetics.snapshot not available — PES concentration gate DISABLED")
            return None

        # Cheap staleness probe — index-only scan, <1 ms regardless of payload size.
        # Scoped to this worker's experiment: each experiment has its own
        # snapshot history, and the worker should sample on the snapshot
        # solved against its own subgraph.
        latest_id = (
            session.query(KineticsSnapshotRow.id)
            .filter(KineticsSnapshotRow.experiment == self.db.experiment)
            .order_by(KineticsSnapshotRow.computed_at.desc())
            .limit(1)
            .scalar()
        )
        if latest_id is None:
            return None

        cached = getattr(self, "_snapshot_cache", None)
        if cached is not None and cached[0] == latest_id:
            return cached[1]

        # Fetch the full payload only when the latest row is new to us.
        row = (
            session.query(KineticsSnapshotRow)
            .filter(KineticsSnapshotRow.id == latest_id)
            .first()
        )
        if row is None:  # raced — snapshot deleted between probe and fetch, unlikely but handle
            return None
        try:
            snap = KineticsSnapshot.from_json(row.payload_jsonb)
        except Exception as e:
            logger.warning(f"Failed to deserialize kinetics snapshot {row.id}: {e}")
            return None
        self._snapshot_cache = (latest_id, snap)
        return snap

    def _sample_batch(self, session, max_contexts, single_context_ratio, max_carbon_count: Optional[int] = None):
        batch = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        n_single = int(max_contexts * single_context_ratio)

        # Load all compounds with their minima for sampling (instance-cached).
        compounds_data = self._get_compounds_data(session, max_carbon_count=max_carbon_count)
        if not compounds_data:
            return batch

        # Kinetic sampling (Phase 5): optionally fetch the latest snapshot
        # from the kinetics_snapshots table. If present AND has at least one
        # sampleable decade, use it to drive merge pair selection. Falls
        # back to uniform on None / exhausted snapshot.
        kinetic_snapshot = None
        smiles_to_compound = None
        if self.kinetic_sampling_enabled:
            kinetic_snapshot = self._load_latest_kinetics_snapshot(session)
            if kinetic_snapshot is not None and kinetic_snapshot.is_ready:
                smiles_to_compound = {c["smiles"]: c for c in compounds_data}
                logger.debug(
                    f"kinetic sampling: using snapshot "
                    f"(n_reactions={kinetic_snapshot.n_reactions}, "
                    f"n_dft={kinetic_snapshot.n_reactions_dft}, "
                    f"ss_species={len(kinetic_snapshot.steady_state_distribution)})"
                )
            else:
                kinetic_snapshot = None  # explicit reset for the batch stats

        # Reaction-count-per-pair, used to cap over-sampled merges.
        # Instance-level cache keyed by non-manual reaction count — cheap
        # index-only probe rebuilds only when a new reaction actually
        # lands.  The old ORM `.options(joinedload(...)).all()` hydrated
        # every Reaction + its reactants + products + compounds every
        # single loop iter (~2948 reactions × 4 participants × full ORM
        # rows = ~40 MB Python object graph) and dominated sampling
        # wall-time; now it's a ~ms staleness probe and a single lean
        # tuple query only when we actually need to refresh.
        MAX_REACTIONS_PER_PAIR = 4
        pair_rxn_counts = self._get_pair_rxn_counts(session)

        # Sample single contexts — use kinetics if available.  No retry
        # budget: _sample_single almost always succeeds (just picks an
        # existing DB compound and wraps it), and any occasional None
        # just means the batch ships with fewer singles, which is fine.
        for _ in range(n_single):
            ctx = self._sample_single(
                compounds_data, device,
                snapshot=kinetic_snapshot,
                smiles_to_compound=smiles_to_compound,
            )
            if ctx is not None:
                batch.append(ctx)

        # Sample merge pairs in bulk.  Replaces the old per-call nested
        # retry loops: one numpy multinomial draw of N×n_needed
        # candidates, one filter pass (composition limit + pair-count
        # cap), take first n_needed.  Duplicate pairs are kept on purpose
        # — each becomes a fresh rotation/noise seed in the TS pipeline.
        from lib.utils import TETROSE_LIMIT
        merge_needed = max_contexts - len(batch)
        merge_pairs, source_tag = self._oversample_merge_pairs(
            snapshot=kinetic_snapshot,
            compounds_data=compounds_data,
            smiles_to_compound=smiles_to_compound,
            n_needed=merge_needed,
            pair_rxn_counts=pair_rxn_counts,
            max_reactions_per_pair=MAX_REACTIONS_PER_PAIR,
            composition_limit=TETROSE_LIMIT,
        )

        n_kinetic = 0
        n_uniform = 0
        for c1, c2 in merge_pairs:
            m1 = random.choice(c1["minima"])
            m2 = random.choice(c2["minima"])
            conf1 = Conformer(
                positions=torch.tensor(m1["positions"], dtype=torch.float32),
                atomic_numbers=torch.tensor(m1["atomic_numbers"].flatten(), dtype=torch.long),
                energy=m1["energy"],
                charge=c1["charge"],
            ).center()
            conf2 = Conformer(
                positions=torch.tensor(m2["positions"], dtype=torch.float32),
                atomic_numbers=torch.tensor(m2["atomic_numbers"].flatten(), dtype=torch.long),
                energy=m2["energy"],
                charge=c2["charge"],
            ).center()
            ctx = merge_conformers_to_context(
                conf1, conf2,
                conformer_1_id=ExplorationContext.compute_id(conf1),
                conformer_2_id=ExplorationContext.compute_id(conf2),
                conformer_1_smiles=c1["smiles"],
                conformer_2_smiles=c2["smiles"],
            )
            if ctx is None:
                continue
            if ctx.merged_conformer.n_atoms < self.fragment_threshold:
                continue
            batch.append(ctx.to(device))
            if source_tag == "kinetic":
                n_kinetic += 1
            else:
                n_uniform += 1

        # Stash sampling stats on the batch list for the worker to harvest
        # into the BatchLog row. The worker reads these via the `_sampling_stats`
        # attribute (set on the batch list object — ugly but zero-diff for
        # callers that just iterate the batch).
        self._last_sampling_stats = {
            "n_kinetic": n_kinetic,
            "n_uniform": n_uniform,
            "snapshot_n_reactions": kinetic_snapshot.n_reactions if kinetic_snapshot else None,
            "snapshot_n_reactions_dft": kinetic_snapshot.n_reactions_dft if kinetic_snapshot else None,
            "snapshot_used": kinetic_snapshot is not None,
        }
        return batch

    def _get_pair_rxn_counts(self, session) -> dict[frozenset, int]:
        """Instance-cached reactant-∪-product frozenset → count map.

        Used by `_sample_batch` to avoid re-sampling already well-covered
        compound pairs.  Key is direction-agnostic (frozenset of SMILES).
        Scoped to this worker's experiment so coverage from another
        experiment doesn't suppress sampling here.
        """
        from sqlalchemy import text as _text, func as _func
        from packages.db.models import Reaction as ReactionRow

        current_count = session.query(_func.count(ReactionRow.id)).filter(
            (ReactionRow.discovery_method != "manual_equilibrium") |
            (ReactionRow.discovery_method.is_(None))
        ).filter(
            ReactionRow.experiments.any(self.db.experiment)
        ).scalar() or 0

        cached = getattr(self, "_pair_counts_cache", None)
        if cached is not None and cached[0] == current_count:
            return cached[1]

        # Rebuild with a lean tuple query — no ORM hydration, no joinedload.
        # Two UNION ALL branches so a reaction's reactant+product compounds
        # all land in the same reaction_id bucket for frozenset assembly.
        rows = session.execute(_text("""
            SELECT r.id, c.smiles
              FROM reactions r
              JOIN reaction_reactants rr ON rr.reaction_id = r.id
              JOIN compounds c ON c.id = rr.compound_id
             WHERE (r.discovery_method != 'manual_equilibrium'
                    OR r.discovery_method IS NULL)
               AND :experiment = ANY(r.experiments)
            UNION ALL
            SELECT r.id, c.smiles
              FROM reactions r
              JOIN reaction_products rp ON rp.reaction_id = r.id
              JOIN compounds c ON c.id = rp.compound_id
             WHERE (r.discovery_method != 'manual_equilibrium'
                    OR r.discovery_method IS NULL)
               AND :experiment = ANY(r.experiments)
        """), {"experiment": self.db.experiment}).fetchall()

        rxn_smiles: dict[int, set[str]] = {}
        for rxn_id, smi in rows:
            rxn_smiles.setdefault(rxn_id, set()).add(smi)

        counts: dict[frozenset, int] = {}
        for smis in rxn_smiles.values():
            key = frozenset(smis)
            counts[key] = counts.get(key, 0) + 1

        self._pair_counts_cache = (current_count, counts)
        return counts

    def _get_compounds_data(self, session, max_carbon_count: Optional[int] = None):
        """Instance-cached compound + minima payload for sampling.

        The underlying `_load_compounds_for_sampling` is expensive — load
        once per (compound_count, max_minimum_id, max_carbon_count) signature
        per worker. Probe scoped to this experiment so a non-formose
        compound landing in main does not invalidate the formose sampler's
        cache.
        """
        from sqlalchemy import text as _text
        probe = session.execute(_text("""
            SELECT (SELECT COUNT(*) FROM compounds
                     WHERE :experiment = ANY(experiments)) AS n_comp,
                   (SELECT COALESCE(MAX(id), 0) FROM minima
                     WHERE local_id >= 0
                       AND :experiment = ANY(experiments)) AS max_min_id
        """), {"experiment": self.db.experiment}).fetchone()
        signature = (int(probe[0]), int(probe[1]), max_carbon_count)

        cached = getattr(self, "_compounds_data_cache", None)
        if cached is not None and cached[0] == signature:
            return cached[1]

        data = self._load_compounds_for_sampling(session, max_carbon_count=max_carbon_count)
        self._compounds_data_cache = (signature, data)
        return data

    def _load_compounds_for_sampling(self, session, max_carbon_count: Optional[int] = None):
        """Load compound + minima data for sampling, scoped to experiment."""
        from packages.db.models import Compound as CompoundRow, Minimum
        from sqlalchemy.orm import defer

        # Scoped to this worker's experiment via the GIN ANY filter on
        # CompoundRow.experiments — sampling never reaches outside the
        # current experiment's compound set. Compounds whose `frontier_in`
        # array contains this experiment (boundary species discovered as
        # products of a closed-subgraph reaction) are visible in the
        # graph view but excluded from the sampling pool — workers don't
        # burn GPU re-exploring boundary nodes.
        query = (
            session.query(Minimum, CompoundRow)
            .options(defer(Minimum.hessian))
            .join(CompoundRow, Minimum.compound_id == CompoundRow.id)
            .filter(Minimum.local_id >= 0)
            .filter(CompoundRow.experiments.any(self.db.experiment))
        )
        if self.db.experiment:
            query = query.filter(
                ~CompoundRow.frontier_in.any(self.db.experiment)
            )
        # Group minima by compound
        compound_minima: dict[int, tuple[Any, list]] = {}
        for m, c in query.all():
            if c.id not in compound_minima:
                compound_minima[c.id] = (c, [])
            compound_minima[c.id][1].append(m)

        result = []
        for compound_id, (row, minima) in compound_minima.items():
            if max_carbon_count is not None:
                sorted_anum = deserialize_ndarray(row.sorted_atomic_numbers).flatten().astype(int)
                n_carbons = int(np.sum(sorted_anum == 6))
                if n_carbons > max_carbon_count:
                    continue

            result.append({
                "smiles": row.smiles,
                "formula": row.formula,
                "charge": row.charge,
                "n_atoms": row.n_atoms,
                "sorted_atomic_numbers": deserialize_ndarray(row.sorted_atomic_numbers),
                "minima": [
                    {
                        "local_id": m.local_id,
                        "positions": deserialize_ndarray(m.positions),
                        "energy": m.energy,
                        "atomic_numbers": deserialize_ndarray(row.sorted_atomic_numbers),
                    }
                    for m in minima
                ],
            })
        return result

    def _sample_single(self, compounds_data, device, snapshot=None, smiles_to_compound=None):
        """Sample a single-molecule context.

        Uses kinetics snapshot (entropy-weighted decade → species) when available,
        falling back to size-weighted sampling.
        """
        # Filter out fragments
        eligible = [c for c in compounds_data if c["n_atoms"] >= self.fragment_threshold]
        if not eligible:
            return None

        # Try kinetic sampling: pick a species from the snapshot
        compound = None
        if snapshot is not None and smiles_to_compound is not None:
            try:
                from packages.kinetics.sampler import sample_pair_from_snapshot
                pair = sample_pair_from_snapshot(snapshot)
                if pair is not None:
                    smi1, smi2 = pair
                    # Pick one of the two species at random
                    smi = random.choice([smi1, smi2])
                    if smi in smiles_to_compound:
                        compound = smiles_to_compound[smi]
            except Exception:
                pass

        if compound is None:
            # Fallback: size-weighted sampling
            weights = [c["n_atoms"] for c in eligible]
            compound = random.choices(eligible, weights=weights, k=1)[0]

        # Boltzmann-weighted conformer sampling
        minima = compound["minima"]
        k_b_T = 0.0257  # eV at 300K
        min_e = min(m["energy"] for m in minima)
        boltz = [np.exp(-(m["energy"] - min_e) / k_b_T) for m in minima]
        minimum = random.choices(minima, weights=boltz, k=1)[0]

        conformer = Conformer(
            positions=torch.tensor(minimum["positions"], dtype=torch.float32),
            atomic_numbers=torch.tensor(minimum["atomic_numbers"].flatten(), dtype=torch.long),
            energy=minimum["energy"],
            charge=compound["charge"],
        ).center()

        ctx = create_single_mol_context(
            start_conformer=conformer.to(device),
            source_compound_smiles=compound["smiles"],
        )
        return ctx

    def _oversample_merge_pairs(
        self,
        snapshot,
        compounds_data,
        smiles_to_compound,
        n_needed: int,
        pair_rxn_counts: Optional[dict],
        max_reactions_per_pair: int,
        composition_limit: dict,
    ) -> tuple[list[tuple[dict, dict]], str]:
        """Vectorized merge-pair sampling.

        Draws `oversample_factor * n_needed` candidate pairs from the
        steady-state distribution (or uniform size-weighted fallback if
        no snapshot), filters by composition cap and per-pair reaction
        cap in one pass, returns up to n_needed `(c1, c2)` tuples.

        Duplicates are preserved — each sampled pair becomes a different
        rotation/noise seed in the TS pipeline downstream.

        Returns (pairs, source_tag).  source_tag ∈ {'kinetic','uniform'}.
        """
        if n_needed <= 0 or len(compounds_data) < 1:
            return [], "uniform"

        n_cand = n_needed * self.oversample_factor
        rng = np.random.default_rng()

        # Decide sampling source: kinetic (snapshot) or uniform fallback.
        use_kinetic = False
        if snapshot is not None and smiles_to_compound is not None:
            dist = snapshot.steady_state_distribution
            smiles_list = list(dist.keys())
            weights = np.asarray(list(dist.values()), dtype=np.float64)
            wsum = float(weights.sum())
            if len(smiles_list) >= 2 and wsum > 0:
                probs = weights / wsum
                i1 = rng.choice(len(smiles_list), n_cand, p=probs)
                i2 = rng.choice(len(smiles_list), n_cand, p=probs)
                candidates = [
                    (smiles_to_compound.get(smiles_list[a]),
                     smiles_to_compound.get(smiles_list[b]))
                    for a, b in zip(i1, i2)
                ]
                use_kinetic = True

        if not use_kinetic:
            # uniform size-weighted on one side, uniform on the other —
            # matches the old fallback behaviour.
            w = np.asarray([c["n_atoms"] for c in compounds_data], dtype=np.float64)
            probs = w / w.sum() if w.sum() > 0 else None
            i1 = rng.choice(len(compounds_data), n_cand, p=probs)
            i2 = rng.choice(len(compounds_data), n_cand)
            candidates = [(compounds_data[a], compounds_data[b]) for a, b in zip(i1, i2)]

        # Composition per compound, cached by object id.  Avoids re-parsing
        # sorted_atomic_numbers for every candidate pair.
        comp_cache: dict[int, tuple[int, int, int]] = {}

        def _comp(c):
            k = id(c)
            if k not in comp_cache:
                a = np.asarray(c.get("sorted_atomic_numbers")).flatten()
                comp_cache[k] = (int((a == 6).sum()), int((a == 1).sum()), int((a == 8).sum()))
            return comp_cache[k]

        c_cap = composition_limit.get("C", 999)
        h_cap = composition_limit.get("H", 999)
        o_cap = composition_limit.get("O", 999)

        selected: list[tuple[dict, dict]] = []
        for c1, c2 in candidates:
            if c1 is None or c2 is None:
                continue
            co1, co2 = _comp(c1), _comp(c2)
            if co1[0] + co2[0] > c_cap: continue
            if co1[1] + co2[1] > h_cap: continue
            if co1[2] + co2[2] > o_cap: continue
            if pair_rxn_counts is not None:
                key = frozenset({c1["smiles"], c2["smiles"]})
                if pair_rxn_counts.get(key, 0) >= max_reactions_per_pair:
                    continue
            selected.append((c1, c2))
            if len(selected) >= n_needed:
                break

        return selected, ("kinetic" if use_kinetic else "uniform")

    # =========================================================================
    # PES Work Queue
    # =========================================================================

    def get_compound_to_postprocess(
        self,
        pes_backlog_tolerance: float = 0.0,
        min_compound_age_s: float = 0.0,
        min_conc: float = 0.0,
        job_kind_filter: Optional[str] = None,
    ) -> Optional[dict]:
        """Claim a compound for PES exploration from the DB work queue.

        Args:
            pes_backlog_tolerance: back-off threshold — if pending fraction
                is below this, return None (let PES work drain).
            min_compound_age_s: wait gate. A compound must have been in the
                DB at least this long before its PES work items are eligible.
                Gives the kinetics solver time to establish a concentration.
            min_conc: concentration gate. Requires the latest kinetics
                snapshot to assign compound's steady-state concentration
                ≥ min_conc before claiming. If no snapshot is ready, the
                gate is not applied (prevents cold-start starvation).

        Returns dict with work_id, compound_id, minimum_id or None.
        """
        with self.db.session() as session:
            n_pending, n_in_progress, n_total = self.db.get_pes_backlog(session)
            if n_total > 0 and n_pending > 0:
                backlog_fraction = n_pending / n_total
                if backlog_fraction <= pes_backlog_tolerance:
                    logger.debug(
                        f"PES backlog within tolerance: {n_pending}/{n_total} "
                        f"({backlog_fraction:.2f}) <= {pes_backlog_tolerance:.2f}"
                    )
                    return None

            # Build concentration lookup from latest kinetics snapshot.
            # If we can't get concentrations, skip PES entirely — don't waste GPU.
            conc_lookup: dict = {}
            if min_conc > 0.0:
                snap = self._load_latest_kinetics_snapshot(session)
                if snap is None:
                    logger.warning("PES conc gate: no kinetics snapshot — skipping PES")
                    return None
                if not getattr(snap, "is_ready", False):
                    logger.warning("PES conc gate: snapshot not ready — skipping PES")
                    return None
                log_concs = getattr(snap, "steady_state_log_concs", {}) or {}
                if not log_concs:
                    logger.warning("PES conc gate: empty steady_state_log_concs — skipping PES")
                    return None
                conc_lookup = {smi: 10.0 ** lc for smi, lc in log_concs.items()}

            return self.db.claim_pes_work(
                session,
                min_compound_age_s=min_compound_age_s,
                conc_lookup=conc_lookup,
                min_conc=min_conc,
                job_kind_filter=job_kind_filter,
            )

    def complete_compound_postprocessing(self, work_id: int, failed: bool = False):
        """Complete PES exploration work item."""
        with self.db.session() as session:
            if failed:
                self.db.fail_pes_work(session, work_id)
            else:
                self.db.complete_pes_work(session, work_id)

    def release_compound_postprocessing(self, work_id: int):
        """Release a claimed PES work item back to pending (e.g. on spot preemption)."""
        with self.db.session() as session:
            self.db.release_pes_work(session, work_id)

    def get_compound(self, smiles: str) -> Optional[Compound]:
        """Load a Compound by SMILES with its full PES graph.

        Returns the Compound or None if not found.
        """
        with self.db.session() as session:
            row = self.db.get_compound_by_smiles(session, smiles)
            if row is None:
                return None
            result = self._load_compound_from_row(session, row)
            if result is None:
                return None
            compound, _ = result
            return compound

    def _load_compound_from_row(self, session, row) -> Optional[tuple[Compound, list]]:
        """Build a Compound with PESGraph from a CompoundRow.

        Returns (Compound, unexplored_minima_db_ids) or None.
        """
        sorted_anum = tuple(deserialize_ndarray(row.sorted_atomic_numbers).flatten().astype(int).tolist())

        # Load minima
        # Only load positive-local_id minima (exclude temp/negative ones
        # that haven't been dedup-resolved yet).
        all_minima = self.db.get_minima_for_compound(session, row.id)
        minima_rows = [m for m in all_minima if m.local_id >= 0]
        if not minima_rows:
            return None

        # Find lowest energy minimum to use as initial
        lowest = min(minima_rows, key=lambda m: m.energy)
        compound = Compound(
            smiles=row.smiles,
            sorted_atomic_numbers=sorted_anum,
            initial_positions=deserialize_ndarray(lowest.positions),
            initial_energy=lowest.energy,
            is_seed=row.is_seed,
        )

        # Load all minima into PES graph using insert_minimum_direct — bypasses
        # dedup so DB local_ids are preserved as PES graph local_ids exactly.
        # The old approach used add_minimum() which ran dedup, assigned NEW
        # sequential IDs, and created a mismatch: save_pes_results would then
        # write new minima with PESGraph IDs that collided with DIFFERENT
        # existing DB rows, producing ghost duplicates.
        #
        # The initial Compound.__post_init__ already added the lowest-energy
        # minimum as PESGraph id 0 via add_minimum(). We need to replace it
        # with the correct DB local_id.
        compound.pes_graph.minima.clear()
        compound.pes_graph.graph.clear()
        compound.pes_graph._next_min_id = 0

        db_id_map: dict[int, int] = {}  # DB minimum.id → PESGraph local_id
        for m in minima_rows:
            compound.pes_graph.insert_minimum_direct(
                min_id=m.local_id,
                positions=deserialize_ndarray(m.positions),
                energy=m.energy,
                explored=m.explored,
                hessian=deserialize_ndarray_optional(m.hessian) if m.hessian else None,
                name=m.name or "",
                n_merged=m.n_merged,
                max_merge_rmsd=m.max_merge_rmsd,
                discovery_timestamp=m.discovery_timestamp,
            )
            db_id_map[m.id] = m.local_id

        # Load existing intra TSs using insert_ts_direct — bypasses dedup and
        # endpoint resolution, preserving exact DB local_ids.
        ts_rows = self.db.get_intra_ts_for_compound(session, row.id)
        for ts in ts_rows:
            fwd_local = db_id_map.get(ts.min_fwd_id)
            bwd_local = db_id_map.get(ts.min_bwd_id)
            if fwd_local is None or bwd_local is None:
                continue
            compound.pes_graph.insert_ts_direct(
                ts_id=ts.local_id,
                positions=deserialize_ndarray(ts.positions),
                energy=ts.energy,
                min_fwd_id=fwd_local,
                min_bwd_id=bwd_local,
                barrier_fwd=ts.barrier_fwd or 0.0,
                barrier_bwd=ts.barrier_bwd or 0.0,
                eigenvalue=ts.eigenvalue or 0.0,
                hessian=deserialize_ndarray_optional(ts.hessian) if hasattr(ts, 'hessian') and ts.hessian else None,
                rmsd_to_fwd_min=ts.rmsd_to_fwd_min or 0.0,
                rmsd_to_bwd_min=ts.rmsd_to_bwd_min or 0.0,
                endpoint_to_endpoint_rmsd=ts.endpoint_to_endpoint_rmsd or 0.0,
                name=ts.name or "",
                discovery_timestamp=ts.discovery_timestamp or 0.0,
            )

        # Collect unexplored minimum DB IDs
        unexplored_db_ids = [m.id for m in minima_rows if not m.explored]

        return compound, unexplored_db_ids

    def load_compound_for_pes(self, compound_id: int, minimum_id: int) -> Optional[tuple[Compound, int, int]]:
        """Load a Compound with its PESGraph from the DB for PES exploration.

        Args:
            compound_id: DB compound ID.
            minimum_id: DB minimum ID of the specific minimum to explore.

        Returns (Compound, minimum_db_id, conformer_id) or None.
            conformer_id is the PES graph local ID for the target minimum.
        """
        with self.db.session() as session:
            from packages.db.models import Compound as CompoundRow, Minimum as MinimumRow

            row = session.query(CompoundRow).filter(CompoundRow.id == compound_id).first()
            if row is None:
                return None

            result = self._load_compound_from_row(session, row)
            if result is None:
                return None

            compound, _unexplored_db_ids = result

            # Resolve minimum_id (DB row id) → PES graph local_id.
            # With insert_minimum_direct, PES graph local_ids == DB local_ids,
            # so we just look up the DB row's local_id.  Scalar query
            # avoids loading the hessian blob.
            conformer_id = session.query(MinimumRow.local_id).filter(MinimumRow.id == minimum_id).scalar()
            if conformer_id is None:
                return None
            if conformer_id not in compound.pes_graph.minima:
                logger.warning(f"minimum local_id={conformer_id} not in PES graph for compound {compound_id}")
                return None

            return compound, minimum_id, conformer_id

    def save_pes_results(
        self,
        compound_id: int,
        compound: Compound,
        explored_minimum_db_id: int,
    ):
        """Save PES exploration results back to DB.

        Saves all minima and intra TS from the compound's PES graph,
        enriching existing entries with new data (hessians, merge stats).
        Only marks the single explored minimum as explored.
        New minima automatically get work queue entries via db.add_minimum.
        """
        from packages.db.models import Minimum as MinimumRow

        with self.db.session() as session:
            # Save all minima (new ones get created, existing ones get enriched)
            local_id_to_db_id = {}
            for min_id, minimum in compound.pes_graph.minima.items():
                min_name = getattr(minimum, 'name', '') or self._min_namer.generate("min")
                db_id, _ = self.db.add_minimum(
                    session=session,
                    compound_id=compound_id,
                    local_id=min_id,
                    positions=minimum.positions,
                    energy=minimum.energy,
                    hessian=getattr(minimum, 'hessian', None),
                    explored=minimum.explored,
                    discovery_timestamp=getattr(minimum, 'discovery_timestamp', time.time()),
                    name=min_name,
                    n_merged=getattr(minimum, 'n_merged', 0),
                    max_merge_rmsd=getattr(minimum, 'max_merge_rmsd', 0.0),
                )
                local_id_to_db_id[min_id] = db_id

            # Save all intra TS (new ones get created, existing ones get hessian enriched)
            for ts_id, ts in compound.pes_graph.transition_states.items():
                fwd_db_id = local_id_to_db_id.get(ts.min_fwd_id)
                bwd_db_id = local_id_to_db_id.get(ts.min_bwd_id)

                if fwd_db_id is None or bwd_db_id is None:
                    # Fallback: look up by local_id in DB.  Scalar queries
                    # skip the hessian blob.
                    fwd_db_id = session.query(MinimumRow.id).filter_by(
                        compound_id=compound_id, local_id=ts.min_fwd_id
                    ).scalar()
                    bwd_db_id = session.query(MinimumRow.id).filter_by(
                        compound_id=compound_id, local_id=ts.min_bwd_id
                    ).scalar()
                    if fwd_db_id is None or bwd_db_id is None:
                        logger.warning(f"Cannot save intra TS {ts_id}: endpoint minimum not found")
                        continue

                self.db.add_intra_ts(
                    session=session,
                    compound_id=compound_id,
                    local_id=ts_id,
                    positions=ts.positions,
                    energy=ts.energy,
                    eigenvalue=getattr(ts, 'eigenvalue', 0.0),
                    min_fwd_db_id=fwd_db_id,
                    min_bwd_db_id=bwd_db_id,
                    barrier_fwd=ts.barrier_fwd,
                    barrier_bwd=ts.barrier_bwd,
                    hessian=getattr(ts, 'hessian', None),
                    rmsd_to_fwd_min=getattr(ts, 'rmsd_to_fwd_min', 0.0),
                    rmsd_to_bwd_min=getattr(ts, 'rmsd_to_bwd_min', 0.0),
                    endpoint_to_endpoint_rmsd=getattr(ts, 'endpoint_to_endpoint_rmsd', 0.0),
                    fwd_trajectory=getattr(ts, 'fwd_trajectory', None),
                    bwd_trajectory=getattr(ts, 'bwd_trajectory', None),
                    discovery_timestamp=getattr(ts, 'discovery_timestamp', time.time()),
                )

            # Mark explored minima in DB — the PES graph's in-memory
            # explored flags are authoritative after _explore_from_minimum.
            # Previously only the claimed minimum was marked, leaving other
            # minima that were explored in the same iteration as unexplored
            # in the DB (causing redundant re-exploration).
            for min_id, minimum in compound.pes_graph.minima.items():
                if minimum.explored:
                    db_id = local_id_to_db_id.get(min_id)
                    if db_id is not None:
                        self.db.mark_minimum_explored(session, db_id)

    def handle_dedup_job(self, compound_id: int, minimum_id: int, work_id: int, calc):
        """Run RMSD+NEB dedup for a temp minimum against the compound's PES graph.

        Called by the PES worker for job_kind='dedup'. The PES queue's per-compound
        serialization ensures this doesn't race with explore jobs.

        If the temp minimum duplicates an existing one: merge references + delete.
        If genuinely new: promote to positive local_id + enqueue explore job.
        """
        from packages.db.models import Minimum as MinimumRow
        from packages.db.serialization import deserialize_ndarray as _deser

        with self.db.session() as session:
            # Load temp minimum (hessian not needed — defer it)
            from sqlalchemy.orm import defer
            temp_min = (
                session.query(MinimumRow)
                .options(defer(MinimumRow.hessian))
                .filter(MinimumRow.id == minimum_id)
                .first()
            )
            if temp_min is None or temp_min.local_id >= 0:
                # Already resolved or not a temp min
                self.db.complete_pes_work(session, work_id)
                return

            temp_local_id = temp_min.local_id
            temp_positions = _deser(temp_min.positions)
            temp_energy = temp_min.energy

            # Load compound's PES graph (positive-local_id minima only)
            from packages.db.models import Compound as CompoundRow
            row = session.query(CompoundRow).filter(CompoundRow.id == compound_id).first()
            if row is None:
                logger.warning(f"Dedup: compound {compound_id} not found")
                self.db.complete_pes_work(session, work_id)
                return

            result = self._load_compound_from_row(session, row)
            if result is None:
                logger.warning(f"Dedup: could not load compound {compound_id}")
                self.db.complete_pes_work(session, work_id)
                return

            compound, _ = result

            # Set calculator for NEB dedup
            compound.pes_graph.calc = calc

            # Canonicalize temp positions and run dedup
            canonical_pos, _, _ = compound.pes_graph._canonicalize(temp_positions)
            match_id, match_rmsd, neb_time = compound.pes_graph._find_matching_minimum(
                canonical_pos, temp_energy
            )

            if match_id is not None:
                # Duplicate — merge into existing
                logger.info(
                    f"Dedup: temp min {temp_local_id} of compound {compound_id} "
                    f"matches min {match_id} (RMSD={match_rmsd:.4f}, NEB={neb_time:.2f}s)"
                )
                self.db.resolve_temp_minimum_as_duplicate(
                    session, compound_id, minimum_id, temp_local_id, match_id
                )
            else:
                # Genuinely new — promote to positive local_id
                new_lid = self.db.promote_temp_minimum(
                    session, compound_id, minimum_id, temp_local_id
                )
                logger.info(
                    f"Dedup: temp min {temp_local_id} of compound {compound_id} "
                    f"promoted to min {new_lid} (NEB={neb_time:.2f}s)"
                )

    # =========================================================================
    # Stats
    # =========================================================================

    def record_batch_stats(self, batch_summary: dict) -> None:
        """Record batch exploration statistics to DB."""
        with self.db.session() as session:
            # Build updates dict from batch summary
            updates = {}
            base_stage_keys = [
                "pipeline_contexts_submitted", "pipeline_survived_denoising",
                "pipeline_passed_fwd_barrier", "pipeline_passed_irc",
                "pipeline_passed_fragmentation", "pipeline_passed_bwd_barrier",
                "pipeline_added_to_graph",
                "merge_submitted", "merge_valid", "single_submitted", "single_valid",
                # Per-type stage counters (P3)
                "merge_passed_fwd_barrier", "single_passed_fwd_barrier",
                "merge_passed_irc", "single_passed_irc",
                "merge_passed_fragmentation", "single_passed_fragmentation",
                "merge_passed_bwd_barrier", "single_passed_bwd_barrier",
            ]
            irc_base_keys = [
                "irc_hessian_pass", "irc_hessian_fail",
                "irc_relax_fwd_fail", "irc_relax_bwd_fail",
                "irc_endpoints_neither_match", "irc_endpoints_both_match",
                "irc_endpoints_one_match",
                "irc_greedy_fallback_used", "irc_greedy_accepted",
                # Granular greedy-reject reasons (see _validate_chemical_difference)
                "greedy_reject_frag_invalid",
                "greedy_reject_too_many_fragments",
                "greedy_reject_smiles_fail",
                "greedy_reject_same_signature",
                "greedy_reject_spectator",
            ]
            # Each irc_* key also has _merge / _single mirrors (see _bump_irc)
            irc_keys = irc_base_keys + [k + s for k in irc_base_keys for s in ("_merge", "_single")]
            for key in base_stage_keys + irc_keys:
                if key in batch_summary:
                    updates[key] = batch_summary[key]

            # Barrier histogram bins (dynamic keys): keep anything prefixed
            # with "barrier_fwd_bin_" or "barrier_bwd_bin_". Accumulates across
            # batches in exploration_stats.stats_json.
            for key, val in batch_summary.items():
                if (key.startswith("barrier_fwd_bin_")
                    or key.startswith("barrier_bwd_bin_")):
                    updates[key] = val

            for key in ["gen_diffusion_s", "gen_energy_s", "gen_irc_s", "gen_fragmentation_s", "gen_graph_add_s"]:
                if key in batch_summary:
                    updates["total_" + key] = batch_summary[key]

            if "wall_time_s" in batch_summary:
                updates["total_gen_wall_time_s"] = batch_summary["wall_time_s"]

            n_added = batch_summary.get("pipeline_added_to_graph", 0)
            updates["reactions_generative"] = n_added

            # Noise histogram
            noise_level = batch_summary.get("noise_level")
            if noise_level is not None:
                updates["noise_histogram"] = {
                    int(noise_level): {"batches": 1, "reactions": n_added}
                }

            self.db.update_exploration_stats(session, updates)
            self.db.append_batch_log(session, batch_summary)

    def record_pes_exploration_stats(self, **kwargs) -> None:
        """Record PES exploration statistics to DB."""
        with self.db.session() as session:
            updates = {
                "compounds_explored": 1,
                "total_intramol_ts_discovered": kwargs.get("n_intramol_ts", 0),
                "total_escaped_reactions": kwargs.get("n_escaped", 0),
                "total_escaped_valid": kwargs.get("n_escaped_valid", 0),
                "total_pes_time_s": kwargs.get("wall_time_s", 0.0),
                "reactions_pes_exploration": kwargs.get("n_escaped_valid", 0),
            }
            step_timings = kwargs.get("step_timings", {})
            for key in ["pes_md_s", "pes_prfo_s", "pes_validation_s", "pes_escaped_s", "pes_neb_dedup_s"]:
                if key in step_timings:
                    updates["total_" + key] = step_timings[key]

            self.db.update_exploration_stats(session, updates)

    # =========================================================================
    # Exploration Completion
    # =========================================================================

    def is_exploration_complete(self, max_compounds: int) -> bool:
        with self.db.session() as session:
            return self.db.is_exploration_complete(session, max_compounds)

    # =========================================================================
    # Utility
    # =========================================================================

    def minimum_to_conformer(self, minimum_data: dict) -> Conformer:
        """Convert a minimum data dict to a Conformer."""
        return Conformer(
            positions=torch.tensor(minimum_data["positions"], dtype=torch.float32),
            atomic_numbers=torch.tensor(minimum_data["atomic_numbers"].flatten(), dtype=torch.long),
            energy=minimum_data["energy"],
        ).center()


@dataclass
class CompoundInfo:
    """Lightweight compound info returned from registration (no PESGraph)."""
    compound_id: int
    smiles: str
    formula: str
    charge: int
    n_atoms: int
    sorted_atomic_numbers: tuple
    is_seed: bool = False
