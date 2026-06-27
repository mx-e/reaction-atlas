"""CPU worker for CRN exploration: handles both CREST conformer search and
PBE0 DFT barrier refinement.

Polls two work queues — DFT first (kinetics-critical), CREST second
(benchmark/stats). Each iteration claims one job from whichever queue has
work, runs it, and persists the result. The two job kinds share the same
container, heartbeat, and SIGTERM handling.

Usage:
    python worker.py

Environment:
    DATABASE_URL    — PostgreSQL connection string (required)
    CREST_CPUS      — CPUs for CREST (default: 4)
    CREST_EWIN      — CREST energy window kcal/mol (default: 6)
    DFT_METHOD      — PySCF functional (default: PBE0)
    DFT_BASIS       — PySCF basis set (default: def2-TZVPP)
    DFT_MAX_MEMORY_MB — PySCF mol.max_memory (default: 8000)
    WORK_TIMEOUT    — Seconds before stale in_progress items are reclaimed (default: 7200)
    POLL_INTERVAL   — Seconds between empty-queue polls (default: 5)
"""

import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

sys.path.insert(0, str(Path(__file__).parent.parent))

from packages.db.models import (
    Base,
    Compound,
    CrestResult,
    CrestWorkQueue,
    DftWorkQueue,
    Minimum,
    WorkerHeartbeat,
)
from packages.db.serialization import deserialize_ndarray, serialize_ndarray
from dft_runner import (
    HARTREE_EV,
    _pyscf_ts_hessian,
    run_dft_reaction_job,
    ts_ml_invalid_from_hessian,
)
from rmsd_match import compute_rmsd_match, kabsch_rmsd, parse_multi_xyz
from ts_corrected_runner import run_ts_correction

_ml_calc = None  # lazy-loaded ML energy calculator


def get_ml_calculator():
    """Lazy-load the md-et energy model (CPU). Cached after first call."""
    global _ml_calc
    if _ml_calc is None:
        try:
            import torch
            from md_et import load_calculator
            _ml_calc = load_calculator(variant="12l", device="cpu")
            logger.info("Loaded md-et energy model (CPU)")
        except Exception as e:
            logger.warning(f"Could not load md-et model: {e}")
    return _ml_calc


def compute_ml_energies_for_conformers(
    conformers_xyz: bytes, charge: int = 0
) -> list[float] | None:
    """Compute ML energies (eV) for each conformer in a multi-XYZ blob."""
    calc = get_ml_calculator()
    if calc is None:
        return None
    import torch
    from ase import Atoms

    parsed = parse_multi_xyz(conformers_xyz.decode("utf-8", errors="replace"))
    if not parsed:
        return None

    energies = []
    with torch.no_grad():
        for positions, atomic_numbers in parsed:
            atoms = Atoms(numbers=atomic_numbers, positions=positions)
            atoms.info["charge"] = charge
            atoms.calc = calc
            energies.append(float(atoms.get_potential_energy()))
    return energies


DATABASE_URL = os.environ["DATABASE_URL"]
CREST_CPUS = int(os.environ.get("CREST_CPUS", "4"))
CREST_EWIN = int(os.environ.get("CREST_EWIN", "6"))
WORK_TIMEOUT = int(os.environ.get("WORK_TIMEOUT", "7200"))
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))
# DFT_ONLY=1 skips the CREST claim path entirely. Used on slim arm64 builds
# (Mac native, no xtb/crest binaries upstream-published for aarch64) so the
# worker only does PBE0 single-points and never tries to claim CREST work.
DFT_ONLY = os.environ.get("DFT_ONLY", "").strip().lower() in ("1", "true", "yes")

# COMPUTE_TS_HESSIAN=1 enables the lowest-priority dataset-backfill path:
# pick a random reaction with ts_hessian_pbe0 IS NULL and compute the
# analytical PBE0 Hessian on the TS geometry. Only fires when DFT and PES
# queues (within EXPERIMENTS_TO_DRAIN) are both empty, so a flagged worker
# still does experiment-driving work first and only switches to backfill
# once everything else is drained.
COMPUTE_TS_HESSIAN = os.environ.get("COMPUTE_TS_HESSIAN", "").strip().lower() in ("1", "true", "yes")

# Reclaim window for the corrected-TS claim. ORCA OptTS+NumFreq runs ~33 min
# median on 4 cores but tails past an hour for larger systems. Must stay ≥
# TS_OPT_TIMEOUT_S in ts_corrected_runner.py — otherwise a still-running ORCA
# on a healthy worker could have its row stolen by another worker mid-flight.
TS_CORRECTED_WORK_TIMEOUT = int(os.environ.get("TS_CORRECTED_WORK_TIMEOUT", str(6 * 3600)))

# EXPERIMENT scoping. Unlike the GPU worker, cpu-workers can drain queues
# from multiple experiments because their work product (CREST conformers,
# PBE0 single-point energies) is purely geometry-dependent — the result is
# the same regardless of which experiment originated the request.
#
# Comma-separated list (e.g. EXPERIMENT="main,formose-drilldown") shares a
# single worker pool across both queues; a single value (e.g.
# EXPERIMENT="main") behaves like the GPU-worker case (dedicated).
_raw = os.environ.get("EXPERIMENT")
if not _raw:
    raise RuntimeError(
        "EXPERIMENT env var is required (e.g. 'main' or 'main,formose-drilldown'). "
        "Refusing to start a cpu-worker without an explicit scope."
    )
EXPERIMENTS_TO_DRAIN: list[str] = [s.strip() for s in _raw.split(",") if s.strip()]
if not EXPERIMENTS_TO_DRAIN:
    raise RuntimeError(f"EXPERIMENT was set but parses to no values: {_raw!r}")
from packages.db.experiments import EXPERIMENTS as _KNOWN_EXPERIMENTS
for _e in EXPERIMENTS_TO_DRAIN:
    if _e not in _KNOWN_EXPERIMENTS:
        raise RuntimeError(
            f"Unknown EXPERIMENT '{_e}'. Known: {sorted(_KNOWN_EXPERIMENTS)}."
        )
# Heartbeat / monitoring uses the first experiment as the "primary" label
# even when the worker is sharing across multiple. The current_task field
# carries enough granularity to tell what the worker is actually doing.
EXPERIMENT = EXPERIMENTS_TO_DRAIN[0]

# Graceful shutdown
shutdown_requested = False


def handle_signal(signum, frame):
    global shutdown_requested
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    shutdown_requested = True


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


ELEMENT_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar",
}


def positions_to_xyz(positions: np.ndarray, atomic_numbers: np.ndarray, charge: int = 0) -> str:
    """Convert positions + atomic numbers to XYZ format string."""
    n = len(atomic_numbers)
    lines = [str(n), f"charge={charge}"]
    for i in range(n):
        sym = ELEMENT_SYMBOLS.get(int(atomic_numbers[i]), "X")
        x, y, z = positions[i]
        lines.append(f"{sym} {x:.10f} {y:.10f} {z:.10f}")
    return "\n".join(lines) + "\n"


def parse_crest_output(crest_out: str) -> dict:
    """Parse CREST output for conformer count and S_conf.

    Handles both CREST 3.x and 2.x output formats:
      - CREST 3.x: ``ensemble entropy (J/mol K, cal/mol K) :   9.134   2.183``
        We take the second number (cal/mol K) to match upstream collect_results.py
        and the database column units (cal/(mol·K)).
      - CREST 2.x (legacy): ``Sconf = 2.183`` or ``S_conf : 2.183``
    """
    result = {"n_conformers": 0, "s_conf": None}

    for line in crest_out.split("\n"):
        # Conformer count — both CREST versions emit "N conformers" somewhere.
        m = re.search(r"(\d+)\s+conformer", line, re.IGNORECASE)
        if m:
            result["n_conformers"] = int(m.group(1))

        # CREST 3.x: "ensemble entropy (J/mol K, cal/mol K) :   9.134   2.183"
        if "ensemble entropy" in line and "cal/mol" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                vals = parts[-1].strip().split()
                if len(vals) >= 2:
                    try:
                        result["s_conf"] = float(vals[1])  # cal/(mol·K)
                        continue
                    except ValueError:
                        pass

        # CREST 2.x: "Sconf = 2.183" / "S_conf : 2.183"
        m = re.search(r"[Ss]_?conf\s*[=:]\s*([\d.]+)", line)
        if m:
            try:
                result["s_conf"] = float(m.group(1))
            except ValueError:
                pass

    return result


def run_crest(xyz_content: str, charge: int, workdir: Path) -> dict:
    """Run CREST in the given directory. Returns parsed results."""
    input_xyz = workdir / "input.xyz"
    input_xyz.write_text(xyz_content)

    cmd = [
        "crest", "input.xyz",
        "--gfn2",
        "--ewin", str(CREST_EWIN),
        "--chrg", str(charge),
        "--T", str(CREST_CPUS),
        "--noreftopo",
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    t0 = time.time()

    result = subprocess.run(
        cmd,
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=WORK_TIMEOUT - 60,  # Leave margin for DB writes
    )

    elapsed = time.time() - t0
    logger.info(f"CREST finished in {elapsed:.1f}s, exit code {result.returncode}")

    parsed = parse_crest_output(result.stdout)

    # Read conformer geometries if available
    conformers_path = workdir / "crest_conformers.xyz"
    conformers_xyz = None
    if conformers_path.exists():
        conformers_xyz = conformers_path.read_bytes()
        # Count from file if not found in output
        if parsed["n_conformers"] == 0:
            parsed["n_conformers"] = conformers_xyz.count(b"\n") // (
                len(xyz_content.strip().split("\n")) + 1
            ) or 1

    parsed["conformers_xyz"] = conformers_xyz
    parsed["output_tail"] = result.stdout[-2000:] if result.stdout else ""

    if result.returncode == 0 and conformers_xyz is not None:
        # Clean success — CREST found conformers
        parsed["success"] = True
    elif result.returncode != 0 and conformers_xyz is None:
        # CREST failed — but for rigid molecules (no rotatable bonds),
        # this is expected. Record as 1 conformer (the input geometry).
        parsed["success"] = True
        parsed["n_conformers"] = 1
        parsed["conformers_xyz"] = xyz_content.encode()
    else:
        # Unexpected state — partial results
        parsed["success"] = conformers_xyz is not None

    return parsed


def claim_crest_work(session, worker_id: str):
    """Claim a pending CREST work item from any experiment this worker drains.

    `experiment = ANY(:exps)` matches against the comma-separated list passed
    via EXPERIMENT — so a worker started with EXPERIMENT="main,formose-drilldown"
    drains both queues by FIFO across the union (lowest queue id first).
    """
    timeout_threshold = datetime.now(timezone.utc).timestamp() - WORK_TIMEOUT

    result = session.execute(
        text("""
            UPDATE crest_work_queue
            SET status = 'in_progress',
                worker_id = :worker_id,
                claimed_at = now()
            WHERE id = (
                SELECT id FROM crest_work_queue
                WHERE experiment = ANY(:experiments)
                  AND (status = 'pending'
                       OR (status = 'in_progress'
                           AND claimed_at < to_timestamp(:timeout_threshold)))
                ORDER BY id
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, compound_id, experiment
        """),
        {
            "worker_id": worker_id,
            "timeout_threshold": timeout_threshold,
            "experiments": EXPERIMENTS_TO_DRAIN,
        },
    )
    row = result.fetchone()
    if row is None:
        return None
    return {"work_id": row[0], "compound_id": row[1], "experiment": row[2]}


def claim_dft_work(session, worker_id: str):
    """Claim a pending DFT work item from any experiment this worker drains.

    Claim order: lowest barrier_forward first (most kinetically relevant
    reactions get refined earliest), tiebroken by queue id. Skips
    manual_equilibrium reactions defensively even though they should never
    be enqueued (db.create_reaction filters them).
    """
    timeout_threshold = datetime.now(timezone.utc).timestamp() - WORK_TIMEOUT

    result = session.execute(
        text("""
            UPDATE dft_work_queue
            SET status = 'in_progress',
                worker_id = :worker_id,
                claimed_at = now()
            WHERE id = (
                SELECT q.id FROM dft_work_queue q
                JOIN reactions r ON r.id = q.reaction_id
                WHERE q.experiment = ANY(:experiments)
                  AND (q.status = 'pending'
                       OR (q.status = 'in_progress'
                           AND q.claimed_at < to_timestamp(:timeout_threshold)))
                  AND (r.discovery_method != 'manual_equilibrium'
                       OR r.discovery_method IS NULL)
                ORDER BY r.barrier_forward ASC NULLS LAST, q.id ASC
                LIMIT 1
                FOR UPDATE OF q SKIP LOCKED
            )
            RETURNING id, reaction_id, experiment
        """),
        {
            "worker_id": worker_id,
            "timeout_threshold": timeout_threshold,
            "experiments": EXPERIMENTS_TO_DRAIN,
        },
    )
    row = result.fetchone()
    if row is None:
        return None
    return {"work_id": row[0], "reaction_id": row[1], "experiment": row[2]}


def _run_crest_job(session, work, heartbeat_fn) -> tuple[bool, float]:
    """Process one CREST work item end-to-end. Returns (success, wall_time_s).

    Side effects: writes CrestResult, updates CrestWorkQueue status.
    """
    compound_id = work["compound_id"]
    work_id = work["work_id"]

    compound = session.query(Compound).filter(Compound.id == compound_id).first()
    if compound is None:
        logger.warning(f"Compound {compound_id} not found, marking CREST work failed")
        session.query(CrestWorkQueue).filter(CrestWorkQueue.id == work_id).update(
            {"status": "failed", "completed_at": datetime.now(timezone.utc)}
        )
        return False, 0.0

    minimum = (
        session.query(Minimum)
        .filter(Minimum.compound_id == compound_id)
        .order_by(Minimum.energy.asc())
        .first()
    )
    if minimum is None:
        logger.warning(f"No minima for compound {compound_id}, marking CREST work failed")
        session.query(CrestWorkQueue).filter(CrestWorkQueue.id == work_id).update(
            {"status": "failed", "completed_at": datetime.now(timezone.utc)}
        )
        return False, 0.0

    positions = deserialize_ndarray(minimum.positions)
    atomic_numbers = deserialize_ndarray(compound.sorted_atomic_numbers).flatten()
    charge = compound.charge

    logger.info(
        f"CREST: {compound.smiles} (id={compound_id}, charge={charge}, atoms={len(atomic_numbers)})"
    )
    heartbeat_fn(session, "crest", compound.smiles, current_job_kind="crest")
    session.commit()
    t0 = time.time()

    xyz = positions_to_xyz(positions, atomic_numbers, charge)

    with tempfile.TemporaryDirectory(prefix="crest_") as tmpdir:
        try:
            result = run_crest(xyz, charge, Path(tmpdir))
        except subprocess.TimeoutExpired:
            logger.warning(f"CREST timed out for {compound.smiles}")
            result = {"success": False, "n_conformers": 0, "s_conf": None,
                      "conformers_xyz": None, "output_tail": "TIMEOUT"}
        except Exception as e:
            logger.error(f"CREST failed for {compound.smiles}: {e}")
            result = {"success": False, "n_conformers": 0, "s_conf": None,
                      "conformers_xyz": None, "output_tail": str(e)}

    elapsed = time.time() - t0

    if result["success"]:
        # Post-processing: per-CREST-conformer best Kabsch RMSD against all
        # of our PES minima for this compound. Cheap (pure numpy), runs while
        # the worker is still hot from CREST. None if anything goes wrong —
        # the column stays NULL and the frontend hides the RMSD card for this
        # compound. The backfill admin endpoint can recompute later.
        rmsd_match_payload = None
        try:
            our_minima_rows = (
                session.query(Minimum)
                .filter(Minimum.compound_id == compound_id)
                .all()
            )
            our_positions = [deserialize_ndarray(m.positions) for m in our_minima_rows]
            rmsd_match_payload = compute_rmsd_match(
                our_positions, result["conformers_xyz"]
            )
            if rmsd_match_payload:
                logger.info(
                    f"CREST RMSD match: {compound.smiles}: "
                    f"{len(rmsd_match_payload['best_rmsds'])} conformers vs "
                    f"{rmsd_match_payload['n_our_minima']} of our minima"
                )
        except Exception as e:
            logger.warning(f"RMSD match post-processing failed for {compound.smiles}: {e}")

        # ML energy for each CREST conformer — enables S_conf comparison
        # using the same conformer set for both CREST (xTB) and us (ML).
        try:
            ml_energies = compute_ml_energies_for_conformers(
                result["conformers_xyz"], charge=charge
            )
            if ml_energies and rmsd_match_payload:
                rmsd_match_payload["ml_energies"] = [round(e, 6) for e in ml_energies]
                logger.info(f"ML energies for {len(ml_energies)} CREST conformers of {compound.smiles}")
        except Exception as e:
            logger.warning(f"ML energy post-processing failed for {compound.smiles}: {e}")

        crest_result = CrestResult(
            compound_id=compound_id,
            n_conformers=result["n_conformers"],
            s_conf=result["s_conf"],
            conformers_xyz=result["conformers_xyz"],
            crest_output=result["output_tail"],
            charge=charge,
            rmsd_match=rmsd_match_payload,
        )
        session.add(crest_result)
        session.query(CrestWorkQueue).filter(CrestWorkQueue.id == work_id).update(
            {"status": "completed", "completed_at": datetime.now(timezone.utc)}
        )
        logger.info(f"CREST done for {compound.smiles}: {result['n_conformers']} conformers in {elapsed:.0f}s")
        return True, elapsed
    else:
        session.query(CrestWorkQueue).filter(CrestWorkQueue.id == work_id).update(
            {"status": "failed", "completed_at": datetime.now(timezone.utc)}
        )
        logger.warning(f"CREST failed for {compound.smiles}")
        return False, elapsed


def claim_ts_hessian_reaction(session, worker_id: str):
    """Claim a random reaction whose TS PBE0 Hessian is missing.

    Inline-claim model: the `reactions` table itself is the work pool; we
    set `ts_hessian_pbe0_claimed_{by,at}` atomically and return the geometry
    needed to compute. Stale claims (older than WORK_TIMEOUT) are reusable.
    Permanent failures are sticky (`ts_hessian_pbe0_failed=true`) and skipped.

    Not experiment-scoped: a Hessian is purely geometry-dependent, so any
    flagged cpu-worker may pick from the global pool of null rows. The
    *gating* of when to invoke this (DFT+PES empty in scope) is handled by
    the caller.
    """
    timeout_threshold = datetime.now(timezone.utc).timestamp() - WORK_TIMEOUT

    result = session.execute(
        text("""
            UPDATE reactions
            SET ts_hessian_pbe0_claimed_by = :worker_id,
                ts_hessian_pbe0_claimed_at = now()
            WHERE id = (
                SELECT id FROM reactions
                WHERE ts_hessian_pbe0 IS NULL
                  AND NOT ts_hessian_pbe0_failed
                  AND (ts_hessian_pbe0_claimed_at IS NULL
                       OR ts_hessian_pbe0_claimed_at < to_timestamp(:timeout_threshold))
                ORDER BY random()
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, ts_conformer_positions, ts_conformer_atomic_numbers, ts_conformer_charge
        """),
        {"worker_id": worker_id, "timeout_threshold": timeout_threshold},
    )
    row = result.fetchone()
    if row is None:
        return None
    return {
        "reaction_id": row[0],
        "ts_positions_blob": row[1],
        "ts_atomic_numbers_blob": row[2],
        "ts_charge": row[3],
    }


def claim_ts_corrected_for_reaction(session, worker_id: str, reaction_id: int) -> bool:
    """Try to atomically claim the corrected-TS slot for a specific reaction.

    Called inline right after a Hessian backfill flags ts_ml_invalid=TRUE: we
    already have the geometry hot in memory, so we'd like to fold the ORCA
    OptTS into the same worker iteration without re-fetching. Losing the race
    (rare — another worker grabbed it from the backlog drain pool first) is
    fine; we drop and the backlog handler will pick it up later.
    """
    result = session.execute(
        text("""
            UPDATE reactions
            SET ts_pbe0_corrected_claimed_by = :worker_id,
                ts_pbe0_corrected_claimed_at = now()
            WHERE id = :rid
              AND ts_ml_invalid = TRUE
              AND ts_pbe0_corrected_positions IS NULL
              AND NOT ts_pbe0_corrected_failed
              AND ts_pbe0_corrected_claimed_at IS NULL
            RETURNING id
        """),
        {"worker_id": worker_id, "rid": reaction_id},
    )
    return result.fetchone() is not None


def claim_ts_corrected_reaction(session, worker_id: str):
    """Claim a random reaction needing PBE0 TS correction (backlog drain).

    Picks from the pool of rows with ts_ml_invalid=TRUE that don't have a
    correction yet, aren't sticky-failed, and aren't actively claimed (or
    whose claim is stale beyond TS_CORRECTED_WORK_TIMEOUT). Returns the
    geometry + the existing energy_TS_pbe0 single-point on the ML TS (used
    for the ΔE column; may be NULL if DFT hasn't been run for this row).
    """
    timeout_threshold = datetime.now(timezone.utc).timestamp() - TS_CORRECTED_WORK_TIMEOUT

    result = session.execute(
        text("""
            UPDATE reactions
            SET ts_pbe0_corrected_claimed_by = :worker_id,
                ts_pbe0_corrected_claimed_at = now()
            WHERE id = (
                SELECT id FROM reactions
                WHERE ts_ml_invalid = TRUE
                  AND ts_pbe0_corrected_positions IS NULL
                  AND NOT ts_pbe0_corrected_failed
                  AND (ts_pbe0_corrected_claimed_at IS NULL
                       OR ts_pbe0_corrected_claimed_at < to_timestamp(:timeout_threshold))
                ORDER BY random()
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, ts_conformer_positions, ts_conformer_atomic_numbers,
                      ts_conformer_charge, "energy_TS_pbe0"
        """),
        {"worker_id": worker_id, "timeout_threshold": timeout_threshold},
    )
    row = result.fetchone()
    if row is None:
        return None
    return {
        "reaction_id": row[0],
        "ts_positions_blob": row[1],
        "ts_atomic_numbers_blob": row[2],
        "ts_charge": row[3],
        "energy_TS_pbe0": row[4],
    }


def _execute_ts_correction(
    session,
    worker_id: str,
    reaction_id: int,
    ml_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    charge: int,
    ml_energy_TS_pbe0_ev,
    heartbeat_fn,
) -> tuple[bool, float]:
    """Run ORCA OptTS+NumFreq on one row whose corrected-TS claim this worker
    already holds, then persist either the corrected geometry/energy/ΔE/RMSD
    columns (success) or set ts_pbe0_corrected_failed=TRUE (failure). Always
    releases the claim before returning.

    The ML PBE0 single-point energy (eV) is used as the ΔE reference; if it
    hasn't been computed for this reaction yet we store ΔE=NULL and the
    column stays sparse.
    """
    logger.info(
        f"TS correction: reaction {reaction_id} (charge={charge}, "
        f"atoms={len(atomic_numbers)}) starting ORCA OptTS+NumFreq"
    )
    heartbeat_fn(session, "ts_correction", f"reaction {reaction_id}",
                 current_job_kind="ts_correction")
    session.commit()
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix=f"ts_opt_{reaction_id}_") as tmpdir:
        result = run_ts_correction(
            ml_positions, atomic_numbers, charge, Path(tmpdir),
        )
    wall_s = time.time() - t0

    if not result.success:
        logger.warning(
            f"TS correction FAILED for reaction {reaction_id} in {wall_s:.0f}s: "
            f"{result.error}"
        )
        session.execute(
            text(
                "UPDATE reactions "
                "SET ts_pbe0_corrected_failed = TRUE, "
                "    ts_pbe0_corrected_wall_s = :wall_s, "
                "    ts_pbe0_corrected_claimed_by = NULL, "
                "    ts_pbe0_corrected_claimed_at = NULL "
                "WHERE id = :rid"
            ),
            {"rid": reaction_id, "wall_s": float(wall_s)},
        )
        session.commit()
        return False, wall_s

    # Pure rigid-body Kabsch RMSD against the ML guess (no equivalent-atom
    # permutation: ORCA preserves the input atom order so the identity
    # permutation IS the right alignment for "how much did the geometry move").
    try:
        rmsd_a = float(kabsch_rmsd(ml_positions, result.positions))
    except Exception as e:
        logger.warning(f"Kabsch RMSD failed for reaction {reaction_id}: {e}")
        rmsd_a = None

    e_corrected_ev = result.energy_hartree * HARTREE_EV
    de_ev = (e_corrected_ev - ml_energy_TS_pbe0_ev) if ml_energy_TS_pbe0_ev is not None else None

    session.execute(
        text(
            "UPDATE reactions "
            "SET ts_pbe0_corrected_positions = :positions, "
            "    ts_pbe0_corrected_energy = :energy, "
            "    ts_pbe0_corrected_at = now(), "
            "    ts_pbe0_corrected_de = :de, "
            "    ts_pbe0_corrected_rmsd = :rmsd, "
            "    ts_pbe0_corrected_wall_s = :wall_s, "
            "    ts_pbe0_corrected_claimed_by = NULL, "
            "    ts_pbe0_corrected_claimed_at = NULL "
            "WHERE id = :rid"
        ),
        {
            "positions": serialize_ndarray(result.positions),
            "energy": float(e_corrected_ev),
            "de": float(de_ev) if de_ev is not None else None,
            "rmsd": rmsd_a,
            "wall_s": float(wall_s),
            "rid": reaction_id,
        },
    )
    session.commit()
    de_str = f"{de_ev:.4f} eV" if de_ev is not None else "n/a"
    rmsd_str = f"{rmsd_a:.3f} Å" if rmsd_a is not None else "n/a"
    logger.info(
        f"TS correction OK for reaction {reaction_id} in {wall_s:.0f}s "
        f"(mult={result.multiplicity}, ΔE={de_str}, RMSD={rmsd_str})"
    )
    return True, wall_s


def _run_ts_correction_job(session, work, worker_id: str, heartbeat_fn) -> tuple[bool, float]:
    """Backlog-drain entry point. The corrected-TS claim is already held;
    deserialize + dispatch to _execute_ts_correction."""
    positions = deserialize_ndarray(work["ts_positions_blob"])
    atomic_numbers = deserialize_ndarray(work["ts_atomic_numbers_blob"]).flatten()
    charge = int(work["ts_charge"])
    ml_e = work.get("energy_TS_pbe0")  # may be None
    return _execute_ts_correction(
        session, worker_id, work["reaction_id"], positions, atomic_numbers,
        charge, ml_e, heartbeat_fn,
    )


def _run_ts_hessian_job(session, work, worker_id: str, heartbeat_fn) -> tuple[bool, float]:
    """Compute and persist the PBE0 Hessian for one claimed reaction.

    On SCF failure we mark `ts_hessian_pbe0_failed=true` (sticky) so future
    random picks skip the row. Stale-claim-retry handles transient flakes
    via WORK_TIMEOUT — no per-row retry counter needed.

    If the resulting Hessian flags the ML TS as not a true saddle
    (ts_ml_invalid=TRUE), we fold an inline ORCA PBE0 OptTS+NumFreq into
    the same iteration while the geometry is still in memory. The returned
    elapsed time covers both phases.
    """
    reaction_id = work["reaction_id"]
    positions = deserialize_ndarray(work["ts_positions_blob"])
    atomic_numbers = deserialize_ndarray(work["ts_atomic_numbers_blob"]).flatten()
    charge = int(work["ts_charge"])

    logger.info(
        f"TS Hessian: claimed reaction {reaction_id} "
        f"(charge={charge}, atoms={len(atomic_numbers)})"
    )
    # Distinct job_kind so monitoring can tell Hessian backfill apart
    # from regular DFT-barrier refinement (very different cadence + cost).
    heartbeat_fn(session, "hessian", f"hessian reaction {reaction_id}", current_job_kind="hessian")
    session.commit()
    t0 = time.time()

    try:
        h = _pyscf_ts_hessian(positions, atomic_numbers, charge)
    except Exception as e:
        logger.error(f"TS Hessian failed for reaction {reaction_id}: {e}")
        session.rollback()
        session.execute(
            text(
                "UPDATE reactions "
                "SET ts_hessian_pbe0_failed = true, "
                "    ts_hessian_pbe0_claimed_by = NULL, "
                "    ts_hessian_pbe0_claimed_at = NULL "
                "WHERE id = :rid"
            ),
            {"rid": reaction_id},
        )
        session.commit()
        return False, time.time() - t0

    elapsed = time.time() - t0
    blob = serialize_ndarray(h)
    # Classify whether the ML-predicted TS is a true saddle at DFT level.
    # Uses the same TR-projection + thresholds as lib.pes_explorer.prfo.
    # is_transition_state (the canonical exploration check). Errors here
    # shouldn't tank the Hessian writeback — log + NULL.
    try:
        ts_ml_invalid = ts_ml_invalid_from_hessian(h, positions, atomic_numbers, charge=charge)
    except Exception as e:
        logger.warning(f"ts_ml_invalid classification failed for reaction {reaction_id}: {e}")
        ts_ml_invalid = None
    session.execute(
        text(
            "UPDATE reactions "
            "SET ts_hessian_pbe0 = :blob, "
            "    ts_hessian_pbe0_at = now(), "
            "    ts_hessian_pbe0_wall_s = :wall_s, "
            "    ts_ml_invalid = :ts_ml_invalid, "
            "    ts_hessian_pbe0_claimed_by = NULL, "
            "    ts_hessian_pbe0_claimed_at = NULL "
            "WHERE id = :rid"
        ),
        {
            "blob": blob,
            "wall_s": float(elapsed),
            "ts_ml_invalid": ts_ml_invalid,
            "rid": reaction_id,
        },
    )
    logger.info(
        f"TS Hessian done for reaction {reaction_id}: shape={h.shape} in {elapsed:.0f}s "
        f"(ts_ml_invalid={ts_ml_invalid})"
    )

    # Inline TS correction: if the ML TS is not a saddle at DFT, fold the
    # PBE0 OptTS into the same iteration while the geometry is hot in
    # memory. Hessian + ts_ml_invalid are committed BEFORE we attempt the
    # claim so an ORCA crash later doesn't lose the Hessian writeback.
    # Losing the inline claim (rare race vs. backlog drainer) is harmless —
    # the drain pool picks it up.
    if ts_ml_invalid is True:
        session.commit()
        if claim_ts_corrected_for_reaction(session, worker_id, reaction_id):
            ml_e = session.execute(
                text('SELECT "energy_TS_pbe0" FROM reactions WHERE id = :rid'),
                {"rid": reaction_id},
            ).scalar()
            session.commit()
            _, ts_wall = _execute_ts_correction(
                session, worker_id, reaction_id, positions, atomic_numbers,
                charge, ml_e, heartbeat_fn,
            )
            elapsed += ts_wall

    return True, elapsed


def _run_dft_job(session, work, heartbeat_fn) -> tuple[bool, float]:
    """Process one DFT work item end-to-end. Returns (success, wall_time_s).

    Side effects: writes Compound.energy_pbe0, Minimum.energy_pbe0,
    Reaction.{energy_R/TS/P_pbe0, barrier_*_pbe0}, updates DftWorkQueue.
    """
    work_id = work["work_id"]
    reaction_id = work["reaction_id"]

    logger.info(f"DFT: claimed reaction {reaction_id} (work_id={work_id})")
    heartbeat_fn(session, "dft", f"reaction {reaction_id}", current_job_kind="dft")
    session.commit()
    t0 = time.time()

    # Heartbeat callback passed into the DFT runner so it can refresh the
    # worker's last_heartbeat between PySCF single-points (each takes 30s+).
    # Without this, the heartbeat goes stale during a multi-minute DFT job
    # and the worker disappears from the monitoring view.
    def _progress(task_desc: str) -> None:
        heartbeat_fn(session, "dft", task_desc, current_job_kind="dft")
        try:
            session.commit()
        except Exception as e:
            logger.warning(f"DFT progress heartbeat commit failed: {e}")

    try:
        ok = run_dft_reaction_job(session, work_id, reaction_id, progress_cb=_progress)
    except Exception as e:
        logger.error(f"DFT job {work_id} (reaction {reaction_id}) crashed: {e}")
        ok = False
        session.rollback()
        # Re-mark queue item failed in a fresh transaction
        session.query(DftWorkQueue).filter(DftWorkQueue.id == work_id).update(
            {"status": "failed", "completed_at": datetime.now(timezone.utc),
             "error_msg": str(e)[:1000]}
        )

    elapsed = time.time() - t0

    if ok:
        session.query(DftWorkQueue).filter(DftWorkQueue.id == work_id).update(
            {"status": "completed", "completed_at": datetime.now(timezone.utc)}
        )
        logger.info(f"DFT done for reaction {reaction_id} in {elapsed:.0f}s")
    else:
        session.query(DftWorkQueue).filter(DftWorkQueue.id == work_id).update(
            {"status": "failed", "completed_at": datetime.now(timezone.utc)}
        )

    return ok, elapsed


def main():
    worker_id = f"cpu-{uuid.uuid4().hex[:8]}"
    logger.info(f"Starting CPU worker {worker_id}")

    # NullPool: every session opens a fresh DB connection and closes it on
    # session.close(). With ~150 concurrent cpu-workers and a Cloud SQL
    # max_connections of 300 (3 reserved for superuser), the previous
    # pool_size=2 + max_overflow=1 (= up to 3 idle conns per worker) could
    # peak above the ceiling and surface as
    # `FATAL: remaining connection slots are reserved for non-replication
    # superuser connections`. Open-on-demand bounds peak conns to the count
    # of currently-running queries/jobs, which is naturally ≤ #workers.
    # IAP-tunnel connect overhead is ~tens of ms — negligible vs. minute-
    # scale DFT jobs and irrelevant to the 5-second idle poll path.
    engine = create_engine(DATABASE_URL, poolclass=NullPool, pool_pre_ping=True)

    try:
        Base.metadata.create_all(engine)
    except Exception as e:
        logger.debug(f"DDL race (harmless): {e}")

    Session = sessionmaker(bind=engine)
    jobs_completed = 0
    total_wall = 0.0

    def _heartbeat(session, status="idle", task=None, current_job_kind=None):
        now = datetime.now(timezone.utc)
        row = session.query(WorkerHeartbeat).filter(WorkerHeartbeat.worker_id == worker_id).first()
        if row is None:
            session.add(WorkerHeartbeat(
                worker_id=worker_id, worker_type="cpu", status=status,
                current_task=task, current_job_kind=current_job_kind,
                started_at=now, last_heartbeat=now,
                batches_completed=jobs_completed, total_wall_time_s=total_wall,
                experiment=EXPERIMENT,
            ))
        else:
            row.status = status
            row.current_task = task
            row.current_job_kind = current_job_kind
            row.last_heartbeat = now
            row.batches_completed = jobs_completed
            row.total_wall_time_s = total_wall

    # Initial heartbeat
    s = Session()
    _heartbeat(s, "idle", current_job_kind=None)
    s.commit()
    s.close()

    while not shutdown_requested:
        session = Session()
        try:
            # Refresh heartbeat every poll cycle so we never go stale even
            # when both queues are empty (otherwise the monitoring view loses
            # us within 2 minutes — the heartbeat freshness threshold).
            _heartbeat(session, "idle", current_job_kind=None)

            # Prefer DFT (kinetics-critical) over CREST (benchmark/stats).
            dft_work = claim_dft_work(session, worker_id)
            crest_work = None
            if dft_work is None and not DFT_ONLY:
                crest_work = claim_crest_work(session, worker_id)

            # Lowest-priority dataset-backfill: as soon as DFT and CREST
            # are empty in this worker's experiment scope, claim a Hessian.
            # We deliberately do NOT gate on pending PES — under heavy
            # exploration the PES queue is essentially never empty, and
            # gating on it left 100s of CPU workers idle for hours waiting
            # for DFT to materialize via PES. The worker re-polls DFT/CREST
            # at the top of every loop iteration, so a freshly-spawned DFT
            # job preempts the next Hessian claim within the time it takes
            # the current Hessian to finish (minutes, not hours).
            hess_work = None
            ts_corrected_work = None
            if (dft_work is None and crest_work is None
                    and COMPUTE_TS_HESSIAN):
                hess_work = claim_ts_hessian_reaction(session, worker_id)
                # Hessian queue empty → fall through to the corrected-TS
                # backlog drain. These two pools are disjoint by predicate
                # (Hessian-needed has ts_hessian_pbe0 IS NULL; corrected-
                # TS-needed has ts_ml_invalid=TRUE, which means Hessian was
                # already computed), so we never grab the same row twice.
                if hess_work is None:
                    ts_corrected_work = claim_ts_corrected_reaction(session, worker_id)
            session.commit()

            if (dft_work is None and crest_work is None
                    and hess_work is None and ts_corrected_work is None):
                session.close()
                time.sleep(POLL_INTERVAL)
                continue

            if dft_work is not None:
                ok, elapsed = _run_dft_job(session, dft_work, _heartbeat)
            elif crest_work is not None:
                ok, elapsed = _run_crest_job(session, crest_work, _heartbeat)
            elif hess_work is not None:
                ok, elapsed = _run_ts_hessian_job(session, hess_work, worker_id, _heartbeat)
            else:
                ok, elapsed = _run_ts_correction_job(session, ts_corrected_work, worker_id, _heartbeat)

            jobs_completed += 1
            total_wall += elapsed
            _heartbeat(session, "idle", current_job_kind=None)
            session.commit()

        except Exception as e:
            logger.error(f"Worker error: {e}")
            session.rollback()
        finally:
            session.close()

    # Graceful shutdown: release any claimed work + remove heartbeat
    logger.info(f"Worker {worker_id} shutting down")
    session = Session()
    try:
        session.execute(
            text("UPDATE crest_work_queue SET status = 'pending', worker_id = NULL, claimed_at = NULL "
                 "WHERE worker_id = :wid AND status = 'in_progress'"),
            {"wid": worker_id},
        )
        session.execute(
            text("UPDATE dft_work_queue SET status = 'pending', worker_id = NULL, claimed_at = NULL "
                 "WHERE worker_id = :wid AND status = 'in_progress'"),
            {"wid": worker_id},
        )
        # Release any inline Hessian claim still held — the row stays
        # eligible for the next worker's random pick. ts_hessian_pbe0 is
        # untouched (NULL means still to do). Gated on the flag so this
        # never hits the new columns on workers running pre-migration.
        if COMPUTE_TS_HESSIAN:
            session.execute(
                text("UPDATE reactions "
                     "SET ts_hessian_pbe0_claimed_by = NULL, ts_hessian_pbe0_claimed_at = NULL "
                     "WHERE ts_hessian_pbe0_claimed_by = :wid AND ts_hessian_pbe0 IS NULL"),
                {"wid": worker_id},
            )
            # Same idea for corrected-TS: clear our claim on any row that
            # isn't fully written yet (positions still NULL and not sticky-
            # failed) so the next worker can re-acquire immediately.
            session.execute(
                text("UPDATE reactions "
                     "SET ts_pbe0_corrected_claimed_by = NULL, "
                     "    ts_pbe0_corrected_claimed_at = NULL "
                     "WHERE ts_pbe0_corrected_claimed_by = :wid "
                     "  AND ts_pbe0_corrected_positions IS NULL "
                     "  AND NOT ts_pbe0_corrected_failed"),
                {"wid": worker_id},
            )
        session.query(WorkerHeartbeat).filter(WorkerHeartbeat.worker_id == worker_id).delete()
        session.commit()
    except Exception:
        pass
    finally:
        session.close()


if __name__ == "__main__":
    main()
