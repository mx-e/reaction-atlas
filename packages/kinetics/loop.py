"""Background kinetics solver loop — runs inside the API container as an
asyncio task spawned at startup.

One solve per experiment per poll. Each experiment has its own advisory
lock so a) two replicas can solve different experiments in parallel, and
b) the existing single-experiment lock semantics are preserved per scope.

Per-experiment trigger: re-solve only when the network HAS CHANGED for
that experiment specifically. Adding a reaction tagged 'formose-drilldown'
does not invalidate the 'main' snapshot and vice versa — the network
version + DFT version counts on _needs_resolve are filtered by experiment.

Exactly one API replica holds a Postgres advisory lock per experiment
(KINETICS_LOCK_KEYS[exp]) and runs that experiment's solve; other replicas
just serve HTTP requests. The lock is released on shutdown / crash; the
next heartbeat picks it up.

The actual PETSc solve is CPU-bound and blocks the Python interpreter — it
runs via asyncio.to_thread() so HTTP request handling continues unimpeded.
"""

import asyncio
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from loguru import logger
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from packages.db.experiments import EXPERIMENTS
from packages.db.models import (
    KineticsSnapshot as KineticsSnapshotRow, Reaction, Compound,
)
from packages.kinetics.build import build_snapshot
from packages.kinetics.checkpoint_backends import (
    CheckpointBackend, select_backend,
)


# Per-experiment Postgres advisory lock keys. Must not collide with each
# other or with worker seed (42). Stable mapping; adding a new experiment
# requires picking a fresh key here.
KINETICS_LOCK_KEYS: dict[str, int] = {
    "main": 43,
    "formose-drilldown": 44,
}

# Knobs — defaults match upstream production setting
POLL_INTERVAL_S = int(os.environ.get("KINETICS_POLL_INTERVAL_S", "60"))
EXPLORATION_TEMPERATURE = float(os.environ.get("KINETICS_EXPLORATION_TEMPERATURE", "500"))
NETWORK_VERSION_THRESHOLD = int(os.environ.get("KINETICS_NETWORK_VERSION_THRESHOLD", "5"))
MAX_SNAPSHOT_AGE_S = int(os.environ.get("KINETICS_MAX_SNAPSHOT_AGE_S", "600"))
PREFER_DFT = os.environ.get("KINETICS_PREFER_DFT", "true").lower() == "true"

# DB checkpoint: trigger a backend-specific backup every N new compounds.
# Set to 0 to disable. The backend (gcs / local / noop) is selected via
# CHECKPOINT_BACKEND in checkpoint_backends.select_backend().
# Checkpoint is GLOBAL (whole-DB export), not per-experiment.
CHECKPOINT_EVERY_N_COMPOUNDS = int(os.environ.get("CHECKPOINT_EVERY_N_COMPOUNDS", "50"))

_last_checkpoint_count: int = 0
_checkpoint_backend: Optional[CheckpointBackend] = None


def _get_checkpoint_backend() -> CheckpointBackend:
    """Lazy-init the configured backend on first use."""
    global _checkpoint_backend
    if _checkpoint_backend is None:
        _checkpoint_backend = select_backend()
    return _checkpoint_backend


def _count_network_version(session: Session, experiment: str) -> tuple[int, int]:
    """Per-experiment (network_version, dft_version).

    Counts only reactions tagged with the given experiment. This is what
    drives the change-detection trigger — an unrelated experiment's growth
    doesn't invalidate this experiment's snapshot.
    """
    base = (
        session.query(func.count(Reaction.id))
        .filter(Reaction.experiments.any(experiment))
    )
    network_version = (
        base.filter(
            (Reaction.discovery_method != "manual_equilibrium")
            | (Reaction.discovery_method.is_(None))
        )
        .scalar()
    )
    dft_version = (
        session.query(func.count(Reaction.id))
        .filter(Reaction.experiments.any(experiment))
        .filter(Reaction.barrier_forward_separated_pbe0.isnot(None))
        .scalar()
    )
    return int(network_version or 0), int(dft_version or 0)


def _needs_resolve(session: Session, experiment: str) -> tuple[bool, str]:
    """Decide whether to re-run the solver for the given experiment.

    Re-solves when:
      - no snapshot exists for this experiment yet (and >= 2 reactions present)
      - network_version (this experiment) has grown by >= threshold
      - dft_version (this experiment) has grown
      - last snapshot for this experiment is older than max age

    Crucially: each experiment's trigger looks at its OWN snapshot history
    and its OWN reaction count. Growth in another experiment does not
    schedule a re-solve here.
    """
    latest = (
        session.query(KineticsSnapshotRow)
        .filter(KineticsSnapshotRow.experiment == experiment)
        .order_by(KineticsSnapshotRow.computed_at.desc())
        .first()
    )
    network_version, dft_version = _count_network_version(session, experiment)

    if latest is None:
        if network_version >= 2:
            return True, "no snapshot yet"
        return False, f"too few reactions ({network_version})"

    delta_network = network_version - latest.network_version
    delta_dft = dft_version - latest.n_reactions_dft
    age_s = (datetime.now(timezone.utc) - latest.computed_at).total_seconds()

    if delta_network >= NETWORK_VERSION_THRESHOLD:
        return True, f"network grew by {delta_network} reactions"
    if delta_dft > 0:
        return True, f"{delta_dft} new DFT barriers"
    if age_s > MAX_SNAPSHOT_AGE_S:
        return True, f"snapshot age {age_s:.0f}s exceeds max"
    return False, f"up to date (age={age_s:.0f}s, Δn={delta_network}, Δdft={delta_dft})"


def _persist_snapshot(
    session: Session, snapshot, network_version: int, experiment: str,
) -> None:
    row = KineticsSnapshotRow(
        network_version=network_version,
        n_reactions_dft=snapshot.n_reactions_dft,
        temperature=snapshot.temperature,
        payload_jsonb=snapshot.to_json(),
        solve_wall_time_s=snapshot.solve_wall_time_s,
        experiment=experiment,
    )
    session.add(row)
    session.flush()


def _maybe_checkpoint(session: Session) -> Optional[str]:
    """Trigger a backup if the compound count crossed a milestone.

    Global (cross-experiment) — the checkpoint is a whole-DB SQL export;
    splitting it per-experiment would just produce overlapping copies.
    Called once per outer poll iteration, regardless of how many
    experiments solved.
    """
    global _last_checkpoint_count
    if CHECKPOINT_EVERY_N_COMPOUNDS <= 0:
        return None

    backend = _get_checkpoint_backend()
    if isinstance(backend, type(None)) or backend.name == "noop":
        return None

    n_compounds = session.query(func.count(Compound.id)).scalar() or 0

    if _last_checkpoint_count == 0:
        _last_checkpoint_count = backend.scan_last_count()
        if _last_checkpoint_count == 0:
            _last_checkpoint_count = (n_compounds // CHECKPOINT_EVERY_N_COMPOUNDS) * CHECKPOINT_EVERY_N_COMPOUNDS
        logger.info(
            f"checkpoint: init _last_checkpoint_count={_last_checkpoint_count} (n={n_compounds})"
        )

    next_milestone = _last_checkpoint_count + CHECKPOINT_EVERY_N_COMPOUNDS
    logger.debug(
        f"checkpoint: _last={_last_checkpoint_count} next={next_milestone} n={n_compounds}"
    )
    if n_compounds < next_milestone:
        return None

    from packages.kinetics.notify import notify_checkpoint

    logger.info(f"DB checkpoint: {n_compounds} compounds → {backend.name}")
    success, status = backend.trigger_checkpoint(n_compounds)
    if success:
        _last_checkpoint_count = (
            (n_compounds // CHECKPOINT_EVERY_N_COMPOUNDS) * CHECKPOINT_EVERY_N_COMPOUNDS
        )
        logger.info(f"DB checkpoint triggered: {status}")
        notify_checkpoint(n_compounds, status, success=True)
    else:
        logger.error(f"DB checkpoint failed: {status}")
        notify_checkpoint(n_compounds, status, success=False, detail=status[:200])
    return status


def _maybe_send_hourly_status(session: Session) -> None:
    """Send an hourly Telegram status update with key counts.

    Counts are global (across all experiments) — the hourly status is a
    pipeline-health summary, not per-experiment monitoring.
    """
    from packages.kinetics.notify import notify_hourly_status
    from packages.db.models import (
        ExplorationStats, WorkerHeartbeat, BatchLog,
    )
    try:
        n_compounds = session.query(func.count(Compound.id)).scalar() or 0
        n_reactions = session.query(func.count(Reaction.id)).scalar() or 0

        # ExplorationStats is per-experiment now; sum the global counts
        # across all rows for the hourly summary.
        es_rows = session.query(ExplorationStats).all()
        n_gen = n_pes = merge_valid = single_valid = 0
        for es_row in es_rows:
            es = es_row.stats_json or {}
            n_gen += es.get("reactions_generative", 0)
            n_pes += es.get("reactions_pes_exploration", 0)
            merge_valid += es.get("merge_valid", 0)
            single_valid += es.get("single_valid", 0)

        cutoff = datetime.now(timezone.utc) - timedelta(minutes=20)
        n_workers = (
            session.query(func.count(WorkerHeartbeat.worker_id))
            .filter(
                WorkerHeartbeat.last_heartbeat >= cutoff,
                WorkerHeartbeat.worker_type == "exploration",
            )
            .scalar() or 0
        )
        n_batches = session.query(func.count(BatchLog.id)).scalar() or 0

        notify_hourly_status(
            n_compounds, n_reactions, n_gen, n_pes, n_workers, n_batches,
            merge_valid, single_valid,
        )
    except Exception as e:
        logger.debug(f"hourly status notification failed: {e}")


def _maybe_check_health(session: Session) -> None:
    """Check for anomalies and send alerts. Global, not per-experiment."""
    from packages.kinetics.notify import notify_workers_offline, notify_batch_crashes
    from packages.db.models import WorkerHeartbeat, BatchLog

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=20)
        n_gpu = (
            session.query(func.count(WorkerHeartbeat.worker_id))
            .filter(
                WorkerHeartbeat.last_heartbeat >= cutoff,
                WorkerHeartbeat.worker_type == "exploration",
            )
            .scalar() or 0
        )
        if n_gpu == 0:
            notify_workers_offline()

        recent = (
            session.query(BatchLog)
            .order_by(BatchLog.id.desc())
            .limit(20)
            .all()
        )
        if len(recent) >= 10:
            crashes = sum(
                1 for b in recent
                if (b.summary_json or {}).get("pipeline_survived_denoising", 1) == 0
            )
            rate = crashes / len(recent)
            if rate > 0.5:
                notify_batch_crashes(rate, len(recent))
    except Exception as e:
        logger.debug(f"health check notification failed: {e}")


def _solve_experiment_blocking(session_factory, experiment: str) -> Optional[str]:
    """Synchronous solve for one experiment — runs in a thread pool.

    Returns a human-readable status string scoped to this experiment.
    Periodic global notifications (hourly status, health checks) are NOT
    fired here; the outer loop fires them once after iterating experiments.
    """
    session = session_factory()
    try:
        should, reason = _needs_resolve(session, experiment)
        if not should:
            return f"[{experiment}] skip: {reason}"

        network_version, _ = _count_network_version(session, experiment)
        logger.info(f"kinetics solver [{experiment}]: re-solving ({reason})")
        snapshot = build_snapshot(
            session,
            temperature=EXPLORATION_TEMPERATURE,
            prefer_dft=PREFER_DFT,
            experiment=experiment,
        )
        if snapshot is None:
            return f"[{experiment}] skip: empty model"

        _persist_snapshot(session, snapshot, network_version, experiment)
        session.commit()

        return (
            f"[{experiment}] solved: {snapshot.n_reactions} reactions "
            f"({snapshot.n_reactions_dft} DFT) in {snapshot.solve_wall_time_s:.2f}s"
        )
    except Exception as e:
        session.rollback()
        logger.exception(f"kinetics solver error [{experiment}]: {e}")
        from packages.kinetics.notify import notify_error
        notify_error(f"kinetics_solver:{experiment}", str(e))
        return f"[{experiment}] error: {e}"
    finally:
        session.close()


def _global_periodic_blocking(session_factory) -> None:
    """Global per-poll work that doesn't belong to any one experiment:
    DB checkpoint, hourly status, health checks. Runs once per outer
    poll iteration regardless of how many experiments solved.
    """
    session = session_factory()
    try:
        _maybe_checkpoint(session)
        _maybe_send_hourly_status(session)
        _maybe_check_health(session)
    except Exception as e:
        logger.debug(f"global periodic block failed: {e}")
    finally:
        session.close()


async def kinetics_solver_loop(session_factory) -> None:
    """Main polling loop. Runs forever.

    Per iteration: for each experiment, try to acquire that experiment's
    advisory lock, and if held, run the solve. Locks are released between
    iterations so a crashed instance can't hold one indefinitely.
    Different experiments can be solved by different replicas in parallel.
    """
    logger.info(
        f"kinetics solver loop starting "
        f"(poll={POLL_INTERVAL_S}s, T={EXPLORATION_TEMPERATURE}K, prefer_dft={PREFER_DFT}, "
        f"experiments={sorted(EXPERIMENTS)})"
    )
    while True:
        try:
            for experiment in sorted(EXPERIMENTS):
                lock_key = KINETICS_LOCK_KEYS.get(experiment)
                if lock_key is None:
                    logger.warning(
                        f"no advisory lock key registered for experiment "
                        f"'{experiment}' — skipping. Add it to KINETICS_LOCK_KEYS."
                    )
                    continue

                lock_session = session_factory()
                got_lock = False
                try:
                    got_lock = bool(
                        lock_session.execute(
                            text("SELECT pg_try_advisory_lock(:k)"),
                            {"k": lock_key},
                        ).scalar()
                    )
                except Exception as e:
                    logger.warning(
                        f"kinetics solver lock check failed [{experiment}]: {e}"
                    )
                finally:
                    if not got_lock:
                        lock_session.close()

                if got_lock:
                    try:
                        status = await asyncio.to_thread(
                            _solve_experiment_blocking, session_factory, experiment,
                        )
                        logger.info(f"kinetics solver: {status}")
                    finally:
                        try:
                            lock_session.execute(
                                text("SELECT pg_advisory_unlock(:k)"),
                                {"k": lock_key},
                            )
                            lock_session.commit()
                        except Exception:
                            pass
                        lock_session.close()

            # Global per-iteration work (checkpoint, hourly status, health).
            # Independent of experiment locks; runs on whichever replica
            # owns the iteration.
            await asyncio.to_thread(_global_periodic_blocking, session_factory)

        except asyncio.CancelledError:
            logger.info("kinetics solver loop cancelled")
            raise
        except Exception as e:
            logger.exception(f"kinetics solver loop top-level error: {e}")

        await asyncio.sleep(POLL_INTERVAL_S)
