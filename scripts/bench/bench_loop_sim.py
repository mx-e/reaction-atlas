"""Simulate N worker-loop iterations against the local DB.

For each "worker loop tick" we run the queries the worker actually executes
every iteration of its main loop:
  1. is_exploration_complete (counts)
  2. get_compound_to_postprocess:
     a. get_pes_backlog (counts)
     b. _load_latest_kinetics_snapshot (JSONB)
     c. claim_pes_work (fast-path scan)
  3. heartbeat upsert

We run both the OLD and NEW forms back-to-back on the same DB to produce a
head-to-head comparison. Uses in-memory cache where the real code would.

Usage:
    DATABASE_URL=... /tmp/crn-bench/venv/bin/python scripts/bench/bench_loop_sim.py
"""
from __future__ import annotations

import os
import random
import statistics
import time

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DB_URL = os.environ.get("DATABASE_URL", "postgresql://crn:crn@localhost:5433/crn_cloud")
WORK_TIMEOUT_S = 1800.0


def _claim_fast(session, worker_id: str) -> None:
    """The fast-path claim — issued as a rollback-safe read to avoid mutating the DB."""
    t = time.time() - WORK_TIMEOUT_S
    session.execute(text("SAVEPOINT bench"))
    try:
        session.execute(text("""
            UPDATE pes_work_queue
               SET status = 'in_progress',
                   worker_id = :wid,
                   claimed_at = now()
             WHERE id = (
                 SELECT q.id FROM pes_work_queue q
                   JOIN compounds c ON c.id = q.compound_id
                  WHERE (q.status = 'pending'
                         OR (q.status = 'in_progress' AND q.claimed_at < to_timestamp(:t)))
                    AND NOT EXISTS (
                        SELECT 1 FROM pes_work_queue other
                         WHERE other.compound_id = q.compound_id
                           AND other.status = 'in_progress'
                           AND other.claimed_at >= to_timestamp(:t)
                    )
                  ORDER BY q.id
                  LIMIT 1
                  FOR UPDATE OF q SKIP LOCKED
             )
             RETURNING id
        """), {"wid": worker_id, "t": t}).fetchone()
    finally:
        session.execute(text("ROLLBACK TO SAVEPOINT bench"))


def _heartbeat(session, worker_id: str) -> None:
    session.execute(text("""
        INSERT INTO worker_heartbeats (
            worker_id, worker_type, status, started_at, last_heartbeat,
            batches_completed, pes_completed, total_wall_time_s)
        VALUES (:wid, 'exploration', 'idle', now(), now(), 0, 0, 0.0)
        ON CONFLICT (worker_id) DO UPDATE SET last_heartbeat = EXCLUDED.last_heartbeat
    """), {"wid": worker_id})


def old_loop_tick(session, worker_id: str) -> None:
    # 1. is_exploration_complete = 2 counts
    session.execute(text("SELECT COUNT(*) FROM compounds")).scalar()
    session.execute(text("SELECT COUNT(*) FROM pes_work_queue WHERE status='pending'")).scalar()

    # 2a. get_pes_backlog = 3 counts
    session.execute(text("SELECT COUNT(*) FROM pes_work_queue WHERE status='pending'")).scalar()
    session.execute(text("SELECT COUNT(*) FROM pes_work_queue WHERE status='in_progress'")).scalar()
    session.execute(text("SELECT COUNT(*) FROM compounds")).scalar()

    # 2b. snapshot (full payload — 3.85 MB!)
    session.execute(text(
        "SELECT id, payload_jsonb FROM kinetics_snapshots ORDER BY computed_at DESC LIMIT 1"
    )).fetchone()

    # 2c. claim_pes_work fast-path
    _claim_fast(session, worker_id)

    # 3. heartbeat (upsert)
    _heartbeat(session, worker_id)


class NewWorkerState:
    """Holds the caches the real code introduces."""
    def __init__(self) -> None:
        self.snapshot_cache: tuple[int, dict] | None = None
        self.loop_counts_cache: tuple[float, tuple[int, int, int]] | None = None
    def loop_counts(self, session, ttl_s: float = 0.5):
        now = time.monotonic()
        if self.loop_counts_cache and now - self.loop_counts_cache[0] < ttl_s:
            return self.loop_counts_cache[1]
        row = session.execute(text("""
            SELECT (SELECT COUNT(*) FROM compounds),
                   COUNT(*) FILTER (WHERE status='pending'),
                   COUNT(*) FILTER (WHERE status='in_progress')
              FROM pes_work_queue
        """)).fetchone()
        counts = (int(row[0]), int(row[1]), int(row[2]))
        self.loop_counts_cache = (now, counts)
        return counts
    def load_snapshot(self, session):
        latest_id = session.execute(text(
            "SELECT id FROM kinetics_snapshots ORDER BY computed_at DESC LIMIT 1"
        )).scalar()
        if latest_id is None:
            return None
        if self.snapshot_cache and self.snapshot_cache[0] == latest_id:
            return self.snapshot_cache[1]
        row = session.execute(text(
            "SELECT payload_jsonb FROM kinetics_snapshots WHERE id = :i"
        ), {"i": latest_id}).fetchone()
        self.snapshot_cache = (latest_id, row[0] if row else None)
        return self.snapshot_cache[1]


def new_loop_tick(session, state: NewWorkerState, worker_id: str) -> None:
    # 1 + 2a: merged counts (cached within 0.5s)
    state.loop_counts(session)
    # is_exploration_complete also hits the same cache — no second DB call.
    state.loop_counts(session)

    # 2b: cached staleness probe; full load only on change (simulate rare miss)
    state.load_snapshot(session)

    # 2c + 3: same
    _claim_fast(session, worker_id)
    _heartbeat(session, worker_id)


def time_series(fn, n: int) -> list[float]:
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def summarize(name: str, times_ms: list[float]) -> dict:
    s = sorted(times_ms)
    return {
        "name": name,
        "n": len(s),
        "mean_ms": statistics.mean(s),
        "p50_ms": s[len(s) // 2],
        "p95_ms": s[int(len(s) * 0.95)] if len(s) > 20 else s[-1],
        "p99_ms": s[int(len(s) * 0.99)] if len(s) > 100 else s[-1],
        "total_ms": sum(s),
    }


def main() -> None:
    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)

    n = int(os.environ.get("BENCH_N", "200"))

    # Warm caches — load relevant pages
    with engine.connect() as c:
        c.execute(text("SELECT COUNT(*) FROM pes_work_queue"))
        c.execute(text("SELECT id FROM kinetics_snapshots ORDER BY computed_at DESC LIMIT 1"))

    # --- OLD ---
    wid = f"bench-old-{random.randint(0, 10**9)}"
    with Session() as s:
        s.begin()  # everything rolled back at end to avoid mutating DB state
        old_times = time_series(lambda: old_loop_tick(s, wid), n)
        s.rollback()

    # --- NEW (cached) ---
    state = NewWorkerState()
    wid = f"bench-new-{random.randint(0, 10**9)}"
    with Session() as s:
        s.begin()
        new_times = time_series(lambda: new_loop_tick(s, state, wid), n)
        s.rollback()

    # --- NEW (worst case: every loop invalidates cache) ---
    wid = f"bench-new-cold-{random.randint(0, 10**9)}"
    with Session() as s:
        s.begin()
        def cold_tick():
            cold = NewWorkerState()  # fresh cache each tick
            new_loop_tick(s, cold, wid)
        cold_times = time_series(cold_tick, n)
        s.rollback()

    old_s = summarize("OLD per-loop (full snapshot every iter)", old_times)
    new_s = summarize("NEW per-loop (cache hit, typical)", new_times)
    cold_s = summarize("NEW per-loop (cold cache = worst case)", cold_times)

    print(f"{'name':<44} {'n':>4} {'mean':>7} {'p50':>7} {'p95':>7} {'p99':>7} {'total_ms':>10}")
    print("-" * 92)
    for r in (old_s, new_s, cold_s):
        print(
            f"{r['name']:<44} {r['n']:>4} "
            f"{r['mean_ms']:>7.2f} {r['p50_ms']:>7.2f} "
            f"{r['p95_ms']:>7.2f} {r['p99_ms']:>7.2f} {r['total_ms']:>10.1f}"
        )

    speedup = old_s["mean_ms"] / new_s["mean_ms"] if new_s["mean_ms"] > 0 else float("inf")
    print(f"\nSpeedup (mean):  NEW cached  is {speedup:.1f}x faster than OLD")
    db_qps_old = 50 * 1000 / old_s["mean_ms"]
    db_qps_new = 50 * 1000 / new_s["mean_ms"]
    print(f"At 50 workers × 1 loop/sec, DB sustains:")
    print(f"  OLD: needs {50 * old_s['mean_ms']:.0f} ms/sec of DB work = {50 * old_s['mean_ms'] / 10:.0f}% CPU")
    print(f"  NEW: needs {50 * new_s['mean_ms']:.0f} ms/sec of DB work = {50 * new_s['mean_ms'] / 10:.0f}% CPU")


if __name__ == "__main__":
    main()
