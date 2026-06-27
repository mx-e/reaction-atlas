"""Benchmark the hot DB query paths used by GPU workers.

Exercises the actual worker DB code (packages/worker/lib/db.py +
packages/worker/lib/reaction_graph.py) against a local Postgres with
a restored production snapshot. Each path is run N times and we record:
    - mean, p50, p95, p99 latency in ms
    - EXPLAIN ANALYZE for the slowest query
    - pg_stat_statements stats for cumulative comparison

Run with:
    DATABASE_URL=postgresql://crn:crn@localhost:5433/crn_cloud \
    /tmp/crn-bench/venv/bin/python scripts/bench/bench_queries.py
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DB_URL = os.environ.get("DATABASE_URL", "postgresql://crn:crn@localhost:5433/crn_cloud")

# ------------------------------------------------------------------------
# Timing harness
# ------------------------------------------------------------------------
@dataclass
class BenchResult:
    name: str
    n: int
    times_ms: list[float] = field(default_factory=list)

    def record(self, t_ms: float) -> None:
        self.times_ms.append(t_ms)

    def summary(self) -> dict:
        ts = sorted(self.times_ms)
        return {
            "name": self.name,
            "n": len(ts),
            "mean_ms": statistics.mean(ts) if ts else 0.0,
            "p50_ms": ts[len(ts) // 2] if ts else 0.0,
            "p95_ms": ts[int(len(ts) * 0.95)] if len(ts) > 20 else (ts[-1] if ts else 0.0),
            "p99_ms": ts[int(len(ts) * 0.99)] if len(ts) > 100 else (ts[-1] if ts else 0.0),
            "max_ms": ts[-1] if ts else 0.0,
            "total_ms": sum(ts),
        }


@contextmanager
def timed(result: BenchResult):
    t0 = time.perf_counter()
    yield
    result.record((time.perf_counter() - t0) * 1000)


def _print_results(results: list[BenchResult]) -> None:
    print()
    print(f"{'name':<42} {'n':>5} {'mean_ms':>10} {'p50_ms':>10} {'p95_ms':>10} {'p99_ms':>10} {'max_ms':>10} {'total_ms':>10}")
    print("-" * 112)
    for r in results:
        s = r.summary()
        print(
            f"{s['name']:<42} {s['n']:>5} "
            f"{s['mean_ms']:>10.2f} {s['p50_ms']:>10.2f} "
            f"{s['p95_ms']:>10.2f} {s['p99_ms']:>10.2f} "
            f"{s['max_ms']:>10.2f} {s['total_ms']:>10.1f}"
        )
    print()


# ------------------------------------------------------------------------
# DB size & schema info
# ------------------------------------------------------------------------
def print_db_size(engine):
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT n.nspname AS schema,
                   c.relname AS table,
                   pg_size_pretty(pg_total_relation_size(c.oid)) AS total,
                   pg_size_pretty(pg_relation_size(c.oid)) AS data,
                   pg_size_pretty(pg_indexes_size(c.oid)) AS indexes,
                   c.reltuples::bigint AS approx_rows
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'r' AND n.nspname = 'public'
            ORDER BY pg_total_relation_size(c.oid) DESC
            LIMIT 20
        """)).all()
        print("\n=== Top 20 tables by total size ===")
        print(f"{'table':<36} {'total':>12} {'data':>12} {'indexes':>12} {'~rows':>10}")
        for schema, table, total, data, idx, nrows in rows:
            print(f"{table:<36} {total:>12} {data:>12} {idx:>12} {nrows:>10}")


def print_pg_stat_statements(engine, top_n: int = 20):
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT
                calls,
                round(total_exec_time::numeric, 1) AS total_ms,
                round(mean_exec_time::numeric, 2) AS mean_ms,
                round((100.0 * total_exec_time / NULLIF(SUM(total_exec_time) OVER (), 0))::numeric, 1) AS pct,
                substring(query, 1, 180) AS query
            FROM pg_stat_statements
            WHERE query NOT LIKE 'EXPLAIN%'
              AND query NOT LIKE '%pg_stat_statements%'
              AND query NOT LIKE 'SELECT n.nspname%'
              AND query NOT LIKE 'COMMIT%'
              AND query NOT LIKE 'BEGIN%'
            ORDER BY total_exec_time DESC
            LIMIT {top_n}
        """)).all()
        print(f"\n=== pg_stat_statements top {top_n} by total time ===")
        print(f"{'pct':>5} {'calls':>7} {'total_ms':>10} {'mean_ms':>8}  query")
        for calls, total_ms, mean_ms, pct, query in rows:
            print(f"{float(pct or 0):>4.1f}% {calls:>7} {float(total_ms):>10.0f} {float(mean_ms):>8.2f}  {query.strip()[:160]}")


def reset_pg_stat_statements(engine):
    with engine.connect() as conn:
        conn.execute(text("SELECT pg_stat_statements_reset()"))
        conn.commit()


# ------------------------------------------------------------------------
# Query helpers — replicates the worker's DB patterns without importing
# the full worker module (which would pull in torch/etc.)
# ------------------------------------------------------------------------
WORK_TIMEOUT_S = 1800.0  # matches worker's DB.WORK_TIMEOUT_S


def get_compound_count(session):
    return session.execute(text("SELECT COUNT(*) FROM compounds")).scalar()


def get_pes_backlog(session):
    """3x COUNT(*) — called every worker loop iter."""
    n_pending = session.execute(text(
        "SELECT COUNT(*) FROM pes_work_queue WHERE status = 'pending'"
    )).scalar()
    n_in_progress = session.execute(text(
        "SELECT COUNT(*) FROM pes_work_queue WHERE status = 'in_progress'"
    )).scalar()
    n_total = session.execute(text("SELECT COUNT(*) FROM compounds")).scalar()
    return n_pending, n_in_progress, n_total


def is_exploration_complete(session, max_compounds=1_000_000):
    """2x COUNT(*) — called every worker loop iter."""
    n_compounds = session.execute(text("SELECT COUNT(*) FROM compounds")).scalar()
    n_pending = session.execute(text(
        "SELECT COUNT(*) FROM pes_work_queue WHERE status = 'pending'"
    )).scalar()
    return n_compounds >= max_compounds and n_pending == 0


def load_latest_kinetics_snapshot(session):
    """Loads the full JSONB payload — called every worker loop iter."""
    row = session.execute(text("""
        SELECT id, payload_jsonb
        FROM kinetics_snapshots
        ORDER BY computed_at DESC
        LIMIT 1
    """)).fetchone()
    if row is None:
        return None
    return row[1]  # payload


def staleness_probe_snapshot(session):
    """New hot path — cheap id-only probe, representative of cached calls."""
    return session.execute(text(
        "SELECT id FROM kinetics_snapshots ORDER BY computed_at DESC LIMIT 1"
    )).scalar()


def merged_loop_counts(session):
    """Combined count query used by _get_loop_counts()."""
    row = session.execute(text("""
        SELECT
          (SELECT COUNT(*) FROM compounds) AS n_compounds,
          COUNT(*) FILTER (WHERE status = 'pending') AS n_pending,
          COUNT(*) FILTER (WHERE status = 'in_progress') AS n_in_progress
        FROM pes_work_queue
    """)).fetchone()
    return int(row[0]), int(row[1]), int(row[2])


def claim_pes_work_fast(session, worker_id: str, min_compound_age_s: float = 0.0):
    """Fast-path (no conc gate): single-row FOR UPDATE SKIP LOCKED UPDATE."""
    timeout_threshold = time.time() - WORK_TIMEOUT_S
    age_clause = "AND c.created_at < now() - make_interval(secs => :max_age_s)" if min_compound_age_s > 0 else ""
    sql = f"""
        UPDATE pes_work_queue
           SET status = 'in_progress',
               worker_id = :worker_id,
               claimed_at = now()
         WHERE id = (
             SELECT q.id FROM pes_work_queue q
               JOIN compounds c ON c.id = q.compound_id
              WHERE (q.status = 'pending'
                     OR (q.status = 'in_progress'
                         AND q.claimed_at < to_timestamp(:timeout_threshold)))
                AND NOT EXISTS (
                    SELECT 1 FROM pes_work_queue other
                     WHERE other.compound_id = q.compound_id
                       AND other.status = 'in_progress'
                       AND other.claimed_at >= to_timestamp(:timeout_threshold)
                )
                {age_clause}
              ORDER BY q.id
              LIMIT 1
              FOR UPDATE OF q SKIP LOCKED
         )
         RETURNING id, compound_id, minimum_id, job_kind
    """
    params = {"worker_id": worker_id, "timeout_threshold": timeout_threshold}
    if min_compound_age_s > 0:
        params["max_age_s"] = float(min_compound_age_s)
    # Use a subtransaction so we can roll back after measuring (don't actually claim)
    session.execute(text("SAVEPOINT bench"))
    try:
        row = session.execute(text(sql), params).fetchone()
        return row
    finally:
        session.execute(text("ROLLBACK TO SAVEPOINT bench"))


def claim_pes_work_gated_scan(session, min_compound_age_s: float = 0.0, limit: int = 2000):
    """Gated path's scan query (no writes, easy to bench)."""
    timeout_threshold = time.time() - WORK_TIMEOUT_S
    age_clause = "AND c.created_at < now() - make_interval(secs => :max_age_s)" if min_compound_age_s > 0 else ""
    sql = f"""
        SELECT q.id, c.smiles
          FROM pes_work_queue q
          JOIN compounds c ON c.id = q.compound_id
         WHERE (q.status = 'pending'
                OR (q.status = 'in_progress'
                    AND q.claimed_at < to_timestamp(:timeout_threshold)))
           AND NOT EXISTS (
               SELECT 1 FROM pes_work_queue other
                WHERE other.compound_id = q.compound_id
                  AND other.status = 'in_progress'
                  AND other.claimed_at >= to_timestamp(:timeout_threshold)
           )
           {age_clause}
         ORDER BY q.id
         LIMIT :limit
    """
    params = {"timeout_threshold": timeout_threshold, "limit": limit}
    if min_compound_age_s > 0:
        params["max_age_s"] = float(min_compound_age_s)
    return session.execute(text(sql), params).fetchall()


def heartbeat(session, worker_id: str, status: str = "idle"):
    """Upsert by worker_id (PK)."""
    # Look up then update/insert — matches worker behaviour.
    sql = """
        INSERT INTO worker_heartbeats (
            worker_id, worker_type, status, started_at, last_heartbeat,
            batches_completed, pes_completed, total_wall_time_s
        )
        VALUES (:wid, 'exploration', :status, now(), now(), 0, 0, 0.0)
        ON CONFLICT (worker_id) DO UPDATE
        SET status = EXCLUDED.status,
            last_heartbeat = EXCLUDED.last_heartbeat
    """
    session.execute(text(sql), {"wid": worker_id, "status": status})


def sample_batch_pair_reaction_counts(session):
    """The pair_rxn_counts build — already supposedly fixed with joinedload.
    Simulate the post-fix SQL: load reactions + reactants + products + compound smiles.
    """
    # The joinedload version emits a single query with LEFT JOINs. Simulate in raw SQL.
    sql = """
        SELECT r.id,
               rr.compound_id AS r_cid, rc.smiles AS r_smi,
               rp.compound_id AS p_cid, pc.smiles AS p_smi
          FROM reactions r
          LEFT JOIN reaction_reactants rr ON rr.reaction_id = r.id
          LEFT JOIN compounds rc ON rc.id = rr.compound_id
          LEFT JOIN reaction_products rp ON rp.reaction_id = r.id
          LEFT JOIN compounds pc ON pc.id = rp.compound_id
         WHERE (r.discovery_method != 'manual_equilibrium' OR r.discovery_method IS NULL)
    """
    return session.execute(text(sql)).fetchall()


def load_compounds_for_sampling(session):
    """Bulk join of compounds + minima (the post-fix form)."""
    sql = """
        SELECT m.id, m.compound_id, m.local_id, m.energy,
               c.id AS cid, c.smiles, c.formula, c.n_atoms, c.charge
          FROM minima m
          JOIN compounds c ON c.id = m.compound_id
         WHERE m.local_id >= 0
    """
    return session.execute(text(sql)).fetchall()


# ------------------------------------------------------------------------
# Bench driver
# ------------------------------------------------------------------------
def explain_slow_queries(engine):
    """Run EXPLAIN (ANALYZE, BUFFERS) on the suspect queries."""
    queries = [
        ("claim_pes_work_gated_scan", f"""
            SELECT q.id, c.smiles
              FROM pes_work_queue q
              JOIN compounds c ON c.id = q.compound_id
             WHERE (q.status = 'pending'
                    OR (q.status = 'in_progress'
                        AND q.claimed_at < to_timestamp({time.time() - WORK_TIMEOUT_S})))
               AND NOT EXISTS (
                   SELECT 1 FROM pes_work_queue other
                    WHERE other.compound_id = q.compound_id
                      AND other.status = 'in_progress'
                      AND other.claimed_at >= to_timestamp({time.time() - WORK_TIMEOUT_S})
               )
             ORDER BY q.id
             LIMIT 2000
        """),
        ("get_pes_backlog_pending", "SELECT COUNT(*) FROM pes_work_queue WHERE status='pending'"),
        ("get_pes_backlog_in_progress", "SELECT COUNT(*) FROM pes_work_queue WHERE status='in_progress'"),
        ("count_compounds", "SELECT COUNT(*) FROM compounds"),
        ("load_latest_kinetics_snapshot", "SELECT id, payload_jsonb FROM kinetics_snapshots ORDER BY computed_at DESC LIMIT 1"),
        ("load_latest_snapshot_minimal", "SELECT id FROM kinetics_snapshots ORDER BY computed_at DESC LIMIT 1"),
        ("sample_batch_pair_rxn_counts", """
            SELECT r.id,
                   rr.compound_id AS r_cid, rc.smiles AS r_smi,
                   rp.compound_id AS p_cid, pc.smiles AS p_smi
              FROM reactions r
              LEFT JOIN reaction_reactants rr ON rr.reaction_id = r.id
              LEFT JOIN compounds rc ON rc.id = rr.compound_id
              LEFT JOIN reaction_products rp ON rp.reaction_id = r.id
              LEFT JOIN compounds pc ON pc.id = rp.compound_id
             WHERE (r.discovery_method != 'manual_equilibrium' OR r.discovery_method IS NULL)
        """),
        ("load_compounds_for_sampling", """
            SELECT m.id, m.compound_id, m.local_id, m.energy,
                   c.id AS cid, c.smiles, c.formula, c.n_atoms, c.charge
              FROM minima m
              JOIN compounds c ON c.id = m.compound_id
             WHERE m.local_id >= 0
        """),
    ]
    print("\n=== EXPLAIN (ANALYZE, BUFFERS) for suspect queries ===")
    with engine.connect() as conn:
        for name, q in queries:
            print(f"\n--- {name} ---")
            try:
                rows = conn.execute(text(f"EXPLAIN (ANALYZE, BUFFERS) {q}")).all()
                for r in rows:
                    print(r[0])
            except Exception as e:
                print(f"error: {e}")


def bench_hot_queries(engine, n: int = 200):
    Session = sessionmaker(bind=engine)
    results: list[BenchResult] = []

    # --- get_compound_count (plain COUNT on compounds) ---
    r = BenchResult("get_compound_count", n)
    with Session() as s:
        for _ in range(n):
            with timed(r):
                get_compound_count(s)
            s.commit()
    results.append(r)

    # --- get_pes_backlog (3x count) ---
    r = BenchResult("get_pes_backlog (3 counts)", n)
    with Session() as s:
        for _ in range(n):
            with timed(r):
                get_pes_backlog(s)
            s.commit()
    results.append(r)

    # --- is_exploration_complete (2x count) ---
    r = BenchResult("is_exploration_complete (2 counts)", n)
    with Session() as s:
        for _ in range(n):
            with timed(r):
                is_exploration_complete(s)
            s.commit()
    results.append(r)

    # --- load_latest_kinetics_snapshot (full JSONB payload) ---
    r = BenchResult("load_latest_kinetics_snapshot (full)", n)
    with Session() as s:
        for _ in range(n):
            with timed(r):
                load_latest_kinetics_snapshot(s)
            s.commit()
    results.append(r)

    # --- staleness probe (id-only, used by new cached path) ---
    r = BenchResult("snapshot_staleness_probe (id only)", n)
    with Session() as s:
        for _ in range(n):
            with timed(r):
                staleness_probe_snapshot(s)
            s.commit()
    results.append(r)

    # --- merged loop counts (1 query replaces 5) ---
    r = BenchResult("merged_loop_counts (1 query)", n)
    with Session() as s:
        for _ in range(n):
            with timed(r):
                merged_loop_counts(s)
            s.commit()
    results.append(r)

    # --- claim_pes_work fast path ---
    r = BenchResult("claim_pes_work (fast, rollback)", n)
    with Session() as s:
        s.begin()
        wid = f"bench-{random.randint(0, 10**9)}"
        for _ in range(n):
            with timed(r):
                claim_pes_work_fast(s, wid)
        s.rollback()
    results.append(r)

    # --- claim_pes_work gated scan (read-only) ---
    r = BenchResult("claim_pes_work_gated_scan", n)
    with Session() as s:
        for _ in range(n):
            with timed(r):
                claim_pes_work_gated_scan(s)
            s.commit()
    results.append(r)

    # --- heartbeat upsert ---
    r = BenchResult("heartbeat (upsert)", n)
    with Session() as s:
        s.begin()
        for i in range(n):
            wid = f"bench-worker-{i % 50}"  # match ~50 workers
            with timed(r):
                heartbeat(s, wid)
        s.rollback()
    results.append(r)

    # --- sample_batch pair_rxn_counts (joinedload-equivalent SQL) ---
    r = BenchResult("sample_batch_pair_rxn_counts", max(10, n // 5))
    with Session() as s:
        for _ in range(max(10, n // 5)):
            with timed(r):
                sample_batch_pair_reaction_counts(s)
            s.commit()
    results.append(r)

    # --- load_compounds_for_sampling (bulk join) ---
    r = BenchResult("load_compounds_for_sampling", max(10, n // 5))
    with Session() as s:
        for _ in range(max(10, n // 5)):
            with timed(r):
                load_compounds_for_sampling(s)
            s.commit()
    results.append(r)

    return results


def main():
    engine = create_engine(DB_URL, pool_pre_ping=True)

    print(f"Connecting to {DB_URL}")
    with engine.connect() as conn:
        ver = conn.execute(text("SELECT version()")).scalar()
        print(f"Postgres: {ver}")

    print_db_size(engine)

    # Fresh stats for this bench
    reset_pg_stat_statements(engine)

    n = int(os.environ.get("BENCH_N", "200"))
    print(f"\nRunning hot-path bench with n={n} iterations each...")

    # Warm up — load pages into cache so first query isn't disproportionately hit
    with engine.connect() as conn:
        conn.execute(text("SELECT COUNT(*) FROM pes_work_queue"))
        conn.execute(text("SELECT COUNT(*) FROM compounds"))
        conn.execute(text("SELECT COUNT(*) FROM minima"))
        conn.execute(text("SELECT COUNT(*) FROM kinetics_snapshots"))

    results = bench_hot_queries(engine, n=n)
    _print_results(results)

    print_pg_stat_statements(engine, top_n=15)

    explain_slow_queries(engine)


if __name__ == "__main__":
    main()
