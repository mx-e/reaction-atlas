"""Retroactive cleanup for two PBE0 bugs found by audit.

Fix A: in-box DFT barriers (energy_R_pbe0, energy_P_pbe0, barrier_*_pbe0)
were computed at the TS-displaced trajectory[0] frame instead of the
relaxed trajectory[-1] minimum. Stored values are wrong (R/P energies
≈ TS energy, in-box barriers ≈ 0).

Fix B: Compound.energy_pbe0 cache became stale when a new lower-energy
minimum was added after the cache was filled.

Phase 1 (Fix A): NULLs energy_R_pbe0, energy_P_pbe0, barrier_forward_pbe0,
barrier_backward_pbe0 on every reaction with energy_pbe0_method set.
energy_TS_pbe0, separated barriers, and the method/at markers stay —
the per-column guards in dft_runner.py recognize TS as cached and skip
that single-point on re-enqueue. Then re-enqueues affected reactions in
dft_work_queue (status=pending).

Phase 2 (Fix B): NULLs Compound.energy_pbe0 where the cache value no
longer matches the lowest-E Minimum.energy_pbe0 (i.e. a newer lower
minimum exists that hasn't been DFT'd yet). Does NOT re-enqueue the
downstream reactions — their separated_pbe0 values stay as-is and will
refresh organically as new reactions touching the same compounds get
DFT'd. (The cost of an aggressive re-sweep is too high for what is at
worst a small bias in the separated barrier reference energy.)

Usage:
    DATABASE_URL=postgresql://...@localhost:5432/crn_cloud \\
        python3 scripts/fix_pbe0_audit.py [--dry-run]
"""
import argparse
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Report counts but do not modify the database.")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    engine = create_engine(db_url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # ---------- Phase 1: Fix A retroactive ----------
        n_affected = session.execute(text("""
            SELECT COUNT(*) FROM reactions WHERE energy_pbe0_method IS NOT NULL
        """)).scalar()
        print(f"[Phase 1] reactions with PBE0 already populated: {n_affected}")

        if not args.dry_run and n_affected > 0:
            # NULL the bogus columns. Keep energy_TS_pbe0, separated barriers,
            # method/at markers — guards in dft_runner skip TS on re-enqueue.
            session.execute(text("""
                UPDATE reactions
                SET "energy_R_pbe0" = NULL,
                    "energy_P_pbe0" = NULL,
                    barrier_forward_pbe0 = NULL,
                    barrier_backward_pbe0 = NULL
                WHERE energy_pbe0_method IS NOT NULL
            """))
            session.commit()
            print(f"[Phase 1]   nulled R/P/in-box columns on {n_affected} reactions")

            # Re-enqueue. Existing dft_work_queue rows for these reactions
            # are already in 'completed' state — flip them back to 'pending'
            # so workers pick them up. Reactions with no dft_work_queue row
            # (shouldn't happen — every non-manual reaction is auto-enqueued)
            # are insert-on-conflict-skipped.
            n_requeued = session.execute(text("""
                UPDATE dft_work_queue q
                SET status = 'pending',
                    worker_id = NULL,
                    claimed_at = NULL,
                    completed_at = NULL,
                    error_msg = NULL
                FROM reactions r
                WHERE q.reaction_id = r.id
                  AND r.energy_pbe0_method IS NOT NULL
                  AND r.discovery_method IS DISTINCT FROM 'manual_equilibrium'
            """)).rowcount
            session.commit()
            print(f"[Phase 1]   re-enqueued {n_requeued} dft_work_queue rows to 'pending'")

            # Belt-and-suspenders: any reaction that should have a queue row
            # but doesn't (legacy / migration). INSERT ... ON CONFLICT DO
            # NOTHING is safe under the uq_dft_work_reaction constraint.
            n_inserted = session.execute(text("""
                INSERT INTO dft_work_queue (reaction_id, status)
                SELECT r.id, 'pending'
                FROM reactions r
                LEFT JOIN dft_work_queue q ON q.reaction_id = r.id
                WHERE r.energy_pbe0_method IS NOT NULL
                  AND r.discovery_method IS DISTINCT FROM 'manual_equilibrium'
                  AND q.id IS NULL
                ON CONFLICT (reaction_id) DO NOTHING
            """)).rowcount
            session.commit()
            if n_inserted > 0:
                print(f"[Phase 1]   inserted {n_inserted} missing queue rows")

        # ---------- Phase 2: Fix B retroactive ----------
        # Stale = compound has cached PBE0 but the current lowest-E minimum
        # either has no PBE0 or has a different PBE0 value than the cache.
        n_stale = session.execute(text("""
            WITH lowest AS (
                SELECT DISTINCT ON (compound_id) compound_id, energy_pbe0 AS lowest_pbe0
                FROM minima
                ORDER BY compound_id, energy ASC
            )
            SELECT COUNT(*)
            FROM compounds c
            JOIN lowest l ON c.id = l.compound_id
            WHERE c.energy_pbe0 IS NOT NULL
              AND (l.lowest_pbe0 IS NULL OR l.lowest_pbe0 != c.energy_pbe0)
        """)).scalar()
        print(f"[Phase 2] compounds with stale energy_pbe0 cache: {n_stale}")

        if not args.dry_run and n_stale > 0:
            n_invalidated = session.execute(text("""
                WITH lowest AS (
                    SELECT DISTINCT ON (compound_id) compound_id, energy_pbe0 AS lowest_pbe0
                    FROM minima
                    ORDER BY compound_id, energy ASC
                )
                UPDATE compounds c
                SET energy_pbe0 = NULL,
                    energy_pbe0_method = NULL,
                    energy_pbe0_at = NULL
                FROM lowest l
                WHERE c.id = l.compound_id
                  AND c.energy_pbe0 IS NOT NULL
                  AND (l.lowest_pbe0 IS NULL OR l.lowest_pbe0 != c.energy_pbe0)
            """)).rowcount
            session.commit()
            print(f"[Phase 2]   invalidated {n_invalidated} stale compound caches")

        if args.dry_run:
            print("\n(dry-run; no changes committed)")
        else:
            print("\nDone.")

    finally:
        session.close()


if __name__ == "__main__":
    main()
