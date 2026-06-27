"""One-shot backfill: compute reactions.ts_ml_invalid from already-stored
ts_hessian_pbe0 blobs.

Worker code populates ts_ml_invalid as new Hessians land; this script catches
up the rows that were Hessian'd before the column existed. Idempotent — skips
rows where ts_ml_invalid is already set.

Run inside the cpu-worker SIF on a compute node:
  apptainer exec --bind ...:/credentials/sa-key.json:ro crn-cpu-worker.sif \\
    python3 /app/scripts/backfill_ts_ml_invalid.py

Or against an ad-hoc DATABASE_URL:
  DATABASE_URL=postgresql://... python3 scripts/backfill_ts_ml_invalid.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# Allow running from either the repo root or inside the SIF (/app).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/packages")  # for `cpu-worker` siblings if needed

from packages.db.serialization import deserialize_ndarray  # type: ignore

# Import the same classifier the live worker uses, so behavior matches 1:1.
try:
    from packages.cpu_worker.dft_runner import ts_ml_invalid_from_hessian  # repo layout
except ImportError:
    from dft_runner import ts_ml_invalid_from_hessian  # SIF layout (/app/dft_runner.py)


DATABASE_URL = os.environ["DATABASE_URL"]
BATCH = int(os.environ.get("BACKFILL_BATCH", "200"))


def main() -> int:
    engine = create_engine(DATABASE_URL, poolclass=NullPool, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)

    total_done = 0
    total_invalid = 0
    t_start = time.time()

    while True:
        with Session() as s:
            rows = s.execute(
                text(
                    "SELECT id, ts_hessian_pbe0, ts_conformer_positions, "
                    "       ts_conformer_atomic_numbers, ts_conformer_charge "
                    "FROM reactions "
                    "WHERE ts_hessian_pbe0 IS NOT NULL AND ts_ml_invalid IS NULL "
                    "ORDER BY id LIMIT :n"
                ),
                {"n": BATCH},
            ).fetchall()
            if not rows:
                break

            for rid, hblob, posblob, znblob, charge in rows:
                try:
                    h = np.asarray(deserialize_ndarray(hblob), dtype=np.float64)
                    pos = np.asarray(deserialize_ndarray(posblob), dtype=np.float64)
                    z = np.asarray(deserialize_ndarray(znblob)).flatten()
                except Exception as e:
                    logger.warning(f"reaction {rid}: cannot deserialize ({e}); skipping")
                    continue
                try:
                    invalid = ts_ml_invalid_from_hessian(h, pos, z, charge=int(charge))
                except Exception as e:
                    logger.warning(f"reaction {rid}: classification failed ({e}); skipping")
                    continue
                s.execute(
                    text("UPDATE reactions SET ts_ml_invalid = :v WHERE id = :rid"),
                    {"v": bool(invalid), "rid": rid},
                )
                total_done += 1
                if invalid:
                    total_invalid += 1
            s.commit()

        elapsed = time.time() - t_start
        rate = total_done / elapsed if elapsed > 0 else 0.0
        logger.info(
            f"backfill: {total_done} classified ({total_invalid} invalid, "
            f"{100*total_invalid/max(1,total_done):.1f}%), {rate:.1f}/s"
        )

    logger.info(
        f"DONE: {total_done} reactions classified, {total_invalid} invalid "
        f"({100*total_invalid/max(1,total_done):.1f}%)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
