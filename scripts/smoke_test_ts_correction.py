"""Smoke test: pick a small reaction with ts_ml_invalid=TRUE, run ORCA OptTS
on its TS geometry via ts_corrected_runner.run_ts_correction, print the
parsed result. No DB writes — pure ORCA exercise.

Runs inside the cpu-worker SIF on a compute node:
  apptainer exec \\
    --bind /home/local/orca:/home/local/orca:ro \\
    --bind .../ts_corrected_runner.py:/app/ts_corrected_runner.py:ro \\
    --bind .../scripts/ts_opt:/app/scripts/ts_opt:ro \\
    crn-cpu-worker.sif python3 /app/scripts/smoke_test_ts_correction.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from packages.db.serialization import deserialize_ndarray
from ts_corrected_runner import run_ts_correction


def main() -> int:
    db = os.environ.get("DATABASE_URL")
    if not db:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        return 1

    engine = create_engine(db, poolclass=NullPool, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    s = Session()

    # Pick the smallest unclaimed candidate so the ORCA wall time is shortest.
    # length(ts_conformer_atomic_numbers) is bytes; smaller = fewer atoms.
    row = s.execute(text("""
        SELECT id, ts_conformer_positions, ts_conformer_atomic_numbers,
               ts_conformer_charge,
               octet_length(ts_conformer_atomic_numbers) AS anum_bytes
        FROM reactions
        WHERE ts_ml_invalid = TRUE
          AND ts_pbe0_corrected_positions IS NULL
          AND NOT ts_pbe0_corrected_failed
          AND ts_pbe0_corrected_claimed_at IS NULL
        ORDER BY anum_bytes ASC, id ASC
        LIMIT 1
    """)).fetchone()
    if row is None:
        print("No candidate reactions (ts_ml_invalid=TRUE, no correction yet).")
        return 1

    rid = row[0]
    positions = deserialize_ndarray(row[1])
    anum = deserialize_ndarray(row[2]).flatten()
    charge = int(row[3])

    print(f"[smoke] reaction={rid} atoms={len(anum)} charge={charge}")
    print(f"[smoke] positions shape={positions.shape} dtype={positions.dtype}")
    print(f"[smoke] ORCA_BIN={os.environ.get('ORCA_BIN', 'unset')}")
    print(f"[smoke] ORCA_NPROCS={os.environ.get('ORCA_NPROCS', 'unset')}")

    # Persistent scratch — compute node /tmp gets wiped post-job. Use a
    # SMOKE_OUT_DIR env if set, else a fresh dir under $HOME.
    base = Path(os.environ.get("SMOKE_OUT_DIR",
                               os.path.expanduser("~/smoke_ts_opt_scratch")))
    out_dir = base / f"reaction_{rid}_pid{os.getpid()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[smoke] scratch dir: {out_dir}")
    result = run_ts_correction(positions, anum, charge, out_dir)

    print(f"[smoke] success={result.success}")
    print(f"[smoke] converged={result.converged}")
    print(f"[smoke] n_imag>100cm={result.n_imag_above_100cm}")
    print(f"[smoke] multiplicity={result.multiplicity}")
    print(f"[smoke] wall_s={result.wall_s:.1f}")
    if result.energy_hartree is not None:
        print(f"[smoke] energy_hartree={result.energy_hartree}")
    if result.positions is not None:
        print(f"[smoke] optimized positions[0]={result.positions[0]}")
    if result.error:
        print(f"[smoke] error: {result.error}")

    # If something went wrong, dump tails of the key ORCA artifacts for
    # debugging (the scratch dir is persistent now so this is also on disk).
    if not result.success:
        ts_out = out_dir / "ts_opt.out"
        if ts_out.exists():
            print(f"\n[smoke] --- tail of {ts_out} (last 80 lines) ---")
            tail = ts_out.read_text(errors="replace").splitlines()[-80:]
            print("\n".join(tail))
        summary = out_dir / "summary.json"
        if summary.exists():
            print(f"\n[smoke] --- {summary} ---")
            print(summary.read_text(errors="replace"))

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
