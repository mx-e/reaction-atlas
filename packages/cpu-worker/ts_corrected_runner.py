"""ORCA PBE0/def2-TZVPP TS optimization wrapper for the cpu-worker.

Thin adapter around scripts/ts_opt/run_ts_opt.py — the standalone script is
the single source of truth for the ORCA input deck and output parsing; this
module just handles the I/O the worker needs (ndarray in, structured result
out) and keeps ORCA's subprocess environment well-defined.

Inputs:
    positions (N,3) Å, atomic_numbers (N,), charge, scratch out_dir.
Outputs:
    TSOptResult with optimized positions, final energy (Hartree), imag-mode
    count, multiplicity used, wall time, and a success flag (True iff ORCA
    converged AND the result is a first-order saddle AND we read positions).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# Atomic-number → symbol for XYZ generation. Matches dft_runner.ELEMENT_SYMBOLS;
# kept local so this module stays import-light (no PySCF / torch pulled in).
ELEMENT_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 35: "Br", 53: "I",
}

# Standalone script path. Bind-mounted into the SIF via launch.slurm; override
# with TS_OPT_SCRIPT if the layout changes.
TS_OPT_SCRIPT = Path(os.environ.get(
    "TS_OPT_SCRIPT", "/app/scripts/ts_opt/run_ts_opt.py"
))

# Per-reaction subprocess timeout. ORCA TS-opt is ~33 min median on 4 cores for
# small organic TSs; tail extends well past an hour for larger systems. 6h —
# initial 4h cap was killing legitimate 12-15 atom OptTS+NumFreq runs (5 known
# wall-time hits at 4h on May 22), wasting hours of compute. Must stay ≤ the
# corrected-TS claim reclaim window in worker.py (TS_CORRECTED_WORK_TIMEOUT).
TS_OPT_TIMEOUT_S = int(os.environ.get("TS_OPT_TIMEOUT_S", str(6 * 3600)))


@dataclass
class TSOptResult:
    """Structured result from one ORCA OptTS+NumFreq invocation."""
    success: bool                       # converged AND first-order saddle AND have geom
    converged: bool                     # OptTS reported convergence
    n_imag_above_100cm: int             # imaginary modes with |freq| > 100 cm⁻¹
    positions: Optional[np.ndarray]     # (N,3) Å, atom order matches input — None on failure
    energy_hartree: Optional[float]     # FINAL SINGLE POINT ENERGY (Hartree)
    multiplicity: int                   # spin multiplicity used (-1 if unknown)
    wall_s: float                       # subprocess wall time
    error: Optional[str]                # human-readable failure description; None on success


def _write_xyz(path: Path, positions: np.ndarray, atomic_numbers: np.ndarray, charge: int) -> None:
    n = len(atomic_numbers)
    lines = [str(n), f"charge={charge}"]
    for z, p in zip(atomic_numbers, positions):
        sym = ELEMENT_SYMBOLS.get(int(z), "X")
        lines.append(f"{sym} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}")
    path.write_text("\n".join(lines) + "\n")


def _read_xyz_positions(path: Path) -> np.ndarray:
    """Read positions from a single-frame XYZ file (ORCA writes ts_opt.xyz this way)."""
    lines = path.read_text().splitlines()
    n = int(lines[0].strip())
    out = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        parts = lines[2 + i].split()
        out[i] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return out


def _orca_env() -> dict:
    """Subprocess env with ORCA_BIN, ORCA_NPROCS, and ORCA's dir on PATH.

    ORCA needs its own install dir on PATH so the orca driver can find its
    sub-binaries (orca_scf_mpi, orca_grad, …). We prepend rather than replace
    so the container's own /usr/bin still resolves python, etc.
    """
    env = os.environ.copy()
    orca_bin = env.get("ORCA_BIN", "/home/local/orca/orca")
    env["ORCA_BIN"] = orca_bin
    env.setdefault("ORCA_NPROCS", "4")
    orca_dir = os.path.dirname(orca_bin)
    env["PATH"] = f"{orca_dir}:{env.get('PATH', '')}"
    return env


def run_ts_correction(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    charge: int,
    out_dir: Path,
    multiplicity: Optional[int] = None,
) -> TSOptResult:
    """Run ORCA OptTS+NumFreq on the given geometry, return parsed result.

    The caller owns ``out_dir`` and is responsible for cleanup (use
    tempfile.TemporaryDirectory). On any failure path we still try to record
    the wall time and a short error string so the worker can mark the row
    failed informatively.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # The standalone script copies --xyz to <out>/ts_in.xyz, so we have to
    # write our XYZ under a different filename to avoid SameFileError.
    xyz_in = out_dir / "input.xyz"
    _write_xyz(xyz_in, positions, atomic_numbers, charge)

    cmd = [
        sys.executable, str(TS_OPT_SCRIPT),
        "--xyz", str(xyz_in),
        "--charge", str(int(charge)),
        "--out", str(out_dir),
    ]
    if multiplicity is not None:
        cmd += ["--mult", str(int(multiplicity))]

    env = _orca_env()
    t0 = time.time()
    proc_err: Optional[str] = None
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=TS_OPT_TIMEOUT_S, env=env, cwd=str(out_dir),
        )
        proc_err = (proc.stderr or "")[-500:] if proc.returncode != 0 else None
    except subprocess.TimeoutExpired:
        return TSOptResult(False, False, 0, None, None,
                           multiplicity if multiplicity is not None else -1,
                           time.time() - t0, "ORCA wall-time exceeded")
    except FileNotFoundError as e:
        return TSOptResult(False, False, 0, None, None,
                           multiplicity if multiplicity is not None else -1,
                           time.time() - t0, f"binary not found: {e}")
    except Exception as e:
        return TSOptResult(False, False, 0, None, None,
                           multiplicity if multiplicity is not None else -1,
                           time.time() - t0, f"subprocess error: {e}")
    wall_s = time.time() - t0

    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return TSOptResult(False, False, 0, None, None,
                           multiplicity if multiplicity is not None else -1,
                           wall_s,
                           f"no summary.json (ORCA crash?); stderr tail: {proc_err}")

    try:
        summary = json.loads(summary_path.read_text())
    except Exception as e:
        return TSOptResult(False, False, 0, None, None,
                           multiplicity if multiplicity is not None else -1,
                           wall_s, f"summary.json malformed: {e}")

    converged = bool(summary.get("converged"))
    n_imag = int(summary.get("n_imaginary_above_100cm1", 0))
    is_saddle = bool(summary.get("is_first_order_saddle"))
    energy = summary.get("final_energy_hartree")
    mult_used = int(summary.get("multiplicity", multiplicity if multiplicity is not None else -1))

    # ORCA writes the (last-step) optimized geometry to <jobname>.xyz on every
    # successful optimization step, so it exists for non-converged runs too —
    # we only trust it when is_saddle is True.
    opt_pos: Optional[np.ndarray] = None
    opt_xyz = out_dir / "ts_opt.xyz"
    if opt_xyz.exists():
        try:
            opt_pos = _read_xyz_positions(opt_xyz)
            if opt_pos.shape[0] != len(atomic_numbers):
                return TSOptResult(
                    False, converged, n_imag, None,
                    float(energy) if energy is not None else None,
                    mult_used, wall_s,
                    f"ts_opt.xyz atom count mismatch ({opt_pos.shape[0]} vs {len(atomic_numbers)})",
                )
        except Exception as e:
            opt_pos = None
            return TSOptResult(
                False, converged, n_imag, None,
                float(energy) if energy is not None else None,
                mult_used, wall_s, f"failed to read ts_opt.xyz: {e}",
            )

    success = is_saddle and opt_pos is not None and energy is not None
    error: Optional[str] = None
    if not success:
        error = (
            f"converged={converged} is_first_order_saddle={is_saddle} "
            f"n_imag>100cm={n_imag} have_geom={opt_pos is not None} "
            f"have_energy={energy is not None}"
        )
    return TSOptResult(
        success=success,
        converged=converged,
        n_imag_above_100cm=n_imag,
        positions=opt_pos if success else None,
        energy_hartree=float(energy) if energy is not None else None,
        multiplicity=mult_used,
        wall_s=wall_s,
        error=error,
    )
