#!/usr/bin/env python3
"""DFT transition-state optimisation of a single TS guess.

Refines an approximate transition state (e.g. an MLFF-predicted TS) at
PBE0/def2-TZVPP with ORCA: an eigenvector-following saddle optimisation
(`OptTS`, a Bofill-updated quasi-Newton / P-RFO search) followed by a numerical
Hessian (`NumFreq`) to confirm the structure is a genuine first-order saddle
(exactly one imaginary mode).

Usage
-----
    python run_ts_opt.py --xyz guess_ts.xyz --charge 0 --out runs/rxn_0001

`--mult` is optional; if omitted it is guessed from electron parity (even ->
singlet, odd -> doublet). Override it for radicals / open-shell systems.

Environment
-----------
    ORCA_BIN     path to the orca binary   (default: /home/local/orca/orca)
    ORCA_NPROCS  MPI ranks for ORCA        (default: 4)

Outputs (under --out)
---------------------
    ts_opt.inp / ts_opt.out   ORCA input / log
    ts_opt.xyz                optimised transition-state geometry
    summary.json              parsed result: converged?, energy, frequencies,
                              imaginary-mode count

Exit code is 0 if the optimisation converged to a first-order saddle, else 1.
See README.md for the method, validation results and how to scale this to a
whole dataset with a SLURM array.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Level of theory used throughout the CRN TS-validation study.
LEVEL = "PBE0 def2-TZVPP def2/J RIJCOSX TightSCF"

# Imaginary modes smaller than this (in cm^-1) are treated as finite-difference
# Hessian noise rather than real negative curvature. ORCA already projects out
# translations/rotations (they print as 0.00 cm^-1).
IMAG_NOISE_CUTOFF = 100.0

_ELEMENTS = ["X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
             "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"]
_Z = {s: i for i, s in enumerate(_ELEMENTS)}

_FINAL_E = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
_OPT_OK = re.compile(r"THE OPTIMIZATION HAS CONVERGED", re.I)
_FREQ_LINE = re.compile(r"^\s*\d+:\s+(-?\d+\.\d+)\s+cm", re.M)


def orca_bin() -> str:
    p = os.environ.get("ORCA_BIN", "/home/local/orca/orca")
    if not Path(p).exists():
        raise SystemExit(f"ORCA not found at {p!r} -- set $ORCA_BIN")
    return p


def guess_multiplicity(xyz: Path, charge: int) -> int:
    """Lowest-spin guess from electron parity. Even electrons -> singlet (1),
    odd -> doublet (2). NOT valid for triplet ground states / open-shell
    systems -- pass --mult explicitly for those."""
    lines = xyz.read_text().splitlines()
    n = int(lines[0])
    electrons = sum(_Z[ln.split()[0]] for ln in lines[2:2 + n]) - charge
    return 1 if electrons % 2 == 0 else 2


def ts_opt_input(charge: int, mult: int, nprocs: int) -> str:
    """ORCA input for the TS optimisation + Hessian.

    OptTS    eigenvector-following saddle search (Bofill Hessian update)
    NumFreq  numerical Hessian on the optimised TS (-> imaginary-mode check)
    Calc_Hess true / Recalc_Hess 5  start from an exact Hessian and refresh it
             every 5 steps -- TS searches need curvature information to be
             reliable; a purely updated Hessian often drifts off the saddle.
    """
    pal = f"%pal nprocs {nprocs} end\n" if nprocs > 1 else ""
    return (
        f"! {LEVEL} OptTS NumFreq\n"
        f"{pal}"
        f"%maxcore 2000\n"
        f"%geom Calc_Hess true Recalc_Hess 5 MaxIter 200 end\n"
        f"* xyzfile {charge} {mult} ts_in.xyz\n"
    )


def imaginary_frequencies(out_text: str) -> list[float]:
    """Imaginary frequencies (cm^-1, negative) from the last Hessian in `out_text`."""
    if "VIBRATIONAL FREQUENCIES" not in out_text:
        return []
    block = out_text.rsplit("VIBRATIONAL FREQUENCIES", 1)[1].split("NORMAL MODES", 1)[0]
    return [f for f in (float(v) for v in _FREQ_LINE.findall(block)) if f < -1.0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xyz", required=True, type=Path,
                    help="approximate transition-state geometry (xyz)")
    ap.add_argument("--charge", required=True, type=int, help="total charge")
    ap.add_argument("--mult", type=int, default=None,
                    help="spin multiplicity (default: parity guess)")
    ap.add_argument("--out", required=True, type=Path,
                    help="output directory for this reaction")
    args = ap.parse_args()

    if not args.xyz.exists():
        raise SystemExit(f"input geometry not found: {args.xyz}")
    mult = args.mult if args.mult is not None else guess_multiplicity(args.xyz, args.charge)
    nprocs = int(os.environ.get("ORCA_NPROCS", "4"))
    orca = orca_bin()

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.xyz, args.out / "ts_in.xyz")
    (args.out / "ts_opt.inp").write_text(ts_opt_input(args.charge, mult, nprocs))

    print(f"[ts_opt] {args.xyz.name}  charge={args.charge} mult={mult} nprocs={nprocs}",
          flush=True)
    with (args.out / "ts_opt.out").open("w") as fh:
        subprocess.call([orca, "ts_opt.inp"], cwd=args.out,
                        stdout=fh, stderr=subprocess.STDOUT)

    text = (args.out / "ts_opt.out").read_text(errors="ignore")
    energies = _FINAL_E.findall(text)
    imag = imaginary_frequencies(text)
    n_imag_real = sum(1 for f in imag if abs(f) > IMAG_NOISE_CUTOFF)
    converged = bool(_OPT_OK.search(text))
    first_order_saddle = converged and n_imag_real == 1

    summary = {
        "input_xyz": str(args.xyz),
        "charge": args.charge,
        "multiplicity": mult,
        "level": LEVEL,
        "converged": converged,
        "final_energy_hartree": float(energies[-1]) if energies else None,
        "imaginary_frequencies_cm1": imag,
        "n_imaginary_above_100cm1": n_imag_real,
        "is_first_order_saddle": first_order_saddle,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[ts_opt] converged={converged}  imag_modes(>100cm-1)={n_imag_real}  "
          f"first-order saddle={first_order_saddle}")
    if not first_order_saddle:
        print("[ts_opt] WARNING: not a clean first-order saddle -- inspect ts_opt.out")
    return 0 if first_order_saddle else 1


if __name__ == "__main__":
    sys.exit(main())
