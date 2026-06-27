#!/usr/bin/env python3
"""One-off helper: enumerate MAYGEN constitutional isomers for the carb-grid
formulas (1<=nC<=5, nO<=nC, nH<=2*nC), canonicalise with RDKit, and write
data/reference/maygen_carb.json as:

    {
      "per_formula": {
        "CH4":    ["C", ...],            # canonical SMILES list
        "CH2O":   [...],
        ...
      },
      "per_nc": {
        "1": [ ... canonical SMILES union across C1 formulas ... ],
        ...
      },
      "grid_rule": "1<=nC<=5, nO<=nC, nH<=2*nC"
    }

Used by /api/carb-coverage so the network_in_maygen column is a true
set intersection (|net ∩ may|), not the degenerate `len(net)` fallback.

Run: `python3 data/reference/build_maygen_carb.py`
(requires MAYGEN at /tmp/MAYGEN-1.8.jar and Java on PATH). Takes ~4 min.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
MAYGEN = Path(os.environ.get("MAYGEN_JAR", "/tmp/MAYGEN-1.8.jar"))
OUT = HERE / "maygen_carb.json"


def hill(nC: int, nH: int, nO: int) -> str:
    parts = []
    if nC:
        parts.append("C" if nC == 1 else f"C{nC}")
    if nH:
        parts.append("H" if nH == 1 else f"H{nH}")
    if nO:
        parts.append("O" if nO == 1 else f"O{nO}")
    return "".join(parts) or "?"


def canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    for atom in m.GetAtoms():
        atom.SetIsotope(0)
    try:
        Chem.RemoveStereochemistry(m)
    except Exception:
        pass
    try:
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def enumerate_formula(formula: str, outdir: Path) -> list[str]:
    fn = outdir / f"{formula}.smi"
    if fn.exists():
        fn.unlink()
    try:
        subprocess.run(
            ["java", "-jar", str(MAYGEN), "-f", formula, "-smi", "-o", str(outdir)],
            check=True, capture_output=True, timeout=900,
        )
    except subprocess.CalledProcessError:
        return []
    except subprocess.TimeoutExpired:
        print(f"  !! MAYGEN timeout on {formula}", file=sys.stderr)
        return []
    if not fn.exists():
        return []
    seen: set[str] = set()
    out: list[str] = []
    for line in fn.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        c = canon(s)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def main() -> int:
    if not MAYGEN.exists():
        print(f"MAYGEN jar missing at {MAYGEN}", file=sys.stderr)
        return 1

    formulas = []
    for nC in range(1, 6):
        for nO in range(0, nC + 1):
            for nH in range(0, 2 * nC + 1):
                formulas.append((hill(nC, nH, nO), nC, nH, nO))
    print(f"Enumerating MAYGEN for {len(formulas)} formulas in carb grid "
          f"(1<=nC<=5, nO<=nC, nH<=2*nC)...")

    per_formula: dict[str, list[str]] = {}
    per_nc: dict[int, set[str]] = {c: set() for c in range(1, 6)}

    with tempfile.TemporaryDirectory(prefix="maygen_carb_") as td:
        outdir = Path(td)
        for i, (f, nC, nH, nO) in enumerate(formulas, 1):
            smis = enumerate_formula(f, outdir)
            per_formula[f] = smis
            per_nc[nC].update(smis)
            if i % 40 == 0 or i == len(formulas):
                print(f"  {i}/{len(formulas)} — last {f} → {len(smis)} SMILES", flush=True)

    payload = {
        "grid_rule": "1<=nC<=5, nO<=nC, nH<=2*nC",
        "per_formula": per_formula,
        "per_nc": {str(k): sorted(v) for k, v in per_nc.items()},
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    total = sum(len(v) for v in per_nc.values())
    sizes = {f"C{k}": len(v) for k, v in per_nc.items()}
    print(f"\nWrote {OUT} — {total} canonical SMILES total")
    print(f"  per-nC: {sizes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
