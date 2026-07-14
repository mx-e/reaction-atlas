#!/usr/bin/env python3
"""Provenance script for `early_network.npz` (the kinetics-demo network).

`early_network.npz` was produced by running this script against an early
checkpoint of the published exploration run:

    python extract_early_network.py checkpoint_60.sql early_network.npz

`checkpoint_60.sql` is a plain `pg_dump` of the reaction-network database at an
early stage of exploration (64 compounds). The full published database is
released separately on Zenodo (see ../../../docs/data.md); this script is kept
here so the shipped demo network is fully reproducible from it.

It parses the dump's `COPY` blocks WITHOUT restoring the database, and keeps
only the scalar columns the kinetics model builder
(`packages/kinetics/model.py::build_model_from_db`) actually reads: compound
SMILES, reactant/product links, ML/DFT barriers, and manual-equilibrium rate
constants. Every heavy bytea blob (geometries, Hessians, IRC trajectories) is
discarded, so a ~100 MB checkpoint reduces to an ~8 KB pickle-free .npz.

Usage:  python extract_early_network.py <checkpoint.sql> <out.npz>
"""
import sys
from collections import defaultdict

import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
WANT = {"compounds", "reactions", "reaction_reactants", "reaction_products"}


def unescape(field: str):
    """Undo PostgreSQL COPY text-format escaping for a single field."""
    if field == r"\N":
        return None
    if "\\" not in field:
        return field
    out, i, n = [], 0, len(field)
    while i < n:
        ch = field[i]
        if ch == "\\" and i + 1 < n:
            nxt = field[i + 1]
            out.append({"t": "\t", "n": "\n", "r": "\r", "b": "\b",
                        "f": "\f", "v": "\v", "\\": "\\"}.get(nxt, nxt))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def parse_cols(header: str):
    """`COPY public.tbl (a, b, "c") FROM stdin;` -> ['a','b','c']."""
    inner = header[header.index("(") + 1: header.rindex(")")]
    return [c.strip().strip('"') for c in inner.split(",")]


data = {t: [] for t in WANT}
cur = None
with open(SRC, "r", errors="replace") as fh:
    for line in fh:
        if cur is None:
            if line.startswith("COPY public."):
                tbl = line.split("public.", 1)[1].split(" ", 1)[0].split("(")[0].strip()
                if tbl in WANT:
                    cidx = {c: i for i, c in enumerate(parse_cols(line))}
                    cur = (tbl, cidx)
            continue
        if line.rstrip("\n") == r"\.":
            cur = None
            continue
        tbl, cidx = cur
        parts = line.rstrip("\n").split("\t")
        data[tbl].append({c: unescape(parts[i]) for c, i in cidx.items() if i < len(parts)})

id2smiles = {int(r["id"]): r["smiles"] for r in data["compounds"] if r.get("smiles")}

reactants, products = defaultdict(list), defaultdict(list)
for r in data["reaction_reactants"]:
    if r.get("reaction_id") and r.get("compound_id") and int(r["compound_id"]) in id2smiles:
        reactants[int(r["reaction_id"])].append(id2smiles[int(r["compound_id"])])
for r in data["reaction_products"]:
    if r.get("reaction_id") and r.get("compound_id") and int(r["compound_id"]) in id2smiles:
        products[int(r["reaction_id"])].append(id2smiles[int(r["compound_id"])])


def fnum(v):
    return float(v) if v is not None else float("nan")


BARRIER_COLS = ["barrier_forward", "barrier_backward",
                "barrier_forward_separated_pbe0", "barrier_backward_separated_pbe0",
                "manual_k_fwd", "manual_k_bwd"]
kept = []
for r in data["reactions"]:
    rid = int(r["id"])
    rl, pl = reactants.get(rid, []), products.get(rid, [])
    if rl and pl:
        kept.append((rid, rl, pl, r))

react_smiles, react_ptr = [], [0]
prod_smiles, prod_ptr = [], [0]
cols = {c: [] for c in BARRIER_COLS}
methods, names = [], []
for rid, rl, pl, r in kept:
    react_smiles.extend(rl); react_ptr.append(len(react_smiles))
    prod_smiles.extend(pl);  prod_ptr.append(len(prod_smiles))
    for c in BARRIER_COLS:
        cols[c].append(fnum(r.get(c)))
    methods.append(r.get("discovery_method") or "")
    names.append(r.get("name") or f"rxn_{rid}")

payload = {
    "react_smiles": np.array(react_smiles, dtype="<U80"),
    "react_ptr": np.array(react_ptr, dtype=np.int32),
    "prod_smiles": np.array(prod_smiles, dtype="<U80"),
    "prod_ptr": np.array(prod_ptr, dtype=np.int32),
    "discovery_method": np.array(methods, dtype="<U40"),
    "name": np.array(names, dtype="<U80"),
    "source_checkpoint": np.array(SRC.split("/")[-1]),
    "n_compounds": np.array(len(id2smiles)),
    "n_reactions": np.array(len(kept)),
}
for c in BARRIER_COLS:
    payload[c] = np.array(cols[c], dtype=np.float64)

np.savez_compressed(OUT, **payload)
print(f"source={SRC.split('/')[-1]} compounds={len(id2smiles)} kept_reactions={len(kept)} "
      f"manual_equilibria={sum(1 for m in methods if m == 'manual_equilibrium')} -> {OUT}")
