# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "scipy",
#   "httpx",
# ]
# ///
"""Build a small dataset of close-but-distinct conformer pairs.

Pulls the live reaction graph from the API, finds compounds with >=2 visible
minima, fetches their concatenated XYZ blob, parses the minima, computes
permutation-aware RMSD between every visible pair, keeps pairs whose RMSD
falls inside the requested window, then samples N pairs across distinct
compounds for diversity.

Output:
  out_dir/pairs.csv         — manifest (one row per pair)
  out_dir/xyz/<pair>.xyz    — two-frame XYZ per pair

Run:
  uv run scripts/build_close_minima_pairs.py
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

import httpx
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# Inline permutation-aware RMSD (verbatim from pes/pes_graph.py — pulling it
# in via import would drag in torch/loguru from the rest of the package).


def _kabsch_align(pos1: np.ndarray, pos2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos1_c = pos1 - pos1.mean(axis=0)
    pos2_c = pos2 - pos2.mean(axis=0)
    H = pos1_c.T @ pos2_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return pos1_c, pos2_c @ R


def _hungarian_permutation(pos1, pos2, anum):
    n = len(anum)
    perm = np.arange(n)
    for el in np.unique(anum):
        idx = np.where(anum == el)[0]
        if len(idx) <= 1:
            continue
        cost = cdist(pos1[idx], pos2[idx], metric="sqeuclidean")
        _, col = linear_sum_assignment(cost)
        perm[idx[np.arange(len(idx))]] = idx[col]
    return perm


def _flat_brute(pos1, pos2, anum, early_stop=0.01):
    from math import factorial
    from itertools import permutations as iter_perms

    groups, total = [], 1
    for el in np.unique(anum):
        idx = np.where(anum == el)[0]
        if len(idx) > 1:
            groups.append(idx)
            total *= factorial(len(idx))
    if total > 10_000_000 or not groups:
        p1c, p2a = _kabsch_align(pos1, pos2)
        perm = _hungarian_permutation(p1c, p2a, anum)
        return perm, float(np.sqrt(np.mean(np.sum((p1c - p2a[perm]) ** 2, axis=1))))
    best_rmsd, best_perm, stopped = float("inf"), np.arange(len(pos1)), [False]

    def _enum(gi, cur):
        if stopped[0]:
            return
        if gi == len(groups):
            _, p2a = _kabsch_align(pos1, pos2[cur])
            p1c = pos1 - pos1.mean(axis=0)
            r = float(np.sqrt(np.mean(np.sum((p1c - p2a) ** 2, axis=1))))
            nonlocal best_rmsd, best_perm
            if r < best_rmsd:
                best_rmsd = r
                best_perm = cur.copy()
                if r < early_stop:
                    stopped[0] = True
            return
        idx = groups[gi]
        for p in iter_perms(range(len(idx))):
            if stopped[0]:
                return
            t = cur.copy()
            t[idx] = idx[list(p)]
            _enum(gi + 1, t)

    _enum(0, np.arange(len(pos1)))
    return best_perm, best_rmsd


def _best_permutation_rmsd(pos1, pos2, anum, early_stop=0.01):
    from math import factorial
    from itertools import permutations as iter_perms

    n = len(pos1)
    h_mask = anum == 1
    h_idx = np.where(h_mask)[0]
    heavy_idx = np.where(~h_mask)[0]
    if len(h_idx) == 0 or len(heavy_idx) == 0:
        return _flat_brute(pos1, pos2, anum, early_stop)
    p1h_c = pos1[heavy_idx] - pos1[heavy_idx].mean(axis=0)
    sv = np.linalg.svd(p1h_c, compute_uv=False)
    sv = np.pad(sv, (0, max(0, 3 - len(sv))))
    if sv[2] < max(1e-3, 0.01 * sv[0]):
        return _flat_brute(pos1, pos2, anum, early_stop)
    h_anum = anum[heavy_idx]
    h_groups, h_total = [], 1
    for el in np.unique(h_anum):
        idx = np.where(h_anum == el)[0]
        if len(idx) > 1:
            h_groups.append(idx)
            h_total *= factorial(len(idx))
    if h_total > 10_000_000:
        p1c, p2a = _kabsch_align(pos1, pos2)
        perm = _hungarian_permutation(p1c, p2a, anum)
        return perm, float(np.sqrt(np.mean(np.sum((p1c - p2a[perm]) ** 2, axis=1))))

    best_rmsd, best_perm, stopped = float("inf"), np.arange(n), [False]

    def _enum_h(gi, hperm):
        nonlocal best_rmsd, best_perm
        if stopped[0]:
            return
        if gi == len(h_groups):
            full = np.arange(n)
            for i in range(len(heavy_idx)):
                full[heavy_idx[i]] = heavy_idx[hperm[i]]
            p2p = pos2[full]
            p1h = pos1[heavy_idx]
            p2h = p2p[heavy_idx]
            p1hc = p1h - p1h.mean(axis=0)
            p2hc = p2h - p2h.mean(axis=0)
            H = p1hc.T @ p2hc
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1] *= -1
                R = Vt.T @ U.T
            c1 = pos1.mean(axis=0)
            c2 = p2p.mean(axis=0)
            p1c = pos1 - c1
            p2a = (p2p - c2) @ R
            if len(h_idx) > 1:
                hc = cdist(p1c[h_idx], p2a[h_idx], metric="sqeuclidean")
                _, col = linear_sum_assignment(hc)
                hperm_glob = h_idx[col]
                for i, hi in enumerate(h_idx):
                    full[hi] = hperm_glob[i]
                p2p = pos2[full]
                p1c, p2a = _kabsch_align(pos1, p2p)
            r = float(np.sqrt(np.mean(np.sum((p1c - p2a) ** 2, axis=1))))
            if r < best_rmsd:
                best_rmsd = r
                best_perm = full.copy()
                if r < early_stop:
                    stopped[0] = True
            return
        idx = h_groups[gi]
        for p in iter_perms(range(len(idx))):
            if stopped[0]:
                return
            t = hperm.copy()
            t[idx] = idx[list(p)]
            _enum_h(gi + 1, t)

    _enum_h(0, np.arange(len(heavy_idx)))
    return best_perm, best_rmsd


def compute_rmsd(pos1, pos2, anum):
    if anum is not None:
        _, r = _best_permutation_rmsd(pos1, pos2, anum)
        return r
    p1c, p2a = _kabsch_align(pos1, pos2)
    return float(np.sqrt(np.mean(np.sum((p1c - p2a) ** 2, axis=1))))

API = "https://pgx06.elk-court.ts.net"
# Wide seed window — API positions drift ~0.18 Å on average when re-relaxed
# under md-et v12l (model weights have moved since the DB was populated),
# so a tight seed window throws out pairs that would have landed cleanly
# inside the relaxed [0.10, 0.30] target. Cast a wider net here and let
# the v12l-relax pass do the real filtering.
RMSD_LOW = 0.05
RMSD_HIGH = 0.45
TARGET_PAIRS = 150  # over-seed so ~25 survive v12l re-relaxation
CONCURRENCY = 16
# Cap minima checked per compound. Brute-force RMSD scales O(n^2) in pair
# count, and we only want one pair per compound anyway — random-sampling
# 10 minima from a 100-minimum compound gives plenty of chances to find
# one inside the window without blocking the event loop for minutes.
MAX_MINIMA_PER_COMPOUND = 10


def parse_all_xyz(blob: str) -> list[dict]:
    """Parse the concatenated XYZ stream from /all-xyz.

    Each entry:
        <n_atoms>
        id=<int> name=<str> energy=<float> smiles=<str>
        <element> x y z
        ...
    """
    lines = blob.splitlines()
    i = 0
    out = []
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            n = int(line)
        except ValueError:
            i += 1
            continue
        header = lines[i + 1]
        meta = {}
        for part in header.split():
            if "=" in part:
                k, v = part.split("=", 1)
                meta[k] = v
        mid = int(meta.get("id", 0))
        energy = float(meta.get("energy", "nan"))
        elements = []
        coords = []
        for j in range(n):
            tokens = lines[i + 2 + j].split()
            elements.append(tokens[0])
            coords.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
        out.append(
            dict(
                id=mid,
                energy=energy,
                elements=elements,
                positions=np.asarray(coords, dtype=np.float64),
            )
        )
        i += 2 + n
    return out


_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17}


def atomic_numbers(elements: list[str]) -> np.ndarray:
    return np.array([_Z[e] for e in elements], dtype=np.int64)


async def fetch_compound(client: httpx.AsyncClient, smiles: str) -> list[dict] | None:
    try:
        r = await client.get(f"{API}/api/compound/{smiles}/all-xyz", timeout=30)
        if r.status_code != 200:
            return None
        minima = parse_all_xyz(r.text)
        visible = [m for m in minima if m["id"] >= 0]
        # Cap to avoid O(n^2) blow-up in pair enumeration. Random subsample
        # keeps the spread; deterministic via SMILES-based seed.
        if len(visible) > MAX_MINIMA_PER_COMPOUND:
            rng = np.random.default_rng(abs(hash(smiles)) % (2**31))
            visible = [
                visible[i]
                for i in rng.choice(
                    len(visible), MAX_MINIMA_PER_COMPOUND, replace=False
                )
            ]
        return visible
    except (httpx.HTTPError, ValueError):
        return None


def first_pair_in_window(minima: list[dict]) -> tuple[int, int, float] | None:
    """Return the first (a, b) RMSD that lands inside the window. We only
    take one pair per compound, so no point computing the rest."""
    if len(minima) < 2:
        return None
    anum = atomic_numbers(minima[0]["elements"])
    for a in range(len(minima)):
        for b in range(a + 1, len(minima)):
            try:
                r = compute_rmsd(minima[a]["positions"], minima[b]["positions"], anum)
            except Exception:
                continue
            if RMSD_LOW <= r <= RMSD_HIGH:
                return (a, b, float(r))
    return None


async def main(out_dir: Path, target: int) -> None:
    out_xyz = out_dir / "xyz"
    out_xyz.mkdir(parents=True, exist_ok=True)

    print(f"Fetching reaction graph from {API} …", flush=True)
    async with httpx.AsyncClient() as c:
        graph = (await c.get(f"{API}/api/reaction-graph", timeout=60)).json()
    # Neutral, non-zwitterionic only: charge=0 AND no [X+]/[X-] atom in SMILES
    # (a neutral compound with [O+] and [C-] in the SMILES is a zwitterion;
    # the student wants real neutrals).
    def _no_charged_atom(smi: str) -> bool:
        return "+]" not in smi and "-]" not in smi

    candidates = [
        n for n in graph["nodes"]
        if n.get("n_conformers", 0) >= 2
        and n.get("charge", 0) == 0
        and _no_charged_atom(n.get("smiles", ""))
    ]
    # Randomize so we cover diverse compounds early.
    rng = np.random.default_rng(0)
    rng.shuffle(candidates)
    print(f"{len(candidates)} candidate compounds (n_conformers≥2)", flush=True)

    sem = asyncio.Semaphore(CONCURRENCY)
    collected: list[dict] = []
    seen_smiles: set[str] = set()
    processed = 0

    async with httpx.AsyncClient(http2=False) as client:

        last_logged = 0

        async def one(node):
            nonlocal processed
            async with sem:
                minima = await fetch_compound(client, node["smiles"])
            if minima is None or len(minima) < 2:
                processed += 1
                return node, None
            # Run the brute-force RMSD off the event loop so a heavy compound
            # doesn't block all other fetches/computes from progressing.
            pair = await asyncio.to_thread(first_pair_in_window, minima)
            processed += 1
            return node, (minima, pair) if pair is not None else None

        tasks = [asyncio.create_task(one(n)) for n in candidates]
        for fut in asyncio.as_completed(tasks):
            node, result = await fut
            if processed - last_logged >= 200:
                last_logged = processed
                print(f"  processed {processed}, collected {len(collected)} pairs", flush=True)
            if result is None:
                continue
            minima, (a, b, r) = result
            if node["smiles"] in seen_smiles:
                continue
            seen_smiles.add(node["smiles"])
            collected.append(
                dict(
                    smiles=node["smiles"],
                    formula=node["formula"],
                    charge=node["charge"],
                    n_atoms=node["n_atoms"],
                    a=minima[a],
                    b=minima[b],
                    rmsd=r,
                )
            )
            if len(collected) >= target:
                for t in tasks:
                    if not t.done():
                        t.cancel()
                break

    print(f"\nCollected {len(collected)} pairs. Writing to {out_dir} …", flush=True)
    manifest = out_dir / "pairs.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pair_id",
                "smiles",
                "formula",
                "charge",
                "n_atoms",
                "id_a",
                "id_b",
                "energy_a_eV",
                "energy_b_eV",
                "deltaE_eV",
                "rmsd_angstrom",
                "xyz_file",
            ]
        )
        for k, p in enumerate(collected):
            pid = f"pair_{k:03d}"
            xyz_path = out_xyz / f"{pid}.xyz"
            with xyz_path.open("w") as g:
                for label, m in (("a", p["a"]), ("b", p["b"])):
                    g.write(f"{len(m['elements'])}\n")
                    g.write(
                        f"{pid}_{label} smiles={p['smiles']} id={m['id']} "
                        f"energy_eV={m['energy']:.6f}\n"
                    )
                    for el, (x, y, z) in zip(m["elements"], m["positions"]):
                        g.write(f"{el} {x:.6f} {y:.6f} {z:.6f}\n")
            w.writerow(
                [
                    pid,
                    p["smiles"],
                    p["formula"],
                    p["charge"],
                    p["n_atoms"],
                    p["a"]["id"],
                    p["b"]["id"],
                    f"{p['a']['energy']:.6f}",
                    f"{p['b']['energy']:.6f}",
                    f"{p['b']['energy'] - p['a']['energy']:.6f}",
                    f"{p['rmsd']:.4f}",
                    str(xyz_path.relative_to(out_dir)),
                ]
            )
    print(f"Done. Manifest: {manifest}")
    print(f"      XYZs:     {out_xyz}/ ({len(collected)} files)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="conformer_pairs_dataset")
    ap.add_argument("--n", type=int, default=TARGET_PAIRS)
    args = ap.parse_args()
    asyncio.run(main(Path(args.out).resolve(), args.n))
