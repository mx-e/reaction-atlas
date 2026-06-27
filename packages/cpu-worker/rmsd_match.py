"""Kabsch-aligned RMSD comparison between our PES minima and CREST conformers.

This is the post-processing step the cpu-worker runs immediately after a
successful CREST job. For each CREST conformer it computes the minimum
RMSD over all of our PES minima for the same compound, after optimal
rigid-body alignment (Kabsch / SVD).

Uses brute-force permutation of equivalent atoms for small molecules
(total permutations <= 720) because our PES minima are stored in
canonicalized atom order while CREST conformers use the input XYZ order.
For small molecules where equivalent atoms dominate (e.g., 3 oxygens in
HCO3-), Kabsch alone gives wrong results.

This module is dependency-free apart from numpy.
"""
from __future__ import annotations

import numpy as np
from itertools import permutations as iter_perms
from math import factorial


# Default match threshold in Å. Matches the upstream default (the frontend
# slider lets the user override interactively without re-querying the API).
DEFAULT_RMSD_THRESHOLD_A = 0.125


_SYM_TO_Z = {"H": 1, "He": 2, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17}


def parse_multi_xyz(xyz_text: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Parse a multi-conformer XYZ file.

    Returns list of (positions, atomic_numbers) tuples per conformer.
    Standard XYZ format: each conformer is `N\\n<comment>\\n<N atom lines>`.
    Returns an empty list if parsing fails or the file is empty.
    """
    if not xyz_text:
        return []

    lines = xyz_text.splitlines()
    conformers: list[tuple[np.ndarray, np.ndarray]] = []
    i = 0
    n = len(lines)
    while i < n:
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break
        try:
            n_atoms = int(lines[i].strip())
        except ValueError:
            break
        i += 1
        if i >= n:
            break
        i += 1  # comment line
        if i + n_atoms > n:
            break
        positions = np.empty((n_atoms, 3), dtype=np.float64)
        atomic_numbers = np.empty(n_atoms, dtype=np.int32)
        ok = True
        for k in range(n_atoms):
            parts = lines[i + k].split()
            if len(parts) < 4:
                ok = False
                break
            try:
                positions[k] = (float(parts[1]), float(parts[2]), float(parts[3]))
                atomic_numbers[k] = _SYM_TO_Z.get(parts[0], 0)
            except ValueError:
                ok = False
                break
        if not ok:
            break
        conformers.append((positions, atomic_numbers))
        i += n_atoms
    return conformers


def _kabsch_align(pos1: np.ndarray, pos2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center and Kabsch-align pos2 onto pos1. Returns (pos1_centered, pos2_aligned)."""
    p1c = pos1 - pos1.mean(axis=0)
    p2c = pos2 - pos2.mean(axis=0)
    H = p1c.T @ p2c
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return p1c, p2c
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return p1c, p2c @ R


def _hungarian_permutation(
    pos1: np.ndarray, pos2: np.ndarray, atomic_numbers: np.ndarray
) -> np.ndarray:
    """Find optimal atom permutation of pos2 to match pos1 via Hungarian algorithm."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    n = len(atomic_numbers)
    perm = np.arange(n)
    for element in np.unique(atomic_numbers):
        idx = np.where(atomic_numbers == element)[0]
        if len(idx) <= 1:
            continue
        cost = cdist(pos1[idx], pos2[idx], metric="sqeuclidean")
        row_ind, col_ind = linear_sum_assignment(cost)
        perm[idx[row_ind]] = idx[col_ind]
    return perm


_MAX_BRUTEFORCE_PERMS = 1_000_000


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray,
                atomic_numbers: np.ndarray | None = None) -> float:
    """RMSD between P and Q with permutation-aware alignment.

    Matches the GPU worker's compute_rmsd logic exactly:
    - Small molecules (total permutations <= 720): brute-force all
      permutations with fresh Kabsch for each
    - Large molecules: Kabsch + Hungarian fallback
    - No atomic_numbers: plain Kabsch
    """
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3 or P.shape[0] == 0:
        return float("inf")

    if atomic_numbers is None:
        p1c, p2a = _kabsch_align(P, Q)
        return float(np.sqrt(np.mean(np.sum((p1c - p2a) ** 2, axis=1))))

    # Build equivalent-atom groups
    groups: list[np.ndarray] = []
    total_perms = 1
    for elem in np.unique(atomic_numbers):
        idx = np.where(atomic_numbers == elem)[0]
        if len(idx) > 1:
            groups.append(idx)
            total_perms *= factorial(len(idx))

    if total_perms <= _MAX_BRUTEFORCE_PERMS and groups:
        # Brute-force: try every permutation with fresh Kabsch
        best = float("inf")

        def _enum(gi: int, perm: np.ndarray) -> None:
            nonlocal best
            if gi == len(groups):
                p1c, p2a = _kabsch_align(P, Q[perm])
                r = float(np.sqrt(np.mean(np.sum((p1c - p2a) ** 2, axis=1))))
                if r < best:
                    best = r
                return
            idx = groups[gi]
            for p in iter_perms(range(len(idx))):
                t = perm.copy()
                t[idx] = idx[list(p)]
                _enum(gi + 1, t)

        _enum(0, np.arange(len(P)))
        return best

    # Fallback: Kabsch + Hungarian (large molecules)
    p1c, p2a = _kabsch_align(P, Q)
    perm = _hungarian_permutation(p1c, p2a, atomic_numbers)
    p2f = p2a[perm]
    return float(np.sqrt(np.mean(np.sum((p1c - p2f) ** 2, axis=1))))


def compute_rmsd_match(
    our_minima_positions: list[np.ndarray],
    crest_xyz_bytes: bytes,
    threshold: float = DEFAULT_RMSD_THRESHOLD_A,
) -> dict | None:
    """Compute the per-CREST-conformer best RMSD against our PES minima.

    Returns a dict matching the JSON schema the frontend expects:
        {
          "best_rmsds": [float, ...],
          "threshold": float,
          "n_our_minima": int,
        }

    Returns None if either input is empty or unparseable so the caller can
    decide whether to leave the field NULL.
    """
    if not our_minima_positions or not crest_xyz_bytes:
        return None

    try:
        crest_text = crest_xyz_bytes.decode("utf-8", errors="replace")
    except Exception:
        return None

    parsed = parse_multi_xyz(crest_text)
    if not parsed:
        return None

    # Filter our minima to those with matching atom count
    n_atoms = parsed[0][0].shape[0]
    our_filtered = [m for m in our_minima_positions if m.shape == (n_atoms, 3)]
    if not our_filtered:
        return None

    # Get atomic numbers from first CREST conformer (same for all)
    anum = parsed[0][1] if len(parsed[0]) > 1 else None

    best_rmsds: list[float] = []
    for conf_pos, conf_anum in parsed:
        if conf_pos.shape != (n_atoms, 3):
            continue
        best = float("inf")
        for our in our_filtered:
            r = kabsch_rmsd(our, conf_pos, atomic_numbers=anum)
            if r < best:
                best = r
        if np.isfinite(best):
            best_rmsds.append(round(best, 4))

    if not best_rmsds:
        return None

    return {
        "best_rmsds": best_rmsds,
        "threshold": threshold,
        "n_our_minima": len(our_filtered),
    }
