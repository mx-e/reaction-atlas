"""Tests for permutation-aware RMSD computation."""

import numpy as np
import pytest
from math import factorial

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.pes_explorer.pes_graph import compute_rmsd, _best_permutation_rmsd, _MAX_BRUTEFORCE_PERMS


def test_identical_structures():
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    anum = np.array([6, 8, 8])
    assert compute_rmsd(pos, pos.copy(), anum) < 1e-10


def test_translated_structure():
    pos1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    pos2 = pos1 + np.array([5.0, 3.0, -2.0])
    anum = np.array([6, 8, 8])
    assert compute_rmsd(pos1, pos2, anum) < 1e-10


def test_rotated_structure():
    pos1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    # 90 degree rotation around z
    pos2 = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
    anum = np.array([6, 8, 8])
    assert compute_rmsd(pos1, pos2, anum) < 1e-10


def test_permuted_equivalent_atoms():
    """Two oxygens swapped — brute force should find the match."""
    pos1 = np.array([[0, 0, 0], [1.2, 0, 0], [-0.6, 1.0, 0]], dtype=np.float64)
    pos2 = pos1.copy()
    pos2[1], pos2[2] = pos2[2].copy(), pos2[1].copy()  # swap O atoms
    anum = np.array([6, 8, 8])

    # Without permutation awareness, Kabsch gives wrong result
    rmsd_plain = compute_rmsd(pos1, pos2)
    # With permutation awareness, should be ~0
    rmsd_perm = compute_rmsd(pos1, pos2, anum)
    assert rmsd_perm < 1e-10
    assert rmsd_plain > rmsd_perm


def test_hco3_three_oxygens_scrambled():
    """HCO3-: 3 oxygens scrambled. Brute force needed."""
    pos1 = np.array([
        [0, 0, 0],        # C
        [1.2, 0, 0],      # O1
        [-0.6, 1.04, 0],  # O2
        [-0.6, -1.04, 0], # O3
        [1.8, 0.6, 0],    # H
    ], dtype=np.float64)
    anum = np.array([6, 8, 8, 8, 1])

    # Scramble oxygens: O1→O3, O2→O1, O3→O2
    pos2 = pos1.copy()
    pos2[1] = pos1[3]
    pos2[2] = pos1[1]
    pos2[3] = pos1[2]

    rmsd = compute_rmsd(pos1, pos2, anum)
    assert rmsd < 1e-10


def test_different_structures_nonzero_rmsd():
    """Genuinely different structures should have nonzero RMSD."""
    pos1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    pos2 = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float64)
    anum = np.array([6, 8, 8])
    rmsd = compute_rmsd(pos1, pos2, anum)
    assert rmsd > 0.1


def test_no_atomic_numbers_plain_kabsch():
    """Without atomic numbers, plain Kabsch is used."""
    pos1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    rmsd = compute_rmsd(pos1, pos1.copy())
    assert rmsd < 1e-10


def test_single_atom():
    pos1 = np.array([[0, 0, 0]], dtype=np.float64)
    pos2 = np.array([[5, 5, 5]], dtype=np.float64)
    anum = np.array([6])
    # Single atom: RMSD is 0 after centering
    assert compute_rmsd(pos1, pos2, anum) < 1e-10


def test_return_permutation():
    pos1 = np.array([[0, 0, 0], [1.2, 0, 0], [-0.6, 1.0, 0]], dtype=np.float64)
    pos2 = pos1.copy()
    pos2[1], pos2[2] = pos2[2].copy(), pos2[1].copy()
    anum = np.array([6, 8, 8])

    rmsd, perm = compute_rmsd(pos1, pos2, anum, return_permutation=True)
    assert rmsd < 1e-10
    assert len(perm) == 3


def test_co_ch_o_molecule():
    """CO[CH-]O: 9 atoms (5H, 2C, 2O) = 480 perms, brute force."""
    rng = np.random.default_rng(42)
    pos1 = rng.standard_normal((9, 3))
    anum = np.array([1, 1, 1, 1, 1, 6, 6, 8, 8])

    total_perms = factorial(5) * factorial(2) * factorial(2)
    assert total_perms == 480
    assert total_perms <= _MAX_BRUTEFORCE_PERMS

    # Permute: swap H0↔H2, C5↔C6, O7↔O8
    pos2 = pos1.copy()
    pos2[0], pos2[2] = pos2[2].copy(), pos2[0].copy()
    pos2[5], pos2[6] = pos2[6].copy(), pos2[5].copy()
    pos2[7], pos2[8] = pos2[8].copy(), pos2[7].copy()

    rmsd = compute_rmsd(pos1, pos2, anum)
    assert rmsd < 1e-10


def test_noisy_duplicate():
    """Small noise should give small RMSD."""
    rng = np.random.default_rng(123)
    pos1 = rng.standard_normal((6, 3))
    anum = np.array([1, 1, 6, 6, 8, 8])

    pos2 = pos1 + rng.standard_normal((6, 3)) * 0.01
    rmsd = compute_rmsd(pos1, pos2, anum)
    assert rmsd < 0.05


def test_mismatched_shapes():
    pos1 = np.array([[0, 0, 0]], dtype=np.float64)
    pos2 = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    rmsd = compute_rmsd(pos1, pos2)
    assert rmsd == float("inf")


def test_reflection_handling():
    """Kabsch should handle improper rotations (reflections)."""
    pos1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    pos2 = pos1.copy()
    pos2[:, 2] *= -1  # reflect z
    anum = np.array([6, 8, 8, 1])
    rmsd = compute_rmsd(pos1, pos2, anum)
    assert rmsd < 1e-10


def test_brute_force_threshold():
    """Verify the threshold is set to 1M."""
    assert _MAX_BRUTEFORCE_PERMS == 1_000_000
