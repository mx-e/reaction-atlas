"""
Utility functions for molecular structure manipulation and comparison.

Includes RMSD calculations, structure validation, and batch processing utilities.
"""

import os
from typing import TYPE_CHECKING
import torch
from ase import Atoms
import schnetpack.properties as Props
import numpy as np
import rmsd
from tqdm import tqdm

from lib.types import Conformer

if TYPE_CHECKING:
    from lib.compound import Compound


TETROSE_LIMIT = {"C": 5, "H": 10, "O": 5}  # Maximum allowed composition (pentose-sized)
HIGH_RMSD = 1e8  # Used when structures are incompatible


def atomic_symbol(atomic_num: int) -> str:
    """Convert atomic number to symbol (C, H, O only)."""
    if atomic_num == 1:
        return "H"
    elif atomic_num == 6:
        return "C"
    elif atomic_num == 8:
        return "O"
    else:
        return "X"  # Fallback if not C, H, or O


def check_tetrose_limit(atomic_numbers: torch.Tensor) -> bool:
    """
    Check we do not exceed the formula C5H10O5 (pentose-sized).
    We ignore any non-(C,H,O) element or treat it as "X" and forbid those entirely.
    """
    # Count occurrences
    composition = {"C": 0, "H": 0, "O": 0, "X": 0}
    if atomic_numbers.numel() <= 1:
        return True
    for at in atomic_numbers:
        symb = atomic_symbol(int(at.item()))
        if symb not in composition:
            composition["X"] += 1
        else:
            composition[symb] += 1

    # If there's any "X" or we exceed the known limit, return False
    if composition["X"] > 0:
        return False
    for k in ["C", "H", "O"]:
        if composition[k] > TETROSE_LIMIT[k]:
            return False
    return True


# === RMSD CALCULATIONS ===


def rmsd_distance(
    pos_A: np.ndarray, pos_B: np.ndarray, atomic_numbers: np.ndarray | None = None
) -> float:
    """
    Compute RMSD between two position arrays after optimal alignment.

    Uses Kabsch algorithm for rotation alignment. When atomic_numbers is
    provided, additionally uses the Hungarian algorithm to find the optimal
    permutation of same-element atoms (permutation-invariant RMSD).
    """
    from lib.pes_explorer.pes_graph import compute_rmsd

    return compute_rmsd(pos_A, pos_B, atomic_numbers)


def positions_similar(
    pos_A: np.ndarray,
    pos_B: np.ndarray,
    rmsd_th: float,
    return_rmsd: bool = False,
    atomic_numbers: np.ndarray | None = None,
) -> bool | tuple[bool, float]:
    """Check if two position arrays are similar within RMSD threshold."""
    if pos_A.shape != pos_B.shape:
        if return_rmsd:
            return False, HIGH_RMSD
        return False
    rmsd_val = rmsd_distance(pos_A, pos_B, atomic_numbers)
    if return_rmsd:
        return rmsd_val < rmsd_th, rmsd_val
    return rmsd_val < rmsd_th


def rmsd_between_conformers(conf1: Conformer, conf2: Conformer) -> float:
    """
    Compute RMSD distance between two conformers.

    Returns HIGH_RMSD if structures are incompatible (different atom count or types).
    """
    c1 = conf1.to_numpy()
    c2 = conf2.to_numpy()
    pos1 = c1.positions
    pos2 = c2.positions
    z1 = c1.atomic_numbers.flatten()
    z2 = c2.atomic_numbers.flatten()

    if pos1.shape[0] != pos2.shape[0]:
        return HIGH_RMSD

    if sorted(z1.tolist()) != sorted(z2.tolist()):
        return HIGH_RMSD

    return rmsd_distance(pos1.copy(), pos2.copy(), z1)


def min_rmsd_conformer_to_compound(
    conformer: Conformer,
    compound_minima: np.ndarray,
    compound_atomic_numbers: np.ndarray,
) -> float:
    """
    Compute minimum RMSD from a conformer to any minimum in a compound.

    Args:
        conformer: Conformer to compare
        compound_minima: Array of shape (n_minima, n_atoms, 3)
        compound_atomic_numbers: Atomic numbers for the compound

    Returns:
        Minimum RMSD value, or HIGH_RMSD if incompatible
    """
    conf = conformer.to_numpy()
    positions = conf.positions
    atomic_numbers = conf.atomic_numbers.flatten()

    # Check if structures are compatible
    if positions.shape[0] != compound_minima.shape[1]:
        return HIGH_RMSD

    if not np.array_equal(atomic_numbers, compound_atomic_numbers.flatten()):
        return HIGH_RMSD

    # Compute RMSD to each minimum and return the minimum
    z = compound_atomic_numbers.flatten()
    min_rmsd_val = HIGH_RMSD
    for i in range(compound_minima.shape[0]):
        rmsd_val = rmsd_distance(positions.copy(), compound_minima[i].copy(), z)
        if rmsd_val < min_rmsd_val:
            min_rmsd_val = rmsd_val

    return min_rmsd_val


def min_rmsd_positions_to_compound(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    compound: "Compound",
) -> float:
    """
    Compute minimum RMSD from positions to any minimum in a compound.

    Args:
        positions: Positions array, shape (n_atoms, 3)
        atomic_numbers: Atomic numbers array (must match compound's sorted_atomic_numbers)
        compound: Compound object containing minima to compare against

    Returns:
        Minimum RMSD to any conformer in the compound
    """
    atomic_numbers = np.asarray(atomic_numbers).flatten()

    # Atom ordering must match - if not, this is a bug in the calling code
    assert tuple(atomic_numbers) == compound.sorted_atomic_numbers, (
        f"Atomic number mismatch: got {tuple(atomic_numbers)}, "
        f"expected {compound.sorted_atomic_numbers}"
    )

    # Compute RMSD to each minimum
    min_rmsd_val = HIGH_RMSD
    for minimum in compound.pes_graph.minima.values():
        rmsd_val = rmsd_distance(positions.copy(), minimum.positions.copy(), atomic_numbers)
        min_rmsd_val = min(min_rmsd_val, rmsd_val)

    return min_rmsd_val


# === KNN RMSD UTILITIES ===


def get_knn_to_trajectory_rmsd_distance(
    positions: np.ndarray,
    trajectory: np.ndarray,
    k: int,
    atomic_numbers: np.ndarray | None = None,
) -> float:
    """Get k-th nearest neighbor RMSD distance from positions to trajectory."""
    n_samples = trajectory.shape[0]
    assert 1 <= k <= n_samples, "k must be between 1 and the number of samples"
    rmsd_distances = np.zeros((n_samples))
    for i in range(n_samples):
        rmsd_distances[i] = rmsd_distance(positions.copy(), trajectory[i].copy(), atomic_numbers)
    return np.sort(rmsd_distances)[k - 1]


def get_avg_knn_to_trajectory_rmsd_distance(
    positions: np.ndarray,
    trajectory: np.ndarray,
    k: int,
    atomic_numbers: np.ndarray | None = None,
) -> float:
    """Get average of k-nearest neighbor RMSD distances."""
    return np.mean(get_knn_to_trajectory_rmsd_distance(positions, trajectory, k, atomic_numbers))


def is_similar_to_trajectory(
    positions: np.ndarray,
    trajectory: np.ndarray,
    k: int,
    rmsd_th: float,
    return_rmsd: bool = False,
    atomic_numbers: np.ndarray | None = None,
) -> bool | tuple[bool, float]:
    """Check if positions are similar to a trajectory based on KNN RMSD."""
    rmsd_val = get_avg_knn_to_trajectory_rmsd_distance(positions, trajectory, k, atomic_numbers)
    if return_rmsd:
        return rmsd_val < rmsd_th, rmsd_val
    return rmsd_val < rmsd_th


def knn_rmsd_distances(
    conformers: np.ndarray,
    k: int,
    max_evaluations: int = 5_000_000,
    seed: int | None = None,
    atomic_numbers: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute k-nearest neighbor RMSD distances for conformers.

    If the number of pairs exceeds max_evaluations, pairs are randomly sampled.

    Args:
        conformers: Array of shape (n_samples, n_atoms, 3)
        k: Number of nearest neighbors to consider
        max_evaluations: Maximum number of distance evaluations
        seed: Random seed for reproducible sampling
        atomic_numbers: If provided, use Hungarian algorithm for permutation-invariant RMSD
    """
    rng = np.random.default_rng(seed)
    n_samples = conformers.shape[0]
    assert 1 <= k <= n_samples, "k must be between 1 and the number of samples"

    # Generate all unique pairs (i, j) where i < j
    pairs = [(i, j) for i in range(n_samples) for j in range(i + 1, n_samples)]

    # Sample if needed
    if len(pairs) > max_evaluations:
        pairs = rng.choice(pairs, size=max_evaluations, replace=False).tolist()

    # Single loop over pairs
    sample_distances = [[] for _ in range(n_samples)]
    for i, j in tqdm(pairs, desc="Calculating RMSD distances"):
        rmsd_val = rmsd_distance(conformers[i].copy(), conformers[j].copy(), atomic_numbers)
        sample_distances[i].append(rmsd_val)
        sample_distances[j].append(rmsd_val)

    # Compute k-NN distance per sample
    knn_distances = np.zeros(n_samples)
    for i in range(n_samples):
        distances = np.array(sample_distances[i])
        if len(distances) < k:
            knn_distances[i] = np.max(distances) if len(distances) > 0 else 0.0
        else:
            knn_distances[i] = np.sort(distances)[k - 1]
    return knn_distances


# === CONVERSION UTILITIES ===


def conformer_to_atoms(conformer: Conformer) -> Atoms:
    """Convert a Conformer to ASE Atoms object."""
    return conformer.to_ase_atoms()


def write_xyz_to_string(images, comment="", fmt="%22.15f"):
    """Write ASE Atoms to XYZ format string."""
    output = []
    comment = comment.rstrip()
    if "\n" in comment:
        raise ValueError("Comment line should not have line breaks.")
    for atoms in images:
        natoms = len(atoms)
        output.append(f"{natoms}\n{comment}")
        for s, (x, y, z) in zip(atoms.symbols, atoms.positions):
            output.append(f"{s} {fmt % x} {fmt % y} {fmt % z}")
    return "\n".join(output)


# === BATCH PROCESSING ===


def collate_mol_batch(batch):
    """
    Build batch from systems and properties & apply padding

    Args:
        batch (list): List of molecule dicts

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    device = batch[0]["_positions"].device
    elem = batch[0]
    idx_keys = {Props.idx_i, Props.idx_j, Props.idx_i_triples}
    # Atom triple indices must be treated separately
    idx_triple_keys = {Props.idx_j_triples, Props.idx_k_triples}

    coll_batch = {}
    for key in elem:
        if (key not in idx_keys) and (key not in idx_triple_keys):
            coll_batch[key] = torch.cat([d[key] for d in batch], 0)
        elif key in idx_keys:
            coll_batch[key + "_local"] = torch.cat([d[key] for d in batch], 0)

    seg_m = torch.cumsum(coll_batch[Props.n_atoms], dim=0)
    seg_m = torch.cat([torch.zeros((1,), dtype=seg_m.dtype).to(device), seg_m], dim=0)
    idx_m = torch.repeat_interleave(
        torch.arange(len(batch)).to(device), repeats=coll_batch[Props.n_atoms], dim=0
    )
    coll_batch[Props.idx_m] = idx_m

    # Add molecule indices (_idx) if not already present
    if Props.idx not in coll_batch:
        coll_batch[Props.idx] = torch.arange(
            len(batch), dtype=torch.long, device=device
        )

    for key in idx_keys:
        if key in elem.keys():
            coll_batch[key] = torch.cat(
                [d[key] + off for d, off in zip(batch, seg_m)], 0
            )

    # Shift the indices for the atom triples
    for key in idx_triple_keys:
        if key in elem.keys():
            indices = []
            offset = 0
            for idx, d in enumerate(batch):
                indices.append(d[key] + offset)
                offset += d[Props.idx_j].shape[0]
            coll_batch[key] = torch.cat(indices, 0)

    return coll_batch


def unbatch_mols(batch_dict):
    """
    Decompose a batched molecular system back into individual molecules

    Args:
        batch_dict (dict[str->torch.Tensor]): Batched molecular system

    Returns:
        list[dict]: List of individual molecular systems
    """
    # Get number of molecules from idx_m
    n_mols = batch_dict[Props.idx_m].max().item() + 1

    # Initialize list to store individual molecules
    molecules = [{} for _ in range(n_mols)]
    n_atoms_per_mol = torch.bincount(batch_dict[Props.idx_m])
    atom_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long), torch.cumsum(n_atoms_per_mol[:-1], dim=0)]
    )

    # Process each molecule
    for mol_idx in range(n_mols):
        start_idx = atom_offsets[mol_idx]
        end_idx = start_idx + n_atoms_per_mol[mol_idx]
        molecules[mol_idx][Props.idx_m] = torch.zeros(
            n_atoms_per_mol[mol_idx], dtype=torch.long
        )
        # Handle regular features
        for key in batch_dict:
            if (
                key not in {Props.idx_m}
                and not key.endswith("_local")
                and not any(k in key for k in ["triples", "idx_i", "idx_j", "idx_k"])
            ):
                if batch_dict[key].size(0) == len(batch_dict[Props.idx_m]):
                    molecules[mol_idx][key] = batch_dict[key][start_idx:end_idx]
                else:
                    molecules[mol_idx][key] = batch_dict[key][mol_idx : mol_idx + 1]

        # Handle pair indices
        for key in ["idx_i", "idx_j"]:
            if key in batch_dict:
                # Find indices corresponding to this molecule
                mask = batch_dict[Props.idx_m][batch_dict[key]] == mol_idx
                indices = batch_dict[key][mask]
                # Adjust indices relative to this molecule
                adjusted_indices = indices - start_idx
                molecules[mol_idx][key] = adjusted_indices

    return molecules


# === FILE/FOLDER UTILITIES ===


def create_folder_structure(output_path: os.PathLike):
    """Create standard output folder structure."""
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path / "min" / "skeleton_images", exist_ok=True)
    os.makedirs(output_path / "min" / "ase_images", exist_ok=True)
    os.makedirs(output_path / "min" / "xyz", exist_ok=True)
    os.makedirs(output_path / "ts" / "skeleton_images", exist_ok=True)
    os.makedirs(output_path / "ts" / "ase_images", exist_ok=True)
    os.makedirs(output_path / "ts" / "xyz", exist_ok=True)
    os.makedirs(output_path / "complex" / "skeleton_images", exist_ok=True)
    os.makedirs(output_path / "complex" / "ase_images", exist_ok=True)
    os.makedirs(output_path / "complex" / "xyz", exist_ok=True)
