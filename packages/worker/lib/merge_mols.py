"""
Functions for merging two molecular conformers into a single complex.

The merge operation combines two separate molecules by:
1. Applying random rotations to both
2. Finding optimal separation distance (VDW non-overlapping)
3. Sorting atoms by atomic number for deterministic ordering
"""

import numpy as np
import scipy.spatial.transform
from scipy.spatial.distance import cdist

import torch
from ase.data import vdw_radii

from lib.types import Conformer
from lib.exploration import ExplorationContext, create_merge_context
from lib.utils import check_tetrose_limit
from loguru import logger


DEFAULT_VDW_RADIUS = 1.70  # Fallback for elements with missing data (nan in ASE)


def random_rotation(positions: np.ndarray) -> np.ndarray:
    """Apply a random rotation to positions."""
    R = scipy.spatial.transform.Rotation.random()
    return positions @ R.as_matrix()


def random_unit_vector() -> np.ndarray:
    """Generate a random unit vector uniformly distributed on the sphere."""
    vec = np.random.randn(3)
    return vec / np.linalg.norm(vec)


def get_vdw_radii_array(atomic_nums: np.ndarray) -> np.ndarray:
    """Get VDW radii for an array of atomic numbers, handling missing values."""
    radii = vdw_radii[atomic_nums]
    missing_mask = np.isnan(radii)
    if missing_mask.any():
        missing_elements = np.unique(atomic_nums[missing_mask])
        logger.warning(
            f"VDW radii missing in ASE for atomic numbers {missing_elements.tolist()}, "
            f"using default {DEFAULT_VDW_RADIUS} Å"
        )
    return np.where(missing_mask, DEFAULT_VDW_RADIUS, radii)


def compute_min_vdw_gap(
    pos1: np.ndarray,
    pos2: np.ndarray,
    atomic_nums1: np.ndarray,
    atomic_nums2: np.ndarray,
) -> float:
    """
    Compute the minimum gap between two molecules considering VDW radii.

    Returns the minimum of (distance - vdw_sum) over all atom pairs.
    Negative values indicate overlap.
    """
    # Pairwise distances
    distances = cdist(pos1, pos2)

    # VDW radii sums for all pairs
    vdw1 = get_vdw_radii_array(atomic_nums1)
    vdw2 = get_vdw_radii_array(atomic_nums2)
    vdw_sums = vdw1[:, None] + vdw2[None, :]

    # Gap = distance - vdw_sum
    gaps = distances - vdw_sums

    return gaps.min()


def _conformer_sort_key(conformer: Conformer) -> tuple:
    """Generate a canonical sort key for a conformer to ensure deterministic ordering."""
    conf = conformer.to_numpy()
    # Use tuple of sorted atomic numbers as primary key
    atomic_nums_tuple = tuple(sorted(conf.atomic_numbers.flatten()))
    # Use energy as secondary key (if available)
    energy = conf.energy if conf.energy is not None else 0.0
    return (atomic_nums_tuple, energy)


def merge_by_atomic_number(
    atomic_nums_1: np.ndarray,
    atomic_nums_2: np.ndarray,
    pos_1: np.ndarray,
    pos_2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge atoms from two molecules, sorted by atomic number (high to low).

    Returns:
        comb_atomic_nums: Combined atomic numbers sorted high to low
        comb_pos: Combined positions in same order
        component_indices: Array indicating which molecule each atom came from (0 or 1)
    """
    n1 = len(atomic_nums_1)
    n2 = len(atomic_nums_2)

    # Combine all atoms
    comb_atomic_nums = np.concatenate([atomic_nums_1, atomic_nums_2], axis=0)
    comb_pos = np.concatenate([pos_1, pos_2], axis=0)

    # Create component indices: 0 for first molecule, 1 for second
    component_indices = np.concatenate(
        [np.zeros(n1, dtype=np.int64), np.ones(n2, dtype=np.int64)]
    )
    sort_idx = np.argsort(-comb_atomic_nums, kind="stable")

    comb_atomic_nums = comb_atomic_nums[sort_idx]
    comb_pos = comb_pos[sort_idx]
    component_indices = component_indices[sort_idx]

    return comb_atomic_nums, comb_pos, component_indices


def merge_conformers(
    conformer_1: Conformer,
    conformer_2: Conformer,
) -> tuple[Conformer, np.ndarray, bool] | None:
    """
    Merge two conformers into a single complex.

    Atom ordering in the result is deterministic and order-independent:
    merge_conformers(a, b) produces the same atom ordering as merge_conformers(b, a).
    The random rotation/translation may differ, but the atom order is consistent.

    Args:
        conformer_1: First conformer
        conformer_2: Second conformer

    Returns:
        (merged_conformer, component_indices, swapped) or None if merge would exceed limits.
        component_indices indicates which original conformer each atom came from (0 or 1).
        swapped is True if conformers were swapped for canonical ordering (i.e., index 0
        refers to the original conformer_2, and index 1 refers to the original conformer_1).
    """
    # Canonicalize order for deterministic atom ordering
    key1 = _conformer_sort_key(conformer_1)
    key2 = _conformer_sort_key(conformer_2)

    swapped = key1 > key2
    if swapped:
        conformer_1, conformer_2 = conformer_2, conformer_1

    # Convert to numpy for manipulation
    conf1 = conformer_1.to_numpy()
    conf2 = conformer_2.to_numpy()

    atomic_nums1 = conf1.atomic_numbers.flatten()
    atomic_nums2 = conf2.atomic_numbers.flatten()

    # Random rotation of both fragments
    pos1 = random_rotation(conf1.positions)
    pos2 = random_rotation(conf2.positions)

    # Center both at origin (align COMs)
    pos1 -= pos1.mean(axis=0)
    pos2 -= pos2.mean(axis=0)

    # Random direction to separate the fragments
    direction = random_unit_vector()

    # Binary search to find minimum separation distance where VDW gap >= 0
    min_gap_threshold = 0.0  # Just touching (no overlap)
    buffer = 0.2  # Small buffer to add after finding minimum separation (Angstroms)

    low, high = 0.0, 15.0

    # First check if we need any separation at all
    if compute_min_vdw_gap(pos1, pos2, atomic_nums1, atomic_nums2) >= min_gap_threshold:
        separation = 0.0
    else:
        # Binary search for minimum separation
        while high - low > 0.01:
            mid = (low + high) / 2
            pos2_shifted = pos2 + mid * direction
            gap = compute_min_vdw_gap(pos1, pos2_shifted, atomic_nums1, atomic_nums2)
            if gap >= min_gap_threshold:
                high = mid
            else:
                low = mid
        separation = high

    # Apply separation with buffer
    pos2_final = pos2 + (separation + buffer) * direction

    # Merge atoms sorted by atomic number
    comb_atomic_nums, comb_pos, component_indices = merge_by_atomic_number(
        atomic_nums1,
        atomic_nums2,
        pos1,
        pos2_final,
    )

    # Center the merged structure
    comb_pos -= comb_pos.mean(axis=0)

    # Check composition limits
    comb_atomic_nums_torch = torch.tensor(comb_atomic_nums, dtype=torch.int64)
    if not check_tetrose_limit(comb_atomic_nums_torch):
        logger.debug("Skipping merge as it exceeds the C5H10O5 limit.")
        return None

    # Calculate combined energy if both have energies
    combined_energy = None
    if conformer_1.energy is not None and conformer_2.energy is not None:
        combined_energy = conformer_1.energy + conformer_2.energy

    merged = Conformer(
        positions=comb_pos,
        atomic_numbers=comb_atomic_nums,
        energy=combined_energy,
        charge=conformer_1.charge + conformer_2.charge,
    )

    return merged, component_indices, swapped


def enhance_merge_contexts(
    contexts: list,
    energy_fn,
    forces_calc,
    n_rotations: int = 5,
    relax_steps: int = 5,
    relax_fmax: float = 0.05,
    displacement_sigma: float = 0.15,
    max_binding_energy_eV: float = 0.3,
) -> list:
    """Improve merge-context starting geometries before diffusion.

    For each merge context:
      1. Keep the existing merged_conformer as candidate 0.
      2. Generate `n_rotations` additional random orientations of the two
         source components (new random rotations + separation directions).
      3. Score all candidates by ML energy; pick the lowest.
      4. Run up to `relax_steps` FIRE steps on the winner (skip if == 0).
      5. If the relaxed binding energy (E_sum(components) − E_merged)
         exceeds `max_binding_energy_eV`, drop the context entirely:
         deep encounter complexes have no nearby saddle the diffuser can
         find; they just trap the IRC on both sides.
      6. Apply Gaussian displacement of per-atom-per-coordinate sigma
         `displacement_sigma` Å. Breaks the symmetry around the encounter
         minimum so the diffuser has a direction to move in.

    Returns the list of surviving contexts (single-molecule contexts are
    passed through unchanged; deep-complex merges are removed).
    """
    # Lazy import to avoid a cycle with ts_pipeline which already imports
    # from this module.
    from ase import Atoms as _Atoms
    from lib.pes_explorer.newton_minimize import optimize_fire
    from lib.types import Conformer

    survivors: list = []
    n_dropped_deep = 0

    for ctx in contexts:
        if not getattr(ctx, "was_merged", False):
            survivors.append(ctx)
            continue
        if ctx.merge_component_conformers is None:
            survivors.append(ctx)
            continue
        comp1, comp2 = ctx.merge_component_conformers

        # Candidate list: start with the existing merged geometry so we
        # never select something *worse* than the sampler produced.
        candidates: list = []
        if ctx.merged_conformer is not None:
            candidates.append(ctx.merged_conformer)

        for _ in range(n_rotations):
            result = merge_conformers(comp1, comp2)
            if result is None:
                continue
            merged_candidate, _, _ = result
            candidates.append(merged_candidate)

        if not candidates:
            survivors.append(ctx)
            continue

        # Score via ML energy; pick lowest.
        best_energy = float("inf")
        best_conformer = None
        for cand in candidates:
            try:
                e = energy_fn(cand)
            except Exception as ex:
                logger.debug(f"Merge candidate energy eval failed: {ex}")
                continue
            if e < best_energy:
                best_energy = e
                best_conformer = cand
        if best_conformer is None:
            survivors.append(ctx)
            continue

        # FIRE-relax the winner (or skip relax entirely).
        cand_np = best_conformer.to_numpy()
        fire_result = None
        if relax_steps > 0:
            atoms = _Atoms(
                numbers=cand_np.atomic_numbers.flatten(),
                positions=cand_np.positions,
            )
            atoms.info["charge"] = best_conformer.charge
            atoms.calc = forces_calc
            forces_calc.reset()
            try:
                fire_result = optimize_fire(
                    forces_calc, atoms,
                    max_steps=relax_steps,
                    force_max_tol=relax_fmax,
                    force_rms_tol=relax_fmax * 0.5,
                    verbose=False,
                )
                relaxed_pos = np.asarray(fire_result.positions)
            except Exception as e:
                logger.warning(f"Merge pre-relax failed, using K-best without relax: {e}")
                relaxed_pos = cand_np.positions
        else:
            relaxed_pos = cand_np.positions

        # Deep encounter-complex gate (tier B.d): if binding exceeds the
        # cap, this pair is forming a stable complex — diffusing from it
        # wastes GPU time because both IRC sides will relax back into it.
        # Skip the whole context; the sampler will pick another pair.
        if max_binding_energy_eV > 0.0:
            e_components = None
            try:
                e1 = getattr(comp1, "energy", None)
                e2 = getattr(comp2, "energy", None)
                if e1 is not None and e2 is not None:
                    e_components = float(e1) + float(e2)
            except Exception:
                e_components = None
            if e_components is not None:
                binding = e_components - float(best_energy)  # positive = bound
                if binding > max_binding_energy_eV:
                    n_dropped_deep += 1
                    logger.debug(
                        f"Merge dropped: deep encounter complex "
                        f"(binding={binding:.3f} eV > {max_binding_energy_eV:.3f} eV cap)"
                    )
                    continue  # drop context

        # Small random displacement: break symmetry so the diffuser has a
        # direction to move in. Applied AFTER the relax so we start from a
        # physical geometry perturbed slightly off the minimum.
        # Retry displacement up to `max_retries` times if it creates atomic
        # overlaps below `min_pairwise_ang` Å — a single too-close pair
        # produces NaN forces during diffusion and kills the whole batch.
        if displacement_sigma > 0.0:
            min_pairwise_ang = 0.8
            max_retries = 10
            base_pos = relaxed_pos
            safe_pos = None
            for _ in range(max_retries):
                trial = base_pos + np.random.normal(
                    scale=displacement_sigma, size=base_pos.shape
                )
                dmat = np.linalg.norm(
                    trial[:, None, :] - trial[None, :, :], axis=-1
                )
                np.fill_diagonal(dmat, np.inf)
                if dmat.min() >= min_pairwise_ang:
                    safe_pos = trial
                    break
            if safe_pos is None:
                logger.debug(
                    "Merge displacement gave up (overlaps); using un-displaced relaxed geometry"
                )
                safe_pos = base_pos
            relaxed_pos = safe_pos

        # MoreRed diffusion requires COM-centered input (it is a
        # translation-invariant process). FIRE-relaxation and the random
        # displacement both shift the center of mass slightly; re-center
        # here or the diffuser rejects with "input positions not centered".
        relaxed_pos = relaxed_pos - relaxed_pos.mean(axis=0)

        new_conf = Conformer(
            positions=torch.tensor(relaxed_pos, dtype=torch.float64),
            atomic_numbers=torch.tensor(cand_np.atomic_numbers.flatten(), dtype=torch.long),
            charge=best_conformer.charge,
        )
        try:
            new_conf.energy = energy_fn(new_conf)
        except Exception:
            new_conf.energy = best_energy

        ctx.merged_conformer = new_conf
        ctx.merged_energy = new_conf.energy
        survivors.append(ctx)
        logger.debug(
            f"Merge pre-prep: {len(candidates)} candidates, "
            f"best_e={best_energy:.3f} eV, post_e={new_conf.energy:.3f} eV "
            f"(relax_steps={relax_steps}, "
            f"converged={getattr(fire_result, 'converged', '?') if fire_result else 'skip'})"
        )

    if n_dropped_deep > 0:
        logger.info(f"Merge prep dropped {n_dropped_deep} deep-encounter-complex contexts")
    return survivors


def merge_conformers_to_context(
    conformer_1: Conformer,
    conformer_2: Conformer,
    conformer_1_id: int | None = None,
    conformer_2_id: int | None = None,
    conformer_1_smiles: str | None = None,
    conformer_2_smiles: str | None = None,
) -> ExplorationContext | None:
    """
    Merge two conformers and return an ExplorationContext ready for TS exploration.

    Args:
        conformer_1: First conformer
        conformer_2: Second conformer
        conformer_1_id: Optional pre-computed ID for first conformer
        conformer_2_id: Optional pre-computed ID for second conformer
        conformer_1_smiles: Optional SMILES of source compound for first conformer
        conformer_2_smiles: Optional SMILES of source compound for second conformer

    Returns:
        ExplorationContext with merge info, or None if merge failed.
    """
    result = merge_conformers(conformer_1, conformer_2)
    if result is None:
        return None

    merged, component_indices, swapped = result
    component_indices_torch = torch.tensor(component_indices, dtype=torch.int64)

    # Account for swapping in IDs and SMILES
    if swapped:
        actual_conf_1, actual_conf_2 = conformer_2, conformer_1
        actual_id_1, actual_id_2 = conformer_2_id, conformer_1_id
        actual_smiles_1, actual_smiles_2 = conformer_2_smiles, conformer_1_smiles
    else:
        actual_conf_1, actual_conf_2 = conformer_1, conformer_2
        actual_id_1, actual_id_2 = conformer_1_id, conformer_2_id
        actual_smiles_1, actual_smiles_2 = conformer_1_smiles, conformer_2_smiles

    ctx = create_merge_context(
        component_1=actual_conf_1,
        component_2=actual_conf_2,
        merged_conformer=merged,
        atom_indices=component_indices_torch,
        component_1_id=actual_id_1,
        component_2_id=actual_id_2,
        component_1_smiles=actual_smiles_1,
        component_2_smiles=actual_smiles_2,
    )

    if merged.energy is not None:
        logger.debug(
            f"Merged conformers: E1={actual_conf_1.energy:.2f} + E2={actual_conf_2.energy:.2f} "
            f"= {merged.energy:.2f} eV"
        )

    return ctx
