"""
PES Graph: Classes and utilities for building molecular potential energy surface graphs.

Nodes represent local minima, edges represent transition states connecting them.
"""

from dataclasses import dataclass, field
from typing import Optional
import time
import pickle
import numpy as np
import networkx as nx
import ase.units
from ase import Atoms
from loguru import logger
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from lib.energy import compute_barrier_from_trajectory
from lib.naming import NameGenerator
from lib.types import Conformer

k_b = ase.units.kB
T = 300  # Kelvin


@dataclass
class RelaxationTrajectory:
    """Trajectory data from relaxation (e.g., from TS to minimum).

    Stores positions, energies, forces, and hessians at each step
    for assessing smoothness of predictions along critical paths.
    """

    positions: list[np.ndarray]  # List of (n_atoms, 3) arrays
    energies: list[float]  # Energy at each step (eV)
    forces: list[np.ndarray]  # List of (n_atoms, 3) force arrays (eV/Å)
    hessians: list[Optional[np.ndarray]]  # (3*n_atoms, 3*n_atoms) or None if reused

    @property
    def n_steps(self) -> int:
        return len(self.positions)

    def __repr__(self):
        return f"RelaxationTrajectory(n_steps={self.n_steps})"


@dataclass
class Minimum(Conformer):
    """
    A local minimum on the PES.

    Extends Conformer with PES-specific fields for graph tracking.

    Attributes:
        id: Unique identifier within the PES graph
        explored: Whether this minimum has been explored for TSs
    """

    id: int = 0
    explored: bool = False
    name: str = ""  # human-readable name, e.g. "min-amber-prism-0"
    hessian: Optional[np.ndarray] = None  # (3*n_atoms, 3*n_atoms) Hessian at this minimum
    n_merged: int = 0  # how many duplicates were merged into this minimum
    max_merge_rmsd: float = 0.0  # largest RMSD of a merged duplicate
    discovery_timestamp: float = 0.0  # unix timestamp when this minimum was discovered

    def __setstate__(self, state):
        state.setdefault("hessian", None)
        state.setdefault("n_merged", 0)
        state.setdefault("max_merge_rmsd", 0.0)
        state.setdefault("discovery_timestamp", 0.0)
        self.__dict__.update(state)

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if not isinstance(other, Minimum):
            return False
        return self.id == other.id


@dataclass
class TransitionState:
    """A transition state connecting two minima."""

    id: int
    positions: np.ndarray  # Shape: (n_atoms, 3)
    energy: float  # eV
    atomic_numbers: np.ndarray
    min_fwd_id: int  # ID of forward minimum
    min_bwd_id: int  # ID of backward minimum
    barrier_fwd: float  # Barrier height from forward minimum (eV)
    barrier_bwd: float  # Barrier height from backward minimum (eV)
    eigenvalue: float  # Imaginary mode eigenvalue
    hessian: Optional[np.ndarray]  # Shape: (3*n_atoms, 3*n_atoms)
    rmsd_to_fwd_min: float  # RMSD to forward minimum
    rmsd_to_bwd_min: float  # RMSD to backward minimum
    endpoint_to_endpoint_rmsd: (
        float  # RMSD between forward and backward endpoint minima
    )
    # Relaxation trajectories from TS to minima (for smoothness analysis)
    fwd_trajectory: Optional[RelaxationTrajectory] = None
    bwd_trajectory: Optional[RelaxationTrajectory] = None
    metadata: dict = field(default_factory=dict)
    name: str = ""  # human-readable name, e.g. "ts-swift-ketone-0"
    discovery_timestamp: float = 0.0  # unix timestamp when this TS was discovered

    def __setstate__(self, state):
        state.setdefault("discovery_timestamp", 0.0)
        self.__dict__.update(state)


def _kabsch_align(pos1: np.ndarray, pos2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center and Kabsch-align pos2 onto pos1. Returns (pos1_centered, pos2_aligned)."""
    pos1_centered = pos1 - pos1.mean(axis=0)
    pos2_centered = pos2 - pos2.mean(axis=0)

    H = pos1_centered.T @ pos2_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    pos2_aligned = pos2_centered @ R
    return pos1_centered, pos2_aligned


def _hungarian_permutation(
    pos1: np.ndarray, pos2: np.ndarray, atomic_numbers: np.ndarray
) -> np.ndarray:
    """Find optimal atom permutation of pos2 to match pos1 via Hungarian algorithm.

    Only permutes atoms within each element type (e.g., H↔H, C↔C).
    Assumes pos1 and pos2 are already aligned (centered + rotated).

    Returns:
        perm: index array such that pos2[perm] optimally matches pos1
    """
    n = len(atomic_numbers)
    perm = np.arange(n)
    for element in np.unique(atomic_numbers):
        mask = atomic_numbers == element
        idx = np.where(mask)[0]
        if len(idx) <= 1:
            continue
        cost = cdist(pos1[idx], pos2[idx], metric="sqeuclidean")
        row_ind, col_ind = linear_sum_assignment(cost)
        perm[idx[row_ind]] = idx[col_ind]
    return perm


def compute_rmsd(
    pos1: np.ndarray,
    pos2: np.ndarray,
    atomic_numbers: np.ndarray | None = None,
    return_permutation: bool = False,
) -> float | tuple[float, np.ndarray]:
    """Compute RMSD between two structures after optimal alignment (Kabsch).

    For molecules with equivalent atoms (same element), finds the optimal
    permutation. Uses brute-force enumeration when the number of permutations
    is small (< 720), otherwise falls back to Hungarian on Kabsch-aligned
    positions.

    Brute-force is needed for small molecules where equivalent atoms are a
    large fraction (e.g., HCO3- with 3/5 atoms being O). Kabsch gives a bad
    initial rotation when 60% of atoms are scrambled, which traps Hungarian
    in a local minimum. For larger molecules the many unique atoms anchor
    Kabsch correctly and Hungarian works fine.

    Args:
        pos1: Reference positions, shape (n_atoms, 3)
        pos2: Positions to compare, shape (n_atoms, 3)
        atomic_numbers: If provided, handle same-element permutations.
        return_permutation: If True, return (rmsd, perm).

    Returns:
        RMSD value, or (rmsd, perm) if return_permutation=True.
    """
    if atomic_numbers is not None:
        perm, rmsd_val = _best_permutation_rmsd(pos1, pos2, atomic_numbers)
        if return_permutation:
            return rmsd_val, perm
        return rmsd_val

    pos1_centered, pos2_aligned = _kabsch_align(pos1, pos2)
    perm = np.arange(len(pos1))
    rmsd_val = np.sqrt(np.mean(np.sum((pos1_centered - pos2_aligned) ** 2, axis=1)))

    if return_permutation:
        return rmsd_val, perm
    return rmsd_val


# Maximum number of brute-force permutations before falling back to Hungarian.
def _best_permutation_rmsd(
    pos1: np.ndarray, pos2: np.ndarray, atomic_numbers: np.ndarray,
    early_stop_rmsd: float = 0.01,
) -> tuple[np.ndarray, float]:
    """Find the atom permutation that minimizes RMSD.

    Two-stage hierarchical approach:
    1. Brute-force permutations of heavy atoms only (C, O, N, ... — typically
       few equivalent atoms, so small permutation space)
    2. For each heavy-atom permutation, use Hungarian to assign H atoms
       (H atoms are numerous but their optimal assignment is determined once
       the heavy-atom backbone is fixed)

    This reduces e.g. C5H10O5 from 10!×5!×5! = 52B to 5!×5! = 14400 heavy-atom
    perms × one Hungarian per perm ≈ 14400 evaluations.

    Falls back to flat brute-force if there are no hydrogens (all atoms are
    heavy) and to Hungarian if even the heavy-atom perms exceed 10M.
    """
    from math import factorial
    from itertools import permutations as iter_perms

    n = len(pos1)
    h_mask = atomic_numbers == 1
    heavy_mask = ~h_mask
    h_indices = np.where(h_mask)[0]
    heavy_indices = np.where(heavy_mask)[0]

    # If no hydrogens or no heavy atoms, use flat brute-force / Hungarian
    if len(h_indices) == 0 or len(heavy_indices) == 0:
        return _best_permutation_rmsd_flat(pos1, pos2, atomic_numbers, early_stop_rmsd)

    # Heavy-only Kabsch needs ≥3 non-coplanar heavy atoms to constrain the
    # rotation. If the heavy-atom cloud is rank-deficient (≤2 atoms,
    # collinear, or planar), the rotation around the degenerate axis is
    # arbitrary and the subsequent H-atom Hungarian assignment picks a
    # wrong permutation — producing RMSDs that are orders of magnitude
    # larger than the true minimum. Fall back to flat brute-force.
    p1h_c = pos1[heavy_indices] - pos1[heavy_indices].mean(axis=0)
    svals = np.linalg.svd(p1h_c, compute_uv=False)
    svals = np.pad(svals, (0, max(0, 3 - len(svals))))
    if svals[2] < max(1e-3, 0.01 * svals[0]):
        return _best_permutation_rmsd_flat(pos1, pos2, atomic_numbers, early_stop_rmsd)

    # Heavy-atom equivalent groups
    heavy_anum = atomic_numbers[heavy_indices]
    heavy_groups: list[np.ndarray] = []
    heavy_total_perms = 1
    for elem in np.unique(heavy_anum):
        idx = np.where(heavy_anum == elem)[0]
        if len(idx) > 1:
            heavy_groups.append(idx)
            heavy_total_perms *= factorial(len(idx))

    if heavy_total_perms > 10_000_000:
        # Too many even for heavy atoms — full Hungarian fallback
        pos1_c, pos2_a = _kabsch_align(pos1, pos2)
        perm = _hungarian_permutation(pos1_c, pos2_a, atomic_numbers)
        rmsd_val = np.sqrt(np.mean(np.sum((pos1_c - pos2_a[perm]) ** 2, axis=1)))
        return perm, rmsd_val

    best_rmsd = float("inf")
    best_perm = np.arange(n)
    stopped = False

    def _enum_heavy(gi, heavy_perm):
        nonlocal best_rmsd, best_perm, stopped
        if stopped:
            return
        if gi == len(heavy_groups):
            # Build full permutation: heavy atoms permuted, H assigned by Hungarian
            full_perm = np.arange(n)
            # Map heavy atom permutation back to global indices
            for i, local_idx in enumerate(range(len(heavy_indices))):
                full_perm[heavy_indices[i]] = heavy_indices[heavy_perm[local_idx]]

            # Kabsch-align using heavy atoms only (H positions may be scrambled
            # and would corrupt the rotation if included)
            p2_perm = pos2[full_perm]
            p1_heavy = pos1[heavy_indices]
            p2_heavy = p2_perm[heavy_indices]
            p1h_c = p1_heavy - p1_heavy.mean(axis=0)
            p2h_c = p2_heavy - p2_heavy.mean(axis=0)
            H_mat = p1h_c.T @ p2h_c
            U_h, _, Vt_h = np.linalg.svd(H_mat)
            R_h = Vt_h.T @ U_h.T
            if np.linalg.det(R_h) < 0:
                Vt_h[-1] *= -1
                R_h = Vt_h.T @ U_h.T
            # Apply heavy-atom rotation to ALL atoms
            center1 = pos1.mean(axis=0)
            center2 = p2_perm.mean(axis=0)
            p1c = pos1 - center1
            p2a = (p2_perm - center2) @ R_h

            # Hungarian assignment for H atoms on the aligned structures
            if len(h_indices) > 1:
                h_cost = cdist(p1c[h_indices], p2a[h_indices], metric="sqeuclidean")
                _, col_ind = linear_sum_assignment(h_cost)
                h_perm = h_indices[col_ind]
                for i, hi in enumerate(h_indices):
                    full_perm[hi] = h_perm[i]
                # Final alignment with all atoms correctly permuted
                p2_perm = pos2[full_perm]
                p1c, p2a = _kabsch_align(pos1, p2_perm)

            r = float(np.sqrt(np.mean(np.sum((p1c - p2a) ** 2, axis=1))))
            if r < best_rmsd:
                best_rmsd = r
                best_perm = full_perm.copy()
                if r < early_stop_rmsd:
                    stopped = True
            return

        idx = heavy_groups[gi]
        for p in iter_perms(range(len(idx))):
            if stopped:
                return
            trial = heavy_perm.copy()
            trial[idx] = idx[list(p)]
            _enum_heavy(gi + 1, trial)

    _enum_heavy(0, np.arange(len(heavy_indices)))
    return best_perm, best_rmsd


def _best_permutation_rmsd_flat(
    pos1: np.ndarray, pos2: np.ndarray, atomic_numbers: np.ndarray,
    early_stop_rmsd: float = 0.01,
) -> tuple[np.ndarray, float]:
    """Flat brute-force for molecules without hydrogens."""
    from math import factorial
    from itertools import permutations as iter_perms

    groups: list[np.ndarray] = []
    total_perms = 1
    for elem in np.unique(atomic_numbers):
        idx = np.where(atomic_numbers == elem)[0]
        if len(idx) > 1:
            groups.append(idx)
            total_perms *= factorial(len(idx))

    if total_perms > 10_000_000 or not groups:
        pos1_c, pos2_a = _kabsch_align(pos1, pos2)
        perm = _hungarian_permutation(pos1_c, pos2_a, atomic_numbers)
        rmsd_val = np.sqrt(np.mean(np.sum((pos1_c - pos2_a[perm]) ** 2, axis=1)))
        return perm, rmsd_val

    best_rmsd = float("inf")
    best_perm = np.arange(len(pos1))
    stopped = False

    def _enumerate(gi, current_perm):
        nonlocal best_rmsd, best_perm, stopped
        if stopped:
            return
        if gi == len(groups):
            p2_perm = pos2[current_perm]
            _, p2_aligned = _kabsch_align(pos1, p2_perm)
            p1c = pos1 - pos1.mean(axis=0)
            r = np.sqrt(np.mean(np.sum((p1c - p2_aligned) ** 2, axis=1)))
            if r < best_rmsd:
                best_rmsd = r
                best_perm = current_perm.copy()
                if r < early_stop_rmsd:
                    stopped = True
            return
        idx = groups[gi]
        for p in iter_perms(range(len(idx))):
            if stopped:
                return
            trial = current_perm.copy()
            trial[idx] = idx[list(p)]
            _enumerate(gi + 1, trial)

    _enumerate(0, np.arange(len(pos1)))
    return best_perm, best_rmsd


def _compute_neb_forces(
    positions: np.ndarray,
    energies: np.ndarray,
    physical_forces: np.ndarray,
    k_spring: float,
    climb: bool = False,
) -> np.ndarray:
    """Compute NEB forces for interior images using improved tangent method.

    Args:
        positions: (n_images+2, n_atoms, 3) - all images including endpoints
        energies: (n_images+2,) - energy of each image
        physical_forces: (n_images, n_atoms, 3) - forces on interior images only
        k_spring: Spring constant (eV/Å²)
        climb: If True, apply climbing image modification to highest-energy interior image

    Returns:
        (n_images, n_atoms, 3) - NEB forces for interior images
    """
    n_interior = len(physical_forces)
    neb_forces = np.zeros_like(physical_forces)

    # Find climbing image index (relative to interior images, 0-based)
    if climb:
        interior_energies = energies[1:-1]
        climb_idx = int(np.argmax(interior_energies))
    else:
        climb_idx = -1  # no climbing image

    for i in range(n_interior):
        img_idx = i + 1  # index into full band (0 = endpoint)

        # Improved tangent (Henkelman & Jonsson 2000)
        tau_plus = (positions[img_idx + 1] - positions[img_idx]).ravel()
        tau_minus = (positions[img_idx] - positions[img_idx - 1]).ravel()

        e_this = energies[img_idx]
        e_next = energies[img_idx + 1]
        e_prev = energies[img_idx - 1]

        if e_next > e_this > e_prev:
            tau = tau_plus
        elif e_next < e_this < e_prev:
            tau = tau_minus
        else:
            dE_max = max(abs(e_next - e_this), abs(e_prev - e_this))
            dE_min = min(abs(e_next - e_this), abs(e_prev - e_this))
            if e_next > e_prev:
                tau = tau_plus * dE_max + tau_minus * dE_min
            else:
                tau = tau_plus * dE_min + tau_minus * dE_max

        tau_norm = np.linalg.norm(tau)
        if tau_norm < 1e-10:
            tau_hat = tau
        else:
            tau_hat = tau / tau_norm

        f_phys = physical_forces[i].ravel()

        if i == climb_idx:
            # Climbing image: invert force component along band
            neb_forces[i] = (f_phys - 2.0 * np.dot(f_phys, tau_hat) * tau_hat).reshape(
                physical_forces[i].shape
            )
        else:
            # Perpendicular component of physical force
            f_perp = f_phys - np.dot(f_phys, tau_hat) * tau_hat

            # Spring force along tangent
            d_next = np.linalg.norm(tau_plus)
            d_prev = np.linalg.norm(tau_minus)
            f_spring = k_spring * (d_next - d_prev) * tau_hat

            neb_forces[i] = (f_perp + f_spring).reshape(physical_forces[i].shape)

    return neb_forces


@dataclass
class _NEBBandState:
    """Per-band state for batched NEB FIRE optimization."""

    # Image positions: (n_total, n_atoms, 3) including endpoints
    positions: np.ndarray
    # Energies of endpoints (known, not recomputed)
    energy_ep0: float
    energy_ep1: float
    # Energies of all images (updated each step)
    energies: np.ndarray  # (n_total,)
    # Number of interior images
    n_interior: int
    # Atomic numbers for creating Atoms objects
    atomic_numbers: np.ndarray

    # FIRE state for this band
    velocity: np.ndarray  # (n_interior * n_atoms * 3,)
    dt: float = 0.1
    alpha: float = 0.1
    n_pos: int = 0

    # Phase / convergence
    current_step: int = 0
    climb: bool = False
    done: bool = False

    # Result
    barrier: float = float("inf")
    same_basin: bool = False

    # FIRE parameters
    dt_max: float = 0.4
    alpha_start: float = 0.1
    f_inc: float = 1.1
    f_dec: float = 0.5
    f_alpha: float = 0.99
    n_min: int = 5

    def fire_step(self, forces_flat: np.ndarray):
        """Perform one FIRE step using provided NEB forces (flattened interior)."""
        v = self.velocity
        f = forces_flat

        power = np.dot(v, f)
        if power > 0:
            self.n_pos += 1
            if self.n_pos > self.n_min:
                self.dt = min(self.dt * self.f_inc, self.dt_max)
                self.alpha *= self.f_alpha
            v_norm = np.linalg.norm(v)
            f_norm = np.linalg.norm(f)
            if f_norm > 1e-10:
                v = (1.0 - self.alpha) * v + self.alpha * (v_norm / f_norm) * f
        else:
            v *= 0.0
            self.n_pos = 0
            self.dt = max(self.dt * self.f_dec, 0.01)
            self.alpha = self.alpha_start

        # Velocity Verlet half: v += 0.5*dt*f, x += dt*v, v += 0.5*dt*f
        v += self.dt * f
        # Clamp velocity to avoid explosions
        v_max = 2.0  # Å/fs-like units
        v_scale = np.linalg.norm(v)
        if v_scale > v_max:
            v *= v_max / v_scale

        # Update interior positions
        n_atoms = self.positions.shape[1]
        dx = (self.dt * v).reshape(self.n_interior, n_atoms, 3)
        self.positions[1:-1] += dx

        self.velocity = v
        self.current_step += 1


def neb_check_same_basin_batched(
    pairs: list[tuple[np.ndarray, np.ndarray, float, float]],
    atomic_numbers: np.ndarray,
    calc,
    fire_steps: int = 150,
    k_spring: float = 0.1,
    barrier_threshold: float = 0.05,
) -> list[bool]:
    """Check multiple pairs of minima for same-basin using batched NEB.

    All interior images across all bands are evaluated in a single GPU call per
    FIRE step, giving ~70x speedup over sequential NEB checks.

    Args:
        pairs: List of (pos1, pos2, energy1, energy2) tuples.
        atomic_numbers: Shared atomic numbers.
        calc: Calculator with get_batched_forces_and_energy() method.
        fire_steps: Total FIRE steps (2/3 phase 1 + 1/3 phase 2).
        k_spring: Spring constant (eV/Å²).
        barrier_threshold: Same-basin barrier threshold (eV).

    Returns:
        List of bools, one per pair (True = same basin).
    """
    from ase.mep import NEB

    if not pairs:
        return []

    # Initialize bands
    bands: list[_NEBBandState] = []
    for pos1, pos2, e1, e2 in pairs:
        try:
            rmsd, perm = compute_rmsd(pos1, pos2, atomic_numbers, return_permutation=True)
            pos2_perm = pos2[perm]
            n_images = max(5, min(12, int(np.ceil(rmsd / 0.15))))

            # Use ASE for IDPP interpolation only
            atoms1 = Atoms(numbers=atomic_numbers, positions=pos1.copy())
            atoms2 = Atoms(numbers=atomic_numbers[perm], positions=pos2_perm.copy())
            images = [atoms1]
            for _ in range(n_images):
                images.append(atoms1.copy())
            images.append(atoms2)
            neb = NEB(images, k=k_spring, climb=False)
            neb.interpolate(method="idpp")

            # Extract all positions
            all_pos = np.array([img.get_positions() for img in images])  # (n_total, n_atoms, 3)
            n_atoms = all_pos.shape[1]

            # Initialize energies with endpoints known
            energies = np.zeros(len(images))
            energies[0] = e1
            energies[-1] = e2

            band = _NEBBandState(
                positions=all_pos,
                energy_ep0=e1,
                energy_ep1=e2,
                energies=energies,
                n_interior=n_images,
                atomic_numbers=atomic_numbers.copy(),
                velocity=np.zeros(n_images * n_atoms * 3),
            )
            bands.append(band)
        except Exception as e:
            logger.warning(f"NEB band init failed ({type(e).__name__}: {e}), marking not same basin")
            # Create a dummy done band
            dummy = _NEBBandState(
                positions=np.zeros((3, len(atomic_numbers), 3)),
                energy_ep0=0.0,
                energy_ep1=0.0,
                energies=np.zeros(3),
                n_interior=1,
                atomic_numbers=atomic_numbers.copy(),
                velocity=np.zeros(len(atomic_numbers) * 3),
                done=True,
                same_basin=False,
            )
            bands.append(dummy)

    phase1_steps = fire_steps * 2 // 3

    # Unified FIRE loop
    for step in range(fire_steps):
        # Collect all interior images from active bands
        active_atoms_list: list[Atoms] = []
        # Map: (band_idx, interior_idx) for each atom in the batch
        batch_map: list[tuple[int, int]] = []

        for bi, band in enumerate(bands):
            if band.done:
                continue
            for ii in range(band.n_interior):
                img_pos = band.positions[ii + 1]  # skip endpoint 0
                a = Atoms(numbers=band.atomic_numbers, positions=img_pos)
                active_atoms_list.append(a)
                batch_map.append((bi, ii))

        if not active_atoms_list:
            break  # All bands done

        # Single batched GPU call
        results = calc.get_batched_forces_and_energy(active_atoms_list)

        # Scatter results back to bands
        # First collect forces+energies per band
        band_forces: dict[int, list] = {}
        band_energies: dict[int, list] = {}
        for (bi, ii), (forces, energy) in zip(batch_map, results):
            if bi not in band_forces:
                band_forces[bi] = [None] * bands[bi].n_interior
                band_energies[bi] = [None] * bands[bi].n_interior
            band_forces[bi][ii] = forces
            band_energies[bi][ii] = energy

        # Update each active band
        for bi in band_forces:
            band = bands[bi]
            # Update interior energies
            for ii in range(band.n_interior):
                band.energies[ii + 1] = band_energies[bi][ii]
            # Keep endpoint energies fixed
            band.energies[0] = band.energy_ep0
            band.energies[-1] = band.energy_ep1

            # Stack physical forces: (n_interior, n_atoms, 3)
            phys_forces = np.array(band_forces[bi])

            # Compute NEB forces (CPU)
            neb_f = _compute_neb_forces(
                band.positions, band.energies, phys_forces,
                k_spring, climb=band.climb,
            )

            # FIRE step
            band.fire_step(neb_f.ravel())

        # Phase transition: after phase 1, check barriers and switch to CI-NEB
        if step == phase1_steps - 1:
            for band in bands:
                if band.done:
                    continue
                e_ref = max(band.energies[0], band.energies[-1])
                barrier = float(np.max(band.energies) - e_ref)
                if barrier > 1.0:
                    # Unphysical band — bail
                    band.done = True
                    band.barrier = barrier
                    band.same_basin = False
                    logger.debug(
                        f"NEB batched: bail after phase 1, barrier={barrier:.4f} eV"
                    )
                else:
                    band.climb = True  # Switch to CI-NEB

        # Check convergence: max NEB force < 0.05 eV/Å
        for bi in band_forces:
            band = bands[bi]
            if band.done:
                continue
            phys_forces = np.array(band_forces[bi])
            neb_f = _compute_neb_forces(
                band.positions, band.energies, phys_forces,
                k_spring, climb=band.climb,
            )
            max_force = np.max(np.linalg.norm(neb_f.reshape(-1, 3), axis=1))
            if max_force < 0.05:
                band.done = True

    # Compute final barriers
    results_list: list[bool] = []
    for bi, band in enumerate(bands):
        if not band.done:
            # Ran out of steps — compute barrier from last state
            pass
        e_ref = max(band.energies[0], band.energies[-1])
        barrier = float(np.max(band.energies) - e_ref)
        band.barrier = barrier
        band.same_basin = barrier < barrier_threshold

        pair = pairs[bi]
        dE = abs(pair[2] - pair[3])
        logger.debug(
            f"NEB batched pair {bi}: barrier={barrier:.4f} eV (dE={dE:.4f}), "
            f"same_basin={band.same_basin}"
        )
        results_list.append(band.same_basin)

    return results_list


def neb_check_same_basin(
    pos1: np.ndarray,
    pos2: np.ndarray,
    atomic_numbers: np.ndarray,
    calc,
    n_images: int = 5,
    fire_steps: int = 150,
    k_spring: float = 0.1,
    barrier_threshold: float = 0.05,
) -> bool:
    """Check if two minima are in the same basin using NEB.

    Creates a NEB band with IDPP interpolation between pos1 and pos2,
    runs FIRE optimization, and checks if the maximum barrier is below
    the threshold.

    Returns True if barrier < threshold (same basin), False otherwise.
    On any error, returns False (conservative: keep separate).
    """
    from ase import Atoms
    from ase.mep import NEB
    from ase.optimize import FIRE

    try:
        # Permutation-invariant RMSD + optimal atom permutation for NEB endpoints
        rmsd, perm = compute_rmsd(pos1, pos2, atomic_numbers, return_permutation=True)
        pos2 = pos2[perm]
        n_images = max(5, min(12, int(np.ceil(rmsd / 0.15))))

        # Create endpoint images
        atoms1 = Atoms(numbers=atomic_numbers, positions=pos1.copy())
        atoms1.calc = calc
        atoms2 = Atoms(numbers=atomic_numbers[perm], positions=pos2.copy())
        atoms2.calc = calc

        # Create interior images
        images = [atoms1]
        for _ in range(n_images):
            img = atoms1.copy()
            img.calc = calc
            images.append(img)
        images.append(atoms2)

        # Two-phase NEB: regular NEB first to relax images onto the MEP,
        # then CI-NEB to refine the barrier.  Running CI-NEB from the start
        # can lock a badly-interpolated image at absurd energy (the climbing
        # force pushes it further up instead of onto the path).
        neb = NEB(images, k=k_spring, climb=False, allow_shared_calculator=True)
        neb.interpolate(method="idpp")

        # Phase 1: regular NEB — stabilize all images onto the MEP
        phase1_steps = fire_steps * 2 // 3
        optimizer = FIRE(neb, logfile=None)
        optimizer.run(fmax=0.05, steps=phase1_steps)

        # Early bail: if any image is still absurdly high after phase 1,
        # the band is unphysical (IDPP produced atom overlaps).  No point
        # running CI-NEB — the answer is clearly "not same basin".
        energies = [img.get_potential_energy() for img in images]
        e_ref = max(energies[0], energies[-1])
        barrier_after_p1 = max(energies) - e_ref
        if barrier_after_p1 > 1.0:
            dE = abs(energies[0] - energies[-1])
            e_profile = [f"{e - energies[0]:.4f}" for e in energies]
            logger.debug(
                f"NEB check: bail after phase 1, barrier={barrier_after_p1:.4f} eV "
                f"(dE={dE:.4f}), rmsd={rmsd:.3f} Å, n_images={n_images}, "
                f"same_basin=False, FIRE_steps={optimizer.nsteps}, "
                f"profile_vs_ep0=[{', '.join(e_profile)}]"
            )
            return False

        # Phase 2: switch to CI-NEB to push highest image toward saddle
        neb.climb = True
        optimizer = FIRE(neb, logfile=None)
        optimizer.run(fmax=0.05, steps=fire_steps - phase1_steps)

        # Compute barrier above the HIGHER endpoint.
        # This measures whether the higher-energy structure faces a
        # barrier preventing it from relaxing to the lower one.
        # barrier ≈ 0 means monotonically downhill → same basin.
        energies = [img.get_potential_energy() for img in images]
        e_ref = max(energies[0], energies[-1])
        barrier = max(energies) - e_ref
        dE = abs(energies[0] - energies[-1])

        # Log energy profile for diagnostics
        e_profile = [f"{e - energies[0]:.4f}" for e in energies]
        logger.debug(
            f"NEB check: barrier={barrier:.4f} eV (dE={dE:.4f}), "
            f"threshold={barrier_threshold:.4f} eV, "
            f"rmsd={rmsd:.3f} Å, n_images={n_images}, "
            f"same_basin={barrier < barrier_threshold}, "
            f"FIRE_steps={optimizer.nsteps}, "
            f"profile_vs_ep0=[{', '.join(e_profile)}]"
        )
        return barrier < barrier_threshold

    except Exception as e:
        logger.warning(f"NEB check failed ({type(e).__name__}: {e}), keeping separate")
        return False


def neb_compute_band(
    pos1: np.ndarray,
    pos2: np.ndarray,
    atomic_numbers: np.ndarray,
    calc,
    fire_steps: int = 150,
    k_spring: float = 0.1,
    barrier_threshold: float = 0.05,
) -> dict:
    """Run NEB between two minima and return the full optimized band.

    Returns dict with:
        frames: list of (n_atoms, 3) position arrays for each image
        energies: list of energies for each image (eV)
        barrier: barrier height above higher endpoint (eV)
        same_basin: whether barrier < threshold
        rmsd: RMSD between endpoints (Å)
        n_images: number of interior images used
    """
    from ase import Atoms
    from ase.mep import NEB
    from ase.optimize import FIRE

    # Permutation-invariant RMSD + optimal atom permutation for NEB endpoints
    rmsd, perm = compute_rmsd(pos1, pos2, atomic_numbers, return_permutation=True)
    pos2 = pos2[perm]
    n_images = max(5, min(12, int(np.ceil(rmsd / 0.15))))

    atoms1 = Atoms(numbers=atomic_numbers, positions=pos1.copy())
    atoms1.calc = calc
    atoms2 = Atoms(numbers=atomic_numbers[perm], positions=pos2.copy())
    atoms2.calc = calc

    images = [atoms1]
    for _ in range(n_images):
        img = atoms1.copy()
        img.calc = calc
        images.append(img)
    images.append(atoms2)

    neb = NEB(images, k=k_spring, allow_shared_calculator=True)
    neb.interpolate(method="idpp")

    optimizer = FIRE(neb, logfile=None)
    optimizer.run(fmax=0.05, steps=fire_steps)

    energies = [img.get_potential_energy() for img in images]
    frames = [img.get_positions().copy() for img in images]
    e_ref = max(energies[0], energies[-1])
    barrier = max(energies) - e_ref

    return {
        "frames": frames,
        "energies": energies,
        "barrier": barrier,
        "same_basin": barrier < barrier_threshold,
        "rmsd": rmsd,
        "n_images": n_images,
    }


def _reorder_hessian(hessian: np.ndarray, sort_indices: np.ndarray) -> np.ndarray:
    """Reorder a Hessian matrix to match a new atom ordering.

    The Hessian is (3N, 3N) where atom i occupies rows/cols [3i:3i+3].
    Permuting atoms by sort_indices requires permuting the corresponding
    3D blocks in both dimensions.
    """
    idx_3d = (3 * sort_indices[:, None] + np.arange(3)).ravel()
    return hessian[np.ix_(idx_3d, idx_3d)]


class PESGraph:
    """
    Graph representation of a molecular potential energy surface.

    Nodes = local minima
    Edges = transition states connecting minima
    """

    def __init__(
        self,
        atomic_numbers: np.ndarray,
        rmsd_threshold: float = 0.2,
        energy_threshold: float = 0.05,
        calc=None,
        neb_barrier_threshold: float = 0.05,
        neb_n_images: int = 5,
        neb_fire_steps: int = 150,
        neb_spring_constant: float = 0.1,
    ):
        """
        Args:
            atomic_numbers: Atomic numbers of the molecule
            rmsd_threshold: RMSD threshold for structure deduplication (Å)
            energy_threshold: Energy threshold for structure deduplication (eV)
            calc: ASE calculator for NEB checks (None = NEB disabled)
            neb_barrier_threshold: NEB barrier threshold for same-basin (eV)
            neb_n_images: Number of interior NEB images
            neb_fire_steps: FIRE optimization steps on NEB band
            neb_spring_constant: NEB spring constant (eV/Å²)
        """
        self.atomic_numbers = atomic_numbers
        self.rmsd_threshold = rmsd_threshold
        self.energy_threshold = energy_threshold
        self.calc = calc
        self.neb_barrier_threshold = neb_barrier_threshold
        self.neb_n_images = neb_n_images
        self.neb_fire_steps = neb_fire_steps
        self.neb_spring_constant = neb_spring_constant

        self.minima: dict[int, Minimum] = {}
        self.transition_states: dict[int, TransitionState] = {}
        self.graph = nx.Graph()

        self._next_min_id = 0
        self._next_ts_id = 0
        self._min_namer = NameGenerator()
        self._ts_namer = NameGenerator()


    def __getstate__(self):
        """Exclude calc from pickle (contains GPU model state)."""
        state = self.__dict__.copy()
        state["calc"] = None
        return state

    def __setstate__(self, state):
        """Restore from pickle with backward compat for NEB fields."""
        state.setdefault("calc", None)
        state.setdefault("neb_barrier_threshold", 0.05)
        state.setdefault("neb_n_images", 5)
        state.setdefault("neb_fire_steps", 150)
        state.setdefault("neb_spring_constant", 0.1)
        self.__dict__.update(state)

    def _canonicalize(
        self, positions: np.ndarray, hessian: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Canonicalize positions (and optionally a Hessian) for this molecule.

        Uses RDKit canonical atom ranking so that RMSD comparisons are
        permutation-invariant.  Lazy-imports lib.compound to avoid circular
        imports.

        Returns:
            (canonical_positions, canonical_hessian_or_None, sort_indices)
        """
        from lib.compound import canonicalize_structure

        _, canonical_pos, sort_indices = canonicalize_structure(
            self.atomic_numbers, positions
        )
        canonical_hess = None
        if hessian is not None:
            canonical_hess = _reorder_hessian(hessian, sort_indices)
        return canonical_pos, canonical_hess, sort_indices

    def _find_matching_minimum(
        self, positions: np.ndarray, energy: float
    ) -> tuple[Optional[int], float, float]:
        """Find an existing minimum that matches the given structure.

        Two-tier deduplication:
        1. RMSD < rmsd_threshold AND energy_diff < energy_threshold → merge (fast path)
        2. Batched NEB check for all remaining pairs (no energy gate)

        Assumes positions are already in canonical atom ordering (done by
        add_minimum before calling this method).

        Returns:
            (min_id or None, rmsd of match, neb_time_s)
        """
        # Pass 1: fast RMSD + energy scan
        neb_candidates: list[tuple[int, float]] = []  # (min_id, rmsd)
        for min_id, minimum in self.minima.items():
            energy_diff = abs(minimum.energy - energy)
            rmsd = compute_rmsd(minimum.positions, positions, self.atomic_numbers)
            logger.debug(
                f"dedup min {min_id}: RMSD={rmsd:.4f} dE={energy_diff:.4f} "
                f"shapes={minimum.positions.shape}/{positions.shape}"
            )
            # Ultra-low RMSD (< 0.01 Å) = essentially identical structure.
            # Always merge regardless of energy diff (which is ML noise at
            # this geometric proximity).
            if rmsd < 0.01:
                logger.info(f"Merging into min {min_id}: ultra-low RMSD={rmsd:.4f}")
                return min_id, rmsd, 0.0
            if rmsd < self.rmsd_threshold and energy_diff < self.energy_threshold:
                logger.info(f"Merging into min {min_id}: RMSD={rmsd:.4f} dE={energy_diff:.4f}")
                return min_id, rmsd, 0.0  # Fast merge

            if self.calc is not None:
                neb_candidates.append((min_id, rmsd))

        if not neb_candidates:
            return None, 0.0, 0.0

        # Pass 2: batched NEB check
        has_batched = hasattr(self.calc, "get_batched_forces_and_energy")
        if has_batched and len(neb_candidates) > 0:
            pairs = []
            for min_id, _rmsd in neb_candidates:
                minimum = self.minima[min_id]
                pairs.append((minimum.positions, positions, minimum.energy, energy))

            t0 = time.perf_counter()
            results = neb_check_same_basin_batched(
                pairs,
                self.atomic_numbers,
                self.calc,
                fire_steps=self.neb_fire_steps,
                k_spring=self.neb_spring_constant,
                barrier_threshold=self.neb_barrier_threshold,
            )
            neb_time = time.perf_counter() - t0

            for (min_id, rmsd_val), same_basin in zip(neb_candidates, results):
                if same_basin:
                    energy_diff = abs(self.minima[min_id].energy - energy)
                    logger.info(
                        f"NEB merge: min {min_id} ← new (RMSD={rmsd_val:.3f} Å, dE={energy_diff:.4f} eV)"
                    )
                    return min_id, rmsd_val, neb_time
            return None, 0.0, neb_time
        else:
            # Fallback: sequential NEB
            t0 = time.perf_counter()
            for min_id, rmsd_val in neb_candidates:
                minimum = self.minima[min_id]
                if neb_check_same_basin(
                    minimum.positions,
                    positions,
                    self.atomic_numbers,
                    self.calc,
                    n_images=self.neb_n_images,
                    fire_steps=self.neb_fire_steps,
                    k_spring=self.neb_spring_constant,
                    barrier_threshold=self.neb_barrier_threshold,
                ):
                    energy_diff = abs(minimum.energy - energy)
                    logger.info(
                        f"NEB merge: min {min_id} ← new (RMSD={rmsd_val:.3f} Å, dE={energy_diff:.4f} eV)"
                    )
                    neb_time = time.perf_counter() - t0
                    return min_id, rmsd_val, neb_time
            neb_time = time.perf_counter() - t0
            return None, 0.0, neb_time

    def add_minimum(
        self,
        positions: np.ndarray,
        energy: float,
        hessian: Optional[np.ndarray] = None,
    ) -> tuple[int, bool, np.ndarray, float]:
        """
        Add a minimum to the graph, deduplicating if similar exists.

        Positions are canonicalized (reordered to RDKit canonical atom
        ordering) before comparison and storage, making deduplication
        permutation-invariant.

        Args:
            positions: Atomic positions, shape (n_atoms, 3)
            energy: Energy in eV
            hessian: Optional Hessian matrix, shape (3*n_atoms, 3*n_atoms)

        Returns:
            (min_id, is_new, positions, neb_time_s): ID of the minimum, whether it
                was newly added, the stored (canonical) positions, and time spent on
                NEB dedup (seconds).
        """
        canonical_pos, canonical_hess, _ = self._canonicalize(positions, hessian)

        existing_id, match_rmsd, neb_time = self._find_matching_minimum(canonical_pos, energy)
        if existing_id is not None:
            # Update hessian if the existing minimum doesn't have one
            if canonical_hess is not None and self.minima[existing_id].hessian is None:
                self.minima[existing_id].hessian = canonical_hess.copy()
            # Track merge stats on the minimum
            self.minima[existing_id].n_merged += 1
            self.minima[existing_id].max_merge_rmsd = max(
                self.minima[existing_id].max_merge_rmsd, match_rmsd
            )
            return existing_id, False, self.minima[existing_id].positions, neb_time

        min_id = self._next_min_id
        self._next_min_id += 1

        minimum = Minimum(
            positions=canonical_pos.copy(),
            atomic_numbers=self.atomic_numbers.copy(),
            energy=energy,
            id=min_id,
            explored=False,
            name=self._min_namer.generate("min"),
            hessian=canonical_hess.copy() if canonical_hess is not None else None,
            discovery_timestamp=time.time(),
        )

        self.minima[min_id] = minimum
        self.graph.add_node(min_id, energy=energy, type="minimum")

        return min_id, True, minimum.positions.copy(), neb_time

    def insert_minimum_direct(
        self,
        min_id: int,
        positions: np.ndarray,
        energy: float,
        explored: bool = False,
        hessian: Optional[np.ndarray] = None,
        name: str = "",
        n_merged: int = 0,
        max_merge_rmsd: float = 0.0,
        discovery_timestamp: float = 0.0,
    ):
        """Insert a minimum with a specified ID, skipping dedup.

        Used when loading from DB where minima are already deduplicated.
        Avoids the local_id mismatch caused by add_minimum's dedup assigning
        new sequential IDs that don't match the DB's local_ids.
        """
        minimum = Minimum(
            positions=positions.copy(),
            atomic_numbers=self.atomic_numbers.copy(),
            energy=energy,
            id=min_id,
            explored=explored,
            name=name,
            hessian=hessian.copy() if hessian is not None else None,
            n_merged=n_merged,
            max_merge_rmsd=max_merge_rmsd,
            discovery_timestamp=discovery_timestamp,
        )
        self.minima[min_id] = minimum
        self.graph.add_node(min_id, energy=energy, type="minimum")
        # Keep _next_min_id above all inserted IDs so new minima don't collide
        if min_id >= self._next_min_id:
            self._next_min_id = min_id + 1

    def insert_ts_direct(
        self,
        ts_id: int,
        positions: np.ndarray,
        energy: float,
        min_fwd_id: int,
        min_bwd_id: int,
        barrier_fwd: float = 0.0,
        barrier_bwd: float = 0.0,
        eigenvalue: float = 0.0,
        hessian: Optional[np.ndarray] = None,
        rmsd_to_fwd_min: float = 0.0,
        rmsd_to_bwd_min: float = 0.0,
        endpoint_to_endpoint_rmsd: float = 0.0,
        name: str = "",
        discovery_timestamp: float = 0.0,
    ):
        """Insert a TS with a specified ID, skipping dedup + endpoint resolution.

        Used when loading from DB where TSs are already deduplicated and
        endpoint minimum IDs are known. Avoids the local_id mismatch caused
        by add_transition_state's sequential ID assignment.
        """
        ts = TransitionState(
            id=ts_id,
            positions=positions.copy(),
            energy=energy,
            atomic_numbers=self.atomic_numbers.copy(),
            min_fwd_id=min_fwd_id,
            min_bwd_id=min_bwd_id,
            barrier_fwd=barrier_fwd,
            barrier_bwd=barrier_bwd,
            eigenvalue=eigenvalue,
            hessian=hessian.copy() if hessian is not None else None,
            rmsd_to_fwd_min=rmsd_to_fwd_min,
            rmsd_to_bwd_min=rmsd_to_bwd_min,
            endpoint_to_endpoint_rmsd=endpoint_to_endpoint_rmsd,
            name=name,
            discovery_timestamp=discovery_timestamp,
        )
        self.transition_states[ts_id] = ts
        # Add edge in both directions (consistent with add_transition_state)
        lo, hi = (min_fwd_id, min_bwd_id) if min_fwd_id < min_bwd_id else (min_bwd_id, min_fwd_id)
        if not self.graph.has_edge(lo, hi):
            self.graph.add_edge(lo, hi, ts_ids=[ts_id])
        else:
            self.graph[lo][hi].setdefault("ts_ids", []).append(ts_id)
        if ts_id >= self._next_ts_id:
            self._next_ts_id = ts_id + 1

    def add_minima_batched(
        self,
        items: list[tuple[np.ndarray, float, Optional[np.ndarray]]],
    ) -> tuple[list[tuple[int, bool, np.ndarray]], float]:
        """Add multiple minima with cross-checking between them.

        Two-stage deduplication:
        1. Batch all new items against existing minima (fast RMSD + single NEB call)
        2. Batch the survivors against each other (fast RMSD + single NEB call)

        This avoids the N separate NEB calls that sequential add_minimum would trigger
        and catches duplicates between new items themselves.

        Args:
            items: List of (positions, energy, hessian_or_None) tuples.

        Returns:
            (results, neb_time_s) where results is a list of (min_id, is_new,
            canonical_positions) tuples in same order as input.
        """
        if not items:
            return [], 0.0

        # --- Canonicalize all items ---
        canonical: list[tuple[np.ndarray, Optional[np.ndarray]]] = []
        for pos, energy, hessian in items:
            can_pos, can_hess, _ = self._canonicalize(pos, hessian)
            canonical.append((can_pos, can_hess))

        n = len(items)
        # resolved[i] = min_id if matched, None if still unresolved
        resolved: list[Optional[int]] = [None] * n
        match_rmsd: list[float] = [0.0] * n

        # --- Stage 1a: Fast RMSD scan against existing minima ---
        # Compute RMSDs once and cache for potential NEB use
        logger.info(
            f"add_minima_batched: {n} items vs {len(self.minima)} existing minima "
            f"(thresh: RMSD<{self.rmsd_threshold}, dE<{self.energy_threshold})"
        )
        neb_vs_existing: list[tuple[int, int, float]] = []  # (item_idx, min_id, rmsd)
        for i in range(n):
            can_pos_i = canonical[i][0]
            energy_i = items[i][1]
            fast_merged = False
            cached_rmsds: list[tuple[int, float]] = []  # (min_id, rmsd)
            for min_id, minimum in self.minima.items():
                rmsd = compute_rmsd(minimum.positions, can_pos_i, self.atomic_numbers)
                energy_diff = abs(minimum.energy - energy_i)
                cached_rmsds.append((min_id, rmsd))
                if i < 4:  # log detail for first few items
                    logger.debug(
                        f"  Stage1a item {i} vs min {min_id}: "
                        f"RMSD={rmsd:.4f} dE={energy_diff:.5f} "
                        f"shapes={minimum.positions.shape}/{can_pos_i.shape} "
                        f"dtypes={minimum.positions.dtype}/{can_pos_i.dtype}"
                    )
                if not fast_merged:
                    if rmsd < self.rmsd_threshold and energy_diff < self.energy_threshold:
                        resolved[i] = min_id
                        match_rmsd[i] = rmsd
                        fast_merged = True
                        logger.info(
                            f"Fast merge: item {i} → min {min_id} "
                            f"(RMSD={rmsd:.4f}, dE={energy_diff:.5f})"
                        )
                        break  # no need to compute remaining RMSDs
            if not fast_merged and self.calc is not None:
                # All RMSDs already computed — reuse them for NEB pairs
                for min_id, rmsd in cached_rmsds:
                    neb_vs_existing.append((i, min_id, rmsd))

        # --- Stage 1b: Batched NEB against existing minima ---
        total_neb_time = 0.0
        if self.calc is not None and not hasattr(self.calc, "get_batched_forces_and_energy"):
            raise RuntimeError(
                "add_minima_batched requires a calculator with get_batched_forces_and_energy"
            )

        if neb_vs_existing:
            # Build pair list (filter already-resolved items)
            pairs = []
            pair_info = []  # (item_idx, min_id, rmsd)
            for item_idx, min_id, rmsd in neb_vs_existing:
                if resolved[item_idx] is not None:
                    continue
                minimum = self.minima[min_id]
                pairs.append((minimum.positions, canonical[item_idx][0],
                              minimum.energy, items[item_idx][1]))
                pair_info.append((item_idx, min_id, rmsd))

            if pairs:
                t0 = time.perf_counter()
                neb_results = neb_check_same_basin_batched(
                    pairs, self.atomic_numbers, self.calc,
                    fire_steps=self.neb_fire_steps,
                    k_spring=self.neb_spring_constant,
                    barrier_threshold=self.neb_barrier_threshold,
                )
                total_neb_time += time.perf_counter() - t0

                for (item_idx, min_id, rmsd), same_basin in zip(pair_info, neb_results):
                    if same_basin and resolved[item_idx] is None:
                        resolved[item_idx] = min_id
                        match_rmsd[item_idx] = rmsd
                        energy_diff = abs(self.minima[min_id].energy - items[item_idx][1])
                        logger.info(
                            f"NEB merge: min {min_id} ← new (RMSD={rmsd:.3f} Å, "
                            f"dE={energy_diff:.4f} eV)"
                        )

        # --- Stage 2: Cross-check survivors against each other ---
        survivors = [i for i in range(n) if resolved[i] is None]

        if len(survivors) >= 2:
            # Stage 2a: Fast RMSD cross-check
            for idx_a, si in enumerate(survivors):
                if resolved[si] is not None:
                    continue  # merged by earlier cross-check
                for sj in survivors[:idx_a]:
                    if resolved[sj] is not None:
                        continue
                    energy_diff = abs(items[si][1] - items[sj][1])
                    rmsd = compute_rmsd(
                        canonical[si][0], canonical[sj][0], self.atomic_numbers
                    )
                    if rmsd < self.rmsd_threshold and energy_diff < self.energy_threshold:
                        # si merges into sj (sj will be added first since it has lower index)
                        resolved[si] = -(sj + 1)  # negative = index into items (offset by 1)
                        match_rmsd[si] = rmsd
                        break

            # Stage 2b: Batched NEB cross-check for remaining survivors
            remaining = [i for i in survivors if resolved[i] is None]
            if self.calc is not None and len(remaining) >= 2:
                cross_pairs = []
                cross_info = []
                for idx_a in range(len(remaining)):
                    for idx_b in range(idx_a):
                        si, sj = remaining[idx_a], remaining[idx_b]
                        rmsd = compute_rmsd(
                            canonical[si][0], canonical[sj][0], self.atomic_numbers
                        )
                        cross_pairs.append((
                            canonical[sj][0], canonical[si][0],
                            items[sj][1], items[si][1],
                        ))
                        cross_info.append((si, sj, rmsd))

                if cross_pairs:
                    t0 = time.perf_counter()
                    cross_results = neb_check_same_basin_batched(
                        cross_pairs, self.atomic_numbers, self.calc,
                        fire_steps=self.neb_fire_steps,
                        k_spring=self.neb_spring_constant,
                        barrier_threshold=self.neb_barrier_threshold,
                    )
                    total_neb_time += time.perf_counter() - t0

                    for (si, sj, rmsd), same_basin in zip(cross_info, cross_results):
                        if same_basin and resolved[si] is None:
                            resolved[si] = -(sj + 1)
                            match_rmsd[si] = rmsd
                            logger.info(
                                f"NEB cross-merge: item {si} → item {sj} "
                                f"(RMSD={rmsd:.3f} Å)"
                            )

        # --- Summary ---
        n_fast = sum(1 for r in resolved if r is not None and r >= 0)
        n_cross = sum(1 for r in resolved if r is not None and r < 0)
        n_new = sum(1 for r in resolved if r is None)
        logger.info(
            f"add_minima_batched: {n} items → {n_fast} fast-merged, "
            f"{n_cross} cross-merged, {n_new} new "
            f"(existing minima: {len(self.minima)})"
        )

        # --- Materialize: add unique minima to graph, resolve references ---
        # First pass: add minima that are truly new (resolved[i] is None)
        # item_to_min_id maps item index → final min_id
        item_to_min_id: dict[int, int] = {}

        for i in range(n):
            if resolved[i] is not None and resolved[i] >= 0:
                # Matched to existing minimum
                item_to_min_id[i] = resolved[i]
            elif resolved[i] is None:
                # Truly new — add to graph
                can_pos, can_hess = canonical[i]
                energy = items[i][1]

                min_id = self._next_min_id
                self._next_min_id += 1
                minimum = Minimum(
                    positions=can_pos.copy(),
                    atomic_numbers=self.atomic_numbers.copy(),
                    energy=energy,
                    id=min_id,
                    explored=False,
                    name=self._min_namer.generate("min"),
                    hessian=can_hess.copy() if can_hess is not None else None,
                    discovery_timestamp=time.time(),
                )
                self.minima[min_id] = minimum
                self.graph.add_node(min_id, energy=energy, type="minimum")
                item_to_min_id[i] = min_id

        # Second pass: resolve cross-references (negative resolved values)
        for i in range(n):
            if resolved[i] is not None and resolved[i] < 0:
                target_item = -(resolved[i] + 1)
                # Follow the chain (target might also reference another item)
                visited = {i}
                while target_item not in item_to_min_id:
                    if resolved[target_item] is not None and resolved[target_item] < 0:
                        target_item = -(resolved[target_item] + 1)
                        if target_item in visited:
                            logger.warning(f"Cycle in cross-reference chain for item {i}")
                            break
                        visited.add(target_item)
                    else:
                        break
                if target_item in item_to_min_id:
                    item_to_min_id[i] = item_to_min_id[target_item]
                else:
                    # Shouldn't happen — treat as new minimum
                    logger.warning(
                        f"Unresolved cross-ref for item {i} → {target_item}, adding as new"
                    )
                    can_pos, can_hess = canonical[i]
                    energy = items[i][1]
                    min_id = self._next_min_id
                    self._next_min_id += 1
                    minimum = Minimum(
                        positions=can_pos.copy(),
                        atomic_numbers=self.atomic_numbers.copy(),
                        energy=energy,
                        id=min_id,
                        explored=False,
                        name=self._min_namer.generate("min"),
                        hessian=can_hess.copy() if can_hess is not None else None,
                        discovery_timestamp=time.time(),
                    )
                    self.minima[min_id] = minimum
                    self.graph.add_node(min_id, energy=energy, type="minimum")
                    item_to_min_id[i] = min_id

        # Update merge stats
        for i in range(n):
            min_id = item_to_min_id[i]
            is_existing = resolved[i] is not None
            if is_existing:
                self.minima[min_id].n_merged += 1
                self.minima[min_id].max_merge_rmsd = max(
                    self.minima[min_id].max_merge_rmsd, match_rmsd[i]
                )
                # Update hessian if missing
                can_hess = canonical[i][1]
                if can_hess is not None and self.minima[min_id].hessian is None:
                    self.minima[min_id].hessian = can_hess.copy()

        # Build results
        results = []
        for i in range(n):
            min_id = item_to_min_id[i]
            is_new = resolved[i] is None
            results.append((min_id, is_new, self.minima[min_id].positions.copy()))

        return results, total_neb_time

    def add_transition_state(
        self,
        positions: np.ndarray,
        energy: float,
        min_fwd_positions: np.ndarray,
        min_fwd_energy: float,
        min_bwd_positions: np.ndarray,
        min_bwd_energy: float,
        eigenvalue: float = 0.0,
        hessian: Optional[np.ndarray] = None,
        fwd_trajectory: Optional[RelaxationTrajectory] = None,
        bwd_trajectory: Optional[RelaxationTrajectory] = None,
        metadata: dict = None,
        min_fwd_hessian: Optional[np.ndarray] = None,
        min_bwd_hessian: Optional[np.ndarray] = None,
        _preresolved_fwd: Optional[tuple[int, np.ndarray]] = None,
        _preresolved_bwd: Optional[tuple[int, np.ndarray]] = None,
    ) -> tuple[int, bool, float]:
        """
        Add a transition state and its endpoint minima.

        Args:
            positions: TS geometry (n_atoms, 3)
            energy: TS energy (eV)
            min_fwd_positions: Forward minimum geometry
            min_fwd_energy: Forward minimum energy
            min_bwd_positions: Backward minimum geometry
            min_bwd_energy: Backward minimum energy
            eigenvalue: Imaginary mode eigenvalue
            hessian: Hessian at TS
            fwd_trajectory: Relaxation trajectory from TS to forward minimum
            bwd_trajectory: Relaxation trajectory from TS to backward minimum
            metadata: Additional metadata
            min_fwd_hessian: Optional Hessian at the forward minimum
            min_bwd_hessian: Optional Hessian at the backward minimum
            _preresolved_fwd: If provided, (min_id, canonical_positions) for the
                forward endpoint — skips add_minimum for it.
            _preresolved_bwd: If provided, (min_id, canonical_positions) for the
                backward endpoint — skips add_minimum for it.

        Returns:
            (ts_id, is_new, neb_time_s): ID of the TS, whether it was newly added,
                and time spent on NEB dedup for both endpoint minima (seconds).
        """
        # Save original positions for TS-to-endpoint RMSD calculations.
        # These share the same atom ordering as the TS (from the simulation),
        # which may differ from canonical ordering after add_minimum.
        orig_fwd_pos = min_fwd_positions.copy()
        orig_bwd_pos = min_bwd_positions.copy()

        total_neb_time = 0.0
        if _preresolved_fwd is not None:
            min_fwd_id, canonical_fwd_pos = _preresolved_fwd
        else:
            min_fwd_id, _, canonical_fwd_pos, neb_time_fwd = self.add_minimum(
                min_fwd_positions, min_fwd_energy, hessian=min_fwd_hessian
            )
            total_neb_time += neb_time_fwd

        if _preresolved_bwd is not None:
            min_bwd_id, canonical_bwd_pos = _preresolved_bwd
        else:
            min_bwd_id, _, canonical_bwd_pos, neb_time_bwd = self.add_minimum(
                min_bwd_positions, min_bwd_energy, hessian=min_bwd_hessian
            )
            total_neb_time += neb_time_bwd

        # Compute barriers FIRST so the dedup check can use them.
        def _safe_barrier(trajectory, fallback_diff: float) -> float:
            if trajectory is None:
                return fallback_diff
            try:
                return compute_barrier_from_trajectory(trajectory)
            except (ValueError, AttributeError, IndexError) as e:
                logger.debug(f"barrier integration failed ({e}); using endpoint diff")
                return fallback_diff

        barrier_fwd = _safe_barrier(fwd_trajectory, energy - min_fwd_energy)
        barrier_bwd = _safe_barrier(bwd_trajectory, energy - min_bwd_energy)

        # Check if this TS already exists (same endpoints)
        for ts_id, ts in self.transition_states.items():
            endpoints_match = (
                ts.min_fwd_id == min_fwd_id and ts.min_bwd_id == min_bwd_id
            ) or (ts.min_fwd_id == min_bwd_id and ts.min_bwd_id == min_fwd_id)
            if endpoints_match:
                rmsd = compute_rmsd(ts.positions, positions, self.atomic_numbers)
                energy_diff = abs(ts.energy - energy)
                # Ultra-low RMSD = same TS regardless of energy noise
                if rmsd < 0.01:
                    return ts_id, False, total_neb_time
                # Barrier-profile dedup: same-endpoint TSs with similar barriers
                # are the same saddle point from different MD trajectories.
                # Check both orientations (fwd/bwd can flip).
                BARRIER_TOL = 0.1  # eV
                same = abs(ts.barrier_fwd - barrier_fwd) < BARRIER_TOL and abs(ts.barrier_bwd - barrier_bwd) < BARRIER_TOL
                swap = abs(ts.barrier_fwd - barrier_bwd) < BARRIER_TOL and abs(ts.barrier_bwd - barrier_fwd) < BARRIER_TOL
                if same or swap:
                    return ts_id, False, total_neb_time
                # Standard dedup: RMSD < threshold AND energy < threshold
                if rmsd < self.rmsd_threshold and energy_diff < self.energy_threshold:
                    return ts_id, False, total_neb_time

        ts_id = self._next_ts_id
        self._next_ts_id += 1

        # TS-to-endpoint RMSDs use original positions (consistent atom order with TS).
        # Endpoint-to-endpoint RMSD uses canonical positions (permutation-invariant).
        rmsd_to_fwd_min = compute_rmsd(positions, orig_fwd_pos, self.atomic_numbers)
        rmsd_to_bwd_min = compute_rmsd(positions, orig_bwd_pos, self.atomic_numbers)
        endpoint_to_endpoint_rmsd = compute_rmsd(canonical_fwd_pos, canonical_bwd_pos, self.atomic_numbers)

        ts = TransitionState(
            id=ts_id,
            positions=positions.copy(),
            energy=energy,
            atomic_numbers=self.atomic_numbers.copy(),
            min_fwd_id=min_fwd_id,
            min_bwd_id=min_bwd_id,
            barrier_fwd=barrier_fwd,
            barrier_bwd=barrier_bwd,
            eigenvalue=eigenvalue,
            hessian=hessian.copy() if hessian is not None else None,
            rmsd_to_fwd_min=rmsd_to_fwd_min,
            rmsd_to_bwd_min=rmsd_to_bwd_min,
            endpoint_to_endpoint_rmsd=endpoint_to_endpoint_rmsd,
            fwd_trajectory=fwd_trajectory,
            bwd_trajectory=bwd_trajectory,
            metadata=metadata or {},
            name=self._ts_namer.generate("ts"),
            discovery_timestamp=time.time(),
        )

        self.transition_states[ts_id] = ts

        # Add edge to graph.
        # Convention: edge is always stored as (lower_id, higher_id) so that
        # barrier_fwd / rmsd_to_fwd_min refer to the lower-ID node and
        # barrier_bwd / rmsd_to_bwd_min refer to the higher-ID node.
        # This ensures consistent semantics regardless of discovery direction.
        if min_fwd_id <= min_bwd_id:
            edge_src, edge_tgt = min_fwd_id, min_bwd_id
            b_src, b_tgt = barrier_fwd, barrier_bwd
            r_src, r_tgt = rmsd_to_fwd_min, rmsd_to_bwd_min
        else:
            edge_src, edge_tgt = min_bwd_id, min_fwd_id
            b_src, b_tgt = barrier_bwd, barrier_fwd
            r_src, r_tgt = rmsd_to_bwd_min, rmsd_to_fwd_min

        self.graph.add_edge(
            edge_src,
            edge_tgt,
            ts_id=ts_id,
            ts_energy=energy,
            barrier_fwd=b_src,
            barrier_bwd=b_tgt,
            rmsd_to_fwd_min=r_src,
            rmsd_to_bwd_min=r_tgt,
            endpoint_to_endpoint_rmsd=endpoint_to_endpoint_rmsd,
            rmsd_weight=1 / endpoint_to_endpoint_rmsd if endpoint_to_endpoint_rmsd > 0 else 0.0,
            energy_weight=(
                np.exp(-barrier_fwd / (k_b * T)) + np.exp(-barrier_bwd / (k_b * T))
            )
            / 2,
        )

        return ts_id, True, total_neb_time

    def get_unexplored_minima(self) -> list[Minimum]:
        """Get all minima that haven't been explored yet."""
        return [m for m in self.minima.values() if not m.explored]

    def mark_explored(self, min_id: int):
        """Mark a minimum as explored."""
        if min_id in self.minima:
            self.minima[min_id].explored = True

    def get_stats(self) -> dict:
        """Get statistics about the graph."""
        return {
            "n_minima": len(self.minima),
            "n_ts": len(self.transition_states),
            "n_edges": self.graph.number_of_edges(),
            "n_unexplored": len(self.get_unexplored_minima()),
            "n_connected_components": nx.number_connected_components(self.graph),
        }

    def to_networkx(self) -> nx.Graph:
        """Get the networkx graph with all attributes."""
        G = self.graph.copy()

        # Add minimum data as node attributes
        for min_id, minimum in self.minima.items():
            G.nodes[min_id]["positions"] = minimum.positions.tolist()
            G.nodes[min_id]["explored"] = minimum.explored

        return G

    def export_gexf(self, path: str):
        """Export graph to GEXF format for visualization."""
        G = self.to_networkx()

        # GEXF doesn't support numpy arrays, convert to strings
        for node in G.nodes():
            if "positions" in G.nodes[node]:
                G.nodes[node]["positions"] = str(G.nodes[node]["positions"])

        nx.write_gexf(G, path)

    def save(self, path: str):
        """Save the full PESGraph to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "PESGraph":
        """Load a PESGraph from a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self):
        stats = self.get_stats()
        return (
            f"PESGraph(n_minima={stats['n_minima']}, "
            f"n_ts={stats['n_ts']}, "
            f"n_unexplored={stats['n_unexplored']})"
        )
