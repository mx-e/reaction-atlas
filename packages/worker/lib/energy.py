"""
Energy computation for molecular structures.

Uses md-et pip package to compute formation energies from conformers.
Simplified from the original version that loaded from local training runs.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Callable
from loguru import logger

from lib.types import Conformer


def get_energy_model(run_dir_path=None, device=None, checkpoint_name="best_model"):
    """Load energy model using md-et package.

    For backwards compatibility, accepts the same args as the original
    but ignores run_dir_path and checkpoint_name.
    """
    from md_et import load_calculator

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # The md-et calculator acts as both forces calculator and energy model.
    # We load a separate instance for energy-only computations.
    calc = load_calculator(variant="12l", device=device, filter_forces=False)
    return calc


def create_energy_fn(energy_model) -> Callable[[Conformer], float]:
    """Create an energy function from an md-et calculator.

    The md-et calculator provides energy via the ASE interface.
    """
    from ase import Atoms

    @torch.no_grad()
    def energy_fn(conformer: Conformer) -> float:
        conf = conformer.to_numpy()
        atoms = Atoms(
            numbers=conf.atomic_numbers.flatten(),
            positions=conf.positions,
        )
        atoms.info['charge'] = conformer.charge or 0
        atoms.calc = energy_model
        energy = atoms.get_potential_energy()  # eV
        return float(energy)

    return energy_fn


def get_conformer_energy(conformer: Conformer, energy_model) -> float:
    """Compute energy for a single conformer."""
    fn = create_energy_fn(energy_model)
    return fn(conformer)


def assign_conformer_energies(
    conformers: List[Conformer],
    energy_model,
) -> List[Conformer]:
    """Assign energies to a list of conformers."""
    energy_fn = create_energy_fn(energy_model)
    for conformer in conformers:
        energy_value = energy_fn(conformer)
        if energy_value >= 0:
            logger.warning(f"WARNING: Conformer has non-negative energy: {energy_value}")
        conformer.energy = energy_value
    return conformers


def compute_barrier_from_trajectory(trajectory, use_hessians: bool = True) -> float:
    """Force integration along an IRC trajectory.

    Trajectory goes TS → endpoint (positions[0] = TS, positions[-1] = minimum).
    Returns barrier height in eV (positive = TS above endpoint).

    Args:
        trajectory: RelaxationTrajectory with positions, forces, and optionally hessians.
        use_hessians: If True, use quadratic model (E ≈ E₀ - F·δr + ½ δrᵀ H δr)
            at frames with Hessians. Best for short trajectories (PES, ~15 frames).
            If False, use trapezoidal rule only. Best for long trajectories
            (reactions, ~50-350 frames) where Hessian noise accumulates.
    """

    positions = trajectory.positions
    forces = trajectory.forces
    hessians = getattr(trajectory, 'hessians', None)

    if not positions or not forces or len(positions) < 2:
        raise ValueError("Trajectory must have >= 2 frames with positions and forces")

    # Integrate energy change along path: TS → endpoint
    integrated = 0.0
    have_hessians = use_hessians and hessians and any(h is not None for h in hessians)

    for i in range(len(positions) - 1):
        dr = positions[i + 1] - positions[i]

        if have_hessians and hessians[i] is not None:
            # Quadratic model: ΔE = -F·δr + ½ δrᵀ H δr
            dr_flat = dr.flatten()
            F = forces[i].flatten()
            H = hessians[i]
            integrated += -np.dot(F, dr_flat) + 0.5 * np.dot(dr_flat, H @ dr_flat)
        else:
            # Trapezoidal: ΔE ≈ -½(F_i + F_{i+1})·δr
            f_avg = 0.5 * (forces[i] + forces[i + 1])
            integrated += -np.sum(f_avg * dr)

    # integrated = E(endpoint) - E(TS), so barrier = -integrated
    return -integrated


def make_collate_fn(device: str):
    """Compatibility stub — not needed with md-et package.

    The md-et package handles input preparation internally.
    Kept for any code that still references this function.
    """
    raise NotImplementedError(
        "make_collate_fn is not needed with the md-et pip package. "
        "Use md_et.load_calculator() instead."
    )
