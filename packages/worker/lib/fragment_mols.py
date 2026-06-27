"""
Functions for fragmenting molecules into disconnected components.

Uses connected component detection based on covalent radii to identify
separate molecular fragments after a decomposition reaction.
"""

from typing import Optional, Callable
from pathlib import Path
from loguru import logger
import numpy as np
import torch
from ase.io import read
from ase.data import covalent_radii
from ase.calculators.calculator import Calculator

from lib.types import Conformer
from lib.exploration import ExplorationContext
from lib.pes_explorer.newton_minimize import optimize_fire
from lib.utils import atomic_symbol, check_tetrose_limit


def load_fragment_conformers(
    fragment_path: Path,
    prepare_inputs_fn: Callable,
    energy_fn: Callable[[Conformer], float],
    device: str,
) -> list[Conformer]:
    """
    Load fragment molecules from xyz files with energies.

    Args:
        fragment_path: Directory containing .xyz files
        prepare_inputs_fn: Function to prepare ASE Atoms for model (e.g., ts_model.model.prepare_inputs)
        energy_fn: Function to compute energy from conformer
        device: Device for tensor operations

    Returns:
        List of Conformer objects with energies set
    """
    fragment_conformers = []
    for fragment_file in fragment_path.glob("*.xyz"):
        fragment = read(fragment_file)
        fragment_mol = prepare_inputs_fn([fragment])
        fragment_mol = {k: v.to(device) for k, v in fragment_mol.items()}
        fragment_conformer = Conformer.from_batch(fragment_mol)

        # Compute energy using energy model
        fragment_conformer.energy = energy_fn(fragment_conformer)
        fragment_conformer = fragment_conformer.to_numpy()

        fragment_conformers.append(fragment_conformer)
        logger.debug(
            f"Loaded fragment {fragment_file.stem} with energy {fragment_conformer.energy:.6f} eV"
        )

    return fragment_conformers


def is_valid_fragment(
    atomic_numbers: torch.Tensor, positions: torch.Tensor = None
) -> bool:
    """
    Check if a fragment is chemically valid.

    Rules:
    - Non-empty
    - Single atoms must be H
    - Must satisfy composition limit (C5H10O5, pentose-sized)
    """
    if atomic_numbers.numel() == 0:
        return False
    # Single atom fragments: only allow H atoms
    if atomic_numbers.numel() == 1:
        if atomic_symbol(atomic_numbers.item()) != "H":
            logger.debug("invalid fragment: single atom that is not H")
            return False
    if not check_tetrose_limit(atomic_numbers):
        return False
    return True


def get_connected_components(
    positions: torch.Tensor, atomic_numbers: torch.Tensor
) -> list[torch.Tensor]:
    """
    Find connected components based on covalent bond distances.

    Args:
        positions: Atomic positions, shape (n_atoms, 3)
        atomic_numbers: Atomic numbers, shape (n_atoms,)

    Returns:
        List of index tensors, one per connected component
    """
    n_atoms = atomic_numbers.numel()
    adj = torch.zeros((n_atoms, n_atoms), dtype=bool)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            try:
                radius_i = covalent_radii[atomic_numbers[i].item()]
                radius_j = covalent_radii[atomic_numbers[j].item()]
            except IndexError:
                radius_i, radius_j = 0.7, 0.7  # Default for unknown atoms
            cutoff = 1.3 * (radius_i + radius_j)
            dist = torch.norm(positions[i] - positions[j]).item()
            if dist < cutoff:
                adj[i, j] = adj[j, i] = True

    visited = [False] * n_atoms
    components = []

    for i in range(n_atoms):
        if not visited[i]:
            stack = [i]
            component = []
            while stack:
                current = stack.pop()
                if not visited[current]:
                    visited[current] = True
                    component.append(current)
                    neighbors = torch.where(adj[current])[0]
                    stack.extend(neighbors.tolist())
            components.append(torch.tensor(component, dtype=torch.int64))

    return [torch.sort(c)[0] for c in components]


def split_conformer_into_fragments(
    conformer: Conformer,
    stats_tracker: Optional[dict] = None,
) -> list[Conformer] | None:
    """
    Split a conformer into disconnected fragments.

    Args:
        conformer: The conformer to split
        stats_tracker: Optional dict to track decomposition statistics

    Returns:
        List of fragment Conformers (without energies), or None if fragmentation is invalid.
        Returns [conformer] if only one component (no fragmentation).
    """
    conf = conformer.to_torch()
    positions = conf.positions.cpu().clone()
    atomic_numbers = conf.atomic_numbers.clone()

    components = get_connected_components(positions, atomic_numbers)
    logger.debug(f"Found {len(components)} connected components")

    if stats_tracker:
        stats_tracker["attempted"] += 1

    # Single component - no fragmentation
    if len(components) == 1:
        if stats_tracker:
            stats_tracker["single_component"] += 1
        return [conformer]

    # First pass: check if ALL components are valid
    all_valid = True
    invalid_reasons = []
    for i, component_idxs in enumerate(components):
        component_anum = atomic_numbers[component_idxs]
        component_pos = positions[component_idxs]
        if not is_valid_fragment(component_anum, component_pos):
            all_valid = False
            composition = {"C": 0, "H": 0, "O": 0}
            for at in component_anum:
                symb = atomic_symbol(int(at))
                if symb in composition:
                    composition[symb] += 1
            invalid_reasons.append(f"Component {i}: {composition}")
            break

    # If any component is invalid, reject entire fragmentation
    if not all_valid:
        if stats_tracker:
            stats_tracker["rejected_invalid_fragments"] += 1
        logger.debug(f"Fragmentation rejected: invalid fragments - {invalid_reasons}")
        return None

    # Second pass: create fragment conformers
    from lib.compound import get_smiles_and_charge_from_structure

    fragment_conformers = []
    for component_idxs in components:
        component_anum = atomic_numbers[component_idxs]
        component_pos = positions[component_idxs]

        # Center the fragment
        component_pos = component_pos - component_pos.mean(dim=0, keepdim=True)

        # Determine charge for this fragment
        fragment_charge = 0
        result = get_smiles_and_charge_from_structure(
            component_anum.numpy(), component_pos.numpy()
        )
        if result is not None:
            fragment_charge = result[1]

        fragment_conf = Conformer(
            positions=component_pos,
            atomic_numbers=component_anum,
            energy=None,  # Energy assigned by caller
            charge=fragment_charge,
        )
        fragment_conformers.append(fragment_conf)

    # Charge conservation check
    original_charge = conformer.charge if conformer.charge is not None else 0
    fragment_charges_sum = sum(f.charge if f.charge is not None else 0 for f in fragment_conformers)
    if fragment_charges_sum != original_charge:
        logger.warning(
            f"Charge conservation violated: original charge={original_charge}, "
            f"sum of fragment charges={fragment_charges_sum} "
            f"(fragments: {[f.charge for f in fragment_conformers]})"
        )

    logger.debug(f"Created {len(fragment_conformers)} valid fragments")
    return fragment_conformers


def relax_conformer(
    fragment: Conformer,
    calc: Calculator,
    fmax: float = 0.005,
    max_steps: int = 200,
) -> Optional[Conformer]:
    """
    Relax a single fragment to its individual PES minimum using FIRE.

    After splitting a combined system into fragments, each fragment's geometry
    is a minimum of the combined PES but not of the individual fragment's PES.
    This function relaxes the fragment on its own PES.

    Args:
        fragment: Fragment conformer to relax
        calc: ASE Calculator for force evaluation
        fmax: Max force convergence criterion (eV/A)
        max_steps: Maximum FIRE steps

    Returns:
        Relaxed Conformer with energy set, or None if relaxation fails.
    """
    atoms = fragment.to_ase_atoms()
    atoms.calc = calc

    try:
        initial_forces = atoms.get_forces()
        if not np.isfinite(initial_forces).all():
            logger.warning("relax_conformer: initial forces contain NaN/inf")
            return None
    except Exception as e:
        logger.warning(f"relax_conformer: failed to compute initial forces: {e}")
        return None

    try:
        result = optimize_fire(
            calc,
            atoms,
            max_steps=max_steps,
            force_max_tol=fmax,
            force_rms_tol=fmax * 0.5,
            hessian_retrace_interval=0,
        )

        is_valid = result.converged or result.final_force_max < fmax * 10
        if not is_valid:
            logger.debug(
                f"relax_conformer: not converged "
                f"(force_max={result.final_force_max:.4f} eV/A)"
            )
            return None

        # Center the relaxed positions
        relaxed_positions = result.positions - result.positions.mean(axis=0, keepdims=True)

        return Conformer(
            positions=relaxed_positions,
            atomic_numbers=fragment.to_numpy().atomic_numbers,
            energy=result.energy,
            charge=fragment.charge,
        )
    except Exception as e:
        logger.warning(f"relax_conformer: FIRE failed: {e}")
        return None


def relax_fragments(
    fragments: list[Conformer],
    calc: Calculator,
    fmax: float = 0.005,
    max_steps: int = 200,
) -> Optional[list[Conformer]]:
    """
    Relax all fragments to their individual PES minima.

    All-or-nothing: returns None if any fragment fails to relax.

    Args:
        fragments: List of fragment conformers
        calc: ASE Calculator
        fmax: Max force convergence criterion (eV/A)
        max_steps: Maximum FIRE steps per fragment

    Returns:
        List of relaxed Conformers with energies set, or None if any fails.
    """
    relaxed = []
    for i, frag in enumerate(fragments):
        calc.reset()
        relaxed_frag = relax_conformer(frag, calc, fmax=fmax, max_steps=max_steps)
        if relaxed_frag is None:
            logger.warning(f"Fragment relaxation failed for fragment {i}")
            return None
        relaxed.append(relaxed_frag)
    return relaxed


def populate_context_fragments(
    ctx: ExplorationContext,
    energy_fn: Callable[[Conformer], float],
    calc: Optional[Calculator] = None,
    stats_tracker: Optional[dict] = None,
) -> bool:
    """
    Check for fragmentation in the MIN endpoint and populate context accordingly.

    This function checks if ctx.min_conformer has fragmented into multiple
    disconnected components. If so, it populates min_fragments, min_fragment_ids,
    and min_fragment_energies.

    When calc is provided and multiple fragments are found, each fragment is
    relaxed on its own PES before energy assignment. This corrects for
    inter-fragment interactions that distort individual fragment geometries.

    Args:
        ctx: ExplorationContext with min_conformer set
        energy_fn: Function to compute energy from conformer (takes Conformer, returns float)
        calc: Optional ASE Calculator for fragment relaxation. When None,
            fragments are used as-is (original behavior).
        stats_tracker: Optional dict for statistics

    Returns:
        True if successful (single molecule or valid fragments),
        False if fragmentation was invalid.
    """
    if ctx.min_conformer is None:
        logger.warning("populate_context_fragments called with no min_conformer")
        return False

    fragments = split_conformer_into_fragments(ctx.min_conformer, stats_tracker)

    if fragments is None:
        # Invalid fragmentation
        return False

    if len(fragments) == 1:
        # No fragmentation - min_conformer stays as is
        # Set energy and ID if not already set
        if ctx.min_energy is None:
            ctx.min_energy = energy_fn(ctx.min_conformer)
        if ctx.min_id is None:
            ctx.min_id = ExplorationContext.compute_id(ctx.min_conformer)
        return True

    # Multiple fragments - optionally relax each on its own PES
    if calc is not None:
        relaxed = relax_fragments(fragments, calc)
        if relaxed is not None:
            fragments = relaxed
        else:
            logger.warning(
                "Fragment relaxation failed, falling back to unrelaxed geometries"
            )

    # Compute energies and IDs
    fragment_energies = []
    for f in fragments:
        if f.energy is not None:
            # Energy already set by relax_conformer
            fragment_energies.append(f.energy)
        else:
            energy = energy_fn(f)
            f.energy = energy
            fragment_energies.append(energy)

    ctx.min_fragments = fragments
    ctx.min_fragment_ids = [ExplorationContext.compute_id(f) for f in fragments]
    ctx.min_fragment_energies = fragment_energies

    # Clear single-molecule fields since we have fragments
    ctx.min_conformer = None
    ctx.min_id = None
    ctx.min_energy = None

    return True
