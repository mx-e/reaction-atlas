"""
Generative TS pipeline: diffusion-based TS generation, IRC verification, and energy filtering.

This module handles the full pipeline for discovering new reactions:
1. Diffuse starting structures to generate TS candidates
2. Filter by forward barrier
3. Verify TS via IRC (intrinsic reaction coordinate)
4. Check product fragmentation
5. Validate backward barrier
6. Add valid reactions to graph
"""

import math
import os
import time
from typing import Optional
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from torch import nn

from schnetpack import properties as Props
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
)

from lib.ts_model import TSDenoiser
from lib.types import Conformer
from lib.exploration import ExplorationContext
from lib.reaction_graph import ReactionGraph
from lib.compound import get_smiles_from_conformer
from lib.pes_explorer.prfo import compute_hessian_eigenvalues
from lib.pes_explorer.newton_minimize import optimize_fire
from lib.pes_explorer.pes_graph import RelaxationTrajectory
from lib.utils import (
    collate_mol_batch,
    unbatch_mols,
    min_rmsd_conformer_to_compound,
)
from lib.fragment_mols import (
    split_conformer_into_fragments,
    relax_fragments,
    populate_context_fragments,
)
from lib.energy import create_energy_fn, get_conformer_energy
from lib.constants import ENERGY_THRESHOLD_EV


# Barrier histogram bin edges (eV). Used to bucket barrier_fwd / barrier_bwd for
# diagnostics. Anything below the first edge lands in "<<first_edge>"; anything
# above the last edge lands in ">last_edge".
_BARRIER_BINS_EV = [-0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def _barrier_bin_key(value: float, direction: str, context_type: str) -> str:
    """Return a bin label for `value` using _BARRIER_BINS_EV. Shape:
    'barrier_{fwd|bwd}_bin_{context_type}_<lt_m05|m05_m02|...|gt_2>'.

    `direction` is "forward" or "backward"; we normalize to "fwd"/"bwd"
    so keys stay compact and match the record_batch_stats prefix filter."""
    dir_short = {"forward": "fwd", "backward": "bwd"}.get(direction, direction)
    edges = _BARRIER_BINS_EV
    if value < edges[0]:
        suffix = f"lt_{edges[0]:+.2f}"
    else:
        idx = None
        for i in range(len(edges) - 1):
            if edges[i] <= value < edges[i + 1]:
                idx = i
                break
        if idx is None:
            suffix = f"gt_{edges[-1]:+.2f}"
        else:
            suffix = f"{edges[idx]:+.2f}_{edges[idx + 1]:+.2f}"
    suffix = suffix.replace("+", "p").replace("-", "m").replace(".", "")
    return f"barrier_{dir_short}_bin_{context_type}_{suffix}"


def filter_contexts_by_barrier(
    contexts: list[ExplorationContext],
    energy_stats: dict,
    direction: str,
    threshold: float = ENERGY_THRESHOLD_EV,
    bin_stats: Optional[dict] = None,
) -> list[ExplorationContext]:
    """
    Filter contexts by energy barrier validation.

    Args:
        contexts: Contexts to filter
        energy_stats: Dict to update with validation statistics
        direction: "forward" (MIN -> TS) or "backward" (TS -> MIN)
        threshold: Maximum allowed barrier in eV
        bin_stats: Optional dict to write per-batch barrier-value histogram
            bins (incremented in-place). Keys are
            "barrier_{dir}_bin_{merge|single}_<bucket>".
    """
    validate_fn = {
        "forward": ExplorationContext.validate_forward_barrier,
        "backward": ExplorationContext.validate_backward_barrier,
    }[direction]
    invalid_stat_key = {
        "forward": "min_to_ts_invalid",
        "backward": "ts_to_min_invalid",
    }[direction]
    get_barrier_fn = {
        "forward": ExplorationContext.get_ts_barrier_forward,
        "backward": ExplorationContext.get_ts_barrier_backward,
    }[direction]

    valid_contexts = []
    for ctx in contexts:
        is_valid, reason, stat_key = validate_fn(ctx, threshold)

        # Emit a diagnostic histogram of the raw barrier value, regardless of
        # accept/reject. Useful for tuning BARRIER_FWD_MIN_EV without changing
        # policy. Split by context type (merge/single) so we can see where
        # merges land in particular.
        if bin_stats is not None:
            barrier_val = get_barrier_fn(ctx)
            if barrier_val is not None and math.isfinite(barrier_val):
                ctype = "merge" if ctx.was_merged else "single"
                key = _barrier_bin_key(float(barrier_val), direction, ctype)
                bin_stats[key] = bin_stats.get(key, 0) + 1

        if is_valid:
            energy_stats[stat_key] = energy_stats.get(stat_key, 0) + 1
            valid_contexts.append(ctx)
            logger.debug(f"Valid {direction} barrier: {reason}")
        else:
            energy_stats[invalid_stat_key] = energy_stats.get(invalid_stat_key, 0) + 1
            if stat_key == "threshold_violation":
                energy_stats["threshold_violations"] = energy_stats.get("threshold_violations", 0) + 1
            energy_stats["branches_terminated"] = energy_stats.get("branches_terminated", 0) + 1
            logger.warning(f"Invalid {direction} barrier: {reason}")

    return valid_contexts


def diffuse_new_ts(
    mols: list[dict],
    ts_model: TSDenoiser,
    t: int,
    max_denoising_steps: int,
    device: str,
) -> list[dict]:
    batch = collate_mol_batch(mols)

    # sometimes fails randomly (retry shoud be fine because of randomness)
    @retry(
        stop=stop_after_attempt(3),  # Retry up to 3 times
        retry=retry_if_exception_type(ValueError),  # Only retry on ValueError
        reraise=True,  # Reraise the exception after all retries are exhausted
        before_sleep=lambda retry_state: logger.warning(
            f"Retry {retry_state.attempt_number} after denosiing error: {retry_state.outcome.exception()}"
        ),
    )
    def denoise_with_retry():
        copied_batch = deepcopy(batch)
        copied_batch = {k: v.to(device) for k, v in copied_batch.items()}
        copied_batch[Props.R], _ = ts_model.diffuser.diff_proc.diffuse(
            copied_batch[Props.R],
            copied_batch[Props.idx_m],
            t=torch.tensor(t, device=device),
        )
        relaxed_batch, _, _ = ts_model.model.denoise(
            copied_batch, max_steps=max_denoising_steps
        )
        return relaxed_batch

    logger.info("Denoising ...")
    with torch.no_grad():
        try:
            relaxed_batch = denoise_with_retry()
        except ValueError as e:
            with open("error_log.txt", "a") as file:
                file.write(f"Node ID: {batch['_idx_m']}\n")
                file.write(f"Positions: {batch['_positions'].cpu().numpy()}\n")
            logger.error(
                "Error in denoising after multiple retries",
                batch["_idx_m"],
                "see log: error_log.txt",
            )
            return None
        batch.update(relaxed_batch)
        batch = {k: v.to("cpu") for k, v in batch.items()}
        updated_mols = unbatch_mols(batch)
    return updated_mols


def _relax_from_displaced_newton(
    start_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    calc: Calculator,
    fmax: float = 0.005,
    max_steps: int = 200,
    charge: int = 0,
) -> tuple[Optional[torch.Tensor], Optional[RelaxationTrajectory]]:
    """
    Relax from displaced geometry using trust-region Newton optimizer.

    Collects full trajectory data (positions, energies, forces, hessians).
    Returns (relaxed_positions_centered, trajectory) or (None, None) if failed.
    """
    atoms = Atoms(numbers=atomic_numbers, positions=start_positions.copy())
    atoms.info['charge'] = charge
    atoms.calc = calc

    # Validate initial forces
    try:
        initial_forces = atoms.get_forces()
        if not np.isfinite(initial_forces).all():
            return None, None
        max_force = np.abs(initial_forces).max()
        if max_force < 1e-10:
            logger.debug(
                f"Newton relax skipped: forces near zero (max={max_force:.2e})"
            )
            return None, None
    except Exception as e:
        logger.warning(f"Newton relax skipped: failed to compute initial forces: {e}")
        return None, None

    try:
        result = optimize_fire(
            calc,
            atoms,
            max_steps=max_steps,
            force_max_tol=fmax,
            force_rms_tol=fmax * 0.5,
            hessian_retrace_interval=5,
            verbose=False,
        )

        # Accept if converged OR forces within 10x tolerance
        is_valid = result.converged or result.final_force_max < fmax * 10
        if not is_valid:
            logger.debug(
                f"Newton relax rejected: not converged "
                f"(force_max={result.final_force_max:.4f} eV/A)"
            )
            return None, None

        # Center the positions
        relaxed_positions = torch.tensor(result.positions, dtype=torch.float64)
        relaxed_positions = relaxed_positions - relaxed_positions.mean(
            dim=0, keepdim=True
        )

        # Convert MinimizationTrajectory to RelaxationTrajectory
        trajectory = RelaxationTrajectory(
            positions=result.trajectory.positions,
            energies=result.trajectory.energies,
            forces=result.trajectory.forces,
            hessians=result.trajectory.hessians,
        )

        return relaxed_positions, trajectory

    except Exception as e:
        logger.warning(f"Newton relax failed: {e}")
        return None, None


def compute_irc_displacement(
    eigenvalue: float,
    base_displacement: float = 0.2,
    min_displacement: float = 0.05,
    max_displacement: float = 0.5,
) -> float:
    curvature = np.abs(eigenvalue)
    if curvature < 1e-6:
        return max_displacement

    displacement = base_displacement / np.sqrt(curvature)
    displacement = np.clip(displacement, min_displacement, max_displacement)

    return float(displacement)


def _bump_irc(irc_stats: Optional[dict], key: str, was_merged: bool) -> None:
    """Increment an irc_stats counter and its per-type mirror.

    Writes both the global key (e.g. "irc_hessian_fail") and the per-type
    variant ("irc_hessian_fail_merge" or "irc_hessian_fail_single") so we can
    see where merges vs singles die in the IRC stage.
    """
    if irc_stats is None:
        return
    irc_stats[key] = irc_stats.get(key, 0) + 1
    suffix = "_merge" if was_merged else "_single"
    irc_stats[key + suffix] = irc_stats.get(key + suffix, 0) + 1


def _validate_ts_hessian(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    charge: int,
    calc: Calculator,
    eigenvalue_threshold: float = -0.01,
) -> Optional[tuple[float, np.ndarray]]:
    """
    Validate TS character via Hessian and extract imaginary mode.

    Returns (imaginary_eigenvalue, imaginary_mode_3d) if the geometry is a valid TS,
    or None if it fails validation.
    """
    atoms = Atoms(numbers=atomic_numbers, positions=positions)
    atoms.info['charge'] = charge
    atoms.calc = calc

    try:
        eigenvalues, eigenvectors = compute_hessian_eigenvalues(
            calc, atoms, project_tr=True
        )
    except Exception as e:
        logger.warning(f"IRC failed: could not compute Hessian: {e}")
        return None

    # Filter out near-zero eigenvalues
    significant_mask = np.abs(eigenvalues) > 1e-4
    significant_eigenvalues = eigenvalues[significant_mask]

    # Check TS character - should have at least 1 negative eigenvalue
    n_negative = np.sum(significant_eigenvalues < eigenvalue_threshold)
    if n_negative == 0:
        lowest = significant_eigenvalues[0] if len(significant_eigenvalues) > 0 else 0.0
        logger.debug(
            f"IRC: Not a TS - no negative eigenvalues. "
            f"Lowest eigenvalue: {lowest:.4f}"
        )
        return None

    # Get the imaginary mode
    negative_indices = np.where(eigenvalues < eigenvalue_threshold)[0]
    imaginary_mode_idx = negative_indices[0]
    imaginary_mode = eigenvectors[:, imaginary_mode_idx]
    imaginary_eigenvalue = eigenvalues[imaginary_mode_idx]

    # Compute displacement magnitude and build 3D mode vector
    displacement_magnitude = compute_irc_displacement(imaginary_eigenvalue)
    logger.debug(
        f"IRC: Valid TS with imaginary eigenvalue "
        f"{imaginary_eigenvalue:.4f} eV/Å², displacement={displacement_magnitude:.3f} Å"
    )

    mode_3d = imaginary_mode.reshape(-1, 3)
    mode_norm = np.linalg.norm(mode_3d)
    if mode_norm < 1e-10:
        logger.warning("IRC: Imaginary mode has near-zero norm")
        return None
    mode_3d = mode_3d / mode_norm * displacement_magnitude

    return imaginary_eigenvalue, mode_3d


def _irc_displace_and_relax(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    charge: int,
    mode_3d: np.ndarray,
    calc: Calculator,
    relaxation_fmax: float = 0.005,
) -> tuple[
    Optional[tuple[torch.Tensor, RelaxationTrajectory, torch.Tensor, RelaxationTrajectory]],
    bool,
    bool,
]:
    """
    Displace along imaginary mode in both directions and relax.

    Returns (result, fwd_ok, bwd_ok) where result is
    (forward_relaxed, forward_trajectory, backward_relaxed, backward_trajectory)
    or None if either relaxation fails. fwd_ok/bwd_ok indicate per-direction success.
    """
    forward_start = positions + mode_3d
    backward_start = positions - mode_3d

    # Both directions must succeed (one = reactant, other = product), so bail
    # early if the first one fails to avoid a wasted relaxation.
    calc.reset()
    forward_relaxed, forward_trajectory = _relax_from_displaced_newton(
        forward_start, atomic_numbers, calc, fmax=relaxation_fmax, max_steps=350,
        charge=charge,
    )
    if forward_relaxed is None:
        logger.debug("IRC: forward direction failed to relax, skipping backward")
        return None, False, False

    calc.reset()
    backward_relaxed, backward_trajectory = _relax_from_displaced_newton(
        backward_start, atomic_numbers, calc, fmax=relaxation_fmax, max_steps=350,
        charge=charge,
    )
    if backward_relaxed is None:
        logger.debug("IRC: backward direction failed to relax")
        return None, True, False

    return (forward_relaxed, forward_trajectory, backward_relaxed, backward_trajectory), True, True


def _check_matches_reactant(
    relaxed_positions: torch.Tensor,
    atomic_numbers: np.ndarray,
    ctx: ExplorationContext,
    graph: ReactionGraph,
    rmsd_threshold: float,
) -> tuple[bool, float]:
    """
    Check if a relaxed IRC endpoint matches the reactant.

    For merge contexts: splits into fragments and matches against both source compounds.
    For single-molecule contexts: matches against the source compound's conformers.

    Returns (is_match, rmsd).
    """
    relaxed_conf = Conformer(
        positions=relaxed_positions,
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
    )

    if ctx.was_merged:
        return _check_matches_reactant_merged(
            relaxed_conf, ctx, graph, rmsd_threshold
        )

    # Single-molecule: check against source compound
    assert ctx.source_compound_smiles is not None, (
        "Single-molecule context must have source compound for IRC validation"
    )
    compound = graph.get_compound(ctx.source_compound_smiles)
    if compound is not None and compound.pes_graph is not None:
        compound_minima = np.array(
            [m.positions for m in compound.pes_graph.minima.values()]
        )
        compound_atomic_numbers = np.array(compound.sorted_atomic_numbers)
        min_rmsd = min_rmsd_conformer_to_compound(
            relaxed_conf, compound_minima, compound_atomic_numbers
        )
        return min_rmsd < rmsd_threshold, min_rmsd
    return False, 9999.0


def _check_matches_reactant_merged(
    relaxed_conf: Conformer,
    ctx: ExplorationContext,
    graph: ReactionGraph,
    rmsd_threshold: float,
) -> tuple[bool, float]:
    """
    Check if a relaxed IRC endpoint matches both reactants of a merge reaction.

    Splits the relaxed geometry into fragments and tries both fragment-to-compound
    assignments, returning the best match.
    """
    assert len(ctx.merge_component_smiles) == 2, (
        "Expected exactly 2 merge components SMILES for merged context"
    )
    fragments = split_conformer_into_fragments(relaxed_conf)
    if fragments is None or len(fragments) != 2:
        logger.debug(
            f"IRC merge check: expected 2 fragments, got "
            f"{len(fragments) if fragments else 0}"
        )
        return False, 9999.0

    smiles_1, smiles_2 = ctx.merge_component_smiles
    compound_1 = graph.get_compound(smiles_1)
    compound_2 = graph.get_compound(smiles_2)

    if compound_1 is None or compound_2 is None:
        logger.warning(
            f"Merge endpoint check skipped: compound not found "
            f"({smiles_1}={'found' if compound_1 else 'missing'}, "
            f"{smiles_2}={'found' if compound_2 else 'missing'})"
        )
        return False, 9999.0
    comp1_minima = np.array(
        [m.positions for m in compound_1.pes_graph.minima.values()]
    )
    comp1_atomic_numbers = np.array(compound_1.sorted_atomic_numbers)
    comp2_minima = np.array(
        [m.positions for m in compound_2.pes_graph.minima.values()]
    )
    comp2_atomic_numbers = np.array(compound_2.sorted_atomic_numbers)

    # Try both assignments: (frag0->comp1, frag1->comp2) and vice versa
    best_total_rmsd = 9999.0
    best_assignment = None
    for frag_a, frag_b in [
        (fragments[0], fragments[1]),
        (fragments[1], fragments[0]),
    ]:
        frag_a_anum = np.sort(frag_a.to_torch().atomic_numbers.cpu().numpy())
        frag_b_anum = np.sort(frag_b.to_torch().atomic_numbers.cpu().numpy())

        if len(frag_a_anum) != len(comp1_atomic_numbers) or len(
            frag_b_anum
        ) != len(comp2_atomic_numbers):
            logger.debug(
                f"IRC merge check: fragment atom count does not match component - "
                f"frag_a={len(frag_a_anum)} vs comp1={len(comp1_atomic_numbers)}, "
                f"frag_b={len(frag_b_anum)} vs comp2={len(comp2_atomic_numbers)}"
            )
            continue

        if not np.array_equal(
            frag_a_anum, comp1_atomic_numbers
        ) or not np.array_equal(frag_b_anum, comp2_atomic_numbers):
            logger.debug(
                f"IRC merge check: fragment atomic numbers do not match component - "
                f"frag_a={frag_a_anum} vs comp1={comp1_atomic_numbers}, "
                f"frag_b={frag_b_anum} vs comp2={comp2_atomic_numbers}"
            )
            continue

        rmsd_a = min_rmsd_conformer_to_compound(
            frag_a, comp1_minima, comp1_atomic_numbers
        )
        rmsd_b = min_rmsd_conformer_to_compound(
            frag_b, comp2_minima, comp2_atomic_numbers
        )

        total_rmsd = max(rmsd_a, rmsd_b)  # Use max to ensure both match well
        if total_rmsd < best_total_rmsd:
            best_total_rmsd = total_rmsd
            best_assignment = (rmsd_a, rmsd_b, smiles_1, smiles_2)

    is_match = best_total_rmsd < rmsd_threshold
    if best_assignment is not None:
        rmsd_a, rmsd_b, s1, s2 = best_assignment
        logger.debug(
            f"IRC merge check: {s1} rmsd={rmsd_a:.4f}, {s2} rmsd={rmsd_b:.4f}, "
            f"max_rmsd={best_total_rmsd:.4f}, match={is_match}"
        )
    else:
        logger.debug(
            f"IRC merge check: no valid fragment assignment found for "
            f"{smiles_1} + {smiles_2}"
        )

    return is_match, best_total_rmsd


def _validate_endpoints(
    forward_relaxed: torch.Tensor,
    backward_relaxed: torch.Tensor,
    atomic_numbers: np.ndarray,
    charge: int,
    ctx: ExplorationContext,
    graph: ReactionGraph,
    rmsd_threshold: float,
    irc_stats: Optional[dict] = None,
) -> bool:
    """
    Validate that the IRC found a reaction connected to the known compound.

    Checks that exactly one endpoint matches the known reactant (via RMSD).
    If both match -> no new product. If neither matches -> unrelated TS.
    Direction assignment is NOT done here (always energy-based).
    """
    fwd_match, fwd_rmsd = _check_matches_reactant(
        forward_relaxed, atomic_numbers, ctx, graph, rmsd_threshold
    )
    bwd_match, bwd_rmsd = _check_matches_reactant(
        backward_relaxed, atomic_numbers, ctx, graph, rmsd_threshold
    )
    logger.debug(
        f"IRC endpoint validation: "
        f"forward_match={fwd_match} (rmsd={fwd_rmsd:.4f}), "
        f"backward_match={bwd_match} (rmsd={bwd_rmsd:.4f})"
    )

    was_merged = bool(ctx.was_merged)

    if not fwd_match and not bwd_match:
        logger.debug("IRC: Neither endpoint matches known reactant")
        _bump_irc(irc_stats, "irc_endpoints_neither_match", was_merged)
        return False

    if fwd_match and bwd_match:
        logger.debug("IRC: Both endpoints match known reactant - no new product")
        _bump_irc(irc_stats, "irc_endpoints_both_match", was_merged)
        return False

    _bump_irc(irc_stats, "irc_endpoints_one_match", was_merged)
    return True


def _validate_chemical_difference(
    forward_relaxed: torch.Tensor,
    backward_relaxed: torch.Tensor,
    atomic_numbers: np.ndarray,
    charge: int,
    was_merged: bool = False,
    irc_stats: Optional[dict] = None,
) -> bool:
    """
    Greedy validation: check that IRC endpoints are chemically different.

    Fragments both sides, compares canonical SMILES. Rejects spectator
    reactions (A+B -> A+C where A is unchanged) and >2 fragments.

    Rejection reasons are tallied in irc_stats (split by merge/single) so
    we can see where the greedy validator is losing contexts:
      greedy_reject_frag_invalid       — fragmentation returned None
      greedy_reject_too_many_fragments — 3+ fragments on either side
      greedy_reject_smiles_fail        — SMILES determination failed
      greedy_reject_same_signature     — both sides have same canonical SMILES
      greedy_reject_spectator          — A+B → A+C with shared species A
    """
    atomic_numbers_tensor = torch.tensor(atomic_numbers, dtype=torch.long)
    fwd_conf = Conformer(positions=forward_relaxed, atomic_numbers=atomic_numbers_tensor, charge=charge)
    bwd_conf = Conformer(positions=backward_relaxed, atomic_numbers=atomic_numbers_tensor, charge=charge)

    fwd_frags = split_conformer_into_fragments(fwd_conf)
    bwd_frags = split_conformer_into_fragments(bwd_conf)

    if fwd_frags is None or bwd_frags is None:
        logger.debug("IRC greedy: fragmentation invalid on one or both sides")
        _bump_irc(irc_stats, "greedy_reject_frag_invalid", was_merged)
        return False

    # Reject 3+ fragments on either side (likely artifact, not a real elementary reaction)
    if len(fwd_frags) > 2 or len(bwd_frags) > 2:
        logger.debug(
            f"IRC greedy: too many fragments (fwd={len(fwd_frags)}, bwd={len(bwd_frags)})"
        )
        _bump_irc(irc_stats, "greedy_reject_too_many_fragments", was_merged)
        return False

    fwd_smiles_list = [get_smiles_from_conformer(f) for f in fwd_frags]
    bwd_smiles_list = [get_smiles_from_conformer(f) for f in bwd_frags]

    if any(s is None for s in fwd_smiles_list) or any(s is None for s in bwd_smiles_list):
        logger.debug("IRC greedy: SMILES determination failed for one or more fragments")
        _bump_irc(irc_stats, "greedy_reject_smiles_fail", was_merged)
        return False

    fwd_smiles = tuple(sorted(fwd_smiles_list))
    bwd_smiles = tuple(sorted(bwd_smiles_list))

    if fwd_smiles == bwd_smiles:
        logger.debug("IRC greedy: both sides have same SMILES signature, no reaction")
        _bump_irc(irc_stats, "greedy_reject_same_signature", was_merged)
        return False

    # Reject spectator reactions: A + B -> A + C where A is unchanged
    # (intramolecular B -> C with A as bystander). Use Counter (multiset)
    # comparison so that 2A -> A + B is NOT rejected (stoichiometry changed).
    if len(fwd_smiles_list) == 2 and len(bwd_smiles_list) == 2:
        from collections import Counter as _Counter
        fwd_counts = _Counter(fwd_smiles_list)
        bwd_counts = _Counter(bwd_smiles_list)
        common_smiles = set(fwd_smiles_list) & set(bwd_smiles_list)
        # True spectator: at least one species has identical count on both sides
        # AND the other species changed. This means one molecule just sat there.
        if common_smiles and any(fwd_counts[s] == bwd_counts[s] for s in common_smiles):
            # But only reject if a species is truly unchanged (same count both sides)
            unchanged = {s for s in common_smiles if fwd_counts[s] == bwd_counts[s]}
            if unchanged and fwd_counts != bwd_counts:
                logger.debug(
                    f"IRC greedy: spectator species {unchanged} unchanged on both sides, "
                    f"rejecting as pseudo-bimolecular"
                )
                _bump_irc(irc_stats, "greedy_reject_spectator", was_merged)
                return False

    logger.debug(
        f"IRC greedy: chemical validation passed "
        f"({' + '.join(fwd_smiles)} vs {' + '.join(bwd_smiles)})"
    )
    return True


def verify_ts_and_get_product(
    ctx: ExplorationContext,
    forces_model_calculator: Calculator,
    graph: ReactionGraph,
    energy_model: nn.Module,
    eigenvalue_threshold: float = -0.01,
    rmsd_threshold: float = 1.0,
    greedy: bool = False,
    relaxation_fmax: float = 0.005,
    irc_stats: Optional[dict] = None,
) -> tuple[
    bool,
    Optional[Conformer],
    Optional[Conformer],
    Optional[RelaxationTrajectory],
    Optional[RelaxationTrajectory],
]:
    """
    Verify TS via IRC and return product/reactant conformers with trajectories.

    Direction is always determined by energy: the higher-energy endpoint is
    the reactant, the lower-energy endpoint is the product. This gives a
    consistent view of reactions in the graph (forward = downhill in energy).

    Validation:
    - Normal: exactly one endpoint must match the known compound (RMSD)
    - Greedy fallback: endpoints must be chemically different (SMILES)

    Returns:
        (is_valid, product_conformer, reactant_conformer, reactant_trajectory, product_trajectory)
    """
    _fail = (False, None, None, None, None)

    if ctx.ts_conformer is None:
        return _fail

    ts_conf = ctx.ts_conformer.to_torch()
    positions = ts_conf.positions.cpu().numpy()
    atomic_numbers = ts_conf.atomic_numbers.cpu().numpy()
    charge = ctx.ts_conformer.charge

    was_merged = bool(ctx.was_merged)

    # 1. Validate TS character via Hessian
    hessian_result = _validate_ts_hessian(
        positions, atomic_numbers, charge, forces_model_calculator, eigenvalue_threshold
    )
    if hessian_result is None:
        _bump_irc(irc_stats, "irc_hessian_fail", was_merged)
        return _fail
    _, mode_3d = hessian_result
    _bump_irc(irc_stats, "irc_hessian_pass", was_merged)

    # 2. Displace along imaginary mode and relax both directions
    irc_result, fwd_ok, bwd_ok = _irc_displace_and_relax(
        positions, atomic_numbers, charge, mode_3d, forces_model_calculator,
        relaxation_fmax=relaxation_fmax,
    )
    if irc_result is None:
        if not fwd_ok:
            _bump_irc(irc_stats, "irc_relax_fwd_fail", was_merged)
        elif not bwd_ok:
            _bump_irc(irc_stats, "irc_relax_bwd_fail", was_merged)
        return _fail
    forward_relaxed, forward_trajectory, backward_relaxed, backward_trajectory = irc_result

    # 3. Compute energies and assign direction (lower energy = reactant)
    atomic_numbers_tensor = torch.tensor(atomic_numbers, dtype=torch.long)
    fwd_conf = Conformer(positions=forward_relaxed, atomic_numbers=atomic_numbers_tensor, charge=charge)
    bwd_conf = Conformer(positions=backward_relaxed, atomic_numbers=atomic_numbers_tensor, charge=charge)
    fwd_energy = get_conformer_energy(fwd_conf, energy_model)
    bwd_energy = get_conformer_energy(bwd_conf, energy_model)

    if fwd_energy >= bwd_energy:
        reactant_conf, product_conf = fwd_conf, bwd_conf
        reactant_traj, product_traj = forward_trajectory, backward_trajectory
    else:
        reactant_conf, product_conf = bwd_conf, fwd_conf
        reactant_traj, product_traj = backward_trajectory, forward_trajectory

    logger.debug(
        f"IRC energy-based direction: "
        f"fwd_energy={fwd_energy:.4f} eV, bwd_energy={bwd_energy:.4f} eV"
    )

    # 4. Validate the reaction
    is_valid = _validate_endpoints(
        forward_relaxed, backward_relaxed,
        atomic_numbers, charge, ctx, graph, rmsd_threshold,
        irc_stats=irc_stats,
    )

    if not is_valid:
        if not greedy:
            return _fail
        _bump_irc(irc_stats, "irc_greedy_fallback_used", was_merged)
        if not _validate_chemical_difference(
            forward_relaxed, backward_relaxed, atomic_numbers, charge,
            was_merged=was_merged, irc_stats=irc_stats,
        ):
            return _fail
        _bump_irc(irc_stats, "irc_greedy_accepted", was_merged)

    return True, product_conf, reactant_conf, reactant_traj, product_traj


def verify_ts_relaxations_irc(
    contexts: list[ExplorationContext],
    forces_model_calculator: Calculator,
    energy_model: nn.Module,
    device: str,
    graph: ReactionGraph,
    greedy_merge: bool = False,
    greedy_single: bool = False,
    relaxation_fmax: float = 0.005,
    irc_stats: Optional[dict] = None,
) -> list[ExplorationContext]:
    """
    Verify TS relaxations via IRC and populate product conformers.

    Direction is always determined by energy (higher = reactant). The context
    is rebuilt from the actual IRC endpoints so that the graph consistently
    represents reactions going downhill in energy.

    If greedy_merge/greedy_single is True, falls back to accepting any
    valid reaction when the normal reactant-matching validation fails.
    """
    logger.info("Verifying TS relaxations via IRC")
    valid_contexts = []

    for ctx in tqdm(contexts, desc="IRC verification"):
        forces_model_calculator.reset()

        greedy = greedy_merge if ctx.was_merged else greedy_single

        is_valid, product_conformer, reactant_conformer, reactant_trajectory, product_trajectory = (
            verify_ts_and_get_product(
                ctx=ctx,
                forces_model_calculator=forces_model_calculator,
                graph=graph,
                energy_model=energy_model,
                greedy=greedy,
                relaxation_fmax=relaxation_fmax,
                irc_stats=irc_stats,
            )
        )

        if not is_valid or product_conformer is None:
            continue

        # Rebuild context from energy-determined reactant
        _rebuild_context_from_irc(
            ctx, reactant_conformer, energy_model, device,
            calc=forces_model_calculator,
        )

        # Set the product conformer as min_conformer
        ctx.min_conformer = product_conformer
        ctx.min_id = ExplorationContext.compute_id(product_conformer)

        # Set IRC trajectories
        ctx.reactant_trajectory = reactant_trajectory
        ctx.product_trajectory = product_trajectory

        # Compute energy for product
        ctx.min_energy = get_conformer_energy(product_conformer, energy_model)

        valid_contexts.append(ctx)

    return valid_contexts


def _rebuild_context_from_irc(
    ctx: ExplorationContext,
    reactant_conformer: Conformer,
    energy_model: nn.Module,
    device: str,
    calc: Optional[Calculator] = None,
) -> None:
    """
    Rebuild an ExplorationContext from the energy-determined reactant conformer.

    The IRC-determined reactant (lower-energy endpoint) may differ from the
    original context's starting compound. This rebuilds the context fields
    (start_conformer/merge fields) to match the actual reaction direction.
    """
    # Fragment reactant side
    reactant_frags = split_conformer_into_fragments(reactant_conformer)
    if reactant_frags is None:
        reactant_frags = [reactant_conformer]

    if len(reactant_frags) == 1:
        # Single-molecule reactant
        frag = reactant_frags[0]
        energy = get_conformer_energy(frag, energy_model)
        frag.energy = energy

        ctx.was_merged = False
        ctx.start_conformer = frag
        ctx.start_id = ExplorationContext.compute_id(frag)
        ctx.start_energy = energy
        ctx.source_compound_smiles = get_smiles_from_conformer(frag)  # already validated
        # Clear merge fields
        ctx.merge_component_conformers = None
        ctx.merge_component_ids = None
        ctx.merge_component_energies = None
        ctx.merge_component_smiles = None
        ctx.merge_atom_indices = None
        ctx.merged_conformer = None
        ctx.merged_energy = None
    elif len(reactant_frags) == 2:
        # Two-fragment reactant → relax fragments then rebuild as merge context
        if calc is not None:
            relaxed = relax_fragments(reactant_frags, calc)
            if relaxed is not None:
                reactant_frags = relaxed
                frag1, frag2 = reactant_frags
            else:
                logger.debug("IRC greedy rebuild: fragment relaxation failed, using unrelaxed")
                frag1, frag2 = reactant_frags
        else:
            frag1, frag2 = reactant_frags
        e1 = frag1.energy if frag1.energy is not None else get_conformer_energy(frag1, energy_model)
        e2 = frag2.energy if frag2.energy is not None else get_conformer_energy(frag2, energy_model)
        frag1.energy = e1
        frag2.energy = e2
        id1 = ExplorationContext.compute_id(frag1)
        id2 = ExplorationContext.compute_id(frag2)
        smiles1 = get_smiles_from_conformer(frag1)
        smiles2 = get_smiles_from_conformer(frag2)

        ctx.was_merged = True
        ctx.merge_component_conformers = (frag1, frag2)
        ctx.merge_component_ids = (id1, id2)
        ctx.merge_component_energies = (e1, e2)
        ctx.merge_component_smiles = (smiles1, smiles2)
        ctx.merge_atom_indices = None  # Not meaningful for greedy-discovered reactions
        ctx.merged_conformer = reactant_conformer
        ctx.merged_energy = e1 + e2
        # Clear single-mol fields
        ctx.start_conformer = None
        ctx.start_id = None
        ctx.start_energy = None
        ctx.source_compound_smiles = None
    else:
        # 3+ fragments — shouldn't happen (filtered in verify_ts_and_get_product)
        logger.warning(f"IRC greedy: unexpected {len(reactant_frags)} reactant fragments")


def explore_and_process_batch(
    contexts: list[ExplorationContext],
    t: int,
    max_denoising_steps: int,
    ts_model: TSDenoiser,
    energy_model: nn.Module,
    forces_model_calculator: Calculator,
    graph: ReactionGraph,
    device: str,
    greedy_merge: bool = False,
    greedy_single: bool = False,
    relaxation_fmax: float = 0.005,
) -> dict:
    """
    Explore new reactions from ExplorationContexts.

    Pipeline:
    1. Diffuse starting structures to generate TS candidates
    2. Assign TS energies and filter by forward barrier
    3. Verify TS via IRC (returns both endpoints)
    4. Check for fragmentation in products
    5. Validate full energy profile
    6. Add valid contexts to graph

    Returns:
        Batch summary dict with pipeline funnel counts, IRC breakdown,
        context-type counts, noise level, and wall-clock time.
    """
    t_start = time.perf_counter()
    n_submitted = len(contexts)
    n_merge = sum(1 for ctx in contexts if ctx.was_merged)
    n_single = n_submitted - n_merge

    # Per-batch barrier histogram accumulator (populated by filter_contexts_by_barrier)
    barrier_bin_stats: dict = {}

    # Per-step timing (seconds); remains 0.0 if step is skipped via early exit
    gen_diffusion_s = 0.0
    gen_energy_s = 0.0
    gen_irc_s = 0.0
    gen_fragmentation_s = 0.0
    gen_graph_add_s = 0.0

    # IRC breakdown accumulator (passed through to verify_ts_and_get_product)
    irc_stats: dict = {
        "irc_hessian_pass": 0,
        "irc_hessian_fail": 0,
        "irc_relax_fwd_fail": 0,
        "irc_relax_bwd_fail": 0,
        "irc_endpoints_neither_match": 0,
        "irc_endpoints_both_match": 0,
        "irc_endpoints_one_match": 0,
        "irc_greedy_fallback_used": 0,
        "irc_greedy_accepted": 0,
    }

    def _make_summary(**overrides) -> dict:
        summary = {
            "noise_level": t,
            "n_contexts": n_submitted,
            "merge_submitted": n_merge,
            "single_submitted": n_single,
            "pipeline_contexts_submitted": n_submitted,
            "pipeline_survived_denoising": 0,
            "pipeline_passed_fwd_barrier": 0,
            "pipeline_passed_irc": 0,
            "pipeline_passed_fragmentation": 0,
            "pipeline_passed_bwd_barrier": 0,
            "pipeline_added_to_graph": 0,
            "merge_valid": 0,
            "single_valid": 0,
            # Per-type stage counters — how many merge/single contexts survive each step
            "merge_passed_fwd_barrier": 0,
            "single_passed_fwd_barrier": 0,
            "merge_passed_irc": 0,
            "single_passed_irc": 0,
            "merge_passed_fragmentation": 0,
            "single_passed_fragmentation": 0,
            "merge_passed_bwd_barrier": 0,
            "single_passed_bwd_barrier": 0,
            "wall_time_s": time.perf_counter() - t_start,
            "gen_diffusion_s": gen_diffusion_s,
            "gen_energy_s": gen_energy_s,
            "gen_irc_s": gen_irc_s,
            "gen_fragmentation_s": gen_fragmentation_s,
            "gen_graph_add_s": gen_graph_add_s,
            "timestamp": time.time(),
            **irc_stats,
            **barrier_bin_stats,
            **overrides,
        }
        return summary

    def _ms_counts(ctxs) -> tuple[int, int]:
        """(n_merge, n_single) for a context list."""
        m = sum(1 for c in ctxs if c.was_merged)
        return m, len(ctxs) - m

    if not contexts:
        return _make_summary()

    ######### STEP 0: ENHANCE MERGE GEOMETRIES (P1 + P2) #########
    # For each merge context: try K extra random orientations, pick the
    # lowest-ML-energy merged complex, optionally FIRE-relax, drop deep
    # encounter complexes, apply a small random displacement so the
    # diffuser has a direction to move in.
    # Env knobs:
    #   MERGE_PREP_N_ROTATIONS       — extra random orientations (0 disables P2)
    #   MERGE_PREP_RELAX_STEPS       — FIRE steps after best-of-K (0 disables P1)
    #   MERGE_PREP_DISPLACEMENT_SIGMA — Å per-atom-per-axis Gaussian after relax
    #   MERGE_SKIP_BINDING_EV        — if binding > this, drop context
    n_rot = int(os.environ.get("MERGE_PREP_N_ROTATIONS", "5"))
    relax_n = int(os.environ.get("MERGE_PREP_RELAX_STEPS", "5"))
    disp_sigma = float(os.environ.get("MERGE_PREP_DISPLACEMENT_SIGMA", "0.15"))
    max_binding = float(os.environ.get("MERGE_SKIP_BINDING_EV", "0.3"))
    if any(c.was_merged for c in contexts) and (n_rot > 0 or relax_n > 0 or disp_sigma > 0.0):
        from lib.merge_mols import enhance_merge_contexts
        energy_fn = create_energy_fn(energy_model)
        contexts = enhance_merge_contexts(
            contexts, energy_fn, forces_model_calculator,
            n_rotations=n_rot, relax_steps=relax_n,
            displacement_sigma=disp_sigma,
            max_binding_energy_eV=max_binding,
        )

    # Guard: enhance_merge_contexts may have dropped every merge (deep
    # encounter complexes) and the batch could now be empty or single-only.
    # An empty list would crash collate_mol_batch inside diffuse_new_ts.
    if not contexts:
        return _make_summary()

    ######### STEP 1: DIFFUSE TO GENERATE TS CANDIDATES #########
    # Merge contexts use a boosted noise level (larger denoiser excursion)
    # to help escape encounter-complex basins. Singles use the original t.
    merge_noise_boost = int(os.environ.get("MERGE_NOISE_BOOST_T", "0"))

    t_step = time.perf_counter()
    if merge_noise_boost > 0 and any(c.was_merged for c in contexts):
        # Split into merge / single sub-batches, diffuse at different t
        merge_idxs = [i for i, c in enumerate(contexts) if c.was_merged]
        single_idxs = [i for i, c in enumerate(contexts) if not c.was_merged]
        updated_mols = [None] * len(contexts)

        t_merge = t + merge_noise_boost

        if single_idxs:
            single_mols = []
            for i in single_idxs:
                mol = contexts[i].molecule_for_ml("start")
                single_mols.append({k: v.to(device) for k, v in mol.items()})
            single_result = diffuse_new_ts(
                single_mols, ts_model, t, max_denoising_steps, device,
            )
            if single_result is not None:
                for j, idx in enumerate(single_idxs):
                    updated_mols[idx] = single_result[j]

        if merge_idxs:
            merge_mols = []
            for i in merge_idxs:
                mol = contexts[i].molecule_for_ml("start")
                merge_mols.append({k: v.to(device) for k, v in mol.items()})
            merge_result = diffuse_new_ts(
                merge_mols, ts_model, t_merge, max_denoising_steps, device,
            )
            if merge_result is not None:
                for j, idx in enumerate(merge_idxs):
                    updated_mols[idx] = merge_result[j]

        # Drop contexts whose diffusion failed (None)
        paired = [(ctx, mol) for ctx, mol in zip(contexts, updated_mols) if mol is not None]
        if not paired:
            gen_diffusion_s = time.perf_counter() - t_step
            return _make_summary()
        contexts, updated_mols = zip(*paired)
        contexts = list(contexts)
        updated_mols = list(updated_mols)
        logger.info(
            f"Diffusion: t_single={t}, t_merge={t_merge} "
            f"({len(single_idxs)} single, {len(merge_idxs)} merge, "
            f"{len(contexts)} survived)"
        )
    else:
        # Standard path: all contexts at same noise level
        mols = []
        for ctx in contexts:
            mol = ctx.molecule_for_ml("start")
            mols.append({k: v.to(device) for k, v in mol.items()})
        updated_mols = diffuse_new_ts(
            mols=mols,
            ts_model=ts_model,
            t=t,
            max_denoising_steps=max_denoising_steps,
            device=device,
        )
        if updated_mols is None:
            gen_diffusion_s = time.perf_counter() - t_step
            return _make_summary()

    gen_diffusion_s = time.perf_counter() - t_step

    ######### STEP 2: SET TS CONFORMERS AND ASSIGN ENERGIES #########
    t_step = time.perf_counter()
    for ctx, mol in zip(contexts, updated_mols):
        ts_conformer = Conformer.from_batch(mol)
        ctx.ts_conformer = ts_conformer
        ctx.ts_id = ExplorationContext.compute_id(ts_conformer)
        ctx.ts_energy = get_conformer_energy(ts_conformer, energy_model)
        ctx.discovery_method = "generative"
        ctx.discovery_noise_level = t if not ctx.was_merged else t + merge_noise_boost
        ctx.discovery_timestamp = time.time()

    # Filter duplicates by TS ID
    seen_ts_ids = set()
    unique_contexts = []
    for ctx in contexts:
        if ctx.ts_id not in seen_ts_ids:
            seen_ts_ids.add(ctx.ts_id)
            unique_contexts.append(ctx)
    contexts = unique_contexts
    n_survived_denoising = len(contexts)

    # Filter by forward barrier (MIN -> TS)
    contexts = filter_contexts_by_barrier(
        contexts, graph.energy_validation_stats, "forward", graph.energy_threshold,
        bin_stats=barrier_bin_stats,
    )
    n_passed_fwd = len(contexts)
    n_merge_fwd, n_single_fwd = _ms_counts(contexts)
    gen_energy_s = time.perf_counter() - t_step

    if not contexts:
        return _make_summary(
            pipeline_survived_denoising=n_survived_denoising,
            pipeline_passed_fwd_barrier=n_passed_fwd,
            merge_passed_fwd_barrier=n_merge_fwd,
            single_passed_fwd_barrier=n_single_fwd,
        )

    ######### STEP 3: VERIFY TS VIA IRC (RETURNS BOTH ENDPOINTS) #########
    t_step = time.perf_counter()
    contexts = verify_ts_relaxations_irc(
        contexts=contexts,
        forces_model_calculator=forces_model_calculator,
        energy_model=energy_model,
        device=device,
        graph=graph,
        greedy_merge=greedy_merge,
        greedy_single=greedy_single,
        relaxation_fmax=relaxation_fmax,
        irc_stats=irc_stats,
    )
    gen_irc_s = time.perf_counter() - t_step
    n_passed_irc = len(contexts)
    n_merge_irc, n_single_irc = _ms_counts(contexts)

    if not contexts:
        return _make_summary(
            pipeline_survived_denoising=n_survived_denoising,
            pipeline_passed_fwd_barrier=n_passed_fwd,
            merge_passed_fwd_barrier=n_merge_fwd,
            single_passed_fwd_barrier=n_single_fwd,
            pipeline_passed_irc=n_passed_irc,
            merge_passed_irc=n_merge_irc,
            single_passed_irc=n_single_irc,
        )

    ######### STEP 4: CHECK FRAGMENTATION #########
    t_step = time.perf_counter()
    energy_fn = create_energy_fn(energy_model)
    valid_contexts = []
    for ctx in contexts:
        if ctx.min_conformer is None:
            continue
        success = populate_context_fragments(
            ctx=ctx,
            energy_fn=energy_fn,
            calc=forces_model_calculator,
            stats_tracker=graph.decomposition_stats,
        )
        if success:
            valid_contexts.append(ctx)
    contexts = valid_contexts
    gen_fragmentation_s = time.perf_counter() - t_step
    n_passed_frag = len(contexts)
    n_merge_frag, n_single_frag = _ms_counts(contexts)

    if not contexts:
        return _make_summary(
            pipeline_survived_denoising=n_survived_denoising,
            pipeline_passed_fwd_barrier=n_passed_fwd,
            merge_passed_fwd_barrier=n_merge_fwd,
            single_passed_fwd_barrier=n_single_fwd,
            pipeline_passed_irc=n_passed_irc,
            merge_passed_irc=n_merge_irc,
            single_passed_irc=n_single_irc,
            pipeline_passed_fragmentation=n_passed_frag,
            merge_passed_fragmentation=n_merge_frag,
            single_passed_fragmentation=n_single_frag,
        )

    ######### STEP 5: VALIDATE FULL ENERGY PROFILE #########
    t_step = time.perf_counter()
    contexts = filter_contexts_by_barrier(
        contexts, graph.energy_validation_stats, "backward", graph.energy_threshold,
        bin_stats=barrier_bin_stats,
    )
    n_passed_bwd = len(contexts)
    n_merge_bwd, n_single_bwd = _ms_counts(contexts)

    ######### STEP 6: ADD TO GRAPH #########
    n_added = 0
    if contexts:
        n_added = graph.add_contexts_to_graph(contexts, calc=forces_model_calculator)
    gen_graph_add_s = time.perf_counter() - t_step

    # Count merge/single among added contexts. `contexts` still holds all
    # post-bwd-barrier contexts; n_added may be smaller if some were rejected
    # at graph-add. Approximate per-type "valid" by the share of survivors.
    n_merge_valid = sum(1 for ctx in contexts if ctx.was_merged)
    n_single_valid = max(0, n_added - n_merge_valid)

    return _make_summary(
        pipeline_survived_denoising=n_survived_denoising,
        pipeline_passed_fwd_barrier=n_passed_fwd,
        merge_passed_fwd_barrier=n_merge_fwd,
        single_passed_fwd_barrier=n_single_fwd,
        pipeline_passed_irc=n_passed_irc,
        merge_passed_irc=n_merge_irc,
        single_passed_irc=n_single_irc,
        pipeline_passed_fragmentation=n_passed_frag,
        merge_passed_fragmentation=n_merge_frag,
        single_passed_fragmentation=n_single_frag,
        pipeline_passed_bwd_barrier=n_passed_bwd,
        merge_passed_bwd_barrier=n_merge_bwd,
        single_passed_bwd_barrier=n_single_bwd,
        pipeline_added_to_graph=n_added,
        merge_valid=n_merge_valid,
        single_valid=n_single_valid,
    )
