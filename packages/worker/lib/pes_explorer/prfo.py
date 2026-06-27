"""
Partitioned Rational Function Optimization (P-RFO) for transition state search.

Implements saddle point optimization by:
1. Computing the Hessian and projecting out translation/rotation
2. Following the lowest eigenvalue mode (maximizing) while minimizing along others

This module works with any ASE Calculator that implements a `get_hessian(atoms)` method.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from loguru import logger


@runtime_checkable
class HessianCalculator(Protocol):
    """Protocol for calculators that can compute Hessians."""

    def get_hessian(self, atoms: Atoms) -> np.ndarray:
        """
        Compute the Hessian matrix for the given atoms.

        Args:
            atoms: ASE Atoms object

        Returns:
            Hessian matrix in eV/Å², shape (3*n_atoms, 3*n_atoms)
        """
        ...


@runtime_checkable
class BatchedHessianCalculator(Protocol):
    """Protocol for calculators that support batched Hessian and energy/force computation."""

    def get_batched_hessians(
        self, atoms_list: list[Atoms]
    ) -> list[tuple[np.ndarray, float, np.ndarray]]:
        """Compute forces, energy, and Hessian for a batch of Atoms."""
        ...

    def get_batched_forces_and_energy(
        self, atoms_list: list[Atoms]
    ) -> list[tuple[np.ndarray, float]]:
        """Compute forces and energy for a batch of Atoms."""
        ...


@dataclass
class _SaddleSearchState:
    """Mutable state for one saddle-point search within a batched run."""

    atoms: Atoms
    index: int  # Original index in input list
    step: int = 0
    converged: bool = False
    early_stopped: bool = False
    is_first_order: bool = False
    energy: float | None = None
    force_max: float = float("inf")
    force_rms: float = float("inf")
    current_min_trust: float = 0.0
    current_ts_trust: float = 0.0
    previous_mode: np.ndarray | None = None
    position_history: list[np.ndarray] = field(default_factory=list)
    step_norms: list[float] = field(default_factory=list)
    force_history: list[float] = field(default_factory=list)
    energy_history: list[float] = field(default_factory=list)
    consecutive_bad_steps: int = 0


def get_translation_rotation_vectors(
    positions: np.ndarray, masses: np.ndarray
) -> np.ndarray:
    """
    Build orthonormal basis for translation and rotation modes.

    Args:
        positions: Atomic positions, shape (n_atoms, 3)
        masses: Atomic masses, shape (n_atoms,)

    Returns:
        Array of orthonormal vectors spanning translation/rotation space
    """
    n_atoms = len(masses)
    sqrt_m = np.sqrt(masses)
    total_mass = np.sqrt(np.sum(masses))
    com = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
    r = positions - com

    vectors = []

    # Translation (3 modes)
    for i in range(3):
        v = np.zeros((n_atoms, 3))
        v[:, i] = sqrt_m
        vectors.append(v.flatten() / total_mass)

    # Rotation (3 modes)
    for axis in range(3):
        v = np.zeros((n_atoms, 3))
        if axis == 0:
            v[:, 1] = r[:, 2] * sqrt_m
            v[:, 2] = -r[:, 1] * sqrt_m
        elif axis == 1:
            v[:, 0] = -r[:, 2] * sqrt_m
            v[:, 2] = r[:, 0] * sqrt_m
        else:
            v[:, 0] = r[:, 1] * sqrt_m
            v[:, 1] = -r[:, 0] * sqrt_m
        v = v.flatten()
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            vectors.append(v / norm)

    # Gram-Schmidt orthonormalization
    P = []
    for v in vectors:
        for u in P:
            v = v - np.dot(v, u) * u
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            P.append(v / norm)

    return np.array(P)


def project_hessian(
    H: np.ndarray, positions: np.ndarray, masses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project out translation/rotation from Hessian.

    Args:
        H: Hessian matrix, shape (3*n_atoms, 3*n_atoms)
        positions: Atomic positions, shape (n_atoms, 3)
        masses: Atomic masses, shape (n_atoms,)

    Returns:
        (H_projected, P_matrix): Projected Hessian and projection matrix
    """
    P_vecs = get_translation_rotation_vectors(positions, masses)
    n_dof = H.shape[0]
    P_matrix = np.eye(n_dof)
    for v in P_vecs:
        P_matrix -= np.outer(v, v)
    return P_matrix @ H @ P_matrix, P_matrix


def get_forces_energy_and_hessian(
    calc: Calculator, atoms: Atoms
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Get forces, energy, and Hessian from calculator for given atoms.

    Args:
        calc: ASE Calculator with get_hessian method
        atoms: ASE Atoms object

    Returns:
        (forces, energy, hessian): Forces in eV/Å, energy in eV, Hessian in eV/Å²
    """
    # Ensure calculator is attached to atoms
    atoms.calc = calc

    # Get energy and forces via standard ASE interface
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    # Get Hessian via calculator's get_hessian method
    if not hasattr(calc, "get_hessian"):
        raise ValueError(
            f"Calculator {type(calc).__name__} does not implement get_hessian method"
        )
    H = calc.get_hessian(atoms)

    return forces, energy, H


@dataclass
class PRFOStepResult:
    """Result of a single P-RFO step."""

    displacement: np.ndarray  # Shape: (n_atoms, 3)
    n_negative_eigs: int
    ts_mode_eigenvalue: float
    step_norm: float
    ts_mode_vector: np.ndarray | None = None  # For mode-following
    ts_step_norm: float = 0.0  # Step magnitude along TS mode
    min_step_norm: float = 0.0  # Step magnitude in minimization subspace


def _rfo_step_component(eigenvalue: float, gradient: float, maximize: bool) -> float:
    """
    Compute RFO step component using the augmented Hessian formulation.

    Solves the 2x2 augmented Hessian eigenvalue problem:
        [λ  g] [p]     [p]
        [g  0] [1] = μ [1]

    This gives μ = (λ ± sqrt(λ² + 4g²)) / 2, and step p = μ / g.

    Args:
        eigenvalue: Hessian eigenvalue λ for this mode
        gradient: Gradient component g for this mode
        maximize: If True, use positive root (uphill); if False, use negative root (downhill)

    Returns:
        Step component p for this mode
    """
    if abs(gradient) < 1e-12:
        return 0.0

    discriminant = eigenvalue**2 + 4 * gradient**2
    sqrt_disc = np.sqrt(discriminant)

    if maximize:
        # Positive root for maximization (move uphill)
        mu = (eigenvalue + sqrt_disc) / 2
    else:
        # Negative root for minimization (move downhill)
        mu = (eigenvalue - sqrt_disc) / 2

    return mu / gradient


def prfo_step(
    forces: np.ndarray,
    H: np.ndarray,
    positions: np.ndarray,
    masses: np.ndarray,
    trust_radius: float = 0.1,
    ts_trust_radius: float | None = None,
    min_trust_radius: float | None = None,
    follow_mode: int = 0,
    previous_mode: np.ndarray | None = None,
    verbose: bool = False,
    eigenvalue_zero_threshold: float = 0.0062,
) -> PRFOStepResult:
    """
    Compute a Partitioned Rational Function Optimization step.

    Maximizes along the lowest eigenvalue mode, minimizes along all others.
    Uses the augmented Hessian RFO formulation for numerically stable steps.

    Args:
        forces: Forces in eV/Å, shape (n_atoms, 3)
        H: Hessian in eV/Å², shape (3*n_atoms, 3*n_atoms)
        positions: Positions in Å, shape (n_atoms, 3)
        masses: Atomic masses, shape (n_atoms,)
        trust_radius: Maximum total step size in Å (used if partitioned radii not given)
        ts_trust_radius: Maximum step along TS mode (if None, uses trust_radius)
        min_trust_radius: Maximum step in minimization subspace (if None, uses trust_radius)
        follow_mode: Which mode to follow (0 = lowest)
        previous_mode: Previous TS mode eigenvector for mode-following
        verbose: Whether to print diagnostic information

    Returns:
        PRFOStepResult with displacement and diagnostics
    """
    # Default to uniform trust radius if partitioned radii not specified
    if ts_trust_radius is None:
        ts_trust_radius = trust_radius
    if min_trust_radius is None:
        min_trust_radius = trust_radius

    f_flat = forces.flatten()
    n_dof = len(f_flat)

    H_proj, P_matrix = project_hessian(H, positions, masses)
    f_proj = P_matrix @ f_flat

    eigenvalues, eigenvectors = np.linalg.eigh(H_proj)

    # Identify non-zero modes (exclude translation/rotation)
    zero_threshold = eigenvalue_zero_threshold
    nonzero_mask = np.abs(eigenvalues) > zero_threshold
    nonzero_indices = np.where(nonzero_mask)[0]
    sorted_indices = nonzero_indices[np.argsort(eigenvalues[nonzero_indices])]

    # Select mode to follow (maximize along)
    # Use mode-following: track mode with best overlap to previous TS mode
    if previous_mode is not None and len(sorted_indices) > 0:
        overlaps = eigenvectors[:, sorted_indices].T @ previous_mode
        best_overlap_idx = np.argmax(np.abs(overlaps))
        ts_mode_idx = sorted_indices[best_overlap_idx]
        # Ensure sign consistency for the eigenvector
        if overlaps[best_overlap_idx] < 0:
            eigenvectors[:, ts_mode_idx] *= -1
    elif len(sorted_indices) > follow_mode:
        ts_mode_idx = sorted_indices[follow_mode]
    else:
        ts_mode_idx = sorted_indices[0] if len(sorted_indices) > 0 else 0

    # Store the current TS mode for next iteration
    ts_mode_vector = eigenvectors[:, ts_mode_idx].copy()

    # Transform gradient to eigenvector basis (g = -f, gradient of energy)
    g_proj = eigenvectors.T @ (-f_proj)
    step_proj = np.zeros(n_dof)

    # Compute RFO step using augmented Hessian formulation
    for i in range(n_dof):
        if not nonzero_mask[i]:
            continue

        lam = eigenvalues[i]
        g_i = g_proj[i]
        maximize = i == ts_mode_idx

        step_proj[i] = _rfo_step_component(lam, g_i, maximize=maximize)

    # Apply partitioned trust radius constraints
    # 1. Constrain TS mode step
    ts_step_raw = abs(step_proj[ts_mode_idx])
    if ts_step_raw > ts_trust_radius:
        step_proj[ts_mode_idx] *= ts_trust_radius / ts_step_raw

    # 2. Constrain minimization subspace step
    min_mask = nonzero_mask.copy()
    min_mask[ts_mode_idx] = False
    min_step_proj = step_proj[min_mask]
    min_step_raw = np.linalg.norm(min_step_proj)
    if min_step_raw > min_trust_radius:
        step_proj[min_mask] *= min_trust_radius / min_step_raw

    # Transform back to Cartesian coordinates
    step_flat = eigenvectors @ step_proj

    # Record final step magnitudes
    ts_step_norm = abs(step_proj[ts_mode_idx])
    min_step_norm = np.linalg.norm(step_proj[min_mask]) if np.any(min_mask) else 0.0
    step_norm = np.linalg.norm(step_flat)

    n_negative = np.sum(eigenvalues[nonzero_mask] < -zero_threshold)
    n_positive = np.sum(eigenvalues[nonzero_mask] > zero_threshold)

    # if verbose:
    #     print(
    #         f"PRFO step: total={step_norm:.4f} Å, TS={ts_step_norm:.4f} Å, min={min_step_norm:.4f} Å | "
    #         f"Negative eigs: {n_negative}, Positive eigs: {n_positive}"
    #     )
    #     print(f"eigenvalues (non-zero): {eigenvalues[nonzero_mask]}")

    return PRFOStepResult(
        displacement=step_flat.reshape(-1, 3),
        n_negative_eigs=n_negative,
        ts_mode_eigenvalue=eigenvalues[ts_mode_idx],
        step_norm=step_norm,
        ts_mode_vector=ts_mode_vector,
        ts_step_norm=ts_step_norm,
        min_step_norm=min_step_norm,
    )


@dataclass
class SaddleOptResult:
    """Result of saddle point optimization."""

    converged: bool
    early_stopped: bool  # Gave up due to lack of progress (stuck/oscillating)
    is_first_order: bool  # Exactly one negative eigenvalue
    positions: np.ndarray
    energy: float | None
    n_steps: int
    final_force_max: float
    final_force_rms: float
    force_history: list[float]
    energy_history: list[float]


def _compute_predicted_energy_change(
    forces: np.ndarray, H: np.ndarray, step: np.ndarray
) -> float:
    """
    Compute predicted energy change from quadratic model.

    ΔE_pred = -f·p + 0.5·p·H·p

    where f is force and p is displacement step.
    """
    f_flat = forces.flatten()
    step_flat = step.flatten()
    return -np.dot(f_flat, step_flat) + 0.5 * step_flat @ H @ step_flat


def optimize_saddle_point(
    calc: Calculator,
    atoms: Atoms,
    max_steps: int = 150,
    force_max_tol: float = 0.03,
    force_rms_tol: float = 0.01,
    trust_radius: float = 0.1,
    min_trust_radius: float = 0.01,
    max_trust_radius: float = 0.3,
    ts_trust_scale: float = 1.5,
    adaptive_trust: bool = True,
    mode_following: bool = True,
    max_step_retries: int = 5,
    step_accept_threshold: float = 0.1,
    early_stop_window: int = 20,
    early_stop_min_displacement: float = 1e-4,
    early_stop_efficiency_threshold: float = 0.1,
    verbose: bool = False,
    eigenvalue_zero_threshold: float = 0.0062,
) -> SaddleOptResult:
    """
    Walk towards a saddle point using Partitioned-RFO.

    Args:
        calc: ASE Calculator with get_hessian method (e.g., MLCalculator, MaceCalculator)
        atoms: ASE Atoms object with starting geometry
        max_steps: Maximum optimization steps
        force_max_tol: Max force component convergence tolerance in eV/Å
        force_rms_tol: RMS force convergence tolerance in eV/Å
        trust_radius: Initial trust radius for minimization subspace in Å
        min_trust_radius: Minimum trust radius in Å (for adaptive trust)
        max_trust_radius: Maximum trust radius in Å (for adaptive trust)
        ts_trust_scale: Scale factor for TS mode trust radius relative to min trust radius
        adaptive_trust: Whether to adaptively adjust trust radius based on energy changes
        mode_following: Whether to track and follow the TS mode across steps
        max_step_retries: Maximum retries with smaller trust radius before accepting step
        step_accept_threshold: Minimum trust region ratio to accept step without retry
        early_stop_window: Number of steps to look back for progress check
        early_stop_min_displacement: Minimum net displacement over window (Å)
        early_stop_efficiency_threshold: Minimum ratio of net displacement to path length
        verbose: Whether to print diagnostic information

    Returns:
        SaddleOptResult with final positions and convergence info

    Example:
        >>> from ase import Atoms
        >>> from lib.md_et_calculator import get_md_et_calculator
        >>> calc = get_md_et_calculator(run_dir, device="cpu")
        >>> atoms = Atoms(...)  # your molecule
        >>> result = optimize_saddle_point(calc, atoms)
        >>> if result.converged and result.is_first_order:
        ...     print("Found transition state!")
    """
    # Work with a copy to avoid modifying the input
    atoms = atoms.copy()
    atoms.calc = calc

    masses = atoms.get_masses()

    converged = False
    is_first_order = False
    early_stopped = False
    force_history = []
    energy_history = []
    previous_mode = None

    # Early stopping tracking
    position_history: list[np.ndarray] = []
    step_norms: list[float] = []

    # Partitioned trust radii: allow more aggressive steps along TS mode
    current_min_trust = trust_radius
    current_ts_trust = trust_radius * ts_trust_scale

    # Track consecutive bad steps for early termination
    consecutive_bad_steps = 0
    max_consecutive_bad = 20  # Give up after this many bad steps at min trust

    step = 0
    while step < max_steps:
        forces, energy, H = get_forces_energy_and_hessian(calc, atoms)

        # Better convergence criteria: max force and RMS force
        force_max = np.max(np.abs(forces))
        force_rms = np.sqrt(np.mean(forces**2))

        force_history.append(force_rms)
        energy_history.append(energy)

        # Track position for early stopping
        position_history.append(atoms.positions.copy())

        # Check convergence
        if force_max < force_max_tol and force_rms < force_rms_tol:
            # Final Hessian check for TS character
            result = prfo_step(
                forces,
                H,
                atoms.positions,
                masses,
                trust_radius=current_min_trust,
                ts_trust_radius=current_ts_trust,
                min_trust_radius=current_min_trust,
                previous_mode=previous_mode if mode_following else None,
                verbose=False,
                eigenvalue_zero_threshold=eigenvalue_zero_threshold,
            )
            converged = True
            is_first_order = result.n_negative_eigs == 1
            if verbose:
                logger.debug(
                    f"Converged at step {step + 1}: force_max={force_max:.6f}, "
                    f"force_rms={force_rms:.6f} eV/Å, n_negative={result.n_negative_eigs}"
                )
            break

        # Early stopping: check if stuck or oscillating
        if (
            len(position_history) >= early_stop_window
            and len(step_norms) >= early_stop_window - 1
        ):
            start_pos = position_history[-early_stop_window]
            end_pos = position_history[-1]
            net_displacement = np.linalg.norm((end_pos - start_pos).flatten())
            path_length = sum(step_norms[-(early_stop_window - 1) :])
            efficiency = net_displacement / path_length if path_length > 1e-10 else 1.0

            is_stuck = (
                path_length < early_stop_min_displacement
                and net_displacement < early_stop_min_displacement
            )
            is_oscillating = (
                path_length > early_stop_min_displacement
                and efficiency < early_stop_efficiency_threshold
            )

            if is_stuck or is_oscillating:
                early_stopped = True
                reason = (
                    "stuck (not moving)"
                    if is_stuck
                    else f"oscillating (efficiency={efficiency:.3f})"
                )
                if verbose:
                    logger.debug(
                        f"Early stopping at step {step + 1}: {reason}. "
                        f"path_length={path_length:.2e} Å, net_displacement={net_displacement:.2e} Å"
                    )
                break

        # Step rejection loop: try steps with decreasing trust radius until acceptable
        trial_ts_trust = current_ts_trust
        trial_min_trust = current_min_trust
        accepted = False

        for retry in range(max_step_retries):
            # Compute trial step
            result = prfo_step(
                forces,
                H,
                atoms.positions,
                masses,
                trust_radius=trial_min_trust,
                ts_trust_radius=trial_ts_trust,
                min_trust_radius=trial_min_trust,
                previous_mode=previous_mode if mode_following else None,
                verbose=verbose and (retry == 0),
                eigenvalue_zero_threshold=eigenvalue_zero_threshold,
            )

            # Compute predicted energy change
            predicted_change = _compute_predicted_energy_change(
                forces, H, result.displacement
            )

            # Evaluate trial position
            trial_atoms = atoms.copy()
            trial_atoms.positions = atoms.positions + result.displacement
            trial_atoms.calc = calc
            trial_energy = trial_atoms.get_potential_energy()

            actual_change = trial_energy - energy

            # Compute trust region ratio
            if abs(predicted_change) > 1e-10:
                rho = actual_change / predicted_change
            else:
                rho = 1.0

            # Accept if ratio is reasonable (not too negative)
            # Note: ρ > 2 is accepted but will trigger trust reduction later
            if rho > step_accept_threshold:
                accepted = True
                if verbose:
                    quality = (
                        "good"
                        if 0.5 < rho < 1.5
                        else "acceptable"
                        if rho <= 2.0
                        else "poor (underestimate)"
                    )
                    if retry > 0:
                        logger.debug(
                            f"Step {step + 1}: Accepted after {retry + 1} tries (ρ={rho:.3f}, {quality})"
                        )
                    else:
                        logger.debug(f"Step {step + 1}: ρ={rho:.3f} ({quality})")
                break
            else:
                # Check if we're already at minimum - no point retrying
                already_at_min = trial_min_trust <= min_trust_radius * 1.01
                if already_at_min:
                    if verbose:
                        logger.debug(
                            f"Step {step + 1}: At minimum trust, accepting poor step (ρ={rho:.3f})"
                        )
                    break

                # Reject and reduce trust radius for retry
                trial_ts_trust *= 0.5
                trial_min_trust *= 0.5

                # Don't go below minimum
                if trial_min_trust < min_trust_radius:
                    trial_min_trust = min_trust_radius
                    trial_ts_trust = min_trust_radius * ts_trust_scale

                if verbose:
                    logger.debug(
                        f"Step {step + 1}: Retry {retry + 1} (ρ={rho:.3f}), "
                        f"trust → ts={trial_ts_trust:.4f}, min={trial_min_trust:.4f} Å"
                    )

        # If all retries failed, accept the last step anyway
        if not accepted:
            if verbose:
                logger.debug(
                    f"Step {step + 1}: Accepting step after {max_step_retries} retries (ρ={rho:.3f})"
                )

        # Track consecutive bad steps for early termination
        at_min_trust = current_min_trust <= min_trust_radius * 1.01
        is_bad_step = rho < step_accept_threshold or rho > 2.0

        if at_min_trust and is_bad_step:
            consecutive_bad_steps += 1
            if consecutive_bad_steps >= max_consecutive_bad:
                if verbose:
                    logger.debug(
                        f"Step {step + 1}: {consecutive_bad_steps} consecutive bad steps at "
                        f"min trust radius, giving up (n_negative={result.n_negative_eigs})"
                    )
                break
        else:
            consecutive_bad_steps = 0

        # Update trust radii based on final rho
        # Good model: ρ ≈ 1 (between 0.5 and 1.5)
        # Poor model: ρ < 0.25 OR ρ > 2.0 (model significantly wrong in either direction)
        if adaptive_trust:
            ts_hit_boundary = result.ts_step_norm >= trial_ts_trust * 0.99
            min_hit_boundary = result.min_step_norm >= trial_min_trust * 0.99

            if rho < 0.25 or rho > 2.0:
                # Poor model (underestimate or overestimate) - shrink trust radii
                current_min_trust = max(min_trust_radius, current_min_trust * 0.5)
                current_ts_trust = max(
                    min_trust_radius * ts_trust_scale, current_ts_trust * 0.5
                )
            elif 0.5 < rho < 1.5:
                # Good model (close to 1) - expand trust radii if constrained
                if min_hit_boundary:
                    current_min_trust = min(max_trust_radius, current_min_trust * 1.5)
                if ts_hit_boundary:
                    current_ts_trust = min(
                        max_trust_radius * ts_trust_scale, current_ts_trust * 1.5
                    )

        # Store mode for next iteration
        if mode_following and result.ts_mode_vector is not None:
            previous_mode = result.ts_mode_vector

        # Apply the accepted step
        atoms.positions = atoms.positions + result.displacement
        step_norms.append(result.step_norm)
        step += 1

    if not converged:
        if early_stopped:
            logger.debug(
                f"Early stopped at step {step}: "
                f"force_max={force_max:.6f}, force_rms={force_rms:.6f} eV/Å"
            )
        else:
            logger.debug(
                f"Did not converge within {max_steps} steps: "
                f"force_max={force_max:.6f}, force_rms={force_rms:.6f} eV/Å"
            )

    return SaddleOptResult(
        converged=converged,
        early_stopped=early_stopped,
        is_first_order=is_first_order,
        positions=atoms.positions.copy(),
        energy=energy if converged else None,
        n_steps=step,
        final_force_max=force_max,
        final_force_rms=force_rms,
        force_history=force_history,
        energy_history=energy_history,
    )


def optimize_saddle_points_batched(
    calc: Calculator,
    atoms_list: list[Atoms],
    max_steps: int = 150,
    force_max_tol: float = 0.03,
    force_rms_tol: float = 0.01,
    trust_radius: float = 0.1,
    min_trust_radius: float = 0.01,
    max_trust_radius: float = 0.3,
    ts_trust_scale: float = 1.5,
    mode_following: bool = True,
    step_accept_threshold: float = 0.1,
    early_stop_window: int = 20,
    early_stop_min_displacement: float = 1e-4,
    early_stop_efficiency_threshold: float = 0.1,
    max_consecutive_bad: int = 20,
    max_step_retries: int = 5,
    verbose: bool = False,
    eigenvalue_zero_threshold: float = 0.0062,
) -> list[SaddleOptResult]:
    """
    Run multiple saddle-point searches in lockstep, batching GPU calls.

    Each search maintains independent state (trust radii, mode-following vector,
    convergence status). Searches finish at different times; the batch shrinks
    as searches converge or early-stop.

    Falls back to sequential optimize_saddle_point() if the calculator does not
    support batched operations.

    Args:
        calc: ASE Calculator with get_batched_hessians / get_batched_forces_and_energy
        atoms_list: Starting geometries for each search
        max_steps: Maximum steps per search
        force_max_tol: Max force convergence tolerance (eV/Å)
        force_rms_tol: RMS force convergence tolerance (eV/Å)
        trust_radius: Initial trust radius (Å)
        min_trust_radius: Minimum trust radius (Å)
        max_trust_radius: Maximum trust radius (Å)
        ts_trust_scale: Scale factor for TS mode trust radius
        mode_following: Whether to track the TS mode across steps
        step_accept_threshold: Minimum trust region ratio to accept step
        early_stop_window: Steps to look back for progress check
        early_stop_min_displacement: Minimum net displacement over window (Å)
        early_stop_efficiency_threshold: Minimum net/path ratio
        max_consecutive_bad: Give up after this many bad steps at min trust
        max_step_retries: Max retries with smaller trust radius per step
        verbose: Print diagnostic information

    Returns:
        List of SaddleOptResult in the same order as atoms_list.
    """
    if not atoms_list:
        return []

    # Fallback: sequential if calculator doesn't support batched ops
    if not isinstance(calc, BatchedHessianCalculator):
        logger.debug(
            "Calculator does not support batched operations, falling back to sequential"
        )
        results = []
        for atoms in atoms_list:
            results.append(
                optimize_saddle_point(
                    calc,
                    atoms,
                    max_steps=max_steps,
                    force_max_tol=force_max_tol,
                    force_rms_tol=force_rms_tol,
                    trust_radius=trust_radius,
                    min_trust_radius=min_trust_radius,
                    max_trust_radius=max_trust_radius,
                    ts_trust_scale=ts_trust_scale,
                    mode_following=mode_following,
                    step_accept_threshold=step_accept_threshold,
                    early_stop_window=early_stop_window,
                    early_stop_min_displacement=early_stop_min_displacement,
                    early_stop_efficiency_threshold=early_stop_efficiency_threshold,
                    verbose=verbose,
                )
            )
        return results

    # Initialize per-search state
    states: list[_SaddleSearchState] = []
    for i, atoms in enumerate(atoms_list):
        a = atoms.copy()
        a.calc = calc
        states.append(
            _SaddleSearchState(
                atoms=a,
                index=i,
                current_min_trust=trust_radius,
                current_ts_trust=trust_radius * ts_trust_scale,
            )
        )

    for _global_step in range(max_steps):
        # Collect active (non-finished) searches
        active = [s for s in states if not s.converged and not s.early_stopped]
        if not active:
            break

        # --- Phase 1: Batched Hessian (GPU) ---
        active_atoms = [s.atoms for s in active]
        hessian_results = calc.get_batched_hessians(active_atoms)

        # Distribute results and store per-search forces/energy/hessian
        per_search_data: list[dict] = []
        for s, (forces, energy, H) in zip(active, hessian_results):
            forces = forces.reshape(-1, 3)
            s.energy = energy
            s.force_max = float(np.max(np.abs(forces)))
            s.force_rms = float(np.sqrt(np.mean(forces**2)))
            s.force_history.append(s.force_rms)
            s.energy_history.append(energy)
            per_search_data.append({"forces": forces, "energy": energy, "H": H})

        # --- Phase 2: Convergence + early stopping (CPU) ---
        still_active: list[tuple[_SaddleSearchState, dict]] = []
        for s, data in zip(active, per_search_data):
            # Check convergence
            if s.force_max < force_max_tol and s.force_rms < force_rms_tol:
                masses = s.atoms.get_masses()
                result = prfo_step(
                    data["forces"],
                    data["H"],
                    s.atoms.positions,
                    masses,
                    trust_radius=s.current_min_trust,
                    ts_trust_radius=s.current_ts_trust,
                    min_trust_radius=s.current_min_trust,
                    previous_mode=s.previous_mode if mode_following else None,
                    eigenvalue_zero_threshold=eigenvalue_zero_threshold,
                )
                s.converged = True
                s.is_first_order = result.n_negative_eigs == 1
                if verbose:
                    logger.debug(
                        f"Search {s.index}: converged at step {s.step + 1}, "
                        f"force_max={s.force_max:.6f}, n_negative={result.n_negative_eigs}"
                    )
                continue

            # Check early stopping
            if (
                len(s.position_history) >= early_stop_window
                and len(s.step_norms) >= early_stop_window - 1
            ):
                start_pos = s.position_history[-early_stop_window]
                end_pos = s.position_history[-1]
                net_displacement = np.linalg.norm((end_pos - start_pos).flatten())
                path_length = sum(s.step_norms[-(early_stop_window - 1) :])
                efficiency = (
                    net_displacement / path_length if path_length > 1e-10 else 1.0
                )

                is_stuck = (
                    path_length < early_stop_min_displacement
                    and net_displacement < early_stop_min_displacement
                )
                is_oscillating = (
                    path_length > early_stop_min_displacement
                    and efficiency < early_stop_efficiency_threshold
                )

                if is_stuck or is_oscillating:
                    s.early_stopped = True
                    if verbose:
                        reason = (
                            "stuck"
                            if is_stuck
                            else f"oscillating (eff={efficiency:.3f})"
                        )
                        logger.debug(
                            f"Search {s.index}: early stop at step {s.step + 1}: {reason}"
                        )
                    continue

            # Check max consecutive bad steps
            if s.consecutive_bad_steps >= max_consecutive_bad:
                s.early_stopped = True
                if verbose:
                    logger.debug(
                        f"Search {s.index}: {s.consecutive_bad_steps} consecutive bad steps, giving up"
                    )
                continue

            # Check per-search step limit
            if s.step >= max_steps:
                if verbose:
                    logger.debug(f"Search {s.index}: reached max steps ({max_steps})")
                continue

            still_active.append((s, data))

        if not still_active:
            continue

        # --- Phases 3-5: PRFO step + trial energy + accept/reject with retries ---
        # Matches sequential retry loop: try progressively smaller trust radii
        # until accepted or at minimum trust. Only rejected searches retry;
        # the GPU batch shrinks each round.

        # Initialize per-search retry state (all searches start pending)
        all_searches: list[dict] = []
        for s, data in still_active:
            all_searches.append({
                "state": s,
                "data": data,
                "trial_min_trust": s.current_min_trust,
                "trial_ts_trust": s.current_ts_trust,
                "result": None,
                "rho": None,
            })
        needs_retry = list(range(len(all_searches)))

        for _retry in range(max_step_retries):
            if not needs_retry:
                break

            # Phase 3: PRFO step (CPU)
            for idx in needs_retry:
                p = all_searches[idx]
                s = p["state"]
                masses = s.atoms.get_masses()
                result = prfo_step(
                    p["data"]["forces"],
                    p["data"]["H"],
                    s.atoms.positions,
                    masses,
                    trust_radius=p["trial_min_trust"],
                    ts_trust_radius=p["trial_ts_trust"],
                    min_trust_radius=p["trial_min_trust"],
                    previous_mode=s.previous_mode if mode_following else None,
                    eigenvalue_zero_threshold=eigenvalue_zero_threshold,
                )
                p["result"] = result
                p["predicted_change"] = _compute_predicted_energy_change(
                    p["data"]["forces"], p["data"]["H"], result.displacement
                )

            # Phase 4: Batched trial energy (GPU) — only for searches needing retry
            trial_atoms_list: list[Atoms] = []
            for idx in needs_retry:
                p = all_searches[idx]
                trial = p["state"].atoms.copy()
                trial.positions = p["state"].atoms.positions + p["result"].displacement
                trial.calc = calc
                trial_atoms_list.append(trial)

            trial_results = calc.get_batched_forces_and_energy(trial_atoms_list)

            # Phase 5: Accept/reject
            next_needs_retry = []
            for idx, (_, trial_energy) in zip(needs_retry, trial_results):
                p = all_searches[idx]
                actual_change = trial_energy - p["data"]["energy"]

                if abs(p["predicted_change"]) > 1e-10:
                    rho = actual_change / p["predicted_change"]
                else:
                    rho = 1.0
                p["rho"] = rho

                if rho > step_accept_threshold:
                    continue  # Accepted — done with this search

                # Rejected — at min trust means no point retrying
                if p["trial_min_trust"] <= min_trust_radius * 1.01:
                    continue

                # Shrink trial trust for next retry
                p["trial_ts_trust"] *= 0.5
                p["trial_min_trust"] *= 0.5
                if p["trial_min_trust"] < min_trust_radius:
                    p["trial_min_trust"] = min_trust_radius
                    p["trial_ts_trust"] = min_trust_radius * ts_trust_scale

                next_needs_retry.append(idx)

            needs_retry = next_needs_retry

        # --- Apply final step + update state for ALL searches ---
        for p in all_searches:
            s = p["state"]
            result = p["result"]
            rho = p["rho"]

            # Track consecutive bad steps (uses current trust, not trial)
            at_min_trust = s.current_min_trust <= min_trust_radius * 1.01
            is_bad_step = rho < step_accept_threshold or rho > 2.0

            if at_min_trust and is_bad_step:
                s.consecutive_bad_steps += 1
            else:
                s.consecutive_bad_steps = 0

            # Update trust radii based on final rho
            ts_hit_boundary = result.ts_step_norm >= p["trial_ts_trust"] * 0.99
            min_hit_boundary = result.min_step_norm >= p["trial_min_trust"] * 0.99

            if rho < 0.25 or rho > 2.0:
                s.current_min_trust = max(min_trust_radius, s.current_min_trust * 0.5)
                s.current_ts_trust = max(
                    min_trust_radius * ts_trust_scale, s.current_ts_trust * 0.5
                )
            elif 0.5 < rho < 1.5:
                if min_hit_boundary:
                    s.current_min_trust = min(
                        max_trust_radius, s.current_min_trust * 1.5
                    )
                if ts_hit_boundary:
                    s.current_ts_trust = min(
                        max_trust_radius * ts_trust_scale, s.current_ts_trust * 1.5
                    )

            # Always apply the step (accepted or last retry, matches sequential)
            s.atoms.positions = s.atoms.positions + result.displacement
            s.step_norms.append(result.step_norm)
            s.position_history.append(s.atoms.positions.copy())

            # Update mode-following vector
            if mode_following and result.ts_mode_vector is not None:
                s.previous_mode = result.ts_mode_vector

            s.step += 1

    # Assemble results in original input order
    results: list[SaddleOptResult | None] = [None] * len(atoms_list)
    for s in states:
        results[s.index] = SaddleOptResult(
            converged=s.converged,
            early_stopped=s.early_stopped,
            is_first_order=s.is_first_order,
            positions=s.atoms.positions.copy(),
            energy=s.energy if s.converged else None,
            n_steps=s.step,
            final_force_max=s.force_max,
            final_force_rms=s.force_rms,
            force_history=s.force_history,
            energy_history=s.energy_history,
        )
    return results  # type: ignore[return-value]


def compute_hessian_eigenvalues(
    calc: Calculator,
    atoms: Atoms,
    project_tr: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Hessian eigenvalues and eigenvectors for analysis.

    Args:
        calc: ASE Calculator with get_hessian method
        atoms: ASE Atoms object
        project_tr: If True, project out translation/rotation modes

    Returns:
        (eigenvalues, eigenvectors): Sorted by eigenvalue (lowest first)
    """
    atoms = atoms.copy()
    atoms.calc = calc

    H = calc.get_hessian(atoms)

    if project_tr:
        masses = atoms.get_masses()
        H, _ = project_hessian(H, atoms.positions, masses)

    eigenvalues, eigenvectors = np.linalg.eigh(H)

    return eigenvalues, eigenvectors


def is_transition_state(
    calc: Calculator,
    atoms: Atoms,
    eigenvalue_threshold: float = -1e-4,
) -> tuple[bool, int, np.ndarray]:
    """
    Check if the geometry is a first-order saddle point (transition state).

    Args:
        calc: ASE Calculator with get_hessian method
        atoms: ASE Atoms object
        eigenvalue_threshold: Threshold for counting negative eigenvalues

    Returns:
        (is_ts, n_negative, eigenvalues): Whether it's a TS, count of negative
            eigenvalues, and all eigenvalues
    """
    eigenvalues, _ = compute_hessian_eigenvalues(calc, atoms, project_tr=True)

    # Filter out near-zero eigenvalues (translation/rotation)
    significant_mask = np.abs(eigenvalues) > 1e-5
    significant_eigenvalues = eigenvalues[significant_mask]

    n_negative = np.sum(significant_eigenvalues < eigenvalue_threshold)
    is_ts = n_negative == 1

    return is_ts, n_negative, eigenvalues
