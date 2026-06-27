"""
Minimization utilities for PES exploration.

Provides two optimizers:
- optimize_fire: Fast FIRE-based minimizer (force-only, ~14x fewer model
  evaluations per step than Newton). Preferred for IRC relaxations.
- optimize_minimum: Trust-region Newton with exact Hessians and trajectory
  collection. Use when Hessians are needed during optimization.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from loguru import logger

from .prfo import (
    BatchedHessianCalculator,
    _rfo_step_component,
    _compute_predicted_energy_change,
    get_forces_energy_and_hessian,
    project_hessian,
)


@dataclass
class MinimizationTrajectory:
    """Trajectory data collected during Hessian-based minimization."""

    positions: list[np.ndarray]
    energies: list[float]
    forces: list[np.ndarray]
    hessians: list[Optional[np.ndarray]]  # None for steps where Hessian was reused


@dataclass
class MinimizationResult:
    """Result of Hessian-based minimization."""

    converged: bool
    early_stopped: bool  # Gave up due to lack of progress (not at stationary point)
    positions: np.ndarray
    energy: float
    n_steps: int
    final_force_max: float
    final_force_rms: float
    trajectory: MinimizationTrajectory


def optimize_fire(
    calc: Calculator,
    atoms: Atoms,
    max_steps: int = 200,
    force_max_tol: float = 0.01,
    force_rms_tol: float = 0.005,
    energy_increase_tol: float = 0.01,
    hessian_retrace_interval: int = 5,
    verbose: bool = False,
) -> MinimizationResult:
    """
    Fast minimization using the FIRE algorithm (force-only, no Hessian needed).

    Much faster than Newton for IRC relaxations: one force evaluation per
    accepted step vs ~14 for Newton. Steps that increase energy by more than
    energy_increase_tol are rejected. On successful convergence, the trajectory
    is retraced to compute Hessians at regular intervals.

    Args:
        calc: ASE Calculator
        atoms: ASE Atoms with starting geometry
        max_steps: Maximum optimization steps
        force_max_tol: Max force convergence criterion (eV/Å)
        force_rms_tol: RMS force convergence criterion (eV/Å)
        energy_increase_tol: Maximum allowed energy increase per step (eV).
            Steps exceeding this are rejected and velocity is reset.
        hessian_retrace_interval: Compute Hessians every N trajectory points
            during post-convergence retrace. Set to 0 to skip retrace.
        verbose: Print progress

    Returns:
        MinimizationResult with trajectory (hessians filled by retrace if converged)
    """
    atoms = atoms.copy()
    atoms.calc = calc

    # FIRE parameters
    dt = 0.1
    dt_max = 1.0
    alpha_start = 0.1
    alpha = alpha_start
    N_min = 5
    N_pos = 0
    v = np.zeros_like(atoms.positions)

    # Trajectory
    traj_positions: list[np.ndarray] = []
    traj_energies: list[float] = []
    traj_forces: list[np.ndarray] = []

    # Initial state
    forces = atoms.get_forces()
    energy = atoms.get_potential_energy()

    traj_positions.append(atoms.positions.copy())
    traj_energies.append(energy)
    traj_forces.append(forces.copy())

    converged = False

    for step in range(max_steps):
        # Check convergence
        force_max = np.max(np.abs(forces))
        force_rms = np.sqrt(np.mean(forces**2))

        if force_max < force_max_tol and force_rms < force_rms_tol:
            converged = True
            break

        # FIRE velocity mixing
        power = np.sum(v * forces)

        if power > 0:
            N_pos += 1
            if N_pos > N_min:
                dt = min(dt * 1.1, dt_max)
                alpha *= 0.99
            f_norm = np.linalg.norm(forces)
            if f_norm > 1e-10:
                f_hat = forces / f_norm
                v = (1 - alpha) * v + alpha * np.linalg.norm(v) * f_hat
        else:
            # Going uphill — reset velocity
            v[:] = 0
            alpha = alpha_start
            dt *= 0.5
            N_pos = 0

        # MD step (unit masses): v += F*dt, x += v*dt
        v += forces * dt
        new_positions = atoms.positions + v * dt

        # Evaluate new position
        prev_energy = energy
        atoms.positions = new_positions
        forces = atoms.get_forces()
        energy = atoms.get_potential_energy()

        # Reject meaningful energy increase
        if energy > prev_energy + energy_increase_tol:
            # Revert to last accepted state
            atoms.positions = traj_positions[-1].copy()
            forces = traj_forces[-1].copy()
            energy = prev_energy
            v[:] = 0
            alpha = alpha_start
            dt *= 0.5
            N_pos = 0
            if verbose:
                logger.debug(
                    f"FIRE step {step}: rejected (ΔE=+{energy - prev_energy:.6f} eV)"
                )
            continue

        traj_positions.append(atoms.positions.copy())
        traj_energies.append(energy)
        traj_forces.append(forces.copy())

        if verbose and step % 20 == 0:
            logger.debug(
                f"FIRE step {step}: E={energy:.6f} eV, "
                f"force_max={force_max:.4f} eV/Å, dt={dt:.4f}"
            )

    # Final convergence check
    if not converged:
        force_max = np.max(np.abs(forces))
        force_rms = np.sqrt(np.mean(forces**2))
        converged = force_max < force_max_tol and force_rms < force_rms_tol

    if not converged and verbose:
        logger.debug(
            f"FIRE did not converge in {max_steps} steps: "
            f"force_max={force_max:.4f}, force_rms={force_rms:.4f} eV/Å"
        )

    # Retrace trajectory to compute Hessians (only on convergence)
    traj_hessians: list[Optional[np.ndarray]] = [None] * len(traj_positions)
    if converged and hessian_retrace_interval > 0:
        retrace_indices = [
            i for i in range(len(traj_positions))
            if i % hessian_retrace_interval == 0
        ]

        if retrace_indices and isinstance(calc, BatchedHessianCalculator):
            # Batched retrace: single GPU call for all retrace points
            retrace_atoms_list: list[Atoms] = []
            for i in retrace_indices:
                a = atoms.copy()
                a.positions = traj_positions[i]
                a.calc = calc
                retrace_atoms_list.append(a)
            retrace_results = calc.get_batched_hessians(retrace_atoms_list)
            for i, (_, _, H) in zip(retrace_indices, retrace_results):
                traj_hessians[i] = H.copy()
        elif retrace_indices:
            # Sequential fallback
            for i in retrace_indices:
                atoms.positions = traj_positions[i]
                _, _, H = get_forces_energy_and_hessian(calc, atoms)
                traj_hessians[i] = H.copy()

    trajectory = MinimizationTrajectory(
        positions=traj_positions,
        energies=traj_energies,
        forces=traj_forces,
        hessians=traj_hessians,
    )

    return MinimizationResult(
        converged=converged,
        early_stopped=False,
        positions=traj_positions[-1].copy(),
        energy=traj_energies[-1],
        n_steps=len(traj_positions) - 1,
        final_force_max=force_max,
        final_force_rms=force_rms,
        trajectory=trajectory,
    )


def _newton_step(
    forces: np.ndarray,
    H: np.ndarray,
    positions: np.ndarray,
    masses: np.ndarray,
    trust_radius: float = 0.1,
    min_eigenvalue_magnitude: float = 0.01,
    eigenvalue_zero_threshold: float = 0.0062,
) -> tuple[np.ndarray, float, float]:
    """
    Compute a trust-region Newton step for minimization.

    Uses the RFO formulation with negative root for all modes (pure minimization).

    Args:
        forces: Forces in eV/Å, shape (n_atoms, 3)
        H: Hessian in eV/Å², shape (3*n_atoms, 3*n_atoms)
        positions: Positions in Å, shape (n_atoms, 3)
        masses: Atomic masses, shape (n_atoms,)
        trust_radius: Maximum step size in Å
        min_eigenvalue_magnitude: Floor for eigenvalue magnitudes in eV/Å².
            Prevents pathologically large steps along soft modes (common near TSs).

    Returns:
        (displacement, step_norm, min_eigenvalue): Displacement, norm, and smallest eigenvalue
    """
    f_flat = forces.flatten()
    n_dof = len(f_flat)

    H_proj, P_matrix = project_hessian(H, positions, masses)
    f_proj = P_matrix @ f_flat

    eigenvalues, eigenvectors = np.linalg.eigh(H_proj)

    # Identify non-zero modes (exclude translation/rotation)
    zero_threshold = eigenvalue_zero_threshold
    nonzero_mask = np.abs(eigenvalues) > zero_threshold
    nonzero_eigenvalues = eigenvalues[nonzero_mask]
    min_eigenvalue = nonzero_eigenvalues.min() if len(nonzero_eigenvalues) > 0 else 0.0

    # Transform gradient to eigenvector basis (g = -f, gradient of energy)
    g_proj = eigenvectors.T @ (-f_proj)
    step_proj = np.zeros(n_dof)

    # Compute RFO step for pure minimization (all modes use negative root)
    for i in range(n_dof):
        if not nonzero_mask[i]:
            continue

        lam = eigenvalues[i]
        g_i = g_proj[i]

        # Apply eigenvalue floor to prevent huge steps along soft modes
        # Preserves sign (important for negative curvature) but bounds magnitude
        if abs(lam) < min_eigenvalue_magnitude:
            lam = (
                np.sign(lam) * min_eigenvalue_magnitude
                if lam != 0
                else min_eigenvalue_magnitude
            )

        # Always minimize: use negative root
        step_proj[i] = _rfo_step_component(lam, g_i, maximize=False)

    step_flat = eigenvectors @ step_proj

    # Apply trust radius
    step_norm = np.linalg.norm(step_flat)
    if step_norm > trust_radius:
        step_flat *= trust_radius / step_norm
        step_norm = trust_radius

    return step_flat.reshape(-1, 3), step_norm, min_eigenvalue


def optimize_minimum(
    calc: Calculator,
    atoms: Atoms,
    max_steps: int = 100,
    force_max_tol: float = 0.05,
    force_rms_tol: float = 0.02,
    trust_radius: float = 0.3,
    min_trust_radius: float = 0.005,
    max_trust_radius: float = 2.0,
    adaptive_trust: bool = True,
    max_step_retries: int = 5,
    step_accept_threshold: float = 0.1,
    line_search: bool = True,
    min_eigenvalue_magnitude: float = 0.01,
    early_stop_window: int = 20,
    early_stop_min_displacement: float = 1e-4,
    early_stop_efficiency_threshold: float = 0.1,
    early_stop_force_guard: float = 5.0,
    hessian_update_interval: int = 1,
    energy_increase_tol: float = 0.001,
    verbose: bool = False,
) -> MinimizationResult:
    """
    Minimize energy using exact Hessians with trust-region Newton.

    This is more efficient than BFGS when Hessians are cheap, and collects
    the Hessians as part of the trajectory (no need to recompute later).

    Args:
        calc: ASE Calculator with get_hessian method
        atoms: ASE Atoms object with starting geometry
        max_steps: Maximum optimization steps
        force_max_tol: Max force component convergence tolerance in eV/Å
        force_rms_tol: RMS force convergence tolerance in eV/Å
        trust_radius: Initial trust radius in Å
        min_trust_radius: Minimum trust radius in Å
        max_trust_radius: Maximum trust radius in Å
        adaptive_trust: Whether to adaptively adjust trust radius
        max_step_retries: Maximum retries with smaller trust radius
        step_accept_threshold: Minimum trust region ratio to accept step
        line_search: Whether to use backtracking line search for rejected steps
        min_eigenvalue_magnitude: Floor for eigenvalue magnitudes in eV/Å².
            Prevents pathologically large steps along soft modes near TSs.
        early_stop_window: Number of steps to look back for progress check
        early_stop_min_displacement: Minimum net displacement over window to avoid
            "stuck" detection (Å). If both path is short AND displacement is tiny.
        early_stop_efficiency_threshold: Minimum ratio of net displacement to path
            length. Below this, we're oscillating (moving but going nowhere).
        early_stop_force_guard: Don't early-stop when force_max is within this
            factor of force_max_tol. Prevents killing optimization that is
            close to convergence but appears to oscillate.
        hessian_update_interval: Recompute the Hessian every N steps (1 = every
            step). Between updates, only forces/energy are computed and the last
            Hessian is reused for Newton steps. Higher values give large speedups
            since Hessian computation dominates cost (~3*N_atoms forward passes).
        energy_increase_tol: Maximum allowed energy increase per step (eV).
            Accommodates noise in approximate (ML) energy models. Steps with
            energy increases within this tolerance are accepted.
        verbose: Whether to print diagnostic information

    Returns:
        MinimizationResult with final positions, energy, and full trajectory
    """
    atoms = atoms.copy()
    atoms.calc = calc
    masses = atoms.get_masses()

    # Trajectory collection
    traj_positions: list[np.ndarray] = []
    traj_energies: list[float] = []
    traj_forces: list[np.ndarray] = []
    traj_hessians: list[Optional[np.ndarray]] = []

    # Early stopping tracking
    position_history: list[np.ndarray] = []  # For computing net displacement
    step_norms: list[float] = []  # For computing path length

    converged = False
    early_stopped = False
    current_trust = trust_radius
    H: Optional[np.ndarray] = None  # Current Hessian (reused between updates)
    force_hessian_recompute = False  # Set after rejected steps

    step = 0
    while step < max_steps:
        # Compute forces + energy every step (cheap)
        # Compute Hessian only every hessian_update_interval steps (expensive)
        # Also recompute after rejected steps (stale Hessian may be the cause)
        compute_hessian = (
            (step % hessian_update_interval == 0)
            or H is None
            or force_hessian_recompute
        )
        force_hessian_recompute = False

        if compute_hessian:
            forces, energy, H = get_forces_energy_and_hessian(calc, atoms)
            traj_hessians.append(H.copy())
        else:
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            traj_hessians.append(None)

        # Collect trajectory data
        traj_positions.append(atoms.positions.copy())
        traj_energies.append(energy)
        traj_forces.append(forces.copy())

        # Check convergence
        force_max = np.max(np.abs(forces))
        force_rms = np.sqrt(np.mean(forces**2))

        # Track position history for early stopping
        position_history.append(atoms.positions.copy())

        if force_max < force_max_tol and force_rms < force_rms_tol:
            converged = True
            if verbose:
                logger.debug(
                    f"Minimization converged at step {step + 1}: "
                    f"force_max={force_max:.6f}, force_rms={force_rms:.6f} eV/Å"
                )
            break

        # Early stopping: check if stuck or oscillating
        # Compare path length (sum of steps) vs net displacement over window
        if (
            len(position_history) >= early_stop_window
            and len(step_norms) >= early_stop_window - 1
        ):
            # Net displacement: how far we moved from start to end of window
            start_pos = position_history[-early_stop_window]
            end_pos = position_history[-1]
            net_displacement = np.linalg.norm((end_pos - start_pos).flatten())

            # Path length: sum of individual step norms over window
            path_length = sum(step_norms[-(early_stop_window - 1) :])

            # Efficiency: are we making progress or just wandering?
            efficiency = net_displacement / path_length if path_length > 1e-10 else 1.0

            # Stuck: tiny path AND tiny displacement (not moving at all)
            is_stuck = (
                path_length < early_stop_min_displacement
                and net_displacement < early_stop_min_displacement
            )

            # Oscillating: moving a lot but ending up close to start
            is_oscillating = (
                path_length > early_stop_min_displacement
                and efficiency < early_stop_efficiency_threshold
            )

            if is_stuck or is_oscillating:
                # Don't early-stop if forces are already close to convergence
                if force_max < force_max_tol * early_stop_force_guard:
                    if verbose:
                        logger.debug(
                            f"Suppressing early stop at step {step + 1}: "
                            f"forces near tolerance (force_max={force_max:.6f} eV/Å, "
                            f"tol={force_max_tol:.6f})"
                        )
                else:
                    early_stopped = True
                    reason = (
                        "stuck (not moving)"
                        if is_stuck
                        else f"oscillating (efficiency={efficiency:.3f})"
                    )
                    if verbose:
                        logger.debug(
                            f"Early stopping at step {step + 1}: {reason}. "
                            f"path_length={path_length:.2e} Å, "
                            f"net_displacement={net_displacement:.2e} Å, "
                            f"force_max={force_max:.4f} eV/Å"
                        )
                    break

        # Compute Newton step (RFO handles negative curvature correctly)
        displacement, step_norm, min_eigenvalue = _newton_step(
            forces,
            H,
            atoms.positions,
            masses,
            trust_radius=current_trust,
            min_eigenvalue_magnitude=min_eigenvalue_magnitude,
        )

        # Compute predicted energy change
        predicted_change = _compute_predicted_energy_change(forces, H, displacement)

        # Evaluate trial position
        trial_atoms = atoms.copy()
        trial_atoms.positions = atoms.positions + displacement
        trial_atoms.calc = calc
        trial_energy = trial_atoms.get_potential_energy()

        actual_change = trial_energy - energy

        # Compute trust region ratio
        if abs(predicted_change) > 1e-10:
            rho = actual_change / predicted_change
        else:
            rho = 1.0

        # If step is rejected, try line search or trust reduction
        if rho < step_accept_threshold:
            accepted = False

            if line_search and actual_change > 0:
                # Energy increased - try backtracking line search along the direction
                alpha = 0.5
                for _ in range(max_step_retries):
                    trial_disp = alpha * displacement
                    trial_atoms.positions = atoms.positions + trial_disp
                    trial_energy = trial_atoms.get_potential_energy()
                    actual_change = trial_energy - energy

                    # Accept if energy decreased
                    if actual_change < 0:
                        displacement = trial_disp
                        step_norm = np.linalg.norm(displacement.flatten())
                        rho = (
                            actual_change / (alpha * predicted_change)
                            if abs(predicted_change) > 1e-10
                            else 1.0
                        )
                        accepted = True
                        if verbose:
                            logger.debug(
                                f"Step {step + 1}: Line search accepted at α={alpha:.3f}"
                            )
                        break
                    alpha *= 0.5

            if not accepted:
                # Fall back to trust radius reduction
                trial_trust = current_trust * 0.5
                for _ in range(max_step_retries):
                    if trial_trust < min_trust_radius:
                        trial_trust = min_trust_radius

                    displacement, step_norm, _ = _newton_step(
                        forces,
                        H,
                        atoms.positions,
                        masses,
                        trust_radius=trial_trust,
                        min_eigenvalue_magnitude=min_eigenvalue_magnitude,
                    )
                    predicted_change = _compute_predicted_energy_change(
                        forces, H, displacement
                    )

                    trial_atoms.positions = atoms.positions + displacement
                    trial_energy = trial_atoms.get_potential_energy()
                    actual_change = trial_energy - energy

                    if abs(predicted_change) > 1e-10:
                        rho = actual_change / predicted_change
                    else:
                        rho = 1.0

                    if rho > step_accept_threshold:
                        break

                    if trial_trust <= min_trust_radius:
                        break
                    trial_trust *= 0.5

        # Reject steps with energy increases beyond tolerance
        if actual_change > energy_increase_tol:
            current_trust = max(min_trust_radius, current_trust * 0.5)
            force_hessian_recompute = True  # Stale Hessian may be the cause
            step_norms.append(0.0)
            step += 1
            if verbose:
                logger.debug(
                    f"Step {step}: rejected (ΔE=+{actual_change:.6f} eV > "
                    f"tol={energy_increase_tol:.4f}), trust→{current_trust:.4f} Å"
                )
            continue

        # Update trust radius adaptively
        if adaptive_trust:
            step_hit_boundary = step_norm >= current_trust * 0.99
            is_positive_definite = min_eigenvalue > 1e-4

            if rho < 0.25 or rho > 2.0:
                # Poor model - shrink trust
                current_trust = max(min_trust_radius, current_trust * 0.5)
            elif 0.5 < rho < 1.5:
                # Good model
                if step_hit_boundary:
                    # Step was constrained, expand trust
                    # More aggressive expansion for positive definite Hessian (true minimum)
                    expansion = 2.0 if is_positive_definite else 1.5
                    current_trust = min(max_trust_radius, current_trust * expansion)
            elif 0.25 <= rho <= 0.5 or 1.5 <= rho <= 2.0:
                # Marginal model - keep trust unchanged
                pass

        # Apply step
        atoms.positions = atoms.positions + displacement
        step_norms.append(step_norm)  # Track for early stopping
        step += 1

        if verbose:
            logger.debug(
                f"Step {step}: E={energy:.6f} eV, ΔE={actual_change:.6f} eV, "
                f"ρ={rho:.3f}, trust={current_trust:.4f} Å, λ_min={min_eigenvalue:.4f}"
            )

    if not converged and verbose:
        if early_stopped:
            logger.debug(
                f"Minimization early stopped at step {step}: stuck with "
                f"force_max={force_max:.6f}, force_rms={force_rms:.6f} eV/Å"
            )
        else:
            logger.debug(
                f"Minimization did not converge within {max_steps} steps: "
                f"force_max={force_max:.6f}, force_rms={force_rms:.6f} eV/Å"
            )

    trajectory = MinimizationTrajectory(
        positions=traj_positions,
        energies=traj_energies,
        forces=traj_forces,
        hessians=traj_hessians,
    )

    return MinimizationResult(
        converged=converged,
        early_stopped=early_stopped,
        positions=atoms.positions.copy(),
        energy=energy,
        n_steps=step,
        final_force_max=force_max,
        final_force_rms=force_rms,
        trajectory=trajectory,
    )


@dataclass
class _MinSearchState:
    """Mutable state for one minimization within a batched run."""

    atoms: Atoms
    index: int  # Original index in input list
    step: int = 0
    converged: bool = False
    early_stopped: bool = False
    energy: float = 0.0
    force_max: float = float("inf")
    force_rms: float = float("inf")
    current_trust: float = 0.3
    H: Optional[np.ndarray] = None
    force_hessian_recompute: bool = False
    position_history: list[np.ndarray] = field(default_factory=list)
    step_norms: list[float] = field(default_factory=list)
    # Trajectory data
    traj_positions: list[np.ndarray] = field(default_factory=list)
    traj_energies: list[float] = field(default_factory=list)
    traj_forces: list[np.ndarray] = field(default_factory=list)
    traj_hessians: list[Optional[np.ndarray]] = field(default_factory=list)


def optimize_minima_batched(
    calc: Calculator,
    atoms_list: list[Atoms],
    max_steps: int = 100,
    force_max_tol: float = 0.05,
    force_rms_tol: float = 0.02,
    trust_radius: float = 0.3,
    min_trust_radius: float = 0.005,
    max_trust_radius: float = 2.0,
    adaptive_trust: bool = True,
    max_step_retries: int = 5,
    step_accept_threshold: float = 0.1,
    min_eigenvalue_magnitude: float = 0.01,
    early_stop_window: int = 20,
    early_stop_min_displacement: float = 1e-4,
    early_stop_efficiency_threshold: float = 0.1,
    early_stop_force_guard: float = 5.0,
    hessian_update_interval: int = 1,
    energy_increase_tol: float = 0.001,
    verbose: bool = False,
) -> list[MinimizationResult]:
    """
    Run multiple minimizations in lockstep, batching GPU calls.

    Each search maintains independent state (trust radius, Hessian, convergence).
    Searches finish at different times; the batch shrinks as searches converge
    or early-stop.

    Falls back to sequential optimize_minimum() if the calculator does not
    support batched operations.

    Args:
        calc: ASE Calculator (ideally with batched Hessian support)
        atoms_list: Starting geometries for each minimization
        max_steps: Maximum steps per search
        force_max_tol: Max force convergence tolerance (eV/Å)
        force_rms_tol: RMS force convergence tolerance (eV/Å)
        trust_radius: Initial trust radius (Å)
        min_trust_radius: Minimum trust radius (Å)
        max_trust_radius: Maximum trust radius (Å)
        adaptive_trust: Adaptively adjust trust radius
        max_step_retries: Max retries with smaller trust radius per step
        step_accept_threshold: Minimum trust region ratio to accept step
        min_eigenvalue_magnitude: Floor for eigenvalue magnitudes (eV/Å²)
        early_stop_window: Steps to look back for progress check
        early_stop_min_displacement: Minimum net displacement over window (Å)
        early_stop_efficiency_threshold: Minimum net/path ratio
        early_stop_force_guard: Don't early-stop when forces are within this
            factor of tolerance
        hessian_update_interval: Recompute Hessian every N steps
        energy_increase_tol: Maximum allowed energy increase per step (eV)
        verbose: Print diagnostic information

    Returns:
        List of MinimizationResult in the same order as atoms_list.
    """
    if not atoms_list:
        return []

    # Fallback: sequential if calculator doesn't support batched ops
    if not isinstance(calc, BatchedHessianCalculator):
        logger.debug(
            "Calculator does not support batched operations, falling back to sequential"
        )
        return [
            optimize_minimum(
                calc,
                atoms,
                max_steps=max_steps,
                force_max_tol=force_max_tol,
                force_rms_tol=force_rms_tol,
                trust_radius=trust_radius,
                min_trust_radius=min_trust_radius,
                max_trust_radius=max_trust_radius,
                adaptive_trust=adaptive_trust,
                max_step_retries=max_step_retries,
                step_accept_threshold=step_accept_threshold,
                min_eigenvalue_magnitude=min_eigenvalue_magnitude,
                early_stop_window=early_stop_window,
                early_stop_min_displacement=early_stop_min_displacement,
                early_stop_efficiency_threshold=early_stop_efficiency_threshold,
                early_stop_force_guard=early_stop_force_guard,
                hessian_update_interval=hessian_update_interval,
                energy_increase_tol=energy_increase_tol,
                verbose=verbose,
            )
            for atoms in atoms_list
        ]

    logger.debug(
        f"optimize_minima_batched: B={len(atoms_list)}, max_steps={max_steps}"
    )

    # Initialize per-search state
    states: list[_MinSearchState] = []
    for i, atoms in enumerate(atoms_list):
        a = atoms.copy()
        a.calc = calc
        states.append(
            _MinSearchState(
                atoms=a,
                index=i,
                current_trust=trust_radius,
            )
        )

    for _global_step in range(max_steps):
        # Collect active (non-finished) searches
        active = [s for s in states if not s.converged and not s.early_stopped
                  and s.step < max_steps]
        if not active:
            break

        # --- Phase 1: Batched Hessian or force/energy (GPU) ---
        # Split active searches into those needing Hessian vs force-only
        need_hessian: list[_MinSearchState] = []
        need_forces: list[_MinSearchState] = []
        for s in active:
            compute_hessian = (
                (s.step % hessian_update_interval == 0)
                or s.H is None
                or s.force_hessian_recompute
            )
            if compute_hessian:
                need_hessian.append(s)
            else:
                need_forces.append(s)

        # Batched Hessian call
        if need_hessian:
            hessian_results = calc.get_batched_hessians([s.atoms for s in need_hessian])
            for s, (forces, energy, H) in zip(need_hessian, hessian_results):
                forces = forces.reshape(-1, 3)
                s.energy = energy
                s.H = H
                s.force_hessian_recompute = False
                s.force_max = float(np.max(np.abs(forces)))
                s.force_rms = float(np.sqrt(np.mean(forces**2)))
                s.traj_positions.append(s.atoms.positions.copy())
                s.traj_energies.append(energy)
                s.traj_forces.append(forces.copy())
                s.traj_hessians.append(H.copy())
                s.position_history.append(s.atoms.positions.copy())

        # Batched force-only call
        if need_forces:
            force_results = calc.get_batched_forces_and_energy(
                [s.atoms for s in need_forces]
            )
            for s, (forces, energy) in zip(need_forces, force_results):
                forces = forces.reshape(-1, 3)
                s.energy = energy
                s.force_max = float(np.max(np.abs(forces)))
                s.force_rms = float(np.sqrt(np.mean(forces**2)))
                s.traj_positions.append(s.atoms.positions.copy())
                s.traj_energies.append(energy)
                s.traj_forces.append(forces.copy())
                s.traj_hessians.append(None)
                s.position_history.append(s.atoms.positions.copy())

        # --- Phase 2: Convergence + early stopping (CPU) ---
        still_active: list[_MinSearchState] = []
        for s in active:
            # Check convergence
            if s.force_max < force_max_tol and s.force_rms < force_rms_tol:
                s.converged = True
                if verbose:
                    logger.debug(
                        f"Search {s.index}: converged at step {s.step + 1}, "
                        f"force_max={s.force_max:.6f}, force_rms={s.force_rms:.6f}"
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
                path_length = sum(s.step_norms[-(early_stop_window - 1):])
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
                    # Don't early-stop if forces are close to convergence
                    if s.force_max < force_max_tol * early_stop_force_guard:
                        pass  # Suppress early stop
                    else:
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

            still_active.append(s)

        if not still_active:
            continue

        # --- Phase 3: Newton step (CPU) ---
        search_data: list[dict] = []
        for s in still_active:
            masses = s.atoms.get_masses()
            displacement, step_norm, min_eigenvalue = _newton_step(
                s.traj_forces[-1],
                s.H,
                s.atoms.positions,
                masses,
                trust_radius=s.current_trust,
                min_eigenvalue_magnitude=min_eigenvalue_magnitude,
            )
            predicted_change = _compute_predicted_energy_change(
                s.traj_forces[-1], s.H, displacement
            )
            search_data.append({
                "state": s,
                "displacement": displacement,
                "step_norm": step_norm,
                "min_eigenvalue": min_eigenvalue,
                "predicted_change": predicted_change,
            })

        # --- Phase 4: Batched trial energy (GPU) ---
        trial_atoms_list: list[Atoms] = []
        for d in search_data:
            s = d["state"]
            trial = s.atoms.copy()
            trial.positions = s.atoms.positions + d["displacement"]
            trial.calc = calc
            trial_atoms_list.append(trial)

        trial_results = calc.get_batched_forces_and_energy(trial_atoms_list)

        # --- Phase 5: Accept/reject with retry loop ---
        # First pass: compute rho for each search
        for d, (_, trial_energy) in zip(search_data, trial_results):
            d["trial_energy"] = trial_energy
            d["actual_change"] = trial_energy - d["state"].energy
            if abs(d["predicted_change"]) > 1e-10:
                d["rho"] = d["actual_change"] / d["predicted_change"]
            else:
                d["rho"] = 1.0

        # Identify rejected searches needing retry
        rejected: list[int] = [
            i for i, d in enumerate(search_data)
            if d["rho"] < step_accept_threshold
        ]

        # --- Phase A: Exhaustive line search (for searches with energy increase) ---
        # Matches sequential: try up to max_step_retries halvings before
        # falling through to trust reduction.
        needs_line_search = [
            idx for idx in rejected if search_data[idx]["actual_change"] > 0
        ]
        alpha = 0.5
        for _ls_retry in range(max_step_retries):
            if not needs_line_search:
                break

            ls_atoms: list[Atoms] = []
            ls_indices: list[int] = []
            for idx in needs_line_search:
                d = search_data[idx]
                trial_disp = alpha * d["displacement"]
                trial = d["state"].atoms.copy()
                trial.positions = d["state"].atoms.positions + trial_disp
                trial.calc = calc
                ls_atoms.append(trial)
                ls_indices.append(idx)
                d["_trial_disp"] = trial_disp
                d["_alpha"] = alpha

            ls_results = calc.get_batched_forces_and_energy(ls_atoms)
            still_need_ls = []
            for idx, (_, trial_energy) in zip(ls_indices, ls_results):
                d = search_data[idx]
                actual_change = trial_energy - d["state"].energy
                if actual_change < 0:
                    # Accept line search result
                    d["displacement"] = d["_trial_disp"]
                    d["step_norm"] = float(
                        np.linalg.norm(d["_trial_disp"].flatten())
                    )
                    d["actual_change"] = actual_change
                    d["trial_energy"] = trial_energy
                    alpha_pred = d["_alpha"] * d["predicted_change"]
                    d["rho"] = (
                        actual_change / alpha_pred
                        if abs(alpha_pred) > 1e-10
                        else 1.0
                    )
                else:
                    still_need_ls.append(idx)
            needs_line_search = still_need_ls
            alpha *= 0.5

        # --- Phase B: Exhaustive trust reduction (for all still-rejected) ---
        # Includes searches where line search failed AND those that had
        # actual_change <= 0 from the start.
        needs_trust = [
            idx for idx in rejected
            if search_data[idx]["rho"] < step_accept_threshold
        ]
        for _tr_retry in range(max_step_retries):
            if not needs_trust:
                break

            tr_atoms: list[Atoms] = []
            tr_indices: list[int] = []
            for idx in needs_trust:
                d = search_data[idx]
                s = d["state"]
                trial_trust = s.current_trust * (0.5 ** (_tr_retry + 1))
                if trial_trust < min_trust_radius:
                    trial_trust = min_trust_radius

                masses = s.atoms.get_masses()
                displacement, step_norm, min_eig = _newton_step(
                    s.traj_forces[-1],
                    s.H,
                    s.atoms.positions,
                    masses,
                    trust_radius=trial_trust,
                    min_eigenvalue_magnitude=min_eigenvalue_magnitude,
                )
                d["displacement"] = displacement
                d["step_norm"] = step_norm
                d["min_eigenvalue"] = min_eig
                d["predicted_change"] = _compute_predicted_energy_change(
                    s.traj_forces[-1], s.H, displacement
                )

                trial = s.atoms.copy()
                trial.positions = s.atoms.positions + displacement
                trial.calc = calc
                tr_atoms.append(trial)
                tr_indices.append(idx)

            tr_results = calc.get_batched_forces_and_energy(tr_atoms)
            still_need_trust = []
            for idx, (_, trial_energy) in zip(tr_indices, tr_results):
                d = search_data[idx]
                d["trial_energy"] = trial_energy
                d["actual_change"] = trial_energy - d["state"].energy
                if abs(d["predicted_change"]) > 1e-10:
                    d["rho"] = d["actual_change"] / d["predicted_change"]
                else:
                    d["rho"] = 1.0

                if d["rho"] < step_accept_threshold:
                    # Still rejected — keep retrying unless we already
                    # tried at min trust (matches sequential: clamp to min,
                    # try once, then give up).
                    current_unclamped = d["state"].current_trust * (0.5 ** (_tr_retry + 1))
                    if current_unclamped > min_trust_radius:
                        still_need_trust.append(idx)
            needs_trust = still_need_trust

        # --- Apply accepted steps + update state ---
        for d in search_data:
            s = d["state"]
            actual_change = d["actual_change"]
            rho = d["rho"]

            # Reject steps with energy increases beyond tolerance
            if actual_change > energy_increase_tol:
                s.current_trust = max(min_trust_radius, s.current_trust * 0.5)
                s.force_hessian_recompute = True
                s.step_norms.append(0.0)
                s.step += 1
                if verbose:
                    logger.debug(
                        f"Search {s.index} step {s.step}: rejected "
                        f"(ΔE=+{actual_change:.6f} eV), trust→{s.current_trust:.4f}"
                    )
                continue

            # Update trust radius adaptively
            if adaptive_trust:
                step_hit_boundary = d["step_norm"] >= s.current_trust * 0.99
                is_positive_definite = d["min_eigenvalue"] > 1e-4

                if rho < 0.25 or rho > 2.0:
                    s.current_trust = max(min_trust_radius, s.current_trust * 0.5)
                elif 0.5 < rho < 1.5:
                    if step_hit_boundary:
                        expansion = 2.0 if is_positive_definite else 1.5
                        s.current_trust = min(max_trust_radius, s.current_trust * expansion)

            # Apply step
            s.atoms.positions = s.atoms.positions + d["displacement"]
            s.step_norms.append(d["step_norm"])
            s.step += 1

    # Assemble results in original input order
    results: list[MinimizationResult | None] = [None] * len(atoms_list)
    for s in states:
        trajectory = MinimizationTrajectory(
            positions=s.traj_positions,
            energies=s.traj_energies,
            forces=s.traj_forces,
            hessians=s.traj_hessians,
        )
        results[s.index] = MinimizationResult(
            converged=s.converged,
            early_stopped=s.early_stopped,
            positions=s.atoms.positions.copy(),
            energy=s.energy,
            n_steps=s.step,
            final_force_max=s.force_max,
            final_force_rms=s.force_rms,
            trajectory=trajectory,
        )
    return results  # type: ignore[return-value]
