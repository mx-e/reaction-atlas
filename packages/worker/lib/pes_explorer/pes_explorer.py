"""
PES Explorer: High-level interface for exploring molecular potential energy surfaces.

Provides a single function `explore_pes` that handles:
- Disk caching and recovery from interrupted explorations
- Iterative exploration from minima via MD and transition state search

The calculator (ML force field) must be instantiated externally and passed in.

Usage:
    from lib.pes_explorer.pes_explorer import explore_pes, ExploreConfig
    from lib.md_et_calculator import get_md_et_calculator

    # Create calculator externally
    calc = get_md_et_calculator(run_dir, device="cuda")

    # Run exploration
    graph = explore_pes(atoms, calc)

    # With custom configuration
    config = ExploreConfig(md_steps=100_000, temperature=500.0)
    graph = explore_pes(atoms, calc, config=config, cache_dir="my_cache")

    # Continue from existing graph
    graph = explore_pes(atoms, calc, existing_graph=partial_graph)
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union
import hashlib

import numpy as np
import torch as th
from ase import Atoms
from ase.calculators.calculator import Calculator
from loguru import logger
from scipy.signal import butter, filtfilt
from lib.md import (
    MDOpts,
    TrajectoryCollector,
    ConsoleLogger,
    run_md_simulation,
)
from lib.pes_explorer.pes_graph import PESGraph, RelaxationTrajectory, compute_rmsd
from lib.pes_explorer.prfo import (
    BatchedHessianCalculator,
    get_forces_energy_and_hessian,
    project_hessian,
    optimize_saddle_point,
    optimize_saddle_points_batched,
)
from lib.pes_explorer.newton_minimize import optimize_minimum, optimize_minima_batched


@dataclass
class ExploreConfig:
    """Configuration for PES exploration."""

    # MD parameters
    md_steps: int = 2_500
    md_save_interval: int = 10
    md_log_interval: int = 2000
    md_step_size_fs: float = 0.5
    temperature: float = 300.0
    thermostat: str = "langevin"
    thermostat_tau: float = 100.0

    # Relaxation
    bfgs_fmax: float = 0.005
    bfgs_steps: int = 120

    # Exploration
    max_iterations: int = 200
    n_ts_samples: int = 32
    ts_top_percentile: float = 10.0
    use_saddle_metric: bool = False

    # TS optimization
    ts_force_tol: float = 0.01
    ts_trust_radius: float = 0.5
    ts_max_steps: int = 120

    # Validation
    ts_validation_displacement: float = (
        0.0  # 0 = adaptive based on curvature, >0 = fixed value in Å
    )
    ts_validation_min_displacement: float = 0.02  # Minimum adaptive displacement (Å)
    ts_validation_max_displacement: float = 0.5  # Maximum adaptive displacement (Å)
    ts_validation_min_rmsd: float = 0.25
    ts_validation_fmax: float = 0.005
    ts_validation_energy_threshold: float = 0.043  # ~1 kcal/mol
    eigenvalue_zero_threshold: float = 0.0062  # eV/Å² — below this, eigenvalue is noise

    # Deduplication
    rmsd_threshold: float = 0.20
    energy_threshold: float = 0.043  # ~1 kcal/mol

    # NEB-based dedup (energy gate + NEB barrier, no RMSD cap)
    neb_barrier_threshold: float = 0.043  # ~1 kcal/mol
    neb_n_images: int = 5
    neb_fire_steps: int = 150
    neb_spring_constant: float = 0.1

    # Filter
    filter_cutoff: float = 0.05
    filter_order: int = 2

    # Caching
    save_interval: int = 1  # Save after every N iterations


def _get_molecule_hash(atoms: Atoms) -> str:
    """Generate a hash identifier for a molecule based on composition and geometry."""
    atomic_numbers = tuple(sorted(atoms.get_atomic_numbers()))
    n_atoms = len(atoms)
    # Use sorted eigenvalues of distance matrix for rotation-invariant hash
    positions = atoms.get_positions()
    dist_matrix = np.sqrt(
        np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=2)
    )
    eigenvalues = np.linalg.eigvalsh(dist_matrix)
    eigenvalues_rounded = tuple(np.round(eigenvalues, 3))

    hash_input = f"{atomic_numbers}_{n_atoms}_{eigenvalues_rounded}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def _get_cache_path(
    atoms: Atoms, cache_dir: Path, cache_name: Optional[str] = None
) -> Path:
    """Generate cache file path for a molecule."""
    mol_hash = _get_molecule_hash(atoms)
    if cache_name:
        return cache_dir / f"pes_{mol_hash}_{cache_name}.pkl"
    return cache_dir / f"pes_{mol_hash}.pkl"


# -----------------------------------------------------------------------------
# Internal helper functions
# -----------------------------------------------------------------------------


def _low_pass_filter(
    data: np.ndarray, cutoff: float = 0.05, order: int = 2
) -> np.ndarray:
    """Apply a low pass Butterworth filter."""
    data = np.array(data)
    if len(data) < 10:
        return data
    if np.any(np.isnan(data)):
        mask = ~np.isnan(data)
        data = np.interp(np.arange(len(data)), np.arange(len(data))[mask], data[mask])
    b, a = butter(order, cutoff, btype="low")
    return filtfilt(b, a, data)


def _compute_saddle_metric(
    energies: np.ndarray, forces_norms: np.ndarray, momenta_norms: np.ndarray
) -> np.ndarray:
    """Compute saddle-point likelihood score for each frame."""
    E_min, E_max = energies.min(), energies.max()
    F_min, F_max = forces_norms.min(), forces_norms.max()
    M_min, M_max = momenta_norms.min(), momenta_norms.max()

    E_norm = (energies - E_min) / (E_max - E_min + 1e-10)
    F_norm = (forces_norms - F_min) / (F_max - F_min + 1e-10)
    M_norm = (momenta_norms - M_min) / (M_max - M_min + 1e-10)

    return E_norm - F_norm - 2.0 * M_norm


def _validate_ts(
    calc: Calculator,
    ts_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    config: ExploreConfig,
    charge: int = 0,
):
    """Validate TS by relaxing in both directions along imaginary mode."""
    n_atoms = len(atomic_numbers)

    ts_atoms = Atoms(numbers=atomic_numbers, positions=ts_positions.copy())
    ts_atoms.info['charge'] = charge
    ts_atoms.calc = calc
    masses = ts_atoms.get_masses()

    forces, ts_energy, H = get_forces_energy_and_hessian(calc, ts_atoms)
    H_proj, _ = project_hessian(H, ts_positions, masses)
    eigenvalues, eigenvectors = np.linalg.eigh(H_proj)

    negative_indices = np.where(eigenvalues < -config.eigenvalue_zero_threshold)[0]
    if len(negative_indices) == 0:
        logger.debug(
            f"No negative eigenvalues found in TS validation, min value: {eigenvalues.min():.4f} eV/Å²"
        )
        return None

    ts_mode_idx = negative_indices[0]
    ts_eigenvector = eigenvectors[:, ts_mode_idx].reshape(n_atoms, 3)
    ts_eigenvector = ts_eigenvector / np.linalg.norm(ts_eigenvector)
    ts_eigenvalue = eigenvalues[ts_mode_idx]

    # Calculate adaptive displacement based on local curvature
    # Using harmonic approximation: E = E_0 + 0.5 * λ * x²
    # To achieve target energy drop ΔE: x = sqrt(2 * |ΔE| / |λ|)
    target_energy_drop = (
        config.ts_validation_energy_threshold * 2
    )  # Aim for 2x the validation threshold
    adaptive_displacement = np.sqrt(2 * target_energy_drop / abs(ts_eigenvalue))

    # Clamp to reasonable bounds from config
    adaptive_displacement = np.clip(
        adaptive_displacement,
        config.ts_validation_min_displacement,
        config.ts_validation_max_displacement,
    )

    # Use adaptive displacement if ts_validation_displacement is 0, otherwise use fixed
    if config.ts_validation_displacement > 0:
        displacement = config.ts_validation_displacement
    else:
        displacement = adaptive_displacement

    logger.debug(
        f"TS validation: eigenvalue={ts_eigenvalue:.4f} eV/Å², "
        f"adaptive_displacement={adaptive_displacement:.4f} Å, using={displacement:.4f} Å"
    )

    endpoints = []
    trajectories = []
    for sign in [+1, -1]:
        displaced = ts_positions + sign * displacement * ts_eigenvector
        atoms = Atoms(numbers=atomic_numbers, positions=displaced.copy())
        atoms.info['charge'] = charge
        atoms.calc = calc

        try:
            result = optimize_minimum(
                calc,
                atoms,
                max_steps=config.bfgs_steps,
                force_max_tol=config.ts_validation_fmax,
                force_rms_tol=config.ts_validation_fmax * 0.5,
                early_stop_window=config.bfgs_steps + 1,  # disable early stopping
                hessian_update_interval=2,
                energy_increase_tol=0.005,
                verbose=False,
            )

            # Check if relaxation produced a valid endpoint
            # Accept if converged, OR if forces are within 10x tolerance (partially converged)
            final_forces = (
                result.trajectory.forces[-1]
                if result.trajectory.forces
                else atoms.get_forces()
            )
            max_force = np.linalg.norm(final_forces, axis=1).max()
            final_energy = result.energy

            is_valid = result.converged or max_force < config.ts_validation_fmax * 10

            # Also check that energy decreased (we're going downhill from TS)
            energy_decreased = (
                final_energy < ts_energy + 0.1
            )  # Allow small increase for numerical noise

            if is_valid and energy_decreased:
                endpoints.append(
                    {
                        "positions": result.positions.copy(),
                        "energy": final_energy,
                    }
                )
                # Convert MinimizationTrajectory to RelaxationTrajectory
                trajectories.append(
                    RelaxationTrajectory(
                        positions=result.trajectory.positions,
                        energies=result.trajectory.energies,
                        forces=result.trajectory.forces,
                        hessians=result.trajectory.hessians,
                    )
                )
            else:
                reason = []
                if not is_valid:
                    reason.append(f"max_force={max_force:.4f} eV/Å")
                if not energy_decreased:
                    reason.append(
                        f"energy increased by {final_energy - ts_energy:.4f} eV"
                    )
                logger.debug(
                    f"TS validation relaxation rejected ({sign:+d} direction): {', '.join(reason)}"
                )
        except Exception as e:
            logger.debug(
                f"TS validation relaxation failed ({sign:+d} direction): {type(e).__name__}: {e}"
            )

        # Both directions must succeed — bail early if first one failed
        if sign == +1 and len(endpoints) == 0:
            logger.debug("TS validation: first direction failed, skipping second")
            return None

    if len(endpoints) != 2:
        logger.debug(
            f"TS validation failed: expected 2 endpoints, got {len(endpoints)}"
        )
        return None

    left_diff = endpoints[0]["energy"] - ts_energy
    right_diff = endpoints[1]["energy"] - ts_energy
    both_lower = (
        left_diff < -config.ts_validation_energy_threshold
        and right_diff < -config.ts_validation_energy_threshold
    )

    ep_rmsd = compute_rmsd(endpoints[0]["positions"], endpoints[1]["positions"], atomic_numbers)
    different = ep_rmsd > config.ts_validation_min_rmsd

    if not (both_lower and different):
        if not both_lower:
            logger.debug(
                f"TS validation failed: endpoints not sufficiently lower in energy ({left_diff:.4f}, {right_diff:.4f} eV)"
            )
        if not different:
            logger.debug(
                f"TS validation failed: endpoints too similar (RMSD = {ep_rmsd:.4f} Å)"
            )
        return None

    # Convention: forward = downhill (fwd = higher energy reactant, bwd = lower energy product)
    if endpoints[0]["energy"] >= endpoints[1]["energy"]:
        fwd_idx, bwd_idx = 0, 1
    else:
        fwd_idx, bwd_idx = 1, 0

    # Extract the last non-None hessian from each endpoint's relaxation trajectory
    def _last_hessian(traj: RelaxationTrajectory):
        for h in reversed(traj.hessians):
            if h is not None:
                return h
        return None

    fwd_hessian = _last_hessian(trajectories[fwd_idx])
    bwd_hessian = _last_hessian(trajectories[bwd_idx])
    if fwd_hessian is None:
        logger.error("Forward minimum has no hessian — relaxation trajectory missing hessian data")
    if bwd_hessian is None:
        logger.error("Backward minimum has no hessian — relaxation trajectory missing hessian data")
    if H is None:
        logger.error("Transition state has no hessian")

    return {
        "ts_positions": ts_positions,
        "ts_energy": ts_energy,
        "eigenvalue": eigenvalues[ts_mode_idx],
        "hessian": H,
        "fwd_positions": endpoints[fwd_idx]["positions"],
        "fwd_energy": endpoints[fwd_idx]["energy"],
        "bwd_positions": endpoints[bwd_idx]["positions"],
        "bwd_energy": endpoints[bwd_idx]["energy"],
        "fwd_trajectory": trajectories[fwd_idx],
        "bwd_trajectory": trajectories[bwd_idx],
        "fwd_hessian": fwd_hessian,
        "bwd_hessian": bwd_hessian,
    }


def _validate_ts_batched(
    calc: Calculator,
    ts_positions_list: list[np.ndarray],
    atomic_numbers: np.ndarray,
    config: ExploreConfig,
    charge: int = 0,
) -> list[Optional[dict]]:
    """Validate multiple TSes in batch: Hessian check + batched relaxation.

    Falls back to sequential _validate_ts if calculator doesn't support batching.

    Args:
        calc: Calculator (ideally BatchedHessianCalculator)
        ts_positions_list: List of TS position arrays to validate
        atomic_numbers: Atomic numbers for all structures (shared)
        config: Exploration config
        charge: Net formal charge

    Returns:
        List of validation dicts (or None) in same order as input.
    """
    if not ts_positions_list:
        return []

    n_atoms = len(atomic_numbers)

    # Fallback to sequential if no batched support
    if not isinstance(calc, BatchedHessianCalculator):
        return [
            _validate_ts(calc, pos, atomic_numbers, config, charge=charge)
            for pos in ts_positions_list
        ]

    logger.debug(
        f"Batched TS validation: {len(ts_positions_list)} candidates"
    )

    # --- Step 1: Batched Hessian for all TS candidates ---
    ts_atoms_list: list[Atoms] = []
    for pos in ts_positions_list:
        a = Atoms(numbers=atomic_numbers, positions=pos.copy())
        a.info["charge"] = charge
        a.calc = calc
        ts_atoms_list.append(a)

    hessian_results = calc.get_batched_hessians(ts_atoms_list)

    # --- Step 2: CPU eigendecomposition + displacement computation per TS ---
    # Track which TSes are valid and their displaced geometries
    valid_ts_indices: list[int] = []  # Index into ts_positions_list
    ts_data: list[dict] = []  # Per-valid-TS data (eigenvalue, eigenvector, etc.)
    displaced_atoms: list[Atoms] = []  # All displaced geometries (2 per valid TS)
    displaced_ts_map: list[tuple[int, int]] = []  # (ts_data_idx, sign_idx) per displaced atom

    for i, (forces, ts_energy, H) in enumerate(hessian_results):
        ts_positions = ts_positions_list[i]
        masses = ts_atoms_list[i].get_masses()

        H_proj, _ = project_hessian(H, ts_positions, masses)
        eigenvalues, eigenvectors = np.linalg.eigh(H_proj)

        negative_indices = np.where(eigenvalues < -config.eigenvalue_zero_threshold)[0]
        if len(negative_indices) == 0:
            logger.debug(
                f"TS {i}: no negative eigenvalues, min={eigenvalues.min():.4f} eV/Å²"
            )
            continue

        ts_mode_idx = negative_indices[0]
        ts_eigenvector = eigenvectors[:, ts_mode_idx].reshape(n_atoms, 3)
        ts_eigenvector = ts_eigenvector / np.linalg.norm(ts_eigenvector)
        ts_eigenvalue = eigenvalues[ts_mode_idx]

        # Adaptive displacement
        target_energy_drop = config.ts_validation_energy_threshold * 2
        adaptive_displacement = np.sqrt(2 * target_energy_drop / abs(ts_eigenvalue))
        adaptive_displacement = np.clip(
            adaptive_displacement,
            config.ts_validation_min_displacement,
            config.ts_validation_max_displacement,
        )

        if config.ts_validation_displacement > 0:
            displacement = config.ts_validation_displacement
        else:
            displacement = adaptive_displacement

        logger.debug(
            f"TS {i}: eigenvalue={ts_eigenvalue:.4f} eV/Å², "
            f"displacement={displacement:.4f} Å"
        )

        ts_data_idx = len(ts_data)
        valid_ts_indices.append(i)
        ts_data.append({
            "ts_positions": ts_positions,
            "ts_energy": ts_energy,
            "H": H,
            "eigenvalue": eigenvalues[ts_mode_idx],
        })

        # Create +1 and -1 displaced geometries
        for sign_idx, sign in enumerate([+1, -1]):
            displaced = ts_positions + sign * displacement * ts_eigenvector
            a = Atoms(numbers=atomic_numbers, positions=displaced.copy())
            a.info["charge"] = charge
            a.calc = calc
            displaced_atoms.append(a)
            displaced_ts_map.append((ts_data_idx, sign_idx))

    if not displaced_atoms:
        logger.debug("No valid TS candidates after eigendecomposition")
        return [None] * len(ts_positions_list)

    # --- Step 3: Batched relaxation of all displaced geometries ---
    logger.debug(
        f"Batched TS validation: relaxing {len(displaced_atoms)} displaced geometries "
        f"({len(ts_data)} TSes × 2 directions)"
    )
    relax_results = optimize_minima_batched(
        calc,
        displaced_atoms,
        max_steps=config.bfgs_steps,
        force_max_tol=config.ts_validation_fmax,
        force_rms_tol=config.ts_validation_fmax * 0.5,
        early_stop_window=config.bfgs_steps + 1,  # disable early stopping
        hessian_update_interval=2,
        energy_increase_tol=0.005,
        verbose=False,
    )

    # --- Step 4: Post-process: pair up +/- results per TS ---
    # Group relaxation results by TS
    per_ts_results: list[list[Optional[tuple]]] = [
        [None, None] for _ in range(len(ts_data))
    ]

    for relax_idx, result in enumerate(relax_results):
        ts_data_idx, sign_idx = displaced_ts_map[relax_idx]
        td = ts_data[ts_data_idx]
        ts_energy = td["ts_energy"]

        # Check validity (same criteria as sequential)
        if result.trajectory.forces:
            final_forces = result.trajectory.forces[-1]
        else:
            # Fallback: recompute forces at final positions (matches sequential)
            fallback_atoms = Atoms(
                numbers=atomic_numbers, positions=result.positions.copy()
            )
            fallback_atoms.info["charge"] = charge
            fallback_atoms.calc = calc
            final_forces = fallback_atoms.get_forces()

        max_force = np.linalg.norm(final_forces, axis=1).max()
        is_valid = result.converged or max_force < config.ts_validation_fmax * 10
        energy_decreased = result.energy < ts_energy + 0.1

        if is_valid and energy_decreased:
            per_ts_results[ts_data_idx][sign_idx] = (
                result.positions.copy(),
                result.energy,
                RelaxationTrajectory(
                    positions=result.trajectory.positions,
                    energies=result.trajectory.energies,
                    forces=result.trajectory.forces,
                    hessians=result.trajectory.hessians,
                ),
            )
        else:
            reason = []
            if not is_valid:
                reason.append(f"max_force={max_force:.4f}")
            if not energy_decreased:
                reason.append(f"energy increased by {result.energy - ts_energy:.4f}")
            sign = +1 if sign_idx == 0 else -1
            logger.debug(
                f"TS {valid_ts_indices[ts_data_idx]}: relaxation rejected "
                f"({sign:+d} direction): {', '.join(reason)}"
            )

    # --- Step 5: Assemble validation results ---
    output: list[Optional[dict]] = [None] * len(ts_positions_list)

    for ts_data_idx, td in enumerate(ts_data):
        orig_idx = valid_ts_indices[ts_data_idx]
        plus_result = per_ts_results[ts_data_idx][0]
        minus_result = per_ts_results[ts_data_idx][1]

        if plus_result is None or minus_result is None:
            failed_dirs = []
            if plus_result is None:
                failed_dirs.append("+1")
            if minus_result is None:
                failed_dirs.append("-1")
            logger.debug(
                f"TS {orig_idx}: validation failed, missing direction(s): "
                f"{', '.join(failed_dirs)}"
            )
            continue

        endpoints = [
            {"positions": plus_result[0], "energy": plus_result[1]},
            {"positions": minus_result[0], "energy": minus_result[1]},
        ]
        trajectories = [plus_result[2], minus_result[2]]
        ts_energy = td["ts_energy"]

        # Energy check
        left_diff = endpoints[0]["energy"] - ts_energy
        right_diff = endpoints[1]["energy"] - ts_energy
        both_lower = (
            left_diff < -config.ts_validation_energy_threshold
            and right_diff < -config.ts_validation_energy_threshold
        )
        # RMSD check
        ep_rmsd = compute_rmsd(
            endpoints[0]["positions"], endpoints[1]["positions"], atomic_numbers
        )
        different = ep_rmsd > config.ts_validation_min_rmsd

        if not (both_lower and different):
            if not both_lower:
                logger.debug(
                    f"TS {orig_idx}: endpoints not sufficiently lower "
                    f"({left_diff:.4f}, {right_diff:.4f} eV)"
                )
            if not different:
                logger.debug(
                    f"TS {orig_idx}: endpoints too similar (RMSD={ep_rmsd:.4f} Å)"
                )
            continue

        # Convention: forward = downhill (fwd = higher energy reactant)
        if endpoints[0]["energy"] >= endpoints[1]["energy"]:
            fwd_idx, bwd_idx = 0, 1
        else:
            fwd_idx, bwd_idx = 1, 0

        def _last_hessian(traj: RelaxationTrajectory):
            for h in reversed(traj.hessians):
                if h is not None:
                    return h
            return None

        fwd_hessian = _last_hessian(trajectories[fwd_idx])
        bwd_hessian = _last_hessian(trajectories[bwd_idx])
        if fwd_hessian is None:
            logger.error(f"TS {orig_idx}: forward minimum missing hessian")
        if bwd_hessian is None:
            logger.error(f"TS {orig_idx}: backward minimum missing hessian")

        output[orig_idx] = {
            "ts_positions": td["ts_positions"],
            "ts_energy": ts_energy,
            "eigenvalue": td["eigenvalue"],
            "hessian": td["H"],
            "fwd_positions": endpoints[fwd_idx]["positions"],
            "fwd_energy": endpoints[fwd_idx]["energy"],
            "bwd_positions": endpoints[bwd_idx]["positions"],
            "bwd_energy": endpoints[bwd_idx]["energy"],
            "fwd_trajectory": trajectories[fwd_idx],
            "bwd_trajectory": trajectories[bwd_idx],
            "fwd_hessian": fwd_hessian,
            "bwd_hessian": bwd_hessian,
        }

    n_valid = sum(1 for v in output if v is not None)
    logger.debug(
        f"Batched TS validation: {n_valid}/{len(ts_positions_list)} passed"
    )
    return output


def _run_md_from_minimum(atoms: Atoms, calc: Calculator, config: ExploreConfig):
    """Run MD starting from a minimum, return trajectory."""
    atoms = atoms.copy()
    atoms.calc = calc

    md_opts = MDOpts(
        steps=config.md_steps,
        save_interval=config.md_save_interval,
        log_interval=config.md_log_interval,
        step_size_fs=config.md_step_size_fs,
        temperature=config.temperature,
        bgfs_steps=config.bfgs_steps,
        bgfs_fmax=config.bfgs_fmax,
        thermostat=config.thermostat,
        thermostat_tau=config.thermostat_tau,
    )

    collector = TrajectoryCollector(atoms)
    console_logger = ConsoleLogger(atoms, md_opts.log_interval)

    attachments = [
        (collector, md_opts.save_interval),
        (console_logger, md_opts.log_interval),
    ]

    run_md_simulation(atoms, calc, md_opts, attachments)

    return collector.trajectory, np.array(collector.energies)


def _sample_ts_candidates(
    trajectory, energies: np.ndarray, config: ExploreConfig
) -> list[int]:
    """Sample frames likely to be near transition states."""
    n_frames = len(trajectory)
    n_samples = min(config.n_ts_samples, n_frames)

    if n_samples == n_frames:
        return list(range(n_frames))

    if not config.use_saddle_metric:
        return np.random.choice(n_frames, size=n_samples, replace=False).tolist()

    forces_norms = np.array(
        [np.linalg.norm(atoms.get_forces(), axis=1).mean() for atoms in trajectory]
    )
    momenta_norms = np.array(
        [np.linalg.norm(atoms.get_momenta(), axis=1).mean() for atoms in trajectory]
    )

    energies_filt = _low_pass_filter(
        energies, config.filter_cutoff, config.filter_order
    )
    forces_filt = _low_pass_filter(
        forces_norms, config.filter_cutoff, config.filter_order
    )
    momenta_filt = _low_pass_filter(
        momenta_norms, config.filter_cutoff, config.filter_order
    )

    scores = _compute_saddle_metric(energies_filt, forces_filt, momenta_filt)

    sorted_idx = np.argsort(scores)
    n_top = max(n_samples, int(len(sorted_idx) * config.ts_top_percentile / 100))
    top_indices = sorted_idx[-n_top:]

    return np.random.choice(top_indices, size=n_samples, replace=False).tolist()


def _explore_from_minimum(
    minimum_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    calc: Calculator,
    config: ExploreConfig,
    pes_graph: PESGraph,
    ts_filter: Optional[Callable[[dict], bool]] = None,
    verbose: bool = True,
    charge: int = 0,
) -> tuple[int, list[dict], dict]:
    """Explore PES starting from a minimum: run MD, find TS, add to graph.

    Args:
        ts_filter: Optional filter for validated TSs. Called with the validation dict.
            Returns True to accept (add to PES graph), False to reject (collect as escaped).
            If None, all validations are accepted.
        charge: Net formal charge of the molecule.

    Returns:
        (n_new_ts, escaped_validations, timings): Number of new TSs added to PES graph,
        list of validation dicts rejected by the filter, and per-step timing dict.
    """
    atoms = Atoms(numbers=atomic_numbers, positions=minimum_positions.copy())
    atoms.info['charge'] = charge

    logger.info("Running MD simulation...")
    t_step = time.perf_counter()
    trajectory, energies = _run_md_from_minimum(atoms, calc, config)
    time_md_s = time.perf_counter() - t_step
    logger.info(f"Collected {len(trajectory)} frames")

    candidate_indices = _sample_ts_candidates(trajectory, energies, config)
    sampling_method = "saddle metric" if config.use_saddle_metric else "random"
    logger.info(f"Sampled {len(candidate_indices)} TS candidates ({sampling_method})")

    n_new_ts = 0
    escaped_validations: list[dict] = []
    total_neb_time = 0.0

    # Collect candidate atoms for batched TS optimization
    candidate_atoms_list: list[Atoms] = []
    for idx in candidate_indices:
        start_atoms = trajectory[idx].copy()
        start_atoms.info["charge"] = charge
        candidate_atoms_list.append(start_atoms)

    logger.info(f"Running batched TS optimization on {len(candidate_atoms_list)} candidates...")
    t_step = time.perf_counter()
    results = optimize_saddle_points_batched(
        calc,
        candidate_atoms_list,
        max_steps=config.ts_max_steps,
        force_max_tol=config.ts_force_tol,
        trust_radius=config.ts_trust_radius,
        verbose=verbose,
        eigenvalue_zero_threshold=config.eigenvalue_zero_threshold,
    )
    time_prfo_s = time.perf_counter() - t_step

    n_converged = sum(1 for r in results if r.converged and r.is_first_order)
    logger.info(f"Batched TS optimization: {n_converged}/{len(results)} converged to first-order saddle")

    # Collect converged TS positions for batched validation
    converged_positions: list[np.ndarray] = []
    converged_indices: list[int] = []  # Index into results list
    for i, result in enumerate(results):
        if result.converged and result.is_first_order:
            converged_positions.append(result.positions)
            converged_indices.append(i)
        else:
            logger.debug(
                f"TS optimization from frame {candidate_indices[i]} did not converge to first-order saddle point"
            )

    # Batched TS validation
    t_step = time.perf_counter()
    if converged_positions:
        logger.info(f"Running batched TS validation on {len(converged_positions)} converged TSes...")
        validations = _validate_ts_batched(
            calc, converged_positions, atomic_numbers, config, charge=charge
        )
    else:
        validations = []
    time_validation_s = time.perf_counter() - t_step

    # Separate accepted vs escaped validations
    accepted_validations = []
    for j, validation in enumerate(validations):
        if validation is None:
            continue
        if ts_filter is not None and not ts_filter(validation):
            logger.info("Valid TS rejected by filter, collecting as escaped reaction")
            escaped_validations.append(validation)
            continue
        accepted_validations.append(validation)

    # Batch-add all endpoint minima (cross-checks new endpoints against each other)
    if accepted_validations:
        endpoint_items = []
        for v in accepted_validations:
            endpoint_items.append((v["fwd_positions"], v["fwd_energy"], v.get("fwd_hessian")))
            endpoint_items.append((v["bwd_positions"], v["bwd_energy"], v.get("bwd_hessian")))

        logger.info(
            f"Batch-adding {len(endpoint_items)} endpoint minima "
            f"from {len(accepted_validations)} TSes..."
        )
        endpoint_results, neb_time = pes_graph.add_minima_batched(endpoint_items)
        total_neb_time += neb_time

        # Add each TS with pre-resolved endpoints
        for j, validation in enumerate(accepted_validations):
            fwd_min_id, _, fwd_pos = endpoint_results[2 * j]
            bwd_min_id, _, bwd_pos = endpoint_results[2 * j + 1]

            ts_id, is_new, ts_neb_time = pes_graph.add_transition_state(
                positions=validation["ts_positions"],
                energy=validation["ts_energy"],
                min_fwd_positions=validation["fwd_positions"],
                min_fwd_energy=validation["fwd_energy"],
                min_bwd_positions=validation["bwd_positions"],
                min_bwd_energy=validation["bwd_energy"],
                eigenvalue=validation["eigenvalue"],
                hessian=validation["hessian"],
                fwd_trajectory=validation["fwd_trajectory"],
                bwd_trajectory=validation["bwd_trajectory"],
                min_fwd_hessian=validation.get("fwd_hessian"),
                min_bwd_hessian=validation.get("bwd_hessian"),
                _preresolved_fwd=(fwd_min_id, fwd_pos),
                _preresolved_bwd=(bwd_min_id, bwd_pos),
            )
            total_neb_time += ts_neb_time

            if is_new:
                n_new_ts += 1
                logger.debug(f"Added new TS {ts_id}")
            else:
                logger.debug(f"TS {ts_id} already exists in graph, skipping addition.")

    timings = {
        "pes_md_s": time_md_s,
        "pes_prfo_s": time_prfo_s,
        "pes_validation_s": time_validation_s,
        "pes_neb_dedup_s": total_neb_time,
    }
    return n_new_ts, escaped_validations, timings


def _relax_structure(
    atoms: Atoms, calc: Calculator, config: ExploreConfig
) -> tuple[np.ndarray, float] | None:
    """Relax a structure to a local minimum. Returns None if optimization failed."""
    result = optimize_minimum(
        calc,
        atoms,
        max_steps=config.bfgs_steps,
        force_max_tol=config.bfgs_fmax,
        force_rms_tol=config.bfgs_fmax * 0.5,
        verbose=False,
    )
    # Accept if converged OR forces within 10x tolerance (including early-stopped)
    is_valid = result.converged or result.final_force_max < config.bfgs_fmax * 10
    if not is_valid:
        logger.debug(
            f"Relaxation failed (force_max={result.final_force_max:.4f} eV/Å, "
            f"converged={result.converged}, early_stopped={result.early_stopped})"
        )
        return None
    return result.positions.copy(), result.energy


# -----------------------------------------------------------------------------
# Main interface function
# -----------------------------------------------------------------------------


def explore_pes(
    atoms: Atoms,
    calc: Calculator,
    config: Optional[ExploreConfig] = None,
    existing_graph: Optional[PESGraph] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    cache_name: Optional[str] = None,
    use_cache: bool = True,
    verbose: bool = True,
    seed: int = 42,
    charge: int = 0,
) -> PESGraph:
    """
    Explore the potential energy surface of a molecule.

    This function iteratively:
    1. Runs MD simulations from known minima
    2. Identifies and validates transition states from high-energy frames
    3. Builds a graph of minima (nodes) and transition states (edges)

    The exploration continues until all minima have been explored or
    max_iterations is reached.

    Args:
        atoms: Initial molecular structure (ASE Atoms object)
        calc: ASE Calculator with get_hessian support (e.g., from lib.md_et_calculator).
              Must be instantiated externally.
        config: Exploration configuration. If None, uses defaults.
        existing_graph: Optional existing PESGraph to continue exploration from.
                       The graph's thresholds will be updated to match config.
        cache_dir: Directory for caching results. If None and use_cache=True,
                  uses "./pes_cache".
        cache_name: Optional name suffix for cache file (e.g., model name).
                   If None, only molecule hash is used.
        use_cache: Whether to use disk caching. If True and a cached graph exists,
                  it will be loaded and exploration continued from there.
        verbose: Whether to show progress bars and detailed logging.
        seed: Random seed for reproducibility.

    Returns:
        PESGraph containing all discovered minima and transition states.

    Example:
        >>> from ase.build import molecule
        >>> from lib.pes_explorer.pes_explorer import explore_pes, ExploreConfig
        >>> from lib.md_et_calculator import get_md_et_calculator
        >>>
        >>> # Create calculator externally
        >>> calc = get_md_et_calculator(run_dir, device="cuda")
        >>>
        >>> # Basic usage
        >>> atoms = molecule("C2H5OH")
        >>> graph = explore_pes(atoms, calc)
        >>>
        >>> # With custom settings
        >>> config = ExploreConfig(md_steps=100_000, temperature=500.0, max_iterations=50)
        >>> graph = explore_pes(atoms, calc, config=config, cache_dir="my_results")
        >>>
        >>> # Continue from existing graph
        >>> graph = explore_pes(atoms, calc, existing_graph=partial_graph)
        >>>
        >>> # With cache name to distinguish different models
        >>> graph = explore_pes(atoms, calc, cache_name="mace_mh1")
        >>>
        >>> # Access results
        >>> print(f"Found {len(graph.minima)} minima, {len(graph.transition_states)} TS")
        >>> graph.export_gexf("pes.gexf")
    """
    # Set random seed
    np.random.seed(seed)
    th.manual_seed(seed)

    # Initialize config
    if config is None:
        config = ExploreConfig()

    # Setup cache directory
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
    elif use_cache:
        cache_dir = Path("./pes_cache")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = _get_cache_path(atoms, cache_dir, cache_name)
    else:
        cache_path = None

    atomic_numbers = atoms.get_atomic_numbers()
    atoms.info['charge'] = charge

    # Try to load from cache or existing graph
    pes_graph = existing_graph
    if pes_graph is not None:
        # Update thresholds and calc on existing graph
        pes_graph.rmsd_threshold = config.rmsd_threshold
        pes_graph.energy_threshold = config.energy_threshold
        pes_graph.calc = calc
        pes_graph.neb_barrier_threshold = config.neb_barrier_threshold
        pes_graph.neb_n_images = config.neb_n_images
        pes_graph.neb_fire_steps = config.neb_fire_steps
        pes_graph.neb_spring_constant = config.neb_spring_constant
    if (
        pes_graph is None
        and use_cache
        and cache_path is not None
        and cache_path.exists()
    ):
        logger.info(f"Loading cached PES graph from {cache_path}")
        pes_graph = PESGraph.load(cache_path)
        # Update thresholds and calc to match current config
        pes_graph.rmsd_threshold = config.rmsd_threshold
        pes_graph.energy_threshold = config.energy_threshold
        pes_graph.calc = calc
        pes_graph.neb_barrier_threshold = config.neb_barrier_threshold
        pes_graph.neb_n_images = config.neb_n_images
        pes_graph.neb_fire_steps = config.neb_fire_steps
        pes_graph.neb_spring_constant = config.neb_spring_constant
        stats = pes_graph.get_stats()
        logger.info(
            f"Loaded graph with {stats['n_minima']} minima and {stats['n_ts']} transition states"
        )

    # Initialize new graph if needed
    if pes_graph is None:
        logger.info("Relaxing initial structure...")
        relax_result = _relax_structure(atoms, calc, config)
        if relax_result is None:
            raise ValueError(f"Initial structure relaxation failed for {len(atomic_numbers)}-atom molecule")
        initial_positions, initial_energy = relax_result

        pes_graph = PESGraph(
            atomic_numbers=atomic_numbers,
            rmsd_threshold=config.rmsd_threshold,
            energy_threshold=config.energy_threshold,
            calc=calc,
            neb_barrier_threshold=config.neb_barrier_threshold,
            neb_n_images=config.neb_n_images,
            neb_fire_steps=config.neb_fire_steps,
            neb_spring_constant=config.neb_spring_constant,
        )

        min_id, _, _, _ = pes_graph.add_minimum(initial_positions, initial_energy)
        logger.info(f"Added initial minimum with energy {initial_energy:.4f} eV")

    # Exploration loop
    for iteration in range(config.max_iterations):
        unexplored = pes_graph.get_unexplored_minima()

        if not unexplored:
            logger.info("All minima explored, stopping")
            break

        # Pick minimum to explore (lowest energy unexplored)
        unexplored_sorted = sorted(unexplored, key=lambda m: m.energy)
        minimum = unexplored_sorted[0]

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Iteration {iteration + 1}/{config.max_iterations}")
        logger.info(f"Exploring minimum {minimum.id} (E = {minimum.energy:.4f} eV)")
        logger.info(f"Graph: {pes_graph}")
        logger.info(f"{'=' * 60}")

        n_new_ts, _, _ = _explore_from_minimum(
            minimum.positions,
            atomic_numbers,
            calc,
            config,
            pes_graph,
            verbose=verbose,
            charge=charge,
        )

        pes_graph.mark_explored(minimum.id)

        logger.info(f"Found {n_new_ts} new transition states")
        logger.info(f"Graph stats: {pes_graph.get_stats()}")

        # Save periodically
        if cache_path is not None and (iteration + 1) % config.save_interval == 0:
            pes_graph.save(cache_path)
            logger.debug(f"Saved checkpoint to {cache_path}")

    # Final save
    if cache_path is not None:
        pes_graph.save(cache_path)
        logger.info(f"Saved final PES graph to {cache_path}")

    # Log summary
    stats = pes_graph.get_stats()
    logger.info("\n" + "=" * 60)
    logger.info("EXPLORATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Minima found: {stats['n_minima']}")
    logger.info(f"Transition states found: {stats['n_ts']}")
    logger.info(f"Connected components: {stats['n_connected_components']}")

    if pes_graph.minima:
        energies = [m.energy for m in pes_graph.minima.values()]
        logger.info(f"Energy range: [{min(energies):.4f}, {max(energies):.4f}] eV")

    return pes_graph
