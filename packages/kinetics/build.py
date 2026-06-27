"""Glue: solve a model and convert the result into a KineticsSnapshot.

This is what the API loop.py calls synchronously in a thread pool. The heavy
work happens in petsc_solver.solve_ode (PETSc BDF) and the light-weight
post-processing here builds the JSON-ready snapshot from the solution.
"""

import math
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from packages.kinetics.constants import DECADE_EXPONENTS, DECADE_TIMES
from packages.kinetics.initial_conditions import NOISE_FLOOR_M
from packages.kinetics.model import build_model_from_db, ODEModel
from packages.kinetics.snapshot import KineticsSnapshot


def build_snapshot(
    session: Session,
    temperature: float,
    prefer_dft: bool = True,
    t_max: float = 1e8,
    n_plot_points: int = 200,
    experiment: Optional[str] = None,
) -> Optional[KineticsSnapshot]:
    """Build + solve + snapshot. Returns None if the model is empty.

    Side-effect-free w.r.t. the database — the caller is responsible for
    persisting the returned snapshot into kinetics_snapshots.

    `experiment` scopes the reaction set; passes through to
    build_model_from_db.
    """
    t0 = time.time()

    model = build_model_from_db(
        session,
        temperature=temperature,
        prefer_dft=prefer_dft,
        experiment=experiment,
    )
    if model.n_reactions == 0:
        logger.info("build_snapshot: model has 0 reactions, skipping solve")
        return None

    logger.info(
        f"build_snapshot: solving {model.n_reactions} reactions "
        f"({model.n_reactions_dft} DFT, {model.n_manual_equilibria} equilibria, "
        f"{model.n_species} species) at T={temperature}K"
    )

    # Import scipy solver lazily so model-only callers (e.g. SBML export)
    # don't need to pull in numba + scipy eagerly at import time.
    from packages.kinetics.scipy_solver import solve_ode
    sol_fn = solve_ode(model, t_max=t_max)

    # Build the continuous time series for the frontend plot. Include EVERY
    # species in the model — even ones whose concentration stays at zero
    # throughout the simulation. The frontend hides negligible traces
    # visually, but the data is there if the user wants to inspect them.
    plot_times = np.geomspace(1e-12, t_max, n_plot_points)
    # Evaluate concentrations at each time point. Round to 6 significant
    # figures to cut JSON serialization size (~40% smaller than full float64
    # repr) without visible loss on the frontend plot.
    def _sig6(x: float) -> float:
        if x == 0.0:
            return 0.0
        from math import log10, floor, isfinite
        if not isfinite(x):
            return 0.0
        return round(x, -int(floor(log10(abs(x)))) + 5)

    plot_concs: dict[str, list[float]] = {smi: [] for smi in model.smiles_list}
    for t_pt in plot_times:
        y = sol_fn(t_pt)
        for i, smi in enumerate(model.smiles_list):
            plot_concs[smi].append(_sig6(float(y[i])))
    plot_times = [_sig6(t) for t in plot_times.tolist()]
    filtered_concs = plot_concs

    # Build per-decade distributions (kept for frontend display / backward compat)
    decade_distributions: list[dict[str, float]] = []
    for decade_t in DECADE_TIMES:
        y = sol_fn(decade_t)
        above_mask = y > NOISE_FLOOR_M
        n_active = int(above_mask.sum())
        if n_active < 2:
            decade_distributions.append({})
            continue

        active_y = np.maximum(y[above_mask], NOISE_FLOOR_M)
        log_c = np.log10(active_y)
        log_c_shifted = log_c - log_c.min() + 1.0
        weights = log_c_shifted / log_c_shifted.sum()

        active_idx = np.where(above_mask)[0]
        dist = {
            model.smiles_list[idx]: round(float(w), 6)
            for idx, w in zip(active_idx, weights)
        }
        decade_distributions.append(dist)

    # Build steady-state sampling distribution: softmax(1 * log10(conc))
    # evaluated at the final time point (t_max). Species below the noise
    # floor are excluded.
    y_final = sol_fn(t_max)
    above_mask_ss = y_final > NOISE_FLOOR_M
    steady_state_distribution: dict[str, float] = {}
    steady_state_log_concs: dict[str, float] = {}
    if int(above_mask_ss.sum()) >= 2:
        active_y_ss = np.maximum(y_final[above_mask_ss], NOISE_FLOOR_M)
        log_c_ss = np.log10(active_y_ss)
        # softmax(1 * log(conc)) — the temperature parameter is 1
        log_c_shifted_ss = log_c_ss - log_c_ss.max()  # shift for numerical stability
        exp_weights = np.exp(log_c_shifted_ss)
        softmax_weights = exp_weights / exp_weights.sum()

        active_idx_ss = np.where(above_mask_ss)[0]
        for idx, w, lc in zip(active_idx_ss, softmax_weights, log_c_ss):
            smi = model.smiles_list[idx]
            steady_state_distribution[smi] = round(float(w), 8)
            steady_state_log_concs[smi] = round(float(lc), 4)

    # Build per-reaction summary for the frontend reaction table
    reactions_summary: list[dict] = []
    for j in range(model.n_reactions):
        reactions_summary.append({
            "name": model.reaction_names[j],
            "reactants": [
                {"smiles": model.smiles_list[idx], "count": count}
                for idx, count in model.reactant_stoich[j]
            ],
            "products": [
                {"smiles": model.smiles_list[idx], "count": count}
                for idx, count in model.product_stoich[j]
            ],
            "barrier_fwd_eV": model.barrier_forwards_eV[j],
            "barrier_bwd_eV": model.barrier_backwards_eV[j],
            "source": model.barrier_sources[j],
        })

    # Initial concentrations (non-zero entries only)
    initial_concs_dict = {
        model.smiles_list[i]: round(float(model.initial_concs[i]), 6)
        for i in range(model.n_species)
        if model.initial_concs[i] > 0
    }

    # Pre-compute Shannon entropy per decade
    decade_entropies: list[float] = []
    for dist in decade_distributions:
        total = sum(dist.values())
        if total <= 0:
            decade_entropies.append(0.0)
            continue
        h = 0.0
        for w in dist.values():
            if w > 0:
                p = w / total
                h -= p * math.log2(p)
        decade_entropies.append(round(h, 3))

    solve_wall = time.time() - t0
    snapshot = KineticsSnapshot(
        smiles_list=list(model.smiles_list),
        times=plot_times,
        concentrations=filtered_concs,
        decade_times=list(DECADE_TIMES),
        decade_exponents=list(DECADE_EXPONENTS),
        decade_distributions=decade_distributions,
        temperature=temperature,
        n_species=model.n_species,
        n_reactions=model.n_reactions,
        n_reactions_dft=model.n_reactions_dft,
        n_manual_equilibria=model.n_manual_equilibria,
        reactions_summary=reactions_summary,
        initial_concentrations=initial_concs_dict,
        decade_entropies=decade_entropies,
        steady_state_distribution=steady_state_distribution,
        steady_state_log_concs=steady_state_log_concs,
        solve_wall_time_s=solve_wall,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )
    n_sampleable = sum(1 for d in decade_distributions if len(d) >= 2)
    logger.info(
        f"build_snapshot: solved in {solve_wall:.2f}s, "
        f"{len(filtered_concs)} species above noise, "
        f"{n_sampleable}/{len(decade_distributions)} decades sampleable"
    )
    return snapshot
