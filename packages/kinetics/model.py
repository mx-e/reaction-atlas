"""Build an ODE model from the database.

Reads Reaction rows + reactant/product joins, dedupes by (reactant_set,
product_set), applies barrier-mode + DFT-preference policy to choose which
barrier feeds Eyring, layers in the manual buffer equilibria using their
literal rate constants, and packs everything into an ODEModel ready for
either the PETSc solver or the SBML exporter.

The model builder has NO numba or PETSc dependency — it's safe to import
from anywhere (API container, SBML route, etc.).
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from packages.db.models import (
    Compound,
    Minimum,
    Reaction,
    ReactionProduct,
    ReactionReactant,
)
from packages.kinetics.initial_conditions import (
    DEFAULT_INITIAL_CONCS,
    UNIFORM_INITIAL_CONC_M,
    DIFFUSION_LIMIT_M_PER_S,
    H_EV_S,
    KB_EV,
)


@dataclass
class ODEModel:
    """Self-contained mass-action ODE problem ready for the PETSc solver."""

    smiles_list: list[str]
    n_species: int
    n_reactions: int  # number of mass-action reactions (incl. manual equilibria)
    n_reactions_discovered: int  # subset that came from discovered (non-manual) Reaction rows
    n_reactions_dft: int  # subset that used PBE0 separated barriers (the rest fell back to ML)
    n_manual_equilibria: int

    # Stoichiometry-flattened arrays (one row per reaction). Each reaction's
    # reactants/products are described by lists of (species_idx, count) pairs.
    k_fwd: np.ndarray              # shape (n_reactions,)
    k_bwd: np.ndarray              # shape (n_reactions,)
    reactant_stoich: list[list[tuple[int, int]]]  # per-reaction list of (idx, stoich)
    product_stoich: list[list[tuple[int, int]]]

    initial_concs: np.ndarray      # shape (n_species,)

    # Diagnostic info per reaction (used by SBML export annotations)
    barrier_sources: list[str]     # 'dft', 'ml', 'manual' per reaction
    barrier_forwards_eV: list[Optional[float]]   # input barrier (None for manual)
    barrier_backwards_eV: list[Optional[float]]
    reaction_names: list[str]      # for SBML id generation


def _eyring_rate(barrier_eV: float, temperature_K: float) -> float:
    """k = (kBT/h) * exp(-Ea/kBT), capped at the diffusion limit."""
    kBT = KB_EV * temperature_K
    prefactor = kBT / H_EV_S
    # Cap input barrier at 10 eV to avoid overflow in exp(-100)
    barrier = max(0.0, min(barrier_eV, 10.0))
    rate = prefactor * math.exp(-barrier / kBT)
    return min(rate, DIFFUSION_LIMIT_M_PER_S)


def _resolve_kinetics_barriers(
    rxn: Reaction, prefer_dft: bool
) -> tuple[Optional[float], Optional[float], str]:
    """Pick the best (forward, backward) barriers for the kinetics solver.

    Priority order (with prefer_dft=True, the default):
      1. ``barrier_*_separated_pbe0``  — DFT separated reference (best, when
         the cpu-worker has refined the reaction with PBE0)
      2. ``barrier_*`` (in-box)        — ML force-integrated barrier along the
         IRC trajectory. The ML force head is more reliable than the energy
         head, so trapezoidal force integration gives a better estimate than
         the ML separated barrier (which uses energy-head differences between
         two completely different geometries).

    With prefer_dft=False the order swaps to ML in-box → DFT separated.

    Returns (bf, bb, source) where source ∈ {'dft_separated', 'ml_inbox', 'none'}.
    Reactions where neither source is available are dropped (do NOT silently
    fall back to ML separated — that uses energy-head differences and is the
    least reliable variant).
    """
    dft_bf = rxn.barrier_forward_separated_pbe0
    dft_bb = rxn.barrier_backward_separated_pbe0
    dft_ok = dft_bf is not None and dft_bb is not None

    ml_bf = rxn.barrier_forward
    ml_bb = rxn.barrier_backward
    ml_ok = ml_bf is not None and ml_bb is not None

    if prefer_dft:
        if dft_ok:
            return dft_bf, dft_bb, "dft_separated"
        if ml_ok:
            return ml_bf, ml_bb, "ml_inbox"
    else:
        if ml_ok:
            return ml_bf, ml_bb, "ml_inbox"
        if dft_ok:
            return dft_bf, dft_bb, "dft_separated"
    return None, None, "none"


def build_model_from_db(
    session: Session,
    temperature: float,
    prefer_dft: bool = True,
    barrier_cutoff_eV: float = 10.0,
    initial_concs_override: Optional[dict[str, float]] = None,
    experiment: Optional[str] = None,
) -> ODEModel:
    """Read all reactions from the DB and assemble an ODEModel.

    Args:
        session: SQLAlchemy session
        temperature: Kelvin — sets the Eyring kBT for k computation
        prefer_dft: if True (default), use barrier_*_separated_pbe0 over ML
        barrier_cutoff_eV: skip reactions where BOTH fwd and bwd separated
            barriers exceed this (effectively unreachable, just bloats the
            ODE system)
        initial_concs_override: optional dict to merge over DEFAULT_INITIAL_CONCS
        experiment: if set, only reactions tagged with this experiment are
            included (Reaction.experiments @> ARRAY[experiment]). When None,
            all reactions across experiments are included — used by the SBML
            exporter and by ad-hoc tooling that wants the global view.

    Returns:
        ODEModel ready to feed into petsc_solver or sbml exporter.
    """
    # ----- Species universe -----
    # Collect all SMILES that participate in any non-zero reaction we're
    # going to keep. We'll do this in two passes: pass 1 selects/dedupes
    # reactions, pass 2 builds the species index from those reactions only.
    #
    # CRITICAL: query ONLY the columns the solver actually reads. The Reaction
    # table has several LargeBinary columns (ts_conformer_positions,
    # reactant_trajectory, product_trajectory) which are multi-MB per row;
    # session.query(Reaction).all() pulled gigabytes of trajectory blobs over
    # the wire and routinely tripped uvicorn's 30 s pong timeout, killing
    # workers mid-solve.
    import time as _time
    from loguru import logger as _logger
    _t0 = _time.perf_counter()
    rxn_q = session.query(
        Reaction.id,
        Reaction.discovery_method,
        Reaction.manual_k_fwd,
        Reaction.manual_k_bwd,
        Reaction.barrier_forward,
        Reaction.barrier_backward,
        Reaction.barrier_forward_separated_pbe0,
        Reaction.barrier_backward_separated_pbe0,
        Reaction.name,
    )
    if experiment is not None:
        rxn_q = rxn_q.filter(Reaction.experiments.any(experiment))
    rxn_cols = rxn_q.all()
    _logger.info(f"build_model: loaded {len(rxn_cols)} reaction rows in {_time.perf_counter()-_t0:.1f}s")

    # Pre-fetch reactant/product joins for all reactions in two queries
    # (avoids N+1).
    rxn_ids = [r.id for r in rxn_cols]
    if not rxn_ids:
        return ODEModel(
            smiles_list=[], n_species=0, n_reactions=0,
            n_reactions_discovered=0, n_reactions_dft=0, n_manual_equilibria=0,
            k_fwd=np.zeros(0), k_bwd=np.zeros(0),
            reactant_stoich=[], product_stoich=[],
            initial_concs=np.zeros(0),
            barrier_sources=[], barrier_forwards_eV=[], barrier_backwards_eV=[],
            reaction_names=[],
        )

    _t1 = _time.perf_counter()
    reactant_rows = session.query(ReactionReactant).filter(
        ReactionReactant.reaction_id.in_(rxn_ids)
    ).all()
    product_rows = session.query(ReactionProduct).filter(
        ReactionProduct.reaction_id.in_(rxn_ids)
    ).all()
    _logger.info(f"build_model: loaded {len(reactant_rows)} reactant + {len(product_rows)} product rows in {_time.perf_counter()-_t1:.1f}s")

    # compound_id → smiles map for the compounds touched by these joins
    touched_compound_ids = {r.compound_id for r in reactant_rows} | {p.compound_id for p in product_rows}
    if touched_compound_ids:
        compound_rows = session.query(Compound).filter(
            Compound.id.in_(touched_compound_ids)
        ).all()
    else:
        compound_rows = []
    compound_id_to_smiles = {c.id: c.smiles for c in compound_rows}

    # rxn_id → list of reactant compound_ids (with multiplicity from join rows)
    rxn_reactants: dict[int, list[int]] = {}
    rxn_products: dict[int, list[int]] = {}
    for rr in reactant_rows:
        rxn_reactants.setdefault(rr.reaction_id, []).append(rr.compound_id)
    for rp in product_rows:
        rxn_products.setdefault(rp.reaction_id, []).append(rp.compound_id)

    # ----- Pass 1: select discovered reactions, dedupe, layer in equilibria -----
    selected = []  # list of dicts: {rxn, kf, kb, source, reactant_smiles_counter, product_smiles_counter}
    n_reactions_dft = 0
    n_manual = 0

    # Track best (lowest min(bf, bb)) per (sorted reactant tuple, sorted product tuple)
    best_by_pair: dict[tuple, dict] = {}

    for rxn in rxn_cols:
        is_manual = rxn.discovery_method == "manual_equilibrium"
        r_ids = rxn_reactants.get(rxn.id, [])
        p_ids = rxn_products.get(rxn.id, [])
        if not r_ids or not p_ids:
            continue

        r_smiles = [compound_id_to_smiles.get(cid) for cid in r_ids]
        p_smiles = [compound_id_to_smiles.get(cid) for cid in p_ids]
        if any(s is None for s in r_smiles + p_smiles):
            continue
        r_counter = Counter(r_smiles)
        p_counter = Counter(p_smiles)

        if is_manual:
            # Manual equilibria use literal rate constants — no Eyring, no T dependence.
            if rxn.manual_k_fwd is None or rxn.manual_k_bwd is None:
                continue
            kf = rxn.manual_k_fwd
            kb = rxn.manual_k_bwd
            source = "manual"
            bf_eV = None
            bb_eV = None
            # Don't dedupe equilibria — they're all distinct by purpose.
            selected.append({
                "rxn": rxn,
                "kf": kf, "kb": kb, "source": source,
                "reactant_counter": r_counter, "product_counter": p_counter,
                "bf_eV": bf_eV, "bb_eV": bb_eV,
            })
            n_manual += 1
            continue

        # Discovered reaction — pick the best barrier source available.
        # Priority: DFT separated → ML in-box (force-integrated). Drops the
        # reaction if neither is available. See _resolve_kinetics_barriers
        # for the rationale (ML separated uses energy-head differences and
        # is the least reliable variant).
        bf_eV, bb_eV, source = _resolve_kinetics_barriers(rxn, prefer_dft)
        if bf_eV is None or bb_eV is None:
            continue  # no usable barriers yet — drop entirely
        # Drop encounter-complex reactions: dot-disconnected SMILES are van der
        # Waals dimers stabilized by dispersion, not chemical reactions. They
        # appear with negative Ea (dimer below separated reactants) and would
        # run at the diffusion limit after clamping, dragging connected species
        # into spurious fast equilibria.
        if any('.' in s for s in r_smiles + p_smiles):
            continue
        # Drop negative-Ea artifacts. PES/IRC sometimes lands on dianion or
        # zwitterion minima with energies far below the separated reactant
        # pair, producing unphysical Ea < 0. Tolerate small numerical noise
        # (-0.05 eV); reject anything more negative — these would otherwise
        # be clamped to the diffusion limit and distort steady state.
        if bf_eV < -0.05 or bb_eV < -0.05:
            continue
        if bf_eV > barrier_cutoff_eV and bb_eV > barrier_cutoff_eV:
            continue

        # Direction-agnostic dedupe key: {reactant_multiset, product_multiset}
        # as a frozenset, so A+B → C and its reverse C → A+B collapse into a
        # single reaction with the lowest barrier.  Storing both as separate
        # mass-action reactions double-counts the net flux for the same
        # elementary step and inflates the ODE Jacobian unnecessarily —
        # solver wall time scales with reaction count on stiff systems, so
        # direction-agnostic dedup is both physically correct and a
        # significant perf win on networks with lots of discovered reverse
        # pairs.
        rkey = tuple(sorted(r_counter.items()))
        pkey = tuple(sorted(p_counter.items()))
        pair_key = frozenset((rkey, pkey))
        candidate_min = min(bf_eV, bb_eV)
        if pair_key in best_by_pair and best_by_pair[pair_key]["min_barrier"] <= candidate_min:
            continue

        kf = _eyring_rate(bf_eV, temperature)
        kb = _eyring_rate(bb_eV, temperature)
        best_by_pair[pair_key] = {
            "rxn": rxn,
            "kf": kf, "kb": kb, "source": source,
            "reactant_counter": r_counter, "product_counter": p_counter,
            "bf_eV": bf_eV, "bb_eV": bb_eV,
            "min_barrier": candidate_min,
        }

    # Promote best_by_pair into selected (after manual reactions)
    discovered = list(best_by_pair.values())
    for entry in discovered:
        if entry["source"] == "dft_separated":
            n_reactions_dft += 1
        # Strip the dedup-internal field before storing
        entry.pop("min_barrier", None)
    # Manual equilibria first, then discovered (order doesn't affect the math
    # but is convenient for SBML output where you want the buffers grouped).
    selected = [s for s in selected if s["source"] == "manual"] + discovered

    _logger.info(f"build_model: pass 1 done — selected {len(selected)} reactions (manual={n_manual}, dft={n_reactions_dft})")
    # ----- Pass 2: build species index from selected reactions -----
    species_set = set()
    # Always include species that have a default initial concentration so the
    # solver injects them into the system even if no reaction touches them
    # (avoids cold-start zero-out for water/CH2O/CO2).
    species_set.update(DEFAULT_INITIAL_CONCS.keys())
    for entry in selected:
        species_set.update(entry["reactant_counter"].keys())
        species_set.update(entry["product_counter"].keys())

    smiles_list = sorted(species_set)
    smiles_to_idx = {s: i for i, s in enumerate(smiles_list)}
    n_species = len(smiles_list)

    # Initial concentrations: uniform baseline for every species (including
    # runtime-discovered ones), then overlay the seed-specific overrides
    # from DEFAULT_INITIAL_CONCS and the caller-supplied override dict.
    initial_concs = np.full(n_species, UNIFORM_INITIAL_CONC_M, dtype=np.float64)
    concs_map = dict(DEFAULT_INITIAL_CONCS)
    if initial_concs_override:
        concs_map.update(initial_concs_override)
    for smi, c in concs_map.items():
        if smi in smiles_to_idx:
            initial_concs[smiles_to_idx[smi]] = c

    # Build the per-reaction stoich lists
    n_reactions = len(selected)
    k_fwd_arr = np.zeros(n_reactions)
    k_bwd_arr = np.zeros(n_reactions)
    reactant_stoich: list[list[tuple[int, int]]] = []
    product_stoich: list[list[tuple[int, int]]] = []
    barrier_sources: list[str] = []
    barrier_forwards_eV: list[Optional[float]] = []
    barrier_backwards_eV: list[Optional[float]] = []
    reaction_names: list[str] = []

    for j, entry in enumerate(selected):
        k_fwd_arr[j] = entry["kf"]
        k_bwd_arr[j] = entry["kb"]
        reactant_stoich.append([
            (smiles_to_idx[smi], int(count)) for smi, count in entry["reactant_counter"].items()
        ])
        product_stoich.append([
            (smiles_to_idx[smi], int(count)) for smi, count in entry["product_counter"].items()
        ])
        barrier_sources.append(entry["source"])
        barrier_forwards_eV.append(entry["bf_eV"])
        barrier_backwards_eV.append(entry["bb_eV"])
        reaction_names.append(entry["rxn"].name or f"rxn_{entry['rxn'].id}")

    _logger.info(f"build_model: pass 2 done — n_species={n_species} n_reactions={n_reactions} total_time={_time.perf_counter()-_t0:.1f}s")
    return ODEModel(
        smiles_list=smiles_list,
        n_species=n_species,
        n_reactions=n_reactions,
        n_reactions_discovered=len(discovered),
        n_reactions_dft=n_reactions_dft,
        n_manual_equilibria=n_manual,
        k_fwd=k_fwd_arr,
        k_bwd=k_bwd_arr,
        reactant_stoich=reactant_stoich,
        product_stoich=product_stoich,
        initial_concs=initial_concs,
        barrier_sources=barrier_sources,
        barrier_forwards_eV=barrier_forwards_eV,
        barrier_backwards_eV=barrier_backwards_eV,
        reaction_names=reaction_names,
    )
