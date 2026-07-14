#!/usr/bin/env python3
"""ReactionAtlas kinetics demo — self-contained, CPU-only, no GPU/PostgreSQL.

Loads a small *real* early-exploration reaction network (extracted from a
checkpoint of the published run — 64 compounds, ~80 reactions, ~8 KB) and runs
the **actual production kinetics pipeline** on it:

    packages.kinetics.build.build_snapshot
        -> packages.kinetics.model.build_model_from_db   (barrier policy + Eyring)
        -> packages.kinetics.scipy_solver.solve_ode       (numba RHS/Jacobian + scipy BDF)

The network is loaded into an in-memory SQLite database so that `build_snapshot`
sees exactly the same SQLAlchemy schema it uses in production; nothing about the
solver or the model builder is re-implemented here.

Output:
  - a deterministic text summary of the solved steady-state distribution
    (printed to stdout; compare against expected_output.txt)
  - concentrations.png : concentration-vs-time trajectories (log-log)

Run from anywhere:
    python demo/kinetics/run_demo.py
    # or:  uv run python demo/kinetics/run_demo.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# --- make the repo-root `packages` importable regardless of CWD ---
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from packages.db.models import (
    Base,
    Compound,
    Reaction,
    ReactionReactant,
    ReactionProduct,
)
from packages.kinetics.build import build_snapshot

NETWORK_NPZ = Path(__file__).resolve().parent / "data" / "early_network.npz"
PLOT_PATH = Path(__file__).resolve().parent / "concentrations.png"

TEMPERATURE_K = 500.0   # matches the exploration temperature used in the paper
T_MAX_S = 1e8           # integrate out to ~3 years (well past steady state)


def _dummy_blob() -> bytes:
    # Placeholder bytes for the NOT-NULL geometry columns the schema requires
    # but the kinetics builder never reads (it only needs SMILES, links, barriers).
    import io
    buf = io.BytesIO()
    np.save(buf, np.zeros(1, dtype=np.float64))
    return buf.getvalue()


def load_network_into_sqlite(npz_path: Path) -> Session:
    """Materialize the shipped network as an in-memory SQLite DB and return a
    Session — the exact schema `build_snapshot` expects in production."""
    d = np.load(npz_path)  # pickle-free: all arrays are numeric or fixed-width unicode
    react_smiles, react_ptr = d["react_smiles"], d["react_ptr"]
    prod_smiles, prod_ptr = d["prod_smiles"], d["prod_ptr"]
    methods, names = d["discovery_method"], d["name"]
    bf, bb = d["barrier_forward"], d["barrier_backward"]
    sf, sb = d["barrier_forward_separated_pbe0"], d["barrier_backward_separated_pbe0"]
    mkf, mkb = d["manual_k_fwd"], d["manual_k_bwd"]
    n_reactions = int(d["n_reactions"])

    engine = create_engine("sqlite://")  # in-memory
    Base.metadata.create_all(engine)
    session = Session(engine)

    # --- one Compound row per unique SMILES ---
    all_smiles = sorted(set(react_smiles.tolist()) | set(prod_smiles.tolist()))
    blob = _dummy_blob()
    smiles_to_id: dict[str, int] = {}
    for i, smi in enumerate(all_smiles, start=1):
        session.add(Compound(
            id=i, smiles=smi, formula=smi, charge=0, n_atoms=0,
            sorted_atomic_numbers=blob, is_seed=False, experiments=["demo"],
        ))
        smiles_to_id[smi] = i
    session.flush()

    def _f(x):
        return None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)

    # --- one Reaction (+ join rows) per network edge ---
    for j in range(n_reactions):
        rxn = Reaction(
            id=j + 1,
            ts_id=j + 1,                       # unique synthetic id
            ts_conformer_positions=blob,
            ts_conformer_atomic_numbers=blob,
            ts_conformer_charge=0,
            ts_energy=0.0,
            barrier_forward=_f(bf[j]) or 0.0,
            barrier_backward=_f(bb[j]) or 0.0,
            barrier_forward_separated_pbe0=_f(sf[j]),
            barrier_backward_separated_pbe0=_f(sb[j]),
            manual_k_fwd=_f(mkf[j]),
            manual_k_bwd=_f(mkb[j]),
            discovery_method=str(methods[j]),
            name=str(names[j]),
            experiments=["demo"],
        )
        session.add(rxn)
        for smi in react_smiles[react_ptr[j]:react_ptr[j + 1]]:
            session.add(ReactionReactant(
                reaction_id=j + 1, compound_id=smiles_to_id[str(smi)],
                conformer_local_id=0,
            ))
        for smi in prod_smiles[prod_ptr[j]:prod_ptr[j + 1]]:
            session.add(ReactionProduct(
                reaction_id=j + 1, compound_id=smiles_to_id[str(smi)],
                conformer_local_id=0, energy=0.0,
            ))
    session.commit()
    return session


def main() -> int:
    if not NETWORK_NPZ.exists():
        print(f"ERROR: shipped network not found at {NETWORK_NPZ}", file=sys.stderr)
        return 1

    d = np.load(NETWORK_NPZ)
    src = str(d["source_checkpoint"])
    n_comp = int(d["n_compounds"])
    n_rxn = int(d["n_reactions"])

    print("=" * 68)
    print("ReactionAtlas — kinetics demo")
    print("=" * 68)
    print(f"Input network      : {NETWORK_NPZ.name}  (from published checkpoint {src})")
    print(f"Compounds / edges  : {n_comp} compounds, {n_rxn} raw reaction edges")
    print(f"Temperature        : {TEMPERATURE_K:.0f} K")
    print(f"Integration horizon: {T_MAX_S:.0e} s")
    print("-" * 68)

    session = load_network_into_sqlite(NETWORK_NPZ)

    t0 = time.perf_counter()
    snap = build_snapshot(
        session,
        temperature=TEMPERATURE_K,
        prefer_dft=True,
        t_max=T_MAX_S,
        experiment=None,   # single-experiment demo DB -> no experiment filter
    )
    wall = time.perf_counter() - t0

    if snap is None:
        print("ERROR: model had 0 usable reactions (nothing to solve).", file=sys.stderr)
        return 1

    print("Solved reaction network (via packages.kinetics.build.build_snapshot):")
    print(f"  species in ODE system     : {snap.n_species}")
    print(f"  reactions in ODE system   : {snap.n_reactions}")
    print(f"    of which manual equilibria: {snap.n_manual_equilibria}")
    print(f"    of which using DFT (PBE0) : {snap.n_reactions_dft}")
    print("-" * 68)

    # Deterministic steady-state distribution (softmax over log10 concentration
    # at t_max, computed inside build_snapshot). Print the dominant species.
    ss = snap.steady_state_distribution
    ranked = sorted(ss.items(), key=lambda kv: kv[1], reverse=True)
    print(f"Steady-state sampling distribution (top {min(15, len(ranked))} of {len(ranked)} active species):")
    print(f"  {'weight':>10}   SMILES")
    for smi, w in ranked[:15]:
        print(f"  {w:10.4f}   {smi}")
    print("-" * 68)
    print(f"[timing] end-to-end build+solve wall time: {wall:.2f}s "
          f"(solver reported {snap.solve_wall_time_s:.2f}s)")

    _write_plot(snap)
    print(f"[plot] wrote {PLOT_PATH.relative_to(REPO_ROOT)}")
    return 0


def _write_plot(snap) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.asarray(snap.times, dtype=float)
    concs = snap.concentrations
    # top species by peak concentration
    peak = {smi: max(v) for smi, v in concs.items() if max(v) > 0}
    top = sorted(peak, key=peak.get, reverse=True)[:10]

    fig, ax = plt.subplots(figsize=(8, 5))
    for smi in top:
        y = np.asarray(concs[smi], dtype=float)
        y = np.where(y <= 0, np.nan, y)
        ax.plot(times, y, label=smi, lw=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("concentration (M)")
    ax.set_title(f"ReactionAtlas kinetics demo — top {len(top)} species "
                 f"(T={TEMPERATURE_K:.0f} K)")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)


if __name__ == "__main__":
    raise SystemExit(main())
