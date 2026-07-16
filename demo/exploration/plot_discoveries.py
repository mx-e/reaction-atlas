#!/usr/bin/env python3
"""Plot what the exploration demo discovered so far.

Reads the reaction network from the database at DATABASE_URL (the same one
run_demo.sh writes into) and produces, next to this script:

  - network.png : networkx rendering of the current reaction network,
                  seed compounds vs. discovered compounds highlighted
  - growth.png  : cumulative compounds / reactions / intramolecular TSs
                  over the wall time of the exploration

Run it in another terminal while the demo explores, or after Ctrl-C:

    uv run --extra db python demo/exploration/plot_discoveries.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from packages.db.connection import get_session
from packages.db.models import (
    Compound,
    IntraTransitionState,
    Reaction,
    ReactionProduct,
    ReactionReactant,
)

NETWORK_PLOT_PATH = Path(__file__).resolve().parent / "network.png"
GROWTH_PLOT_PATH = Path(__file__).resolve().parent / "growth.png"

SEED_COLOR = "#f2c14e"
DISCOVERED_COLOR = "#4e79a7"


def _mol_image(smiles: str, px: int = 220):
    """2D depiction of a SMILES as an RGBA numpy array with transparent
    background, or None if rdkit is unavailable / the SMILES fails to parse."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        from PIL import Image
        import io
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    drawer = rdMolDraw2D.MolDraw2DCairo(px, px)
    drawer.drawOptions().setBackgroundColour((1, 1, 1, 0))
    drawer.drawOptions().padding = 0.05
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    return np.asarray(img)


def _load(session):
    compounds = {c.id: c for c in session.query(Compound).all()}
    reactions = session.query(Reaction).all()
    reactants = session.query(ReactionReactant).all()
    products = session.query(ReactionProduct).all()
    intra_ts = session.query(IntraTransitionState).all()
    return compounds, reactions, reactants, products, intra_ts


def _write_network_plot(compounds, reactions, reactants, products) -> None:
    by_rxn_reac: dict[int, list[int]] = {}
    by_rxn_prod: dict[int, list[int]] = {}
    for rr in reactants:
        by_rxn_reac.setdefault(rr.reaction_id, []).append(rr.compound_id)
    for rp in products:
        by_rxn_prod.setdefault(rp.reaction_id, []).append(rp.compound_id)

    G = nx.Graph()
    for c in compounds.values():
        G.add_node(c.smiles, is_seed=c.is_seed)
    for rxn in reactions:
        for r_id in by_rxn_reac.get(rxn.id, []):
            for p_id in by_rxn_prod.get(rxn.id, []):
                r, p = compounds[r_id].smiles, compounds[p_id].smiles
                if r == p:
                    continue
                if G.has_edge(r, p):
                    G[r][p]["n_rxn"] += 1
                else:
                    G.add_edge(r, p, n_rxn=1)

    # Layout: the main connected component is spread across the upper area,
    # and every other component (including isolated single molecules, which are
    # common early in an exploration) is placed on a well-spaced grid below so
    # that no two molecule depictions overlap — even when they are unconnected.
    comps = sorted(nx.connected_components(G), key=len, reverse=True)

    def _normalize(p, x_lo, x_hi, y_lo, y_hi):
        """Rescale a dict of positions into the given bounding box."""
        pts = np.array(list(p.values()), dtype=float)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        span = np.where((hi - lo) == 0, 1.0, hi - lo)
        out = {}
        for n, xy in p.items():
            u = (np.asarray(xy, dtype=float) - lo) / span  # -> [0,1]^2
            out[n] = np.array([x_lo + u[0] * (x_hi - x_lo),
                               y_lo + u[1] * (y_hi - y_lo)])
        return out

    main = G.subgraph(comps[0])
    main_pos = nx.kamada_kawai_layout(main) if len(main) > 1 else {list(main)[0]: np.array([0.0, 0.0])}
    main_pos = nx.spring_layout(main, pos=main_pos, seed=42, iterations=80,
                                k=1.6 / np.sqrt(max(len(main), 1)))
    # Park the main cluster in the upper band, spread across the width.
    pos = _normalize(main_pos, x_lo=-1.6, x_hi=1.6, y_lo=0.5, y_hi=2.1) if len(main) > 1 \
        else {list(main)[0]: np.array([0.0, 1.3])}

    # Satellites: every molecule outside the main cluster gets its own cell on a
    # generously spaced grid below, so no two depictions overlap even when they
    # are unconnected (or only connected to each other). Grouped by component so
    # a connected pair lands in adjacent cells, with its reaction edge drawn.
    sat_nodes = [n for comp in comps[1:] for n in sorted(comp)]
    if sat_nodes:
        step = 0.95                      # >> molecule-image footprint, so no overlap
        per_row = max(1, min(len(sat_nodes), 6))
        x0 = -step * (per_row - 1) / 2.0
        for i, n in enumerate(sat_nodes):
            r, c = divmod(i, per_row)
            pos[n] = np.array([x0 + c * step, -0.3 - r * step])

    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color="#9aa7b1", alpha=0.7,
        width=[0.8 + 0.5 * G[u][v]["n_rxn"] for u, v in G.edges()])

    # nodes: rdkit 2D depictions in colored frames (seed vs discovered),
    # falling back to plain colored dots + SMILES text without rdkit
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    fallback = []
    for n in G.nodes():
        color = SEED_COLOR if G.nodes[n]["is_seed"] else DISCOVERED_COLOR
        img = _mol_image(n)
        if img is None:
            fallback.append(n)
            continue
        ab = AnnotationBbox(
            OffsetImage(img, zoom=0.24), pos[n], frameon=True,
            bboxprops=dict(edgecolor=color, facecolor="white",
                           linewidth=2.0, boxstyle="round,pad=0.1"))
        ab.set_zorder(3)
        ax.add_artist(ab)
    if fallback:
        nx.draw_networkx_nodes(
            G, pos, nodelist=fallback, ax=ax, node_size=300,
            node_color=[SEED_COLOR if G.nodes[n]["is_seed"] else DISCOVERED_COLOR
                        for n in fallback],
            linewidths=0.8, edgecolors="white")
        nx.draw_networkx_labels(
            G, {n: (x, y - 0.09) for n, (x, y) in pos.items()}, ax=ax,
            labels={n: n for n in fallback}, font_size=6, font_color="#555555")

    n_disc = sum(1 for n in G.nodes() if not G.nodes[n]["is_seed"])
    ax.scatter([], [], c=SEED_COLOR, s=80, label="seeded")
    ax.scatter([], [], c=DISCOVERED_COLOR, s=80, label="discovered")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False)
    ax.set_xlim(min(x for x, _ in pos.values()) - 0.5,
                max(x for x, _ in pos.values()) + 0.5)
    ax.set_ylim(min(y for _, y in pos.values()) - 0.5,
                max(y for _, y in pos.values()) + 0.5)
    ax.set_title(f"Exploration demo — {len(G)} compounds "
                 f"({n_disc} discovered), {len(reactions)} reactions", loc="left")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(NETWORK_PLOT_PATH, dpi=130)
    print(f"[plot] wrote {NETWORK_PLOT_PATH.relative_to(REPO_ROOT)}")


def _write_growth_plot(compounds, reactions, intra_ts) -> None:
    from datetime import datetime, timezone

    t0 = min(c.created_at for c in compounds.values())
    # intra-TS rows carry a float epoch `discovery_timestamp` instead of a
    # `created_at` datetime column
    ts_times = [datetime.fromtimestamp(t.discovery_timestamp, tz=timezone.utc)
                for t in intra_ts if t.discovery_timestamp > 0]

    def minutes(times):
        return sorted(max((t - t0).total_seconds(), 0.0) / 60.0 for t in times)

    series = [
        (minutes([c.created_at for c in compounds.values()]), "compounds", DISCOVERED_COLOR),
        (minutes([r.created_at for r in reactions]), "reactions", "#e15759"),
        (minutes(ts_times), "intramolecular TSs", "#59a14f"),
    ]
    # End the axis just past the last event so the staircase reads clearly,
    # rather than stretching a long flat tail out to "now" when the plot is
    # generated well after the run finished.
    last_event = max((t[-1] for t, _, _ in series if t), default=1.0)
    t_end = last_event * 1.15 + 1.0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for t, label, color in series:
        ax.step([0.0] + t + [t_end], np.append(np.arange(len(t) + 1), len(t)),
                where="post", label=label, color=color, lw=1.8)
    ax.set_xlabel("wall time since seeding (min)")
    ax.set_ylabel("cumulative count")
    ax.set_title("Exploration demo — network growth over time")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(GROWTH_PLOT_PATH, dpi=130)
    print(f"[plot] wrote {GROWTH_PLOT_PATH.relative_to(REPO_ROOT)}")


def main() -> int:
    session = get_session()
    compounds, reactions, reactants, products, intra_ts = _load(session)
    if not compounds:
        print("ERROR: database is empty — run ./demo/exploration/run_demo.sh first.",
              file=sys.stderr)
        return 1
    n_disc = sum(1 for c in compounds.values() if not c.is_seed)
    print(f"compounds: {len(compounds)} ({n_disc} discovered)  "
          f"reactions: {len(reactions)}  intra-TS: {len(intra_ts)}")
    _write_network_plot(compounds, reactions, reactants, products)
    _write_growth_plot(compounds, reactions, intra_ts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
