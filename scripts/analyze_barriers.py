#!/usr/bin/env python3
"""Compare stored energy barriers vs force-integrated barriers from IRC trajectories.

Loads the pickle state, finds all reactions with trajectories, and compares:
1. Stored barrier: E(TS) - E(endpoint)
2. Trapezoidal integration: -Σ F_i · Δr_i along the IRC path
3. Hessian-corrected integration: quadratic interpolation between Hessian points

Usage:
    python scripts/analyze_barriers.py test_data/n111/
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Stub classes for unpickling ──
import types

class _Stub:
    def __setstate__(self, state):
        self.__dict__.update(state)

class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        cls = type(name, (_Stub,), {})
        setattr(self, name, cls)
        return cls

# Register stubs
lib_module = _StubModule("lib")
rg_module = _StubModule("lib.reaction_graph")
for name in ["ReactionRegistry", "ReactionEntry", "ReactantEntry",
             "Reaction", "NameGenerator", "ReactionGraph"]:
    setattr(rg_module, name, type(name, (_Stub,), {}))

sys.modules["lib"] = lib_module
sys.modules["lib.reaction_graph"] = rg_module

for submod in ["compound", "types", "pes_explorer", "pes_explorer.pes_graph",
               "pes_explorer.prfo", "energy", "fragment_mols", "naming", "utils",
               "md", "merge_mols", "md_et_calculator", "constants", "graph_analysis"]:
    mod = _StubModule(f"lib.{submod}")
    sys.modules[f"lib.{submod}"] = mod
    parts = submod.split(".")
    parent = lib_module
    for i, p in enumerate(parts[:-1]):
        full = ".".join(parts[:i + 1])
        if full not in sys.modules:
            sys.modules[full] = _StubModule(full)
        parent = sys.modules[full]
    setattr(parent, parts[-1], mod)


def trapezoidal_barrier(positions_list, forces_list, energies_list):
    """Integrate forces along path using trapezoidal rule.

    ΔE = -Σ ½(F_i + F_{i+1}) · (r_{i+1} - r_i)

    Returns integrated energy at each frame (relative to first frame).
    """
    n = len(positions_list)
    integrated = [0.0]

    for i in range(n - 1):
        dr = positions_list[i + 1] - positions_list[i]  # (n_atoms, 3)
        f_avg = 0.5 * (forces_list[i] + forces_list[i + 1])  # (n_atoms, 3)
        # Work = -F · dr (negative because F points downhill)
        work = -np.sum(f_avg * dr)
        integrated.append(integrated[-1] + work)

    return np.array(integrated)


def hessian_corrected_barrier(positions_list, forces_list, hessians_list, energies_list):
    """Integrate using quadratic energy model at Hessian points.

    At points with Hessians:
        E(r + δr) ≈ E(r) - F·δr + ½ δr^T H δr

    Between Hessian points, use trapezoidal on forces.
    Returns integrated energy at each frame.
    """
    n = len(positions_list)
    integrated = [0.0]

    # Find frames with Hessians
    hessian_frames = [i for i, h in enumerate(hessians_list) if h is not None]

    for i in range(n - 1):
        dr = positions_list[i + 1] - positions_list[i]  # (n_atoms, 3)
        dr_flat = dr.flatten()

        # Check if current frame has a Hessian
        if hessians_list[i] is not None:
            H = hessians_list[i]
            F = forces_list[i].flatten()
            # Quadratic model: ΔE = -F·δr + ½ δr^T H δr
            dE = -np.dot(F, dr_flat) + 0.5 * np.dot(dr_flat, H @ dr_flat)
        else:
            # Fall back to trapezoidal
            f_avg = 0.5 * (forces_list[i] + forces_list[i + 1])
            dE = -np.sum(f_avg * dr)

        integrated.append(integrated[-1] + dE)

    return np.array(integrated)


def analyze_compound_pes(compound, compound_name):
    """Analyze PES graph trajectories for a compound."""
    results = []

    pes = getattr(compound, 'pes_graph', None)
    if pes is None:
        return results

    # Get transition states
    tss = getattr(pes, 'transition_states', None) or getattr(pes, '_transition_states', {})
    if not tss:
        return results

    minima = getattr(pes, 'minima', None) or getattr(pes, '_minima', {})

    for ts_id, ts in tss.items():
        # Get trajectory data
        fwd_traj = getattr(ts, 'fwd_trajectory', None)
        bwd_traj = getattr(ts, 'bwd_trajectory', None)

        ts_energy = getattr(ts, 'energy', None)

        for direction, traj in [('forward', fwd_traj), ('backward', bwd_traj)]:
            if traj is None:
                continue

            positions = getattr(traj, 'positions', None)
            energies = getattr(traj, 'energies', None)
            forces = getattr(traj, 'forces', None)
            hessians = getattr(traj, 'hessians', None)

            if positions is None or forces is None or len(positions) < 2:
                continue

            if energies is None or len(energies) < 2:
                continue

            # Stored barrier: energy difference between first and last frame
            stored_energies = np.array(energies)
            stored_barrier = stored_energies[-1] - stored_energies[0]

            # Trapezoidal integration
            trap_integrated = trapezoidal_barrier(positions, forces, energies)
            trap_barrier = trap_integrated[-1]

            # Hessian-corrected integration
            hess_barrier = None
            n_hessians = 0
            if hessians is not None:
                n_hessians = sum(1 for h in hessians if h is not None)
                if n_hessians > 0:
                    hess_integrated = hessian_corrected_barrier(positions, forces, hessians, energies)
                    hess_barrier = hess_integrated[-1]

            results.append({
                'compound': compound_name,
                'ts_id': ts_id,
                'direction': direction,
                'n_frames': len(positions),
                'n_hessians': n_hessians,
                'stored_barrier': stored_barrier,
                'trap_barrier': trap_barrier,
                'hess_barrier': hess_barrier,
                'stored_energies': stored_energies,
                'trap_integrated': trap_integrated,
                'hess_integrated': hess_integrated if hess_barrier is not None else None,
            })

    return results


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_data/n111")
    pkl_path = data_dir / ".reaction_graph_state.pkl"

    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        state = pickle.load(f)

    # Extract compounds from state
    compounds = state['compound_registry']._compounds

    print(f"Found {len(compounds)} compounds")

    # Analyze all compounds
    all_results = []
    for name, compound in compounds.items():
        results = analyze_compound_pes(compound, name)
        all_results.extend(results)

    if not all_results:
        print("\nNo trajectories found! Checking state structure...")
        print(f"State type: {type(state).__name__}")
        if hasattr(state, '__dict__'):
            for k, v in state.__dict__.items():
                print(f"  {k}: {type(v).__name__}", end="")
                if hasattr(v, '__len__'):
                    print(f" (len={len(v)})", end="")
                if hasattr(v, 'pes_graph'):
                    pes = v.pes_graph
                    ts_count = len(getattr(pes, 'transition_states', {}) or getattr(pes, '_transition_states', {}))
                    print(f" [PES: {ts_count} TSs]", end="")
                print()
        return

    # Print results
    print(f"\n{'='*90}")
    print(f"{'Compound':<20} {'Dir':<8} {'Frames':<7} {'Hess':<5} "
          f"{'Stored(eV)':<12} {'Trap(eV)':<12} {'Hess(eV)':<12} {'Trap err%':<10} {'Hess err%'}")
    print(f"{'='*90}")

    trap_errors = []
    hess_errors = []

    for r in all_results:
        trap_err = abs(r['trap_barrier'] - r['stored_barrier'])
        trap_pct = 100 * trap_err / max(abs(r['stored_barrier']), 1e-10)
        trap_errors.append(trap_pct)

        hess_str = "N/A"
        hess_pct_str = "N/A"
        if r['hess_barrier'] is not None:
            hess_err = abs(r['hess_barrier'] - r['stored_barrier'])
            hess_pct = 100 * hess_err / max(abs(r['stored_barrier']), 1e-10)
            hess_errors.append(hess_pct)
            hess_str = f"{r['hess_barrier']:>10.4f}"
            hess_pct_str = f"{hess_pct:>8.2f}%"

        print(f"{r['compound'][:20]:<20} {r['direction']:<8} {r['n_frames']:<7} {r['n_hessians']:<5} "
              f"{r['stored_barrier']:>10.4f}  {r['trap_barrier']:>10.4f}  {hess_str}  "
              f"{trap_pct:>8.2f}%  {hess_pct_str}")

    print(f"\n{'='*90}")
    print(f"Total trajectories analyzed: {len(all_results)}")
    print(f"Trapezoidal error:  mean={np.mean(trap_errors):.2f}%  median={np.median(trap_errors):.2f}%  max={np.max(trap_errors):.2f}%")
    if hess_errors:
        print(f"Hessian-corrected:  mean={np.mean(hess_errors):.2f}%  median={np.median(hess_errors):.2f}%  max={np.max(hess_errors):.2f}%")
    else:
        print("No Hessian data available for comparison")


if __name__ == "__main__":
    main()
