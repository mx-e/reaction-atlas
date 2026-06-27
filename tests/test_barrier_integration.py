"""Tests for Hessian-corrected barrier integration.

Tests compute_barrier_from_trajectory against known analytical cases
and verifies the integration flows through ExplorationContext correctly.

Run: python tests/test_barrier_integration.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "packages", "worker"))

import numpy as np
from lib.energy import compute_barrier_from_trajectory
from lib.pes_explorer.pes_graph import RelaxationTrajectory


def make_trajectory(positions, energies, forces, hessians=None):
    """Helper to build a RelaxationTrajectory."""
    n = len(positions)
    if hessians is None:
        hessians = [None] * n
    return RelaxationTrajectory(
        positions=positions,
        energies=energies,
        forces=forces,
        hessians=hessians,
    )


# ── Analytical test: harmonic potential ──
# E(x) = ½kx², F(x) = -kx, H = k
# Trajectory from x=1 → x=0 (TS at x=1, minimum at x=0)
# True barrier = ½k(1)² - ½k(0)² = ½k

def test_harmonic_1d_exact():
    """Single atom in 1D harmonic potential. Hessian-corrected should be exact."""
    k = 2.0
    n_steps = 10
    xs = np.linspace(1.0, 0.0, n_steps)

    positions = [np.array([[x, 0.0, 0.0]]) for x in xs]
    energies = [0.5 * k * x**2 for x in xs]
    forces = [np.array([[-k * x, 0.0, 0.0]]) for x in xs]
    hessians = [np.diag([k, 0.0, 0.0]) for _ in xs]

    traj = make_trajectory(positions, energies, forces, hessians)
    barrier = compute_barrier_from_trajectory(traj)

    expected = 0.5 * k  # E(TS) - E(min) = ½k - 0
    assert abs(barrier - expected) < 1e-10, f"Harmonic 1D: expected {expected}, got {barrier}"
    print(f"  harmonic_1d_exact: barrier={barrier:.6f}, expected={expected:.6f} ✓")


def test_harmonic_1d_trapezoidal_only():
    """Same harmonic potential but no Hessians — trapezoidal only."""
    k = 2.0
    n_steps = 100  # More steps needed for accuracy without Hessians
    xs = np.linspace(1.0, 0.0, n_steps)

    positions = [np.array([[x, 0.0, 0.0]]) for x in xs]
    energies = [0.5 * k * x**2 for x in xs]
    forces = [np.array([[-k * x, 0.0, 0.0]]) for x in xs]

    traj = make_trajectory(positions, energies, forces)
    barrier = compute_barrier_from_trajectory(traj)

    expected = 0.5 * k
    error_pct = abs(barrier - expected) / expected * 100
    assert error_pct < 1.0, f"Trapezoidal error {error_pct:.2f}% > 1% for 100-step harmonic"
    print(f"  harmonic_1d_trap: barrier={barrier:.6f}, expected={expected:.6f}, error={error_pct:.4f}% ✓")


def test_harmonic_1d_coarse_hessian_vs_trapezoidal():
    """Coarse trajectory (5 steps) — Hessian correction should beat trapezoidal."""
    k = 2.0
    n_steps = 5
    xs = np.linspace(1.0, 0.0, n_steps)

    positions = [np.array([[x, 0.0, 0.0]]) for x in xs]
    energies = [0.5 * k * x**2 for x in xs]
    forces = [np.array([[-k * x, 0.0, 0.0]]) for x in xs]
    hessians = [np.diag([k, 0.0, 0.0]) for _ in xs]

    traj_hess = make_trajectory(positions, energies, forces, hessians)
    traj_trap = make_trajectory(positions, energies, forces)

    barrier_hess = compute_barrier_from_trajectory(traj_hess)
    barrier_trap = compute_barrier_from_trajectory(traj_trap)

    expected = 0.5 * k
    err_hess = abs(barrier_hess - expected)
    err_trap = abs(barrier_trap - expected)

    assert err_hess <= err_trap, f"Hessian error ({err_hess:.6f}) should be <= trapezoidal ({err_trap:.6f})"
    print(f"  coarse_hess_vs_trap: hess_err={err_hess:.6f}, trap_err={err_trap:.6f} ✓")


# ── Multi-atom test ──

def test_two_atom_stretch():
    """Two atoms stretching — H2-like potential along bond axis."""
    k = 3.0
    n_steps = 20
    # Bond lengths from 1.5 (TS) to 0.74 (equilibrium)
    bonds = np.linspace(1.5, 0.74, n_steps)

    positions = [np.array([[0, 0, 0], [0, 0, b]]) for b in bonds]
    # E = ½k(b - b_eq)² where b_eq = 0.74
    energies = [0.5 * k * (b - 0.74)**2 for b in bonds]
    # F on atom 2 along z: F_z = -k(b - b_eq), atom 1 gets -F_z
    forces = [np.array([[0, 0, k * (b - 0.74)], [0, 0, -k * (b - 0.74)]]) for b in bonds]
    # 6x6 Hessian: only z-z block of atoms 1,2 matters
    def make_hessian():
        H = np.zeros((6, 6))
        H[2, 2] = k; H[2, 5] = -k
        H[5, 2] = -k; H[5, 5] = k
        return H
    hessians = [make_hessian() for _ in bonds]

    traj = make_trajectory(positions, energies, forces, hessians)
    barrier = compute_barrier_from_trajectory(traj)

    expected = 0.5 * k * (1.5 - 0.74)**2
    error_pct = abs(barrier - expected) / expected * 100
    assert error_pct < 0.1, f"Two-atom stretch error {error_pct:.2f}% > 0.1%"
    print(f"  two_atom_stretch: barrier={barrier:.6f}, expected={expected:.6f}, error={error_pct:.4f}% ✓")


# ── Sparse Hessian test ──

def test_sparse_hessians():
    """Only every 5th frame has a Hessian — should still be more accurate than pure trapezoidal."""
    k = 2.0
    n_steps = 20
    xs = np.linspace(1.0, 0.0, n_steps)

    positions = [np.array([[x, 0.0, 0.0]]) for x in xs]
    energies = [0.5 * k * x**2 for x in xs]
    forces = [np.array([[-k * x, 0.0, 0.0]]) for x in xs]
    hessians_sparse = [np.diag([k, 0.0, 0.0]) if i % 5 == 0 else None for i in range(n_steps)]

    traj_sparse = make_trajectory(positions, energies, forces, hessians_sparse)
    traj_trap = make_trajectory(positions, energies, forces)

    barrier_sparse = compute_barrier_from_trajectory(traj_sparse)
    barrier_trap = compute_barrier_from_trajectory(traj_trap)

    expected = 0.5 * k
    err_sparse = abs(barrier_sparse - expected)
    err_trap = abs(barrier_trap - expected)

    assert err_sparse <= err_trap, f"Sparse Hessian ({err_sparse:.6f}) should be <= trapezoidal ({err_trap:.6f})"
    print(f"  sparse_hessians: sparse_err={err_sparse:.6f}, trap_err={err_trap:.6f} ✓")


# ── Anharmonic test (Morse potential) ──

def test_morse_potential():
    """Morse potential — anharmonic, tests accuracy beyond quadratic regime."""
    D = 1.0  # Dissociation energy
    a = 2.0  # Width parameter
    r_eq = 0.74
    n_steps = 50

    rs = np.linspace(1.8, r_eq, n_steps)
    positions = [np.array([[0, 0, 0], [0, 0, r]]) for r in rs]
    energies = [D * (1 - np.exp(-a * (r - r_eq)))**2 for r in rs]

    # Force on atom 2 (z-component): F = -dE/dr = -2*D*a*(1-exp(-a*(r-r_eq)))*exp(-a*(r-r_eq))
    forces_z = [-2 * D * a * (1 - np.exp(-a * (r - r_eq))) * np.exp(-a * (r - r_eq)) for r in rs]
    forces = [np.array([[0, 0, -fz], [0, 0, fz]]) for fz in forces_z]

    # Hessian: d²E/dr² = 2*D*a²*exp(-a*(r-r_eq))*(2*exp(-a*(r-r_eq)) - 1)
    def morse_hessian(r):
        exp_term = np.exp(-a * (r - r_eq))
        d2E = 2 * D * a**2 * exp_term * (2 * exp_term - 1)
        H = np.zeros((6, 6))
        H[2, 2] = d2E; H[2, 5] = -d2E
        H[5, 2] = -d2E; H[5, 5] = d2E
        return H
    hessians = [morse_hessian(r) for r in rs]

    traj_hess = make_trajectory(positions, energies, forces, hessians)
    traj_trap = make_trajectory(positions, energies, forces)

    barrier_hess = compute_barrier_from_trajectory(traj_hess)
    barrier_trap = compute_barrier_from_trajectory(traj_trap)

    expected = D * (1 - np.exp(-a * (1.8 - r_eq)))**2  # E(TS) - E(min) = E(1.8) - 0
    err_hess = abs(barrier_hess - expected) / expected * 100
    err_trap = abs(barrier_trap - expected) / expected * 100

    assert err_hess < 1.0, f"Morse Hessian error {err_hess:.2f}% > 1%"
    assert err_trap < 1.0, f"Morse trapezoidal error {err_trap:.2f}% > 1%"
    print(f"  morse_potential: hess_err={err_hess:.4f}%, trap_err={err_trap:.4f}%, expected={expected:.6f} ✓")


# ── Error handling ──

def test_raises_on_none_trajectory():
    """Must raise when trajectory is None."""
    try:
        compute_barrier_from_trajectory(None)
        assert False, "Should have raised"
    except (AttributeError, TypeError):
        print(f"  raises_on_none: ✓")


def test_raises_on_empty_trajectory():
    """Must raise when trajectory has < 2 frames."""
    traj = make_trajectory(
        [np.array([[0, 0, 0]])],
        [0.0],
        [np.array([[0, 0, 0]])],
    )
    try:
        compute_barrier_from_trajectory(traj)
        assert False, "Should have raised"
    except ValueError:
        print(f"  raises_on_empty: ✓")


# ── ExplorationContext integration ──

def test_exploration_context_uses_integration():
    """ExplorationContext.get_ts_barrier_* must use trajectory integration."""
    from lib.exploration import ExplorationContext

    k = 2.0
    xs = np.linspace(1.0, 0.0, 20)
    positions = [np.array([[x, 0.0, 0.0]]) for x in xs]
    energies = [0.5 * k * x**2 for x in xs]
    forces = [np.array([[-k * x, 0.0, 0.0]]) for x in xs]
    hessians = [np.diag([k, 0.0, 0.0]) for _ in xs]

    traj = make_trajectory(positions, energies, forces, hessians)

    ctx = ExplorationContext()
    ctx.reactant_trajectory = traj
    ctx.product_trajectory = traj

    fwd = ctx.get_ts_barrier_forward()
    bwd = ctx.get_ts_barrier_backward()

    expected = 0.5 * k
    assert abs(fwd - expected) < 1e-10, f"Forward barrier {fwd} != {expected}"
    assert abs(bwd - expected) < 1e-10, f"Backward barrier {bwd} != {expected}"
    print(f"  exploration_context: fwd={fwd:.6f}, bwd={bwd:.6f} ✓")


def test_exploration_context_raises_without_trajectory():
    """ExplorationContext must fail if trajectory is missing."""
    from lib.exploration import ExplorationContext

    ctx = ExplorationContext()
    try:
        ctx.get_ts_barrier_forward()
        assert False, "Should have raised"
    except (AttributeError, TypeError):
        pass

    try:
        ctx.get_ts_barrier_backward()
        assert False, "Should have raised"
    except (AttributeError, TypeError):
        pass

    print(f"  context_raises_without_traj: ✓")


# ── Run on real pickle data ──

def test_real_pickle_barriers():
    """Verify integration works on actual trajectory data from the pickle."""
    import pickle
    import types as pytypes

    pkl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "test_data", "n111", ".reaction_graph_state.pkl")
    if not os.path.exists(pkl_path):
        print(f"  real_pickle: SKIPPED (no test data)")
        return

    # Stub classes for unpickling
    class _Stub:
        def __setstate__(self, state): self.__dict__.update(state)
    class _StubModule(pytypes.ModuleType):
        def __getattr__(self, name):
            if name.startswith("_"): raise AttributeError(name)
            cls = type(name, (_Stub,), {}); setattr(self, name, cls); return cls

    # Save and restore sys.modules
    saved = {}
    lib_module = _StubModule("lib")
    rg_module = _StubModule("lib.reaction_graph")
    for n in ["ReactionRegistry", "ReactionEntry", "ReactantEntry", "Reaction", "NameGenerator", "ReactionGraph"]:
        setattr(rg_module, n, type(n, (_Stub,), {}))

    mods_to_register = {"lib": lib_module, "lib.reaction_graph": rg_module}
    for s in ["compound", "types", "pes_explorer", "pes_explorer.pes_graph", "pes_explorer.prfo",
              "energy", "fragment_mols", "naming", "utils", "md", "merge_mols",
              "md_et_calculator", "constants", "graph_analysis"]:
        mods_to_register[f"lib.{s}"] = _StubModule(f"lib.{s}")

    for k, v in mods_to_register.items():
        saved[k] = sys.modules.get(k)
        sys.modules[k] = v

    try:
        with open(pkl_path, "rb") as f:
            state = pickle.load(f)

        compounds = state["compound_registry"]._compounds
        n_tested = 0
        n_valid = 0

        for name, comp in list(compounds.items())[:20]:
            pg = comp.pes_graph
            tss = getattr(pg, "transition_states", {}) or getattr(pg, "_transition_states", {})
            for ts_id, ts in tss.items():
                for attr in ["fwd_trajectory", "bwd_trajectory"]:
                    traj = getattr(ts, attr, None)
                    if traj is None or not traj.positions or len(traj.positions) < 2:
                        continue
                    if not traj.forces:
                        continue

                    barrier = compute_barrier_from_trajectory(traj)
                    n_tested += 1

                    # Basic sanity: barrier should be finite and positive (TS above minimum)
                    assert np.isfinite(barrier), f"Non-finite barrier for {name} TS#{ts_id} {attr}"
                    if barrier > 0:
                        n_valid += 1

        print(f"  real_pickle: tested {n_tested} trajectories, {n_valid} positive barriers ✓")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


if __name__ == "__main__":
    print("Testing compute_barrier_from_trajectory:")
    test_harmonic_1d_exact()
    test_harmonic_1d_trapezoidal_only()
    test_harmonic_1d_coarse_hessian_vs_trapezoidal()
    test_two_atom_stretch()
    test_sparse_hessians()
    test_morse_potential()
    test_raises_on_none_trajectory()
    test_raises_on_empty_trajectory()
    test_exploration_context_uses_integration()
    test_exploration_context_raises_without_trajectory()
    test_real_pickle_barriers()
    print("\nAll tests passed!")
