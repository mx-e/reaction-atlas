"""Full kinetics solver A/B bench — matsolver × lag sweep.

Runs the solver 6 times on a restored prod snapshot (local
`crn-bench-pg` container) to measure which PETSc sparse backend + PC
lag combination is actually worth shipping:

    1. baseline    = pc.setType('lu'), no factorSolverType
    2. + mumps     = pc.setFactorSolverType('mumps')
    3. + klu       = pc.setFactorSolverType('klu')
    4. + umfpack   = pc.setFactorSolverType('umfpack')
    5. winner + PC lag=2
    6. winner + PC lag=5

Accuracy gate: max |Δlog10(y)| on active species (>1e-10 M) between
baseline and the test run.  If lag degrades accuracy substantially we
don't ship it regardless of speedup.

Decision rule (encoded at end): ship a change only if speedup ≥1.5×
AND max|Δlog10| on active species ≤0.1.

Usage (inside the API container so PETSc+numba+RDKit are linked):

    docker run --rm --network host \\
      -e DATABASE_URL=postgresql://crn:crn@host.docker.internal:5433/crn_cloud \\
      -e KINETICS_PETSC_RTOL=1e-8 \\
      -e KINETICS_PETSC_ATOL=1e-10 \\
      -v /Users/maxi/WORKSPACE/crn-cloud/scripts/bench:/bench \\
      us-central1-docker.pkg.dev/reactionatlas/crn-cloud/api:latest \\
      python /bench/bench_solver.py
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

if os.path.isdir("/app/packages"):
    sys.path.insert(0, "/app")
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))


@dataclass
class RunResult:
    label: str
    matsolver: str
    jac_lag: int
    pc_lag: int
    wall_s: float
    y_final: np.ndarray


def _run_once(label: str, matsolver: str, jac_lag: int, pc_lag: int,
              temperature: float) -> RunResult:
    """Single solve with the requested env config."""
    env = {
        "KINETICS_PETSC_MATSOLVER": matsolver,
        "KINETICS_PETSC_JAC_LAG": str(jac_lag),
        "KINETICS_PETSC_PC_LAG": str(pc_lag),
    }
    for k, v in env.items():
        if v: os.environ[k] = v
        else: os.environ.pop(k, None)

    # Fresh imports per run so module-level PETSc state gets a clean slate.
    for mod in list(sys.modules):
        if mod.startswith("packages.kinetics") or mod == "petsc4py" or mod.startswith("petsc4py."):
            sys.modules.pop(mod, None)

    from packages.kinetics.model import build_model_from_db
    from packages.kinetics.scipy_solver import solve_ode

    db_url = os.environ.get("DATABASE_URL",
                            "postgresql://crn:crn@localhost:5433/crn_cloud")
    engine = create_engine(db_url)
    SessionMaker = sessionmaker(bind=engine)
    with SessionMaker() as session:
        model = build_model_from_db(session, temperature=temperature, prefer_dft=True)

    print(f"  [{label}] matsolver={matsolver or 'default':<9s}  jac_lag={jac_lag}  pc_lag={pc_lag}")
    t0 = time.perf_counter()
    sol_fn = solve_ode(model, t_max=1e8)
    y_final = sol_fn(1e8)
    wall = time.perf_counter() - t0
    print(f"  [{label}] wall={wall:.2f}s  n_species={len(model.smiles_list)}  "
          f"n_rxn={model.n_reactions}")
    return RunResult(label, matsolver, jac_lag, pc_lag, wall, y_final)


def _accuracy_diff(y_a: np.ndarray, y_b: np.ndarray) -> dict[str, float]:
    """Max/p99/median |Δlog10(y)| on species with meaningful concentration."""
    eps = 1e-20
    log_a = np.log10(np.maximum(y_a, eps))
    log_b = np.log10(np.maximum(y_b, eps))
    diff = np.abs(log_a - log_b)
    mask = (y_a > 1e-10) | (y_b > 1e-10)
    if mask.sum() == 0:
        return {"n_active": 0, "max": 0.0, "p99": 0.0, "median": 0.0}
    d = diff[mask]
    return {
        "n_active": int(mask.sum()),
        "max": float(d.max()),
        "p99": float(np.percentile(d, 99)),
        "median": float(np.median(d)),
    }


def main():
    temperature = float(os.environ.get("KINETICS_EXPLORATION_TEMPERATURE", "500"))
    print("=" * 70)
    print(f"Kinetics solver A/B suite — T={temperature}K, t_max=1e8")
    print(f"rtol={os.environ.get('KINETICS_PETSC_RTOL', '1e-12')}  "
          f"atol={os.environ.get('KINETICS_PETSC_ATOL', '1e-16')}")
    print("=" * 70)
    print()

    # Phase 1: matsolver sweep at lag=1,1
    print("### Phase 1: matsolver sweep (lag=1,1)")
    results: list[RunResult] = []
    matsolver_candidates = ["", "mumps", "klu", "umfpack"]
    for i, ms in enumerate(matsolver_candidates, 1):
        label = f"{i}-{ms or 'default'}"
        try:
            r = _run_once(label, ms, jac_lag=1, pc_lag=1, temperature=temperature)
            results.append(r)
        except Exception as e:
            print(f"  [{label}] FAILED: {e}")
    print()

    if not results:
        print("All matsolver runs failed — aborting.")
        return

    # Accuracy: use baseline (first run) as reference
    baseline = results[0]
    print("### Phase 1 results")
    print(f"  {'label':<14s} {'wall_s':>8s}  {'speedup':>8s}  {'max|Δlog10|':>12s}  {'p99':>6s}")
    for r in results:
        diffs = _accuracy_diff(baseline.y_final, r.y_final)
        speedup = baseline.wall_s / r.wall_s if r.wall_s > 0 else 0
        print(f"  {r.label:<14s} {r.wall_s:>8.2f}  {speedup:>7.2f}×  "
              f"{diffs['max']:>12.3f}  {diffs['p99']:>6.3f}")
    print()

    # Phase 2: pc_lag sweep on the fastest accurate matsolver
    def _is_accurate(r: RunResult) -> bool:
        return _accuracy_diff(baseline.y_final, r.y_final)["max"] < 0.2
    accurate = [r for r in results if _is_accurate(r)]
    if not accurate:
        print("No matsolver produced accurate-enough results — skipping lag phase.")
        _final_decision(baseline, results)
        return

    winner = min(accurate, key=lambda r: r.wall_s)
    print(f"### Phase 2: pc-lag sweep on winner = {winner.label} "
          f"(matsolver={winner.matsolver or 'default'})")
    print()
    lag_results = [winner]
    for pc_lag in (2, 5):
        label = f"{winner.label}+lag{pc_lag}"
        try:
            r = _run_once(label, winner.matsolver, jac_lag=1, pc_lag=pc_lag,
                          temperature=temperature)
            lag_results.append(r)
        except Exception as e:
            print(f"  [{label}] FAILED: {e}")
    print()

    print("### Phase 2 results")
    print(f"  {'label':<20s} {'wall_s':>8s}  {'speedup':>8s}  {'max|Δlog10|':>12s}")
    for r in lag_results:
        diffs = _accuracy_diff(baseline.y_final, r.y_final)
        speedup = baseline.wall_s / r.wall_s if r.wall_s > 0 else 0
        print(f"  {r.label:<20s} {r.wall_s:>8.2f}  {speedup:>7.2f}×  "
              f"{diffs['max']:>12.3f}")
    print()

    _final_decision(baseline, results + lag_results[1:])


def _final_decision(baseline: RunResult, all_runs: list[RunResult]) -> None:
    """Apply decision rule: ship change only if speedup ≥1.5× AND max|Δlog10|≤0.1
    on active species."""
    print("=" * 70)
    print("### Decision rule: ship if speedup ≥1.5× AND max|Δlog10|≤0.1")
    print("=" * 70)
    candidates = []
    for r in all_runs:
        if r is baseline: continue
        diffs = _accuracy_diff(baseline.y_final, r.y_final)
        speedup = baseline.wall_s / r.wall_s if r.wall_s > 0 else 0
        passes = speedup >= 1.5 and diffs["max"] <= 0.1
        candidates.append((r, speedup, diffs, passes))
    candidates.sort(key=lambda x: -x[1])  # sort by speedup desc

    if not candidates:
        print("  (nothing to decide)")
        return
    print(f"  {'label':<22s} {'speedup':>8s}  {'max|Δ|':>8s}  {'ship?':>6s}")
    for r, sp, d, passes in candidates:
        mark = "✓" if passes else "✗"
        print(f"  {r.label:<22s} {sp:>7.2f}×  {d['max']:>8.3f}  {mark:>6s}")
    winners = [c for c in candidates if c[3]]
    print()
    if winners:
        best = winners[0]
        r = best[0]
        print(f"→ Best shippable: {r.label}")
        print(f"  set env:  KINETICS_PETSC_MATSOLVER={r.matsolver or '(unset)'}, "
              f"KINETICS_PETSC_JAC_LAG={r.jac_lag}, KINETICS_PETSC_PC_LAG={r.pc_lag}")
    else:
        print("→ No configuration meets the ship threshold. Keep current settings.")


if __name__ == "__main__":
    main()
