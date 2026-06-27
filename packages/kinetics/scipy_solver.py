"""ODE solver for the kinetics system.

Uses PETSc BDF (matching the reference implementation) when available,
with scipy BDF as fallback. Both use the same numba-compiled analytical
RHS and Jacobian kernels.

PETSc install recipe for the API Dockerfile:
  apt-get install libpetsc-real3.22-dev libopenmpi-dev
  pip install Cython==3.0.11 mpi4py
  PETSC_DIR=... PETSC_ARCH='' pip install --no-build-isolation petsc4py==3.22.2
"""

import math
from typing import Callable

import numpy as np
from loguru import logger
from scipy import sparse

from packages.kinetics.model import ODEModel


def _prepare_arrays(model: ODEModel):
    """Flatten ODEModel stoichiometry into arrays for numba kernels."""
    from packages.kinetics.numba_kernels import compute_rhs, compute_jacobian

    n_species = model.n_species
    n_reactions = model.n_reactions
    k_fwd = model.k_fwd.astype(np.float64)
    k_bwd = model.k_bwd.astype(np.float64)
    y0 = model.initial_concs.astype(np.float64)

    rs_list, rsc_list, rp_list = [], [], [0]
    ps_list, psc_list, pp_list = [], [], [0]
    for j in range(n_reactions):
        for idx, count in model.reactant_stoich[j]:
            rs_list.append(idx)
            rsc_list.append(int(count))
        rp_list.append(len(rs_list))
        for idx, count in model.product_stoich[j]:
            ps_list.append(idx)
            psc_list.append(int(count))
        pp_list.append(len(ps_list))

    rs = np.array(rs_list, np.int32)
    rsc = np.array(rsc_list, np.int32)
    rp = np.array(rp_list, np.int32)
    ps = np.array(ps_list, np.int32)
    psc = np.array(psc_list, np.int32)
    pp = np.array(pp_list, np.int32)

    # S_net in CSC
    S_r = sparse.lil_matrix((n_species, n_reactions))
    S_p = sparse.lil_matrix((n_species, n_reactions))
    for j in range(n_reactions):
        for idx, count in model.reactant_stoich[j]:
            S_r[idx, j] += count
        for idx, count in model.product_stoich[j]:
            S_p[idx, j] += count
    S_net = (S_p - S_r).tocsc()
    S_data = S_net.data.astype(np.float64)
    S_indices = S_net.indices.astype(np.int32)
    S_indptr = S_net.indptr.astype(np.int32)

    # Jacobian sparsity
    jac_sp = sparse.lil_matrix((n_species, n_species))
    for j in range(n_reactions):
        col_j = S_net[:, j].nonzero()[0]
        for idx, _ in model.reactant_stoich[j]:
            for i in col_j:
                jac_sp[i, idx] = 1
        for idx, _ in model.product_stoich[j]:
            for i in col_j:
                jac_sp[i, idx] = 1
    jac_csr = jac_sp.tocsr()
    jac_indptr = jac_csr.indptr.astype(np.int32)
    jac_indices = jac_csr.indices.astype(np.int32)
    jac_nnz = jac_csr.nnz

    # Column-flattened S_net for Jacobian kernel
    snet_col_idx, snet_col_val, snet_col_ptr = [], [], [0]
    for j in range(n_reactions):
        col = S_net[:, j]
        rows = col.nonzero()[0]
        vals = np.array(col[rows].todense()).flatten() if len(rows) else np.zeros(0)
        for r, v in zip(rows, vals):
            snet_col_idx.append(int(r))
            snet_col_val.append(float(v))
        snet_col_ptr.append(len(snet_col_idx))
    snet_col_idx_arr = np.array(snet_col_idx, np.int32)
    snet_col_val_arr = np.array(snet_col_val, np.float64)
    snet_col_ptr_arr = np.array(snet_col_ptr, np.int32)

    # Lookup table
    jac_lookup = -np.ones((n_species, n_species), dtype=np.int32)
    for row in range(n_species):
        for idx in range(jac_indptr[row], jac_indptr[row + 1]):
            col = jac_indices[idx]
            jac_lookup[row, col] = idx

    # Warmup numba
    _out = np.zeros(n_species)
    compute_rhs(y0, _out, k_fwd, k_bwd, rs, rsc, rp, ps, psc, pp,
                S_data, S_indices, S_indptr, n_species, n_reactions)
    if jac_nnz > 0:
        _jd = np.zeros(jac_nnz)
        compute_jacobian(y0, _jd, k_fwd, k_bwd, rs, rsc, rp, ps, psc, pp,
                         snet_col_idx_arr, snet_col_val_arr, snet_col_ptr_arr,
                         jac_lookup, n_species, n_reactions)

    return dict(
        n_species=n_species, n_reactions=n_reactions,
        k_fwd=k_fwd, k_bwd=k_bwd, y0=y0,
        rs=rs, rsc=rsc, rp=rp, ps=ps, psc=psc, pp=pp,
        S_data=S_data, S_indices=S_indices, S_indptr=S_indptr, S_net=S_net,
        jac_indptr=jac_indptr, jac_indices=jac_indices, jac_nnz=jac_nnz,
        snet_col_idx=snet_col_idx_arr, snet_col_val=snet_col_val_arr,
        snet_col_ptr=snet_col_ptr_arr, jac_lookup=jac_lookup,
        compute_rhs=compute_rhs, compute_jacobian=compute_jacobian,
    )


def _solve_petsc(arrays: dict, t_max: float) -> Callable[[float], np.ndarray]:
    """PETSc BDF solver — matches reference implementation exactly."""
    import petsc4py
    petsc4py.init([])
    from petsc4py import PETSc
    from packages.kinetics.constants import DECADE_TIMES

    a = arrays
    n_species, n_reactions = a["n_species"], a["n_reactions"]
    y0 = a["y0"]

    y_petsc = PETSc.Vec().createSeq(n_species)
    y_petsc.setArray(y0.copy())
    f_petsc = PETSc.Vec().createSeq(n_species)

    J = PETSc.Mat().createAIJ(
        size=(n_species, n_species),
        csr=(a["jac_indptr"], a["jac_indices"], np.zeros(a["jac_nnz"])),
    )
    J.setUp()

    def rhs_function(ts, t, y, f):
        y_arr = y.getArray(readonly=True)
        f_arr = f.getArray(readonly=False)
        a["compute_rhs"](y_arr, f_arr, a["k_fwd"], a["k_bwd"],
                         a["rs"], a["rsc"], a["rp"], a["ps"], a["psc"], a["pp"],
                         a["S_data"], a["S_indices"], a["S_indptr"],
                         n_species, n_reactions)

    # Preallocate the Jacobian-data buffer and reuse across evals.  The old
    # code did `np.zeros(jac_nnz)` every call; with a ~10 k-nnz Jacobian and
    # thousands of evals per solve that's a lot of wasted allocator work.
    jac_data_buf = np.zeros(a["jac_nnz"], dtype=np.float64)
    jac_indptr_arr = a["jac_indptr"]
    jac_indices_arr = a["jac_indices"]

    def jac_function(ts, t, y, Jmat, Pmat):
        y_arr = y.getArray(readonly=True)
        jac_data_buf.fill(0.0)
        a["compute_jacobian"](y_arr, jac_data_buf, a["k_fwd"], a["k_bwd"],
                              a["rs"], a["rsc"], a["rp"], a["ps"], a["psc"], a["pp"],
                              a["snet_col_idx"], a["snet_col_val"], a["snet_col_ptr"],
                              a["jac_lookup"], n_species, n_reactions)
        # One-shot CSR assembly — avoids the 2000× per-row Python→C hop of
        # the old setValues loop.  The matrix was created with this exact
        # sparsity pattern (`createAIJ(csr=...)`), so setValuesCSR just
        # overwrites the existing value buffer in place.
        Pmat.zeroEntries()
        Pmat.setValuesCSR(jac_indptr_arr, jac_indices_arr, jac_data_buf)
        Pmat.assemble()
        if Jmat != Pmat:
            Jmat.assemble()

    eval_times = sorted(set(
        np.geomspace(1e-12, t_max, 400).tolist() + list(DECADE_TIMES)
    ))
    eval_idx = [0]
    snapshots = {}

    def monitor(ts, step, t, y):
        while eval_idx[0] < len(eval_times) and eval_times[eval_idx[0]] <= t:
            snapshots[eval_times[eval_idx[0]]] = y.getArray(readonly=True).copy()
            eval_idx[0] += 1

    ts = PETSc.TS().create()
    ts.setType("bdf")
    ts.setRHSFunction(rhs_function, f_petsc)
    ts.setRHSJacobian(jac_function, J, J)
    ts.setTime(0.0)
    ts.setMaxTime(t_max)
    import os
    petsc_dt0 = float(os.environ.get("KINETICS_PETSC_DT0", "1e-10"))
    ts.setTimeStep(petsc_dt0)
    ts.setMaxSteps(50_000_000)
    petsc_atol = float(os.environ.get("KINETICS_PETSC_ATOL", "1e-16"))
    petsc_rtol = float(os.environ.get("KINETICS_PETSC_RTOL", "1e-12"))
    ts.setTolerances(atol=petsc_atol, rtol=petsc_rtol)
    ts.setExactFinalTime(PETSc.TS.ExactFinalTime.INTERPOLATE)
    ts.setMaxSNESFailures(-1)
    ts.setMonitor(monitor)

    snes = ts.getSNES()
    ksp = snes.getKSP()
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    # Optional: pick a specific sparse factor backend (mumps / superlu /
    # klu / umfpack / superlu_dist).  Default "" lets PETSc pick, which
    # typically means its built-in sequential LU.  MUMPS is usually the
    # fastest direct solver for stiff chemistry Jacobians at this size.
    petsc_matsolver = os.environ.get("KINETICS_PETSC_MATSOLVER", "")
    if petsc_matsolver:
        try:
            pc.setFactorSolverType(petsc_matsolver)
        except Exception as e:
            logger.warning(f"setFactorSolverType({petsc_matsolver!r}) failed: {e} — defaulting")
    # Optional lag knobs — trade (re)compute wall time for slower Newton
    # convergence.  Two independent levels:
    #   JAC_LAG:  reuse the Jacobian matrix for N SNES iterations before
    #             recomputing it via the callback.
    #   PC_LAG:   reuse the LU factorization of the Jacobian for N SNES
    #             iterations.  In our preonly+LU setup the factorization
    #             IS the expensive step, so this is usually the bigger
    #             lever.
    # Defaults (1,1) = current behavior — no lag.  The A/B bench at
    # scripts/bench/bench_solver.py measures the trade-off empirically
    # against a restored prod snapshot.
    petsc_jac_lag = int(os.environ.get("KINETICS_PETSC_JAC_LAG", "1"))
    petsc_pc_lag = int(os.environ.get("KINETICS_PETSC_PC_LAG", "1"))
    # Route lag knobs through PETSc Options rather than SNES methods —
    # petsc4py 3.22.2's SNES doesn't bind setLagPreconditioner, but the
    # -snes_lag_preconditioner command-line option works universally once
    # ts.setFromOptions() has been called (below).
    if petsc_jac_lag > 1:
        PETSc.Options().setValue("-snes_lag_jacobian", str(petsc_jac_lag))
    if petsc_pc_lag > 1:
        PETSc.Options().setValue("-snes_lag_preconditioner", str(petsc_pc_lag))
    ts.setFromOptions()

    try:
        ts.solve(y_petsc)
    except Exception as e:
        logger.warning(f"PETSc partial solve: {e}")

    t_final = ts.getTime()
    if t_final < 1e7:
        logger.warning(f"PETSc only reached t={t_final:.2e}")

    # Per-solve profiling — prints TS/KSP summary with per-stage wall time,
    # # of Jacobian evaluations, linear solve stats.  Gated by env var so
    # prod doesn't spam every solve; set KINETICS_PETSC_VIEW=1 during
    # benchmarking.
    if os.environ.get("KINETICS_PETSC_VIEW", "") == "1":
        try:
            ts.view()
            ksp.view()
        except Exception as e:
            logger.warning(f"ts/ksp view failed: {e}")

    y_final = y_petsc.getArray().copy()
    while eval_idx[0] < len(eval_times):
        snapshots[eval_times[eval_idx[0]]] = y_final.copy()
        eval_idx[0] += 1

    ts.destroy()
    J.destroy()
    y_petsc.destroy()
    f_petsc.destroy()

    if not snapshots:
        raise RuntimeError("PETSc: no time points solved")

    snap_times = sorted(snapshots.keys())
    snap_matrix = np.column_stack([snapshots[t] for t in snap_times])
    log_snap_times = np.log10(np.maximum(np.array(snap_times), 1e-30))

    def sol_fn(t: float) -> np.ndarray:
        if t <= 0:
            return y0.copy()
        log_t = math.log10(t)
        idx = np.searchsorted(log_snap_times, log_t)
        idx = min(max(idx, 0), len(log_snap_times) - 1)
        return np.maximum(snap_matrix[:, idx], 0.0)

    return sol_fn


def _solve_scipy(arrays: dict, t_max: float) -> Callable[[float], np.ndarray]:
    """Scipy BDF fallback when PETSc is not available."""
    from scipy.integrate import solve_ivp

    a = arrays
    n_species = a["n_species"]
    y0 = a["y0"]

    def rhs(t, y):
        out = np.zeros(n_species)
        a["compute_rhs"](y, out, a["k_fwd"], a["k_bwd"],
                         a["rs"], a["rsc"], a["rp"], a["ps"], a["psc"], a["pp"],
                         a["S_data"], a["S_indices"], a["S_indptr"],
                         n_species, a["n_reactions"])
        return out

    jac_dense = np.zeros((n_species, n_species))

    def jac(t, y):
        if a["jac_nnz"] == 0:
            jac_dense[:] = 0.0
            return jac_dense
        jac_data = np.zeros(a["jac_nnz"])
        a["compute_jacobian"](y, jac_data, a["k_fwd"], a["k_bwd"],
                              a["rs"], a["rsc"], a["rp"], a["ps"], a["psc"], a["pp"],
                              a["snet_col_idx"], a["snet_col_val"], a["snet_col_ptr"],
                              a["jac_lookup"], n_species, a["n_reactions"])
        jac_dense[:] = 0.0
        for row in range(n_species):
            for k in range(a["jac_indptr"][row], a["jac_indptr"][row + 1]):
                jac_dense[row, a["jac_indices"][k]] = jac_data[k]
        return jac_dense

    sol = solve_ivp(
        rhs, (0.0, t_max), y0,
        method="BDF", jac=jac, dense_output=True,
        rtol=1e-10, atol=1e-12, first_step=1e-10,
    )

    if not sol.success:
        logger.warning(f"scipy BDF did not succeed: {sol.message}")

    t_final = float(sol.t[-1]) if len(sol.t) > 0 else 0.0
    if t_final < 1e7:
        logger.warning(f"scipy BDF only reached t={t_final:.2e}")

    dense_out = sol.sol

    def sol_fn(t: float) -> np.ndarray:
        if t <= 0 or dense_out is None:
            return y0.copy()
        t_clamped = min(max(t, float(sol.t[0])), float(sol.t[-1]))
        return np.maximum(dense_out(t_clamped), 0.0)

    return sol_fn


def solve_ode(model: ODEModel, t_max: float = 1e8) -> Callable[[float], np.ndarray]:
    """Solve the mass-action ODE. Uses PETSc BDF if available, scipy BDF otherwise."""
    if model.n_species == 0 or model.n_reactions == 0:
        raise RuntimeError("Cannot solve empty ODE system")

    import time as _time
    _t0 = _time.perf_counter()
    arrays = _prepare_arrays(model)
    logger.info(f"solve_ode: arrays prepared in {_time.perf_counter()-_t0:.1f}s (n_species={model.n_species}, n_reactions={model.n_reactions})")

    try:
        _t1 = _time.perf_counter()
        sol_fn = _solve_petsc(arrays, t_max)
        logger.info(f"Kinetics solved with PETSc BDF in {_time.perf_counter()-_t1:.1f}s")
        return sol_fn
    except ImportError:
        logger.info("petsc4py not available, falling back to scipy BDF")
    except Exception as e:
        logger.warning(f"PETSc solver failed ({e}), falling back to scipy BDF")

    sol_fn = _solve_scipy(arrays, t_max)
    logger.info("Kinetics solved with scipy BDF")
    return sol_fn
