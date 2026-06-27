"""Numba-compiled mass-action ODE RHS and analytical Jacobian.

Verbatim port of upstream crn-exploration/lib/kinetic_sampler.py:58-126.
These kernels are hot-path — compiled once (cache=True persists across
container restarts) and called by PETSc on every integration step. Do not
restructure unless you profile first.

Array layout (all int32 unless noted):
  y[n_species]                              species concentrations (float64)
  out[n_species]                            RHS output (float64)
  kf[n_reactions], kb[n_reactions]          rate constants (float64)
  rs[], rsc[]                               flat reactant (species, count) pairs
  rp[n_reactions+1]                         offsets into rs/rsc
  ps[], psc[], pp[n_reactions+1]            same for products
  S_data[nnz], S_indices[nnz], S_indptr[n_reactions+1]
                                             CSC of the net stoichiometry matrix
                                             (S_product - S_reactant)
  jac_data[nnz]                             output CSR data array
  snet_col_idx/val/ptr                      column-flattened S_net used by the Jacobian
  jac_lookup[n_species, n_species]          (row, col) -> position in jac_data
"""

try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    # Fallback: noop decorator so the module still imports for type-checking
    # and local development without numba. Production (API container) always
    # has numba installed — see Phase 8 requirements.txt additions.
    _HAVE_NUMBA = False
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def wrapper(f):
            return f
        return wrapper


@njit(cache=True)
def compute_rhs(y, out, kf, kb, rs, rsc, rp, ps, psc, pp, Sd, Si, Sp, nsp, nrx):
    """Mass-action ODE right-hand side: dy/dt = S_net @ rates."""
    out[:] = 0.0
    for j in range(nrx):
        f = kf[j]
        for k in range(rp[j], rp[j + 1]):
            if rsc[k] == 1:
                f *= y[rs[k]]
            else:
                f *= y[rs[k]] ** rsc[k]
        b = kb[j]
        for k in range(pp[j], pp[j + 1]):
            if psc[k] == 1:
                b *= y[ps[k]]
            else:
                b *= y[ps[k]] ** psc[k]
        rate = f - b
        for k in range(Sp[j], Sp[j + 1]):
            out[Si[k]] += Sd[k] * rate


@njit(cache=True)
def compute_jacobian(y, jac_data, kf, kb, rs, rsc, rp, ps, psc, pp,
                     snet_idx, snet_val, snet_ptr, jac_lookup, nsp, nrx):
    """Analytical Jacobian: J[i,k] = d(dy_i/dt)/dy_k, written into CSR data array."""
    jac_data[:] = 0.0
    for j in range(nrx):
        # Forward rate derivatives w.r.t. each reactant
        for ki in range(rp[j], rp[j + 1]):
            k_idx = rs[ki]
            deriv = kf[j]
            for qi in range(rp[j], rp[j + 1]):
                q_idx = rs[qi]
                q_s = rsc[qi]
                if qi == ki:
                    if q_s != 1:
                        deriv *= q_s * max(y[q_idx], 0.0) ** (q_s - 1)
                else:
                    if q_s == 1:
                        deriv *= y[q_idx]
                    else:
                        deriv *= y[q_idx] ** q_s
            for si in range(snet_ptr[j], snet_ptr[j + 1]):
                row = snet_idx[si]
                loc = jac_lookup[row, k_idx]
                if loc >= 0:
                    jac_data[loc] += snet_val[si] * deriv

        # Backward rate derivatives w.r.t. each product (subtracted)
        for ki in range(pp[j], pp[j + 1]):
            k_idx = ps[ki]
            deriv = kb[j]
            for qi in range(pp[j], pp[j + 1]):
                q_idx = ps[qi]
                q_s = psc[qi]
                if qi == ki:
                    if q_s != 1:
                        deriv *= q_s * max(y[q_idx], 0.0) ** (q_s - 1)
                else:
                    if q_s == 1:
                        deriv *= y[q_idx]
                    else:
                        deriv *= y[q_idx] ** q_s
            for si in range(snet_ptr[j], snet_ptr[j + 1]):
                row = snet_idx[si]
                loc = jac_lookup[row, k_idx]
                if loc >= 0:
                    jac_data[loc] -= snet_val[si] * deriv
