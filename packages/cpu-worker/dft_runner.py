"""PySCF-based DFT single-point computation for the cpu-worker DFT job kind.

For each Reaction we run PBE0/def2-TZVPP single-points on three geometries
(R from reactant_trajectory[-1], TS from ts_conformer_positions, P from
product_trajectory[-1]). For separated barriers we also need PBE0 energies of
the lowest-energy minimum of each participating compound — these are cached on
the Compound row (and on the Minimum row) so subsequent reactions touching the
same compound reuse the result.

Trajectory frame indexing: RelaxationTrajectory.positions[0] is the
TS-displaced starting frame of the IRC relaxation; positions[-1] is the
converged minimum (reactant or product). The in-box barrier wants the
relaxed minimum, so we use [-1].

Method matches upstream paper_experiments/pyscf_calc.py + 02_dft_energy_hessian.py:
  - functional: PBE0 (env DFT_METHOD)
  - basis:      def2-TZVPP (env DFT_BASIS)
  - SCF: conv_tol 1e-9, max_cycle 300; on failure retry once with mf.newton()
  - No Hessian / frequency check (DFT cheap enough for Hessians is wrong as
    often as the model — energies are where the leverage is).
"""

import os
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from packages.db.models import (
    Compound,
    DftWorkQueue,
    Minimum,
    Reaction,
    ReactionProduct,
    ReactionReactant,
)
from packages.db.serialization import (
    deserialize_ndarray,
    deserialize_trajectory,
)


# Defaults match upstream production setting (paper_experiments/pyscf_calc.py).
# Override with env vars to test cheaper methods (e.g. PBE0/def2-SVP).
DFT_METHOD = os.environ.get("DFT_METHOD", "PBE0")
DFT_BASIS = os.environ.get("DFT_BASIS", "def2-TZVPP")
DFT_MAX_MEMORY_MB = int(os.environ.get("DFT_MAX_MEMORY_MB", "8000"))


# ASE Hartree-to-eV constant; we don't import ase.units here to avoid pulling
# all of ASE just for one number.
HARTREE_EV = 27.211386245988


# Periodic table for atomic-number → symbol conversion
ELEMENT_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 35: "Br", 53: "I",
}


class SCFConvergenceError(RuntimeError):
    """Raised when SCF fails to converge even with the Newton fallback."""


def _guess_spin(atomic_numbers: np.ndarray, charge: int) -> int:
    """Guess 2S from electron count: 0 (singlet) or 1 (doublet)."""
    n_electrons = int(np.sum(atomic_numbers)) - charge
    return n_electrons % 2


def _pyscf_single_point(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    charge: int,
    method: str = DFT_METHOD,
    basis: str = DFT_BASIS,
    max_memory: int = DFT_MAX_MEMORY_MB,
) -> float:
    """Run a PBE0 (or other DFT) single-point. Returns energy in eV.

    On SCF failure, retries once with the Newton solver — matches upstream
    pyscf_calc.py:65-71. Raises SCFConvergenceError if both attempts fail.
    """
    from pyscf import gto, dft

    spin = _guess_spin(atomic_numbers, charge)

    mol = gto.Mole()
    mol.atom = [
        (ELEMENT_SYMBOLS.get(int(z), "X"), tuple(map(float, p)))
        for z, p in zip(atomic_numbers, positions)
    ]
    mol.basis = basis
    mol.charge = int(charge)
    mol.spin = spin
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.max_memory = max_memory
    mol.build()

    mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
    mf.xc = method
    mf.verbose = 0
    mf.max_cycle = 300
    mf.conv_tol = 1e-9
    energy_ha = mf.kernel()

    if not mf.converged:
        # Newton fallback (upstream pattern). The Newton solver is slower but
        # more robust for hard SCF cases (small gaps, near-degenerate states).
        logger.warning(f"SCF failed for {mol.atom}, retrying with Newton solver")
        mf2 = mf.newton()
        mf2.verbose = 0
        energy_ha = mf2.kernel()
        if not mf2.converged:
            raise SCFConvergenceError(
                f"SCF failed even with Newton fallback for "
                f"{mol.atom_pure_symbol(0) if mol.natm > 0 else '?'}*{mol.natm}"
            )

    return float(energy_ha) * HARTREE_EV


def _pyscf_ts_hessian(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    charge: int,
    method: str = DFT_METHOD,
    basis: str = DFT_BASIS,
    max_memory: int = DFT_MAX_MEMORY_MB,
) -> np.ndarray:
    """Compute the analytical PBE0 Hessian at the TS geometry.

    Returns the raw (3N, 3N) float64 matrix. No symmetrization — caller can
    enforce H = 0.5*(H+H.T) downstream if needed (PySCF's analytic output
    is symmetric to ~1e-10 already). Same SCF protocol as
    `_pyscf_single_point` (Newton fallback on failure).

    Wall time scales steeply with system size for hybrid functionals: ~5–15
    min per TS at N≈13–20 with def2-TZVPP. This is dataset-only post-hoc
    work — never call from the experiment hot-path.
    """
    from pyscf import gto, dft

    spin = _guess_spin(atomic_numbers, charge)

    mol = gto.Mole()
    mol.atom = [
        (ELEMENT_SYMBOLS.get(int(z), "X"), tuple(map(float, p)))
        for z, p in zip(atomic_numbers, positions)
    ]
    mol.basis = basis
    mol.charge = int(charge)
    mol.spin = spin
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.max_memory = max_memory
    mol.build()

    mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
    mf.xc = method
    mf.verbose = 0
    mf.max_cycle = 300
    mf.conv_tol = 1e-9
    mf.kernel()

    if not mf.converged:
        logger.warning("SCF did not converge for Hessian, retrying with Newton solver")
        mf = mf.newton()
        mf.verbose = 0
        mf.kernel()
        if not mf.converged:
            raise SCFConvergenceError(
                f"SCF failed even with Newton fallback for Hessian on N={mol.natm}"
            )

    # PySCF analytic Hessian: shape (natm, natm, 3, 3). Reshape to (3N, 3N)
    # by transposing so atom indices are outer, cartesian inner.
    h = mf.Hessian().kernel()
    natm = mol.natm
    h = np.asarray(h).reshape(natm, natm, 3, 3).transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)
    return h.astype(np.float64, copy=False)


# Threshold used to distinguish a real imaginary vibrational mode from
# numerical noise / hindered rotation in the DFT Hessian (in mass-weighted
# wavenumber units, cm⁻¹). 100 cm⁻¹ is the standard chemistry cutoff for
# "real imaginary frequency" in TS analysis — anything softer is treated
# as a hindered rotation, large-amplitude floppy mode, or numerical noise
# rather than a genuine vibrational degree of freedom.
TS_IMAGINARY_FREQ_THRESHOLD_CM = 100.0


def ts_ml_invalid_from_hessian(
    hessian_au: np.ndarray,
    positions_ang: np.ndarray,
    atomic_numbers: np.ndarray,
    charge: int = 0,
) -> bool:
    """Return True if the ML-predicted TS is *not* a first-order saddle at DFT.

    Procedure (chemistry-standard, matches PySCF's `harmonic_analysis`):
      1. Reshape the stored (3N, 3N) Hessian back to PySCF's (N, N, 3, 3).
      2. Mass-weight using natural-abundance isotope-averaged masses.
      3. Project out 3 translations + 3 rotations.
      4. Diagonalize → normal-mode frequencies in cm⁻¹ (imaginary modes
         returned by PySCF as complex with non-zero imag part).
      5. Count modes with |Im(ν)| > TS_IMAGINARY_FREQ_THRESHOLD_CM —
         a true first-order saddle has exactly one. Anything else
         (zero ⇒ local minimum, two-plus ⇒ higher-order saddle) is
         flagged invalid and routed to the corrected-TS optimization.

    Note: NOT identical to the exploration's Cartesian-eigenvalue check.
    Mass-weighting matters here because DFT Hessians have small
    Cartesian-negative numerical artifacts (basis incompleteness, grid
    noise, SCF residual) that vanish under mass-weighting — the
    explorer's algorithm worked for ML only because the ML Hessian
    doesn't produce those artifacts.
    """
    from pyscf import gto
    from pyscf.hessian import thermo

    z = np.asarray(atomic_numbers, dtype=int).flatten()
    natm = len(z)
    pos = np.asarray(positions_ang, dtype=np.float64)
    h = np.asarray(hessian_au, dtype=np.float64)
    # Stored layout: (3N, 3N) with rows/cols indexed as 3*atom + cart.
    # PySCF wants (N, N, 3, 3) with axes (atom_i, atom_j, cart_i, cart_j).
    h_butterfly = h.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)

    mol = gto.Mole()
    mol.atom = [
        (ELEMENT_SYMBOLS.get(int(zz), "X"), tuple(map(float, p)))
        for zz, p in zip(z, pos)
    ]
    mol.unit = "Angstrom"
    mol.charge = int(charge)
    # Spin = 2S = N_alpha - N_beta. For closed-shell ⇒ 0; for radicals ⇒ 1.
    nelec = int(z.sum()) - int(charge)
    mol.spin = nelec % 2
    mol.basis = "sto-3g"  # placeholder — harmonic_analysis only uses geometry + masses
    mol.build(verbose=0)

    result = thermo.harmonic_analysis(mol, h_butterfly)
    freqs = result["freq_wavenumber"]
    n_imag = sum(
        1 for f in freqs
        if np.iscomplex(f) and abs(f.imag) > TS_IMAGINARY_FREQ_THRESHOLD_CM
    )
    return n_imag != 1


def _ensure_compound_dft_energy(
    session: Session,
    compound_id: int,
    method: str,
    basis: str,
) -> Optional[float]:
    """Ensure the compound's reference (lowest-E minimum) has a DFT energy.

    If already cached, return it. Otherwise compute via PySCF, write to BOTH
    the Minimum row (per-conformer slot) AND the Compound row (denormalized
    fast-lookup cache), and return the new value.

    Returns None if compute fails (caller should mark the parent reaction job
    failed).
    """
    compound = session.query(Compound).filter(Compound.id == compound_id).first()
    if compound is None:
        return None

    if compound.energy_pbe0 is not None and compound.energy_pbe0_method == f"{method}/{basis}":
        return compound.energy_pbe0

    # Find lowest-E minimum
    minimum = (
        session.query(Minimum)
        .filter(Minimum.compound_id == compound_id)
        .order_by(Minimum.energy.asc())
        .first()
    )
    if minimum is None:
        logger.warning(
            f"compound {compound.smiles} (id={compound_id}) has no minima — "
            f"cannot compute reference DFT energy"
        )
        return None

    positions = deserialize_ndarray(minimum.positions)
    atomic_numbers = deserialize_ndarray(compound.sorted_atomic_numbers).flatten()
    charge = compound.charge

    t0 = time.time()
    try:
        energy_ev = _pyscf_single_point(positions, atomic_numbers, charge, method, basis)
    except SCFConvergenceError as e:
        logger.error(f"reference DFT failed for {compound.smiles}: {e}")
        return None

    elapsed = time.time() - t0
    logger.info(
        f"DFT reference: {compound.smiles} (charge={charge}, "
        f"atoms={len(atomic_numbers)}) → {energy_ev:.4f} eV in {elapsed:.1f}s"
    )

    method_str = f"{method}/{basis}"
    now = datetime.now(timezone.utc)

    # Write to Minimum row (per-conformer slot)
    minimum.energy_pbe0 = energy_ev
    minimum.energy_pbe0_method = method_str
    minimum.energy_pbe0_at = now

    # Write to Compound row (denormalized cache for fast model-builder lookup)
    compound.energy_pbe0 = energy_ev
    compound.energy_pbe0_method = method_str
    compound.energy_pbe0_at = now

    return energy_ev


def _trajectory_relaxed_frame(traj_bytes: Optional[bytes]) -> Optional[np.ndarray]:
    """Extract the relaxed-endpoint frame's positions from a serialized
    trajectory. RelaxationTrajectory[0] is the TS-displaced start; [-1] is
    the converged minimum, which is what we want for in-box DFT barriers."""
    if traj_bytes is None:
        return None
    traj = deserialize_trajectory(traj_bytes)
    if not traj or not traj.get("positions"):
        return None
    return traj["positions"][-1]


def run_dft_reaction_job(
    session: Session,
    work_id: int,
    reaction_id: int,
    method: str = DFT_METHOD,
    basis: str = DFT_BASIS,
    progress_cb: Optional[callable] = None,
) -> bool:
    """Process one DFT reaction job: cache compound refs, then compute R/TS/P
    energies, then derive and persist all four PBE0 barrier variants.

    Returns True on success, False on failure (caller marks queue accordingly).

    progress_cb: optional callable(task_desc: str) invoked between major steps
        so the caller can refresh the worker heartbeat (otherwise it goes
        stale during long PySCF calls and the worker disappears from the
        monitoring view). Each call also commits the session, so any
        already-computed compound DFT cache survives a mid-job crash.
    """
    reaction = session.query(Reaction).filter(Reaction.id == reaction_id).first()
    if reaction is None:
        logger.warning(f"DFT job {work_id}: reaction {reaction_id} not found")
        return False

    if reaction.discovery_method == "manual_equilibrium":
        # Defensive: claim filter should already exclude these.
        logger.info(f"DFT job {work_id}: skipping manual equilibrium reaction {reaction_id}")
        return True  # mark completed; nothing to do

    method_str = f"{method}/{basis}"
    now = datetime.now(timezone.utc)

    # 1. Compound reference energies (cached on Compound.energy_pbe0)
    reactant_rows = (
        session.query(ReactionReactant)
        .filter(ReactionReactant.reaction_id == reaction_id)
        .all()
    )
    product_rows = (
        session.query(ReactionProduct)
        .filter(ReactionProduct.reaction_id == reaction_id)
        .all()
    )

    # Keep multiplicities — a reaction with 2× C=O as products must sum
    # the reference energy twice. Use a set only for which compounds to compute.
    reactant_compound_ids = [r.compound_id for r in reactant_rows]
    product_compound_ids = [p.compound_id for p in product_rows]
    all_compound_ids = list(set(reactant_compound_ids + product_compound_ids))

    compound_dft_energies: dict[int, float] = {}
    for i, cid in enumerate(all_compound_ids):
        if progress_cb is not None:
            progress_cb(f"reaction {reaction_id}: compound ref {i+1}/{len(all_compound_ids)}")
        e = _ensure_compound_dft_energy(session, cid, method, basis)
        if e is None:
            logger.error(
                f"DFT job {work_id}: failed to obtain reference DFT energy for "
                f"compound {cid}, aborting reaction {reaction_id}"
            )
            return False
        compound_dft_energies[cid] = e

    # Flush the compound updates so they're visible to subsequent jobs even if
    # this reaction's TS step blows up.
    session.flush()

    # 2. TS single-point (in-box DFT). Skip if already cached at the same
    # method+basis (e.g. re-enqueued reactions where R/P were nulled out for
    # the trajectory-frame-bug retroactive cleanup).
    ts_positions = deserialize_ndarray(reaction.ts_conformer_positions)
    ts_anum = deserialize_ndarray(reaction.ts_conformer_atomic_numbers).flatten()
    ts_charge = reaction.ts_conformer_charge

    method_matches = reaction.energy_pbe0_method == method_str
    if reaction.energy_TS_pbe0 is not None and method_matches:
        e_ts = reaction.energy_TS_pbe0
        logger.info(f"DFT TS: reaction {reaction_id} → cached {e_ts:.4f} eV")
    else:
        if progress_cb is not None:
            progress_cb(f"reaction {reaction_id}: TS single-point")
        t0 = time.time()
        try:
            e_ts = _pyscf_single_point(ts_positions, ts_anum, ts_charge, method, basis)
        except SCFConvergenceError as e:
            logger.error(f"DFT job {work_id}: TS SCF failed for reaction {reaction_id}: {e}")
            return False
        logger.info(f"DFT TS: reaction {reaction_id} → {e_ts:.4f} eV in {time.time()-t0:.1f}s")

    # 3. R and P frame single-points (also in-box; for the dataset and barrier_*_pbe0)
    r_positions = _trajectory_relaxed_frame(reaction.reactant_trajectory)
    p_positions = _trajectory_relaxed_frame(reaction.product_trajectory)

    e_r = None
    e_p = None
    # Reactant in-box: same atomic_numbers as TS (the trajectory uses the same
    # atom set since IRC is performed on the in-box geometry).
    if reaction.energy_R_pbe0 is not None and method_matches:
        e_r = reaction.energy_R_pbe0
    elif r_positions is not None:
        if progress_cb is not None:
            progress_cb(f"reaction {reaction_id}: R-frame single-point")
        try:
            e_r = _pyscf_single_point(r_positions, ts_anum, ts_charge, method, basis)
        except SCFConvergenceError as e:
            logger.warning(
                f"DFT job {work_id}: R-frame SCF failed for reaction {reaction_id}: {e}"
            )
    if reaction.energy_P_pbe0 is not None and method_matches:
        e_p = reaction.energy_P_pbe0
    elif p_positions is not None:
        if progress_cb is not None:
            progress_cb(f"reaction {reaction_id}: P-frame single-point")
        try:
            e_p = _pyscf_single_point(p_positions, ts_anum, ts_charge, method, basis)
        except SCFConvergenceError as e:
            logger.warning(
                f"DFT job {work_id}: P-frame SCF failed for reaction {reaction_id}: {e}"
            )

    # 4. Derived barriers
    bf_pbe0 = (e_ts - e_r) if e_r is not None else None
    bb_pbe0 = (e_ts - e_p) if e_p is not None else None

    # Separated barriers — sum of reactant/product compound DFT references
    reactant_ref_sum = sum(compound_dft_energies[cid] for cid in reactant_compound_ids)
    product_ref_sum = sum(compound_dft_energies[cid] for cid in product_compound_ids)
    bf_sep_pbe0 = e_ts - reactant_ref_sum
    bb_sep_pbe0 = e_ts - product_ref_sum

    # 5. Persist on the Reaction row
    reaction.energy_R_pbe0 = e_r
    reaction.energy_TS_pbe0 = e_ts
    reaction.energy_P_pbe0 = e_p
    reaction.barrier_forward_pbe0 = bf_pbe0
    reaction.barrier_backward_pbe0 = bb_pbe0
    reaction.barrier_forward_separated_pbe0 = bf_sep_pbe0
    reaction.barrier_backward_separated_pbe0 = bb_sep_pbe0
    reaction.energy_pbe0_method = method_str
    reaction.energy_pbe0_at = now

    logger.info(
        f"DFT job {work_id} complete for reaction {reaction_id}: "
        f"bf_sep_pbe0={bf_sep_pbe0:.3f} eV, bb_sep_pbe0={bb_sep_pbe0:.3f} eV"
    )
    return True
