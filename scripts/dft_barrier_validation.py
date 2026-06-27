#!/usr/bin/env python3
"""Validate ML energy barriers against DFT single-point energies.

Samples random reactions from the pickle state, computes DFT energies
at reactant/TS/product geometries, and compares barriers.

Uses PySCF for DFT (B3LYP/def2-SVP or r2SCAN/def2-SVP).

Usage:
    # Inside Docker (crn-cloud-worker with pyscf installed):
    python scripts/dft_barrier_validation.py test_data/n111/ --n-reactions 50 --n-pes 50

    # With custom DFT level:
    python scripts/dft_barrier_validation.py test_data/n111/ --functional b3lyp --basis def2-svp
"""

import os
import sys
import pickle
import random
import time
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Stub classes for unpickling ──
import types as pytypes

class _Stub:
    def __setstate__(self, state):
        self.__dict__.update(state)

class _StubModule(pytypes.ModuleType):
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        cls = type(name, (_Stub,), {})
        setattr(self, name, cls)
        return cls

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


# ── DFT Engine ──

HARTREE_TO_EV = 27.211386245988

def dft_energy(atomic_numbers, positions, charge=0, multiplicity=1,
               functional="b3lyp", basis="def2-svp"):
    """Compute DFT single-point energy using PySCF.

    Args:
        atomic_numbers: array of atomic numbers
        positions: (n_atoms, 3) in Angstrom
        charge: molecular charge
        multiplicity: spin multiplicity (1=singlet, 2=doublet, ...)
        functional: DFT functional
        basis: basis set

    Returns:
        Energy in eV
    """
    from pyscf import gto, dft, scf, cc, mp

    # Build atom string for PySCF
    element_map = {1: "H", 6: "C", 8: "O", 7: "N"}
    atom_str = ""
    for z, pos in zip(atomic_numbers, positions):
        sym = element_map.get(int(z), f"X{int(z)}")
        atom_str += f"{sym} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}; "

    # Determine total electrons and fix spin if needed
    total_z = sum(int(z) for z in atomic_numbers)
    n_electrons = total_z - charge
    spin = (n_electrons % 2)  # 0 for even, 1 for odd (doublet)

    mol = gto.M(
        atom=atom_str,
        basis=basis,
        charge=charge,
        spin=spin,
        verbose=0,
    )

    is_open_shell = spin > 0

    try:
        if functional.lower() in ("ccsd", "ccsd_t", "ccsd(t)", "mp2"):
            # Post-HF methods
            mf_hf = scf.UHF(mol) if is_open_shell else scf.RHF(mol)
            mf_hf.conv_tol = 1e-10
            mf_hf.max_cycle = 200
            mf_hf.verbose = 0
            mf_hf.kernel()
            if not mf_hf.converged:
                print(f"  WARNING: HF not converged (charge={charge})")
                return None

            if functional.lower() == "mp2":
                pt = mp.UMP2(mf_hf) if is_open_shell else mp.MP2(mf_hf)
                pt.verbose = 0
                pt.kernel()
                return pt.e_tot * HARTREE_TO_EV

            mycc = cc.UCCSD(mf_hf) if is_open_shell else cc.CCSD(mf_hf)
            mycc.verbose = 0
            mycc.kernel()
            if not mycc.converged:
                print(f"  WARNING: CCSD not converged")
                return None

            if functional.lower() in ("ccsd_t", "ccsd(t)"):
                et = mycc.ccsd_t()
                return (mycc.e_tot + et) * HARTREE_TO_EV
            else:
                return mycc.e_tot * HARTREE_TO_EV
        else:
            # DFT
            mf = dft.UKS(mol) if is_open_shell else dft.RKS(mol)
            mf.xc = functional
            mf.conv_tol = 1e-8
            mf.max_cycle = 200
            mf.verbose = 0

            energy_hartree = mf.kernel()
            if not mf.converged:
                print(f"  WARNING: SCF not converged (charge={charge})")
                return None
            return energy_hartree * HARTREE_TO_EV

    except Exception as e:
        print(f"  DFT/post-HF failed: {e}")
        return None


# ── Data extraction ──

@dataclass
class BarrierSample:
    """A single barrier comparison point."""
    source: str           # "pes" or "reaction"
    compound: str
    label: str
    ml_barrier: float     # ML model barrier (eV)
    dft_barrier: float    # DFT barrier (eV)
    ml_energies: dict     # {point_name: energy_eV}
    dft_energies: dict    # {point_name: energy_eV}
    n_atoms: int
    charge: int


def integrate_trajectory(traj):
    """Integrate forces along a trajectory. Returns (trap_barrier, hess_barrier).

    Trajectory goes TS → endpoint, so barrier = -(integrated work).
    Returns (None, None) if trajectory lacks data.
    """
    positions = getattr(traj, "positions", None)
    forces = getattr(traj, "forces", None)
    hessians = getattr(traj, "hessians", None)

    if not positions or not forces or len(positions) < 2:
        return None, None

    trap_int = 0.0
    for j in range(len(positions) - 1):
        dr = positions[j + 1] - positions[j]
        f_avg = 0.5 * (forces[j] + forces[j + 1])
        trap_int += -np.sum(f_avg * dr)
    trap_barrier = -trap_int

    hess_barrier = None
    if hessians and any(h is not None for h in hessians):
        hess_int = 0.0
        for j in range(len(positions) - 1):
            dr = positions[j + 1] - positions[j]
            dr_flat = dr.flatten()
            if hessians[j] is not None:
                H = hessians[j]
                F = forces[j].flatten()
                hess_int += -np.dot(F, dr_flat) + 0.5 * np.dot(dr_flat, H @ dr_flat)
            else:
                f_avg = 0.5 * (forces[j] + forces[j + 1])
                hess_int += -np.sum(f_avg * dr)
        hess_barrier = -hess_int

    return trap_barrier, hess_barrier


def print_barrier_table(dft_barrier, ml_barrier, trap_barrier, hess_barrier, label_suffix=""):
    """Print a comparison table for one direction."""
    s = label_suffix
    print(f"  {'Method':<25} {'Barrier(eV)':>12} {'(kcal/mol)':>12} {'vs DFT(eV)':>12} {'vs DFT(kcal)':>12}")
    print(f"  {'-'*73}")
    print(f"  {'DFT'+s:.<25} {dft_barrier:>12.4f} {dft_barrier*23.06:>12.2f} {'(ref)':>12} {'(ref)':>12}")
    if ml_barrier is not None:
        d = ml_barrier - dft_barrier
        print(f"  {'ML energy'+s:.<25} {ml_barrier:>12.4f} {ml_barrier*23.06:>12.2f} {d:>+12.4f} {d*23.06:>+12.2f}")
    if trap_barrier is not None:
        d = trap_barrier - dft_barrier
        print(f"  {'Trap. integ.'+s:.<25} {trap_barrier:>12.4f} {trap_barrier*23.06:>12.2f} {d:>+12.4f} {d*23.06:>+12.2f}")
    if hess_barrier is not None:
        d = hess_barrier - dft_barrier
        print(f"  {'Hessian-corr.'+s:.<25} {hess_barrier:>12.4f} {hess_barrier*23.06:>12.2f} {d:>+12.4f} {d*23.06:>+12.2f}")


def extract_pes_samples(state, n_samples=50):
    """Extract random PES TS barriers (intra-molecular transitions)."""
    compounds = state["compound_registry"]._compounds

    # Collect all PES TSs with trajectory data
    all_ts = []
    for name, comp in compounds.items():
        pg = comp.pes_graph
        minima = getattr(pg, "minima", {}) or getattr(pg, "_minima", {})
        tss = getattr(pg, "transition_states", {}) or getattr(pg, "_transition_states", {})
        anum = getattr(pg, "atomic_numbers", None)
        charge = getattr(comp, "charge", 0) or 0

        for ts_id, ts in tss.items():
            ts_energy = getattr(ts, "energy", None)
            ts_positions = getattr(ts, "positions", None)
            fwd_id = getattr(ts, "min_fwd_id", None)
            bwd_id = getattr(ts, "min_bwd_id", None)

            if ts_energy is None or ts_positions is None:
                continue
            if fwd_id is None or bwd_id is None:
                continue

            fwd_min = minima.get(fwd_id)
            bwd_min = minima.get(bwd_id)
            if fwd_min is None or bwd_min is None:
                continue

            # Get trajectories for integration comparison
            fwd_traj = getattr(ts, "fwd_trajectory", None)
            bwd_traj = getattr(ts, "bwd_trajectory", None)

            all_ts.append({
                "compound": name,
                "ts_id": ts_id,
                "ts_positions": ts_positions,
                "ts_energy": ts_energy,
                "fwd_positions": fwd_min.positions,
                "fwd_energy": fwd_min.energy,
                "bwd_positions": bwd_min.positions,
                "bwd_energy": bwd_min.energy,
                "atomic_numbers": anum,
                "charge": charge,
                "fwd_traj": fwd_traj,
                "bwd_traj": bwd_traj,
            })

    if len(all_ts) > n_samples:
        all_ts = random.sample(all_ts, n_samples)

    return all_ts


def extract_reaction_samples(state, n_samples=50):
    """Extract random inter-molecular reaction barriers."""
    import torch
    rr = state["reaction_registry"]
    reactions = getattr(rr, "_reactions", {})

    all_rxns = []
    for rxn_id, rxn in reactions.items():
        # TS conformer
        ts_conf = getattr(rxn, "ts_conformer", None)
        if ts_conf is None:
            continue
        ts_positions = ts_conf.positions
        if isinstance(ts_positions, torch.Tensor):
            ts_positions = ts_positions.detach().cpu().numpy()
        ts_anum = ts_conf.atomic_numbers
        if isinstance(ts_anum, torch.Tensor):
            ts_anum = ts_anum.detach().cpu().numpy()
        ts_energy = getattr(ts_conf, "energy", None) or getattr(rxn, "barrier_forward", None)
        charge = getattr(ts_conf, "charge", 0) or 0

        if ts_energy is None or ts_positions is None:
            continue

        # Reactant: use trajectory endpoint (last frame = relaxed minimum)
        fwd_traj = getattr(rxn, "reactant_trajectory", None)
        bwd_traj = getattr(rxn, "product_trajectory", None)

        reactant_positions = None
        reactant_energy = None
        product_positions = None
        product_energy = None

        if fwd_traj is not None and hasattr(fwd_traj, "positions") and len(fwd_traj.positions) > 0:
            reactant_positions = fwd_traj.positions[-1]
            if hasattr(fwd_traj, "energies") and fwd_traj.energies:
                reactant_energy = fwd_traj.energies[-1]

        if bwd_traj is not None and hasattr(bwd_traj, "positions") and len(bwd_traj.positions) > 0:
            product_positions = bwd_traj.positions[-1]
            if hasattr(bwd_traj, "energies") and bwd_traj.energies:
                product_energy = bwd_traj.energies[-1]

        if reactant_positions is None or reactant_energy is None:
            continue

        # ML barriers
        ml_barrier_fwd = getattr(rxn, "barrier_forward", None)
        ml_barrier_bwd = getattr(rxn, "barrier_backward", None)

        all_rxns.append({
            "rxn_id": rxn_id,
            "name": getattr(rxn, "name", f"rxn-{rxn_id}"),
            "ts_positions": ts_positions,
            "ts_energy": ts_conf.energy,
            "ts_anum": ts_anum,
            "reactant_positions": reactant_positions,
            "reactant_energy": reactant_energy,
            "product_positions": product_positions,
            "product_energy": product_energy,
            "ml_barrier_fwd": ml_barrier_fwd,
            "ml_barrier_bwd": ml_barrier_bwd,
            "charge": charge,
            "fwd_traj": fwd_traj,
            "bwd_traj": bwd_traj,
            "discovery_method": getattr(rxn, "discovery_method", "?"),
        })

    if len(all_rxns) > n_samples:
        all_rxns = random.sample(all_rxns, n_samples)

    return all_rxns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?", default="test_data/n111")
    parser.add_argument("--n-pes", type=int, default=50, help="Number of PES barriers to sample")
    parser.add_argument("--n-reactions", type=int, default=50, help="Number of reaction barriers to sample")
    parser.add_argument("--functional", default="b3lyp", help="DFT functional")
    parser.add_argument("--basis", default="def2-svp", help="Basis set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    import builtins
    _orig_print = builtins.print
    def print(*args, **kwargs):
        kwargs.setdefault("flush", True)
        _orig_print(*args, **kwargs)

    random.seed(args.seed)
    data_dir = Path(args.data_dir)
    pkl_path = data_dir / ".reaction_graph_state.pkl"

    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        state = pickle.load(f)

    print(f"DFT level: {args.functional}/{args.basis}")
    print()

    # ── PES Barriers ──
    print("=" * 100)
    print("PES BARRIERS (intramolecular)")
    print("=" * 100)

    pes_samples = extract_pes_samples(state, args.n_pes)
    print(f"Sampled {len(pes_samples)} PES transitions")

    pes_results = []
    for i, sample in enumerate(pes_samples):
        anum = np.array(sample["atomic_numbers"]).flatten().astype(int)
        charge = sample["charge"]
        label = f"{sample['compound']} TS#{sample['ts_id']}"

        print(f"\n[{i+1}/{len(pes_samples)}] {label} ({len(anum)} atoms, charge={charge})")

        t0 = time.time()

        # DFT single points: TS, fwd minimum, bwd minimum
        e_ts = dft_energy(anum, sample["ts_positions"], charge, functional=args.functional, basis=args.basis)
        e_fwd = dft_energy(anum, sample["fwd_positions"], charge, functional=args.functional, basis=args.basis)
        e_bwd = dft_energy(anum, sample["bwd_positions"], charge, functional=args.functional, basis=args.basis)

        dt = time.time() - t0

        if e_ts is None or e_fwd is None:
            print(f"  SKIPPED (DFT failed) [{dt:.1f}s]")
            continue

        ml_barrier_fwd = sample["ts_energy"] - sample["fwd_energy"]
        dft_barrier_fwd = e_ts - e_fwd

        # Compute integrated barriers from trajectories
        trap_barrier_fwd = None
        hess_barrier_fwd = None

        fwd_traj = sample.get("fwd_traj")
        if fwd_traj is not None:
            positions = getattr(fwd_traj, "positions", None)
            forces = getattr(fwd_traj, "forces", None)
            energies = getattr(fwd_traj, "energies", None)
            hessians = getattr(fwd_traj, "hessians", None)

            if positions and forces and len(positions) >= 2:
                # Trapezoidal: integrate forces along trajectory
                # Trajectory goes from TS → minimum, so barrier = -integrated
                trap_integrated = 0.0
                for j in range(len(positions) - 1):
                    dr = positions[j + 1] - positions[j]
                    f_avg = 0.5 * (forces[j] + forces[j + 1])
                    trap_integrated += -np.sum(f_avg * dr)
                # Trajectory goes TS→min, so energy drops. Barrier = E(TS) - E(min) = -trap_integrated
                trap_barrier_fwd = -trap_integrated

                # Hessian-corrected
                if hessians and any(h is not None for h in hessians):
                    hess_integrated = 0.0
                    for j in range(len(positions) - 1):
                        dr = positions[j + 1] - positions[j]
                        dr_flat = dr.flatten()
                        if hessians[j] is not None:
                            H = hessians[j]
                            F = forces[j].flatten()
                            hess_integrated += -np.dot(F, dr_flat) + 0.5 * np.dot(dr_flat, H @ dr_flat)
                        else:
                            f_avg = 0.5 * (forces[j] + forces[j + 1])
                            hess_integrated += -np.sum(f_avg * dr)
                    hess_barrier_fwd = -hess_integrated

        # Print comparison table
        print(f"  {'Method':<20} {'Barrier(eV)':>12} {'(kcal/mol)':>12} {'vs DFT(eV)':>12} {'vs DFT(kcal)':>12}")
        print(f"  {'-'*68}")
        print(f"  {'DFT':.<20} {dft_barrier_fwd:>12.4f} {dft_barrier_fwd*23.06:>12.2f} {'(ref)':>12} {'(ref)':>12}")
        print(f"  {'ML energy':.<20} {ml_barrier_fwd:>12.4f} {ml_barrier_fwd*23.06:>12.2f} {ml_barrier_fwd-dft_barrier_fwd:>+12.4f} {(ml_barrier_fwd-dft_barrier_fwd)*23.06:>+12.2f}")
        if trap_barrier_fwd is not None:
            print(f"  {'Trap. integration':.<20} {trap_barrier_fwd:>12.4f} {trap_barrier_fwd*23.06:>12.2f} {trap_barrier_fwd-dft_barrier_fwd:>+12.4f} {(trap_barrier_fwd-dft_barrier_fwd)*23.06:>+12.2f}")
        if hess_barrier_fwd is not None:
            print(f"  {'Hessian-corrected':.<20} {hess_barrier_fwd:>12.4f} {hess_barrier_fwd*23.06:>12.2f} {hess_barrier_fwd-dft_barrier_fwd:>+12.4f} {(hess_barrier_fwd-dft_barrier_fwd)*23.06:>+12.2f}")
        print(f"  [{dt:.1f}s]")

        pes_results.append({
            "label": label,
            "ml_fwd": ml_barrier_fwd,
            "dft_fwd": dft_barrier_fwd,
            "trap_fwd": trap_barrier_fwd,
            "hess_fwd": hess_barrier_fwd,
            "n_atoms": len(anum),
        })

    # ── Reaction Barriers ──
    print("\n" + "=" * 80)
    print("REACTION BARRIERS (intermolecular)")
    print("=" * 80)

    rxn_samples = extract_reaction_samples(state, args.n_reactions)
    print(f"Sampled {len(rxn_samples)} reactions")

    rxn_results = []
    for i, sample in enumerate(rxn_samples):
        anum = np.array(sample["ts_anum"]).flatten().astype(int)
        charge = sample["charge"]
        label = f"{sample['name']} ({sample['discovery_method']})"

        print(f"\n[{i+1}/{len(rxn_samples)}] {label} ({len(anum)} atoms, charge={charge})")

        t0 = time.time()

        e_ts = dft_energy(anum, sample["ts_positions"], charge, functional=args.functional, basis=args.basis)
        e_reactant = dft_energy(anum, sample["reactant_positions"], charge, functional=args.functional, basis=args.basis)

        e_product = None
        if sample["product_positions"] is not None:
            e_product = dft_energy(anum, sample["product_positions"], charge, functional=args.functional, basis=args.basis)

        dt = time.time() - t0

        if e_ts is None or e_reactant is None:
            print(f"  SKIPPED (DFT failed) [{dt:.1f}s]")
            continue

        # Forward barrier
        dft_barrier_fwd = e_ts - e_reactant
        ml_barrier_fwd = sample["ml_barrier_fwd"]
        trap_fwd, hess_fwd = integrate_trajectory(sample.get("fwd_traj"))

        print(f"  FORWARD (TS ← reactant):")
        print_barrier_table(dft_barrier_fwd, ml_barrier_fwd, trap_fwd, hess_fwd)

        # Backward barrier
        dft_barrier_bwd = None
        ml_barrier_bwd = sample["ml_barrier_bwd"]
        trap_bwd, hess_bwd = integrate_trajectory(sample.get("bwd_traj"))

        if e_product is not None:
            dft_barrier_bwd = e_ts - e_product
            print(f"  BACKWARD (TS ← product):")
            print_barrier_table(dft_barrier_bwd, ml_barrier_bwd, trap_bwd, hess_bwd)

        print(f"  [{dt:.1f}s]")

        rxn_results.append({
            "label": label,
            "ml_fwd": ml_barrier_fwd, "dft_fwd": dft_barrier_fwd,
            "trap_fwd": trap_fwd, "hess_fwd": hess_fwd,
            "ml_bwd": ml_barrier_bwd, "dft_bwd": dft_barrier_bwd,
            "trap_bwd": trap_bwd, "hess_bwd": hess_bwd,
            "n_atoms": len(anum),
        })

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if pes_results:
        def _print_stats(name, errors_eV):
            errors_kcal = [e * 23.06 for e in np.abs(errors_eV)]
            print(f"\n  {name}:")
            print(f"    MAE:    {np.mean(np.abs(errors_eV)):.4f} eV  ({np.mean(errors_kcal):.2f} kcal/mol)")
            print(f"    Median: {np.median(np.abs(errors_eV)):.4f} eV  ({np.median(errors_kcal):.2f} kcal/mol)")
            print(f"    Max:    {np.max(np.abs(errors_eV)):.4f} eV  ({np.max(errors_kcal):.2f} kcal/mol)")
            print(f"    Bias:   {np.mean(errors_eV):+.4f} eV  ({np.mean(errors_eV) * 23.06:+.2f} kcal/mol)")

        print(f"\nPES barriers ({len(pes_results)} samples):")

        ml_errs = np.array([r["ml_fwd"] - r["dft_fwd"] for r in pes_results])
        _print_stats("ML energy model", ml_errs)

        trap_errs = [r["trap_fwd"] - r["dft_fwd"] for r in pes_results if r.get("trap_fwd") is not None]
        if trap_errs:
            _print_stats(f"Trapezoidal integration ({len(trap_errs)} samples)", np.array(trap_errs))

        hess_errs = [r["hess_fwd"] - r["dft_fwd"] for r in pes_results if r.get("hess_fwd") is not None]
        if hess_errs:
            _print_stats(f"Hessian-corrected ({len(hess_errs)} samples)", np.array(hess_errs))

    if rxn_results:
        for direction, key_ml, key_dft, key_trap, key_hess in [
            ("FORWARD", "ml_fwd", "dft_fwd", "trap_fwd", "hess_fwd"),
            ("BACKWARD", "ml_bwd", "dft_bwd", "trap_bwd", "hess_bwd"),
        ]:
            valid = [r for r in rxn_results if r.get(key_dft) is not None]
            if not valid:
                continue
            print(f"\nReaction barriers {direction} ({len(valid)} samples):")

            ml_errs = np.array([r[key_ml] - r[key_dft] for r in valid if r.get(key_ml) is not None])
            if len(ml_errs):
                _print_stats("ML energy model", ml_errs)

            trap_errs = np.array([r[key_trap] - r[key_dft] for r in valid if r.get(key_trap) is not None])
            if len(trap_errs):
                _print_stats(f"Trapezoidal integration ({len(trap_errs)} samples)", trap_errs)

            hess_errs = np.array([r[key_hess] - r[key_dft] for r in valid if r.get(key_hess) is not None])
            if len(hess_errs):
                _print_stats(f"Hessian-corrected ({len(hess_errs)} samples)", hess_errs)

    # Save raw results
    out_path = data_dir / "dft_barrier_validation.npz"
    np.savez(out_path,
             pes_results=pes_results,
             rxn_results=rxn_results,
             functional=args.functional,
             basis=args.basis)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
