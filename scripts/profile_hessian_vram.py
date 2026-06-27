"""
Profile GPU VRAM usage for batched Hessian computation at various batch sizes.

Measures actual peak memory for the md-et calculator's get_batched_hessians()
using jacrev + vmap, so we can determine safe batch sizes for different GPUs.

Usage:
    python scripts/profile_hessian_vram.py [--max-batch 64] [--device cuda]

Produces a table of batch_size → peak VRAM, with a recommendation.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

# Add worker package to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "worker"))


def make_random_molecule(n_atoms: int, seed: int = 0) -> Atoms:
    """Create a random molecule with realistic CHO composition."""
    rng = np.random.RandomState(seed)

    # Realistic CHO compositions by size
    if n_atoms <= 4:
        numbers = [6, 8, 1, 1][:n_atoms]  # H2CO-like
    elif n_atoms <= 8:
        # Glycolaldehyde-like: C2H4O2
        numbers = [6, 6, 8, 8, 1, 1, 1, 1][:n_atoms]
    elif n_atoms <= 12:
        # Glyceraldehyde-like: C3H6O3
        numbers = [6, 6, 6, 8, 8, 8, 1, 1, 1, 1, 1, 1][:n_atoms]
    else:
        # Tetrose-max: C4H8O4
        numbers = [6, 6, 6, 6, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1][:n_atoms]

    positions = rng.randn(n_atoms, 3) * 1.5
    atoms = Atoms(numbers=numbers, positions=positions)
    atoms.info["charge"] = 0
    return atoms


def get_gpu_memory_mb() -> dict:
    """Get current GPU memory stats in MB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**2,
        "reserved": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**2,
    }


def reset_peak_memory():
    """Reset peak memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


def profile_batch(calc, atoms_list: list[Atoms], label: str) -> dict:
    """Profile a single batched Hessian call."""
    reset_peak_memory()
    mem_before = get_gpu_memory_mb()

    t0 = time.perf_counter()
    results = calc.get_batched_hessians(atoms_list)
    dt = time.perf_counter() - t0

    mem_after = get_gpu_memory_mb()
    peak = mem_after["max_allocated"]

    # Verify results
    n_atoms = len(atoms_list[0])
    for forces, energy, H in results:
        assert H.shape == (3 * n_atoms, 3 * n_atoms), f"Bad Hessian shape: {H.shape}"

    return {
        "label": label,
        "batch_size": len(atoms_list),
        "n_atoms": n_atoms,
        "time_s": dt,
        "time_per_mol_ms": dt / len(atoms_list) * 1000,
        "peak_vram_mb": peak,
        "delta_vram_mb": peak - mem_before["allocated"],
        "vram_per_mol_mb": (peak - mem_before["allocated"]) / len(atoms_list),
    }


def main():
    parser = argparse.ArgumentParser(description="Profile Hessian VRAM usage")
    parser.add_argument("--max-batch", type=int, default=64, help="Max batch size to test")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--variant", default="12l", help="md-et model variant")
    parser.add_argument("--mol-sizes", default="4,8,12,16", help="Molecule sizes to test")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu (memory profiling disabled)")
        device = "cpu"

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_total_mb = torch.cuda.get_device_properties(0).total_mem / 1024**2
        print(f"GPU: {gpu_name} ({gpu_total_mb:.0f} MB)")
    else:
        gpu_name = "CPU"
        gpu_total_mb = 0
        print("Running on CPU (timing only, no VRAM profiling)")

    # Load calculator
    print(f"Loading md-et calculator (variant={args.variant}, device={device})...")
    from lib.md_et_calculator import get_md_et_calculator

    calc = get_md_et_calculator(device=device, variant=args.variant)

    baseline_mem = get_gpu_memory_mb()
    print(f"Model loaded. Baseline VRAM: {baseline_mem['allocated']:.0f} MB\n")

    mol_sizes = [int(s) for s in args.mol_sizes.split(",")]
    batch_sizes = [1, 2, 4, 8, 16, 32]
    batch_sizes = [b for b in batch_sizes if b <= args.max_batch]
    if args.max_batch not in batch_sizes:
        batch_sizes.append(args.max_batch)

    all_results = []

    for n_atoms in mol_sizes:
        print(f"{'='*70}")
        print(f"Molecule size: {n_atoms} atoms (3N = {3*n_atoms})")
        print(f"{'='*70}")
        print(
            f"{'Batch':>6} {'Peak VRAM':>10} {'Delta':>10} {'Per Mol':>10} "
            f"{'Time':>8} {'Per Mol':>10} {'Status':>8}"
        )
        print("-" * 70)

        for batch_size in batch_sizes:
            atoms_list = [make_random_molecule(n_atoms, seed=i) for i in range(batch_size)]
            for a in atoms_list:
                a.calc = calc

            label = f"{n_atoms}at_b{batch_size}"
            try:
                result = profile_batch(calc, atoms_list, label)
                all_results.append(result)

                status = "OK"
                if device == "cuda" and result["peak_vram_mb"] > gpu_total_mb * 0.85:
                    status = "WARN"

                print(
                    f"{batch_size:>6} {result['peak_vram_mb']:>9.0f}M "
                    f"{result['delta_vram_mb']:>9.0f}M "
                    f"{result['vram_per_mol_mb']:>9.1f}M "
                    f"{result['time_s']:>7.2f}s "
                    f"{result['time_per_mol_ms']:>8.1f}ms "
                    f"{status:>8}"
                )

            except torch.cuda.OutOfMemoryError:
                print(f"{batch_size:>6} {'---':>10} {'---':>10} {'---':>10} {'---':>8} {'---':>10} {'OOM':>8}")
                reset_peak_memory()
                break

            except Exception as e:
                print(f"{batch_size:>6} ERROR: {e}")
                reset_peak_memory()
                break

        print()

    # Summary and recommendation
    if all_results and device == "cuda":
        print(f"\n{'='*70}")
        print("SUMMARY: Safe batch sizes by GPU")
        print(f"{'='*70}\n")

        # Fit linear model: peak_vram = base + per_mol * batch_size
        # Group by n_atoms
        for n_atoms in mol_sizes:
            results_for_size = [r for r in all_results if r["n_atoms"] == n_atoms]
            if len(results_for_size) < 2:
                continue

            # Simple linear regression
            xs = np.array([r["batch_size"] for r in results_for_size])
            ys = np.array([r["peak_vram_mb"] for r in results_for_size])
            A = np.vstack([xs, np.ones(len(xs))]).T
            slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]

            print(f"  {n_atoms} atoms: VRAM ≈ {intercept:.0f} MB + {slope:.1f} MB × batch_size")

            for gpu_name, gpu_vram_mb in [
                ("T4 (16 GB)", 16000),
                ("L4 (24 GB)", 24000),
                ("A100 40GB", 40000),
            ]:
                # Leave 15% headroom for CUDA allocator fragmentation
                usable = gpu_vram_mb * 0.85
                if slope > 0:
                    max_batch = int((usable - intercept) / slope)
                else:
                    max_batch = 999
                max_batch = max(0, max_batch)
                print(f"    {gpu_name}: max batch ≈ {max_batch}")

            print()


if __name__ == "__main__":
    main()
