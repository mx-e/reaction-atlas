# DFT transition-state optimisation

Refine an approximate transition state (e.g. an MLFF-predicted TS from the CRN)
to a true first-order saddle point with DFT, and confirm it. This is the
"TS-opt" procedure from the CRN TS-validation study, packaged so it can be
scaled to the whole reaction dataset.

## What it does

For one reaction, given an approximate TS geometry and the total charge:

1. **Saddle optimisation** — ORCA `OptTS`: an eigenvector-following / P-RFO
   search with a Bofill-updated quasi-Newton Hessian. It walks the geometry
   *up* along one mode and *down* along all others until it sits on a saddle.
2. **Hessian** — ORCA `NumFreq`: a numerical Hessian on the optimised
   geometry, used to confirm there is **exactly one imaginary mode**
   (the definition of a first-order saddle / genuine transition state).

Level of theory: **PBE0/def2-TZVPP**, `def2/J RIJCOSX TightSCF`.

The ORCA input it generates per reaction:

```
! PBE0 def2-TZVPP def2/J RIJCOSX TightSCF OptTS NumFreq
%pal nprocs 4 end
%maxcore 2000
%geom Calc_Hess true Recalc_Hess 5 MaxIter 200 end
* xyzfile <charge> <mult> ts_in.xyz
```

`Calc_Hess true` / `Recalc_Hess 5` start the search from an exact Hessian and
refresh it every 5 steps — TS searches are unreliable with a purely updated
Hessian, which tends to drift off the saddle.

## Why trust it (validation summary)

100 transition states predicted by the MLFF were re-optimised with this
procedure and compared to the MLFF guess:

| result | value |
|---|---|
| `OptTS` converged | 98 / 100 |
| converged structures that are first-order saddles (1 imaginary mode) | 98 / 98 |
| optimised TS within 0.5 Å (all-atom RMSD) of the MLFF guess | 92 / 98 (94%) |
| optimised TS with identical covalent connectivity to the MLFF guess | 89 / 98 (91%) |
| mean \|ΔE\| between MLFF-guess and optimised saddle | ~1.6 kcal/mol |
| median wall time | ~33 min on 4 CPU cores |

The MLFF TS prediction itself is sub-second; this DFT refinement is the cost.
The optimised saddle is on average ~1.6 kcal/mol *below* the MLFF guess — the
MLFF slightly overestimates barrier tops, consistently and with little scatter.

## Requirements

- **ORCA** 6.x on `$PATH` (or set `$ORCA_BIN`). ORCA needs its own directory on
  `$PATH` to find its sub-tools.
- Python 3.9+ (standard library only — no extra packages).

## Quick start — one reaction

```bash
export ORCA_BIN=/home/local/orca/orca
export ORCA_NPROCS=4

python run_ts_opt.py --xyz guess_ts.xyz --charge 0 --out runs/rxn_0001
```

`--mult` is optional; without it the multiplicity is guessed from electron
parity (even → singlet, odd → doublet). **Pass `--mult` explicitly for
radicals, triplets and other open-shell systems** — the parity guess is only
the lowest-spin closed-shell/doublet assumption.

## Scaling to the whole dataset

`submit_array.slurm` is a SLURM array template — one array task per reaction.
It expects a dataset laid out as:

```
dataset/rxn_0001/ts.xyz      approximate TS geometry
dataset/rxn_0001/meta.json   JSON with an integer field "ts_charge"
dataset/rxn_0002/...
```

Then:

```bash
DATASET=/path/to/dataset RESULTS=/path/to/results \
    sbatch --array=1-1000 submit_array.slurm
```

Edit the `#SBATCH` partition/time to your cluster. Each task runs ORCA in
node-local scratch and writes results to `results/rxn_NNNN/`. Tasks are
independent — re-submitting a sub-range (`--array=12,45,88`) safely reruns
just those reactions.

## Reading the output

Each `results/rxn_NNNN/` contains the ORCA `ts_opt.inp`/`ts_opt.out`, the
optimised geometry `ts_opt.xyz`, and a parsed **`summary.json`**:

```json
{
  "converged": true,
  "final_energy_hartree": -266.957116,
  "imaginary_frequencies_cm1": [-241.7],
  "n_imaginary_above_100cm1": 1,
  "is_first_order_saddle": true
}
```

`run_ts_opt.py` exits 0 only if the optimisation converged **and** the
structure is a clean first-order saddle.

**On the imaginary-mode count:** ORCA Eckart-projects the translational and
rotational modes, which print as `0.00 cm-1`. A mode is counted as imaginary
only if its magnitude exceeds **100 cm⁻¹** — smaller imaginary frequencies are
finite-difference Hessian noise (a floppy torsion, integration-grid artefacts),
not real negative curvature. A genuine transition state has exactly one
imaginary mode above this cutoff.

## Notes / gotchas

- **Multiplicity** is the one chemistry assumption that is *not* free — see the
  `--mult` note above.
- **Cost scales steeply with system size.** ~33 min median is for small organic
  TSs (≤ ~12 heavy atoms). Larger systems need more `--time` and memory.
- **A non-converged or multi-imaginary-mode result is not a silent failure** —
  `summary.json` records it and the script exits non-zero. Inspect `ts_opt.out`;
  common causes are a poor initial guess or an SCF that needs tighter settings.
- The optimiser can converge to a *different* saddle than intended (a competing
  channel). The connectivity check in the validation table (91% match) quantifies
  how often this happens; for production use, compare the optimised connectivity
  to the expected reactant/product.
