# Demo 1 — Kinetics (self-contained, CPU-only)

This demo runs the **actual ReactionAtlas kinetics pipeline** on a small, real
early-exploration reaction network and produces a steady-state distribution and
a concentration-vs-time plot. It needs no GPU, no PostgreSQL, no external
quantum-chemistry binaries, and no downloads.

It is the guaranteed "runs on a normal desktop" demo: a colleague unfamiliar
with the software should be able to reproduce the output below in under a
minute after installation.

## What it does

`run_demo.py` loads `data/early_network.npz` — a 64-compound, 80-reaction slice
of a **published run** (extracted from checkpoint `checkpoint_60.sql`, an early
state of the exploration; see [Provenance](#provenance)) — into an in-memory
SQLite database, then calls the production code path unchanged:

```
packages.kinetics.build.build_snapshot
  → packages.kinetics.model.build_model_from_db   # barrier policy + Eyring rates
  → packages.kinetics.scipy_solver.solve_ode       # numba RHS/Jacobian + scipy BDF
```

Nothing about the model builder or the ODE integrator is re-implemented in the
demo — the SQLite database presents the exact schema the solver expects in
production (`packages/db/models.py`).

## System requirements

- Python 3.11–3.12 (tested on 3.11).
- The base dependencies only: `numpy`, `scipy`, `numba`, `sqlalchemy`,
  `loguru`, `matplotlib` (installed by `uv sync`; see the top-level README).
- ~1 GB RAM. No GPU.

## Run it

From the repository root:

```bash
uv sync                              # one-time: create the CPU-only environment
uv run python demo/kinetics/run_demo.py
```

(Without `uv`: `pip install -e .` into a fresh Python 3.11 venv, then
`python demo/kinetics/run_demo.py`.)

## Expected output

The full text summary is in [`expected_output.txt`](expected_output.txt). The
key lines:

```
Solved reaction network (via packages.kinetics.build.build_snapshot):
  species in ODE system     : 64
  reactions in ODE system   : 49
    of which manual equilibria: 4
    of which using DFT (PBE0) : 37
```

followed by the top of the steady-state sampling distribution (dominated by
CO₂, methoxy/formate esters, and small sugars) and the plot
`concentrations.png` — compare it against the committed reference
[`expected_concentrations.png`](expected_concentrations.png).

## Expected run time

On a normal laptop:

- **First run: ~10–25 s** — one-time numba JIT compilation of the ODE
  right-hand-side / Jacobian kernels, plus the matplotlib font-cache build.
- **Subsequent runs: ~1–2 s** — numba caches the compiled kernels to disk.

## Determinism

The reaction/species counts and the dominant steady-state species are stable
across runs and machines. The last decimal place of the steady-state weights
may vary slightly with the BLAS/LAPACK build behind scipy's BDF integrator;
this does not change the qualitative result.

## Provenance

`data/early_network.npz` was produced by
[`data/extract_early_network.py`](data/extract_early_network.py), which parses
the `COPY` blocks of a plain `pg_dump` checkpoint and keeps only the scalar
columns the kinetics builder reads (compound SMILES, reactant/product links,
ML/DFT barriers, and manual-equilibrium rate constants). All large geometry /
Hessian / trajectory blobs are discarded, which is why an ~100 MB database
checkpoint reduces to an 8 KB network. The full published database is released
separately on Zenodo (see `docs/data.md`).
