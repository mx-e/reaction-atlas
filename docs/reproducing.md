# Reproducing the published runs

A bit-for-bit reproduction of the paper is not practical: the runs
spanned ~10⁵ GPU-hours and ~10⁵ CPU-hours on Cloud Batch. A
**small-scale reproduction** — running each loop on a handful of
seeds against a local Postgres — is straightforward and exercises the
same code paths.

> **Just want to see the solver run?** The self-contained kinetics demo
> (`demo/kinetics/`) needs none of the prerequisites below — no GPU, no
> PostgreSQL, no external binaries. See the top-level README, "Demo".

## Prerequisites

- Python 3.11–3.12 with [`uv`](https://docs.astral.sh/uv/)
- A CUDA-capable GPU is **optional** — the exploration worker auto-selects CPU
  when no CUDA device is present (just slower). The published runs used L4 / A100.
- A local PostgreSQL ≥15 (use `docker compose up -d db`)
- For the DFT/CREST validation stages: `crest`, `xtb`, and `orca` binaries on PATH
- The `pygraphviz` dependency needs the system Graphviz library
  (`brew install graphviz` / `apt-get install graphviz graphviz-dev`).

The diffusion-proposer checkpoint *is* committed
(`packages/worker/models/ts_best_model`). The `md-et` force-field checkpoint is
downloaded from Hugging Face on first use by the `md-et` package (see the
README "System requirements" for the public release / pin).

## 0. Install

```bash
uv sync --extra worker      # exploration environment (torch, md-et, rdkit, ...)
```

`uv sync` (no extras) installs only the CPU-only kinetics-demo environment.

## 1. Bring up the database

```bash
docker compose up -d db
export DATABASE_URL=postgresql://crn:crn@localhost:5432/crn_cloud

# Create the schema. The workers create it automatically on first launch
# (Base.metadata.create_all from packages/db/models.py — the canonical schema),
# so this is only needed for a clean, empty database ahead of time:
uv run --extra db python -c "from packages.db.connection import init_db; init_db()"
```

> The Alembic migrations under `packages/db/alembic/versions/` record the
> production schema deltas; the canonical, complete schema is
> `packages/db/models.py`, applied via `create_all` above.

## 2. Seed an experiment

Pick a starting geometry (anything in `data/start_xyz/`, or your own
XYZ file):

```bash
export START_XYZ_PATH=$PWD/data/start_xyz/glycolaldehyde.xyz
export FRAGMENT_PATH=$PWD/data/fragments
export EXPERIMENT=main   # must be a registered experiment (see packages/db/experiments.py)
```

The first worker launch will canonicalise this compound, drop it into
`pes_work_queue`, and pick up the work. To seed from the published
neutral-seed set, download the seed bundle from
reactionatlas.bifold.berlin/downloads (forthcoming; see `docs/data.md`)
and point `START_XYZ_PATH` at one of its files.

## 3. Run a GPU worker

```bash
# demo/exploration/run_demo.sh is the easiest correct launcher (it sets the
# paths, EXPERIMENT, and PYTHONPATH). To run worker.py directly, put the repo
# root AND packages/worker on PYTHONPATH (worker.py mixes `packages.*` and
# `lib.*` imports):
cd packages/worker
PYTHONPATH="$(cd ../.. && pwd):$PWD" uv run --extra worker python worker.py
```

(Runs on CPU automatically if no CUDA GPU is present — slower, but the same
code path. For a minimal single-seed exploration on a laptop, see
`demo/exploration/`.)

Parameters that control the published runs (see SI §1 for derivations)
are read from environment variables; the relevant defaults are listed
in `lib/pes_explorer/pes_explorer.py` (`ExploreConfig`) and
`lib/ts_pipeline.py`. Notable knobs:

| Variable | Default | What it controls |
|---|---|---|
| `PES_MD_STEPS` | 500 | MD steps per PES seed |
| `PES_MAX_ITERATIONS` | 10 | PES sweeps before exit |
| `TS_BATCH_SIZE` | 32 | Generative-loop batch size |
| `MAX_VALID_NODES` | 1000 | Soft cap on graph size |

The worker exits cleanly when its queue is drained (or on SIGTERM).

## 4. Run a CPU worker (optional, for DFT validation)

```bash
cd packages/cpu-worker
uv run --extra worker python worker.py
```

This consumes `crest_work_queue` first, then `dft_work_queue`. The
PBE0 single-points and TS corrections are by far the slowest stage in
the pipeline.

## 5. Solve the kinetics network

`packages/kinetics/loop.py` is the background solver that runs *inside the API
service* (an asyncio task holding a Postgres advisory lock per experiment).
This repository ships the solver but not the API host, so run a one-shot solve
with the bundled CLI instead:

```bash
uv run --extra db python -m packages.kinetics.run --experiment main
```

It builds the mass-action ODE model from the `reactions` graph, integrates it
(PETSc BDF if `petsc4py` is installed, otherwise scipy BDF), and prints the
steady-state distribution — reading the same tables `loop.py` does. To persist
`kinetics_snapshots` rows instead, run the API service (not in this repo).

## Matching the published numbers

If you have access to the published-runs dump
(reactionatlas.bifold.berlin/downloads, forthcoming; see
`docs/data.md`), load it into Postgres and you can query the
`intra_transition_states`, `reactions`, and `kinetics_snapshots` tables
directly to reproduce the figures in SI §1.7 and SI §3. The schema is
documented in `docs/schema.md`.

For locally-grown reaction graphs the absolute numbers will differ
from the paper (different seeds, different RNG, different fleet
ordering), but the *distributions* (barrier height, IRC pass rate,
endpoint pass rate) should match within statistical noise.

## Known limitations of local execution

- The published runs benefited from many workers racing on the queue;
  `pes_explorer` includes back-off and lock-contention paths that are
  exercised only by a multi-worker setup. A single local worker
  follows the same code path but never observes contention.
- Spot preemption handling (`SpotPreemptionError`, SIGTERM) is wired
  through the worker but inert outside of a preemptible cloud VM.
- The GCS upload paths in `packages/worker/lib/db.py` (full-Hessian
  blob offload) are conditional on `GCS_BUCKET`; leave that unset
  locally and the data is stored inline in Postgres.
