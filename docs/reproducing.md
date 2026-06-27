# Reproducing the published runs

A bit-for-bit reproduction of the paper is not practical: the runs
spanned ~10⁵ GPU-hours and ~10⁵ CPU-hours on Cloud Batch. A
**small-scale reproduction** — running each loop on a handful of
seeds against a local Postgres — is straightforward and exercises the
same code paths.

## Prerequisites

- A CUDA-capable GPU (the published runs used L4 / A100)
- Python 3.11+ with [`uv`](https://docs.astral.sh/uv/)
- A local PostgreSQL ≥15 (use `docker compose up -d db`)
- For the CPU stages: `crest`, `xtb`, and `orca` binaries on PATH

The `md-et` force-field checkpoint is **not** committed in this
repository. It is downloaded from HuggingFace at worker boot; set
`HF_TOKEN` in the environment. The diffusion-proposer checkpoint *is*
committed (`packages/worker/models/ts_best_model`).

## 1. Bring up the database

```bash
docker compose up -d db
export DATABASE_URL=postgresql://crn:crn@localhost:5432/crn_cloud

cd packages/db
uv run alembic upgrade head
```

## 2. Seed an experiment

Pick one neutral seed as the starting compound:

```bash
export START_XYZ_PATH=$PWD/data/start_xyz/glycolaldehyde.xyz
export FRAGMENT_PATH=$PWD/data/fragments
export EXPERIMENT_TAG=local-test
```

The first worker launch will canonicalise this compound, drop it into
`pes_work_queue`, and pick up the work.

## 3. Run a GPU worker

```bash
cd packages/worker
uv run python worker.py
```

Parameters that control the published runs (see SI §1 for derivations)
are read from environment variables; the relevant defaults are listed
in `lib/pes_explorer/pes_explorer.py` (`ExploreConfig`) and
`lib/ts_pipeline.py`. Notable knobs:

| Variable | Default | What it controls |
|---|---|---|
| `PES_MD_STEPS` | 500 | MD steps per PES seed |
| `PES_MAX_ITERATIONS` | 3 | PES sweeps before exit |
| `TS_BATCH_SIZE` | 8 | Generative-loop batch size |
| `MAX_VALID_NODES` | 10000 | Soft cap on graph size |

The worker exits cleanly when its queue is drained (or on SIGTERM).

## 4. Run a CPU worker (optional, for DFT validation)

```bash
cd packages/cpu-worker
uv run python worker.py
```

This consumes `crest_work_queue` first, then `dft_work_queue`. The
PBE0 single-points and TS corrections are by far the slowest stage in
the pipeline.

## 5. Run the kinetics solver

```bash
cd packages/kinetics
uv run python loop.py
```

This polls the DB, builds the rate matrix from
`reactions` × `intra_transition_states`, integrates, and writes
`kinetics_snapshot` rows.

## 6. Analyse

```bash
cd scripts
uv run python analyze_barriers.py        # ML vs DFT barrier comparison
uv run python dft_barrier_validation.py  # sampling-based DFT check
```

Both scripts read from `$DATABASE_URL`.

## Matching the published numbers

If you have access to the published-runs dump (Zenodo; see
`docs/data.md`), load it into Postgres and the analysis scripts above
will reproduce the figures in SI §1.7 and SI §3.

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
