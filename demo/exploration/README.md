# Demo 2: Minimal exploration

This demo runs **the actual ReactionAtlas discovery loop** on a single seed
molecule at a tiny scale. This demo showcases the paper's
core method end-to-end, including all the relevant units:

```
1. generative TS proposal (MoreRed)   
2. MD-ET force-field validation
3. P-RFO saddle search 
4. IRC endpoint check  
5. reaction-graph assembly
```

Unlike the kinetics demo (previous chapter, demo 1),
this one needs the full worker environment and a local
PostgreSQL.
A GPU is recommended but not required and the worker automatically selects a
CPU when no CUDA device is present (the same code path, just slower).

## Prerequisites

1. Worker environment:
   ```bash
   uv sync --extra worker
   ```
   (`pygraphviz` needs system Graphviz first)

2. MD-ET force field: Installed automatically by `--extra worker` (pinned to
   [`md-et`](https://github.com/mx-e/md-et) `v0.1.0`, paper
   [arXiv:2503.01431](https://arxiv.org/abs/2503.01431)).
   The weights are downloaded from Hugging Face automatically when running it
   for the first time (no account or token required).

3. Local database:
   ```bash
   docker compose up -d db
   export DATABASE_URL=postgresql://crn:crn@localhost:5432/crn_cloud
   ```
   No migration step is needed and the worker creates the schema on first launch
   (`Base.metadata.create_all`, see `docs/reproducing.md`).

## Run it

```bash
./demo/exploration/run_demo.sh
```

The script initializes the start molecule (formaldehyde by default, which can be overridden with
`START_XYZ_PATH`), the fragment library, and the four buffer equilibria, then
runs the discovery loop continuously, sampling
compounds and proposing transition states. A single small seed reaches the node
cap slowly, so we recommend to let it run for a few minutes and stop it with Ctrl-C, then
inspect what was explored.
All parameters can be changed, for detailed control, see header of `run_demo.sh`.

For a run that discovers structure faster on a laptop, seed the C2 sugar
glycolaldehyde and give the diffusion proposer more steps:

```bash
START_XYZ_PATH=$PWD/data/start_xyz/glycolaldehyde.xyz \
  MAX_DENOISING_STEPS=500 TS_BATCH_SIZE=4 \
  ./demo/exploration/run_demo.sh
```

## Expected output

The worker logs each stage, i.e. model load, seeding, then the exploration loop
(example lines, where the exact counts and contents vary with config and random seed):

```
Using device: cpu
Seeded compound: CH2O (C=O) / H2O (O) / ...
Seeded equilibrium 'water_autoionization': O ⇌ [HH] + [OH-] ...
Seeding complete, 8 compounds
Entering main exploration loop
Batched TS optimization: 2/3 converged to first-order saddle     # PES / P-RFO
Denoising ...  30it [00:00, 775 it/s]                            # MoreRed proposer
Invalid forward barrier: ... exceeds threshold ...               # barrier validation
Round N: exploring 2 contexts ...
```

Inspect what was explored (in another terminal, or after Ctrl-C):

```bash
uv run --extra db python - <<'PY'
from packages.db.connection import get_session
from packages.db.models import Compound, Reaction, IntraTransitionState
s = get_session()
print("compounds:", s.query(Compound).count())
print("intra-TS :", s.query(IntraTransitionState).count())
print("reactions:", s.query(Reaction).count())
PY

# ...and solve the kinetics of whatever network was explored:
uv run --extra db python -m packages.kinetics.run --experiment main
```

## Run times on a laptop

(Measured on macOS (Apple Silicon, arm64), CPU-only, Python 3.11)

- `uv sync --extra worker` installs the full stack (PyTorch, MoreRed, md-et, …).
- **Model load** (MoreRed diffusion checkpoint + MD-ET `12l`): a few seconds.
- **Seeding** (start molecule + fragments + 4 buffer equilibria): 1 s
- **Exploration loop** is roughly a round per second: batched P-RFO
  saddle search + MoreRed denoising (~500 steps <1 s) + barrier/IRC validation.
- **Discovery:** seeding glycolaldehyde found a new compound and a new
  intramolecular transition state within ca. 30 s (grows slow on laptop, but it illustrates the process)

The machinery all runs on CPU, but meaningful network growth requires a GPU.
Note that the published network used $\sim 10^5$ GPU-hours.
The kinetics demo (`demo/kinetics/`) remains the fully self-contained, pre-verified demo that
needs none of the worker stack.
