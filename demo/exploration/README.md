# Demo 2 — Minimal exploration (representative)

This demo runs **the actual ReactionAtlas discovery loop** on a single seed
molecule at deliberately tiny scale. It is the demo that exercises the paper's
core method end-to-end:

```
generative TS proposal (MoreRed)  →  MD-ET force-field validation
      →  P-RFO saddle search  →  IRC endpoint check  →  reaction-graph assembly
```

Unlike the kinetics demo, this one needs the full worker environment and a local
PostgreSQL. A GPU is **recommended but not required** — the worker auto-selects
CPU when no CUDA device is present (the same code path, just slower).

## Prerequisites

1. **Worker environment:**
   ```bash
   uv sync --extra worker
   ```
   (`pygraphviz` needs system Graphviz first — see the top-level README.)

2. **MD-ET force field.** Installed automatically by `--extra worker` (pinned to
   [`md-et`](https://github.com/mx-e/md-et) `v0.1.0`, paper
   [arXiv:2503.01431](https://arxiv.org/abs/2503.01431)). Its weights are
   downloaded openly from Hugging Face on first use — **no account or token
   required**. (The generative-proposer checkpoint is already committed in this
   repo at `packages/worker/models/ts_best_model`.)

3. **Local database:**
   ```bash
   docker compose up -d db
   export DATABASE_URL=postgresql://crn:crn@localhost:5432/crn_cloud
   ```
   No migration step is needed — the worker creates the schema on first launch
   (`Base.metadata.create_all`; see `docs/reproducing.md`).

## Run it

```bash
./demo/exploration/run_demo.sh
```

The script seeds the start molecule (formaldehyde by default; override
`START_XYZ_PATH`), the fragment library, and the four buffer equilibria, then
runs the discovery loop continuously — like the production worker — sampling
compounds and proposing transition states. A single small seed reaches the node
cap slowly, so **let it run for a few minutes and stop it with Ctrl-C**, then
inspect what it grew. All knobs are overridable — see the header of
`run_demo.sh`.

For a run that discovers structure faster on a laptop, seed the C₂ sugar
glycolaldehyde and give the diffusion proposer more steps:

```bash
START_XYZ_PATH=$PWD/data/start_xyz/glycolaldehyde.xyz \
  MAX_DENOISING_STEPS=500 TS_BATCH_SIZE=4 \
  ./demo/exploration/run_demo.sh
```

## Expected output

The worker logs each stage — model load, seeding, then the exploration loop
(illustrative lines; the exact counts vary with config and RNG):

```
Using device: cpu
Seeded compound: CH2O (C=O) / H2O (O) / ...
Seeded equilibrium 'water_autoionization': O ⇌ [HH] + [OH-] ...
Seeding complete, 8 compounds
Entering main exploration loop
Batched TS optimization: 2/3 converged to first-order saddle      # PES / P-RFO
Denoising ...  30it [00:00, 775 it/s]                             # MoreRed proposer
Invalid forward barrier: ... exceeds threshold ...               # barrier validation
Round N: exploring 2 contexts ...
```

Inspect what it grew (in another terminal, or after Ctrl-C):

```bash
uv run --extra db python - <<'PY'
from packages.db.connection import get_session
from packages.db.models import Compound, Reaction, IntraTransitionState
s = get_session()
print("compounds:", s.query(Compound).count())
print("intra-TS :", s.query(IntraTransitionState).count())
print("reactions:", s.query(Reaction).count())
PY

# ...and solve the kinetics of whatever network you just grew:
uv run --extra db python -m packages.kinetics.run --experiment main
```

## Verified behaviour and run time (this laptop)

Measured on macOS (Apple Silicon, arm64), **CPU-only**, Python 3.11:

- `uv sync --extra worker` installs the full stack (PyTorch, MoreRed, md-et, …).
- **Model load** (MoreRed diffusion checkpoint + MD-ET `12l`): a few seconds.
- **Seeding** (start molecule + fragments + 4 buffer equilibria): ~1 s → 8 compounds.
- **Exploration loop** runs at roughly a round per second: batched P-RFO
  saddle search + MoreRed denoising (~500 steps ≈ <1 s) + barrier/IRC validation.
- **Discovery:** seeding glycolaldehyde found a new compound and a new
  intramolecular transition state within ~30 s; the network then grows slowly.

The machinery all runs on CPU, but **meaningful network growth is
compute-bound** — the published runs used ~10⁵ GPU-hours. Use a GPU (auto-
detected) for real exploration; the CPU run above is a faithful, low-cost
demonstration that the full pipeline installs and runs. The kinetics demo
(`demo/kinetics/`) remains the fully self-contained, pre-verified demo that
needs none of the worker stack.
