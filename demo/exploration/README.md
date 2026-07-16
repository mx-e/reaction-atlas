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

The script initializes the start molecule (the C2 sugar glycolaldehyde by
default, overridable with `START_XYZ_PATH`), the fragment library, and the four
buffer equilibria, then runs the discovery loop continuously, sampling compounds
and proposing transition states. Let it run for a few minutes and stop it with
Ctrl-C, then inspect what was explored. All parameters can be changed; for
detailed control see the header of `run_demo.sh`.

> In the production run, the worker only accepted
> reactions below a physical barrier limit (`ENERGY_THRESHOLD_HARTREE = 0.1 Ha ≈
> 2.72 eV`). Because exploration is slower in the beginning and this demo is not made for
> dedicated hardware, we turned this limit off in order to find reactions faster, i.e.
> for convenience. 

## Expected output

The worker logs each stage, i.e. model load, seeding, then the exploration loop
(example lines, where the exact counts and contents vary with config and random seed):

```
Using device: cpu
Seeded compound: OCC=O (glycolaldehyde) / H2O (O) / CH2O (C=O) / ...
Seeded equilibrium 'water_autoionization': O ⇌ [HH] + [OH-] ...
Seeding complete, 9 compounds
Barrier acceptance cap: 1000.000 eV
Entering main exploration loop
Claimed PES work (explore): compound_id=1 minimum_id=1              # PES / MD escape
Denoising ...  500it [00:02, 190 it/s]                              # MoreRed proposer
Valid escaped reaction from O=CCO: barrier_fwd=2.918 eV             # a discovery!
Round N: exploring 4 contexts ...
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

## Visualize what was explored

Once a discovery or two has landed, render the network and its growth over time:

```bash
uv run --extra worker python demo/exploration/plot_discoveries.py
```

This writes `network.png` and `growth.png` next to the script. The reference
images below are from a 1h GPU run.
A few minutes on a laptop may likely yield less discoveries.

![Reaction network grown from a single seed](expected_network.png)

A yellow frame marks aan initial species and a blue frame a newly discovered one, and 
edges denote reactions. The seeds are the start sugar glycolaldehyde, formaldehyde, and the
water/carbonate buffer (H2O, H3O+, OH-, H2, CO2, carbonic acid, bicarbonate)
wired together by the four buffer equilibria. Around them the exploration has
grown seven new compounds (blue).
Among them is methane and acetic acid off
CO₂, a hydroxycarbene and a dioxetane off the C1/C2 carbonyls. 
Disconnected fragments are spread on their own grid below so
every molecule stays around even before it may join the main network.

![Cumulative discoveries over wall time](expected_growth.png)

Cumulative compounds / reactions / intramolecular TSs
against wall time.

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
The run above was conducted on 1 GPU node.
