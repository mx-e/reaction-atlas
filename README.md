# ReactionAtlas

Companion code for **"ReactionAtlas: ab origine exploration of chemical
reaction networks with machine learning"** (Gugler, Eissler, Kahouli, Müller;
[arXiv:2606.30778](https://arxiv.org/abs/2606.30778)).

ReactionAtlas grows a chemical reaction network from a handful of seed molecules
with no hand-written rules: a machine-learned generative model proposes
transition-state candidates, a machine-learned force field (MD-ET) validates
them, and the discovered products feed back as new seeds. A kinetics solver
integrates the resulting mass-action network. Starting from eight pre-biotic
molecules, the published run discovered ~47,000 reactions among ~12,000
compounds.

This repository is preserved **as-it-ran in production**: no scientific module
has been refactored or renamed for release. The cloud-orchestration layer
(Terraform, Cloud Batch, GKE) and the web API/frontend are hosted separately
and are not included here — a new operator will want their own orchestration.
The full reaction-network database and the seed inputs are released on Zenodo
(see [`docs/data.md`](docs/data.md)); this repository ships the code plus a
small demo network.

## Links & related resources

- **Interactive explorer (hosted):**
  [reactionatlas.bifold.berlin](https://reactionatlas.bifold.berlin) — browse the
  full reaction network and its kinetics online (the frontend/API from
  `crn-cloud`, hosted separately from this code repository).
- **Full dataset (Zenodo):**
  [10.5281/zenodo.21358136](https://doi.org/10.5281/zenodo.21358136) — the
  complete reaction-network database (~26 GB, `pg_restore`) and the seed inputs.
  See [`docs/data.md`](docs/data.md).
- **MD-ET force field:** [github.com/mx-e/md-et](https://github.com/mx-e/md-et)
  · paper [arXiv:2503.01431](https://arxiv.org/abs/2503.01431).
- **Paper (this work):** [arXiv:2606.30778](https://arxiv.org/abs/2606.30778).

---

## 1. System requirements

### Software dependencies

| Component | Version | Notes |
|---|---|---|
| Python | 3.11–3.12 | tested on **3.11** |
| [`uv`](https://docs.astral.sh/uv/) | ≥ 0.5 | environment/dependency manager |
| **Kinetics demo** (base) | numpy ≥ 1.24, scipy ≥ 1.10, numba ≥ 0.59, SQLAlchemy ≥ 2.0, matplotlib ≥ 3.7, loguru ≥ 0.7 | pinned in `uv.lock`; installed by `uv sync` |
| **Exploration worker** (`--extra worker`) | PyTorch 2.4.x, SchNetPack 2.1.1, ASE ≥ 3.23, RDKit ≥ 2024.3.5, [`md-et`](https://arxiv.org/abs/2503.01431), huggingface-hub ≥ 0.20, psycopg 3, Alembic ≥ 1.13, pygraphviz ≥ 1.14 | see [Installation](#2-installation-guide) |
| **Database** (`--extra db`) | PostgreSQL ≥ 15, Alembic ≥ 1.13, psycopg 3 | `docker compose up -d db` |
| DFT validation (optional) | `orca`, `crest`, `xtb` on `PATH` | external binaries, not bundled |

Exact, reproducible versions of the demo environment are locked in
[`uv.lock`](uv.lock).

### Operating systems

Tested on **macOS 14/15 (Apple Silicon, arm64)** and **Linux (Ubuntu 22.04,
x86_64)**. The kinetics demo below was verified on macOS 15 / Python 3.11; the
full exploration pipeline ran on Linux (x86_64 + NVIDIA GPUs) in production.

### Hardware

- **No GPU is required.** Both the diffusion proposer and the force field
  auto-select CPU when no CUDA device is present (the code picks
  `"cuda" if torch.cuda.is_available() else "cpu"`); a GPU only makes
  exploration faster. The published runs used NVIDIA L4 / A100.
- The kinetics demo runs in ~1 GB RAM on any modern laptop.

---

## 2. Installation guide

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then
from the repository root:

```bash
uv sync                     # kinetics-demo environment (CPU-only)
uv sync --extra worker      # + the full exploration worker (PyTorch, md-et, RDKit, ...)
uv sync --extra db          # + database tooling (Alembic migrations)
```

`uv sync` provisions a virtual environment in `.venv/` from the locked
dependencies; the repository is imported from its root (code is run with
`uv run …`, see below) rather than installed as a wheel.

**Typical install time on a normal desktop:**

- Kinetics-demo environment (`uv sync`): **under 1 minute**.
- Full worker environment (`uv sync --extra worker`): **~5–10 minutes**
  (downloads PyTorch and the scientific stack). `pygraphviz` needs the system
  Graphviz library first: `brew install graphviz` (macOS) or
  `apt-get install graphviz graphviz-dev` (Debian/Ubuntu).

---

## 3. Demo

### Demo 1 — Kinetics (self-contained, CPU-only) — **start here**

Runs the actual kinetics pipeline on a small **real** early-exploration network
(64 compounds, 80 reactions, extracted from a published checkpoint; ~8 KB,
shipped in `demo/kinetics/data/`). No GPU, no database, no downloads.

```bash
uv sync
uv run python demo/kinetics/run_demo.py
```

**Expected output** (full text in
[`demo/kinetics/expected_output.txt`](demo/kinetics/expected_output.txt)):
the network is built and solved via `packages.kinetics.build.build_snapshot` —
49 reactions over 64 species (4 buffer equilibria, 37 using DFT barriers) — and
the dominant steady-state species and a concentration-vs-time plot
(`demo/kinetics/concentrations.png`, compare with the committed
[`expected_concentrations.png`](demo/kinetics/expected_concentrations.png)) are
produced.

**Expected run time:** ~10–25 s on the first run (one-time numba JIT
compilation + matplotlib font cache), ~1–2 s afterwards.

See [`demo/kinetics/README.md`](demo/kinetics/README.md) for details.

### Demo 2 — Minimal exploration (representative, CPU)

Runs one iteration of the real discovery loop on a single seed molecule on CPU
— generative TS proposal → MD-ET validation → saddle search → IRC → reaction
graph — writing the discovered minima / transition states / reactions to a
local PostgreSQL. This is the demo that exercises the paper's core method.

```bash
uv sync --extra worker           # includes the public md-et package
docker compose up -d db
uv run python demo/exploration/run_demo.py
```

Expected output, expected run time, and details are in
[`demo/exploration/README.md`](demo/exploration/README.md).

---

## 4. Instructions for use (your own data)

- **Grow a new network from your own seed.** Point the worker at any XYZ start
  geometry and run an exploration against a local PostgreSQL. See
  [`docs/reproducing.md`](docs/reproducing.md) (steps 1–4) and
  `demo/exploration/` for the minimal single-seed case.
- **Solve kinetics on a network you have.** Against any populated database (a
  live run, or one restored from the Zenodo dump):
  ```bash
  export DATABASE_URL=postgresql://crn:crn@localhost:5432/crn_cloud
  uv run --extra db python -m packages.kinetics.run --experiment main
  ```
- **Query the database directly.** The schema is the canonical record of every
  pipeline stage; see [`docs/schema.md`](docs/schema.md).

---

## 5. Reproduction of the published results

A bit-for-bit reproduction is not practical (the runs spanned ~10⁵ GPU-hours
and ~10⁵ CPU-hours). [`docs/reproducing.md`](docs/reproducing.md) gives a
faithful **small-scale** reproduction that exercises the same code paths, and
explains how to restore the full published database from Zenodo
([`docs/data.md`](docs/data.md)) to reproduce the figures.

---

## Repository layout

| Path | Purpose | SI reference |
|---|---|---|
| `packages/worker/` | GPU/CPU worker: PES loop, generative TS loop, IRC, reaction-graph assembly. The diffusion-proposer checkpoint (`models/ts_best_model`) is committed. | SI §1, §2, §6 |
| `packages/cpu-worker/` | CREST conformer search, ORCA TS correction, PBE0 DFT barrier validation. | SI §1.7, §2.6 |
| `packages/kinetics/` | Reaction-network ODE integrator (PETSc / SciPy backends) + `run.py` CLI. | SI §3 |
| `packages/db/` | SQLAlchemy schema + Alembic migrations. | SI §6 |
| `packages/MoreRed_src/` | Vendored [MoreRed](https://github.com/khaledkah/MoreRed) diffusion proposer (MIT). | SI §2.1 |
| `demo/` | Self-contained kinetics demo + minimal exploration demo. | — |
| `data/` | Runtime reference structures (start geometries, fragment library). | SI §1.1, §4 |

### Where to start reading

1. [`docs/architecture.md`](docs/architecture.md) — how the worker, DB, and
   kinetics solver interact.
2. [`docs/schema.md`](docs/schema.md) — the database schema, table by table.
3. [`docs/reproducing.md`](docs/reproducing.md) — commands to rerun each loop.
4. [`docs/data.md`](docs/data.md) — the Zenodo dataset and bundled files.

The PES loop entry point is
`packages/worker/lib/pes_explorer/pes_explorer.py`; the generative-loop pipeline
is `packages/worker/lib/ts_pipeline.py`; the saddle-search optimiser is
`packages/worker/lib/pes_explorer/prfo.py`.

### Where the algorithm is described

A complete, detailed description of the method (with pseudocode) is in the
paper's **Methods** section and **Supplementary Information** (SI §1–§3); the
SI-section cross-references above map each component to the code.

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Third-party components (the
vendored MoreRed under MIT, and the `md-et` force field) are listed in
[`NOTICE`](NOTICE); MD-ET use requires citing
[arXiv:2503.01431](https://arxiv.org/abs/2503.01431).

> ReactionAtlas is the subject of a pending patent application. The code is
> released under Apache-2.0, whose patent grant (§3) covers use of this
> released implementation.

## Citation

If you use this software, please cite the paper (see
[`CITATION.cff`](CITATION.cff)):

```bibtex
@article{reactionatlas2026,
  title   = {ReactionAtlas: ab origine exploration of chemical reaction
             networks with machine learning},
  author  = {Gugler, Stefan and Eissler, Max and Kahouli, Khaled and
             M\"uller, Klaus-Robert},
  journal = {arXiv preprint arXiv:2606.30778},
  year    = {2026}
}
```
