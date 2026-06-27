# reaction-atlas

Companion code for *<paper title — TODO>*.

This repository contains the code that was used to generate, validate, and
analyse the reaction networks reported in the paper. It is preserved
**as-it-ran in production**: no scientific module has been refactored,
renamed, or simplified for release. The cloud-orchestration layer
(Terraform, Cloud Batch, GKE manifests) was specific to Google Cloud and
has not been included — any new operator will want to write their own
orchestration anyway. The published runs used many GPU/CPU workers
coordinated through PostgreSQL work queues; running a single worker
locally against a local Postgres is sufficient to exercise the full
pipeline on a small seed set.

The full reaction-network database that backs the paper's figures is
published separately on Zenodo (DOI: **TODO**). This repository ships
only the code, the seed inputs, the reference structures, and the
supplementary-information sources.

## Repository layout

| Path | Purpose | SI reference |
|---|---|---|
| `packages/worker/` | GPU worker: PES loop, generative TS loop, IRC, reaction-graph assembly. The diffusion-proposer checkpoint (`models/ts_best_model`) is committed. | SI §1, §2, §6 |
| `packages/cpu-worker/` | CPU worker: CREST conformer search, ORCA TS correction, PBE0 DFT barrier validation. | SI §1.7, §2.6 |
| `packages/kinetics/` | Reaction-network ODE integrator (PETSc / SciPy backends) and snapshot writer. | SI §3 |
| `packages/db/` | SQLAlchemy schema + Alembic migrations. The schema is the canonical record of what each pipeline stage produced. | SI §6 |
| `packages/MoreRed_src/` | Source of the diffusion proposer used by the generative loop. | SI §2.1 |
| `scripts/` | Analysis scripts (`analyze_barriers.py`, `dft_barrier_validation.py`) and benchmarks. | SI §1.7 |
| `report/` | LaTeX sources of the supplementary information. `merge_si.sh` builds `si_merged.tex`. | the SI itself |
| `tests/` | Unit tests for the worker libraries. | — |
| `data/` | Reference structures used at runtime (start xyz, fragment library). | SI §1.1, §4 |
| `neutral_seeds/` | The 152 neutral seed molecules described in SI §4. | SI §4 |
| `conformer_pairs_dataset/`, `conformer_pairs_relaxed/` | Conformer-pair validation set. | SI §1.7 |

## Where to start reading

If you came here from the paper, the recommended reading order is:

1. **`report/si_merged.tex`** — the supplementary information; the
   single most useful entry point.
2. **`docs/architecture.md`** — block diagram of how the worker, DB,
   and kinetics solver interact.
3. **`docs/schema.md`** — the database schema, table by table.
4. **`docs/reproducing.md`** — exact commands to rerun each loop on a
   small seed set.
5. **`docs/data.md`** — pointer to the Zenodo dataset and notes on the
   files bundled in this repo.

The PES loop entry point is `packages/worker/lib/pes_explorer/pes_explorer.py`;
the generative-loop pipeline lives in `packages/worker/lib/ts_pipeline.py`;
the saddle-search optimiser is `packages/worker/lib/pes_explorer/prfo.py`.

## Installing

The Python environment is managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync                     # install pinned dependencies
uv pip install -e packages/worker -e packages/db -e packages/kinetics
```

The worker requires a CUDA-capable GPU and a checkpoint for the
`md-et` force field; see `docs/reproducing.md` for the download
procedure.

The CPU worker requires external binaries that are not bundled:
`crest`, `xtb`, and `orca`. Refer to those projects' install guides.

A local PostgreSQL is sufficient for development; `docker-compose.yml`
brings one up.

## Reproducing the published numbers

The published runs ran across many GPU/CPU workers on Google Cloud
Batch. A faithful local reproduction is not practical; a *small-scale*
reproduction on a handful of seeds is. See `docs/reproducing.md`.

## License

TODO — pick a license before making the repository public. The current
state is "private, code preserved for paper review only".

## Citation

```bibtex
@article{TODO,
  title  = {TODO},
  author = {TODO},
  year   = {TODO}
}
```
