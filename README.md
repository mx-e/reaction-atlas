# reaction-atlas

Companion code for the paper *ReactionAtlas: Ab origine exploration of
chemical reaction networks with machine learning* (Gugler, Eissler,
Kahouli, and Müller), [arXiv:2606.30778](https://arxiv.org/abs/2606.30778).

This repository contains the code that was used to generate and validate
the reaction networks reported in the paper. It is preserved
**as-it-ran in production**: no scientific module has been refactored,
renamed, or simplified for release. The cloud-orchestration layer
(Terraform, Cloud Batch, GKE manifests) was specific to Google Cloud and
has not been included; any new operator will want to write their own
orchestration anyway. The published runs used many GPU/CPU workers
coordinated through PostgreSQL work queues; running a single worker
locally against a local Postgres is sufficient to exercise the full
pipeline on a small seed set.

The full reaction-network database that backs the paper's figures, and
the seed inputs that initialised the published runs, will be made
available at <https://reactionatlas.bifold.berlin/downloads>. This
repository ships only the code.

## Repository layout

| Path | Purpose | SI reference |
|---|---|---|
| `packages/worker/` | GPU worker: PES loop, generative TS loop, IRC, reaction-graph assembly. The diffusion-proposer checkpoint (`models/ts_best_model`) is committed. | SI §1, §2, §6 |
| `packages/cpu-worker/` | CPU worker: CREST conformer search, ORCA TS correction, PBE0 DFT barrier validation. | SI §1.7, §2.6 |
| `packages/kinetics/` | Reaction-network ODE integrator (PETSc / SciPy backends) and snapshot writer. | SI §3 |
| `packages/db/` | SQLAlchemy schema + Alembic migrations. The schema is the canonical record of what each pipeline stage produced. | SI §6 |
| `packages/MoreRed_src/` | Source of the diffusion proposer used by the generative loop. | SI §2.1 |
| `data/` | Reference structures used at runtime (start xyz, fragment library). | SI §1.1, §4 |

## Where to start reading

1. **`docs/architecture.md`**: block diagram of how the worker, DB,
   and kinetics solver interact.
2. **`docs/schema.md`**: the database schema, table by table.
3. **`docs/reproducing.md`**: exact commands to rerun each loop on a
   small seed set.
4. **`docs/data.md`**: pointer to the dataset and notes on the
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

Released under the [PolyForm Noncommercial License
1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/); see
[`LICENSE.md`](LICENSE.md). Any noncommercial purpose (academic
research, teaching, evaluation, and reproduction of the paper's results)
is permitted. This license does not grant rights for commercial use.

A patent application covering the method implemented here is pending.
The noncommercial license above does not grant any right to practice
that method for commercial purposes; for commercial or patent
licensing, contact the authors.

## Citation

```bibtex
@article{gugler2026reactionatlas,
  title   = {ReactionAtlas: Ab origine exploration of chemical reaction networks with machine learning},
  author  = {Gugler, Stefan and Eissler, Max and Kahouli, Khaled and M\"uller, Klaus-Robert},
  journal = {arXiv preprint arXiv:2606.30778},
  year    = {2026}
}
```
