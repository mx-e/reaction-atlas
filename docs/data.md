# Data

This repository ships only the **runtime reference structures** the
worker needs to start (fragment libraries, sample start geometries).
Seeds, the full reaction-network database, and the model checkpoints
are released separately.

## Released externally

| Resource | Location | Size | Contents |
|---|---|---|---|
| Full reaction-network DB | Zenodo — [10.5281/zenodo.21358136](https://doi.org/10.5281/zenodo.21358136) | ~26 GB compressed | PostgreSQL dump of all published runs: compounds, minima, transition states, reactions, and DFT validation. The `kinetics_snapshots` are **not** included (they are large and fully regenerable from the reaction graph — see below). Restore with `pg_restore` (see Zenodo README). |
| Seed inputs | Zenodo — [10.5281/zenodo.21358136](https://doi.org/10.5281/zenodo.21358136) | small | The neutral-seed set used to initialise the published runs (SI §4). |
| `md-et` force-field checkpoint | Hugging Face, via the [`md-et`](https://arxiv.org/abs/2503.01431) package | — | Downloaded on first use by `lib/md_et_calculator.py` / `lib/energy.py`. Not in this repo. See the README "System requirements" for the package/checkpoint access. |

The hosted interactive explorer (the `crn-cloud` frontend/API) and the Zenodo
dataset are linked from the top-level README ("Links & related resources"). The
`kinetics_snapshots` table is excluded from the dump because it is large and can
be regenerated from the restored reaction graph with the kinetics solver:
`uv run --extra db python -m packages.kinetics.run --experiment main`.

## Bundled in this repository

| Path | Size | Purpose |
|---|---|---|
| `data/start_xyz/` | small | Single-compound start geometries (e.g. `glycolaldehyde.xyz`) for the local reproduction workflow described in `docs/reproducing.md`. |
| `data/fragments/` | ~7 MB | Fragment library used by `lib/fragment_mols.py` for combinatorial fragment substitution. |
| `data/buffer_fragments/` | small | Buffer-region fragments (charge-neutralising etc.). |
| `data/reference/` | small | Reference small-molecule structures used for sanity checks. |
| `packages/worker/models/ts_best_model` | ~10 MB | Diffusion-proposer checkpoint. Committed because the published runs depended on this exact weights file. The MoreRed source that consumes it is in `packages/MoreRed_src/`. |

## Files explicitly *not* shipped

The following live in the source-of-truth repo but are intentionally
excluded from `reaction-atlas`:

- `final-dump-*.sql.gz` (~30 GB compressed) — superseded by the
  dataset release at
  [reactionatlas.bifold.berlin/downloads](https://reactionatlas.bifold.berlin/downloads)
  *(forthcoming)*.
- `test_data/` (~10 GB) — internal regression artefacts.
- `neutral_pairs_v12l*/` — intermediate build artefacts from a
  re-ranking experiment; not used by the published pipeline.
- `sa-key.json`, `.terraform/`, any `.env` files — credentials.
