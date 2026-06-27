# Data

This repository ships only **small, code-adjacent** datasets — seed
inputs, reference structures, and validation pairs. The full reaction-
network database that backs the paper's figures is published
separately.

## Released externally

| Resource | Location | Size | Contents |
|---|---|---|---|
| Full reaction-network DB | Zenodo — DOI **TODO** | ~30 GB compressed | PostgreSQL dump of all published runs: compounds, minima, transition states, reactions, DFT validation, kinetics snapshots. Restore with `pg_restore` (see Zenodo README). |
| `md-et` force-field checkpoint | HuggingFace (set `HF_TOKEN`) | — | Pulled at worker startup by `lib/md_et_calculator.py`. Not in this repo. |

## Bundled in this repository

| Path | Size | Purpose |
|---|---|---|
| `neutral_seeds/` | 616 KB | The 152 neutral seed molecules described in SI §4. XYZ files plus a `pairs.csv` index. Used as the initial frontier for the published runs. |
| `data/start_xyz/` | small | Single-compound start geometries (e.g. `glycolaldehyde.xyz`) for the local reproduction workflow described in `docs/reproducing.md`. |
| `data/fragments/` | ~7 MB | Fragment library used by `lib/fragment_mols.py` for combinatorial fragment substitution. |
| `data/buffer_fragments/` | small | Buffer-region fragments (charge-neutralising etc.). |
| `data/reference/` | small | Reference small-molecule structures used for sanity checks. |
| `conformer_pairs_dataset/` | 424 KB | Conformer-pair set used in the conformer-RMSD validation (SI §1.7). XYZ + pairs.csv. |
| `conformer_pairs_relaxed/` | 144 KB | Relaxed counterparts of the above. |
| `investigation/29_wikipedia_compounds/` | 12 KB | Auxiliary Wikipedia-compound labelling notes; reference only. |

## Notes on the diffusion-proposer checkpoint

`packages/worker/models/ts_best_model/` (~10 MB) **is** committed to
this repository because the published runs depended on this exact
checkpoint. The MoreRed source that consumes it lives in
`packages/MoreRed_src/`.

## Files explicitly *not* shipped

The following live in the source-of-truth repo but are intentionally
excluded from `reaction-atlas`:

- `final-dump-*.sql.gz` (~30 GB compressed) — superseded by the
  Zenodo release.
- `test_data/` (~10 GB) — internal regression artefacts.
- `neutral_pairs_v12l*/` — intermediate build artefacts from a
  re-ranking experiment; not used by the published pipeline.
- `sa-key.json`, `.terraform/`, any `.env` files — credentials.
