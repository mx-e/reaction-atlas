# Database schema

Canonical definition: `packages/db/models.py`. Migrations:
`packages/db/alembic/versions/`.

The schema is split into three layers: the **chemistry layer** (what
the network *is*), the **work-queue layer** (how it got built), and
the **analysis layer** (what was computed *on* it).

## Chemistry layer

| Table | Purpose |
|---|---|
| `compounds` | Canonical molecules, keyed by SMILES and net charge. Stores formula, atom count, frontier-set membership. Aggregates conformers via `Minimum`. |
| `minima` | A relaxed conformer of a `Compound`. Holds XYZ coordinates, ML and (optionally) PBE0 energies, and the originating `experiment_tag`. |
| `intra_transition_states` | A TS that connects two `Minimum` rows of the *same* compound graph. Stores the saddle geometry, Hessian metadata, ML barrier, PBE0 barriers (forward and backward), and any post-hoc PBE0 TS correction. |
| `reactions` | An edge in the reaction graph: ordered (reactants → products) with a backreference to an `IntraTransitionState`. Carries the IRC path, validation flags (`is_validated`, `validation_method`), and stoichiometry. |
| `reaction_reactants`, `reaction_products` | Stoichiometric join tables: `(reaction_id, compound_id, count)`. |
| `graph_edges` | Legacy edge table from the pre-Reaction schema (migration 001). Kept for compatibility; no longer written. |

## Work-queue layer

These are the coordination tables the Cloud Batch fleet used. They are
still required for a single local worker because the same code path is
exercised.

| Table | Purpose |
|---|---|
| `pes_work_queue` | One row per (compound, experiment) the GPU worker should explore. `status ∈ {pending, in_progress, done, failed}`. |
| `crest_work_queue` | New compounds awaiting CREST conformer search. |
| `dft_work_queue` | TS rows awaiting PBE0 single-point or TS-correction. |
| `worker_heartbeat` | Liveness pings — used to detect crashed workers and unstick `in_progress` rows on the next sweep. |
| `crest_result` | Raw CREST output (conformer XYZ blob + ensemble energies), keyed by compound. |

## Analysis layer

| Table | Purpose |
|---|---|
| `exploration_stats` | Per-batch summary written by the worker: number of candidates proposed, accepted, IRC-rejected, etc. Method tag distinguishes PES vs generative. |
| `batch_log` | Free-form log entries about a worker batch (errors, timings). |
| `kinetics_snapshot` | One row per kinetics integration: input reaction-graph fingerprint, solver config, final concentration vector (as a SBML-style blob), and metadata. |
| `annotation`, `saved_layout` | Frontend-only state (no longer used; frontend is not shipped in this repo). |

## Migrations

Listed in `packages/db/alembic/versions/`:

| Revision | What it added |
|---|---|
| `001_initial_schema.py` | Compounds, minima, intra_transition_states, reactions + stoichiometry. |
| `002_kinetics_dft_schema.py` | DFT result columns, kinetics_snapshot. |
| `003_crest_rmsd_match.py` | CREST result storage + RMSD-match indexing. |
| `004_experiments_tagging.py` | Per-experiment `experiment_tag` on compounds and minima for multi-tenant runs. |
| `005_tag_formose_drilldown.py` | Tag backfill for the formose drilldown experiment. |
| `006_ts_hessian_pbe0.py` | Stored Hessian + PBE0 columns on `intra_transition_states`. |
| `007_compounds_frontier_in.py` | Frontier set membership on `compounds` (used by sampling logic). |
| `007_ts_corrected_pbe0.py` | TS-correction columns (post-hoc ORCA re-optimisation). |

The two `007_*` revisions share the same prefix because they are two
heads merged together in production; Alembic handles them correctly.

## SQLite caveat

`packages/db/models.py` declares JSONB and TEXT[] columns with SQLite
fallbacks via `JSON().with_variant(...)`. Unit tests can therefore run
against an in-memory SQLite, but the array operators degrade to JSON
queries and array `.any()` filtering is not available. The published
runs always used PostgreSQL.
