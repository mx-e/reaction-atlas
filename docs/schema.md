# Database schema

Canonical definition: `packages/db/models.py`. Migrations:
`packages/db/alembic/versions/`.

The schema splits into a **chemistry layer** (what the network *is*),
a **work-queue layer** (how it got built), and an **analysis layer**
(what was computed *on* it). Every entity-style table carries an
`experiments TEXT[]` column so the same database can hold several
independent runs side-by-side; queue/log tables carry a singular
`experiment TEXT` column instead.

## Conventions

- Energies are stored in **eV** (ML) or **eV** (PBE0), absolute or as
  barriers depending on the column; column names disambiguate.
- All geometry blobs (`positions`, `hessian`, `*_trajectory`) are raw
  numpy `float64` byte buffers — see `packages/db/serialization.py`
  for the read/write helpers.
- Timestamps are `TIMESTAMP WITH TIME ZONE`.
- JSON blobs use PostgreSQL JSONB (SQLite falls back to plain JSON).

---

## Chemistry layer

### `compounds` — canonical molecules

| Column | Type | Notes |
|---|---|---|
| `id` | int PK | |
| `smiles` | text UNIQUE NOT NULL | Canonical SMILES; primary identity key. |
| `formula` | text NOT NULL | Hill formula. |
| `charge` | int NOT NULL | Net molecular charge. |
| `n_atoms` | int NOT NULL | |
| `sorted_atomic_numbers` | bytes NOT NULL | Sorted Z vector blob; used for fast compound dedup before SMILES canonicalisation. |
| `is_seed` | bool | True for the initial seed compounds (water, formaldehyde, etc. — buffer chemistry needed by the kinetics solver). |
| `energy_pbe0`, `energy_pbe0_method`, `energy_pbe0_at` | float/text/ts | Denormalised PBE0 single-point at the lowest-energy minimum; used as the separated-barrier reference. Written by the cpu-worker DFT stage. |
| `experiments` | text[] | Multi-experiment membership. |
| `frontier_in` | text[] | Subset of `experiments` in which this compound is treated as a frontier (visible, not explorable). Used by closed-subgraph runs with `RESTRICT_TO_EXISTING_COMPOUNDS=true`. |
| `created_at` | ts | |

### `minima` — relaxed conformers

| Column | Type | Notes |
|---|---|---|
| `id` | int PK | |
| `compound_id` | FK → `compounds` | |
| `local_id` | int | Per-compound sequence (0, 1, 2…). UNIQUE(`compound_id`,`local_id`). |
| `positions` | bytes | XYZ geometry blob. |
| `energy` | float | ML force-field energy. |
| `hessian` | bytes | ML Hessian (optional). |
| `explored` | bool | Whether this minimum has been picked up by the PES loop yet. Driven by the partial index `idx_minima_unexplored`. |
| `name` | text | Human-readable label (often empty). |
| `n_merged`, `max_merge_rmsd` | int / float | Bookkeeping from the dedup pass that merged near-duplicate conformers into this row. |
| `discovery_timestamp` | float | Unix seconds at first discovery. |
| `energy_pbe0`, `energy_pbe0_method`, `energy_pbe0_at` | float/text/ts | PBE0 single-point. Only auto-populated for the lowest-E minimum of each compound. |
| `experiments` | text[] | |

### `intra_transition_states` — single-compound saddles

A TS connecting two `Minimum` rows of the *same* compound (a within-
compound conformational transition). Cross-compound reactions go into
the `reactions` table instead.

| Column | Type | Notes |
|---|---|---|
| `id` | int PK | |
| `compound_id` | FK → `compounds` | |
| `local_id` | bigint | Per-compound sequence. UNIQUE(`compound_id`,`local_id`). |
| `positions` | bytes | TS geometry. |
| `energy` | float | ML TS energy. |
| `eigenvalue` | float | The (most negative) Hessian eigenvalue at the saddle. |
| `hessian` | bytes | Optional ML Hessian blob. |
| `min_fwd_id`, `min_bwd_id` | FK → `minima` | Forward / backward minimum endpoints. |
| `barrier_fwd`, `barrier_bwd` | float | ML barriers (eV). |
| `rmsd_to_fwd_min`, `rmsd_to_bwd_min`, `endpoint_to_endpoint_rmsd` | float | IRC-validation diagnostics. |
| `fwd_trajectory`, `bwd_trajectory` | bytes | IRC trajectories (numpy float64 blobs). |
| `name` | text | |
| `discovery_timestamp` | float | |
| `energy_pbe0`, `barrier_fwd_pbe0`, `barrier_bwd_pbe0`, `energy_pbe0_method`, `energy_pbe0_at` | mixed | Optional PBE0 single-point on the TS geometry and derived in-box barriers. On-demand only. |
| `experiments` | text[] | |

### `reactions` — cross-compound reaction edges

The primary edge type in the reaction graph. Each row is a reaction
(reactants → products) with multi-stoichiometry encoded through the
`reaction_reactants` / `reaction_products` join tables.

The column count is large because the same reaction carries several
barrier variants (different bias-correction strategies) plus a full
post-hoc DFT refinement track. Grouped below by purpose.

**Identity and TS geometry:**

| Column | Type | Notes |
|---|---|---|
| `id` | int PK | |
| `ts_id` | bigint UNIQUE | TS identifier (matches `IntraTransitionState.local_id` when applicable). |
| `ts_conformer_positions`, `ts_conformer_atomic_numbers`, `ts_conformer_charge` | bytes / bytes / int | The TS geometry exactly as proposed (separate from `intra_transition_states` because cross-compound TSs are not stored there). |
| `ts_energy` | float | ML TS energy. |
| `reactant_trajectory`, `product_trajectory` | bytes | IRC trajectories on each side. |
| `discovery_method`, `discovery_noise_level`, `discovery_timestamp` | text / int / float | Provenance — which loop discovered this reaction (`pes`, `generative`, `manual_equilibrium`, …) and at what diffusion noise level. |
| `name`, `created_at`, `experiments` | text / ts / text[] | |

**ML barriers (eV).** Three variants, all stored for downstream comparison; the kinetics solver uses the `*_separated` pair.

| Column | Why a separate variant |
|---|---|
| `barrier_forward`, `barrier_backward` | In-box (TS − trajectory endpoint). Susceptible to ML long-distance artefacts. **Not** used by the solver. |
| `barrier_forward_separated`, `barrier_backward_separated` | Separated (TS − sum of reference-conformer energies of each reactant/product compound). The principled choice; bypasses long-distance artefacts. Used by the kinetics solver. |
| `barrier_forward_ex`, `barrier_backward_ex` | IRC-extremum (TS − min energy along IRC on each side). A weaker fix for the long-distance issue; stored for comparison. |

**DFT barriers (PBE0/def2-TZVPP).** Same in-box vs separated split as the ML barriers. Populated by the cpu-worker DFT stage.

| Column |
|---|
| `energy_R_pbe0`, `energy_TS_pbe0`, `energy_P_pbe0` |
| `barrier_forward_pbe0`, `barrier_backward_pbe0` — in-box DFT |
| `barrier_forward_separated_pbe0`, `barrier_backward_separated_pbe0` — kinetics-solver primary when present |
| `energy_pbe0_method`, `energy_pbe0_at` |

**Post-hoc PBE0 TS Hessian.** Optional ground-truth backfill, gated by `COMPUTE_TS_HESSIAN=1` on a cpu-worker.

| Column | Notes |
|---|---|
| `ts_hessian_pbe0` | Raw (3N,3N) float64 blob, unsymmetrised. |
| `ts_hessian_pbe0_at` | |
| `ts_hessian_pbe0_claimed_by`, `ts_hessian_pbe0_claimed_at` | Atomic-claim slots — the `reactions` table is itself the work pool for this job kind. |
| `ts_hessian_pbe0_failed` | Sticky-failure flag (skipped by future picks). |
| `ts_hessian_pbe0_wall_s` | Compute wall time (so the paper can quote it). |

**Corrected TS (PBE0 saddle re-optimisation).** Populated when `ts_ml_invalid=True`, i.e. when the PBE0 Hessian of the ML TS does not have exactly one imaginary mode.

| Column |
|---|
| `ts_ml_invalid` |
| `ts_pbe0_corrected_positions`, `ts_pbe0_corrected_energy`, `ts_pbe0_corrected_at` |
| `ts_pbe0_corrected_de` — E(corrected) − E(ML at ML geom) |
| `ts_pbe0_corrected_rmsd` — Kabsch RMSD vs ML geometry |
| `ts_pbe0_corrected_wall_s`, `ts_pbe0_corrected_failed` |
| `ts_pbe0_corrected_claimed_by`, `ts_pbe0_corrected_claimed_at` |

**Manual equilibrium rate constants.** A few reactions (water autoionisation, CO₂ hydration, …) use literal rate constants instead of Eyring-from-barriers; the kinetics solver consumes these directly (no temperature dependence).

| Column |
|---|
| `manual_k_fwd`, `manual_k_bwd` |

### `reaction_reactants`, `reaction_products` — stoichiometry

Per-conformer join tables. Multiplicity comes from row count, not a
`count` column.

| Column | Notes |
|---|---|
| `id` | PK |
| `reaction_id` | FK → `reactions` (CASCADE on delete) |
| `compound_id` | FK → `compounds` |
| `conformer_local_id` | int — which specific conformer of the compound participates. Nullable on the reactant side, required on the product side. |
| `energy` *(products only)* | float — ML energy of that product conformer at the trajectory endpoint. |

### `graph_edges` — legacy edge table

Pre-`reactions` representation kept around for the formose subgraph
tagging migration. Not written by the current worker code path.

---

## Work-queue layer

These coordinate workers via `SELECT … FOR UPDATE SKIP LOCKED`. A
single local worker exercises the same code path; many cloud workers
race on the same tables.

### `pes_work_queue`

One row per `(minimum)` the GPU worker should explore (or dedup). The
`job_kind` column distinguishes:

| Column | Notes |
|---|---|
| `id`, `compound_id`, `minimum_id` | UNIQUE(`minimum_id`). |
| `status` | `pending` / `in_progress` / `done` / `failed`. Partial index on `status='pending'`. |
| `job_kind` | `explore` or `dedup`. |
| `worker_id`, `claimed_at`, `completed_at` | Claim metadata. |
| `experiment` | Per-row experiment tag (singular, not array). |

### `crest_work_queue`

One row per compound awaiting CREST conformer search.
UNIQUE(`compound_id`). Status / claim columns analogous to PES.

### `dft_work_queue`

One row per `Reaction` awaiting DFT refinement.
UNIQUE(`reaction_id`). Auto-enqueued by `db.create_reaction()` so
kinetically relevant reactions (sorted by `barrier_forward ASC`) get
picked up first.

### `worker_heartbeats`

Live worker registry — each worker upserts a row every few seconds. PK
is `worker_id`. Records `worker_type` (`exploration` / `cpu`), current
`status` (`idle` / `pes` / `generative` / `dft` / `crest`), wall-time
accumulators, and per-experiment scope. Used to detect crashed workers
and unstick `in_progress` rows on the next sweep.

### `crest_results`

| Column | Notes |
|---|---|
| `id`, `compound_id` | UNIQUE(`compound_id`). |
| `n_conformers`, `s_conf` | Ensemble size and conformational entropy (cal/mol·K). |
| `conformers_xyz` | Raw multi-XYZ blob produced by CREST. |
| `crest_output` | Last lines of `crest.out` for debugging. |
| `charge` | Compound charge passed to CREST. |
| `rmsd_match` | JSON: `{best_rmsds: [float, ...], threshold: 0.125, n_our_minima: int}` — min Kabsch RMSD from each CREST conformer to any of our PES minima. |
| `created_at` | |

---

## Analysis layer

### `exploration_stats`

Per-experiment singleton (one row per experiment, keyed by the
`experiment` UNIQUE column). The `stats_json` blob holds the running
aggregate counters — proposed/accepted/rejected per stage,
discovery-method breakdown, etc. `updated_at` is bumped on every
write.

### `batch_log`

Per-batch summary written by the worker after each batch completes.
`summary_json` carries the structured payload (counts, timings,
errors); `batch_idx` and `experiment` give the dimensions to filter
on.

### `kinetics_snapshots`

One row per kinetics integration. The solver runs as a background
asyncio task (advisory-locked singleton across workers), polls the
reaction graph every ~60 s, and writes a new row when the network has
materially changed.

| Column | Notes |
|---|---|
| `id`, `experiment`, `computed_at` | |
| `network_version` | `COUNT(reactions WHERE discovery_method != 'manual_equilibrium')` at solve time. |
| `n_reactions_dft` | Of which N had separated PBE0 barriers (i.e. were eligible for the Eyring-from-DFT rate law). |
| `temperature` | Kelvin — sets `kBT` for Eyring. |
| `payload_jsonb` | Serialised `KineticsSnapshot` dataclass (the actual concentration trajectory + per-species summaries). Schema is defined in `packages/kinetics/snapshot.py`. |
| `solve_wall_time_s` | |

### `annotations`, `saved_layouts`

Frontend-only state from the production UI (not shipped in this
repo). Retained because the migrations and worker code still reference
the tables; never written by anything in `reaction-atlas`.

---

## Migrations

| Revision | What it added |
|---|---|
| `001_initial_schema.py` | All initial tables: compounds, minima, intra_transition_states, reactions + stoichiometry join tables, graph_edges, pes/crest work queues, worker_heartbeats, crest_results, exploration_stats, batch_log, annotations, saved_layouts. |
| `002_kinetics_dft_schema.py` | Separated and IRC-extremum ML barriers; PBE0 single-point energy slots on compounds, minima, intra_transition_states, reactions; manual equilibrium rate constants on reactions; `dft_work_queue`; `kinetics_snapshots`; `WorkerHeartbeat.current_job_kind`. |
| `003_crest_rmsd_match.py` | `crest_results.rmsd_match` JSONB column. |
| `004_experiments_tagging.py` | `experiments TEXT[]` on entity tables and `experiment TEXT` on queue/log tables; existing rows backfilled to `'main'` then default dropped so writes must set the column explicitly. |
| `005_tag_formose_drilldown.py` | Data-only migration that tags the formose subgraph (sourced from `annotations.label`) with `'formose-drilldown'`. |
| `006_ts_hessian_pbe0.py` | `ts_hessian_pbe0` blob + claim/timing/failure columns on reactions; partial index for the random-pick claim. |
| `007_compounds_frontier_in.py` | `compounds.frontier_in TEXT[]` + GIN index. Records experiments where a compound is a frontier (visible, not explorable). |
| `007_ts_corrected_pbe0.py` | `ts_ml_invalid` flag and the `ts_pbe0_corrected_*` corrected-TS columns on reactions. |

The two `007_*` revisions share the same prefix because they were
parallel branches merged in production; Alembic handles them as
distinct heads.

---

## SQLite caveat

`models.py` declares JSONB and TEXT[] columns with SQLite fallbacks
via `JSON().with_variant(...)`. Schema-load works against in-memory
SQLite, but the array operators degrade to JSON: `.any()` filtering
(which compiles to `:val = ANY(column)` on Postgres) is **not**
available. The published runs always used PostgreSQL.
