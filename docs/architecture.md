# Architecture

Two loops generate transition-state candidates; both write into the same
PostgreSQL schema; a separate kinetics solver consumes the resulting
reaction graph.

```
                  ┌─────────────────────────────────────────┐
                  │       PostgreSQL (packages/db)          │
                  │  compounds, minima, ts, reactions,      │
                  │  work_queues, kinetics_snapshots        │
                  └────────────────┬────────────────────────┘
                                   │
       ┌───────────────────────────┼────────────────────────────┐
       │                           │                            │
┌──────┴──────────┐    ┌───────────┴────────────┐   ┌───────────┴──────────┐
│  GPU worker     │    │   CPU worker           │   │  Kinetics solver     │
│  (packages/     │    │  (packages/            │   │  (packages/          │
│   worker)       │    │   cpu-worker)          │   │   kinetics)          │
├─────────────────┤    ├────────────────────────┤   ├──────────────────────┤
│ • PES loop:     │    │ • CREST conformer      │   │ • Builds rate matrix │
│   MD + P-RFO    │    │   search per compound  │   │   from reactions     │
│   saddle search │    │ • ORCA TS correction   │   │ • Integrates the     │
│                 │    │   (re-optimise TS at   │   │   ODE system         │
│ • Generative    │    │   reference level)     │   │ • Writes snapshots   │
│   loop:         │    │ • PBE0 single-point    │   │   periodically       │
│   MoreRed       │    │   barriers for         │   │                      │
│   diffusion →   │    │   validation           │   │                      │
│   IRC verify    │    │                        │   │                      │
└─────────────────┘    └────────────────────────┘   └──────────────────────┘
```

## Worker (GPU)

`packages/worker/worker.py` is the main loop. On startup it claims an
experiment via `PESWorkQueue`, loads the diffusion checkpoint
(`packages/worker/models/ts_best_model`) and the `md-et` force field,
then alternates between two candidate generators:

- **PES loop** — `lib/pes_explorer/`. Short MD runs from a relaxed
  minimum generate seed geometries; `prfo.py` (Partitioned Rational
  Function Optimisation) refines each one to a first-order saddle and
  rejects anything that does not exit with exactly one negative
  Hessian eigenvalue. The accepted saddle is then validated via IRC
  (`lib/ts_pipeline.py:_irc_displace_and_relax`) to confirm it
  connects two distinct minima.

- **Generative loop** — `lib/ts_pipeline.py`. The MoreRed diffusion
  proposer emits TS candidates; each is Hessian-validated (≥1
  significant negative eigenvalue) and IRC-traced. Higher-order
  saddles are filtered implicitly by the endpoint-identity and
  bidirectional-barrier gates rather than by an explicit eigenvalue
  recount; see SI §2.

Both loops produce `Reaction` rows linking two `Minimum` rows via an
`IntraTransitionState`. Reaction-graph assembly and duplicate
detection happen in `lib/reaction_graph.py`.

## Worker (CPU)

`packages/cpu-worker/worker.py` consumes the CPU queues. Two stages:

- **CREST** (`packages/cpu-worker/`) — for any new compound, search
  the conformer ensemble to canonicalise the lowest-energy conformer
  used in subsequent steps. RMSD matching against existing compounds
  lives in `rmsd_match.py`.

- **DFT** (`packages/cpu-worker/dft_runner.py`,
  `ts_corrected_runner.py`) — re-optimise selected TS geometries with
  ORCA at the reference level (PBE0); compute PBE0 single-point
  energies on both endpoints. Stores back into
  `IntraTransitionState.energy_TS_pbe0` etc. for the barrier
  validation reported in SI §1.7.

## Kinetics

`packages/kinetics/loop.py` repeatedly:

1. Snapshots the current reaction graph from the DB
2. Builds the sparse rate matrix (Arrhenius from validated barriers)
3. Integrates the ODE system (`scipy_solver.py` or PETSc backend in
   `build.py`)
4. Writes a `KineticsSnapshot` row with the final concentrations

This loop runs alongside exploration; later kinetics snapshots see
later reaction networks.

## Coordination via Postgres

There is no message broker or service mesh. Workers are stateless and
coordinate exclusively through PostgreSQL using SELECT … FOR UPDATE
SKIP LOCKED on the queue tables. This was deliberate — it keeps the
methodology re-runnable from any DB snapshot without replaying broker
state. The published runs used Cloud Batch to scale workers; a single
local worker reads the same queues identically.
