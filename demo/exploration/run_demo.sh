#!/usr/bin/env bash
#
# Demo 2 — Minimal single-seed exploration.
#
# Runs the REAL exploration worker (packages/worker/worker.py) at deliberately
# tiny scale on a single seed molecule (glycolaldehyde), on CPU if no GPU is
# present. It seeds the start molecule + fragment library + buffer equilibria,
# then runs the generative TS-proposal / MD-ET-validation / saddle-search / IRC
# loop until the (small) node cap is reached, writing discovered minima,
# transition states, and reactions into a local PostgreSQL.
#
# Prerequisites (see demo/exploration/README.md):
#   1. uv sync --extra worker        # torch, md-et, rdkit, schnetpack, ...
#   2. md-et model access (Hugging Face) — see README
#   3. docker compose up -d db       # local PostgreSQL
#   4. migrations applied            # see step 1 of docs/reproducing.md
#
# Usage:
#   ./demo/exploration/run_demo.sh
#
# Tunable via environment (defaults sized to discover on a laptop in minutes):
#   MAX_VALID_NODES (15)  TS_BATCH_SIZE (4)  PES_MD_STEPS (500)
#   PES_MAX_ITERATIONS (2)  MAX_DENOISING_STEPS (500)  ENERGY_THRESHOLD_EV (1000)

set -euo pipefail

# repo root = two levels up from this script
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export DATABASE_URL="${DATABASE_URL:-postgresql://crn:crn@localhost:5432/crn_cloud}"
export EXPERIMENT="${EXPERIMENT:-main}"

# Inputs shipped in this repository
export START_XYZ_PATH="${START_XYZ_PATH:-$REPO_ROOT/data/start_xyz/glycolaldehyde.xyz}"
export FRAGMENT_PATH="${FRAGMENT_PATH:-$REPO_ROOT/data/fragments}"
export BUFFER_FRAGMENT_PATH="${BUFFER_FRAGMENT_PATH:-$REPO_ROOT/data/buffer_fragments}"

# The generative-proposer checkpoint is committed here; the MD-ET force field
# is loaded from the md-et package (Hugging Face), so its *_MODEL_PATH is unused.
export TS_MODEL_PATH="${TS_MODEL_PATH:-$REPO_ROOT/packages/worker/models}"
export FORCES_MODEL_PATH="${FORCES_MODEL_PATH:-$REPO_ROOT/packages/worker/models}"
export ENERGY_MODEL_PATH="${ENERGY_MODEL_PATH:-$REPO_ROOT/packages/worker/models}"

# Small bounded configuration so a single worker terminates in minutes while
# still exploring enough (longer MD + more denoising) to actually *discover* a
# new compound rather than just re-finding the seeds.
export MAX_VALID_NODES="${MAX_VALID_NODES:-15}"
export TS_BATCH_SIZE="${TS_BATCH_SIZE:-4}"
export PES_MD_STEPS="${PES_MD_STEPS:-500}"
export PES_MAX_ITERATIONS="${PES_MAX_ITERATIONS:-2}"
export MAX_DENOISING_STEPS="${MAX_DENOISING_STEPS:-500}"
# Barrier acceptance cap (eV). Production keeps the physical default
# (ENERGY_THRESHOLD_HARTREE = 0.1 Ha ≈ 2.72 eV), which rejects essentially every
# reaction reachable from a single tiny seed — so a laptop-scale demo would grow
# no network. We relax it here so genuinely higher-barrier reactions (e.g. the
# glycolaldehyde → formaldehyde + hydroxycarbene fragmentation, ~2.9 eV) still
# register as discoveries. Set ENERGY_THRESHOLD_EV=2.72 to reproduce production.
export ENERGY_THRESHOLD_EV="${ENERGY_THRESHOLD_EV:-1000}"
# The demo does not run the background kinetics solver, so the generative loop
# samples compounds uniformly instead of by kinetic concentration.
export KINETIC_SAMPLING_ENABLED="${KINETIC_SAMPLING_ENABLED:-false}"
# Keep the console readable (the worker emits verbose DEBUG polling lines).
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

echo "Seed molecule : $START_XYZ_PATH"
echo "Database      : $DATABASE_URL"
echo "Node cap      : $MAX_VALID_NODES  (worker exits when the graph reaches this size)"
echo

# worker.py mixes `lib.*` imports (needs packages/worker on the path) and
# `packages.*` imports (needs the repo root) — put both on PYTHONPATH.
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/packages/worker${PYTHONPATH:+:$PYTHONPATH}"

cd packages/worker
exec uv run --extra worker python worker.py
