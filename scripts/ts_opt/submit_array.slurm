#!/bin/bash
# SLURM array template for scaling the DFT TS-optimisation to a whole dataset.
#
# Layout assumed (adapt INPUT_XYZ / OUT_DIR / CHARGE to your dataset):
#   <DATASET>/rxn_0001/ts.xyz      approximate TS geometry
#   <DATASET>/rxn_0001/meta.json   contains an integer field "ts_charge"
#   ...
#
# Submit one task per reaction:
#   sbatch --array=1-1000 submit_array.slurm
#
# Each task runs run_ts_opt.py on one reaction (~30 min median on 4 cores for
# small organic TSs; raise --time for larger systems).

#SBATCH --job-name=ts-opt
#SBATCH --partition=cpu-2d
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/ts-opt-%A_%a.out
#SBATCH --error=logs/ts-opt-%A_%a.err

set -euo pipefail

# --- paths: edit these for your dataset --------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-$HERE/dataset}"          # dir of rxn_NNNN/ input folders
RESULTS="${RESULTS:-$HERE/results}"          # dir for rxn_NNNN/ outputs
mkdir -p "$HERE/logs"

# --- ORCA -------------------------------------------------------------------
export ORCA_BIN="${ORCA_BIN:-/home/local/orca/orca}"
export ORCA_NPROCS="${SLURM_CPUS_PER_TASK:-4}"
export PATH="$(dirname "$ORCA_BIN"):$PATH"   # ORCA needs its own dir on PATH
export OMPI_MCA_rmaps_base_oversubscribe=true

# Run ORCA in fast node-local scratch, cleaned up on exit.
SCRATCH="${TMPDIR:-/tmp}/ts-opt-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SCRATCH"; export TMPDIR="$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

IDX=$(printf "%04d" "$SLURM_ARRAY_TASK_ID")
RXN_DIR="$DATASET/rxn_${IDX}"
[ -d "$RXN_DIR" ] || { echo "missing $RXN_DIR -- skipping"; exit 0; }

CHARGE=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['ts_charge'])" \
                  "$RXN_DIR/meta.json")

echo "[task $SLURM_ARRAY_TASK_ID] host=$(hostname) start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python "$HERE/run_ts_opt.py" \
    --xyz    "$RXN_DIR/ts.xyz" \
    --charge "$CHARGE" \
    --out    "$RESULTS/rxn_${IDX}"
echo "[task $SLURM_ARRAY_TASK_ID] done=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
