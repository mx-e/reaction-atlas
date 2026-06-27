#!/bin/bash
# One-shot smoke test for the inline TS-correction integration.
# Picks the smallest unclaimed reaction with ts_ml_invalid=TRUE and runs
# the full PBE0/def2-TZVPP OptTS+NumFreq pipeline via ts_corrected_runner.
# No DB writes — pure ORCA exercise.
#
# Submit with:
#   CRN_DB_PASSWORD=… sbatch scripts/smoke_ts_correction.slurm
#
# ~20-40 min wall for small organic TSs on 4 cores.

#SBATCH --job-name=crn-smoke-ts
#SBATCH --partition=cpu-5h
#SBATCH --account=pyml
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke-ts-%j.out
#SBATCH --error=logs/smoke-ts-%j.err

set -euo pipefail
mkdir -p logs

REPO="${CRN_REPO:-$HOME/MISC_PROJECTS/crn-cloud}"
# Use the original (no-MPI) SIF — host MPI gets bind-mounted in. Avoids a
# SIF rebuild AND pins us to the exact OpenMPI 4.1.2 ORCA was built against.
SIF_PATH="${CRN_CPU_SIF:-$REPO/crn-cpu-worker.sif}"
ORCA_DIR="${CRN_ORCA_DIR:-/home/local/orca}"
DB_PASSWORD="${CRN_DB_PASSWORD:?Set CRN_DB_PASSWORD env}"

# shellcheck source=host_mpi_binds.sh
source "$REPO/scripts/host_mpi_binds.sh"

export APPTAINERENV_DATABASE_URL="postgresql://crn_worker:${DB_PASSWORD}@pgx06.ml.tu-berlin.de:5432/crn_cloud"
export APPTAINERENV_ORCA_BIN="$ORCA_DIR/orca"
export APPTAINERENV_ORCA_NPROCS="${SLURM_CPUS_PER_TASK:-4}"
# Single-host fork launcher — skips the rsh/ssh agent (no ssh in the SIF) and
# bypasses the SLURM auto-detect (which fails inside apptainer since OMPI
# can't see srun). All ORCA MPI ranks fork on the local node anyway.
export APPTAINERENV_OMPI_MCA_plm=isolated
# Without SLURM-aware launcher, OMPI's slot-counting defaults to localhost:1
# and refuses to spawn N>1 ranks. We're on one node and SLURM already allocated
# the CPUs to us, so oversubscribe is the right knob (not actual oversubscription).
export APPTAINERENV_OMPI_MCA_rmaps_base_oversubscribe=true

echo "[smoke] host=$(hostname) cpus=$SLURM_CPUS_PER_TASK orca=$ORCA_DIR/orca"
echo "[smoke] sif=$SIF_PATH"

if [ ! -x "$ORCA_DIR/orca" ]; then
    echo "ERROR: ORCA not at $ORCA_DIR/orca on this node." >&2
    exit 1
fi

# Code + ORCA bind set, plus the host-MPI bind set sourced above.
BIND_ARGS=(
    "${HOST_MPI_BINDS[@]}"
    --bind "$REPO/packages/cpu-worker/ts_corrected_runner.py":/app/ts_corrected_runner.py:ro
    --bind "$REPO/scripts/ts_opt":/app/scripts/ts_opt:ro
    --bind "$REPO/scripts/smoke_test_ts_correction.py":/app/scripts/smoke_test_ts_correction.py:ro
    --bind "$ORCA_DIR":"$ORCA_DIR":ro
)

echo
echo "=== Full ORCA OptTS+NumFreq on one reaction ==="
apptainer exec "${BIND_ARGS[@]}" "$SIF_PATH" \
    python3 /app/scripts/smoke_test_ts_correction.py
echo "=== Done ==="
