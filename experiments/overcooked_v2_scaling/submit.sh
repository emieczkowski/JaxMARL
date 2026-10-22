#!/bin/bash
# Submit the Overcooked V2 scaling sweep as a SLURM job array (run on the login node).
#
# Usage:
#   ./submit.sh              full sweep (all jobs in sweep.sh)
#   ./submit.sh 798,799      only these job indices, e.g. a pilot of the largest configs
#   ./submit.sh --list       print the job index -> configuration mapping and exit
#
# Env overrides: REPO, OUT_DIR, SIF, MAX_CONCURRENT
set -euo pipefail

REPO=${REPO:-/scratch/gpfs/$USER/JaxMARL}
OUT_DIR=${OUT_DIR:-/scratch/gpfs/$USER/overcooked_v2_scaling}
SIF=${SIF:-/scratch/gpfs/$USER/jaxmarl_latest.sif}
MAX_CONCURRENT=${MAX_CONCURRENT:-50}

source "$REPO/experiments/overcooked_v2_scaling/sweep.sh"

if [ "${1:-}" = "--list" ]; then
  for (( i = 0; i < NUM_JOBS; i++ )); do
    decode_job "$i"
    echo "$i agents=$N_AGENTS pots=$N_POTS cook_time=$COOK no_collision=$NO_COLL"
  done
  exit 0
fi

ARRAY_SPEC=${1:-0-$((NUM_JOBS - 1))}

echo "Pulling latest changes from Git..."
git -C "$REPO" pull

# One snapshot for the whole array, so later edits to $REPO don't affect queued jobs
SNAPSHOT_DIR=/scratch/gpfs/$USER/JaxMARL_snapshot_$(date +%Y%m%d_%H%M%S)
echo "Creating snapshot at $SNAPSHOT_DIR..."
rsync -a --delete --exclude wandb --exclude outputs "$REPO/" "$SNAPSHOT_DIR/"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/wandb"

JOB_ID=$(sbatch --parsable \
  --array="${ARRAY_SPEC}%${MAX_CONCURRENT}" \
  --output="$OUT_DIR/logs/%A_%a.out" \
  --export=ALL,SNAPSHOT_DIR="$SNAPSHOT_DIR",OUT_DIR="$OUT_DIR",SIF="$SIF" \
  "$SNAPSHOT_DIR/experiments/overcooked_v2_scaling/run.slurm")
echo "Submitted array job $JOB_ID (tasks $ARRAY_SPEC)"

# Remove the snapshot once every task has finished
sbatch --parsable --dependency=afterany:"$JOB_ID" --job-name=ov2_cleanup \
  --time=0:10:00 --output=/dev/null --wrap="rm -rf $SNAPSHOT_DIR" > /dev/null

echo "Logs: $OUT_DIR/logs   W&B offline runs: $OUT_DIR/wandb/wandb"
echo "Sync to W&B afterwards with: $REPO/experiments/overcooked_v2_scaling/sync.sh"
