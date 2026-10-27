#!/bin/bash
# Submit the Overcooked V2 scaling sweep as a SLURM job array (run on the login node).
#
# Usage:
#   ./submit.sh              full sweep (all jobs in the sweep file)
#   ./submit.sh 798,799      only these job indices, e.g. a pilot of the largest configs
#   ./submit.sh --list       print the job index -> configuration mapping and exit
#
# The sweep file defaults to sweep.sh (the scaling sweep); select another with e.g.
#   SWEEP=sweep_order.sh ./submit.sh
set -euo pipefail

REPO_DIR="/scratch/gpfs/GRIFFITHS/em4240/JaxMARL"
CONDA_ENV="${CONDA_ENV:-/scratch/gpfs/GRIFFITHS/em4240/envs/jaxmarl}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"
SWEEP="${SWEEP:-sweep.sh}"
EXP_DIR="$REPO_DIR/experiments/overcooked_v2_scaling"

if [ "${1:-}" = "--list" ]; then
  source "$(dirname "$0")/$SWEEP"
  for (( i = 0; i < NUM_JOBS; i++ )); do
    decode_job "$i"
    echo "$i ${JOB_ARGS[*]}"
  done
  exit 0
fi

echo "Pulling latest changes from Git..."
git -C "$REPO_DIR" pull

GIT_HASH=$(git -C "$REPO_DIR" rev-parse --short HEAD)

source "$EXP_DIR/$SWEEP"
RUN_DIR="/scratch/gpfs/GRIFFITHS/em4240/runs/${RUN_PREFIX}_${GIT_HASH}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/wandb"

# Copy of the code for this sweep, so later pulls don't change queued jobs
rsync -a --exclude .git --exclude wandb --exclude outputs "$REPO_DIR/" "$RUN_DIR/code/"

source "$RUN_DIR/code/experiments/overcooked_v2_scaling/$SWEEP"
ARRAY_SPEC="${1:-0-$((NUM_JOBS - 1))}"

echo "Run directory: $RUN_DIR"
echo "Git commit: $GIT_HASH"
echo "Sweep: $SWEEP ($NUM_JOBS jobs, config $CONFIG_NAME)"

JOB_ID=$(sbatch \
  --parsable \
  --array="${ARRAY_SPEC}%${MAX_CONCURRENT}" \
  --export=ALL,RUN_DIR="$RUN_DIR",GIT_HASH="$GIT_HASH",CONDA_ENV="$CONDA_ENV",SWEEP="$SWEEP" \
  --output="$RUN_DIR/logs/ov2_%A_%a.out" \
  "$RUN_DIR/code/experiments/overcooked_v2_scaling/run.slurm")

echo "Submitted array job $JOB_ID (tasks $ARRAY_SPEC)"
echo "Sync to W&B afterwards with: $EXP_DIR/sync.sh $RUN_DIR"
