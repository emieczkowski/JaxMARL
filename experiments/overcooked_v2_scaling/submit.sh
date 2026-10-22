#!/bin/bash
# Submit the Overcooked V2 scaling sweep as a SLURM job array (run on the login node).
#
# Usage:
#   ./submit.sh              full sweep (all jobs in sweep.sh)
#   ./submit.sh 798,799      only these job indices, e.g. a pilot of the largest configs
#   ./submit.sh --list       print the job index -> configuration mapping and exit
set -euo pipefail

REPO_DIR="/scratch/gpfs/GRIFFITHS/em4240/JaxMARL"
CONDA_ENV="${CONDA_ENV:-/scratch/gpfs/GRIFFITHS/em4240/envs/jaxmarl}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"
EXP_DIR="$REPO_DIR/experiments/overcooked_v2_scaling"

if [ "${1:-}" = "--list" ]; then
  source "$EXP_DIR/sweep.sh"
  for (( i = 0; i < NUM_JOBS; i++ )); do
    decode_job "$i"
    echo "$i agents=$N_AGENTS pots=$N_POTS cook_time=$COOK no_collision=$NO_COLL"
  done
  exit 0
fi

echo "Pulling latest changes from Git..."
git -C "$REPO_DIR" pull

GIT_HASH=$(git -C "$REPO_DIR" rev-parse --short HEAD)

RUN_DIR="/scratch/gpfs/GRIFFITHS/em4240/runs/OV2_scaling_${GIT_HASH}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/wandb"

# Copy of the code for this sweep, so later pulls don't change queued jobs
rsync -a --exclude .git --exclude wandb --exclude outputs "$REPO_DIR/" "$RUN_DIR/code/"

source "$RUN_DIR/code/experiments/overcooked_v2_scaling/sweep.sh"
ARRAY_SPEC="${1:-0-$((NUM_JOBS - 1))}"

echo "Run directory: $RUN_DIR"
echo "Git commit: $GIT_HASH"

JOB_ID=$(sbatch \
  --parsable \
  --array="${ARRAY_SPEC}%${MAX_CONCURRENT}" \
  --export=ALL,RUN_DIR="$RUN_DIR",GIT_HASH="$GIT_HASH",CONDA_ENV="$CONDA_ENV" \
  --output="$RUN_DIR/logs/ov2_%A_%a.out" \
  "$RUN_DIR/code/experiments/overcooked_v2_scaling/run.slurm")

echo "Submitted array job $JOB_ID (tasks $ARRAY_SPEC)"
echo "Sync to W&B afterwards with: $EXP_DIR/sync.sh $RUN_DIR"
