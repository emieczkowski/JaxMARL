#!/bin/bash
#SBATCH --job-name=ov2_scaling
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00

# One array task = one sweep configuration (5 vmapped seeds). Submitted by submit.sh,
# which sets RUN_DIR, GIT_HASH and CONDA_ENV.

module purge
module load anaconda3/2024.6
conda activate "$CONDA_ENV"

CODE_DIR="$RUN_DIR/code"
source "$CODE_DIR/experiments/overcooked_v2_scaling/sweep.sh"
decode_job "$SLURM_ARRAY_TASK_ID"
echo "Task $SLURM_ARRAY_TASK_ID (commit $GIT_HASH): agents=$N_AGENTS pots=$N_POTS cook_time=$COOK no_collision=$NO_COLL"

# Run this sweep's copy of the code, not the (editable-installed) repo
export PYTHONPATH="$CODE_DIR"
export WANDB_DIR="$RUN_DIR/wandb"

python "$CODE_DIR/baselines/IPPO/ippo_rnn_overcooked_v2.py" \
  --config-name ippo_rnn_overcooked_v2_scaling \
  ENV_KWARGS.num_agents="$N_AGENTS" \
  ENV_KWARGS.num_pots="$N_POTS" \
  ENV_KWARGS.pot_cook_time="$COOK" \
  ENV_KWARGS.no_collision="$NO_COLL" \
  GROUP="$(basename "$RUN_DIR")" \
  GIT_HASH="$GIT_HASH" \
  hydra.run.dir="$RUN_DIR/hydra/$SLURM_ARRAY_TASK_ID"
