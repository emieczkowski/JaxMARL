#!/bin/bash
#SBATCH --job-name=ov2_scaling
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00

# One array task = one sweep configuration (5 vmapped seeds). Submitted by submit.sh,
# which sets SNAPSHOT_DIR, OUT_DIR and SIF.

module purge

source "$SNAPSHOT_DIR/experiments/overcooked_v2_scaling/sweep.sh"
decode_job "$SLURM_ARRAY_TASK_ID"
echo "Task $SLURM_ARRAY_TASK_ID: agents=$N_AGENTS pots=$N_POTS cook_time=$COOK no_collision=$NO_COLL"

apptainer exec --nv \
  --bind "$SNAPSHOT_DIR:/home/workdir/JaxMARL" \
  --bind "$SNAPSHOT_DIR:/home/local_workdir/JaxMARL" \
  --bind "$OUT_DIR:$OUT_DIR" \
  --env PYTHONPATH=/home/local_workdir/JaxMARL:/home/workdir/JaxMARL \
  --env WANDB_DIR="$OUT_DIR/wandb" \
  "$SIF" /usr/bin/python3 /home/workdir/JaxMARL/baselines/IPPO/ippo_rnn_overcooked_v2.py \
  --config-name ippo_rnn_overcooked_v2_scaling \
  ENV_KWARGS.num_agents="$N_AGENTS" \
  ENV_KWARGS.num_pots="$N_POTS" \
  ENV_KWARGS.pot_cook_time="$COOK" \
  ENV_KWARGS.no_collision="$NO_COLL" \
  GROUP="$SLURM_ARRAY_JOB_ID" \
  hydra.run.dir="$OUT_DIR/hydra/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
