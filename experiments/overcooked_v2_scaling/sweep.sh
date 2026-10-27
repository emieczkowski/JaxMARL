# Sweep grid for the Overcooked V2 scaling experiment (sourced by submit.sh and run.slurm).
# Each job trains NUM_SEEDS (=5, see ippo_rnn_overcooked_v2_scaling.yaml) vmapped seeds.
#
# A sweep file defines CONFIG_NAME, RUN_PREFIX, NUM_JOBS and decode_job, which sets
# JOB_ARGS (the Hydra overrides of one job). Select a sweep file with SWEEP=<file> ./submit.sh.

CONFIG_NAME=ippo_rnn_overcooked_v2_scaling
RUN_PREFIX=OV2_scaling

AGENTS=(1 2 3 4 5 6 7 8 9 10)
POTS=(1 2 3 4 5 6 7 8 9 10)
COOK_TIMES=(5 10 30 60)  # env steps
NO_COLLISION=(False True)

NUM_JOBS=$(( ${#AGENTS[@]} * ${#POTS[@]} * ${#COOK_TIMES[@]} * ${#NO_COLLISION[@]} ))

# Decode a job index (0 .. NUM_JOBS-1) into JOB_ARGS.
# Index order (slowest to fastest): agents, pots, cook time, collision.
decode_job() {
  local idx=$1
  local no_coll cook n_pots n_agents
  no_coll=${NO_COLLISION[idx % ${#NO_COLLISION[@]}]}; idx=$(( idx / ${#NO_COLLISION[@]} ))
  cook=${COOK_TIMES[idx % ${#COOK_TIMES[@]}]}; idx=$(( idx / ${#COOK_TIMES[@]} ))
  n_pots=${POTS[idx % ${#POTS[@]}]}; idx=$(( idx / ${#POTS[@]} ))
  n_agents=${AGENTS[idx]}
  JOB_ARGS=(
    ENV_KWARGS.num_agents="$n_agents"
    ENV_KWARGS.num_pots="$n_pots"
    ENV_KWARGS.pot_cook_time="$cook"
    ENV_KWARGS.no_collision="$no_coll"
  )
}
