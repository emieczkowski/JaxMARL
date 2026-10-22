# Sweep grid for the Overcooked V2 scaling experiment (sourced by submit.sh and run.slurm).
# Each job trains NUM_SEEDS (=5, see ippo_rnn_overcooked_v2_scaling.yaml) vmapped seeds.

AGENTS=(1 2 3 4 5 6 7 8 9 10)
POTS=(1 2 3 4 5 6 7 8 9 10)
COOK_TIMES=(5 10 30 60)  # env steps
NO_COLLISION=(False True)

NUM_JOBS=$(( ${#AGENTS[@]} * ${#POTS[@]} * ${#COOK_TIMES[@]} * ${#NO_COLLISION[@]} ))

# Decode a job index (0 .. NUM_JOBS-1) into N_AGENTS, N_POTS, COOK, NO_COLL.
# Index order (slowest to fastest): agents, pots, cook time, collision.
decode_job() {
  local idx=$1
  NO_COLL=${NO_COLLISION[idx % ${#NO_COLLISION[@]}]}; idx=$(( idx / ${#NO_COLLISION[@]} ))
  COOK=${COOK_TIMES[idx % ${#COOK_TIMES[@]}]}; idx=$(( idx / ${#COOK_TIMES[@]} ))
  N_POTS=${POTS[idx % ${#POTS[@]}]}; idx=$(( idx / ${#POTS[@]} ))
  N_AGENTS=${AGENTS[idx]}
}
