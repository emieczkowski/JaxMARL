# Sweep grid for the fixed-work ("order") variant of the scaling experiment (sourced by
# submit.sh and run.slurm when SWEEP=sweep_order.sh). Each episode ends once the team has
# delivered an order of K soups, with K pots; the outcome is the completion time.
# Each job trains NUM_SEEDS (=5, see ippo_rnn_overcooked_v2_order.yaml) vmapped seeds.

CONFIG_NAME=ippo_rnn_overcooked_v2_order
RUN_PREFIX=OV2_order

AGENTS=(1 2 3 4 5 6 7 8 9 10)
ORDER_SIZES=(1 2 5 10)  # soups per order; also the number of pots
COOK_TIMES=(5 20 60)  # env steps
RECIPE_SIZES=(1 3)  # onions per soup
NO_COLLISION=(True False)

NUM_JOBS=$(( ${#AGENTS[@]} * ${#ORDER_SIZES[@]} * ${#COOK_TIMES[@]} * ${#RECIPE_SIZES[@]} * ${#NO_COLLISION[@]} ))

# Decode a job index (0 .. NUM_JOBS-1) into JOB_ARGS.
# Index order (slowest to fastest): agents, order size, cook time, recipe size, collision.
decode_job() {
  local idx=$1
  local no_coll recipe cook order n_agents
  no_coll=${NO_COLLISION[idx % ${#NO_COLLISION[@]}]}; idx=$(( idx / ${#NO_COLLISION[@]} ))
  recipe=${RECIPE_SIZES[idx % ${#RECIPE_SIZES[@]}]}; idx=$(( idx / ${#RECIPE_SIZES[@]} ))
  cook=${COOK_TIMES[idx % ${#COOK_TIMES[@]}]}; idx=$(( idx / ${#COOK_TIMES[@]} ))
  order=${ORDER_SIZES[idx % ${#ORDER_SIZES[@]}]}; idx=$(( idx / ${#ORDER_SIZES[@]} ))
  n_agents=${AGENTS[idx]}
  JOB_ARGS=(
    ENV_KWARGS.num_agents="$n_agents"
    ENV_KWARGS.num_pots="$order"
    ENV_KWARGS.order_size="$order"
    ENV_KWARGS.pot_cook_time="$cook"
    ENV_KWARGS.recipe_size="$recipe"
    ENV_KWARGS.no_collision="$no_coll"
  )
}
