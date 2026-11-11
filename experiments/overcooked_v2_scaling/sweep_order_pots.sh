# Sweep grid for the order variant with pots and order size crossed (sourced by submit.sh
# and run.slurm when SWEEP=sweep_order_pots.sh). Same task as sweep_order.sh, where the
# number of pots always equals the order size K; here they vary independently: K sets the
# total work, the number of pots P how many soups can cook at once (pots are reused when
# K > P, some sit idle when K < P).
# Pots are placed nearest first (pot_placement=nearest: the cells where a soup takes the
# fewest steps), so adding pots adds capacity without bringing a pot closer. sweep_order.sh
# used the spread placement, so none of its runs are reused: all jobs run.
# Each job trains NUM_SEEDS (=5, see ippo_rnn_overcooked_v2_order.yaml) vmapped seeds.

CONFIG_NAME=ippo_rnn_overcooked_v2_order
RUN_PREFIX=OV2_order_pots

AGENTS=(1 2 3 4 5 6 7 8 9 10)
POTS=(1 2 5 10)
ORDER_SIZES=(1 2 5 10)  # soups per order
COOK_TIMES=(5 20 60)  # env steps
RECIPE_SIZES=(1)  # onions per soup (the 3-onion recipe is not run in this sweep)
NO_COLLISION=(True False)

NUM_JOBS=$(( ${#AGENTS[@]} * ${#POTS[@]} * ${#ORDER_SIZES[@]} * ${#COOK_TIMES[@]} * ${#RECIPE_SIZES[@]} * ${#NO_COLLISION[@]} ))

# Decode a job index (0 .. NUM_JOBS-1) into JOB_ARGS.
# Index order (slowest to fastest): agents, pots, order size, cook time, recipe size, collision.
decode_job() {
  local idx=$1
  local no_coll recipe cook order pots n_agents
  no_coll=${NO_COLLISION[idx % ${#NO_COLLISION[@]}]}; idx=$(( idx / ${#NO_COLLISION[@]} ))
  recipe=${RECIPE_SIZES[idx % ${#RECIPE_SIZES[@]}]}; idx=$(( idx / ${#RECIPE_SIZES[@]} ))
  cook=${COOK_TIMES[idx % ${#COOK_TIMES[@]}]}; idx=$(( idx / ${#COOK_TIMES[@]} ))
  order=${ORDER_SIZES[idx % ${#ORDER_SIZES[@]}]}; idx=$(( idx / ${#ORDER_SIZES[@]} ))
  pots=${POTS[idx % ${#POTS[@]}]}; idx=$(( idx / ${#POTS[@]} ))
  n_agents=${AGENTS[idx]}
  JOB_ARGS=(
    ENV_KWARGS.num_agents="$n_agents"
    ENV_KWARGS.num_pots="$pots"
    ENV_KWARGS.pot_placement=nearest
    ENV_KWARGS.order_size="$order"
    ENV_KWARGS.pot_cook_time="$cook"
    ENV_KWARGS.recipe_size="$recipe"
    ENV_KWARGS.no_collision="$no_coll"
  )
}
