# Overcooked V2 scaling sweep

Exploratory sweep of how IPPO performance in Overcooked V2 changes with team size, number of pots, pot cooking time, and agent collisions, all in one layout.

## Where the data is

| | |
|---|---|
| W&B project | `emiecz-princeton/jaxmarl-overcooked-v2-scaling` |
| Full sweep | W&B group `OV2_scaling_<git hash>_<timestamp>` (SLURM array job `16320373`, launched 2026-10-27). 800 runs expected. |
| Pilot (exclude) | W&B group `OV2_scaling_a17b370_20261027_102906`: 2 runs (`a10_p10_c60`, with and without `_nocoll`), cancelled at ~8% of training. |
| Code | `scaling` branch, commit `a17b370`. The commit is also stored as `GIT_HASH` in each run's config. |
| Raw files (Della) | `/scratch/gpfs/GRIFFITHS/em4240/runs/<group>/`: `logs/` (SLURM output), `wandb/` (offline runs, uploaded with `sync.sh`), `code/` (exact code used). |

Runs can appear in W&B gradually while the sweep is running, since they are uploaded with `sync.sh`.

## Design

Full factorial, 10 × 10 × 4 × 2 = **800 configurations × 5 seeds = 4,000 training runs**. Each W&B run is one configuration and contains all 5 seeds.

| Factor | Values | W&B config key |
|---|---|---|
| Agents | 1–10 | `ENV_KWARGS.num_agents` |
| Pots | 1–10 | `ENV_KWARGS.num_pots` |
| Pot cooking time (env steps) | 5, 10, 30, 60 | `ENV_KWARGS.pot_cook_time` |
| Collisions | on (`False`) / off (`True`) | `ENV_KWARGS.no_collision` |

Run names encode the configuration: `ippo_rnn_overcooked_v2_scaling_kitchen_a{agents}_p{pots}_c{cook}`, with a `_nocoll` suffix when collisions are off.

## Task

- **Layout:** `scaling_kitchen`, a single open 10×5 room surrounded by counters. Coordinates are (x, y), with the origin at the top-left. The base layout is shown below; the map with all 10 pots is under "Pot placement".
  ```
  WWWWWPWWWWWW     W counter   0 onion pile   B plate pile
  W          W     X delivery  P pot          A agent
  0          0
  X    A     X
  B          B
  W          W
  WWWWWWWWWWWW
  ```
- **Recipe:** onion soup only (3 onions). The pot starts cooking automatically once it holds 3 onions. Every soup is the correct recipe, so there are no penalised deliveries.
- **Episodes** are 400 steps. Agents start at random empty cells each episode. Every agent sees the full grid (`agent_view_size: null`).
- **Rewards:**
  - **Sparse:** 20 per delivered soup, shared: every agent gets the team's reward.
  - **Shaped:** placing an onion in a pot +3, picking up a dish +5, a useful plate pickup +3. The shaped reward is scaled down linearly from 1 to 0 over the first 1.5e7 env steps, so the second half of training uses the sparse reward only.
- **Collisions on** (the original Overcooked rules): agents block each other and cannot swap places. **Collisions off:** agents can share cells and pass through each other. Counters and pots still block movement, and pot rules are unchanged: 3 ingredients maximum, one soup at a time.
- **Pot placement** is deterministic, and each pot set is contained in the next larger one. Pots are added in this order: (5,0), (11,5), (0,5), (5,6), (10,0), (1,0), (2,6), (8,6), (3,0), (7,0). Each pot replaces a counter, so P=10 has 14 free counters versus 23 at P=1. With all 10 pots:
  ```
  WPWPWPWPWWPW
  W          W
  0          0
  X          X
  B          B
  P          P
  WWPWWPWWPWWW
  ```

## Training

IPPO with an RNN and parameters shared across all agents (`baselines/IPPO/ippo_rnn_overcooked_v2.py`, config `ippo_rnn_overcooked_v2_scaling.yaml`). The network is a CNN encoder followed by a GRU (128 units). Settings:

- 256 parallel envs × 256 steps per update, 457 updates.
- Final `env_step` = **29,949,952** (≈ 3e7).
- LR 2.5e-4 with 5% warmup and cosine decay, 4 epochs × 64 minibatches.
- γ = 0.99, GAE λ = 0.95, clip 0.2, entropy coefficient 0.01.
- The 5 seeds are `jax.random.split(PRNGKey(0), 5)`, the same for every configuration, and train in parallel in one job.

## Logged data

**History.** The x-axis is `env_step`; `update_step` (1–457) is also logged. Each update writes **one row per seed**, containing `env_step`, `update_step` and only that seed's keys `seed_{i}/<metric>`. When reading history (e.g. `run.scan_history()`), other seeds' keys are missing in each row: group rows by `env_step`, or select one seed's keys at a time.

Each metric is the mean over the update's 256 steps and over all agents and envs:

| Metric | Meaning |
|---|---|
| `returned_episode_returns` | Return of the most recent finished episode, sparse reward only. Equals the team return (every agent gets the same reward). **Soups delivered per episode = value ÷ 20.** It is 0 in update 1, before any episode has finished. |
| `returned_episode_recipe_returns/0_0_0` | Same as above (only one recipe exists). |
| `returned_episode_lengths` | Always 400. |
| `returned_episode` | Fraction of steps on which an episode ended (≈ 1/400). |
| `original_reward` | Sparse reward per agent per step. |
| `shaped_reward` | Shaped reward per agent per step, before scaling down. |
| `anneal_factor` | Weight on the shaped reward: 1 at the start, 0 from 1.5e7 env steps onwards. |
| `combined_reward` | Training reward: `original_reward + anneal_factor × shaped_reward`. |

Losses are not logged.

**Summary** (written when a run finishes; missing if it crashed or timed out):

| Key | Meaning |
|---|---|
| `seed_{i}/final_returned_episode_returns` | Mean `returned_episode_returns` over the last 45 updates (last 10% ≈ 3M env steps). |
| `final_returned_episode_returns_mean`, `_std` | Mean and standard deviation (ddof=0) of the above across the 5 seeds. **Main outcome measure.** |

Fetching the data:

```python
import wandb
runs = wandb.Api().runs(
    "emiecz-princeton/jaxmarl-overcooked-v2-scaling",
    filters={"group": "<full sweep group>"},
)
for run in runs:
    kw = run.config["ENV_KWARGS"]
    kw["num_agents"], kw["num_pots"], kw["pot_cook_time"], kw["no_collision"]
    run.summary.get("final_returned_episode_returns_mean")
```

## Caveats for analysis

- **More agents means more experience.** All configurations get the same number of env steps, but each env step contains one action per agent. A 10-agent configuration therefore collects 10× as many agent-steps as a 1-agent configuration.
- **Single-agent duplicates:** with 1 agent, collisions on and off behave identically. With the same seeds, these pairs should give (near-)identical results, which is a useful check on run-to-run noise.
- **Returns are team totals**, not per agent. Divide by the number of agents for per-capita productivity.
- **Shared-cell observations:** with collisions off, the "other agents" observation channels count how many agents share a cell, so values can be above 1.
- **Incomplete runs:** check `run.state` and the presence of the summary keys. A run that hit the 10-hour SLURM limit has history but no `final_*` summary.
