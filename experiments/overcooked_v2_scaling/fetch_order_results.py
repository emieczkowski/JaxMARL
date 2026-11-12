"""Download the order sweep's results from W&B into CSV files for analysis.

Usage (from anywhere, with the jaxmarl env; needs `wandb login`):
    python fetch_order_results.py                      # all groups in the project
    python fetch_order_results.py --group OV2_order_6712a70_20261029_133840
    python fetch_order_results.py --no-history         # summaries only (fast)

A configuration can have several runs, e.g. a failed run in the original sweep and its
rerun in a later group. For each configuration the script selects one run: the most
recent run that finished training (has its final_* summary). It then prints the
configurations that have no finished run, with the submit.sh command that reruns them.

Writes to --out (default: results/order next to this script):

runs.csv     One row per W&B run (one configuration, 5 seeds trained together), including
             failed and superseded runs. `selected` marks the run used for its
             configuration; analyses should filter on it.
             Config: layout, num_agents, num_pots, order_size, pot_cook_time,
             recipe_size, no_collision, max_steps (censoring limit), pot_placement
             ("spread" for runs from before it existed). sweep_order.sh: scaling_kitchen,
             num_pots = order_size, spread pots; sweep_order_pots.sh: scaling_stations,
             pots and order size crossed, balanced pots. Run info: run_id, name, group,
             state, created_at, git_hash, runtime_h, sweep_index (job index in
             sweep_order_pots.sh), has_final (False if the run did not finish training: its
             final_* columns are empty). Outcomes (mean/std across seeds):
               final_length_*     completion time: mean returned_episode_lengths over the
                                  last 10% of updates; episodes that did not complete the
                                  order count as max_steps (censored)
               final_completed_*  completion rate over the same updates
               final_return_*     sparse return (20 per soup)
               rollout_*_mean     one recorded episode per seed with the final policy
seeds.csv    One row per selected run x seed: the config columns plus run_id, seed,
             final_length, final_completed, final_return, rollout_length, rollout_return,
             rollout_soups.
history.csv.gz  One row per selected run x seed x update: run_id, seed, update_step, env_step and
             every logged metric (returned_episode_lengths, returned_episode_order_completed,
             returned_episode_returns, original_reward, shaped_reward, anneal_factor, ...),
             each the mean over that update's steps, envs and agents. For learning curves
             and convergence checks.
metadata.json  Project, groups, fetch time, run counts, configurations still missing.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import wandb

PROJECT = "emiecz-princeton/jaxmarl-overcooked-v2-order"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP_FILE = "sweep_order_pots.sh"  # the crossed grid (scaling_stations); sweep_order.sh runs are outside it
CONFIG_KEYS = ["layout", "num_agents", "order_size", "num_pots", "pot_placement", "pot_cook_time", "recipe_size", "no_collision", "max_steps"]
DEFAULTS = {"pot_placement": "spread"}  # for runs from before the key existed
# Swept keys: together they identify a configuration
SWEPT_KEYS = ["layout", "num_agents", "num_pots", "pot_placement", "order_size", "pot_cook_time", "recipe_size", "no_collision"]
# summary key suffix -> column name
FINAL_KEYS = {
    "final_returned_episode_lengths": "final_length",
    "final_returned_episode_order_completed": "final_completed",
    "final_returned_episode_returns": "final_return",
    "rollout_length": "rollout_length",
    "rollout_return": "rollout_return",
    "rollout_soups": "rollout_soups",
}
SEED_KEY = re.compile(r"^seed_(\d+)/(.+)$")


def sweep_indices():
    """{configuration (SWEPT_KEYS values): job index} for every job in the sweep file."""
    listing = subprocess.run(
        ["bash", "-c", f'source {SWEEP_FILE}; for ((i = 0; i < NUM_JOBS; i++)); do decode_job $i; echo "$i ${{JOB_ARGS[*]}}"; done'],
        cwd=SCRIPT_DIR, capture_output=True, text=True, check=True,
    ).stdout
    indices = {}
    for line in listing.splitlines():
        index, *args = line.split()
        kwargs = dict(arg.removeprefix("ENV_KWARGS.").split("=") for arg in args)
        indices[config_key({**DEFAULTS, **kwargs})] = int(index)
    return indices


def array_spec(indices):
    """Sorted job indices as a SLURM array spec, e.g. [1, 2, 3, 7] -> "1-3,7"."""
    ranges = []
    for index in indices:
        if ranges and index == ranges[-1][1] + 1:
            ranges[-1][1] = index
        else:
            ranges.append([index, index])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def config_key(row):
    """Configuration as a hashable tuple, from a runs.csv row or a sweep file's overrides."""

    def value(k):
        v = row[k]
        if k == "no_collision":
            return v in (True, "True")
        return str(v) if k in ("layout", "pot_placement") else int(v)

    return tuple(value(k) for k in SWEPT_KEYS)


def run_rows(run):
    """(runs.csv row, seeds.csv rows) for one run."""
    summary = run.summary._json_dict
    env_kwargs = run.config["ENV_KWARGS"]
    info = {
        "run_id": run.id,
        "name": run.name,
        "group": run.group,
        "state": run.state,
        "created_at": run.created_at,
        "git_hash": run.config.get("GIT_HASH"),
        "runtime_h": summary.get("_runtime", np.nan) / 3600,
        **{k: env_kwargs.get(k, DEFAULTS.get(k)) for k in CONFIG_KEYS},
    }
    seeds = []
    for seed in range(run.config["NUM_SEEDS"]):
        row = {"run_id": run.id, **{k: env_kwargs.get(k, DEFAULTS.get(k)) for k in CONFIG_KEYS}, "seed": seed}
        for key, col in FINAL_KEYS.items():
            row[col] = summary.get(f"seed_{seed}/{key}", np.nan)
        seeds.append(row)

    seed_df = pd.DataFrame(seeds)
    info["has_final"] = bool(seed_df["final_length"].notna().all())
    for col in ("final_length", "final_completed", "final_return"):
        info[f"{col}_mean"] = seed_df[col].mean()
        info[f"{col}_std"] = seed_df[col].std(ddof=0)
    for col in ("rollout_length", "rollout_return", "rollout_soups"):
        info[f"{col}_mean"] = seed_df[col].mean()
    return info, seeds


def run_history(run):
    """Learning curves of one run, one row per seed x update."""
    by_seed = {}
    for row in run.scan_history(page_size=5000):
        if row.get("update_step") is None:  # the final row holding the rollout GIFs
            continue
        for key, value in row.items():
            match = SEED_KEY.match(key)
            if match is None or value is None:
                continue
            seed, metric = int(match.group(1)), match.group(2)
            entry = by_seed.setdefault((seed, row["update_step"]), {"env_step": row["env_step"]})
            entry[metric] = value
    rows = [
        {"run_id": run.id, "seed": seed, "update_step": update_step, **values}
        for (seed, update_step), values in by_seed.items()
    ]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", help="only runs in this W&B group (default: all)")
    parser.add_argument("--out", default=os.path.join(SCRIPT_DIR, "results", "order"))
    parser.add_argument("--no-history", action="store_true", help="skip learning curves")
    parser.add_argument("--workers", type=int, default=8, help="parallel history downloads")
    args = parser.parse_args()

    api = wandb.Api(timeout=120)
    filters = {"group": args.group} if args.group else {}
    runs = list(api.runs(PROJECT, filters=filters, per_page=500))
    print(f"{len(runs)} runs in {PROJECT}" + (f" (group {args.group})" if args.group else ""))

    infos, seeds = [], []
    for run in runs:
        info, seed_rows = run_rows(run)
        infos.append(info)
        seeds.extend(seed_rows)
    runs_df = pd.DataFrame(infos)

    # One run per configuration: the most recent one that finished training
    indices = sweep_indices()
    runs_df["sweep_index"] = [indices.get(config_key(row)) for _, row in runs_df.iterrows()]
    runs_df = runs_df.sort_values(["has_final", "created_at"], ascending=False)
    runs_df["selected"] = runs_df["has_final"] & ~runs_df.duplicated(SWEPT_KEYS)
    runs_df = runs_df.sort_values(["layout", "pot_placement", "num_pots", "order_size", "pot_cook_time", "recipe_size", "no_collision", "num_agents", "created_at"])
    selected_ids = set(runs_df.loc[runs_df["selected"], "run_id"])
    seeds_df = pd.DataFrame(seeds)
    seeds_df = seeds_df[seeds_df["run_id"].isin(selected_ids)]

    os.makedirs(args.out, exist_ok=True)
    runs_df.to_csv(os.path.join(args.out, "runs.csv"), index=False)
    seeds_df.to_csv(os.path.join(args.out, "seeds.csv"), index=False)

    done = {config_key(row) for _, row in runs_df[runs_df["selected"]].iterrows()}
    missing = sorted(index for key, index in indices.items() if key not in done)
    metadata = {
        "project": PROJECT,
        "group_filter": args.group,
        "fetched_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "num_runs": len(runs_df),
        "runs_by_group": runs_df["group"].value_counts().to_dict(),
        "runs_by_state": runs_df["state"].value_counts().to_dict(),
        "configs_in_sweep": len(indices),
        "configs_with_finished_run": len(done),
        "missing_sweep_indices": missing,
    }
    print(json.dumps({k: v for k, v in metadata.items() if k != "missing_sweep_indices"}, indent=2))
    if missing:
        print(f"\n{len(missing)} configurations have no finished run. Rerun them on Della with:")
        print(f"  SWEEP={SWEEP_FILE} ./submit.sh {array_spec(missing)}")

    if not args.no_history:
        selected_runs = [run for run in runs if run.id in selected_ids]
        with ThreadPoolExecutor(args.workers) as pool:
            histories = []
            for i, history in enumerate(pool.map(run_history, selected_runs), 1):
                histories.append(history)
                if i % 25 == 0 or i == len(selected_runs):
                    print(f"history: {i}/{len(selected_runs)} runs")
        history_df = pd.concat(histories, ignore_index=True)
        history_df.to_csv(os.path.join(args.out, "history.csv.gz"), index=False)
        metadata["history_rows"] = len(history_df)

    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
