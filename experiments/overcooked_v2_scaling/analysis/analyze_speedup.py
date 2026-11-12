"""Compare the order sweep's measured team performance with the structural predictions.

Inputs (in results/order/): runs.csv and seeds.csv from fetch_order_results.py,
predictions.csv from amdahl_predictions.py (T1, W, C per condition). Only selected runs
(finished training, one per configuration) are used.

Seeds that never learned the task (completion rate < LEARNED, almost always ~0: a
training failure, mostly at 1 agent) are excluded and counted separately.
Measured completion time T(n) of a configuration is final_length (mean over the last
10% of training of the episode length, censored at max_steps) averaged over its learned
seeds.

Structural bound: n agents cannot finish faster than the labor split evenly (W/n) or
the critical path (C), so T(n) >= max(W/n, C). Amdahl prediction: T(n) = C + (T1 - C)/n.

Speedup is measured against the optimal solo time T1 from the structure,
S*(n) = T1 / T(n), not against the trained solo agent: a solo agent slower than optimal
would otherwise inflate every team's speedup. S* is the completion time rescaled by a
constant per condition, so it is bounded by T1 / max(W/n, C) <= T1 / C, and S*(1) is the
solo agent's efficiency. The actual speedup over the trained solo agent, T(1) / T(n), is
plotted separately: it can exceed the bound when the solo agent is slower than optimal.

Outputs (in results/order/figures/). The top-level figures show sweep_order.sh
(scaling_kitchen, as many pots as soups); pots/ shows sweep_order_pots.sh (scaling_stations,
pots crossed with order size):
  speedup_optimal_r{1,3}.png  S*(n) = T1*/T(n) vs agents (speedup over an optimal solo
                          agent); rows = order size K, columns = cook time; collisions off /
                          on against their own predictions (gray where collisions don't
                          change them)
  speedup_actual_r{1,3}.png   the same for the actual speedup T(1)/T(n), over the trained
                          solo agent. All four speedup figures share one y-axis
  time_r{1,3}.png         the same for completion time (steps)
  pots/{speedup_optimal,speedup_actual,time}_r1_c{5,20,60}.png
                          the same three kinds per recipe size and cook time, with
                          rows = order size K and columns = number of pots P
  ceiling.png             measured peak S* vs predicted ceiling T1/C, and the team size
                          reaching 90% of the peak vs the predicted knee
  summary.csv             one row per condition (layout, placement, P, K, r, T, collisions)

Usage: python analyze_speedup.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "order")
FIGURES = os.path.join(RESULTS, "figures")
CONDITION = ["layout", "pot_placement", "num_pots", "order_size", "recipe_size", "pot_cook_time", "no_collision"]
LEARNED = 0.5  # seeds with a lower completion rate never learned the task: excluded
COMPLETE = 0.99  # completion rate (learned seeds) below which a point is drawn hollow (time is censored)

# Validated two-slot categorical palette and chart chrome (light mode)
SERIES = {True: ("#2a78d6", "collisions off"), False: ("#eb6834", "collisions on")}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "legend.frameon": False, "text.color": INK,
})


def load():
    runs = pd.read_csv(os.path.join(RESULTS, "runs.csv"))
    seeds = pd.read_csv(os.path.join(RESULTS, "seeds.csv"))
    pred = pd.read_csv(os.path.join(RESULTS, "predictions.csv"))
    runs = runs[runs["selected"]]
    for key, default in (("layout", "scaling_kitchen"), ("pot_placement", "spread")):
        if key not in runs:  # runs.csv from before the key was a factor
            runs = runs.assign(**{key: default})
    seeds = seeds[seeds["run_id"].isin(runs["run_id"])]
    return runs, seeds, pred


def predicted(T1, W, C, n):
    """Predicted completion times and speedups over the optimal solo time T1."""
    t_amdahl = C + (T1 - C) / n
    t_bound = np.maximum(W / n, C)
    return {"T_amdahl": t_amdahl, "T_bound": t_bound, "S_amdahl": T1 / t_amdahl, "S_bound": T1 / t_bound}


def measure(runs, seeds, pred):
    """One row per configuration: measured time and speedups, and the predictions."""
    seeds = seeds.assign(learned=seeds["final_completed"] >= LEARNED)
    learned = seeds[seeds["learned"]]
    per_run = learned.groupby("run_id").agg(
        T_emp=("final_length", "mean"), completed=("final_completed", "mean"), seeds_learned=("seed", "size"))
    per_run["seeds_total"] = seeds.groupby("run_id").size()
    df = runs[CONDITION + ["num_agents", "run_id"]].merge(per_run, left_on="run_id", right_index=True)
    t1 = df[df["num_agents"] == 1].set_index(CONDITION)["T_emp"].rename("T_emp_1")
    df = df.join(t1, on=CONDITION).merge(pred, on=CONDITION)

    seed_t = learned.merge(df[["run_id", "T1", "T_emp_1"]], on="run_id")
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    df["S_opt"] = df["T1"] / df["T_emp"]
    df["S_opt_se"] = df["run_id"].map((seed_t["T1"] / seed_t["final_length"]).groupby(seed_t["run_id"]).agg(se))
    df["S_solo"] = df["T_emp_1"] / df["T_emp"]
    df["S_solo_se"] = df["run_id"].map((seed_t["T_emp_1"] / seed_t["final_length"]).groupby(seed_t["run_id"]).agg(se))
    df["T_emp_se"] = df["run_id"].map(seed_t.groupby("run_id")["final_length"].agg(se))
    for key, value in predicted(df["T1"], df["W"], df["C"], df["num_agents"]).items():
        df[key] = value
    return df.sort_values(CONDITION + ["num_agents"])


def grid(df, kind, rows, cols, title, path, ylim=None):
    """Small multiples of one figure. rows / cols: (column, label function) of the factor
    varied down / across; df holds the panels' configurations (other factors fixed)."""
    (row_key, row_label), (col_key, col_label) = rows, cols
    row_values, col_values = sorted(df[row_key].unique()), sorted(df[col_key].unique())
    fig, axes = plt.subplots(len(row_values), len(col_values), figsize=(2.5 * len(col_values) + 2.5, 2.6 * len(row_values)),
                             sharex=True, sharey=kind != "time", squeeze=False)
    n_line = np.linspace(1, 10, 200)
    key = "T" if kind == "time" else "S"
    column = {"optimal": "S_opt", "actual": "S_solo", "time": "T_emp"}[kind]
    for i, rv in enumerate(row_values):
        for j, cv in enumerate(col_values):
            ax = axes[i, j]
            panel = df[(df[row_key] == rv) & (df[col_key] == cv)]
            # one gray prediction when collisions don't change it
            shared = panel[["T1", "W", "C"]].drop_duplicates().shape[0] == 1
            for no_coll, (color, _) in SERIES.items():
                d = panel[panel.no_collision == no_coll]
                if d.empty:
                    continue
                p = d.iloc[0]
                curves = predicted(p["T1"], p["W"], p["C"], n_line)
                pred_color = MUTED if shared else color
                ax.plot(n_line, curves[f"{key}_amdahl"], color=pred_color, lw=1.2, ls="--")
                ax.plot(n_line, curves[f"{key}_bound"], color=pred_color, lw=1.2, ls=":")
                y, err = d[column], d[f"{column}_se"]
                ax.plot(d["num_agents"], y, color=color, lw=2, zorder=3)
                ax.errorbar(d["num_agents"], y, yerr=err, fmt="none", ecolor=color, elinewidth=1, zorder=3)
                done = d["completed"] >= COMPLETE
                ax.scatter(d["num_agents"][done], y[done], s=28, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=4)
                ax.scatter(d["num_agents"][~done], y[~done], s=28, facecolor=SURFACE, edgecolor=color, linewidth=1.5, zorder=4)
            if kind != "time":
                ax.axhline(1, color=AXIS, lw=0.8, zorder=1)
                if ylim:
                    ax.set_ylim(*ylim)
            else:
                ax.set_ylim(bottom=0)
            ax.set_xticks(range(1, 11))
            if i == 0:
                ax.set_title(col_label(cv))
            if j == 0:
                ax.set_ylabel(f"{row_label(rv)}\n" + {"optimal": "speedup T1*/T(n)", "actual": "speedup T(1)/T(n)",
                                                      "time": "completion time (steps)"}[kind])
            if i == len(row_values) - 1:
                ax.set_xlabel("agents n")
    bound = "max(W/n, C)" if kind == "time" else "T1*/max(W/n, C)"
    handles = [plt.Line2D([], [], color=c, lw=2, marker="o", label=f"measured, {l}") for c, l in SERIES.values()]
    handles += [plt.Line2D([], [], color=MUTED, lw=1.2, ls="--", label="Amdahl prediction"),
                plt.Line2D([], [], color=MUTED, lw=1.2, ls=":", label=f"structural bound {bound}"),
                plt.Line2D([], [], color=MUTED, lw=0, marker="o", markerfacecolor=SURFACE,
                           label=f"hollow: < {COMPLETE:.0%} of episodes completed (time is a lower limit)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    what = {"optimal": "Speedup over an optimal solo agent, T1*/T(n),",
            "actual": "Actual speedup over the trained solo agent, T(1)/T(n),", "time": "Completion time"}[kind]
    fig.suptitle(f"{what} vs team size, {title}", y=1.035, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_name(kind):
    return "time" if kind == "time" else f"speedup_{kind}"


def summarize(df):
    rows = []
    for key, d in df.groupby(CONDITION):
        d = d.sort_values("num_agents")
        best = d.loc[d["S_opt"].idxmax()]
        p = d.iloc[0]
        rows.append({
            **dict(zip(CONDITION, key)),
            "agents_available": ",".join(map(str, d["num_agents"])),
            "T1_pred": p["T1"], "W_pred": p["W"], "C_pred": p["C"],
            "S_max_pred": p["T1"] / p["C"], "knee_pred": p["T1"] / p["C"],
            "T_emp_1": p["T_emp_1"], "solo_efficiency": p["T1"] / p["T_emp_1"],
            "S_opt_peak": best["S_opt"], "n_peak": int(best["num_agents"]),
            "n_90": int(d.loc[d["S_opt"] >= 0.9 * best["S_opt"], "num_agents"].min()),
            "T_best_over_C": best["T_emp"] / p["C"],
            "S_solo_peak": d["S_solo"].max(),
            "min_completion_rate": d["completed"].min(),
            "seeds_failed_1agent": int(p["seeds_total"] - p["seeds_learned"]) if p["num_agents"] == 1 else None,
            "seeds_failed_total": int((d["seeds_total"] - d["seeds_learned"]).sum()),
        })
    return pd.DataFrame(rows)


def ceiling_plot(summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    markers = {1: "o", 3: "s"}
    for no_coll, (color, label) in SERIES.items():
        for r, m in markers.items():
            s = summary[(summary.no_collision == no_coll) & (summary.recipe_size == r)]
            axes[0].scatter(s["S_max_pred"], s["S_opt_peak"], color=color, marker=m, s=40,
                            edgecolor=SURFACE, linewidth=1, label=f"{label}, recipe {r}")
            axes[1].scatter(np.minimum(s["knee_pred"], 10.5), s["n_90"], color=color, marker=m, s=40,
                            edgecolor=SURFACE, linewidth=1)
    lim = max(summary["S_max_pred"].max(), summary["S_opt_peak"].max()) * 1.05
    axes[0].plot([1, lim], [1, lim], color=AXIS, lw=1, zorder=0)
    axes[0].set(xlabel="predicted ceiling T1*/C", ylabel="measured peak speedup T1*/T(n)", xscale="log", yscale="log",
                title="Speedup ceiling: predicted vs measured")
    axes[1].plot([1, 10], [1, 10], color=AXIS, lw=1, zorder=0)
    axes[1].set(xlabel="predicted knee T1*/C (capped at 10.5)", ylabel="measured team size reaching 90% of peak",
                title="Saturation team size: predicted vs measured", yticks=range(1, 11))
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    path = os.path.join(FIGURES, "ceiling.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    os.makedirs(FIGURES, exist_ok=True)
    runs, seeds, pred = load()
    df = measure(runs, seeds, pred)
    df.to_csv(os.path.join(RESULTS, "speedup.csv"), index=False)
    # one speedup scale for every panel of both figures
    top = max(df["S_opt"].max(), df["S_solo"].max(), df["S_bound"].max()) * 1.05
    onions = lambda r: f"recipe = {r} onion{'s' if r > 1 else ''}"
    soups = ("order_size", lambda K: f"K = {K} soup{'s' if K > 1 else ''}")
    kinds = {"optimal": (0, top), "actual": (0, top), "time": None}
    # sweep_order.sh (scaling_kitchen, P = K, spread pots): rows = order size, columns = cook time
    diagonal = df[(df["layout"] == "scaling_kitchen") & (df["pot_placement"] == "spread")
                  & (df["num_pots"] == df["order_size"])]
    for r in sorted(diagonal["recipe_size"].unique()):
        for kind, ylim in kinds.items():
            print(grid(diagonal[diagonal.recipe_size == r], kind, soups, ("pot_cook_time", lambda T: f"cook time {T}"),
                       f"{onions(r)} (order of K soups, K pots)", os.path.join(FIGURES, f"{figure_name(kind)}_r{r}.png"), ylim))
    # sweep_order_pots.sh (scaling_stations, pots crossed with order size): rows = order
    # size, columns = pots
    for (r, T), d in df[df["layout"] == "scaling_stations"].groupby(["recipe_size", "pot_cook_time"]):
        for kind, ylim in kinds.items():
            print(grid(d, kind, soups, ("num_pots", lambda P: f"P = {P} pot{'s' if P > 1 else ''}"),
                       f"{onions(r)}, cook time {T} (workstation kitchen, order of K soups, P pots)",
                       os.path.join(FIGURES, "pots", f"{figure_name(kind)}_r{r}_c{T}.png"), ylim))
    summary = summarize(df)
    summary.to_csv(os.path.join(FIGURES, "summary.csv"), index=False)
    print(ceiling_plot(summary))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 2):
        print(summary.drop(columns=["agents_available"]).to_string(index=False))


if __name__ == "__main__":
    main()
