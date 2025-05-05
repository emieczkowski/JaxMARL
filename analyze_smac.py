import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from scipy.stats import f_oneway

df = pd.read_csv("smac_expt2/combined.csv")
# df2 = pd.read_csv("smac_expt1/wandb_runs_chunk2.csv")
# df = pd.concat([df1, df2], ignore_index=True)

df = df[df["jsd"].notna()]

# Group by trial and compute average JSD
df["trial"] = df["name"].apply(lambda x: "_".join(x.split("_")[:3]))
df[["num_agents", "num_enemies"]] = df["trial"].str.extract(r"(\d+)m_vs_(\d+)m").astype(int)
df["S"] = df["num_agents"]

def max_winrate_jsd(group):
    max_win = group["win_rate"].max()
    return group[group["win_rate"] == max_win]["jsd"].mean()

# Apply that to each trial
jsd_at_max_winrate = df.groupby("trial").apply(max_winrate_jsd).reset_index(name="jsd")

jsd_at_max_winrate[["num_agents", "num_enemies"]] = jsd_at_max_winrate["trial"].str.extract(r"(\d+)m_vs_(\d+)m").astype(int)
jsd_at_max_winrate["S"] = jsd_at_max_winrate["num_agents"]

pearson_corr, _ = pearsonr(jsd_at_max_winrate["S"], jsd_at_max_winrate["jsd"])
print(f"Pearson correlation (S vs. JSD at max win): {pearson_corr:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(jsd_at_max_winrate["S"], jsd_at_max_winrate["jsd"], color='blue')
plt.xlabel("Parallelizability")
plt.ylabel("JSD at Max Win Rate")
plt.title("JSD vs. Parallelizability (Max Win Rate Trials)")
plt.grid(True)
plt.tight_layout()
plt.show()

grouped_by_agents = (
    jsd_at_max_winrate
    .groupby("num_agents")["jsd"]
    .agg(['mean', 'sem'])
    .reset_index()
)

plt.figure(figsize=(8, 5))
plt.bar(
    grouped_by_agents["num_agents"],
    grouped_by_agents["mean"],
    yerr=grouped_by_agents["sem"],
    capsize=5,
    color='skyblue',
    alpha=0.9
)
plt.xlabel("Number of Agents")
plt.ylabel("JSD (at Max Win Rate)")
plt.title("JSD vs. Number of Agents (Highest Win Trials)")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

grouped = df.groupby("trial").agg({
    "jsd": "mean",
    "num_agents": "first",
    "num_enemies": "first",
    "S": "first"
}).reset_index()

# Extract agent/enemy counts and parallelizability
# grouped[["num_agents", "num_enemies"]] = grouped["trial"].str.extract(r"(\d+)m_vs_(\d+)m").astype(int)
grouped["S"] = grouped["num_agents"]

# grouped = grouped[grouped["num_enemies"] == 2]

pearson_corr, _ = pearsonr(grouped["S"], grouped["jsd"])
print(f"Pearson correlation (S vs. JSD): {pearson_corr:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(grouped["S"], grouped["jsd"], color='blue')
plt.xlabel("Parallelizability")
plt.ylabel("Average JSD")
plt.title("JSD vs. Parallelizability")
plt.grid(True)
plt.tight_layout()
plt.show()

grouped_by_agents = (
    df
    .groupby("num_agents")["jsd"]
    .agg(['mean', 'sem'])  
    .reset_index()
)

plt.figure(figsize=(8, 5))
plt.bar(
    grouped_by_agents["num_agents"],
    grouped_by_agents["mean"],
    yerr=grouped_by_agents["sem"],
    capsize=5,
    color='skyblue',
    alpha=0.9
)
plt.xlabel("Number of Agents")
plt.ylabel("Average JSD")
plt.title("Average JSD vs. Number of Agents (with Error Bars)")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

filtered_df = df[df["num_enemies"] == 2]

grouped_jsds = [
    group["jsd"].values
    for _, group in filtered_df.groupby("num_agents")
    if len(group) > 1 
]

f_stat, p_value = f_oneway(*grouped_jsds)
print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_value:.4e}")
