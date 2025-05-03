import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from scipy.stats import f_oneway

df = pd.read_csv("smac_expt1/wandb_runs_chunk1.csv")
# df2 = pd.read_csv("smac_expt1/wandb_runs_chunk2.csv")
# df = pd.concat([df1, df2], ignore_index=True)

df = df[df["jsd"].notna()]

# Group by trial and compute average JSD
df["trial"] = df["name"].apply(lambda x: "_".join(x.split("_")[:3]))
# avg_jsd_per_trial = df.groupby("trial")["jsd"].mean().reset_index()
df[["num_agents", "num_enemies"]] = df["trial"].str.extract(r"(\d+)m_vs_(\d+)m").astype(int)
df["S"] = df["num_agents"]

grouped = df.groupby("trial").agg({
    "jsd": "mean",
    "num_agents": "first",
    "num_enemies": "first",
    "S": "first"
}).reset_index()

# Extract agent/enemy counts and parallelizability
grouped[["num_agents", "num_enemies"]] = grouped["trial"].str.extract(r"(\d+)m_vs_(\d+)m").astype(int)
grouped["S"] = grouped["num_agents"]

grouped = grouped[grouped["num_enemies"] == 3]

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

grouped_by_agents = grouped.groupby("num_agents")["jsd"].mean().reset_index()

plt.figure(figsize=(8, 5))
plt.bar(grouped_by_agents["num_agents"], grouped_by_agents["jsd"], color='skyblue')
plt.xlabel("Number of Agents")
plt.ylabel("Average JSD")
plt.title("Average JSD vs. Number of Agents")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

filtered_df = df[df["num_enemies"] == 3]

grouped_jsds = [
    group["jsd"].values
    for _, group in filtered_df.groupby("num_agents")
    if len(group) > 1 
]

f_stat, p_value = f_oneway(*grouped_jsds)
print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_value:.4e}")
