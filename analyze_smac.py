import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

df = pd.read_csv("smac_expt1/wandb_runs_chunk1.csv")

# Group by trial and compute average JSD
df["trial"] = df["name"].apply(lambda x: "_".join(x.split("_")[:3]))
avg_jsd_per_trial = df.groupby("trial")["jsd"].mean().reset_index()
df[["num_agents", "num_enemies"]] = df["trial"].str.extract(r"(\d+)m_vs_(\d+)m").astype(int)

# print(avg_jsd_per_trial)

# df["S"] = np.minimum(df["num_agents"], df["num_enemies"]) # treating enemies as bottlenecks
# df["S"] = df["num_agents"] / df["num_enemies"]  # treating enemies as workload
# df["S"] = df["num_agents"] / np.minimum(df["num_agents"]-1, df["num_enemies"]) 
df["S"] = df["num_agents"]

grouped = df.groupby("trial").agg({
    "jsd": "mean",
    "num_agents": "first",
    "num_enemies": "first",
    "S": "first"
}).reset_index()

print(grouped)

pearson_corr, _ = pearsonr(grouped["S"], grouped["jsd"])
print(f"Pearson correlation (S vs. JSD): {pearson_corr:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(grouped["S"], grouped["jsd"], color='blue')
plt.xlabel("Parallelizability")
plt.ylabel("Average JSD")
plt.title("JSD vs. Parallelizability")
plt.grid(True)
plt.tight_layout()
# plt.show()

# grouped_by_agents = grouped.groupby("num_agents")["jsd"].mean().reset_index()

# plt.figure(figsize=(8, 5))
# plt.bar(grouped_by_agents["num_agents"], grouped_by_agents["jsd"], color='skyblue')
# plt.xlabel("Number of Agents")
# plt.ylabel("Average JSD")
# plt.title("Average JSD vs. Number of Agents")
# plt.grid(axis='y', alpha=0.3)
# plt.tight_layout()
# plt.show()

# Prepare data for grouped bar chart
grouped_bar = grouped.groupby(["num_agents", "num_enemies"])["jsd"].mean().reset_index()

# Pivot to have enemy counts as separate columns
pivoted = grouped_bar.pivot(index="num_agents", columns="num_enemies", values="jsd")

# Plot
pivoted.plot(kind="bar", figsize=(10, 6))
plt.xlabel("Number of Agents")
plt.ylabel("Average JSD")
plt.title("Average JSD by Number of Agents and Enemies")
plt.legend(title="Number of Enemies")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
