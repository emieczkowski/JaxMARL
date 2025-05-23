import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

df = pd.read_csv("mpe_expt1/wandb_runs_chunk1.csv")

df = df.dropna(subset=["jsd"])
df["num_agents"] = df["num_agents"].astype(int)
df["num_landmarks"] = df["num_landmarks"].astype(int)

# Group by trial and compute average JSD
# df["trial"] = df["name"].apply(lambda x: "_".join(x.split("_")[:3]))
# avg_jsd_per_trial = df.groupby("trial")["jsd"].mean().reset_index()
# df[["num_agents", "num_enemies"]] = df["trial"].str.extract(r"(\d+)m_vs_(\d+)m").astype(int)

df["S"] = np.minimum(df["num_agents"], df["num_landmarks"]) 

grouped = df.groupby(["num_agents", "num_landmarks"]).agg({
    "jsd": "mean",
    "S": "first",  
    "reward": "mean" 
}).reset_index()

print(grouped)

# df["S"] = np.minimum(df["num_agents"], df["num_enemies"]) # treating enemies as bottlenecks
# df["S"] = df["num_agents"] / df["num_enemies"]  # treating enemies as workload
# df["S"] = df["num_agents"] / np.minimum(df["num_agents"]-1, df["num_enemies"]) 

# grouped = df.groupby("trial").agg({
#     "jsd": "mean",
#     "num_agents": "first",
#     "num_enemies": "first",
#     "S": "first"
# }).reset_index()

# print(grouped)

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

grouped = grouped[grouped["num_landmarks"] == 3]
grouped_by_agents = grouped.groupby("num_agents")["reward"].mean().reset_index()

plt.figure(figsize=(8, 5))
plt.bar(grouped_by_agents["num_agents"], grouped_by_agents["reward"], color='green')
plt.xlabel("Number of Agents")
plt.ylabel("Average Reward")
plt.title("Average Reward vs. Number of Agents")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
