import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import pearsonr

df_smac = pd.read_csv("smac_expt1/wandb_runs_chunk1.csv")
df_smac = df_smac[df_smac["jsd"].notna()]
df_smac["trial"] = df_smac["name"].apply(lambda x: "_".join(x.split("_")[:3]))
df_smac[["num_agents", "num_enemies"]] = df_smac["trial"].str.extract(r"(\d+)m_vs_(\d+)m").astype(int)
df_smac["S"] = df_smac["num_agents"]

grouped_smac = df_smac.groupby("trial").agg({
    "jsd": "mean",
    "num_agents": "first",
    "num_enemies": "first",
    "S": "first"
}).reset_index()
grouped_smac["env"] = "SMAC"

df_mpe = pd.read_csv("mpe_expt1/wandb_runs_chunk1.csv")
df_mpe = df_mpe.dropna(subset=["jsd"])
df_mpe["num_agents"] = df_mpe["num_agents"].astype(int)
df_mpe["num_landmarks"] = df_mpe["num_landmarks"].astype(int)
# df_mpe["S"] = np.minimum(df_mpe["num_agents"], df_mpe["num_landmarks"])
df_mpe["S"] = 1

grouped_mpe = df_mpe.groupby(["num_agents", "num_landmarks"]).agg({
    "jsd": "mean",
    "S": "first"
}).reset_index()
grouped_mpe["env"] = "MPE"

combined = pd.concat([
    grouped_smac[["S", "jsd", "env"]],
    grouped_mpe[["S", "jsd", "env"]]
], ignore_index=True)

plt.figure(figsize=(8, 5))
for env, color in zip(["SMAC", "MPE"], ["blue", "orange"]):
    subset = combined[combined["env"] == env]
    plt.scatter(subset["S"], subset["jsd"], label=env, alpha=0.7, color=color)

plt.xlabel("Parallelizability (S)")
plt.ylabel("Average JSD")
plt.title("JSD vs. Parallelizability across Environments")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

median_jsd = 0.5
combined["jsd_high"] = (combined["jsd"] > median_jsd).astype(int)

X = combined[["S"]].values
y = combined["jsd_high"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
log_reg = LogisticRegression(class_weight="balanced", max_iter=500)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print("LogReg Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

corr, p_value = pearsonr(combined["S"], combined["jsd"])
print(f"Pearson correlation (JSD vs. S): {corr:.4f}, p = {p_value:.4e}")