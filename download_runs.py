import wandb
import pandas as pd

def main():
    wandb.init(
        entity="ruaridhmw",
        project="jaxmarl-mpe-exp1",
        mode="online",
    )

    api = wandb.Api(timeout=1000)
    entity = "ruaridhmw"
    project = "jaxmarl-mpe-exp1"
    runs = api.runs(f"{entity}/{project}")
    
    chunk_size = 1000  # Number of runs per CSV chunk
    chunk_count = 1
    rows = []
    
    for i, run in enumerate(runs, start=1):
        env_kwargs = run.config.get("ENV_KWARGS", {})
        rows.append({
            "name": run.name,
            "num_agents": env_kwargs.get("num_agents"),
            "num_landmarks": env_kwargs.get("num_landmarks"),
            "reward": run.summary.get("rollout_reward"),
            "jsd": run.summary.get("Multi-Agent JSD"),
            "laststep_jsd": run.summary.get("Last-Step Multi-Agent JSD")
        })
    
    if rows:
        df = pd.DataFrame(rows)
        file_name = f"mpe_expt1/wandb_runs_chunk{chunk_count}.csv"
        df.to_csv(file_name, index=False)
        print(f"Saved final chunk with {len(df)} runs to {file_name}")

if __name__ == "__main__":
    main()
