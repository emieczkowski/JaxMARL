import wandb
import pandas as pd

def main():
    wandb.init(
        entity="ruaridhmw",
        project="jaxmarl-smac-expt-9",
        mode="online",
    )

    api = wandb.Api(timeout=1000)
    entity = "ruaridhmw"
    project = "jaxmarl-smac-expt-9"
    runs = api.runs(f"{entity}/{project}")
    
    chunk_size = 1000  # Number of runs per CSV chunk
    chunk_count = 1
    rows = []
    
    for i, run in enumerate(runs, start=1):
        env_kwargs = run.config.get("ENV_KWARGS", {})
        rows.append({
            "name": run.name,
            "win_rate": run.summary.get("win_rate"),
            "jsd": run.summary.get("avg_generalized_jsd"),
        })
    
    if rows:
        df = pd.DataFrame(rows)
        file_name = f"smac_expt2/3enemies.csv"
        df.to_csv(file_name, index=False)
        print(f"Saved final chunk with {len(df)} runs to {file_name}")

if __name__ == "__main__":
    main()
