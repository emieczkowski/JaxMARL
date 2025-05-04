#!/bin/bash

# Pull latest changes
echo "Pulling latest changes from Git..."
git -C /scratch/gpfs/$USER/JaxMARL pull

# Run seeds 0 through 9
for SEED in {0..9}; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)_s$SEED
    SNAPSHOT_DIR="/scratch/gpfs/$USER/JaxMARL_snapshot_$TIMESTAMP"

    echo "Creating snapshot at $SNAPSHOT_DIR for seed $SEED..."
    rsync -a --delete /scratch/gpfs/$USER/JaxMARL/ "$SNAPSHOT_DIR/"

    echo "Submitting job for seed $SEED..."
    sbatch --export=ALL,SNAPSHOT_DIR="$SNAPSHOT_DIR",SEED=$SEED run_apptainer_mounted.slurm

    sleep 1  # avoid same timestamp collisions
done
