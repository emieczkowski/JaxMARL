#!/bin/bash

# Step 1: Pull the latest changes from your Git repository (login node)
echo "Pulling latest changes from Git..."
git -C /scratch/gpfs/$USER/JaxMARL pull

# Step 2: Create a timestamped snapshot directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)  # Format: YYYYMMDD_HHMMSS
SNAPSHOT_DIR="/scratch/gpfs/$USER/JaxMARL_snapshot_$TIMESTAMP"

# Step 3: Copy the repository to the timestamped snapshot directory
echo "Creating a snapshot of the Git repository at $SNAPSHOT_DIR..."
rsync -a --delete /scratch/gpfs/$USER/JaxMARL/ "$SNAPSHOT_DIR/"

# Step 4: Submit the training job to the GPU node, passing the snapshot directory as an argument
echo "Submitting training job to GPU node..."
TRAIN_JOB_ID=$(sbatch --parsable --export=SNAPSHOT_DIR="$SNAPSHOT_DIR" run_apptainer_mounted.slurm)