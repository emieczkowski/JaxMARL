#!/bin/bash
#SBATCH --job-name=your_job_name      # Your job name
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=20
#SBATCH --time=0:40:00               # Set a time limit (e.g., 15 minutes)

module purge

# Step 1: Define the path to the config file (inside the container)
CONFIG_FILE="/home/workdir/JaxMARL/config.txt"

# Step 2: Extract the Python script path from the config file
PYTHON_SCRIPT_PATH=$(apptainer exec --nv --bind "$SNAPSHOT_DIR:/home/workdir/JaxMARL" \
  /scratch/gpfs/em4240/jaxmarl_latest.sif bash -c "grep 'PYTHON_SCRIPT_PATH' $CONFIG_FILE | cut -d '=' -f 2")

# Step 3: Check if the script path was correctly loaded
if [ -z "$PYTHON_SCRIPT_PATH" ]; then
  echo "Error: Python script path not found in $CONFIG_FILE"
  exit 1
fi

# Step 4: Execute the Python script using Apptainer, pointing to the snapshot directory
# apptainer exec --nv --bind "$SNAPSHOT_DIR:/home/workdir/JaxMARL" \
#   --bind "$SNAPSHOT_DIR:/home/local_workdir/JaxMARL" \
#   --env PYTHONPATH=/home/local_workdir/JaxMARL:/home/workdir/JaxMARL \
#   /scratch/gpfs/em4240/jaxmarl_latest.sif /usr/bin/python3 "/home/workdir/JaxMARL/$PYTHON_SCRIPT_PATH"
# Append --config.SEED=$SEED to override seed dynamically
apptainer exec --nv --bind "$SNAPSHOT_DIR:/home/workdir/JaxMARL" \
  --bind "$SNAPSHOT_DIR:/home/local_workdir/JaxMARL" \
  --env PYTHONPATH=/home/local_workdir/JaxMARL:/home/workdir/JaxMARL \
  /scratch/gpfs/em4240/jaxmarl_latest.sif /usr/bin/python3 "/home/workdir/JaxMARL/$PYTHON_SCRIPT_PATH" --config.SEED=$SEED

# Ensure cleanup runs on script exit
trap 'rm -rf "$SNAPSHOT_DIR"' EXIT

# Step 5: Clean up - remove the snapshot directory after the job is complete
echo "Cleaning up the snapshot directory..."
rm -rf "$SNAPSHOT_DIR"
