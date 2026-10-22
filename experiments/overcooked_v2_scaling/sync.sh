#!/bin/bash
# Upload the offline W&B runs of a sweep (run on the login node, which has internet).
# Safe to re-run: runs already synced are skipped.
#
# Usage: ./sync.sh RUN_DIR
set -euo pipefail

RUN_DIR="$1"
SIF="${SIF:-/scratch/gpfs/GRIFFITHS/em4240/jaxmarl_latest.sif}"

shopt -s nullglob
to_sync=()
for run_dir in "$RUN_DIR"/wandb/wandb/offline-run-*; do
  synced=("$run_dir"/*.wandb.synced)
  [ ${#synced[@]} -eq 0 ] && to_sync+=("$run_dir")
done

echo "${#to_sync[@]} run(s) to sync"
[ ${#to_sync[@]} -eq 0 ] && exit 0

apptainer exec --bind "$RUN_DIR:$RUN_DIR" "$SIF" \
  /usr/bin/python3 -m wandb sync "${to_sync[@]}"
