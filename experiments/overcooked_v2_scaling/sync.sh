#!/bin/bash
# Upload the offline W&B runs of a sweep (run on the login node, which has internet,
# with the jaxmarl conda env activated). Safe to re-run: runs already synced are skipped.
#
# Usage: ./sync.sh RUN_DIR
set -euo pipefail

RUN_DIR="$1"

shopt -s nullglob
to_sync=()
for run_dir in "$RUN_DIR"/wandb/wandb/offline-run-*; do
  synced=("$run_dir"/*.wandb.synced)
  [ ${#synced[@]} -eq 0 ] && to_sync+=("$run_dir")
done

echo "${#to_sync[@]} run(s) to sync"
[ ${#to_sync[@]} -eq 0 ] && exit 0

wandb sync "${to_sync[@]}"
