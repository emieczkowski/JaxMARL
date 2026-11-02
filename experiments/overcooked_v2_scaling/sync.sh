#!/bin/bash
# Upload the offline W&B runs of a sweep (run on the login node, which has internet,
# with the jaxmarl conda env activated). Safe to re-run, also while jobs are running:
# runs with no new data since their last sync are skipped.
#
# Usage: ./sync.sh RUN_DIR
set -euo pipefail

RUN_DIR="$1"

shopt -s nullglob
to_sync=()
for run_dir in "$RUN_DIR"/wandb/wandb/offline-run-*; do
  for data in "$run_dir"/run-*.wandb; do
    [ "$data.synced" -nt "$data" ] || to_sync+=("$run_dir")
  done
done

echo "${#to_sync[@]} run(s) to sync"
[ ${#to_sync[@]} -eq 0 ] && exit 0

# --include-synced: without it, wandb sync skips any run synced before, even one with new data
wandb sync --include-synced "${to_sync[@]}"
