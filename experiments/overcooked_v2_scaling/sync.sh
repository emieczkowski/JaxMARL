#!/bin/bash
# Upload offline W&B runs of the scaling sweep (run on the login node, which has internet).
# Safe to re-run: runs already synced are skipped.
set -euo pipefail

OUT_DIR=${OUT_DIR:-/scratch/gpfs/$USER/overcooked_v2_scaling}
SIF=${SIF:-/scratch/gpfs/$USER/jaxmarl_latest.sif}

shopt -s nullglob
to_sync=()
for run_dir in "$OUT_DIR"/wandb/wandb/offline-run-*; do
  synced=("$run_dir"/*.wandb.synced)
  [ ${#synced[@]} -eq 0 ] && to_sync+=("$run_dir")
done

echo "${#to_sync[@]} run(s) to sync"
[ ${#to_sync[@]} -eq 0 ] && exit 0

apptainer exec --bind "$OUT_DIR:$OUT_DIR" "$SIF" \
  /usr/bin/python3 -m wandb sync "${to_sync[@]}"
