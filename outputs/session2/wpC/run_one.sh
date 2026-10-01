#!/bin/bash
# Run one WP-C config, resumable (re-running continues from the checkpoint).
# Usage (from anywhere): outputs/session2/wpC/run_one.sh <config name without .json>
# Result: outputs/session2/wpC/results/<name>.json (atomic, full parameters)
# Log:    outputs/session2/wpC/logs/<name>.log
# Checkpoints: checkpoints/session2/wpC/<name>/ (never committed)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
NAME="$1"
cd "$REPO"
mkdir -p outputs/session2/wpC/results outputs/session2/wpC/logs "checkpoints/session2/wpC/$NAME"
export NUMBA_NUM_THREADS=2
exec caffeinate -i "$REPO/.venv/bin/python" -m swirl.engine.run \
    "outputs/session2/wpC/configs/$NAME.json" \
    --out "outputs/session2/wpC/results/$NAME.json" \
    --checkpoint-dir "checkpoints/session2/wpC/$NAME" --verbose \
    >> "outputs/session2/wpC/logs/$NAME.log" 2>&1
