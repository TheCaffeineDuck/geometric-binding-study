#!/bin/bash
# Mac-local copy of run_one.sh: results_mac/, logs_mac/, checkpoints/session2/wpC_mac/<name>. Resumable.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
NAME="$1"
cd "$REPO"
mkdir -p outputs/session2/wpC/results_mac outputs/session2/wpC/logs_mac "checkpoints/session2/wpC_mac/$NAME"
[ -f "outputs/session2/wpC/results_mac/$NAME.json" ] && exit 0
export NUMBA_NUM_THREADS=2
echo "start $(date +%s) $NAME" >> outputs/session2/wpC/logs_mac/times.txt
caffeinate -i "$REPO/.venv/bin/python" -m swirl.engine.run \
    "outputs/session2/wpC/configs/$NAME.json" \
    --out "outputs/session2/wpC/results_mac/$NAME.json" \
    --checkpoint-dir "checkpoints/session2/wpC_mac/$NAME" --verbose \
    >> "outputs/session2/wpC/logs_mac/$NAME.log" 2>&1
echo "end $(date +%s) $NAME" >> outputs/session2/wpC/logs_mac/times.txt
