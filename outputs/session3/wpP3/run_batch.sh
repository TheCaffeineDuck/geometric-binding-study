#!/bin/bash
# Launch the WP-P3 batch on one c8g.4xlarge (16 cores): the six jobs at once, 2 threads each (12 cores).
# Each job is resumable; a job whose result says completed = true is skipped.
# Usage (from the repo root, after make_sources.py N175, N250 and make_configs.py):
#   nohup outputs/session3/wpP3/run_batch.sh > outputs/session3/wpP3/logs/batch.log 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PYTHON:-$REPO/.venv/bin/python}"
mkdir -p "$REPO/outputs/session3/wpP3/logs"
for NAME in P3_c0230_far P3_c0242_far P3_c0230 P3_c0242 P3_inphase P3_c0230_lin; do
    OUT="$REPO/outputs/session3/wpP3/results/$NAME.json"
    if [ -f "$OUT" ] && "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$OUT'))['completed'] else 1)"; then
        echo "$(date) skip $NAME (completed)"
        continue
    fi
    echo "$(date) start $NAME"
    ( "$REPO/outputs/session3/wpP3/run_one.sh" "$NAME" || echo "$(date) FAILED $NAME" ) &
done
wait
echo "$(date) all done"
