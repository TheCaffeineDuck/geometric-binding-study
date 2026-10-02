#!/bin/bash
# Launch one phase of the WP-P3 batch, each job in the background with 2 threads (packable with other batches on a
# c8g.4xlarge, 8 slots of 2 threads). Each job is resumable; a job whose result says completed = true is skipped.
# Usage (from the repo root):
#   phase 1 (after make_sources.py N175, N250 and make_configs.py A):
#     nohup outputs/session3/wpP3/run_batch.sh P3_inphase P3_c0230_lin > outputs/session3/wpP3/logs/phase1.log 2>&1 &
#   phase 2 (after P3_inphase completed and make_configs.py B wrote T_choice.json and the configs):
#     nohup outputs/session3/wpP3/run_batch.sh P3_c0230 P3_c0242 P3_iso_N100 P3_iso_N175 P3_iso_N250 \
#         > outputs/session3/wpP3/logs/phase2.log 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PYTHON:-$REPO/.venv/bin/python}"
mkdir -p "$REPO/outputs/session3/wpP3/logs"
for NAME in "$@"; do
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
