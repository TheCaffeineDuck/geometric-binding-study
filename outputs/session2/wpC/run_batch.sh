#!/bin/bash
# Run a list of WP-C configs one after another (one slot of 2 threads). Start one
# instance per free 2-thread slot with different lists. Each run is resumable; a run
# whose result says completed=true is skipped.
# Usage: nohup outputs/session2/wpC/run_batch.sh outputs/session2/wpC/batch_slotA.txt > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
LIST="$1"
while read -r NAME; do
    [ -z "$NAME" ] && continue
    case "$NAME" in \#*) continue ;; esac
    RES="$REPO/outputs/session2/wpC/results/$NAME.json"
    if [ -f "$RES" ] && "$REPO/.venv/bin/python" -c "import json,sys; sys.exit(0 if json.load(open('$RES'))['completed'] else 1)"; then
        echo "$(date) skip $NAME (completed)"
        continue
    fi
    echo "$(date) start $NAME"
    "$REPO/outputs/session2/wpC/run_one.sh" "$NAME" || echo "$(date) FAILED $NAME"
    echo "$(date) end $NAME"
done < "$LIST"
