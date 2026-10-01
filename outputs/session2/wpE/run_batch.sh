#!/bin/bash
# Run a list of WP-E configs one after another (one slot of 2 threads). Start one instance per
# free slot with different lists. Each run is resumable; a run whose result says completed=true
# is skipped. Lines starting with # are ignored.
# Usage: nohup outputs/session2/wpE/run_batch.sh LIST [configs_dir] [results_dir] > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
LIST="$1"
CONF="${2:-outputs/session2/wpE/configs}"
RES="${3:-outputs/session2/wpE/results}"
PY="${PYTHON:-$REPO/.venv/bin/python}"
while read -r NAME; do
    [ -z "$NAME" ] && continue
    case "$NAME" in \#*) continue ;; esac
    OUT="$REPO/$RES/$NAME.json"
    if [ -f "$OUT" ] && "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$OUT'))['completed'] else 1)"; then
        echo "$(date) skip $NAME (completed)"
        continue
    fi
    echo "$(date) start $NAME"
    "$REPO/outputs/session2/wpE/run_one.sh" "$NAME" "$CONF" "$RES" || echo "$(date) FAILED $NAME"
    echo "$(date) end $NAME"
done < "$LIST"
