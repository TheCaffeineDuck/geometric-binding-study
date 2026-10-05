#!/bin/sh
# Phase 3b: the two linear twins. Waits for the main chain (mac_entry.sh) to finish, then runs them (2 workers).
# Does not touch the running chain. Resumable: rerun the same line.
set -u
REPO=$(cd "$(dirname "$0")/../../.." && pwd); cd "$REPO"
PY=${PYTHON:-/Users/aaronramos/Desktop/SWIRL/.venv/bin/python}
export PYTHONPATH="$REPO" OMP_NUM_THREADS=1
L=logs/wpl3_mac.log
until grep -q "wpl3 mac finished" $L; do sleep 60; done
echo "$(date +%T) phase 3b: linear twins" >> $L
$PY -m swirl.engine.l3 run --jobs outputs/session3/wpL3/jobs_mac_twins.json --workers 2 --threads 2 --stages 6 >> $L 2>&1
echo "$(date +%T) wpl3 mac 3b finished" >> $L
