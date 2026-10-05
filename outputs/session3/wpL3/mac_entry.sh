#!/bin/sh
# WP-L3 reduced set on the Mac: 2 workers x 2 threads, phases in order. Resumable: rerun the same line.
# Jobs: outputs/session3/wpL3/jobs_mac.json (subset of jobs.json, same definitions; job_hours set to a priority
# order only). Phase 2 runs only if gatecheck_primary passed (else the driver reports "blocked" and exits).
set -u
REPO=$(cd "$(dirname "$0")/../../.." && pwd); cd "$REPO"
PY=${PYTHON:-/Users/aaronramos/Desktop/SWIRL/.venv/bin/python}
export PYTHONPATH="$REPO" OMP_NUM_THREADS=1
mkdir -p logs
J=outputs/session3/wpL3/jobs_mac.json
L=logs/wpl3_mac.log
echo "$(date +%T) phase 1: primary gates" >> $L
$PY -m swirl.engine.l3 run --jobs $J --workers 2 --threads 2 --starts primary --stages 1,2 >> $L 2>&1
echo "$(date +%T) phase 2: primary collisions (skipped by the driver if the gate check failed)" >> $L
$PY -m swirl.engine.l3 run --jobs $J --workers 2 --threads 2 --starts primary --stages 4 >> $L 2>&1
echo "$(date +%T) phase 3: src_dip and dip (d) runs" >> $L
$PY -m swirl.engine.l3 run --jobs $J --workers 2 --threads 2 --starts dip --stages 1,3 >> $L 2>&1
echo "$(date +%T) wpl3 mac finished" >> $L
