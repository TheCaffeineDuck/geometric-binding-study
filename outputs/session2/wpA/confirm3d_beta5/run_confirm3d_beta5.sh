#!/bin/bash
# Resumable. Usage: run_confirm3d_beta5.sh confirm3d_b5_primary | confirm3d_b5_twin
set -euo pipefail
REPO=/Users/aaronramos/Desktop/SWIRL; NAME="$1"; cd "$REPO"
D=outputs/session2/wpA/confirm3d_beta5
mkdir -p $D/results $D/logs "checkpoints/session2/wpA_confirm3d_beta5/$NAME"
[ -f $D/results/$NAME.json ] && exit 0
export NUMBA_NUM_THREADS=2
echo "start $(date +%s) $NAME" >> $D/logs/times.txt
caffeinate -i $REPO/.venv/bin/python -m swirl.engine.run $D/$NAME.json --out $D/results/$NAME.json \
  --checkpoint-dir checkpoints/session2/wpA_confirm3d_beta5/$NAME --verbose >> $D/logs/$NAME.log 2>&1
echo "end $(date +%s) $NAME" >> $D/logs/times.txt
