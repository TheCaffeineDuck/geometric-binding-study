#!/bin/sh
# WP-L3 3D batch, one c8g.4xlarge (16 cores): 8 jobs x 2 threads. Resumable: rerun the same line after an
# interruption (finished jobs are skipped; runs resume from checkpoints/session3/wpL3/<job>/).
#
# Usage:  outputs/session3/wpL3/entry.sh ROLE
#   core     all starts, stages 1, 2, 3, 4, 5, 6 (seed 1 only for the random multiplier)     recommended
#   core3s   core plus stage 3s (random-multiplier seeds 2 and 3 at every start)
#   seeds    stage 3s only (after core, on the same disk)
#   A / B    two-instance split of core: A = dip and primary, B = b5
#   4b       optional: b5 gate, gate check and b5 collisions (after core or B, on the same disk)
#
# Collisions start only after gatecheck_<source> passes on this instance (gate at both speeds and theta = 0, 90,
# and the t = 0 cross term of every collision below 1e-4). If it fails, that source's collisions and their checks
# are skipped and the reason is in outputs/session3/wpL3/results/gatecheck_<source>.json.fail.
#
# Needs: Python 3.12 with numpy 2.5, scipy 1.18, numba 0.68; the repo at the WP-L3 commit; the WP-E production
# source at outputs/session2/wpE/source/src_prod_N160_L88.npz (66 MB, sha256 6193e7b1..., checked by job
# src_primary). Results: outputs/session3/wpL3/results/*.json (atomic) and outputs/session3/wpL3/summary.json;
# source metas: outputs/session3/wpL3/sources/*.json; log: logs/wpl3_<ROLE>.log.
set -eu
ROLE=${1:?role: core | core3s | seeds | A | B | 4b}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO"
PY=${PYTHON:-python3}
export PYTHONPATH="$REPO" OMP_NUM_THREADS=1
mkdir -p logs
case "$ROLE" in
  core)   STARTS=dip,primary,b5; STAGES=1,2,3,4,5,6 ;;
  core3s) STARTS=dip,primary,b5; STAGES=1,2,3,3s,4,5,6 ;;
  seeds)  STARTS=dip,primary,b5; STAGES=1,3s ;;
  A)      STARTS=dip,primary;    STAGES=1,2,3,4,5,6 ;;
  B)      STARTS=b5;             STAGES=1,3,5,6 ;;
  4b)     STARTS=b5;             STAGES=1,4b ;;
  *) echo "unknown role $ROLE"; exit 2 ;;
esac
STATUS=0
$PY -m swirl.engine.l3 run --jobs outputs/session3/wpL3/jobs.json --workers 8 --threads 2 \
    --stages "$STAGES" --starts "$STARTS" >> "logs/wpl3_$ROLE.log" 2>&1 || STATUS=$?
$PY -m swirl.engine.l3 post --jobs outputs/session3/wpL3/jobs.json >> "logs/wpl3_$ROLE.log" 2>&1 || true
echo "wpl3 $ROLE finished, run status $STATUS" >> "logs/wpl3_$ROLE.log"
exit $STATUS
