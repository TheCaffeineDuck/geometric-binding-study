#!/bin/sh
# WP-L3 3D batch: one role per c8g.4xlarge (16 cores), 8 jobs x 2 threads, longest ready job first.
# Resumable: rerun the same line after an interruption (finished jobs are skipped; runs resume from
# checkpoints/session3/wpL3/<job>/).
#
# Usage:  outputs/session3/wpL3/entry.sh ROLE        (ROLE = A | B | C | D)
#   A  6 WP-L workers. dip: builds src_dip, gates, gate check, the 6 dip collisions, their 1.5x box checks
#      (v 0.02, 0.2), the resolution check (v 0.2), the collision twin                          ~12.7 h
#   B  7 workers. primary: checks the shipped source, gates, gate check, 6 collisions, 8 (d) runs  ~10.3 h
#   C  8 workers. dip: builds src_dip (its own copy), 8 (d) runs, 4 seed runs, 2 spot checks, (d) twin  ~8.9 h
#   D  6 workers. b5: builds src_b5, 4 (d) runs, the spot check, the 1.5x box run                   ~13.0 h
# The slots left free are for the P3 jobs (other branch, run by their own driver, 2 threads each):
#   A 2 slots: the two candidates (~12 h each); B 1 slot: one single-lump run (~7 h);
#   D 2 slots: single-lump run + in-phase control (7 + 3.3 h), single-lump run + twin (7 + 1.7 h).
#
# Collisions start only after gatecheck_<source> passes on this instance (gate at v 0.02 and 0.2, theta 0 and
# 90, and the t = 0 cross term of every collision below 1e-4). If it fails, that source's collisions and their
# checks are skipped and the reason is in outputs/session3/wpL3/results/gatecheck_<source>.json.fail.
#
# Needs: Python 3.12 with numpy 2.5, scipy 1.18, numba 0.68; the repo at the WP-L3 commit. Role B also needs the
# WP-E production source at outputs/session2/wpE/source/src_prod_N160_L88.npz (66 MB, sha256 6193e7b1...).
# The dip source is built on A and on C from the same config (no S3 copy needed); each records its sha256.
# Results: outputs/session3/wpL3/results/*.json (atomic) and summary.json per instance (merge the results
# directories, then run `python -m swirl.engine.l3 post` once on the merged set); log: logs/wpl3_<ROLE>.log.
set -eu
ROLE=${1:?role: A | B | C | D}
case "$ROLE" in A|B|C|D) ;; *) echo "unknown role $ROLE"; exit 2 ;; esac
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO"
PY=${PYTHON:-python3}
export PYTHONPATH="$REPO" OMP_NUM_THREADS=1
mkdir -p logs
case "$ROLE" in A) W=6 ;; B) W=7 ;; C) W=8 ;; D) W=6 ;; esac
STATUS=0
$PY -m swirl.engine.l3 run --jobs outputs/session3/wpL3/jobs.json --workers "$W" --threads 2 --role "$ROLE" \
    >> "logs/wpl3_$ROLE.log" 2>&1 || STATUS=$?
$PY -m swirl.engine.l3 post --jobs outputs/session3/wpL3/jobs.json >> "logs/wpl3_$ROLE.log" 2>&1 || true
echo "wpl3 $ROLE finished, run status $STATUS" >> "logs/wpl3_$ROLE.log"
exit $STATUS
