#!/bin/bash
# usage: run_jobs.sh jobsfile  -- runs each line through run_evolve5 with 2 workers, serial per job
cd /Users/aaronramos/Desktop/SWIRL/.claude/worktrees/s5-trackF
export NUMBA_NUM_THREADS=1 PYTHONPATH=.
J=$1
cat "$J" | xargs -P 2 -I{} bash -c 'tag=$(echo "{}" | sed "s/.*--tag //;s/ .*//"); echo "$(date +%T) start $tag" >> outputs/session5/trackF/logs/progress.log; /Users/aaronramos/Desktop/SWIRL/.venv/bin/python -m swirl.radial5.run_evolve5 {} --out-dir outputs/session5/trackF/runs > outputs/session5/trackF/logs/$tag.log 2>&1; echo "$(date +%T) done $tag rc=$?" >> outputs/session5/trackF/logs/progress.log'
