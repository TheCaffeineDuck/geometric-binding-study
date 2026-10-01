#!/bin/zsh
# WP3 local launch. Usage: outputs/step2/launch.sh run1_primary [run2_primary_twin ...]
# Each run: detached, under caffeinate, threads from the config (set to 2), NUMBA_NUM_THREADS=2.
# Re-running the same command resumes from checkpoints/step2/<run>/ (the engine does this itself).
ROOT=/Users/aaronramos/Desktop/SWIRL
cd $ROOT || exit 1
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-2}
mkdir -p logs/step2 checkpoints/step2 outputs/step2/results
for run in "$@"; do
  nohup caffeinate -i $ROOT/.venv/bin/python -m swirl.engine.run outputs/step2/configs/$run.json \
    --out outputs/step2/results/$run.json --checkpoint-dir checkpoints/step2/$run --verbose \
    >> logs/step2/$run.log 2>&1 &
  echo $! > logs/step2/$run.pid
  echo "$run pid $(cat logs/step2/$run.pid)"
done
