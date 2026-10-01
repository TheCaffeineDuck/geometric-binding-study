#!/bin/zsh
# Waits for run1_primary and run2_primary_twin (pid files in logs/step2), then launches the secondary pair.
# Start detached:  nohup outputs/step2/launch_secondary.sh > logs/step2/launch_secondary.log 2>&1 &
ROOT=/Users/aaronramos/Desktop/SWIRL
cd $ROOT || exit 1
for run in run1_primary run2_primary_twin; do
  pid=$(cat logs/step2/$run.pid)
  while kill -0 $pid 2>/dev/null; do sleep 30; done
  echo "$(date) $run (pid $pid) finished"
done
echo "$(date) launching secondary pair"
$ROOT/outputs/step2/launch.sh run1b_secondary run2b_secondary_twin
