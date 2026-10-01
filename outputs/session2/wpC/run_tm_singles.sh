#!/bin/bash
# Job A: negative-phase single-lump runs, two at a time, resumable.
cd /Users/aaronramos/Desktop/SWIRL
outputs/session2/wpC/run_one.sh single_x12_tm22.5 &
outputs/session2/wpC/run_one.sh single_x12_tm45.0 &
wait
outputs/session2/wpC/run_one.sh single_x12_tm67.5
