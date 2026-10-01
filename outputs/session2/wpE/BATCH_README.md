# WP-E arrangement batch (launched 2026-10-02 ~02:51 +07, Mac, two slots of 2 threads)

Configs: `outputs/session2/wpE/configs/` (from `candidates_wpE.json`, built by `build_candidates_wpE.py`; window (26, 30) in every config; the generator's `_lin/_res/_box` files renamed to E15_twin_E01, E15_res_E03, E15_box_E03). t = 0 check: `t0_check.py` -> `t0_check.json`.
Slot lists: `slotA.txt`, `slotB.txt`. Results `outputs/session2/wpE/results/<name>.json`; logs `outputs/session2/wpE/logs/<name>.log` and `slotA.log`/`slotB.log`; checkpoints `checkpoints/session2/wpE/<name>/` (every 50 time units).

## Launch (from the repo root; PIDs in logs/session2/wpE_slotA.pid and wpE_slotB.pid)

    cd /Users/aaronramos/Desktop/SWIRL
    nohup caffeinate -i outputs/session2/wpE/run_batch.sh outputs/session2/wpE/slotA.txt > outputs/session2/wpE/logs/slotA.log 2>&1 &
    echo $! > logs/session2/wpE_slotA.pid
    nohup caffeinate -i outputs/session2/wpE/run_batch.sh outputs/session2/wpE/slotB.txt > outputs/session2/wpE/logs/slotB.log 2>&1 &
    echo $! > logs/session2/wpE_slotB.pid

## Progress

    cat outputs/session2/wpE/logs/slotA.log outputs/session2/wpE/logs/slotB.log     # start/end/skip lines
    tail -n 3 outputs/session2/wpE/logs/<name>.log                                  # step count and steps/s of the running config
    ls outputs/session2/wpE/results/                                                # finished runs (completed = true inside)
    ps -p $(cat logs/session2/wpE_slotA.pid),$(cat logs/session2/wpE_slotB.pid)

## Resume after a restart

Run the same two launch commands again. `run_batch.sh` skips runs whose result says completed = true and `run_one.sh` resumes an unfinished run from its checkpoint. Do not start a list twice at once. Order: A: E01 E03 E05 E07 E09 E11 E13 E15_twin_E01 E15_box_E03; B: E02 E04 E06 E08 E10 E12 E15_res_E03 E14.
A run that fails is logged as FAILED in the slot log and the slot moves on.
