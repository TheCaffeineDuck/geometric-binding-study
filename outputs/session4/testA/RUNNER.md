# Test A: runner commands

All commands from the worktree root, with `PY=/Users/aaronramos/Desktop/SWIRL/.venv/bin/python`.
The driver sets `NUMBA_NUM_THREADS=1` in each pool worker; the parent never imports numba. At most 4 workers.
Every run checkpoints every 50 time units under `checkpoints/session4/testA/<name>/` and resumes on restart.
Results: `outputs/session4/testA/results/<name>.json` (full parameters inside; written .tmp then os.replace).
t = 0 checks: `outputs/session4/testA/t0/<name>.json`.

## 0. Sources

- `primary`: the Session 2 WP-E production source, read in place from the main checkout
  (`outputs/session2/wpE/source/src_prod_N160_L88.npz`, sha256 6193e7b1...8c72, checked at every load).
- `dip`, `big` (unequal pair) and `b5`: built by the driver jobs `src_dip`, `src_big`, `src_b5` into
  `outputs/session4/testA/sources/` (field arrays; gitignored). If they were built in another worktree, copy the
  `.npz` and `.json` files into this worktree's `outputs/session4/testA/sources/`; the hash is checked at load.

## 1. Job list

    $PY -m swirl.session4.testA.driver jobs          # without b5 (108 jobs)
    $PY -m swirl.session4.testA.driver jobs --b5     # with b5 (118 jobs; the cost gate passes, see the report)
    $PY -m swirl.session4.testA.driver cost          # cost table from outputs/session4/testA/cost_rate.json

## 2. Production (detached, logged, Mac kept awake)

    mkdir -p outputs/session4/testA/logs
    nohup caffeinate -i env PYTHONPATH=. $PY -m swirl.session4.testA.driver run --workers 4 \
        >> outputs/session4/testA/logs/production.log 2>&1 &

Restart with the same command after any interruption: finished jobs are skipped, running ones resume from their
checkpoint. Subsets: `--group primary controls` (sources always included), `--only NAME ...`.

## 3. Analysis

    $PY -m swirl.session4.testA.analysis static      # A3 verdicts for the static potentials
    $PY -m swirl.session4.testA.analysis all         # V_N, V_F, fits, verdicts, controls -> analysis.json

## Notes

- Pair runs stop early once kappa |d - r| > 0.13 (t >= 60); `stopped_early` is recorded in the result.
- Never commit `checkpoints/` or `outputs/session4/testA/sources/*.npz`. Raw result JSONs are about 3 to 5 MB each.
