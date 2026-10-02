# WP-L3 3D batch: what the cloud worker runs

Code: `swirl/engine/l3.py` (jobs, driver, post), `swirl/engine/boost.py` (initial data, 3D kicks, shape), `swirl/engine/run.py` (engine; new optional `shape` block).
- Job list: `jobs.json` (97 jobs, configs embedded). Per-job cost: `cost.json`. Entry: `entry.sh ROLE`.
- Driver smoke test: `tests/engine/test_wpl3_driver.py`. It passed locally on the dx = 0.8 test source; it also checks that the gate check stops on a large cross term and writes the reason.

## Instance and run

- c8g.4xlarge (16 cores, 64 GB); Python 3.12 with numpy 2.5.3, scipy 1.18.1, numba 0.68.0; the repo at the WP-L3 commit.
- **Ship** the WP-E production source to `outputs/session2/wpE/source/src_prod_N160_L88.npz` (66 MB, sha256 6193e7b109b7369c...).
- Run `nohup outputs/session3/wpL3/entry.sh core &`. Log: `logs/wpl3_core.log`, one line per job start and end. Rerun the same line to resume.
- Collect `outputs/session3/wpL3/results/*.json`, `results/*.fail`, `summary.json`, `sources/*.json` and the logs. Do not bring back npz or checkpoints. The result JSONs are 2 to 6 MB each.

## Conventions

- Production grid dx = 0.55, dt = 0.04, default sponge. The resolution check uses dx = 0.367, dt = 0.0267, with the source resampled spectrally.
- Lumps are settled snapshot copies: window (26, 30) at beta = 4/3, (36, 42) at beta = 5; 32 phases.
- **Collisions:** lumps at x = -d/2 and +d/2. Each moves at v toward the other, so the closing speed is 2v.
  - v = 0.02 per lump (Xue et al. 2025 used 0.01 to 0.04) and v = 0.2 (the kinetic case).
  - d = 30 at beta = 4/3 and d = 56 at beta = 5.
- **Phase label** = relative phase of lump B minus lump A at the meeting point x = 0. It holds there at every t by symmetry; at offset x in the overlap the relative phase is label + 2 omega gamma v x. The label is recorded in every collision result.
- **Times at beta = 4/3:**
  - Meeting (contact at 2 hw): (d - 2 hw)/(2v) = 580 at v = 0.02 and 58 at v = 0.2.
  - Horizon: contact + 1500, so T = 2080 and 1560.
  - Sponge arrival: a lump leaving at v reaches the sponge (Phi = 1e-3 Phi0 at its inner radius) after 1531 (v = 0.02) and 153 (v = 0.2).
  - Two-lump outcomes (bounce, pass-through) are classified from the count windows before that time; merges are read to the horizon. Both classifications are stored.
- **Velocity guard:** every other moving lump (the gates) stays off the sponge for its whole run. This is checked when the job list is built.
- **Gate check (automatic, per source, on the instance that runs the collisions):** it passes when, at v = 0.02 and 0.2 and at theta = 0 and 90:
  - speed, E/(gamma E0), omega_lab/(omega/gamma) and ball-energy drift are within 3x of 1e-2, 1e-4, 1e-4 and 1e-3;
  - the t = 0 cross term of every collision of that source is below 1e-4.
  - Otherwise that source's collisions and checks are skipped and `results/gatecheck_<source>.json.fail` holds the reason.

## Jobs

Rate model: 3.56 steps/s per 2-thread job at N = 160, scaled as N^3 ln N (Session 2's 33.5 steps/s total for 8 N = 160 jobs on c8g.4xlarge, less 15% for diagnostics).

| stage | jobs | grid (N, L) | T | steps each | job-h each | job-h |
|---|---|---|---|---|---|---|
| 1 sources | src_dip, src_b5 (settled to 500), src_primary (shipped, hash check) | 160, 88 / 216, 118.8 | 500 | 13,700 | 1.07 / 2.79 | 3.9 |
| 2 gate | dip, primary: v = 0; v = 0.02 and 0.2 at theta 0 and 90 (from x = 0, -2.4, -12); gate checks | 160, 88 | 240 / 240 / 120 | 6,000 / 6,000 / 3,000 | 0.47 / 0.47 / 0.23 | 3.8 |
| 3 (d) | 3 starts x {control; stretch eps 0.05, 0.1, 0.2; random a 0.02, 0.05, 0.1, 0.15 seed 1} | 160, 88; b5 216, 118.8 | 2000 | 50,000 | 3.90 / 10.17 | 143.8 |
| 3s (d) seeds | random a 0.02 to 0.15, seeds 2 and 3, 3 starts | as 3 | 2000 | 50,000 | 3.90 / 10.17 | 143.8 |
| 4 (e) | dip, primary x phase 0, 90, 180 x v 0.02, 0.2; d = 30 | 180, 99 | 2080 / 1560 | 52,000 / 39,000 | 5.91 / 4.43 | 62.1 |
| 5 spot | dip field -0.1, dip removal 0.2, b5 field +0.1 | as 3 | 2000 | 50,000 | 3.90 / 10.17 | 18.0 |
| 6 checks | twins of d_dip_q0.1 and coll_dip_p0_v0.02 (T 500); 1.5x box (270, 148.5) and resolution (270, 99) of coll_dip_p0 at v 0.02 (T 1080) and 0.2 (T 560); 1.5x box of d_b5_r0.1_s1 (324, 178.2, T 500) | — | — | 12,500 to 40,500 | 0.98 to 16.75 | 54.0 |
| 4b optional | b5 gate (5) + gate check; b5 collisions, d = 56 | 216, 118.8 / 270, 148.5 | 240 / 2570 / 1610 | up to 64,250 | 26.6 / 16.7 | 134.6 |

## Cost at 0.6358 USD per instance-hour (8 jobs x 2 threads per instance)

| what | job-h | lower bound (job-h / 8) | 1 instance: wall, USD | 2 instances (A, B): wall, USD |
|---|---|---|---|---|
| **core = stages 1-6 without 3s (`entry.sh core`), recommended** | 285 | 22.7 | **45.4 h, 28.8** | 33.4 + 23.1 h, 36.0 |
| core + 3s for dip and primary only | 348 | 27.7 | 53.2 h, 33.8 | 41.3 + 23.1 h, 40.9 |
| core + 3s at all starts (`entry.sh core3s`) | 429 | 34.1 | 63.3 h, 40.3 | 41.3 + 33.3 h, 47.4 |
| 4b (b5 collisions), extra | 135 | 10.7 | about 30 h, about 19 | — |

The project has 40.31 USD left, and 9.69 has been spent. Core alone takes total spend to about 38.5. Core3s (all seeds) or 4b takes it past the 40 USD mark at which Aaron must be told; both together exceed the 50 USD budget. Memory is at most about 5.5 GB per job, so 8 jobs fit in 64 GB.

## What each job produces for the predictions

- **Sources:** settled omega, its drift, crop loss per window, nonlinear share.
- **Gate:** speed, E/(gamma E0), omega_lab/omega_rest against 1/gamma, ball-energy and I_xx/I_yy drift, at each speed and theta; the cross terms (`results/gatecheck_*.json`).
- **(d)**, against the start's control:
  - frac(t) = E(r < 3 hw)/E_control(0); min frac and first time below 0.5 (the broken-outright test);
  - whole-period ball N and omega in windows;
  - a2, a3 smoothed over 3 periods (first, max, end);
  - E + absorbed; nonlinear share.
  - a = 0.15 gives Delta N / N = 2.25%, past the dip's 1.7% width.
  - The random-multiplier runs are the dip start's first 3D test with no symmetry; the controls keep exact cubic symmetry.
- **(e):**
  - outcome (merge / bounce / pass-through / fragmented / dispersed, from the debris-floored count of the last three windows and the tracked centroids), at the end and before sponge arrival;
  - measured contact time;
  - remnant energy (group sphere, last 300 time units) over E0, and remnant omega;
  - lump energies and speeds; energy absorbed;
  - phase label; cross term at t = 0; the minimum tracker distance (to judge tracker swaps).
- **Spot checks:** the (d) quantities. `post --radial outputs/session3/wpL/runs` compares them with the radial runs, locally, after the L1 scan.
- **Checks:** twins (frac at T, dispersal); box and resolution (outcome, contact, remnant against the base run); the b5 box check (frac, N, a2 against the base to T = 500).
