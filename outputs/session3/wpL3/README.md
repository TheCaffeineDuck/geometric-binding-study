# WP-L3 3D batch: what the cloud worker runs

Code: `swirl/engine/l3.py` (jobs, driver, post), `swirl/engine/boost.py` (initial data, 3D kicks, shape), `swirl/engine/run.py` (engine; new optional `shape` block).
- Job list: `jobs.json` (60 jobs after the budget trim, configs embedded, each with its instance role). Per-job cost: `cost.json`. Entry: `entry.sh ROLE`.
- Driver smoke test: `tests/engine/test_wpl3_driver.py`. It passed locally on the dx = 0.8 test source; it also checks that the gate check stops on a large cross term and writes the reason.

## Instance and run

- c8g.4xlarge (16 cores, 64 GB); Python 3.12 with numpy 2.5.3, scipy 1.18.1, numba 0.68.0; the repo at the WP-L3 commit.
- Four instances, one role each: `nohup outputs/session3/wpL3/entry.sh A &` (and B, C, D on the others).
- **Ship** the WP-E production source to instance B only, at `outputs/session2/wpE/source/src_prod_N160_L88.npz` (66 MB, sha256 6193e7b109b7369c...).
- The dip source is built on A and on C from the same config, so no S3 copy is needed. The b5 source is built on D. Each source result records its sha256; the two dip builds are expected to be identical (same code, same instance type).
- The P3 jobs run on the same instances under their own driver, in the slots WP-L leaves free (below).
- Log: `logs/wpl3_<ROLE>.log`, one line per job start and end. Rerun the same line to resume.
- Collect from every instance `outputs/session3/wpL3/results/*.json`, `results/*.fail`, `sources/*.json` and the logs. Merge the results directories, then run `python -m swirl.engine.l3 post` once on the merged set; each instance's own `summary.json` covers only its jobs.
- Do not bring back npz or checkpoints. The result JSONs are 2 to 6 MB each.

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

## Jobs (after the budget trim)

Rate model: 3.56 steps/s per 2-thread job at N = 160, scaled as N^3 ln N (Session 2's 33.5 steps/s total for 8 N = 160 jobs on c8g.4xlarge, less 15% for diagnostics). Grid dx = 0.55, dt = 0.04 unless stated.

| stage | jobs | grid (N, L) | T | job-h each | job-h |
|---|---|---|---|---|---|
| 1 sources | src_dip (built on A and on C), src_b5 (D), src_primary (shipped to B, hash check) | 160, 88 / 216, 118.8 | 500 | 1.07 / 2.79 | 3.9 |
| 2 gate | dip, primary: v = 0; v = 0.02 and 0.2 at theta 0 and 90; gate checks | 160, 88 | 240 / 240 / 120 | 0.47 / 0.47 / 0.23 | 3.8 |
| 3 (d) | dip, primary: control; stretch 0.05, 0.1, 0.2; random 0.02, 0.05, 0.1, 0.15 (seed 1). b5: control, stretch 0.2, random 0.1 and 0.15 (seed 1) | 160, 88; 216, 118.8 | 2000 | 3.90 / 10.17 | 103.1 |
| 3s seeds | dip only, random 0.1 and 0.15, seeds 2 and 3 | 160, 88 | 2000 | 3.90 | 15.6 |
| 4 (e) | dip, primary x phase 0, 90, 180 x v 0.02, 0.2; d = 30 | 180, 99 | 2080 / 1560 | 5.91 / 4.43 | 62.1 |
| 5 spot | dip field -0.1, dip removal 0.2, b5 field +0.1 | as stage 3 | 2000 | 3.90 / 10.17 | 18.0 |
| 6 checks | twins of d_dip_q0.1 and coll_dip_p0_v0.02 (T 500); 1.5x box (270, 148.5) of coll_dip_p0 at v 0.02 (T 1080) and 0.2 (T 560); resolution (270, 99, dx 0.367, dt 0.0267) of coll_dip_p0 at v 0.2 (T 560); 1.5x box of d_b5_r0.1_s1 (324, 178.2, T 500) | — | — | 0.98 to 11.17 | 37.3 |
| **WP-L total** | 60 jobs | | | | **243.7** |
| P3 allowance | 2 candidates ~12 h, 3 single-lump runs ~7 h, in-phase control 3.3 h, twin 1.7 h | | | | 50.0 |

Dropped and parked: the b5 collisions (4b) and b5 gate. Also dropped: the other b5 (d) runs, seeds 2 and 3 except at the dip for a = 0.1 and 0.15, and the resolution check at v = 0.02.

## Instances (c8g.4xlarge, 16 cores, 2 threads per job, longest ready job first)

| role | WP-L workers | WP-L jobs (job-h) | P3 slots (planned job-h in sequence) | wall h | USD |
|---|---|---|---|---|---|
| A | 6 | src_dip; dip gates and gate check; 6 dip collisions; box of coll_dip_p0 at v 0.02 and 0.2; resolution at v 0.2; collision twin (61.1) | 2 slots: the two candidates (12, 12) | 12.7 | 8.09 |
| B | 7 | src_primary check; primary gates and gate check; 6 primary collisions; 8 primary (d) runs (64.1) | 1 slot: single-lump run (7) | 10.3 | 6.55 |
| C | 8 | src_dip (own build); 8 dip (d) runs; 4 seed runs; 2 dip spot checks; (d) twin (56.7) | none | 8.9 | 5.64 |
| D | 6 | src_b5; 4 b5 (d) runs; b5 spot check; b5 box run (62.9) | 2 slots: single-lump + control (7 + 3.3), single-lump + twin (7 + 1.7) | 13.0 | 8.24 |
| **total** | | 243.7 WP-L + 50 P3 job-h | | 44.9 instance-h | **28.52** |

- Wall times come from a longest-ready-job-first schedule with the dependencies, at the rate model above. Each instance is billed until its last job ends.
- About 9.69 USD has been spent, so this batch takes the total to about 38.2 USD, below the 40 USD notice mark.
- The P3 slot plan assumes the P3 job lengths given by the coordinator. Longer P3 jobs on A or D extend those instances.

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
- **(e)** (dip and primary only):
  - outcome (merge / bounce / pass-through / fragmented / dispersed, from the debris-floored count of the last three windows and the tracked centroids), at the end and before sponge arrival;
  - measured contact time;
  - remnant energy (group sphere, last 300 time units) over E0, and remnant omega;
  - lump energies and speeds; energy absorbed;
  - phase label; cross term at t = 0; the minimum tracker distance (to judge tracker swaps).
- **Spot checks:** the (d) quantities. `post --radial outputs/session3/wpL/runs` compares them with the radial runs, locally, after the L1 scan.
- **Checks:** twins (frac at T, dispersal); box and resolution (outcome, contact, remnant against the base run); the b5 box check (frac, N, a2 against the base to T = 500).
