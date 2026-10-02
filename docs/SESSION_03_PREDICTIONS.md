# Session 3 predictions on record

Written by the orchestrator before the runs they refer to. Each block carries its commit, so the order can be checked in `git log`. Each prediction sits next to what the literature expects, with a source from `docs/PRIOR_ART.md` (standard 9). "CD" marks a guess carried over from Claude Desktop in `SESSION_03_PLAN.md`.

Units: Set-B-like (m = 1, g4 = 0.30, g6 = 0.09 beta) unless stated. "frac" is E(r < 3 initial half-widths)/E0. "hw" is the half-width. N is the lump's charge, (omega/2) times the integral of the squared profile; disturbances are stated as the fractional change they make in N, with the energy delivered as a second number.

This file was started by the runner in WP0 with the definitions only. It holds no predictions yet; the orchestrator writes them in blocks below, before the runs they refer to.

## Definitions fixed before any run

### The corrected bonded rule

Replaces the Session 2 rule (which labelled the far-apart control and E11's first run "bonded"; `SESSION_02_REPORT.md` section 7 and section 13). Code: `swirl/engine/bonded.py`, function `bonded_rule(run, control, contact_time, **parameters)`; inputs from a stored WP-E result by `series_from_result(result)`. The old rule is kept there as `bonded_rule_session2`, superseded. Tests: `tests/engine/test_bonded.py`.

A group of K lumps is **bonded** only if all of the following hold, with the defaults below (every number is an argument of the function):

1. **Count.** The number of distinct lumps equals K in every count window from t = 50 on (`t_from`).
2. **Distance.** Every lump's nearest-neighbour distance stays within 10% of its start value (`dist_tol` = 0.10).
3. **Trend.** No lump's nearest-neighbour distance has a linear trend (fit over t from 50 on) whose magnitude exceeds 3 times (`trend_factor`) the far-apart control's own trend. The control's trend is floored at the Session 2 noise floor of E14: 5e-4 in the distance ratio over T = 500 (`noise_nn`).
4. **N held.** No N exchange between lumps: each lump's relative N change from its first sample, minus the mean change of all lumps, stays within 1e-3 (`N_tol`, 5 times the floor 2e-4, `noise_N`). The common drift is removed because an isolated lump radiates: in E14 every lump's N falls by 1.5e-3 over T = 500, while the spread between its lumps is at most 1.5e-4. The 2e-4 quoted for E14 in Session 2 is that spread, not the per-lump change.
5. **Phase steps held.** Every neighbour phase step changes by no more than 0.25 degrees from its first value (`steps_tol_deg`, 5 times the floor 0.05 degrees, `noise_steps_deg`).
6. **Duration.** The run lasts at least 5 interaction times (`interaction_periods`). The unit is the in-phase control's contact time, an input with no default (Session 2: E2, 160.4, so five units are 802).
7. **Response.** The run's response differs from the far-apart control's: the largest distance change, N exchange or phase-step change exceeds 3 times (`response_factor`) the control's value, each floored at the E14 noise floors.

Verdicts, in this order of precedence: **not bonded** (any of 1 to 5 fails, at any run length); **no interaction** (1 to 5 hold but 7 fails: the run did what the control did); **too short** (1 to 5 and 7 hold but 6 fails); **bonded** (every clause holds). The function returns the verdict, each clause's measured value and limit, and pass or fail. A bonded outcome gets a second run at finer resolution and a longer time before it is reported (plan, WP-P3).

Measured on the saved Session 2 runs (not predictions): E14 against itself gives "no interaction" (distance change 4.3e-4, N exchange 7e-5, phase-step change 0.036 degrees); E11's first run (T = 700) gives "not bonded" (distance change 0.41, trend 3.4e-4 per unit time against a limit of 3.4e-6, N exchange 0.21, phase-step change 18 degrees) and is also too short. The 3x and 5x scalings are the runner's reading of "a few times"; the orchestrator may change them before a run, in the arguments.

### Lump count

The lump finder counts maxima of the period-averaged energy density above the larger of 1% of the current maximum and an absolute floor of 0.05 times the first count window's maximum (`count_floor_frac`, `count_floor_ref` in the lumps config; `swirl/engine/lumps.py`). A field with no maximum above the floor has zero lumps. The reason and the check against saved runs are in the module docstring and `tests/engine/test_lump_floor.py`.

### Group energy

`E_residence_end` (old name `E_retained_end`) is the energy inside the group sphere, period-averaged, over its t = 0 value. It measures residence in the sphere, not survival: lumps that leave the sphere alive lower it. Quote it only with the sphere radius.

### WP-L: when a lone lump is broken

Copied from `SESSION_03_PLAN.md`, section 2, WP-L, "Definitions".

- **Broken outright:** within 2,000 time units, the energy inside 3 half-widths of the tracked centre falls below half its value just before the disturbance, while the undisturbed control stays above 0.9.
- **Degraded:** the loss rate and the omega shift after the disturbance, relative to the undisturbed control, with the radial solver giving the long horizon.

## Blocks

(None yet. The orchestrator writes each block before the runs it refers to, with its literature column and source for every row.)
