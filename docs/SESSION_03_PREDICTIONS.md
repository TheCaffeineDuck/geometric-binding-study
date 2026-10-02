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

## Block P. WP-P2 point-model scan. Written 2026-10-02 13:55 (+07), before the P2 scan

**What was known when this was written.** The P1 build (branch `worktree-agent-a8f6bbfab55407fab`, commit 1e378b0, `docs/WPP_BUILD_NOTES.md` there) and its 24-run smoke test: 12 flew apart, 8 merged, 2 bounded for the run, 2 left the family's omega range, all "not bonded". Its linearization: every rotating balance of equal lumps at 3.6 to 8 hw has an unstable mode with an e-folding time of 0.001 to 0.14 orbital periods; circular orbits under the tail law are radially unstable for kappa d > (1 + sqrt 5)/2, and kappa d is about 5 or more over the whole valid range. Beat period over orbital period at 4.8 hw: 0.98, 0.26, 0.18 for N ratios 1.1, 1.5, 2. The beat-averaged force in the model is attractive in 11 of 12 cases, 1e-4 to 0.25 of the in-phase force. The WP-S brief (`PRIOR_ART.md` T4, T5, T7, N34). So these predictions lean on the build's own analysis; they test the scan against it.

**Decisions fixed before the scan.**
- **Phase clause for moving and unequal groups.** Clause 5 of the bonded rule is applied (a) for moving groups to the effective step D_eff = theta_j - theta_i + (delta_i + delta_j)/2 (the step the interaction depends on; it equals the raw step at rest), and (b) for unequal frequencies as "locked": each edge's linear phase drift zero within the far-apart control's noise and its excursion within the clause-5 tolerance. A group that meets every other clause but whose steps drift at the lumps' frequency difference is labelled **bounded, beating**. It is a 3D candidate (P3), never a bonded claim: in this model it would be held only by second-order forces the model leaves out.
- **What the point model can claim.** A point-model "bonded" or "bounded, beating" is a candidate only. Anything closer than 3.6 hw at any time is flagged and not counted. The model omits the phase-independent second-order attraction of unequal lumps (N34), so a negative result for unequal lumps in this scan does not settle the question; P3 in 3D does.
- **Interaction unit** for the duration clause: the in-phase contact time of equal lumps at the same K and spacing (the build's choice). Runs are capped at T = 10,000; a group still bounded at the cap with fewer than five units is "too short".

| # | Prediction | Basis | Literature |
|---|---|---|---|
| P1 | **Equal lumps in orbit** (pairs and rotating rings, every phase pattern, 3.6 to 8 hw, 0.6 to 1.4 times the circular speed): none bonded. At least 90% of the runs at 3.6 to 6 hw merge or fly apart within 2 orbital periods. At 7.2 and 8 hw, where the cap is under one period, most runs end "too short" or with an event | Linearization: fastest e-fold 0.001 to 0.14 T_orb from a 1e-3 seed, so about 7 e-folds, under one period, to reach order one; circular orbits radially unstable at kappa d > 1.62 | Coherent solitons admit no stable orbiting pair: negative phase mass, saddle points (Kivshar, Desyatnikov & Torner review sec. 6.1; Malomed 1998; `PRIOR_ART.md` T5). Classical: circular orbits in a Yukawa potential are unstable beyond a critical kappa r (source to be supplied by the scholar in WP-F) |
| P2 | **Static unequal pairs and rings**: none bonded, none "bounded, beating" in the valid range. At N ratio 1.1 the outcome follows the starting phase as for equal lumps (the beat is about as slow as the dynamics). At N ratio 1.5 and above, the pair drifts together on the beat-averaged attraction, and contact comes at least 3 times later than for an equal in-phase pair at the same spacing, or not within the cap | Model: phase runs at omega_1 - omega_2; beat-averaged force attractive in 11 of 12 cases and small | First-order force averages to zero (Axenides 2000; Battye & Sutcliffe 2000). Unequal Q-balls at rest repel with little charge transfer in 1D (Battye & Sutcliffe): that is the opposite sign to this prediction. NLS unequal pairs bind (1D, Al Khawaja 2010) or spiral (3D incoherent, Buryak 1998) through a second-order attraction (T4) |
| P3 | **Unequal orbiting pairs and alternating-size rings**: none bonded and none "bounded, beating" in the valid range. Beating removes the coherent phase instability, but the radial instability of circular orbits does not depend on phase, so each orbit spirals in or escapes within a few orbital periods | kappa d > 1.62 everywhere in the valid range | No published real-field orbiting pair; every stable published orbit needs a second component, nonlocality or a trap (T5) |
| P4 | **Overall**: zero bonded and zero "bounded, beating" groups at spacings of 3.6 hw and more over the whole scan. If the scan returns any candidate, it is one that spends time inside 3.6 hw | P1 to P3 | as above |
| P-CD | Claude Desktop's guess (plan, no source): unequal frequencies make the force average near zero, so unequal lumps neither bind nor merge quickly; orbital motion is the more likely route. My reading: the first half holds in the model; "neither merge quickly" holds at N ratios of 1.5 and above (P2); "orbital motion is the more likely route" fails in the model (P1, P3) | as above | first half published (T7); "orbits are the likelier route" has no source for a real field |

Scored after the scan from the scan's summary file. A miss is recorded in `FAILURES.md` and investigated, not rerun.

