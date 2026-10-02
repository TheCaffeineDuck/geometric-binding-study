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


## Block L1. WP-L1 radial break and degradation scan. Written 2026-10-02 14:00 (+07), before the scan

**What was known when this was written.** The L1/L2 build (branch `worktree-agent-a511a354d573f02da`, commit 4213d63, `docs/WPL_BUILD_NOTES.md` there): the t = 0 table (a field or velocity kick of epsilon changes N by about 2 epsilon: -0.0198 at -0.01, -0.3575 at -0.2, -0.5065 at -0.3; removal fraction s = 0.2 gives -0.28, 0.5 gives -0.59); after relaxation Delta N / N = 1.908 epsilon and omega shifts by -0.117 epsilon; packets of 5% of the lump energy change N by -0.1% to -0.7% (+0.13% for the slow packet at beta = 5); three single demo runs, none broken, with loss rates after the kick of 2.7e-5 (dip, field +0.1), 6.2e-5 (dip, removal 0.2), 1.8e-5 (primary, field +0.1), 8e-9 (beta = 5, field +0.1) against controls of 6e-9, 4.2e-6, 8e-10. The Session 2 lifetime table at beta = 4/3 and 5 (rates and lifetimes against N/N_min). The WP-S brief (T1 to T3, T7, N31 to N33).

**Starts.** dip: beta = 4/3, omega 0.906505, N/N_min 1.83. primary: beta = 4/3, omega 0.9091, N/N_min 1.73. b5: beta = 5, omega 0.975598, N/N_min 1.97. Disturbances are scored by the Delta N / N they impose (the t = 0 value), with Delta E / E as the second number.

**The family rule (the main idea under test).** After a spherical disturbance relaxes, the lump behaves like the undisturbed lump of the family with the same N: same loss rate and omega, because lumps of one beta follow one adiabatic curve (Session 2, F-007 diagnostics: the run's own rates reproduce its lifetime to 1%).

| # | Prediction | Basis | Literature |
|---|---|---|---|
| L1-1 | **Break by losing N.** Negative field kicks, negative velocity kicks and core removal break the 4/3 starts outright (within 2,000) only once the imposed Delta N / N is beyond -0.30. Threshold (midpoint of the bracketing scan values) between -0.30 and -0.45 at the dip, between -0.28 and -0.42 at the primary. In epsilon for field kicks: between -0.15 and -0.25 | A lump at N_min at 4/3 dies at about 1,830 (Session 2 table); N31 ceiling 1 - N_min/N = 0.45 and 0.42; relaxation sheds a little more N | No published threshold for a real oscillon (T0 item 1). Death at the critical charge (Mukaida 2017; Ibe 2019). N31 |
| L1-2 | **beta = 5 is hardest to break outright.** Its threshold lies beyond -0.45 in Delta N / N (beyond the 4/3 thresholds), because at beta = 5 even a lump at N/N_min = 1.0 lives past 30,000; it breaks only when pushed below N_min (ceiling 0.49) | Session 2: b5 starts at N/N_min 1.00 to 1.02 alive at 30,000 with frac 0.97 | Claude Desktop (plan): beta = 5 most robust. No source (T7) |
| L1-3 | **No outright break from positive kicks** of the field or velocity at epsilon up to +0.5, at any start. Above +0.5 (Delta E / E above 1) no prediction | Larger lumps exist and live long at both betas; the excess is shed as radiation | Kicked lumps relax and shed the excess: NLS, Q-balls, 2D oscillons (Carles 2021; Multamaki 2001; Hindmarsh & Salmi 2006) |
| L1-4 | **No outright break from packets** at either wavenumber up to a packet energy of 0.5 E_lump. Above that no prediction | Demo packets of 0.05 E change N by under 1% | Plane waves of a few percent push a lump and leave it intact (Xue 2025); no source for a spherical or tuned packet (T2) |
| L1-5 | **Family rule, rate.** For spherical disturbances with \|Delta N / N\| up to 0.2, the loss rate measured after relaxation (window t_d + 500 to t_d + 2,000, and late windows of the 30,000 runs) equals the undisturbed family's rate at the same settled N within a factor of 3, wherever a family start lies within 1% in N/N_min (the comparison is made only there). Scored per run; hit if at least 80% of the comparable runs pass | one adiabatic curve per beta | Published lifetime estimates assume this path (Ibe 2019 eq. 55; Li 2025 eq. 2.30) |
| L1-6 | **Family rule, omega.** After relaxation the omega shift equals the family's d omega/dN times the settled Delta N, within 20% for \|Delta N / N\| up to 0.1 | linearity run: -0.117 epsilon | as L1-5 |
| L1-7 | **Sign asymmetry at the dip.** For epsilon = -0.01 and -0.02 (Delta N / N = -2%, -4%) the late loss rate is at least 100 times the control's, and at least 10 times the rate after +0.01 and +0.02. Removal leaves the dip for good; addition stays on the high-N side (N/N_min 1.87 to 1.90, family rate about 1e-7) | the dip is 1.7% wide in N; N/N_min 1.80 has rate 1.7e-5, 1.857 has 1.1e-7, the dip start 3.5e-9 | N32 (derived); Li 2025 Fig. 3 (lifetime changes 25 times for a 17% change in energy) |
| L1-8 | **Degradation ordering (rate after over control's rate, at matched \|Delta N / N\| of 2% to 20%).** dip is the most degraded (ratio 100 to 1e4); primary is degraded least in ratio (its control already loses 4.2e-6; ratio 0.3 to 30); b5 keeps an absolute rate below 1e-7 for every \|Delta N / N\| up to 0.2 | Session 2 family rates | Claude Desktop (plan): the dip start is the most fragile in lifetime terms. Sources half agree: the effect depends on the sign (T7) |
| L1-9 | **Packets at k3 couple less to the dip than to the primary.** At equal packet energy, \|Delta N / N\| after the packet is at least 2 times larger for the primary than for the dip | at a dip the lump's 3 omega source nearly vanishes | N33 (derived; Zhang 2020; Ibe 2019) |
| L1-10 | **Linear twins** of the disturbed starts disperse: energy inside 3 hw below 0.05 of its start by t_d + 100 at beta = 4/3 and by t_d + 1,000 at beta = 5 | Session 2 twins: 0.02 at T = 100 (4/3), 2e-3 by T = 1000 (5) | standard 5 |

Scored from the scan's summary file when it is in. A miss goes to `FAILURES.md` and is investigated, not rerun.
