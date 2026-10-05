# Session 3 report

Date: 2026-10-02 to 2026-10-05. Plan: `docs/SESSION_03_PLAN.md`, WP0 to WP-F. All work is on `main`; nothing has been pushed. **The SWIRL AWS project was closed on 2026-10-05; the last cloud batch's results were not copied before the closure and are lost to this repo** (section 12). **AWS reported about 125 USD of spend against the 50 USD budget; this ledger estimated 16.7 to 37.4 USD** (section 12).

**Where things are recorded.**
- **Predictions:** `docs/SESSION_03_PREDICTIONS.md`, Blocks P, L1, L3 and P3. Each block was committed before the runs it refers to, and its "What was known" paragraph says what was in hand.
- **Misses:** `docs/FAILURES.md`, F-020 to F-029, each with a literature check.
- **Literature:** `docs/PRIOR_ART.md`.
  - The "Session 3 brief": T0 to T11, notes N31 to N35, searches 120 to 151.
  - Addenda G to J: searches from 152, including the novelty check in Addendum I.
- **Established results:** `docs/FINDINGS.md`, new this session, FND-01 to FND-23.
- **Build notes** with the reviewer's corrections: `docs/WPP_BUILD_NOTES.md`, `docs/WPL_BUILD_NOTES.md`.
- **Resume file:** `docs/SESSION_03_STATUS.md`, which holds every decision and deviation with its time.
- **Result files:** under `outputs/session3/` in git.
  - The main checkout's `outputs/` is a link to an external drive and does not hold them.
  - They are in the worktree `.claude/worktrees/integration` and in the commits (section 11).

**Terms used below.** As in the Session 2 report, plus these:
- **dN/N:** the fractional change a disturbance makes in a lump's N, at t = 0 unless "settled". **Settled** means averaged over whole periods 750 to 2,000 time units after the disturbance, as a ratio to the undisturbed control.
- **Broken outright:** within 2,000 time units, the energy inside 3 half-widths falls below half its value just before the disturbance, while the control stays above 0.9 (plan definition).
- **Starts:**
  - **dip:** beta = 4/3, omega 0.9065, N/N_min 1.83, inside the long-life dip;
  - **primary:** beta = 4/3, omega 0.9091, N/N_min 1.754, the Session 1 primary, 4% below the dip in N;
  - **b5:** beta = 5, omega 0.9756, N/N_min 1.97.
- **Family rule:** after a disturbance relaxes, the lump behaves like the undisturbed lump of the same N.
- **Bounded, beating:** a group that meets every clause of the corrected bonded rule except phase locking, its phase steps drifting at the lumps' frequency difference. A 3D candidate, never a bonded claim.
- **Two out:** a collision after which two lumps recede. For identical lumps at phase 0 or 180, bounce and pass-through are the same field by symmetry.

## Verdict

**In the radial (spherical) solver, a lone swirl takes a large disturbance to break outright: about 40% of its energy.**
- In the radial solver, kicks, core removal and incoming radiation were scanned at three starts (384 runs, every threshold bisected).
- A lump breaks within 2,000 time units only when the disturbance leaves it unable to settle at or above the bottom of its family.
- **Measured in energy delivered, the three kick types agree to 4.5% or better:** 40 to 42% of the energy at the dip, 38 to 39% at the primary, 47% at beta = 5. Measured in charge they spread more: 38 to 49% of N removed. The energy reading was found after the scan (post hoc, F-023).
- The last lump that survives settles near N_min, with omega within 1.2% of the omega at the family's minimum.
- Kicks that add N break a lump only beyond a 3.2- to 3.7-fold increase of N. Velocity kicks up to +1.0, and spherical packets up to twice the lump's energy, broke none (these are the largest values tried, not thresholds).
- These thresholds are radial (spherical) results. Their 3D spot checks were lost (section 5).

**Below that threshold a disturbance changes lifetime, and by how much depends on where the lump sits in its family.**
- **The dip start** kept 99.5% of its energy to T = 300,000 in Session 2.
  - After a kick that removes 4% of its N, it dies at 25,023 (field) or 23,367 (velocity).
  - After 2%, it holds 0.69 (field) or 0.60 (velocity) of its energy at T = 30,000, still falling; it was not run to death.
  - Kicks that add 2% raise its loss rate 130 to 320 times at first, then recover toward the family's rate.
- **The Session 1 primary can be improved by a kick:** +4% of N puts it at the dip's N and lowers its early loss rate 150-fold (velocity kick) or 34-fold (field kick). It then sits 8 to 10 times (velocity kick) or 38 to 69 times (field kick) above the dip start's own rate.
- **At beta = 5**, no kick below the threshold raised the loss rate measurably above the noise floor late in the run, and no run died within 30,000.
- **The family rule:**
  - After relaxation, omega follows the family's d omega/dN to within 20% in 84 of 85 runs (reviewer's L1 recomputation; the exception is the b5 slow packet at 0.01 E, ratio 1.28, at the measurement floor). In 3D it does so to 4% for the dip distortions.
  - The loss rate follows the family's within 3x for 7 of 9 comparable kicks in the early window (5 of 9 late; 78% early is under the 80% target even for kicks alone), and apparently not after radiation packets. A diagnostic with family starts matched to 0.33% in N removed that exception: the packet runs follow the family rate within 0.88 to 1.15 (F-024). The family rate changes 5-fold over 0.33% in N near the dip, so the 1% match in the prediction was too loose.

**In 3D at the dip start, non-spherical distortions of up to 2.4% of the energy relax:** no break, no growing shape mode to T = 2,000 (stretch and random multiplier only; no non-spherical threshold was measured). A stretch with no change in N still sheds 0.5 to 1.8% of N. A random multiplier keeps about 37% of the N it adds. Both shift the lump along its family as the omega shift says.

**Collisions (3D, the primary start).** In phase, the lumps merge at both speeds tried, 0.02 and 0.2 per lump: one lump with 0.59 to 0.61 of the pair's energy, still shedding. Anti-phase they come out as two intact lumps. At 90 degrees N flows from the leading to the lagging lump, and two unequal lumps come out. Claude Desktop's "in phase absorbs, anti-phase bounces" holds at both speeds; my prediction that in-phase lumps would not merge at 0.2 was wrong. A diagnostic at 0.4 per lump (0.69c relative) also shows no pass-through to t = 150: the pair rebounds once and stays together, still shedding.

**No group of distinct swirls is bonded in the point model with unequal lumps or orbital motion.**
- 1,822 point-model runs: no group is bonded.
- **Orbital motion does not hold a group** in the model's valid range (3.6 to 8 hw).
  - The radial part follows from the classical Yukawa criterion: circular orbits are unstable for kappa d > 1.618, and kappa d is 5 or more there.
  - The phase and N modes grow in 0.1% to 14% of an orbital period.
- **Unequal lumps beat:** their first-order force averages out.
  - Three weakly interacting unequal pairs at 7.2 hw stayed "bounded, beating" for the run. Adding the omitted second-order force makes them fail the trend clause; run three times longer they drift together by 1 to 10%, though two of them stayed between d = 22.2 and 24.1 to 60,000 in the reviewer's run (stored nowhere).
  - Their larger lumps (N/N_min 3.1 and 4.4) do not hold their N on the run's timescale. In the radial solver they fall below 0.90 of their energy at t = 330 and 558, and their 3D sources failed the settling guard.
  - So the candidates are not static unequal pairs, and they were not run in 3D.

**Against the theory's first claim.**
- **"A modest disturbance can dissipate a lone swirl"** is not supported for an outright break.
- **It is supported in lifetime terms at beta = 4/3 (radial)**, strongly for a lump in a long-life dip: a 4% loss of N (or core removal s = 0.02, dN/N -3.1%) ends a life of over 300,000 time units by 23,000 to 29,000; a 2% loss leaves it at 0.60 to 0.69 of its energy at 30,000, still falling, not run to death.
- **At beta = 5 it is not supported.**

Section 10 has the details.

## 1. Housekeeping (WP0)

- `docs/FINDINGS.md`: 16 entries from Sessions 1 and 2. The runner drafted them, and I wrote the file, because the harness stops subagents from creating report files.
- **Corrected bonded rule** in `swirl/engine/bonded.py`, with tests. The far-apart control E14 now reads "no interaction" and E11's first run "not bonded"; the old rule called both "bonded".
  - Before the P2 scan, the phase clause was fixed for moving groups (the effective step D_eff) and for unequal frequencies ("locked", otherwise "bounded, beating"). Block P records this.
- **Lump finder:** an absolute floor of 0.05 of the first window's maximum. E3, E7 and E4 now end "merged". E1, E2, E14 and a lone lump are unchanged.
- `E_retained_end` is documented as residence in the group sphere and renamed `E_residence_end`.

## 2. Prior art (WP-S)

`docs/PRIOR_ART.md`, Session 3 brief, and Addenda G and H.
- **No published break threshold** for a settled real 3D oscillon, in any variable. The scholar derived a ceiling (N31): the imposed N removal that breaks a lump is below 1 - N_min/N, that is 0.45, 0.43 and 0.49 for the three starts. In energy form the ceiling holds at all nine measured thresholds. In t = 0 dN/N it is exceeded by 0.009 to 0.015 for field kicks and core removal at 4/3, because those lumps regain N as they settle (F-023).
- **The sign of a kick should matter at a dip (N32):** removing N takes a lump out of the dip for good; adding N leaves it above the dip. Measured: yes in the late windows, no in the early ones (section 4).
- **Collisions:** for identical lumps at slow speed, in phase merges and anti-phase bounces (Amin & Mou 2020; Xue 2025). Pass-through at high speed is published in 2D and for Q-balls.
- **Unequal frequencies:** the first-order force averages out (Axenides 2000; Battye & Sutcliffe 2000). A second-order phase-independent attraction binds or spirals NLS pairs (Al Khawaja 2010; Buryak 1998). The point model lacks it (N34). The reviewer sized it: net of the sextic term, about 0.03 of the in-phase first-order force at 4.8 hw and 0.0008 at 7.2 hw.
- **Orbiting pairs:** none published for a real field. Every published stable orbit needs a second component, nonlocality or a trap.
- **The held second field (literature only):**
  - Q-balls of a complex field are absolutely stable above a critical charge, classically stable but metastable between two critical charges, and unstable on the upper branch.
  - A complex partner of our lumps (E/(mN) about 0.975, below 1) would probably be absolutely stable. That conflicts with "a lone swirl is metastable"; this is the scholar's inference (N35).
  - A two-field model with co-located unequal-frequency oscillons exists (Murai, Ogawa, Takahashi 2026). No source found for stepped-phase clusters whose lone lumps are metastable.

## 3. Binding search in the point model (WP-P1, P2)

### Build (P1)

- **Model.**
  - Each lump has its own N, omega, tail amplitude A and kappa along the beta = 4/3 family (39 profiles, omega 0.850 to 0.945).
  - The pair law for unequal lumps: U = -(2 pi/d)[w_1 Q_1(kappa_2) A_2 e^{-kappa_2 d} + w_2 Q_2(kappa_1) A_1 e^{-kappa_1 d}] cos D, with w_i = Omega_i/(Omega_1 + Omega_2).
  - It reduces exactly to the Session 2 law for equal lumps. The reviewer re-derived it and checked it by quadrature to 6.3e-5.
  - Moving lumps have inertia gamma E and a phase that runs at Omega/gamma, with a boost phase.
  - Conservation laws and time reversal hold to 1e-9 or better.
- **Regression on the 15 Session 2 3D pairs:**
  - outcome class 15 of 15;
  - in-phase contact time 0.8%;
  - d(t) above target at d20_45 (1.80x) and d16_90 (1.28x), passes with a note.
  - The reviewer notes that the boost form was chosen on the same two contact times the regression then counts as a pass, so that row is not out of sample.
- **The structural result.**
  - For an orbiting pair, (lambda_phase/Omega_orb)^2 = |d omega/dN| E d/(kappa + 1/d), independent of the coupling strength: 2.09 at 4.8 hw.
  - Phase e-folding time over orbital period: 0.14 at 3.6 hw falling to 0.08 at 8 hw.
  - Rings: fastest e-fold 0.001 to 0.10 of a period.
  - Circular orbits under the tail law are radially unstable for kappa d > (1 + sqrt 5)/2 (READ in Aguirregabiria, Hernandez & Rivas, physics/0407049, eq. 30).
  - The reviewer recomputed two rows and the threshold, and found no stable rotating balance under boost and coupling variations.

### Scan (P2)

1,822 runs, each with a far-apart control, T up to 10,000, starting perturbation 1e-3.

| Family | Runs | Merged | Flew apart | Left the family's range | Bounded for the run | Bonded | Bounded, beating |
|---|---|---|---|---|---|---|---|
| Equal lumps in orbit | 672 | 176 | 446 | 44 | 6 | 0 | 0 |
| Unequal static pairs | 250 | 62 | 78 | 23 | 87 | 0 | 3 |
| Unequal static rings | 312 | 148 | 46 | 52 | 66 | 0 | 0 |
| Unequal orbiting pairs | 252 | 7 | 240 | 5 | 0 | 0 | 0 |
| Alternating-size rings | 336 | 50 | 271 | 12 | 3 | 0 | 0 |

798 of the 1,822 runs come closer than 3.6 hw at some time, where the model is not valid (this count includes the five added by the reviewer's dense-output reflag).

### Against the predictions (Block P)

| # | Prediction | Measured | Outcome |
|---|---|---|---|
| P1 | Equal lumps in orbit: none bonded; at least 90% at 3.6 to 6 hw merge or fly apart within 2 orbital periods | none bonded; 418 of 448 (93.3%); the other 30 leave the family's range within 2 periods | hit |
| P2 | Static unequal pairs and rings: none bonded or bounded-beating; at N ratio 1.5 and above, contact at least 3x later than equal in-phase pairs, or not within the cap | none bonded; 3 bounded-beating pairs at 7.2 hw; at 3.6 hw contact no later than for equal pairs (0.76 to 0.97 of it) | **miss** on both clauses (F-020, F-021) |
| P3 | Unequal orbiting pairs and alternating-size rings: none bonded or bounded-beating | none | hit |
| P4 | Zero bonded and zero bounded-beating at 3.6 hw and above | 3 bounded-beating, all at 7.2 hw | **miss** (F-022; the same three runs as F-020) |
| P-CD | Claude Desktop: unequal frequencies average the force to near zero, so unequal lumps neither bind nor merge quickly; orbital motion is the more likely route | first half holds; "neither merge quickly" fails at 3.6 hw (where contact is itself an extrapolation at the model's validity edge); "orbital motion is the more likely route" fails in the model | one clause of three |

### The misses

**F-020 and F-022: three "bounded, beating" pairs.** Runs p2_0230 and p2_0234 (N ratio 1.75, start phase 0 and 180) and p2_0242 (ratio 2.5, phase 90), all at 7.2 hw.
- **Expected:** none.
- **Measured:** they meet every clause except phase locking. They are marginal on two clauses: distance trend 2.2 to 2.6e-7 against a limit of 3.3e-7, and N exchange 3.4 to 8.3e-4 against 1e-3.
- **Diagnostics run:**
  - D1: with the omitted phase-independent force added (as a diagnostic, not a model change), all three fail the trend clause by 4 to 6 times, closing by 0.65 to 0.87% over the run.
  - D2: run three times longer without it, they stay bounded but drift (p2_0242 by 10%).
  - The reviewer ran them to 60,000: p2_0230 and p2_0234 swing between d = 22.2 and 24.1. p2_0242 closes to 19.4 and then leaves the family's range. Those runs printed their numbers and stored nothing, so they cannot be rechecked from files.
- **Causes, ranked as in the ledger:**
  1. The run length: the label is a threshold crossing within a run of exactly five contact units; longer runs leave it.
  2. The model omits the second-order force, which at 7.2 hw is as large as the model's own residual (D1).
  3. A start-phase-dependent mean drift that happens to cancel: the scholar's fit |A + B sin(start phase)| predicts the 90-degree value to 2% for N ratios 1.75 to 2.5, but fails at 2.95 (0.9 predicted, 9.1 measured). Derived, untested.
- **Rules in:** "bounded, beating" in the point model is a run-length and model-content artefact at this spacing. **Rules out:** the label as evidence of binding. It does not rule out binding in 3D; without N34, two of the pairs stay bounded to 60,000 in the model.
- **3D:** the candidates' larger lumps do not hold their N on the run's timescale (radial and 3D), so the candidates were not run in 3D (section 6).

**F-021: no delay of contact at 3.6 hw.**
- **Expected:** for N ratio 1.5 and above, contact at least 3 times later than equal in-phase pairs.
- **Measured:** 0.96 to 0.97 of it, and 0.76 to 0.88 at ratios 2.5 and 2.95. All 25 runs at 3.6 hw merge at the edge of the model's validity, so the contact time is itself an extrapolation (ledger cause 2).
- **Diagnostic D3:** at 3.6 hw, contact takes 0.07 to 0.24 of a beat period. The relative phase has turned only 52 to 93 degrees by merger, so the pair merges like an equal in-phase pair.
- At 4.2 and 4.8 hw, pairs with ratios 1.2 to 2 do not merge, but some started in phase flew apart rather than drifting together (ratio 1.5 at 4.8 hw; ratios 1.5 and 1.75 at 4.2 hw). So the mechanism I wrote ("drifts together on the beat-averaged attraction") is wrong even where the outcome matched. The reviewer showed the held-position average is not a sound predictor: free dynamics can push the pair apart.
- **Rules in:** the ratio of interaction time to beat period decides whether unequal lumps behave as coherent. **Rules out:** a slow beat-averaged attraction as the general rule.
- 3D does not test this: P3 runs at 7.2 hw only.

## 4. Lone-lump break test, radial (WP-L1)

Numbers from `outputs/session3/wpL/scan_summary.json` as corrected, and the reviewer's recomputation from the raw runs. The reviewer agrees with every break flag, bracket and 30,000-unit value.

### Build and checks

- **Disturbances:**
  - (a) field or velocity kicks of 1 + eps inside 2 to 3 hw, tapered;
  - (b) core removal, scaling the field inside 1 to 1.5 hw by 1 - s (s is not a fraction of N: s = 0.2 gives dN/N = -0.28);
  - (c) ingoing spherical packets, at k = 0.3 and near k3 = sqrt(9 omega^2 - 1).
- **Settled starts:** at t_s of about 1,000.
- **t = 0 values:** dN and dE against independent quadrature, to 1.5e-8 and 9.1e-6. The reviewer recomputed four rows to 3.4e-6 or better.
- **Conservation and checks:**
  - E + absorbed to 6.3e-6;
  - eps = 0 identical to the control;
  - linear response 1.99977 against 2;
  - standing-mode N check 2.4e-8 against 1e-8, a pass with a note.
- **Doubled box:** at most 1.8e-4 while the lump lives in frac (dip field -0.02); death times agree to 5e-4.
- **Regime:**

  | | omega/m | nonlinear share at start (abs) | mean over run |
  |---|---|---|---|
  | dip | 0.907 | 0.36 | 0.15 |
  | primary | 0.910 | 0.35 | — |
  | b5 | 0.976 | 0.15 | 0.06 |

  After field kicks of +0.2 the share is 0.48 (4/3) and 0.22 (b5); at the positive break thresholds it is 0.75 (dip), 0.80 (primary) and 0.53 (b5), deep in the nonlinear regime.
- **Linear twins:** energy inside 3 hw is 0.016 to 0.025 at t_d + 100 (4/3) and 0.0011 to 0.0016 at t_d + 1,000 (b5).

### Break thresholds (radial solver, spherical symmetry)

Bisected to a bracket width of 0.01 in dN/N or better. All dynamic: the first fall below half comes 106 to 1,346 time units after the kick.

| Start | Field kick, eps (t = 0 dN/N; dE/E) | Velocity kick, eps (dN/N = dE/E) | Core removal, s (dN/N; dE/E) | Settled N/N_min at the edge |
|---|---|---|---|---|
| dip | -0.272 (-0.467; -0.415) | -0.234 (-0.411) | 0.366 (-0.469; -0.398) | 1.00 to 1.01 |
| primary | -0.253 (-0.439; -0.390) | -0.216 (-0.382) | 0.347 (-0.445; -0.377) | 1.01 to 1.02 |
| b5 | -0.284 (-0.485; -0.472) | -0.272 (-0.467) | 0.378 (-0.486; -0.468) | 1.015 to 1.03 |

- **Positive field kicks** break at eps of about +0.81 (dip), +0.83 (primary) and +0.93 (b5), that is dN/N +2.2 to +2.7 and dE/E +3.9 to +5.1, deep in the nonlinear regime (share 0.53 to 0.80).
- **Positive velocity kicks** up to +1.0 (dN/N +3.0) and **packets** up to twice the lump's energy never break a lump.

### Degradation (T = 30,000)

- **The dip start.**
  - The control keeps frac 0.99999 at 30,000.
  - Every negative kick of 4% or more in N, every removal, and the k3 packets at 0.2 and 0.5 E die before 30,000: field -0.02 (dN/N -4%) at 25,023, velocity -0.02 at 23,367, removal s = 0.2 at 10,066. The -2% kicks are alive at 30,000 at 0.69 (field) and 0.60 (velocity), still falling.
  - Positive kicks and slow packets survive.
- **The primary** (whose control dies at 22,932): every negative kick and every k3 packet dies earlier, at 9,510 to 21,271. Positive kicks of +0.02 and +0.05 land at or near the dip and survive.
- **b5:** every run survives. Kick rates are at most 5.4e-9 early and within noise late; the noise floor is about 5e-10.
- **Rate after kick over the control's** (t_d + 500 to 2,000, |settled dN/N| 0.02 to 0.20):
  - dip: 316 to 1.6e4;
  - primary: 0.0066 to 13.9;
  - the primary's low values are the +0.02 kicks landing in the dip.

### Against the predictions (Block L1)

| # | Prediction | Measured | Outcome |
|---|---|---|---|
| L1-1 | Break by losing N: threshold between -0.30 and -0.45 (dip), -0.28 and -0.42 (primary); field eps -0.15 to -0.25 | dip -0.467 / -0.411 / -0.469; primary -0.439 / -0.382 / -0.445; eps -0.272, -0.253 | **miss as written** (field and removal beyond the range by 0.017 to 0.025; velocity inside). Within 1.06x of the band edge: a pass with a note under the CLAUDE.md 3x rule (F-023) |
| L1-2 | b5 hardest to break: beyond -0.45 and beyond the 4/3 thresholds | -0.485 / -0.467 / -0.486; beyond the dip by 0.018, 0.056, 0.018 | hit |
| L1-3 | No break from positive kicks up to +0.5 | none | hit |
| L1-4 | No break from packets up to 0.5 E | none up to 2 E | hit |
| L1-5 | Family rule, rate: at least 80% of comparable runs within 3x | 8 of 15 early, 7 of 15 late; kicks 7 of 9; packets mostly 4 to 31x | **miss** (F-024). The 1% N match allows up to about 190x at the dip, so the test is partly unsound as designed |
| L1-6 | Family rule, omega: within 20% for \|dN/N\| up to 0.1 | 84 of 85 (reviewer's recomputation; exception b5 slow packet at 0.01 E, 1.28, at the floor) | hit |
| L1-7 | Sign asymmetry at the dip: late rate after -0.01 and -0.02 at least 100x the control and at least 10x the matching positive kick | late rate over the control 3,229 to 11,451 (reviewer, 1000-unit windows; 3,900 to 8,000 from the summary's late windows); late negative over positive 20 to 465 (reviewer, latest window with both alive) or 34 to 57 (summary's late windows) | hit as written. Early-window ratios to the control 188 to 1,830; early asymmetry only 1.4 to 1.5: positive kicks degrade at first too |
| L1-8 | Ordering: dip 100 to 1e4; primary 0.3 to 30; b5 below 1e-7 absolute | dip 316 to 1.6e4; primary 0.0066 to 13.9; b5 kicks at most 5.4e-9, slow packet 1.5e-7 | **miss** (F-025). Dip and b5 within 3x of the target; the primary's lower edge 45x out |
| L1-9 | k3 packets: primary/dip \|dN/N\| at least 2 | 36, 7.1, 2.6 at 0.01 to 0.05 E; 1.07 to 1.73 above | **miss** (F-026); within 1.9x of the target |
| L1-10 | Linear twins disperse | yes | hit |
| CD | Claude Desktop: the dip start is the most fragile in lifetime terms; b5 the most robust | dip: yes in lifetime (radial): a 4% loss ends its life by about 25,000; 2% leaves 0.60 to 0.69 at 30,000; b5: yes, no break below N_min and no measurable degradation from kicks | agrees |

### The misses

**F-023 (L1-1): the threshold is set by energy, not N.**
- In t = 0 dE/E the three kick types agree to 4.5% at each start; in dN/N they spread 14 to 16% at 4/3.
- Field kicks and core removal at 4/3 regain N as they settle (+0.016 to +0.021); velocity kicks shed it (-0.034 to -0.036).
- My basis ("relaxation sheds more N, so the threshold sits inside the ceiling") had the sign wrong for two of the three kick types. The Block L1 build numbers already showed it: relaxed slope 1.908 eps against 1.986 at t = 0.
- **Rules in:** a lump survives as long as it can settle at or above the bottom of its family.
- **Rules out:** a threshold fixed by imposed N alone.
- **Literature:** decay on reaching the family's end is published for slow radiative evolution (Saffin & Tranberg 2006; Salmi & Hindmarsh 2012). No source found for a kick threshold of a settled real lump.

**F-024 (L1-5): the family rule for the rate.**
- Kicks follow it (7 of 9 early).
- Packets at almost unchanged N have 4 to 31 times the family rate, except two (dip slow 0.5 E; b5 k3 late).
- **Ranked causes:**
  1. The packets' own N shifts (-0.07 to -0.45%) move the family rate 1.4 to 10x at the dip, where the family rate changes about 2x per 0.13% in N.
  2. A lasting internal excitation, of the kind published for Q-balls and 1+1D lumps (Ciurla 2024; Barashenkov & Oxtoby 2009).
- The two are not separated. The late clause compared a path average with a point value.
- **Diagnostic run:** family starts at N/N_min 1.821, 1.824, 1.827. Against them the three packet runs are 0.88 to 1.15 of the family rate. The excess was the family flank at the packets' settled N (cause 1), not a lasting excitation. The miss is in the test's N matching.

**F-025 (L1-8): ordering.**
- The primary's lower edge: +0.02 kicks put it at N/N_min 1.831 to 1.839, the dip's centre. N32 was applied to the dip start only, though it applies to any start within a few percent of the dip.
- The dip's upper edge is 1.6x over (within 3x), and the dip controls themselves differ 1.9x with the kick time.

**F-026 (L1-9): k3 packets.**
- The dip's response grows as efrac^3 at small amplitude and efrac^1.3 at 0.2 to 0.5 E. The primary's grows as efrac^0.7 to efrac^1.3.
- N33's argument (the dip's 3 omega source vanishes) holds at small amplitude. **Diagnostic run:** packets at 0.03 to 0.3 E give primary/dip ratios 4.14, 3.10, 2.58, 2.06, 1.50, 1.27; the ratio crosses 2 at about 0.075 E, where the dip lump's own |dN/N| is about 40% of the dip's width. Consistent with self-detuning out of the dip; fixed packet energy and fixed |dN/N| are not separated.
- **Literature:** no source found for nonlinear absorption of a strong wave by an oscillon.

## 5. 3D: distortions, collisions, spot checks (WP-L3)

**What ran.** Only the reduced set on the Mac (2026-10-02 17:42 to 2026-10-03 12:04): the primary gate on the production grid, six primary collisions, the dip source, five dip distortion runs (control, stretch eps = 0.1 and 0.2, random multiplier a = 0.1 and 0.15 with seed 1), and two linear twins.

**Lost to this repo** with the AWS closure (section 12): the four β = 5 distortion runs and the three spot checks (X); the dip gates, six dip collisions, their twin, box and resolution checks (Z2). The instances synced to S3 every 5 minutes; the bucket's state after the closure is unknown. **Never run** (dropped in the budget trim or not in Aaron's approved items): random seeds 2 and 3, stretch eps 0.05, random a = 0.02 and 0.05, all primary (d) runs, the b5 1.5x box check and the b5 collisions.

**frac in 3D** is E(r < 3 hw) over the control's t = 0 sphere energy (README), so a run's delivered energy is included. Normalised by each run's own t = 0 energy, the differences from the control are stretch -6.0e-3 and -2.3e-2, random -6.5e-3 and -1.5e-2 (the random sign flips). The scores below use the definition as written.

Results: `outputs/session3/wpL3/results/` and `summary.json` (commit 977e1b2).

**Review.** The reviewer recomputed every outcome from the stored series (section 5 of the status file).
- Every stored number is reproduced.
- Four summary readings are corrected:
  - the slow 90-degree collision is two out, not a merge;
  - the fast 180- and 90-degree "contact" times are the lumps entering the sponge;
  - speeds and energies "at the end" were read inside the sponge;
  - the collision twin's 0.12 was measured around the origin, not around the lumps.

### Gate on the production grid (primary source, dx 0.55)

| Speed, start phase | Speed error | E/(gamma E0) - 1 | omega_lab/(omega/gamma) - 1 |
|---|---|---|---|
| 0.02, 0 | -4.0e-4 | 2.3e-5 | -1.0e-6 |
| 0.02, 90 | -2.0e-6 | 5.2e-6 | 3.5e-7 |
| 0.2, 0 | -3.0e-5 | 1.3e-5 | 1.7e-6 |
| 0.2, 90 | -6.1e-6 | 7.9e-6 | -4.4e-6 |

- Targets are 1e-2, 1e-4 and 1e-4.
- The t = 0 cross terms of the six collisions are at most 4.8e-5.
- E + absorbed is conserved to 8.3e-6 or better in every 3D run.

### Distortions at the dip (T = 2,000)

| Run | dE/E at t = 0 | frac(2000) - control | Settled dN/N | omega shift | a2: first, end | Late loss rate |
|---|---|---|---|---|---|---|
| control | 0 | 0 | 0 | 0 | 0 | 6e-9 |
| stretch 0.1 | +1.8e-3 | -4.4e-3 | -0.49% | +3.0e-4 | 0.087, 2.8e-5 | 6.5e-8 |
| stretch 0.2 | +6.4e-3 | -1.75e-2 | -1.81% | +1.12e-3 | 0.171, 7.4e-5 | 6.1e-7 |
| random 0.1 | +1.0e-2 | +2.9e-3 | +0.36% | -2.2e-4 | 0.077, 3.1e-5 | 4.5e-8 |
| random 0.15 | +2.4e-2 | +7.3e-3 | +0.88% | -5.3e-4 | 0.114, 3.7e-5 | about 2.3e-7 |

Notes on the table:
- The first a2 is the raw t = 0 value. The a2 decays roughly as a power law, to 3e-5 by t = 2,000.
- The random runs' a3 decays from 0.03 to 0.04 down to under 1e-4.
- The late loss rate is fitted over t = 1,000 to 2,000.
- No run breaks, and no shape mode grows. The stretch runs' a3 is round-off (1e-10 to 1e-8) and rises slowly, as the control's does at 1e-14.
- The frac changes are explained to about 1e-3 by dE = omega x (settled dN). The omega shifts follow the family's d omega/dN (-2.07e-5) to 4%.
- The stretch sheds 3.3 times the energy it delivers, by t = 300.
- The random multiplier keeps 36 to 39% of the N it adds.

**Regime.** At the start, omega/m is 0.906 to 0.908. The nonlinear share is 0.36 instantaneous (0.34 to 0.35 for the random runs), and its period mean is 0.15.

**Standards not met for the 3D runs:** no box doubling for any (d) run or collision; linear twins exist for one of the five (d) runs and one of the six collisions.

### Collisions at the primary start (d = 30)

| Speed per lump, phase | Outcome | Contact (free estimate) | Closest approach | Afterwards |
|---|---|---|---|---|
| 0.02, 0 | merged | 393.6 (582.8) | — | one lump, 0.614 of 2 gamma E0, omega 0.8967, still shedding (4e-5 E0 per time unit) |
| 0.02, 180 | two out, no contact | — | 16.8 = 5.02 hw at t = 401 | two intact lumps at 0.998 of start energy, speed 0.0200, N equal to 1e-5 |
| 0.02, 90 | two out, unequal, no contact | — | 14.35 = 4.29 hw at t = 425 | N flows from leader to lagger at 4.3 to 5 hw; asymmetry 0.115; lumps +10% and -12% in N, speeds 0.031 and 0.037 |
| 0.2, 0 | merged | 56.0 (58.3) | — | one lump at rest, 0.593 of 2 gamma E0, omega 0.8975, still shedding; about 32% of the energy radiated at the collision |
| 0.2, 180 | two out | — | 7.50 = 2.24 hw at t = 63 | intact, 0.996 of start energy, speed 0.1975 |
| 0.2, 90 | two out, unequal | 63.0 (1.97 hw) | 6.60 = 1.97 hw | asymmetry 0.35; lumps 1.31 and 0.63 of start energy, speeds 0.143 and 0.294 |

- The merged remnants are near-equilibrium lumps at N/N_min of about 2.2: E = 3485 against 3477 on the family at that N.
- The nonlinear share peaks at 0.43 to 0.46 in the merges.
- The linear twins disperse: energy inside 3 hw at t = 100 is 0.021 (distortion) and 0.016 to 0.033 (collision, depending on the centre used).
- **Not validated:** no box or resolution check exists for any collision (lost). These are single runs.

### Against the predictions (Block L3)

| # | Prediction | Measured | Outcome |
|---|---|---|---|
| L3-1 | No shape instability; a2 and a3 below t = 0 values or 1e-3; no break | dip: no break, a2 decays to 3e-5, random-run a3 to under 1e-4; stretch-run a3 is round-off rising from 3e-11 to 4e-10 (and 5e-10 to 3e-8) | **partial**: no break, a2 and the random-run a3 as predicted; the stretch a3 clause fails as written (round-off, read as numerical); b5 part lost; primary part never run |
| L3-2 | Stretch: frac within 1e-3 (eps 0.1) and 1e-2 (eps 0.2) of the control | -4.4e-3 and -1.75e-2 | **miss** (F-027): eps 0.1 is 4.4x the target; eps 0.2 is 1.75x (a pass with a note by the 3x rule) |
| L3-3 | Random multiplier at the dip: frac within 2e-3 of the control; omega shift to 30% | +2.9e-3 and +7.3e-3; omega to 4% | **miss** on frac (F-028): a = 0.15 is 3.7x the target; a = 0.1 is 1.45x (a pass with a note). Hit on omega |
| L3-4 | Slow, phase 0: merged, contact before the free meeting time, remnant 0.6 to 0.85, omega below the start's | merged; 393.6 before 582.8; 0.614; 0.8967 against 0.9095 | hit (primary start, not dip) |
| L3-5 | Slow, phase 180: two out, no contact, closest approach 4 to 6 hw, N equal, speed near 0.02 | 5.02 hw; N equal to 1e-5; 0.0200 | hit |
| L3-6 | Slow, phase 90: two out, unequal, asymmetry above 0.10 | no contact; asymmetry 0.115; leader to lagger | hit |
| L3-7 | Fast: two out at every phase, no merger | phase 0 merged; phase 180 turned at 2.24 hw; phase 90 two unequal | **miss** at phase 0 (F-029) |
| L3-8 | 3D spot checks match the radial runs to 1e-3 | — | **lost** |
| L3-9 | Gate on the production grid passes | passes (primary source; the dip gate passed on Z2 but its file is lost) | hit |
| L3-10 | Linear twins disperse | 0.021; 0.016 to 0.033 | hit |
| L3-11 | Box and resolution checks of collisions agree | — | **lost** |
| CD | Claude Desktop: in phase absorbs, anti-phase bounces | holds at both speeds | agrees |

### The misses

**F-027 (L3-2).**
- **Expected:** a stretch with no change in N sheds only the energy it delivers.
- **Measured:** it sheds 3.3 times that, and 0.5 to 1.8% of N, within about 60 time units.
- **Reading:** the lump does not return to the same family member; it settles lower in N.
- **Rules in:** shape excitation costs N as well as energy.
- **Rules out:** that a zero-dN distortion is free for a lump in a dip. The late loss rates rise to 6.5e-8 and 6.1e-7 (11 and 100 times the control's), which still imply lives of order 1e6; the 3D control sits 0.6% below the radial dip start in N, so the radial lifetimes do not transfer directly.

**F-028 (L3-3).**
- **Expected:** frac within 2e-3 of the control.
- **Measured:** the frac excess is the N the lump keeps (36 to 39% of +a^2), times omega.
- **Reading:** my band ignored that frac counts the energy kept. The omega clause of the same prediction held to 4%.
- **Rules in:** the family rule holds in 3D for generic perturbations.

**F-029 (L3-7).**
- **Expected:** at 0.2 per lump, in-phase lumps would not merge, because the kinetic energy (116) is comparable to the tail-law overlap energy at 2 hw (186). My prediction text says "a third": that is an arithmetic slip, as 116/186 = 0.62.
- **Measured:** they merge, after radiating about 32% of the energy at the collision.
- **Reading:** the tail law is not valid at contact. The relevant comparison is with the energy radiated in the collision, which is several times the kinetic energy.
- **Rules in:** in-phase merger up to at least 0.4 relative speed, for this potential and this lump size. **Diagnostic run** at 0.4 per lump (0.69c relative): no pass-through to t = 150; the pair meets, rebounds to separation 10 and returns to 3.0 to 3.4, with 0.41 of the energy left in the group sphere and falling. One box, one resolution; the long-time remnant is not measured.
- **Literature:** the energy criterion for merging leaves out the friction from radiation and excitation (Cotner 2016). Here the pair radiated about 1,850, which is 16 times the kinetic energy. The published merger statement ("up to 0.1c", Amin & Mou 2020) is the largest speed tried there, not a threshold. Our 0.385c relative is below every published pass-through speed (0.5 per lump in 2D; about 0.6c for 3D Q-balls). The remnant's energy is the same at both speeds (3,437 against 3,485): the extra kinetic energy is radiated (Addendum J).

## 6. 3D: unequal pairs (WP-P3)

**Not run for the candidates, and the planned controls were lost.**
- **The sources failed the settling guard.** Two larger lumps were needed: 1.75 N0 and 2.5 N0, at N/N_min 3.1 and 4.4. Their 3D sources failed the snapshot settling guard (omega drift over 100 time units below 1e-4):
  - at t_s = 300: 1.42e-3 and 8.35e-4;
  - at t_s = 1500 (fix attempt 1): 4.70e-4 and 1.49e-3.
- **A radial check shows this is the lumps, not the 3D preparation.** It reproduces the drift (5.1e-4 and 1.5e-3 at t = 1,500) and the settled omega (0.8954 against 0.8955 in 3D). These lumps shed energy fast: frac 0.72 and 0.58 at t = 2,000; T_90 330 and 558. The dip start under the same check: drift 3.5e-7, frac 0.997.
- **So the candidates are not static unequal pairs.** Over the planned 7,500-unit run their larger lumps would lose a large part of their N. The point model, which treats each lump as a fixed family member, had no way to see this.
- **The second fix attempt was not used.** Relaxing the guard would have run a test whose premise does not hold.
- **The in-phase control** (P3_inphase, for P3-1) and the N0 isolated run ran on Z2. The isolated run finished. Both results are lost.

| # | Prediction | Outcome |
|---|---|---|
| P3-1 | In-phase control merges, contact 800 to 1300 | **lost** |
| P3-2, P3-3, P3-4 | Candidates not bonded; phase beats; no N exchange | **not run**: the premise fails (above). Recorded in F-020 and F-022 |

**What this means for WP-P:**
- The point model's "bounded, beating" candidates are artefacts twice over: of run length and of a model that holds each lump's N fixed.
- A 3D test of unequal binding needs two long-lived family members, for example N/N_min 1.83 and 5.2 at beta = 4/3. That is in the parking lot.

## 7. Hits and misses

- **WP-P (Block P):** hits P1, P3. Misses P2 (both clauses; F-020, F-021) and P4 (F-022). Claude Desktop's guess holds by half.
- **WP-L1 (Block L1):** hits L1-2, L1-3, L1-4, L1-6, L1-7 (as written; late window), L1-10. Misses L1-1 (F-023), L1-5 (F-024), L1-8 (F-025), L1-9 (F-026). L1-1, L1-9 and the dip and b5 clauses of L1-8 are within 3x of their targets (passes with a note by the CLAUDE.md tolerance rule). The primary clause of L1-8 (45x) and the L1-5 packet runs are not. Claude Desktop's fragility ordering agrees.
- **WP-L3 (Block L3):** L3-1 partial; hits L3-4, L3-5, L3-6, L3-9, L3-10. Misses L3-2 (F-027), L3-3 frac clause (F-028), L3-7 phase 0 (F-029). Lost L3-8, L3-11. Claude Desktop's collision guess agrees.
- **WP-P3 (Block P3):** P3-1 lost; P3-2 to P3-4 not run.
- Every miss has a ledger entry with causes, diagnostics and a literature check.

**Cheap diagnostics run for the misses** (2026-10-05, `outputs/session3/diagnostics/`, commit 400d1ec; each is in its ledger entry):

| Entry | Diagnostic | Result |
|---|---|---|
| F-023 | kick shape varied (phase 45 needs a code change; not run) | threshold in t = 0 dE/E -0.408 to -0.414 for six shape and kind combinations (1.5% spread); dN/N spreads 13%: energy is the common coordinate |
| F-024 | family starts matched to 0.33% in N | packets 0.88 to 1.15 of the family rate: no lasting excitation; the miss was the test's N matching |
| F-026 | k3 packets at five more energies | the ratio crosses 2 at about 0.075 E, at about 40% of the dip width in the lump's own dN/N |
| F-027 | field at one point per axis, r = 25 (a shell needs a code change) | extra radiation at omega about 1.07, just above the gap; little at 3 omega |
| F-028 | spectral share of the added N (calculation) | the kept 36 to 39% lies between the kappa and 2/hw cuts; the proposed 1/hw cut gives 5 to 10% |
| F-029 | collision at 0.4 per lump after a gate | no pass-through to t = 150; the pair stays together |

## 8. Novelty

The scholar checked each candidate statement (`PRIOR_ART.md` Addendum I, searches 179 to 212). "No source found" means no source turned up in web search; it is not proof of absence. Nothing here is claimed as new without that qualifier.

| Statement | Wording the record supports |
|---|---|
| Radial: break threshold set by energy; survivor settles near the family's bottom | Partly published: decay on reaching the family's end is published (Saffin & Tranberg 2006; Salmi & Hindmarsh 2012; Ibe 2019). No source found for a kick or removal threshold of a settled real lump, or for the agreement of kick kinds in dE/E |
| Positive kicks and packets up to 2 E do not break | Partly published, qualitatively (Multamaki 2001; Sakai & Tamaki 2012). No source found for the measured limits |
| Radial: a 4% loss of N ends a dip lump's long life by about 25,000; 2% leaves 0.60 to 0.69 at 30,000; adding N recovers | Partly published: lifetime sensitivity near dips for starts (Li 2025; Zhang 2020). No source found for a disturbance applied to a lump already in a dip |
| A kick can move a lump into a dip | Partly published (mechanism). No source found for the measured effect |
| omega follows the family after a disturbance | Published as the picture (Andersen & Tranberg 2012; Ibe 2019); measured here as a consistency check, not new |
| No stable rotating balance in the point model | The radial part is the classical Yukawa criterion (Aguirregabiria et al. 2004); the phase instability of coherent pairs is published for NLS. No source found for the application to oscillon rings |
| Interaction time over beat period decides coherence | Mechanism published (Battye & Sutcliffe 2000; Xue 2025). No source found for the ratio as a criterion |
| Lumps at N/N_min 3.1 and 4.4 at beta = 4/3 do not hold their N | No source found (own result) |
| 3D collisions: in phase merges at 0.02 and 0.2 per lump, anti-phase bounces | Published for slow identical lumps (Amin & Mou 2020; Xue 2025); see F-029 for the speed |

## 9. Workers

Every worker stated its model from its own context; all match `CLAUDE.md`. No worker could see a named effort level. Numeric values seen: numerics 15; reviewer and scholar 30; cloud 10; runner 5 once, otherwise not visible. CLAUDE.md lists high, xhigh, xhigh, high and medium. The mapping is unconfirmed, as in Session 2.

| Worker | Packages | What it did well | What review or I caught |
|---|---|---|---|
| Orchestrator (Opus 5.5) | plan, predictions, judging, merges, report, the final cloud launch | — | my errors are listed below |
| numerics (Opus 5.5) | WP-P1 model, P2 scan and diagnostics, P3 prep; WP-L1/L2 build, L3 prep, fixes | derivations checked by quadrature; the boosted-lump gate; reported its own misses and judgment calls | P: the boost form was tuned on the same contact times used as the regression pass; the scan ran on an uncommitted labeller. L: notes overstated the first-period agreement; checkpoints keyed without code; summary defects |
| runner (Sonnet 5.5) | WP0, L1 scan, Mac L3, radial P3 check | exact reporting, disclosed deviations, found the post-processor's wrong twin value | the primary N/N_min of 1.73 in the summary; a units slip in the radial check |
| reviewer (Sonnet 5.5) | both builds, L1 and L3 recomputations | recomputed everything with its own code; corrected four L3 classifications and the P1 count; sized the omitted second-order force | — |
| scholar (Sonnet 5.5) | brief, Addenda G to J, ledger F-020 to F-029, FINDINGS, novelty | corrected my wording on P3 and the dip, found the energy reading of L1-1, the Yukawa source | — |
| cloud (Sonnet 5.5) | benchmark, cost tables | stopped at the cap and at each permission denial without routing around it | the projection excluded per-instance tests |

**My own errors this session.**
- I wrote L1-1's basis with the sign of the post-kick N change wrong for two of three kick types, although the build numbers in hand already showed it.
- I wrote L3-2 and L3-3 bands that ignored that a distortion changes N, and that frac counts the energy kept.
- I wrote L3-7 from the tail law at contact, where it is not valid.
- I wrote P2's mechanism ("drift together on the beat-averaged attraction") from a held-position average that the reviewer showed is not a predictor.
- I overstated the P3 premise ("these lumps decay in 3D") before the radial check, and the dip result ("a 2 to 4% loss ends its life") before the -2% runs were read. The scholar caught both.
- Git, around the `outputs/` link:
  - I moved `main` once to a commit that lacked a status commit; I restored it within minutes.
  - I wiped a worker's uncommitted ledger edits with a `checkout -- docs`; the worker re-applied them.
  - zsh word-splitting broke two of my commands, one of them an AWS resource-creation step. I checked what had been created before retrying.
- I wrote a wrong time into the status file once, from memory.
- **I never copied the last batch's results from S3 to the Mac while the sign-in was valid.** I planned collection at the end of the batch, and the closure made that impossible. A per-hour copy to the Mac would have saved what had finished.
- The P3 sources were never built before the batch. A source build on the Mac before renting would have shown the settling failure for free.

## 10. What the results mean for claim 1

Claim 1 has two halves: "left alone a swirl persists" and "a modest disturbance can dissipate it".

- **Persists.**
  - Session 2 supported this at beta of 4/3 and above.
  - This session adds the first non-spherical test of the long-lived dip start in 3D. Stretches and random perturbations of up to 2.4% of the energy relax without a growing shape mode to T = 2,000. They leave the lump shifted along its family, in N, omega and loss rate.
  - 3D still covers only 2,000 time units of lumps whose radial lives exceed 300,000.
- **Dissipated by a modest disturbance.** It depends on what "dissipate" means.
  - **As an outright break** (half the energy gone within 2,000), not supported for spherical disturbances (radial): it takes about 40% of the lump's energy at beta = 4/3 and 47% at beta = 5, the amount that leaves the lump unable to settle above the family's minimum; a 2E radiation packet and a tripling of N leave it intact. Non-spherical disturbances were tested only to 2.4% of the energy and T = 2,000 in 3D, with no break.
  - **As a shortened life**, supported at beta = 4/3 (radial) and strongly so for a lump in a long-life dip. There, removing 4% of N turns a life of over 300,000 time units into about 25,000, and 2% turns it into one that has lost 31 to 40% of its energy by 30,000, still falling. The 3D lifetime cost was not measured (the spot checks were lost; the 3D distortions ran 2,000 units).
  - **At beta = 5**, not supported in either sense within the runs made.
  - So at beta = 4/3 the theory's "modest disturbance" is true of lifetime, not of immediate destruction, and its size depends on where the lump sits relative to a dip.
- **Collisions.** In-phase lumps absorb each other at both speeds tried. The result is one lump of about 0.6 of the pair's energy, still shedding, not a bigger stable lump. Anti-phase lumps repel or bounce intact. Collisions are the one disturbance tried that makes two swirls into one, as the theory's "absorption" expects.
- **Claims 2 and 3** concern bonded groups. This session found none, in the point model with unequal lumps or orbital motion. Its candidates turned out to involve lumps that do not hold their N. The first-order exclusion of Session 2 (attraction needs a relative phase that is unstable), the classical instability of Yukawa orbits, and the beat of unequal lumps leave no route to a bonded group of distinct real-field swirls that this project has found. The routes not tested are second-order forces in a full treatment, unequal pairs made of two long-lived family members, and a second field.

**Proposals for what to do next** (for Aaron and Claude Desktop; not decisions):
1. **The 3D lifetime cost of disturbances at the dip**, on the Mac. Extend the four dip distortion runs and their control from 2,000 to 30,000, to see whether a 0.5 to 1.8% loss of N from a distortion shortens the dip lump's life as radial kicks of 2 to 4% did. Cost: about 31 to 36 h per run at 2 threads, about 170 to 190 job-hours for five runs, about 4 days on the Mac running 2 at a time. Checkpoints at T = 2,000 are in `.claude/worktrees/integration/checkpoints/session3/wpL3/` (not in git).
2. **Unequal pairs built from long-lived lumps** (N/N_min 1.83 and 5.2 at beta = 4/3) in 3D at 6 to 7 hw, with isolated controls, on the Mac. This is the one binding route the model left open that this session could not test. The N/N_min 5.2 lump has hw 4.9, so it needs a larger box than N = 160, L = 88 (to be sized); at the measured Mac rate a T = 5,000 pair run is of order 50 to 100 h. Expect the first-order force to average out; the question is the second-order force.
3. **Collision speed scan at phase 0** from 0.4 per lump upward (0.4 does not pass through to t = 150), on the Mac, to find where merging stops, with longer runs to measure the remnant (each short run about 0.5 to 1 h). Also the lost dip collisions with box and resolution checks (about 100 job-hours).
4. **Decide the second-field question** in the light of the scholar's T6: a complex partner field would likely make lone lumps absolutely stable, against "metastable" in claim 1.
5. **Bookkeeping:**
   - Restore a layout in which `outputs/` can be committed from the main checkout.
   - Read billed cost, not only estimate it, before and during any future rented batch.

## 11. Deviations, open issues, parking lot

**Deviations.**
- **Repository layout.**
  - `outputs/` and `checkpoints/` in the main checkout are links to an external drive (made 2026-10-02 11:41), and `git merge` cannot run there.
  - Code merges were made in the worktree `.claude/worktrees/integration`, and `main` was moved with `git update-ref`.
  - The main checkout's `git status` still shows every tracked output file as deleted. The files are on the drive.
- **WP0:** `docs/FINDINGS.md` was written by me from the runner's draft (the harness stops subagents from creating report files).
- **Block P:** the bonded rule's phase clause was settled for moving and unequal groups before the scan. The scan's interaction unit and T cap were the builder's choices.
- **WP-L1:**
  - Collision speeds were set to 0.02 and 0.2 per lump, not the builder's 0.05, to match the published slow range.
  - L1 scoring uses the reviewer's recomputation (primary N/N_min 1.754, not 1.73).
  - The thresholds are radial only.
- **WP-L3:**
  - The full set was trimmed for budget (b5 distortions halved, b5 collisions dropped) before any launch.
  - The relaunch was blocked twice by the permission system. Aaron then had the orchestrator launch it from the main session.
  - Most of it was then lost (section 12).
  - The Mac ran a reduced set that Aaron did not ask for explicitly: the fallback I announced when he had not answered for an hour.
- **P3:**
  - The horizon was fixed at 7,500 to remove a dependency between instances.
  - Isolated single lumps replaced the far-apart control.
  - The candidates were not run (section 6).
- **CLAUDE.md:** rented compute marked unavailable, at Aaron's instruction.
- **Diagnostics:** more than three for some misses (F-020 to F-022 drew on D1 to D3 and the reviewer's runs).

**Open issues.**
- `tests/family/test_profile.py::test_derrick_identity[1.0]` fails on aarch64 (Graviton) with a node in the continuation at omega 0.715. It passes on the Mac and is not used by any run here. Two attempts were not made; it is logged as open.
- No box or resolution check of any 3D collision.
- Two proposed diagnostics need code changes and were not run: a phase-45 kick (F-023) and a field record on a shell (F-027).
- The cloud stop of 2026-10-02 19:35Z is unexplained (no stop call in CloudTrail).
- The spend difference (section 12).
- The AWS resources of the last batch were not deleted by this session.
- Session 2's open issues stand: the 45-degree pair excess, the linear-twin interaction test, the 90-degree ring N exchange.

**Parking lot.**
- The extra packet wavenumbers k2 and k5 (N33).
- b5 collisions.
- A spinning single lump.
- A 0.1% N-matched family comparison for packets (F-024).
- A direct test of the energy-threshold reading with kicks designed to deliver equal energy and unequal N.

## 12. Rented compute

| Batch | Instances | Hours | Estimated cost (USD) | Results on the Mac |
|---|---|---|---|---|
| Sessions 1 and 2 | — | — | 9.69 | yes |
| Benchmark, 2026-10-02 07:50Z | 1 c8g.4xlarge | 0.27 | 0.18 | benchmark records |
| Z attempt 1 (test gate failed) | 1 | 0.9 | 0.6 | test logs |
| P (P3 sources) | 1 | 0.72 | 0.46 | source metadata and logs |
| X and Z2, to the stop at 19:35Z | 2 | 4.9 + 4.0 | 5.7 | **none: lost** |
| X and Z2, resumed 2026-10-03 00:48Z | 2 | unknown; at most 16.1 each to their timers | up to 20.7 | **none: lost** |
| **Ledger estimate** | | | **16.6 to 37.3** (rows; the ledger quotes 16.7 to 37.4) | |

- **AWS reported, per Aaron, 100.17 USD month-to-date, plus about 25 USD after a reactivation: about 125 USD.**
  - That is far above this ledger's estimate and above the 50 USD budget.
  - The difference is unexplained here (`docs/CLOUD_LEDGER.md`, Closure).
  - The ledger estimated from list prices and never read the billed amount. That was my process failure: the 40 USD notice and the 50 USD stop could not work on an estimate that was this far off.
  - By the AWS-reported figure the 40 USD mark and the 50 USD budget were both passed; by this ledger's estimate (at most 37.4) neither was. This is the notice to Aaron that CLAUDE.md requires at 40 USD.
  - Resources left in the account by this session: bucket `swirl-batch3-156084119557-use2`, role `swirl-batch3-role` (inline policy `swirl-batch3-s3`, AmazonSSMManagedInstanceCore attached), instance profile `swirl-batch3-profile`, security group `sg-0a7678933c595a7a4`; possibly the stopped or running instances X and Z2 and their 60 GB volumes. Their state after the closure is unknown here.
- **The project was closed on 2026-10-05.** Lost runs, by instance:
  - **X:** the four b5 distortions, spot_b5_field+0.1, spot_dip_field-0.1 (at t = 1,800 of 2,000 at the resume), spot_dip_removal+0.2, and both sources.
  - **Z2:** the dip gates and gate check (passed, known only from the absence of a .fail file), the six dip collisions and their twin, the box checks at both speeds, the resolution check, P3_iso_N100 (finished) and P3_inphase (state at the stop unknown).
  - No results or checkpoints from X, Z2 or P were copied to the Mac before the closure.
- **Other events.**
  - The permission system blocked the cloud worker twice ("Real-World Transactions"); it stopped each time.
  - The sign-in lasted about 9 hours the first time (12:46 to before 21:35 +07) and lapsed within about 2 hours the third time (07:47 to 09:59 +07); the second lapse time is not recorded.
  - Both instances stopped without explanation after about 5 hours. Aaron chose to resume them, and the sign-in expired again before collection.

## 13. Tests on main

`.venv/bin/python -m pytest` on `main` at commit a38050d (after the merge of the thread fix), run in the main checkout on 2026-10-05. Every commit after it changes only `docs/` and files under `outputs/`.

```
collected 212 items
tests/arrange/test_angular_momentum.py .                                 [  0%]
tests/arrange/test_claude_desktop.py .                                   [  0%]
tests/arrange/test_counting.py ...                                       [  2%]
tests/arrange/test_dynamics.py ...                                       [  3%]
tests/arrange/test_forces.py ...                                         [  5%]
tests/arrange/test_p2_label.py .........                                 [  9%]
tests/arrange/test_two_lump.py ...                                       [ 10%]
tests/arrange/test_unequal.py .............                              [ 16%]
tests/engine/test_bonded.py ...........                                  [ 22%]
tests/engine/test_check1_linear_exact.py ..                              [ 23%]
tests/engine/test_check2_energy.py ...                                   [ 24%]
tests/engine/test_check3_ebind.py ......                                 [ 27%]
tests/engine/test_check4_absorber.py .x....                              [ 30%]
tests/engine/test_checkpoint_resume.py ...                               [ 31%]
tests/engine/test_diagnostics.py .........                               [ 35%]
tests/engine/test_lump_floor.py ..............                           [ 42%]
tests/engine/test_lumps.py .............                                 [ 48%]
tests/engine/test_lumps_k.py .............                               [ 54%]
tests/engine/test_snapshot.py ..x.x.......                               [ 60%]
tests/engine/test_snapshot_multi.py .                                    [ 60%]
tests/engine/test_wpe_postprocess.py .........                           [ 65%]
tests/engine/test_wpl3_driver.py .                                       [ 65%]
tests/engine/test_wpl_initial.py ..........s..                           [ 71%]
tests/family/test_profile.py .......................                     [ 82%]
tests/family/test_radial.py ......                                       [ 85%]
tests/family/test_wpa_pathintegral.py .......                            [ 88%]
tests/family/test_wpa_run.py ......x                                     [ 91%]
tests/family/test_wpl.py .............ss.                                [ 99%]
tests/test_import.py .                                                   [100%]
============ 205 passed, 3 skipped, 4 xfailed in 1698.29s (0:28:18) ============
```

- **The first full run, at 977e1b2, failed: 12 failed, 193 passed, 3 skipped, 4 xfailed.**
  - All 12 failures were in `tests/family/test_wpl.py`, with the same numba error. `swirl/engine/l3.py` and `wpl.scan` set `NUMBA_NUM_THREADS` in the parent pytest process after numba had launched its threads, and the next compile raised.
  - Fixed on the first attempt: the thread count is set only in worker processes (04b10b1). A regression check was added to `tests/engine/test_wpl3_driver.py`.
- **The four expected failures are strict records of misses:** the Session 1 absorber check (F-001), the beta = 0.75 box doubling (F-008), and two WP-E tooling targets. If any of them ever passes, the suite fails.
- **The three skips** are checks that read session3 records (`gate.json`, `box_check.json`, `estimator_check.json`). Those are in git, but not under the main checkout's `outputs/`, which is the external-drive link. In the integration worktree they run.
- **Two open caveats:**
  - `tests/family/test_profile.py::test_derrick_identity[1.0]` fails on aarch64 (Graviton); it passes here (section 11).
  - The suite rewrites some small check records under `outputs/`; in the main checkout these land on the external drive.
