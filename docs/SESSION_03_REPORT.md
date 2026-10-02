# Session 3 report

**DRAFT.** Sections 5 (WP-L3) and 6 (WP-P3) wait for the 3D runs. Sections 9 and 12 (tests, rented compute) are filled in at the end.

Date: 2026-10-02 to 2026-10-03. Plan: `docs/SESSION_03_PLAN.md`, WP0 to WP-F. All work is on `main`; nothing has been pushed.

**Where things are recorded.**
- Predictions: `docs/SESSION_03_PREDICTIONS.md`, Blocks P, L1, L3 and P3. Each block was committed before the runs it refers to, and its "What was known" paragraph says what was in hand.
- Misses: `docs/FAILURES.md`, F-020 to F-026 so far, each with a literature check.
- Literature: `docs/PRIOR_ART.md`, "Session 3 brief" (T0 to T11, notes N31 to N35, searches 120 to 151) and Addenda G and H (searches 152 to 178).
- Established results: `docs/FINDINGS.md`, new this session.
- Build notes with the reviewer's corrections: `docs/WPP_BUILD_NOTES.md`, `docs/WPL_BUILD_NOTES.md`.
- Resume file: `docs/SESSION_03_STATUS.md`.
- Result files: under `outputs/session3/` in git. The main checkout's `outputs/` is a link to an external drive and does not hold them; they are in the worktree `.claude/worktrees/integration` and in the commits (section 11).

**Terms used below.** As in the Session 2 report, plus these:
- **dN/N:** the fractional change a disturbance makes in a lump's N, at t = 0 unless "settled". **Settled** means averaged over whole periods 750 to 2,000 time units after the disturbance, as a ratio to the undisturbed control.
- **Broken outright:** within 2,000 time units, the energy inside 3 half-widths falls below half its value just before the disturbance, while the control stays above 0.9 (plan definition).
- **Starts:**
  - **dip:** beta = 4/3, omega 0.9065, N/N_min 1.83, inside the long-life dip;
  - **primary:** beta = 4/3, omega 0.9091, N/N_min 1.754, the Session 1 primary, 4% below the dip in N;
  - **b5:** beta = 5, omega 0.9756, N/N_min 1.97.
- **Family rule:** after a disturbance relaxes, the lump behaves like the undisturbed lump of the same N.
- **Bounded, beating:** a group that meets every clause of the corrected bonded rule except phase locking, its phase steps drifting at the lumps' frequency difference. A 3D candidate, never a bonded claim.

## Verdict

[To be finalised after the 3D runs.]

**A lone swirl survives any disturbance that leaves it at or above the bottom of its family, and breaks when the disturbance takes it below.** In the radial solver, kicks, removal of N from the core, and incoming radiation were scanned at three starts (384 runs). A lump breaks outright only when the disturbance takes it past the end of its family:
- Measured as charge, the threshold lies at 38 to 49% of N removed.
- Measured as energy, the three kick types agree to 4.5%: 40 to 42% of the energy at the dip, 37 to 39% at the primary, 47% at beta = 5.
- The last lump that survives has settled near N_min, with omega within 1.2% of the omega at the minimum.
- Positive kicks break a lump only when they more than triple its N. Radiation packets up to twice the lump's energy never break it.

**Below that threshold, a disturbance changes lifetime, and how much depends on where the lump sits in its family.**
- The dip start kept 99.5% of its energy to T = 300,000 in Session 2. After losing 2% of its N it holds only 0.69 of its energy at T = 30,000, still falling; after losing 4% it dies at 25,023.
- A kick that adds 2% raises its loss rate 130 to 320 times at first, then recovers toward the family's rate.
- The Session 1 primary can be improved by a kick: +4% of N puts it into the dip and lowers its loss rate up to 150 times.
- At beta = 5 no disturbance below the threshold produced a loss rate measurably above the noise floor for kicks, or a death within 30,000.
- After relaxation, omega follows the family's d omega/dN to within 20% in 84 of 85 runs. The loss rate follows the family's within 3x in 7 of 9 comparable kicks, but not after radiation packets.

**No group of distinct swirls is bonded in the point model with unequal lumps or orbital motion.**
- 1,822 point-model runs of unequal pairs and rings, orbiting pairs and rotating rings gave no bonded group.
- Orbital motion cannot outrun the instabilities: in the valid range (3.6 to 8 hw) every rotating balance has an unstable mode growing in 0.1% to 14% of an orbital period.
- Circular orbits under the tail force are radially unstable wherever kappa d > 1.618, which covers the whole valid range (kappa d is 5 or more). This is the classical Yukawa result (Aguirregabiria et al.).
- Unequal lumps beat: the first-order force averages out, and what binds or separates them is a second-order force the point model does not contain. Three weakly interacting unequal pairs at 7.2 hw stayed "bounded, beating" for the run. Adding the omitted force, or running three times longer, makes them drift together. [3D test: section 6.]

**Against the theory's first claim.** "A modest disturbance can dissipate a lone swirl" is not supported for outright break: it takes the removal of about 40% of the lump's energy, or 38 to 49% of its N, to break one within 2,000 time units. It is supported in lifetime terms at beta = 4/3, where a 4% loss of N shortens the life of the dip start from beyond 300,000 to about 25,000, and a 2% loss leaves it at 0.69 of its energy by 30,000. At beta = 5 it is not supported. Section 10 has the details.

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

[Written from Block P scoring, the build and review notes, F-020 to F-022.]

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

798 of the 1,822 runs come closer than 3.6 hw at some time, where the model is not valid (five more after the reviewer's dense-output reflag).

### Against the predictions (Block P)

| # | Prediction | Measured | Outcome |
|---|---|---|---|
| P1 | Equal lumps in orbit: none bonded; at least 90% at 3.6 to 6 hw merge or fly apart within 2 orbital periods | none bonded; 418 of 448 (93.3%); the other 30 leave the family's range within 2 periods | hit |
| P2 | Static unequal pairs and rings: none bonded or bounded-beating; at N ratio 1.5 and above, contact at least 3x later than equal in-phase pairs, or not within the cap | none bonded; 3 bounded-beating pairs at 7.2 hw; at 3.6 hw contact no later than for equal pairs (0.76 to 0.98 of it) | **miss** on both clauses (F-020, F-021) |
| P3 | Unequal orbiting pairs and alternating-size rings: none bonded or bounded-beating | none | hit |
| P4 | Zero bonded and zero bounded-beating at 3.6 hw and above | 3 bounded-beating, all at 7.2 hw | **miss** (F-022; the same three runs as F-020) |
| P-CD | Claude Desktop: unequal frequencies average the force to near zero, so unequal lumps neither bind nor merge quickly; orbital motion is the more likely route | first half holds; "neither merge quickly" fails at 3.6 hw; "orbital motion is the more likely route" fails in the model | half |

### The misses

**F-020 and F-022: three "bounded, beating" pairs.** Runs p2_0230 and p2_0234 (N ratio 1.75, start phase 0 and 180) and p2_0242 (ratio 2.5, phase 90), all at 7.2 hw.
- **Expected:** none.
- **Measured:** they meet every clause except phase locking. They are marginal on two clauses: distance trend 2.2 to 2.6e-7 against a limit of 3.3e-7, and N exchange 3.4 to 8.3e-4 against 1e-3.
- **Diagnostics run:**
  - D1: with the omitted phase-independent force added (as a diagnostic, not a model change), all three fail the trend clause by 4 to 6 times, closing by 0.65 to 0.87% over the run.
  - D2: run three times longer without it, they stay bounded but drift (p2_0242 by 10%).
  - The reviewer ran them to 60,000: p2_0230 and p2_0234 swing between d = 22.2 and 24.1. p2_0242 closes to 19.4 and leaves the family's range at 17,419. Those runs printed their numbers and stored nothing, so they cannot be rechecked from files.
- **Causes, ranked:**
  1. These are runs where a start-phase-dependent mean drift happens to cancel. The scholar fitted the trend over its limit as |A + B sin(start phase)| for N ratios 1.75 to 2.5, predicting the 90-degree value to 2% (derived, untested).
  2. The model omits the second-order force, which at 7.2 hw is as large as the model's own residual.
- **Rules in:** "bounded, beating" in the point model is a run-length and model-content artefact at this spacing. **Rules out:** a point-model bound state of unequal lumps.
- [3D result: section 6.]

**F-021: no delay of contact at 3.6 hw.**
- **Expected:** for N ratio 1.5 and above, contact at least 3 times later than equal in-phase pairs.
- **Measured:** 0.96 to 0.98 of it, and 0.76 to 0.88 at ratios 2.5 and 2.95.
- **Diagnostic D3:** at 3.6 hw, contact takes 0.07 to 0.24 of a beat period. The relative phase has turned only 52 to 93 degrees by merger, so the pair merges like an equal in-phase pair.
- At 4.2 and 4.8 hw, pairs with ratios 1.2 to 2 do not merge, but ratios 1.5 and 1.75 started in phase flew apart rather than drifting together. So the mechanism I wrote ("drifts together on the beat-averaged attraction") is wrong even where the outcome matched. The reviewer showed the held-position average is not a sound predictor: free dynamics can push the pair apart.
- **Rules in:** the ratio of interaction time to beat period decides whether unequal lumps behave as coherent. **Rules out:** a slow beat-averaged attraction as the general rule.
- 3D does not test this: P3 runs at 7.2 hw only.

## 4. Lone-lump break test, radial (WP-L1)

[Numbers from `outputs/session3/wpL/scan_summary.json` as corrected, and the reviewer's recomputation from the raw runs. The reviewer agrees with every break flag, bracket and 30,000-unit value.]

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

  After field kicks the share reaches 0.48 (4/3) and 0.22 (b5).
- **Linear twins:** energy inside 3 hw is 0.016 to 0.025 at t_d + 100 (4/3) and 0.0011 to 0.0016 at t_d + 1,000 (b5).

### Break thresholds

Bisected to a bracket width of 0.01 in dN/N or better. All dynamic: the first fall below half comes 106 to 1,346 time units after the kick.

| Start | Field kick, eps (t = 0 dN/N; dE/E) | Velocity kick, eps (dN/N = dE/E) | Core removal, s (dN/N; dE/E) | Settled N/N_min at the edge |
|---|---|---|---|---|
| dip | -0.272 (-0.467; -0.415) | -0.234 (-0.411) | 0.366 (-0.469; -0.398) | 1.00 to 1.01 |
| primary | -0.253 (-0.439; -0.390) | -0.216 (-0.382) | 0.347 (-0.445; -0.377) | 1.01 to 1.02 |
| b5 | -0.284 (-0.485; -0.472) | -0.272 (-0.467) | 0.378 (-0.486; -0.468) | 1.015 to 1.03 |

- **Positive field kicks** break at eps of about +0.81 (dip), +0.83 (primary) and +0.93 (b5), that is dN/N +2.2 to +2.7 and dE/E +3.9 to +5.0.
- **Positive velocity kicks** up to +1.0 (dN/N +3.0) and **packets** up to twice the lump's energy never break a lump.

### Degradation (T = 30,000)

- **The dip start.**
  - The control keeps frac 0.99999 at 30,000.
  - Every negative kick, every removal and the k3 packets at 0.2 and 0.5 E die before 30,000. Field -0.02 (dN/N -4%) dies at 25,023; removal s = 0.2 dies at 10,066; field -0.01 (dN/N -2%) is alive at 0.69 and falling.
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
| L1-6 | Family rule, omega: within 20% for \|dN/N\| up to 0.1 | 84 of 85 | hit |
| L1-7 | Sign asymmetry at the dip: late rate after -0.01 and -0.02 at least 100x the control and at least 10x the matching positive kick | 188 to 1,830x the control early; late negative over positive 20 to 465 | hit as written. Early-window asymmetry 1.4 to 1.5: positive kicks degrade at first too |
| L1-8 | Ordering: dip 100 to 1e4; primary 0.3 to 30; b5 below 1e-7 absolute | dip 316 to 1.6e4; primary 0.0066 to 13.9; b5 kicks at most 5.4e-9, slow packet 1.5e-7 | **miss** (F-025). Dip and b5 within 3x of the target; the primary's lower edge 45x out |
| L1-9 | k3 packets: primary/dip \|dN/N\| at least 2 | 36, 7.1, 2.6 at 0.01 to 0.05 E; 1.07 to 1.73 above | **miss** (F-026); within 1.9x of the target |
| L1-10 | Linear twins disperse | yes | hit |
| CD | Claude Desktop: the dip start is the most fragile in lifetime terms; b5 the most robust | dip: yes, a 2 to 4% loss of N ends its long life; b5: yes, no break below N_min and no measurable degradation from kicks | agrees |

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
- **Proposed diagnostic, not run:** the same comparison with an N match of 0.1%, using family starts made for the purpose.

**F-025 (L1-8): ordering.**
- The primary's lower edge: +0.02 kicks put it at N/N_min 1.831 to 1.839, the dip's centre. N32 was applied to the dip start only, though it applies to any start within a few percent of the dip.
- The dip's upper edge is 1.6x over (within 3x), and the dip controls themselves differ 1.9x with the kick time.

**F-026 (L1-9): k3 packets.**
- The dip's response grows as efrac^3 at small amplitude and efrac^1.3 at 0.2 to 0.5 E. The primary's grows as efrac^0.7 to efrac^1.3.
- N33's argument (the dip's 3 omega source vanishes) holds while the packet's own dN stays inside the dip's width. Larger packets move the lump out of the dip.
- **Literature:** no source found for nonlinear absorption of a strong wave by an oscillon.

## 5. 3D: distortions, collisions, spot checks (WP-L3)

[Pending: the Mac runs end about 11:00 to 12:00 on 2026-10-03. Gate on the production grid: passed (speed error at most 4e-4, E/(gamma E0) at most 2.3e-5, omega/gamma at most 3.6e-6, at v = 0.02 and 0.2, phases 0 and 90; collision cross terms at most 4.8e-5). Collision speeds per lump: 0.02 and 0.2 (deviation from the builder's 0.05; reason in section 11).]

## 6. 3D: unequal pairs (WP-P3)

[Pending, or not run: needs the cloud batch.]

## 7. Hits and misses

[Collected at the end.]

## 8. Novelty

[Scholar's novelty check at the end.]

## 9. Workers

[At the end. Effort values seen from inside: numerics 15, reviewer and scholar 30, cloud 10, runner 5 or not visible. CLAUDE.md lists high, xhigh, xhigh, high and medium. Models all match.]

## 10. What the results mean for claim 1

[At the end.]

## 11. Deviations, open issues, parking lot

[At the end. Items so far:]
- **Repository layout.** `outputs/` and `checkpoints/` in the main checkout are links to an external drive (made 2026-10-02 11:41). `git merge` cannot run there. Code merges were made in a worktree, and `main` was moved with `git update-ref`.
  - Twice this caused a slip of mine: `main` briefly lost a status commit (restored within minutes), and a `checkout -- docs` wiped a worker's uncommitted ledger edits (re-applied by the worker from its backup).
- **Collision speeds** 0.02 and 0.2 per lump, not the builder's 0.05; the reviewer pointed to the published slow range of 0.01 to 0.04.
- **P3 horizon** fixed at T = 7500.
- **P3 control:** isolated single lumps instead of a far-apart pair (cost).
- **Rented compute:**
  - Trimmed L3 (b5 distortions halved, b5 collisions dropped).
  - A benchmark found the rate 14 to 19% below the assumption, and the batch was stopped before any run (0.18 USD).
  - The relaunch at a 37 USD batch cap was blocked by the permission system ("Real-World Transactions").
  - Aaron was asked at 15:16. With no answer after an hour, a reduced L3 set ran on the Mac.
- **Block L1** quotes the primary at N/N_min 1.73; the right value is 1.754.
- **Scoring:** misses are reported as written, with the CLAUDE.md 3x reading noted.

## 12. Rented compute

[At the end.]

## 13. Tests on main

[At the end.]
