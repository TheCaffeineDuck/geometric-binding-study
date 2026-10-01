# Session 2 predictions on record

Written by the orchestrator before the runs they refer to. Each block carries its commit, so the order can be checked in `git log`. Each prediction sits next to what the literature expects, with a source from `docs/PRIOR_ART.md` (standard 9). "CD" marks a guess carried over from Claude Desktop in `SESSION_02_PLAN.md`.

Units: Set-B-like (m = 1, g4 = 0.30, g6 = 0.09 beta) unless stated. "frac" is E(r < 3 initial half-widths)/E0. "hw" is the half-width.

## Block 1. Written 2026-10-01, before any Session 2 run

### WP-A. Lifetime against beta (radial)

Definitions fixed here. T_90: first time frac falls below 0.90 (the measure behind the "10,630" of Session 1). T_death: first time E_in falls below half its t = 50 value (the Session 1 death rule). Both are also reported in oscillations, T omega / (2 pi).

| # | Prediction | Basis | Literature |
|---|---|---|---|
| A1 | The path-integral estimate reproduces the beta = 0.85 death (T = 1146) and the beta = 4/3 T_90 (10,630) each to within 30%, using a Gamma(E) map that does not include the validation runs themselves | Ibe eq. 55 and Li eq. 2.30 are this integral; our runs follow one adiabatic path (F-004 D1) | Published practice (Ibe 2019, Li 2025, `READ`). Zhang 2020 warns that estimates are poor where dips exist (`READ`) |
| A2 | Lifetime of the best profile rises monotonically with beta over 0.75 to 5.0. No fall inside the range | Radiation at k_3 is suppressed roughly exponentially in k_3 L with (k_3 L)^2 = 2.4 beta - 9 alpha^2 (PRIOR_ART N7, N9, `DERIVED`) | Olle 2020 heuristic: very long life needs V'' not very negative; V''_min/m^2 = 1 - 3/(2 beta) reaches 0 at beta = 1.5 (N5). Amin-Easther-Finkel 2010 ran beta of 17 to 33 on a lattice. No source for lifetime against beta in this model (search 29, 32) |
| A2-CD | Lifetime keeps rising for some way above 4/3, then falls as the stable branch narrows | CD, no source | as above |
| A3 | beta = 0.75: T_death between 350 and 600. beta = 1.0: T_death between 3,000 and 12,000. beta = 1.15: T_90 between 3,000 and 15,000 | Interpolating Session 1: 344 at 0.70, 576 at 0.80, 1146 at 0.85, alive at 3000 with 0.77 at 1.0 | none below 4/3 |
| A4 | For beta >= 1.6 the best profile is alive at T = 30,000 with frac above 0.90, so the map is capped there and ranked by the path integral | A2 | Mukaida 2017 ran beta = 4/3 to T = 1e5 without death from R_ini = 10 (`READ`) |
| A5 | No beta in the list is permanent. Every profile has a finite path-integral lifetime | Energetic death is generic for power-series potentials (van Dissel 2023, `READ`) | same |

### WP-B. 3D primary to T = 5000

| # | Prediction | Basis | Literature |
|---|---|---|---|
| B1 | The 3D frac curve tracks the radial curve to 1e-3 at every sampled time up to T = 5000 (plan target). Expected size of the difference: 1e-4 to 5e-4. frac(5000) is about 0.956 | 3D matched radial to 7e-5 to T = 1000. The radial curve moves by about 0.016 at T = 5000 for a shift of 0.0009 in omega, so a grid-induced shift of 1e-5 gives 2e-4 | Nodeless states do not fragment in 3D (van Dissel 2023, sine-Gordon, `READ`). No source for non-radial stability of the sextic family |
| B2 | No growth of non-spherical structure: energy moments stay isotropic at the level of run 1 | cubic symmetry of the data is exact; run 5 saw no growth to T = 1000 | same |

### WP-C. Single-lump phase control (run before any pair run)

Initial data for phase theta: phi = Phi(r) cos(theta), phi_t = -omega Phi(r) sin(theta), at beta = 4/3, omega = 0.9091.

**Pass criterion, fixed here.** For theta = 45, 90, 135 and 180 degrees, against theta = 0: |frac_theta(T) - frac_0(T)| <= 0.01 at T = 300, 500 and 1000; measured omega within 0.005; no death. A larger difference is a finding and the pair runs wait (plan, section 4).

| # | Prediction | Basis | Literature |
|---|---|---|---|
| C0a | theta = 180 equals theta = 0 to rounding | phi -> -phi is an exact symmetry | — |
| C0b | theta = 45, 90, 135 pass, with differences of order 1e-3 or less | early shedding at beta = 4/3 is 0.03% (F-002 D3), so the start phase changes little | A phi_t = 0 single-frequency start is not the quasi-breather (van Dissel 2023, `READ`). Pair runs in the literature use phase-shifted copies (Amin & Mou 2020, Xue 2025, `READ`). No source for phase dependence of the settling |

## Block 1b. WP-A supplement. Written 2026-10-02 00:12 (+07), before the supplement runs

What I know when writing this: the first sweep (18 long runs to T = 30,000, `outputs/session2/wpA/lifetime_table.json`). Above beta = 4/3 every best start is alive at the cap, so the sweep cannot rank those beta values by measurement. The supplement extends the best starts at beta = 4/3, 1.6, 2.0, 3.0 and 5.0 to T = 300,000 (a deviation from the plan's cap, to answer its question "is there a beta where a lone lump is effectively permanent"), and adds the reviewer's safeguard starts and matched-size starts.

| # | Prediction | Basis | Literature |
|---|---|---|---|
| A6 | beta = 4/3, start omega 0.9065: T_90 falls between 50,000 and 300,000; the lump is not permanent | measured rate at or below 2e-8 now, but the lump drifts out of its dip (F-004 mechanism) | Published beta = 4/3 lifetimes are 1e4 to 1e5; Mukaida 2017 ran to 1e5 without death (`READ`) |
| A7 | beta = 3.0 and 5.0: the best starts are alive at T = 300,000 with frac above 0.95. beta = 1.6 and 2.0: alive with frac above 0.90 | rates at or below 1e-9 to 1e-8 | Geometric decoupling at large beta (A&S, `READ`); a "longevity cliff" near beta 1.6 to 2.1 is possible (N17, low confidence), which would make 1.6 or 2.0 fall short |
| A8 | At matched N/N_min = 1.3, 1.8 and 2.5 the lifetime rises monotonically with beta | Session 1 D3 scan, monotone to T = 600 | no source |

## Block 1c. 3D confirmation of the long-lived beta = 4/3 start. Written 2026-10-02 02:03 (+07), before the run

Radial result in hand: the beta = 4/3 start at omega 0.906505 keeps frac 0.9946 at T = 300,000 (`long_beta1.3333_w0.906505_T300k.json`). At the T = 300,000 horizon no other beta beats it by a wide margin in retained energy; beta = 5 beats it in rate (zero within error against 1.7e-8). The plan asks for one 3D confirmation of any beta that beats 4/3 by a wide margin. I run the 4/3 start itself in 3D now on the Mac, because it is the candidate for Step 4 at the verified beta, and list a beta = 5 confirmation as pending on rented compute.

| # | Prediction | Basis | Literature |
|---|---|---|---|
| A9 | 3D (N = 128, L = 70, dx = 0.547, dt = 0.04, T = 3000) tracks the radial frac curve of the same start to 1e-3 at every sampled time; measured omega 0.9069 within 0.0005; twin below 0.10 by T = 100 | Session 1: this grid reproduced the primary to 1.6e-5; 3D matched radial to 7e-5 | no source for 3D at this omega; nodeless states do not fragment (van Dissel 2023) |

## Block 1 outcome for the WP-C phase control (recorded 2026-10-01, before Block 2)

Radial runs, `outputs/session2/wpC/phase_control_radial.json`. C0a: hit (2.9e-15). C0b: hit at 45 (9.3e-4) and 135 (1.1e-3); **miss at 90**: the difference in frac is 6.6e-3, 1.09e-2 and 2.46e-2 at T = 300, 500 and 1000 against the 0.01 criterion, and omega is off by 0.0066 to 0.0079 against 0.005. No death. Cause, from three diagnostics: the t = 0 energy of the single-frequency start depends on theta (2843.8 at 0, 2605.2 at 90), so the 90-degree start is a lighter lump (N 9.5% lower, omega 0.9157) that behaves exactly like the family member with its N. Ledger entry F-009.

**What I did with the gate, and why.** The plan's control is "a single lump started at each phase used behaves like the phase-0 lump". The criterion numbers above stay as written. The pair design changed in response to the finding: phases are assigned symmetrically, theta = -/+ dtheta/2, and dtheta = 180 uses (0, 180), so the phases used are 0, +-22.5, +-45, +-67.5 and 180. No 90-degree start is used. Every used phase passes the criterion in the radial runs (worst: +-67.5 at 8.5e-3 at T = 1000, 3.6e-3 at T = 500). The same control is repeated in 3D as a gate (below) and the pair runs start only if it passes. This is a deviation from Block 1, which listed theta = 90 as a used phase; it is reported as such.

## Block 2. WP-C pair runs. Written 2026-10-01 23:20 (+07), before any pair run of the batch

Design: `docs/WPC_BUILD_NOTES.md` section 4. beta = 4/3, omega = 0.909149 profile (A = 56.70, kappa = 0.41647, E = 2801.7, N = 2865.5, hw = 3.344). L = 88, N = 160, dx = 0.55, dt = 0.04. Lumps at x = -/+ d/2, d = 16, 20, 24 (4.8, 6.0, 7.2 hw), dtheta = 0, 45, 90, 135, 180. T = 300 at d = 16, T = 500 otherwise.

Derivation on record: `docs/WPC_DERIVATION.md`. F = 2 pi A^2 cos(dtheta) e^(-kappa d) (kappa/d + 1/d^2) = -dU/dd with U = -2 pi A^2 cos(dtheta) e^(-kappa d)/d; a = F/E; N flows from the lump that leads in phase to the one that lags at 2 pi A^2 sin(dtheta) e^(-kappa d)/d; the relative phase obeys dtheta'' = K sin(dtheta), which runs away from 0 and toward 180.

**What I already knew when writing this.** The numerics worker ran test-sized pair runs while validating the pipeline (N = 96, T <= 70, not the batch; disclosed in `WPC_BUILD_NOTES.md` section 7): early force over prediction 1.004 in phase and 1.041 anti-phase at d = 16, and an early repulsion of -0.065 for the 90-degree pair at d = 16. The sign prediction and the formula were on record before those runs (plan WP-C; `PRIOR_ART.md` section 4.2). Rows C1 and C2 at d = 16 are therefore not blind.

### Gate (fixed criteria; if either fails the pair runs do not start)

| # | Run | Criterion |
|---|---|---|
| G1 | `single_x12_t00.0`: one lump at x = +12, no partner, T = 500 | centroid drift below 0.01 in magnitude at T = 500 (sponge-asymmetry force) |
| G2 | `single_x12_t22.5`, `t45.0`, `t67.5` against `t00.0`, 3D | difference in frac (energy in the 3 hw ball over its t = 0 value) at most 0.01 at T = 300 and 500; omega within 0.005; no death. Same numbers as Block 1; T = 1000 is dropped because no pair run exceeds T = 500 |

Predicted: both pass. G1 about 1e-4 (noise run: 1.3e-6 over T = 80). G2 as in the radial table (worst 3.6e-3 at 67.5, T = 500).

### Pair predictions

"Shift" is per lump, positive toward the partner. Force comparisons use the prediction at the measured separation, measured relative phase and measured omega of the run.

| # | Prediction | Tolerance | Literature |
|---|---|---|---|
| C1 | Sign: dtheta = 0 and 45 attract at first, 135 and 180 repel, at all three separations | sign | Agrees: Battye & Sutcliffe 2000, Amin & Mou 2020, Xue 2025, Burlak & Malomed 2018 (`READ`) |
| C2 | Size: measured force over predicted = 1 | within 15% at dtheta = 0 and 180; within 25% at 45 and 135; early half-space value within 20% (radiation contaminates it by about 10% at d = 16) | Formula published for the non-relativistic limit (Malomed 1998, `READ`); no measured value in any source |
| C3 | Decay constant from dtheta = 0 and 180 across the three separations, with the exact 1/d^2 term: kappa = 0.416 | within 0.03 | same |
| C4 | Cosine law: F/F0 against cos(measured dtheta) has slope 1 and intercept 0 | slope within 0.15, intercept within 0.1 | cos law only for equal frequencies (Bowcock 2008, `READ`) |
| C5a | dtheta = 90: leading force vanishes at the start: early force below 0.15 F0(d) in magnitude | — | first-order force zero at 90 (Malomed 1998 via Burlak & Malomed, `READ`) |
| C5b | dtheta = 90: N flows from the leading lump to the lagging one at 1.61, 0.244, 0.038 per unit time at d = 16, 20, 24 | direction; rate within 30% over the early window | Direction and rate: no source (N14 is the scholar's derivation, same result). Q-ball charge transfer at intermediate phase: Battye & Sutcliffe 2000 |
| C5c | dtheta = 90: the phase runs toward 180: dtheta at the end = 160 (d = 20), 102 (d = 24); beyond 180 at d = 16 (model outside its validity there) | within 30 degrees at d = 20 and 24 | Phase has negative effective mass (Malomed 1998, `READ`) |
| C5d | dtheta = 90: the pair ends repelling. Shift -2.8 (d = 16, T = 300), -0.89 (d = 20), -0.03 (d = 24) | sign at d = 16, 20; within 50% at d = 20 | Repulsion with an N split of 0.76 at kappa d = 2.5 (Burlak & Malomed 2018, `READ`); ours is kappa d = 6.7 to 10 |
| C5-CD | "At 90 degrees the leading force vanishes and N flows from one lump to the other" (plan) | — | as C5a, C5b; the sources add that the pair then repels |
| C6 | dtheta = 45: attracts first, then the phase runs past 90 and the force reverses. d = 16: not merged, separating by T = 300 (low confidence: the model is at the edge of its validity). d = 20: turning, d about 17 at T = 500. d = 24: still approaching, shift 0.59 | outcome at d = 16, 20; shift within 30% at d = 24 | Disagrees at short range: 45 degrees merges at kappa d = 2.5 in 3D cubic-quintic NLS (Burlak & Malomed 2018). No source at our separations |
| C7 | dtheta = 0: merged by T = 300 at d = 16 (contact near t = 126) and by T = 500 at d = 20 (contact near 346). d = 24: approaching, shift 0.90 at T = 500. Where a merger completes inside the run, the merged lump keeps 0.6 to 0.9 of E1 + E2 | contact times within 25%; shift within 25% | Merge: all sources. 0.7 (E1 + E2): Amin & Mou 2020, a different potential |
| C8 | dtheta = 180: separated at every d. Shift -5.7 (d = 16, T = 300), -3.2 (d = 20), -0.71 (d = 24). N of the two lumps stays equal (exact symmetry) | shift within 25%; N difference at rounding | Rebound, no charge transfer at pi (Battye & Sutcliffe 2000; Burlak & Malomed 2018). Plane stress non-negative at every instant (N15, derived) |
| C9 | dtheta = 135: separated at every d; phase moves toward 180. Shift -4.9, -2.6, -0.52 | shift within 30% | no source at 135 |
| C10 | **No pair is bonded.** Bonded would mean: two distinct lumps at the end, and d(t) showing at least one minimum and one maximum inside the run while staying between 2 hw and d0 + 1. "Bounded at T" for slow pairs (all d = 24 runs, d = 20 at 45) is transit, not a bond | — | No static pair in any source; orbiting two-soliton states are unstable (Malomed 1998). Breathing pairs exist for Q-balls in 1D and 2D (Axenides 2000) |
| C11 | Linear twins: energy in each ball below 0.05 of its start by T = 100; no centroid attraction | — | free packets disperse (Copeland 1995); Session 1 twins 0.018 at T = 100 |
| C12 | Resolution check (d = 20, in phase, N = 224, T = 300): d(t) within 0.1 of the N = 160 run. Box check (d = 24, in phase, L = 132, T = 300): d(t) within 0.02 | targets | — |
| C13 | Angular momentum about z and about x stays at zero to noise in every pair run (collinear lumps) | — | N12 (derived): 0/180 patterns carry none |

### Block 2a. Amendment to how the force is measured. Written 2026-10-01 23:50 (+07), before any pair run has started (the rented batch is still on its benchmark and gate runs)

Source: the independent review of the WP-C code. It ran a single lump at x = +12 on the production grid (no partner) and found that the momentum in the far half-space swings by -0.56 (theta = 0) to -1.12 (theta = 45) at t of about 46 and recovers by t of about 88: the lump's own start-up radiation. The slope over the early window [10, 60] is -0.006 to -0.011 per lump, against a predicted pair force of 0.0176 at d = 24. The early half-space slope is therefore not a measurement of the pair force at d = 20 and 24. No pair data informed this amendment. The predicted values in Block 2 are unchanged.

- **C2, C3, C4 are judged on two measures:** (i) the sliding half-space force against F(d(t), dtheta(t)) at the measured omega, on windows centred at t >= 100 while two distinct lumps remain; (ii) the centroid separation d(t) against the point model run with the measured dtheta(t). The early half-space slope over [10, 60] is reported for every run but counts as a criterion only at d = 16 (tolerance as in C2). Where the single-lump runs allow it (d = 24, same geometry and phases), the early slope is also reported with the single-lump contribution subtracted.
- **C5a restated:** at dtheta = 90 the measured force on windows at t >= 100 equals F0(d) cos(measured dtheta) to within 0.15 F0(d). (The start is not exactly at 90: the +-45 pair reads 92.3 degrees.)
- **Per-lump N and energy at "start"** are read from the first window centred at t >= 15; the first whole-period window is a start-up artefact.
- **Known limits, stated before the runs:** the box check is 1.5 times the box (L = 132 against 88), not the 2 times of standard 4; only in-phase linear twins are in the rented batch (one anti-phase twin at d = 16 is added on the Mac); lumps started at +theta and -theta settle 8e-5 to 1.2e-4 apart in omega, a static phase drift of about 8e-5 rad per unit time, about a tenth of the predicted pendulum drift at d = 24.

### Block 2b. Notes added 2026-10-02 00:25 (+07). No criterion or predicted value is changed

- **Sign slip in the "Derivation on record" line above.** F there is positive for attraction, so F = +dU/dd; the force along increasing separation is -dU/dd = -F. The predicted values and signs in the table are unaffected.
- **What I know now that I did not when Block 2 was written.** While testing the revised analysis script the numerics worker ran 19 coarse test pairs (N = 96, dx = 0.92, T = 150, scratch, not committed) and reported their criteria table to me. Block 2 and 2a were committed before I saw it. From here on no criterion is amended.
- **Criteria expected to miss for mechanical reasons, flagged by the worker before the batch finished** (they stay as written and will be judged as written, with the mechanical cause analysed in the report): C2 at d = 16 in phase (the only windows at t >= 100 are near contact, where the tail formula overpredicts); C2 at 45 and 135 degrees (ratios amplified where the predicted force passes near zero); C5b at d = 24 (the +theta and -theta lumps settle at different rates, larger than the interaction rate of 0.038; needs single-lump runs at -theta to subtract); C11 "no centroid attraction" (ball centroids of a dispersing linear twin are not lump positions).

Stop rule from the plan, section 4: if the measured pair law contradicts the derivation in sign (C1), stop after WP-C.

## Block 3. WP-E arrangement runs. Written 2026-10-02 02:27 (+07), before any arrangement run

**What I know when writing this.** The point-lump scan (`docs/WPD_NOTES.md` with its review corrections; `outputs/session2/wpD/predictions_point_model.json`), built on the derived pair law, not yet on the measured one: the rented pair results cannot be fetched (AWS sign-in expired), and a local copy of the d = 16 pair runs is still running. From its logs I have seen only that the in-phase pair has absorbed 811 of its energy by t = 250 and the anti-phase pair 10.6. I have also seen one coarse smoke run from the tooling build (a 3-lump ring with 120 degree steps expanded from spacing 10.4 to 15.5 by t = 60). No arrangement run starts until the local pair runs confirm the sign of the force (C1).

**Starts.** Lumps are copies of one settled lump taken at different times (`swirl/engine/snapshot.py`), so lumps at every phase are the same lump: equal to about 4e-6 in N and 3e-6 in E in the tooling tests, against up to 9.5% for the single-frequency start (F-009). Grid dx = 0.55, dt = 0.04, default sponge, box from the generator's guard.

**Outcome criteria, fixed here** (parameters of `outputs/session2/wpE/criteria.json`). Samples from t = 50 to the end of the run.
- **Bonded:** the number of distinct lumps found from coordinates equals K, and every lump's nearest-neighbour distance stays between 0.5 and 1.5 times its initial value, in at least 95% of the samples and in the last sample; the run completed.
- **Merged:** the distinct-lump count is below K at the end.
- **Flew apart:** the count is K at the end and some nearest-neighbour distance is above 1.5 times its initial value.
- **Not distinct at the start:** fewer than K distinct lumps in the first count window. Such a start cannot be called bonded whatever follows; what it becomes is described.
- Anything else is reported as "bounded at T, not bonded" with the trend of the distances.
- A bonded outcome is not a result until a second run at finer resolution and longer time confirms it (plan, WP-E).

**Runs, in priority order, with the point-model outcome (Model A unless stated) and my prediction.** a is the nearest-neighbour spacing; hw = 3.344. "Contact" is the first time the count drops below K or a pair comes within 2 hw.

| # | Run | T | Point model | Prediction | Tolerance |
|---|---|---|---|---|---|
| E1 | 5-ring, 72 degree steps, a = 16 (4.8 hw) | 600 | merger at 276 (B: 286) | merged; contact later than its in-phase control E2 by a factor 1.85; per-lump N equal to 2% and steps within 10 degrees of 72 until t = 100 | contact 190 to 420; factor 1.3 to 2.5 |
| E2 | 5-ring, all in phase, a = 16 | 400 | merger at 149 | merged | contact 105 to 200 |
| E3 | 5-ring, 72 degree steps, a = 8.11 (2.43 hw; Model B balance) | 500 | B: merger at 135 | merged by T = 300; not bonded | — |
| E4 | 5-ring, 72 degree steps, a = 3.02 (0.90 hw; CD's balance, kappa R = 1.07) | 400 | collapse at 12 | fewer than 5 distinct lumps at the start or by t = 50; one object or dispersal; not bonded | — |
| E5 | 4-ring, alternating 0/180 (mirror control, N15), a = 16 | 500 | flies apart at 230 | flew apart: nearest-neighbour distance passes 1.5 times its start; no merger; per-lump N stay equal | passes 1.5 at 160 to 320 |
| E6 | 4-ring, 90 degree steps, a = 12 | 600 | flies apart at 318 (B: 292) | flew apart | passes 1.5 at 200 to 450 |
| E7 | 6-ring, 60 degree steps, a = 7.32 (2.19 hw; Model B balance) | 400 | B: merger at 174 | merged by T = 300; not bonded | — |
| E8 | octahedron around an anti-phase centre, centre-to-shell 3.82 (1.14 hw; CD's balance, kappa R = 1.59) | 300 | collapse at 17 to 24 | fewer than 7 distinct lumps at the start or by t = 50; not bonded | — |
| E9 | cube around an anti-phase centre, centre-to-corner 8.51 (2.54 hw; Model A balance) | 400 | merger at 86 | not bonded: merged or flew apart by T = 250 | — |
| E10 | 8-ring, 90 degree steps (two turns), a = 12 | 500 | merger at 248 | merged | contact 170 to 370 |
| E11 | 6-ring around an anti-phase centre, R = 15.77 (4.71 hw; Model B balance resting on second-order terms; a probe, not a candidate) | 700 | B: merger at 463 | merged; low confidence | contact 320 to 650 |
| E12 | 5-ring, phases 0, 0, 180, 180, 0 (rounded control), a = 16 | 400 | merger at 117 | merged (the in-phase neighbours first) | contact 80 to 160 |
| E13 | 5-ring, 144 degree steps (step above 90, control), a = 16 | 500 | not in the scan; neighbours repel at first order | no merger; expanding: nearest-neighbour distance above 1.2 times its start at T = 500 | — |
| E14 | 5-ring, 72 degree steps, a = 30 (far-apart control) | 500 | nothing resolves | nothing happens: count 5, distances within 1%, per-lump N equal to 1e-3 | — |
| E15 | linear twin of E1 (T = 300); resolution check of E3 (dx 0.393); box check of E3 (1.5 times the box) | — | — | twin: no lumps left, group energy below 0.1 by T = 100. Checks: same outcome class as E3, contact time within 15% | — |

| # | Overall prediction | Literature |
|---|---|---|
| E-all | **No arrangement is bonded.** Stepped rings below 90 degrees per step end in a delayed merger; rings with steps of 90 degrees or more and patterns with a phase-exchanging mirror plane expand; arrangements at the point-model balances (all in the overlap region) merge or collapse | Optical and 3D cubic-quintic clusters with stepped phases are metastable for a few revolutions and need a complex field (Desyatnikov & Kivshar 2002, Crasovan 2003, `READ`); necklace beams expand (Soljacic 1998, `READ`); no radiation-free bound state with a phase-exchanging mirror plane (N15, derived; Cunha 2022 for complex fields); any pair-law balance with attractive single-lump energy is unstable (N16, derived). No source for real-field oscillon arrangements (S4, searches 41 to 82) |
| E-rot | Stepped rings carry angular momentum from the start (4 to 13% of N per lump at 4.8 hw by the overlap formula) and it is conserved to the level of the absorbed radiation. I do not predict whether the ring rotates | Optical rings rotate with angular velocity J/I to 2% (Desyatnikov & Kivshar 2002) |
| E-CD | CD's two numbers: the 5-ring at kappa R = 1.07 and the octahedral shell at kappa R = 1.59 do not hold in 3D (E4, E8) | reproduced in the point model by the scholar and by WP-D; both in the overlap region |

Runs are on the Mac until rented compute is available again; anything not run is listed as pending in the report.

### Block 3a. Notes added 2026-10-02 02:47 (+07), before any arrangement run. No criterion or prediction is changed

From the independent review of the tooling:
- **The equality figures quoted above are the best case.** "4e-6 in N, 3e-6 in E" is the first window on a dx = 0.8 test source. Over a whole record the per-lump N differ by up to 6.4e-5. Copies are not exactly identical: the source frequency still rises by 4.4e-5 over 200 time units and carries a modulation of rms 2.8e-5. All of this is at or below 1e-4, against signals of 1e-3 or more.
- **The window must be (26, 30)** in every run's config; the example file still carried (14, 18), with which lumps differ by up to 4.3e-4 in N.
- **The post-processing did not yet implement the bonded rule as written** (it lacked the last-sample condition, the "not distinct at the start" label, and it used "at least K" for "equals K"). It is being corrected to the rule above; the rule is not being changed to fit the code. With samples every 25 time units from t = 50, a T = 500 run has 18 samples, so 95% means no sample may fail there; T = 600 and 700 runs tolerate one.
- Per-lump tracked series (N, phase, energy of each lump) are not meaningful once the distinct-lump count is below K and will not be read there.

### Block 3b. Second run for the one outcome the criterion labels bonded. Written 2026-10-02 06:00 (+07), before the run

**What I know.** The batch is in. E11 (6-ring around an anti-phase centre, R = 15.77) is labelled "bonded" by the rule above: 7 distinct lumps in all 26 samples to T = 700, nearest-neighbour ratios 0.66 to 0.85 at the end. Its plot shows the ring contracting at a growing rate (tracked minimum ratio 0.59 at the last window), so the label looks like a run that stopped before contact. The plan requires a second run at finer resolution and longer time before a bonded outcome is reported. E14, the far-apart control, is also labelled "bonded" because nothing happens in it; that is a flaw in my criterion (it does not ask the outcome to differ from the control), noted in the report.

| # | Run | Prediction | Tolerance |
|---|---|---|---|
| E11-long | E11 rerun to T = 1200 at the same resolution (dx = 0.55) | merged; contact (count below 7 or a pair within 2 hw) between t = 700 and 800; not bonded | contact 700 to 850 |
| E11-fine | E11 at dx = 0.393 to T = 1000 (source lump resampled) | same outcome; d(t) of the ring agrees with E11-long to 0.1 up to t = 600; contact within 10% | — |

Literature: none specific; the point model gives merger at 440 to 463 for this start, so the 3D run is already 1.5 times slower than the model.
