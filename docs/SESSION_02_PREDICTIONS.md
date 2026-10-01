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

## Block 1 outcome for the WP-C phase control (recorded 2026-10-02, before Block 2)

Radial runs, `outputs/session2/wpC/phase_control_radial.json`. C0a: hit (2.9e-15). C0b: hit at 45 (9.3e-4) and 135 (1.1e-3); **miss at 90**: the difference in frac is 6.6e-3, 1.09e-2 and 2.46e-2 at T = 300, 500 and 1000 against the 0.01 criterion, and omega is off by 0.0066 to 0.0079 against 0.005. No death. Cause, from three diagnostics: the t = 0 energy of the single-frequency start depends on theta (2843.8 at 0, 2605.2 at 90), so the 90-degree start is a lighter lump (N 9.5% lower, omega 0.9157) that behaves exactly like the family member with its N. Ledger entry F-009.

**What I did with the gate, and why.** The plan's control is "a single lump started at each phase used behaves like the phase-0 lump". The criterion numbers above stay as written. The pair design changed in response to the finding: phases are assigned symmetrically, theta = -/+ dtheta/2, and dtheta = 180 uses (0, 180), so the phases used are 0, +-22.5, +-45, +-67.5 and 180. No 90-degree start is used. Every used phase passes the criterion in the radial runs (worst: +-67.5 at 8.5e-3 at T = 1000, 3.6e-3 at T = 500). The same control is repeated in 3D as a gate (below) and the pair runs start only if it passes. This is a deviation from Block 1, which listed theta = 90 as a used phase; it is reported as such.

## Block 2. WP-C pair runs. Written 2026-10-02 00:50 (+07), before any pair run of the batch

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

Stop rule from the plan, section 4: if the measured pair law contradicts the derivation in sign (C1), stop after WP-C.

## Block 3. WP-E arrangement runs

To be written after WP-D and before any arrangement run.
