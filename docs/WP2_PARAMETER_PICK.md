# WP2 parameter pick and WP3 predictions on record

Date: 2026-10-01. Written by the orchestrator before any 3D Step 2 run started. Inputs: WP1A (branch `worktree-agent-a63e9fc3746934e76`, commit b138396; `outputs/family/`), WP1D (`outputs/benchmark/`), `docs/PRIOR_ART.md`.

## Why not the plan's default (beta = 0.70)

The radial check contradicts the Step 1 prediction at beta = 0.611 and 0.70: every profile, on both branches, dies before T = 500 (death times 120 to 345; 25-profile Gamma map at beta = 0.70, omega 0.71 to 0.92, all die). A beta = 0.70 profile cannot meet the Step 2 target (90% at T = 1000) even with spherical symmetry imposed, so a 3D run at beta = 0.70 would only re-measure a known radial death and would say nothing about non-spherical survival. The plan's rule for this case applies: pick the profile the radial check shows is longest-lived, and record the contradiction as a finding (`docs/FAILURES.md`, F2).

## Pick

| | Primary | Secondary |
|---|---|---|
| beta | 4/3 | 0.85 |
| Units (Set-B-like) | m = 1, g4 = 0.30, g6 = 0.12 | m = 1, g4 = 0.30, g6 = 0.0765 |
| omega (profile) | 0.9091 | 0.7793 |
| Branch | dN/domega < 0 (omega_c = 0.9691) | dN/domega < 0 (omega_c = 0.9493) |
| E/(mN) | 0.978 | 0.878 |
| Half-width (Phi = Phi0/2) | 3.344 | 3.636 |
| E_loc radius, 3 half-widths | 10.03 | 10.91 |
| Radial frac at T = 500 | 0.995 | 0.957 |
| Radial Gamma (last 200) | 3.0e-6 | 5.9e-5 |
| Radial shedding | < 0.1% | 2.2% |
| Nonlinear share at t = 0 | -0.32 | -0.79 |
| False vacuum (exists for beta < 5/6) | no | no |

Reasons:
- **Primary, beta = 4/3, omega = 0.909.** The longest-lived radial profile found (0.995 at T = 500, 0.956 at T = 5000), at a radiation-rate dip that matches Ibe et al. 2019 (Q = 10^2.14) — the only point with an independent literature anchor. It sits inside the plan's E/(mN) window. Because radial survival is not in doubt, a 3D failure would be attributable to the 3D question Step 2 asks (non-spherical perturbations), which is what the gate is for. Cost of this choice: the potential is further from the legacy Set B (beta = 0.611) and is the same potential as published radial work, so the gate adds only the 3D question.
- **Secondary, beta = 0.85, omega = 0.779.** The lowest tested beta with a radial survivor, closer to the plan's intent, no published 3D point, no false vacuum (beta > 5/6). Its radial dip is sharp (omega = 0.789 and 0.799 die), so it is fragile; it is run as runs 1 and 2 only.
- **Grid.** dx = 0.365 (N = 192, L = 70). Points per half-width: 9.2 (primary), 10.0 (secondary); target >= 6. Third harmonic k3 = sqrt(9 omega^2 - 1) = 2.54 (primary): 6.8 points per wavelength (target >= 6). Fifth harmonic (brief, section 6) k5 = 4.43: 3.9 points per wavelength, below Nyquist limit k_max = 8.6 by 2x; the resolution run checks it.
- **Box.** L = 70 with the default sponge (gamma_max = 2, width 12, cubic, r_in = 23). E_loc radius 10-11 sits well inside r_in. WP1B check 4 found interior radius >= ~22 needed for slow free packets to T = 75; r_in = 23 meets it. Box validation against twice the size: L = 70 vs L = 140 at a coarser common dx (0.547, N = 128 vs 256) to T = 300.
- **Time step.** dt = 0.04 (about dx/9). WP1B: dt = dx/8 at dx = 0.78 gave 2.75e-5 drift over T = 100 (RK4 dt^5); at dx = 0.365 and dt = 0.04 expect ~1e-6 per T = 100 from the main mode. Measured E + absorbed is reported for every run.
- **Initial data.** phi = Phi(r) from the WP1A profile, phi_t = 0, centred at the origin. Distortion run: `quad_eps` = 0.02 (volume-preserving l = 2 stretch).

## Predictions on record for WP3 (written before running)

Literature next to each (source: `docs/PRIOR_ART.md`).

| Run | Prediction | Basis | Literature |
|---|---|---|---|
| 1, primary | E_loc(r < 10.03, T = 1000)/E0 = 0.99 +/- 0.01; measured omega = 0.909 +/- 0.005 < m. Go. | radial frac 0.995 at T = 500 and Gamma = 3e-6 | Ibe 2019 (radial, beta = 4/3): lifetime 1e4+; Gamma dip near Q = 10^2.14. No published 3D non-radial run at this beta |
| 2, primary twin (g4 = g6 = 0) | E_loc/E0 < 0.01 by T = 100 and < 0.01 at T = 1000 | radial twin 1e-5 at T = 500 | Copeland 1995: linear packets leave in ~5 |
| 3, primary, quad_eps = 0.02 | survives like run 1 (E_loc at T = 1000 within 0.02 of run 1); quadrupole moment decays or stays bounded | nodeless radial ground state | van Dissel et al. 2023: nodeless states did not fragment in 3D (sine-Gordon); Cyncynates & Giurgica-Tiron 2021: higher l generally more stable. No source for the sextic family in 3D |
| 4, primary, N = 288 (dx = 0.243), T = 300 | E_loc and omega agree with run 1 to 1e-3 at matching times | spectral convergence at 4-7 points per wavelength | — |
| 1b, secondary | E_loc(r < 10.91, T = 1000)/E0 = 0.93 (range 0.85-0.96); marginal Go; risk that the 3D grid detunes the dip and the lump dies as its neighbours at omega = 0.789 do | radial frac 0.957 at T = 500, Gamma = 5.9e-5 | no published point at beta = 0.85 |
| 2b, secondary twin | < 0.01 by T = 100 | as run 2 | as run 2 |
| Box check | L = 70 vs L = 140: E_loc curves agree to 1e-3 to T = 300 | sponge absorbs fast radiation to ~1e-6 (WP1B) | Salmi & Hindmarsh shell check |

Step 2 targets (unchanged from the plan): Go if >= 90% of energy inside 3 half-widths at T = 1000 with measured omega < m; the linear twin must fall below 10% in the same time or the diagnostic is broken; No-go only if no member survives for any beta >= 5/8. A result between targets is reported as it is.

Status: subject to the reviewer's independent re-run of the WP1A radial result. If that re-run overturns the low-beta deaths, this pick is revisited before WP3 and the change is recorded here.
