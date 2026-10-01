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

## Block 2. WP-C pair runs

To be written after the WP-C derivation and before any pair run.

## Block 3. WP-E arrangement runs

To be written after WP-D and before any arrangement run.
