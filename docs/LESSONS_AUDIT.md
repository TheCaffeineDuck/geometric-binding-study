# LESSONS audit (WP1C)

Date: 2026-10-01. Worker: runner, claude-sonnet-5-5 (reasoning effort setting in my environment read 5, not the "medium" in CLAUDE.md; noted as a mismatch).
Scope: items 1, 2, 3, 7, 8, 9, 10 of `docs/LESSONS_2026-10-01.md`, recomputed from `legacy/outputs` and legacy code. LESSONS was not edited and `legacy/` was not modified (the legacy evolver was copied to `outputs/audit/scratch/` for grid energies and the item 7 run).
All numbers: `outputs/audit/audit_results.json`. Scripts: `outputs/audit/audit_itemNN.py` (shared helpers in `audit_common.py`). Legacy parameters: Set B, phi0 = 0.5, R = 2.5, d = 6, N = 64, L = 50, Gaussian `phi0 exp(-r^2/(2R^2))` as in the legacy code.

Agree means within the rounding shown in LESSONS unless a tolerance is stated.

| Item | Quantity | LESSONS | Recomputed | Verdict | Source |
|---|---|---|---|---|---|
| 1 | Nonlinear (quartic+sextic) share of packet energy | about 0.06% | 0.178% net (quartic -0.0240, sextic +0.00002, E = 13.4618) | **DISAGREE** (3x) | audit_item01.py, grid and closed form agree |
| 1 | Frequency vs m | 1.113 vs m = 1 | 1.113 hard-coded (phase12); 1.11256 calibrated (continuous_phase_summary.json); sqrt(m^2+3/2R^2) = 1.11355 | agree (see note 1b) | audit_item01.py |
| 1 | max abs phi / phi0 at t = 50 | about 5% | 4.19% at t = 50.0; 4.88% max over 45 to 55 | agree (about) | phase1/set_B_baseline.json |
| 1 | max abs phi / phi0 at t = 500 | toward 24% | 20.0% at t = 500.0 (23.0% max over 495 to 500) | **DISAGREE** (20.0 vs 24) | same |
| 1 | Free dispersal at t = 50 | 4.4% | 4.37% | agree | formula |
| 2 | J(6) | 5.203 | 5.2030 | agree | audit_item02.py |
| 2 | J(10) | 0.239 | 0.23903 | agree | |
| 2 | Legacy PDE pair d = 6 / d = 10 | 5.172 / 0.239 | 5.1723 / 0.23849 (T = 500) | agree | distance_sweep_summary.json |
| 2 | Cube checkerboard, closed / PDE | -51.65 / -51.53 | -51.654 (CE = 12 bipartite pattern) / -51.53 (L=50, T=500) | agree | boundary_test/comparison.json |
| 2 | Cube anti-tetrahedral, closed / PDE | -23.98 / -23.94 | -23.982 / -23.945 (T = 500) | agree on numbers; **pattern label disagrees** (note 2a) | phase2/cube/polarized_T1.json |
| 2 | Icosahedron all-same, closed / PDE | 165.96 / 164.11 | 165.958 / 164.106 (T = 500) | agree | phase2/icosahedron/ce_0.json |
| 2 | 50 seeds at T = 1000: correlation | 0.999997 | 0.9999967 | agree | phase12/seed_*.json |
| 2 | 50 seeds: largest miss | 0.16 | 0.161 (seed 0) | agree | |
| 2 | 50 seeds: E_bind range | -40 to +78 | -39.65 to +77.43 | agree | |
| 3 | Predicted slopes ico / cube / oct | 3.83 / 7.61 / 2.85 | 3.8317 / 7.6112 / 2.8542 | agree | audit_item03.py |
| 3 | Measured slope, ico | 3.79 | 3.787 (paper Table 4 values, R2 0.997); 3.786 (grid t = 0 energies, 6 classes) | agree | paper_draft_v9.md Table 4; not a legacy/outputs file (note 3a) |
| 3 | Measured slope, cube | 7.64 | not recoverable from stored output; grid t = 0 fit gives 7.53 (R2 0.973) | **NOT RECOMPUTED / does not reproduce** | note 3b |
| 3 | Measured slope, oct | 2.81 | not recoverable from stored output; grid t = 0 fit gives 2.81 (CE = 6), 2.83 (CE = 8) | agree on the grid t = 0 stand-in only | note 3b |
| 7 | Stiffness term in evolver, absent from energy | present, absent | `legacy/engine/evolver.py:78` computes `sigma*dx**4*K4*phi_hat`; `:80` subtracts it in phi_ddot; `compute_energy` is `:97-111` and has no K4 term | agree | audit_item07.py |
| 7 | Energy step | 13.46178 to 13.46374 within 10 time units | 13.461782 at t = 0, 13.463738 at t = 10 (already at 13.46374 by t = 1.0); final 13.463815 | agree | phase1/set_B_baseline.json |
| 7 | Cause is the term | stated | scratch run T = 20: sigma = 0.01 gives +0.00196 by t = 10; sigma = 0 gives -1.3e-6; adding 0.5*sigma*dx^4*sum(K4 abs(phi_hat)^2) to E makes sigma = 0.01 flat (spread 2.6e-6, same as sigma = 0) | agree | scratch run |
| 8 | Seeds bound at t = 0 / T = 1000 | 30 / 30 | 30 / 30, same seeds | agree | audit_item08.py |
| 8 | Sign changes | 0 | 0 (also none at any of the 101 recorded times) | agree | |
| 8 | Largest change in E_bind | 0.03 | 0.032 (seed 23) | agree | |
| 9 | Samples / spacing / length | 101 / 10 / 1010 | 101 / 10 / N*dt = 1010 (last minus first = 1000) | agree | audit_item09.py |
| 9 | Periods 1010, 505, 252 | window / 1, 2, 4 | 1010, 505, 252.5; they are also the three largest bins of an FFT of my ensemble-mean S(t) (k = 4, 2, 1) | agree | |
| 10 | Ico CE = 15 classes: rot / rot+refl / rot+flip / all | 12 / 10 / 6 / 5 | 12 / 10 / 6 / 5 (544 assignments, group sizes 60 and 120 from coordinates) | agree | audit_item10.py |
| 10 | Cube CE = 6 classes: same order | 7 / 6 / 5 / 4 | 7 / 6 / 5 / 4 (64 assignments, groups 24 and 48) | agree | |

## Notes on disagreements and items not fully recomputed

**1a. Nonlinear share, 0.178% vs "about 0.06%".** Grid and analytic integrals agree (quartic -0.024032, sextic +0.0000200, total energy 13.461782 equal to the stored E_series[0]). The ratio is 0.178%, roughly 3 times the stated figure. I tried the alternative Gaussian conventions exp(-r^2/R^2) and exp(-r^2/(4R^2)); they give 0.149% and 0.198%, so a convention difference does not produce 0.06%. I could not find a definition giving 0.06%. The conclusion of the lesson (nonlinear share is of order 0.1%, packet is effectively linear) is unaffected, but the number is stated differently from what I measure. Not reconciled.

**1b. Frequency.** LESSONS item 12 quotes sqrt(m^2 + 3/(2R^2)) = 1.1136; I get 1.11355, which rounds to 1.114. The legacy calibrated value in `continuous_phase_summary.json` is 1.11256 (1.113 as used in phase12). The formula and calibration differ by 0.001. The statement that 1.113 is above m = 1 holds.

**1c. max|phi| at t = 500.** The stored value is 0.19994 of phi0 at t = 500 (amplitude_retention_final), not 24%; the maximum over 495 to 500 is 23.0%. Larger returns occur earlier: the run's late maximum is 45.4% at t = 444 and about 40% at t = 450. The qualitative claim (a rise from 5% as waves wrap the box) holds.

**2a. Cube "anti-tetrahedral" pattern.** The -23.94 PDE value belongs to `phase2/cube/polarized_T1.json`, whose stored phases are [-1,1,-1,1,1,-1,1,-1] with CE = 8, not a tetrahedral pattern: the minus vertices {0,2,5,7} include an edge (0 and 2). Closed form for that stored pattern is -23.982, matching. The same name "polarized_T1" in `studies/02a_cube_phase2.py` lists [-1,1,1,-1,1,-1,-1,1] (the bipartite pattern, CE = 12), which gives -51.65 closed form and -51.53 PDE (`boundary_test`, and `phase2/cube/cube_polarized_T1_CE12.json` -51.526). So the phase2 file and the code disagree on which pattern was run, a label/ordering mismatch of the kind in lesson 13. The tetrahedral (bipartite) pattern is the one called "checkerboard" in LESSONS (-51.65); the true tetrahedral-vs-anti distinction needs a definition from Aaron. The legacy file named `checkerboard` in phase2 is a CE = 6 pattern, PDE -3.86, closed form -3.806. Grid t = 0 energies from the legacy energy function: bipartite -51.529, CE = 8 pattern -23.947, ico all-same 164.119, pair d = 6 5.1727, pair d = 10 0.23851.

**3a. Measured ico slope.** `legacy/outputs` has only class A of the six CE = 15 icosahedron classes; the other five E_bind values exist only in the paper draft Table 4, which I used (slope 3.787, intercept -5.033, R2 0.9970). The grid t = 0 energy fit of the six rot+flip classes reproduces it (3.786), consistent with E_bind being fixed at t = 0. The paper table lists six classes under rotations+flip; under all symmetries there are five.

**3b. Measured cube and oct slopes.** No stored legacy output contains several cube or octahedron configurations at one cross-edge count (cube phase2 has one config per CE; oct has one per CE). The stored PDE energies cannot give a fixed-CE slope. As a stand-in I computed static t = 0 grid energies with the legacy energy function (PDE/static = 0.9999 per `oct_audit_results.md`): oct gives 2.8115 (CE = 6) and 2.8344 (CE = 8) against LESSONS 2.81; cube CE = 6 gives 7.53 against LESSONS 7.64. The 7.64 is not reproduced by the 4-class regression; two-point slopes among the classes range 6.28 to 8.80, so the cube value depends on which classes enter (R2 of the fit is 0.973, because body diagonals at 10.39 are not accounted for in N*J(d2)). Source of 7.64 unknown to me. The predicted slopes 3.8317, 7.6112, 2.8542 reproduce exactly from N*J(d2): ico 12*J(9.708), cube 8*J(8.485), oct 3*J(8.485).

**7. Not a number disagreement, but a precision note.** The jump to 13.46374 happens by t = 1.0, much earlier than "within 10 time units". After t = 10 the energy wanders between 13.46300 and 13.46433, not strictly flat.

**9. FFT method.** The legacy FFT code is not in `legacy/studies`; I used an FFT of the ensemble-mean order parameter. The bin periods are 1010/k only if the window is taken as N*dt = 1010 rather than the 1000 spanned; with the span the periods would be 1000, 500, 250.

**10. Legacy C and D arrays.** Not present anywhere in `legacy/` (07b builds classes at runtime; `paper_v4_data.json` is absent). Not recomputed. From coordinates: under reflections exactly one pair of rot+flip classes merges for each solid, both pairs with equal sigma^2_NS (ico 0.9167, cube 0.25) and equal closed-form energy, so a mirror pair is geometrically consistent with "D and E" (ico) and "C and D" (cube), but I cannot map the legacy letters to my classes. For the icosahedron the three sigma^2 = 0.9167 classes C, D, E all have the same closed-form energy (-1.2772); PDE values differ by 0.5% (paper).

## Open issues

- Item 1: share 0.18% vs 0.06% and 20% vs 24% unexplained; Aaron or the LESSONS author should state the definition behind 0.06%.
- Item 3: cube 7.64 and oct 2.81 as "measured" have no stored source in legacy/outputs.
- Item 2: pattern naming for the -23.94 cube config.
- Item 10: the C/D arrays cannot be located, so the mirror-pair identification is unconfirmed.
