# Session 5 Track F (WP-D): floor comparison in 3D radial symmetry

Branch s5-trackF. Every run: h 0.05, dt 0.0125, sponge 100-200 (O) / 200-400 (R) with the box doubling validated by the build; linear twin to the same T unless stated. Run JSON (full parameters, series, per-period tables, guard extremes) in `runs/`, boundary bisections in `boundary/`, scripts `run_jobs.py`, `run_evolve_guess.py`, `analysis_fa.py`, `make_tables.py`. Families are the existing `outputs/radial5/family_*.json`, not regenerated.

## 0. Read first: definition findings (not rescued, reported as found)

1. **Rule 2.1 "frac at least 0.90" cannot be met by a R start as written.** frac is normalised by the energy at t=0 over the whole box, region 3 hw. Wide-tailed R profiles have frac(t=0) of 0.71 to 0.92 (table below, column frac(t=0)), so frac < 0.90 from the first step. The class label uses the rule as coded (`evolve.classify`). I did not change it; the table shows frac(t=0) next to every frac so the retained fraction can be read against it.
2. **Rule 2.1 "rate on the forbidden side by more than the spectral resolution"**: the resolution is 2 pi / 200 = 0.0314 (core tail of 200 time units). The whole R family spans 0.0054 (bump) and 0.0135 (pinch) below the floor, so this clause cannot be passed by any R start. Zero-crossing rates are quoted instead; late rates of all R runs are 0.9975 to 1.033, i.e. at or above the floor (the surviving energy has become linear waves).
3. **frac at 1000 cycles** in the table is the cycle-averaged region energy over E0 from the per-period table (rule 2.1 wording). `evolve.classify` uses the instantaneous frac series, which differs in the third decimal (O bump 0.99 scale: cycle-averaged 0.9003, instantaneous below 0.90, class "disperses"). Both are stored in `lifetime_all.json`.
4. The first R bump a0.2 run (T=2000) is superseded by `B_R_bump_a0.2_T6600` (checkpoint parameters conflict made a new tag necessary). Both rows are in the lifetime table.
5. Upper-branch R pinch starts (amp 0.4384, 0.4861, 0.51) and the O pinch amp 1.828 start were reached through `run_evolve_guess.py`, a wrapper that passes the family table's eps (sqrt(1 - omega^2)) as the shooting guess, because the CLI's default guess lands on the ansatz wall. No solver code changed.
6. A 1% rule for the O bump baseline fails (0.99 times amplitude disperses), but 1.01, 1.02, 1.05 times amplitude hold longer than the baseline. Wider scans (0.98, 1.02, 1.05, 1.02 to T=100,000) were three diagnostics added after seeing the 1% result.


## 1. F-A Family tables (existing family JSON; rate = omega, hw = half-width)

| family | points | rate range | amplitude param range | folds in rate (index, rate, E) | energy minima (rate, E) | hw range | E range | E/cycle range | nonlinear share range | X_max,cycle max | end of table |
|---|---|---|---|---|---|---|---|---|---|---|---|
| O_bump | 52 | 0.80064 to 0.999950 | 0.2239 to 7.131 | none in rate (fold in amplitude: (46, rate 0.8759, amp 7.131)) | (0.97113, 1659.0) | 2.94 to 56.3 | 1659.0 to 81196 | 10733 to 637204 | -0.286 to -0.000 | 39.544 | solver limit: shooting precision (thin wall) |
| O_pinch | 78 | 0.79675 to 0.999950 | 0.001951 to 2.223 | none in rate (fold in amplitude: (68, rate 0.8880, amp 2.223)) | (0.95580, 7926.2) | 6.02 to 162.9 | 7926.2 to 184019 | 52012 to 1451171 | -0.295 to -0.000 | 30.948 | solver limit: shooting precision (thin wall) |
| R_bump | 36 | 0.99459 to 0.999950 | 0.07086 to 1.005 | (31, 0.99459, 165.0) | (0.99494, 145.7) | 3.00 to 56.2 | 145.7 to 2515 | 920 to 15804 | +0.000 to +0.046 | 1.000 | ansatz wall: omega F_peak -> 1 (0.999877 at the last point) |
| R_pinch (partial) | 104 | 0.98652 to 1.000000 | 0.0006172 to 0.5194 | (58, 0.98652, 494.7) | (0.99725, 307.9) | 4.48 to 162.8 | 307.9 to 14674 | 1940 to 92201 | +0.000 to +0.156 | 0.970 | solver limit: stopped by hand after 102 points (about 20 min): the upp |

| family | branch (index range, rate range) | n | Spearman rho(rate, hw) | rule 2.2 |
|---|---|---|---|---|
| O_bump | all_points_rate_vs_hw | 52 | +0.8569 | no monotone relation |
| O_bump | branch0_idx0-51_w0.80064-0.99995 | 52 | +0.8569 | no monotone relation |
| O_pinch | all_points_rate_vs_hw | 78 | +0.9314 | tighter-slower (floor) |
| O_pinch | branch0_idx0-77_w0.79675-0.99995 | 78 | +0.9314 | tighter-slower (floor) |
| R_bump | all_points_rate_vs_hw | 36 | +0.9949 | tighter-slower (floor) |
| R_bump | branch0_idx0-31_w0.99459-0.99995 | 32 | +1.0000 | tighter-slower (floor) |
| R_bump | branch1_idx31-35_w0.99459-0.99495 | 5 | -1.0000 | n < 8: not scored |
| R_pinch | all_points_rate_vs_hw | 104 | -0.2687 | no monotone relation |
| R_pinch | branch0_idx0-58_w0.98652-0.99995 | 59 | +1.0000 | tighter-slower (floor) |
| R_pinch | branch1_idx58-103_w0.98652-1.00000 | 46 | -0.9870 | tighter-faster (ceiling) |

Notes: R pinch is "partial" (stopped by hand at 104 points; the upper branch tends to a fixed core with omega -> 1, E 362.7 and rising). O families end at the shooting-precision limit (thin wall) at rate 0.80. "Nonlinear share" is negative for O (the sextic and quartic terms lower the energy relative to the quadratic part) and positive for R. Rate relative to 1 (regime check): O 0.80 to 0.99995, R bump 0.99459 to 0.99995, R pinch 0.98652 to 1.0. Spearman is rate against hw per rule 2.2; where the family folds in rate (R bump at 0.99459, R pinch at 0.98652) it is given branch by branch. The R bump upper branch has 5 points (rule needs 8: not scored). O bump rho 0.857 is below the 0.9 threshold because hw has an interior minimum (2.94 at rate 0.95).

Energy per cycle 2 pi E / omega follows E (rate varies by < 1.5% outside the O fold branches): d ln(Epc)/d ln E along the families has median 0.999 (O bump, O pinch, R bump, R pinch alike), see summary.json (`o8_family`).

Plots: `plot_families.png` (E, E per cycle and hw against rate, four families), `plot_bump_vs_pinch_E.png`.

## 2. F-B Lifetimes (production runs; T = 20,000 for O, 2,000 for R, extended where alive)

"frac" = cycle-averaged energy inside 3 hw over E0; "(at)" says whether that is at 1000 cycles or at the end of the run. T_90 and T_death in time units with cycles in brackets (measured periods up to that time). Twin frac is at the same time as the frac column. Late rate is the zero-crossing rate over the last 100 time units. Nonlinear share is the profile's cycle-averaged value (the run value at t 50-150 is in `lifetime_all.json`); for R the X_max over the run is also stored.

| run | model obj | start | hw | E | frac(t=0) | frac, cycle-avg (at) | T_90 time (cycles) | T_death time (cycles) | twin frac (at t) | late rate | nonlinear share (profile) | end class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| O_bump_w0.82 | O bump | w=0.82 | 7.81 | 2.767e+04 | 1.000 | 0.2906 (1000 cyc) | 185.5 (24 cyc) | 429 (56 cyc) | 1.83e-04 @7356.9 | 0.8587 | -0.254 | disperses |
| O_bump_w0.86 | O bump | w=0.86 | 4.81 | 7246 | 1.000 | 0.4429 (1000 cyc) | 1680 (230 cyc) | 4083 (567 cyc) | 5.51e-05 @7104.9 | 0.9052 | -0.195 | disperses |
| O_bump_w0.91 | O bump | w=0.909149280617113 | 3.34 | 2802 | 0.997 | 0.9599 (1000 cyc) | 1.063e+04 (1542 cyc) | > 20000 | 2.61e-05 @6902.64 | 0.9582 | -0.133 | held |
| O_bump_w0.94 | O bump | w=0.94 | 2.97 | 1938 | 0.987 | 0.8276 (1000 cyc) | 1985 (298 cyc) | 8279 (1260 cyc) | 3.52e-05 @6588.23 | 1.0007 | -0.094 | disperses |
| O_bump_w0.965 | O bump | w=0.965 | 3.01 | 1664 | 0.967 | 0.0004 (1000 cyc) | 1791 (275 cyc) | 1854 (285 cyc) | 7.21e-05 @6325.2 | 1.0004 | -0.059 | disperses |
| O_bump_w0.98 | O bump | w=0.98 | 3.43 | 1729 | 0.949 | 0.0006 (1000 cyc) | 2657 (408 cyc) | 2722 (418 cyc) | 2.11e-04 @6357.21 | 1.0007 | -0.036 | disperses |
| O_bump_w0.995 | O bump | w=0.995 | 5.91 | 2726 | 0.926 | 0.0102 (1000 cyc) | 388 (61 cyc) | 575 (91 cyc) | 5.12e-03 @6272.31 | 0.9998 | -0.010 | disperses |
| O_pinch_a1.828 | O pinch | amp=1.828 | 10.2 | 5.961e+04 | 1.000 | 0.5099 (1000 cyc) | 233 (30 cyc) | > 20000 | 4.97e-06 @7473.3 | 0.8427 | -0.258 | alive at cap |
| O_pinch_w0.86 | O pinch | w=0.86 | 7.12 | 1.982e+04 | 1.000 | 0.9818 (1000 cyc) | > 20000 | > 20000 | 1.83e-07 @7299.91 | 0.8618 | -0.202 | held |
| O_pinch_w0.86_T100k | O pinch | w=0.86 | 7.12 | 1.982e+04 | 1.000 | 0.9818 (1000 cyc) | 9.083e+04 (12465 cyc) | 95412 (13105 cyc) | - | 0.9423 | -0.202 | disperses |
| O_pinch_w0.91 | O pinch | w=0.91 | 6.04 | 1.001e+04 | 1.000 | 0.7956 (1000 cyc) | 794 (115 cyc) | > 20000 | 6.36e-08 @6719.02 | 0.9535 | -0.139 | alive at cap |
| O_pinch_w0.94 | O pinch | w=0.94 | 6.21 | 8207 | 1.000 | 0.9643 (1000 cyc) | 1.362e+04 (2056 cyc) | 13666 (2063 cyc) | 1.52e-07 @6652.03 | 1.0040 | -0.099 | disperses |
| O_pinch_w0.965 | O pinch | w=0.965 | 7.17 | 8029 | 1.000 | 0.0001 (1000 cyc) | 417 (64 cyc) | 462 (71 cyc) | 2.11e-06 @6261.12 | 1.0008 | -0.062 | disperses |
| O_pinch_w0.98 | O pinch | w=0.98 | 8.86 | 9029 | 1.000 | 0.0024 (1000 cyc) | 3262 (491 cyc) | 4110 (621 cyc) | 8.77e-06 @6463.72 | 1.0012 | -0.037 | disperses |
| O_pinch_w0.995 | O pinch | w=0.995 | 16.6 | 1.545e+04 | 1.000 | 0.5274 (1000 cyc) | 727.5 (111 cyc) | 8035 (1205 cyc) | 5.35e-04 @6694.36 | 1.0016 | -0.010 | disperses |
| R_bump_a0.2 | R bump | amp=0.2 | 19.8 | 882.9 | 0.915 | 0.8979 (end (317 cyc)) | 1945 (309 cyc) | > 2000 | 2.23e-01 @1993.76 | 1.0001 | +0.002 | alive at cap |
| R_bump_a0.2_T6600 | R bump | amp=0.2 | 19.8 | 882.9 | 0.915 | 0.1374 (1000 cyc) | 1945 (309 cyc) | 4106 (653 cyc) | 2.06e-02 @6284.34 | 1.0000 | +0.002 | disperses |
| R_bump_a0.33 | R bump | amp=0.33 | 11.8 | 525.2 | 0.910 | 0.1548 (end (317 cyc)) | 518.5 (82 cyc) | 1304 (207 cyc) | 2.85e-02 @1992.75 | 1.0003 | +0.007 | disperses |
| R_bump_a0.55 | R bump | amp=0.55 | 6.83 | 298.2 | 0.895 | 0.0046 (end (317 cyc)) | 0.5 (0 cyc) | 376 (59 cyc) | 1.88e-03 @1992.03 | 1.0002 | +0.018 | disperses |
| R_bump_a0.79 | R bump | amp=0.79 | 4.38 | 188.5 | 0.851 | 0.0004 (end (317 cyc)) | 0.5 (0 cyc) | 140 (22 cyc) | 2.03e-04 @1991.66 | 1.0002 | +0.036 | disperses |
| R_bump_a0.90 | R bump | amp=0.90 | 3.61 | 158.1 | 0.803 | 0.0001 (end (317 cyc)) | 0.5 (0 cyc) | 102 (16 cyc) | 9.16e-05 @1991.54 | 1.0000 | +0.044 | disperses |
| R_bump_a0.95 | R bump | amp=0.95 | 3.3 | 148.8 | 0.766 | 0.0001 (end (317 cyc)) | 0.5 (0 cyc) | 94 (14 cyc) | 6.89e-05 @1991.5 | 1.0000 | +0.046 | disperses |
| R_bump_a1.0 | R bump | amp=1.0 | 3.02 | 145.6 | 0.714 | 0.0001 (end (317 cyc)) | 0.5 (0 cyc) | 90 (14 cyc) | 5.67e-05 @1991.48 | 0.9999 | +0.046 | disperses |
| R_pinch_a0.0384 | R pinch | amp=0.0384 | 20.4 | 1802 | 1.000 | 0.0465 (end (317 cyc)) | 849.5 (135 cyc) | 1152 (183 cyc) | 3.52e-03 @1992.06 | 1.0003 | +0.019 | disperses |
| R_pinch_a0.0916 | R pinch | amp=0.0916 | 12.9 | 1109 | 1.000 | 0.0003 (end (317 cyc)) | 293.5 (46 cyc) | 412 (65 cyc) | 5.36e-05 @1991.06 | 1.0022 | +0.044 | disperses |
| R_pinch_a0.2 | R pinch | amp=0.2 | 8.34 | 661.4 | 0.999 | 0.0000 (end (318 cyc)) | 104.5 (16 cyc) | 153 (24 cyc) | 1.57e-06 @1992.29 | 1.0332 | +0.092 | disperses |
| R_pinch_a0.36 | R pinch | amp=0.36 | 5.76 | 394.1 | 0.994 | 0.0000 (end (319 cyc)) | 45 (7 cyc) | 74 (11 cyc) | 2.55e-07 @1991.48 | 1.0309 | +0.145 | disperses |
| R_pinch_a0.4384 | R pinch | amp=0.4384 | 5.02 | 325.7 | 0.980 | 0.0000 (end (319 cyc)) | 34.5 (5 cyc) | 67 (10 cyc) | 2.18e-07 @1991.11 | 1.0308 | +0.156 | disperses |
| R_pinch_a0.4861 | R pinch | amp=0.4861 | 4.67 | 307.9 | 0.945 | 0.0000 (end (317 cyc)) | 28.5 (4 cyc) | 65 (10 cyc) | 4.27e-07 @1990.76 | 1.0089 | +0.151 | disperses |
| R_pinch_a0.51 | R pinch | amp=0.51 | 4.53 | 320.9 | 0.884 | 0.0000 (end (316 cyc)) | 0.5 (0 cyc) | 66 (9 cyc) | 1.42e-06 @1990.68 | 0.9975 | +0.138 | disperses |

Plot: `plot_frac_vs_time.png` (frac against time, four panels, twin dotted).

Reading: O bump holds only near rate 0.909 (T_90 10,630 = 1,542 cycles; frac 0.597 at T = 20,000). O pinch at 0.86 holds to T = 20,000 (frac 0.972) and to T_90 = 90,832, T_death = 95,412 (about 13,100 cycles) in the extension; its late rate drifts from 0.86 to 0.94 before death. Both O objects disperse at rates 0.94 and above. R: every start disperses; the longest-lived R bump (amp 0.2, rate 0.9996) has T_death 4,106 (653 cycles, twin 2e-2 at 6,284); the matched-energy R bump E = 1000 has T_death 5,403 (860 cycles; twin T_death 1,563). R pinch lifetimes: 65 to 1,152 time units (9 to 183 cycles). No R start is "held" under rule 2.1.

## 3. F-C The 1% rule (Block 6), starts that meet rule 2.1

Only two starts meet rule 2.1: O bump rate 0.909 and O pinch rate 0.86. R: none, so none to rerun.

| start | scale | late rate | T_90 | T_death | frac at 1000 cyc (cycle-avg) | twin frac | end class |
|---|---|---|---|---|---|---|---|
| O bump | 0.99 | 1.0006 | 6880 | 19608 | 0.9003 | 2.7e-05 | disperses |
| O bump | 1.00 | 0.9582 | 10630 | > 20000 | 0.9599 | 2.6e-05 | held |
| O bump | 1.01 | 0.9142 | > 20000 | > 20000 | 0.9881 | 2.5e-05 | held |
| O pinch | 0.99 | 0.8620 | > 20000 | > 20000 | 0.9920 | 1.8e-07 | held |
| O pinch | 1.00 | 0.8618 | > 20000 | > 20000 | 0.9818 | 1.8e-07 | held |
| O pinch | 1.01 | 0.8617 | > 20000 | > 20000 | 0.9688 | 1.9e-07 | held |
| (diagnostic: wider scan, O bump w=0.909, twin to T=2000 only) | | | | | | | |
| O bump | 0.98 | 1.0032 | 4983 | 17222 | 0.8187 (1000 cyc) | 2.0e-04 | disperses |
| O bump | 1.02 | 0.9072 | > 20000 | > 20000 | 0.9968 (1000 cyc) | 2.0e-04 | alive at cap |
| O bump | 1.05 | 0.9058 | > 20000 | > 20000 | 0.9776 (1000 cyc) | 2.0e-04 | alive at cap |
| O bump | 1.02 | 0.9088 | > 100000 | > 100000 | 0.9968 (1000 cyc) | 0.0e+00 | alive at cap |

Reading: O pinch 0.86 passes (held at 0.99, 1.00, 1.01; frac at 1000 cycles 0.992, 0.982, 0.969): a trap by Block 6. O bump 0.909 fails: at 0.99 times amplitude frac at 1000 cycles is 0.9003 cycle-averaged (instantaneous below 0.90, T_90 6,880, T_death 19,608), so by the coded rule it disperses; the baseline is at the edge of a steep amplitude dependence (T_90 4,983 at 0.98, 6,880 at 0.99, 10,630 at 1.00, none at 1.01 to 1.05). It is not a tuned peak that falls on both sides: lifetime rises monotonically from 0.98 to 1.02 and 1.05 is still alive. At 1.02 the bump holds to T = 100,000 (frac 0.9716, 14,437 cycles, T_90 none, rate 0.909). The Session 1 baseline is therefore the low-amplitude edge of a longer-lived range, and the scored statement is "fails the 1% rule as written".

## 4. F-D Fair scan for R away from the family ansatz (boundary5, T = 1000, bisection to 1% in amplitude, with h-refinement of trips by the driver)

| shape | obj | A_lo (non-trip) | A_hi (trip) | rel width | low side: class, T_death (TU) | high side: class, trip time | collapse time, refined (h, h/2, h/4) at threshold | holds to 1000 cycles? |
|---|---|---|---|---|---|---|---|---|
| gauss sigma=2 | bump | 0.92439 | 0.93022 | 0.0063 | disperses, 62.5 | singular, 6.500 | t=3.822 (thr 1e-01): 3.825, 3.825, 3.822 | no: T_death 62.5 TU = 10 cycles at A_lo |
| gauss sigma=3 | bump | 0.84546 | 0.85005 | 0.0054 | disperses, 64.5 | singular, 13.000 | t=12.847 (thr 1e-02): 12.850, 12.850, 12.847 | no: T_death 64.5 TU = 10 cycles at A_lo |
| gauss sigma=5 | bump | 0.60976 | 0.61527 | 0.0090 | disperses, 147.5 | singular, 45.000 | t=42.003 (thr 1e-01): 42.013, 42.006, 42.003 | no: T_death 147.5 TU = 23 cycles at A_lo |
| shell r0=10 sigma=2 v_in=0 | bump | 0.21300 | 0.21468 | 0.0079 | disperses, 129.0 | singular, 34.000 | t=36.131 (thr 1e-02): -, 36.131, 36.131 | no: T_death 129.0 TU = 21 cycles at A_lo |
| D_R_pinch_gauss_s3 | pinch | 0.3 | 1.5 | - | bracket not valid: A=0.3 disperses; A=1.5 disperses | | | |
| gauss sigma=3 | pinch | 1.71538 | 1.72545 | 0.0059 | disperses, 97.5 | 3D ill-posed, 37.913 | 3D ill-posed: trip t=37.913 (no h-refinement by the driver; see D_pinch_trip) | no: T_death 97.5 TU = 16 cycles at A_lo |

Notes. The first pinch bracket (0.3, 1.5) was invalid (both ends disperse, the row marked bracket not valid); the second attempt with A_hi = 6 converged (second pinch row). Collapse times are first crossings of min(1 - X) below the stated threshold at h = 0.05, 0.025, 0.0125 (the deepest threshold common to the finest two grids); the columns agree to 0.1% (sigma 2, 3, 5) except the shell (1e-2: 36.131 at both finer grids, none on the coarsest). Class of each trip: singular, centre collapse (the bump; shell collapses at the centre as well) with t* above. The pinch trips are "3D ill-posed" (transverse condition, r about 2.3, off the centre): the driver does not refine this class, so I ran the trip start (A = 1.72545) at h = 0.05, 0.025, 0.0125 (`runs/D_pinch_trip_h*.json`): trip at t = 37.9125, 37.925, 37.925, r = 2.325, 2.3125, 2.319, min(1 - X) = 0.046 at that time: converged in h, a property of the solution, not of the grid. X stays well below 1 there.
Low side of every boundary: dispersal with T_death 62.5 to 147.5 time units (10 to 23 cycles); high side: trip, with trip time growing as the boundary is approached (sigma 5: 14.5, 23.5, 30, 45; shell: 16.5, 18.5, 21, 24, 26.5, 29.5, 34; pinch 11.2, 18.6, 28.2, 37.9) while the low-side T_death grows more slowly (sigma 5: 86, 90, 118, 140, 147.5; pinch 59, 62, 69, 84, 97.5; sigma 2 and 3 flat at 62 to 65). No start within 1% of any boundary lives 1000 cycles: the best low-side lifetime is 147.5 time units (23 cycles). The boundary amplitudes are for Gaussian shapes at rest: sigma 2 0.927, sigma 3 0.848, sigma 5 0.612 (A sigma^1.5-ish scaling not fitted), shell r0 = 10 sigma 2: 0.214, pinch sigma 3: 1.72. Plot: `plot_boundary_scan.png`.

## 5. F-E Pinch against bump, matched rate and matched energy

| model | pairing | bump run: rate, E, hw | pinch run: rate, E, hw | E ratio pinch/bump | hw ratio | bump: T_90, T_death, frac_end | pinch: T_90, T_death, frac_end | bump end class | pinch end class |
|---|---|---|---|---|---|---|---|---|---|
| O | matched rate 0.86 | w=0.8600, 7246, 4.81 | w=0.8600, 19821, 7.12 | 2.74 | 1.48 | 1680, 4083, 0.418 | > 20000, > 20000, 0.972 | disperses | held |
| O | matched rate 0.91 | w=0.9091, 2802, 3.34 | w=0.9100, 10012, 6.04 | 3.57 | 1.80 | 10630, > 20000, 0.597 | 794, > 20000, 0.785 | held | alive at cap |
| O | matched rate 0.94 | w=0.9400, 1938, 2.97 | w=0.9400, 8207, 6.21 | 4.23 | 2.09 | 1985, 8279, 0.000 | 13620, 13666, 0.000 | disperses | disperses |
| O | matched rate 0.965 | w=0.9650, 1664, 3.01 | w=0.9650, 8029, 7.17 | 4.82 | 2.38 | 1791, 1854, 0.000 | 417, 462, 0.000 | disperses | disperses |
| O | matched rate 0.98 | w=0.9800, 1729, 3.43 | w=0.9800, 9029, 8.86 | 5.22 | 2.58 | 2657, 2722, 0.000 | 3262, 4110, 0.000 | disperses | disperses |
| O | matched rate 0.995 | w=0.9950, 2726, 5.91 | w=0.9950, 15450, 16.62 | 5.67 | 2.81 | 388, 575, 0.000 | 728, 8035, 0.000 | disperses | disperses |
| O | matched rate 0.82 | w=0.8200, 27667, 7.81 | w=0.8193, 59609, 10.23 | 2.15 | 1.31 | 186, 429, 0.286 | 233, > 20000, 0.504 | disperses | alive at cap |
| O | matched E ~ 2.0e4 (narrow) | w=0.8312, 17467, 6.64 | w=0.8600, 19821, 7.12 | 1.13 | 1.07 | 232, 2008, 0.451 | > 20000, > 20000, 0.972 | disperses | held |
| O | matched E ~ 1.0e4 (narrow) | w=0.8531, 8682, 5.14 | w=0.9120, 9837, 6.03 | 1.13 | 1.17 | 15410, > 20000, 0.898 | 911, > 20000, 0.799 | alive at cap | alive at cap |
| O | matched E ~ 1.0e4 (wide) | w=0.9997, 9997, 22.29 | w=0.9854, 9995, 10.14 | 1.00 | 0.45 | 2900, 5090, 0.027 | 640, 7524, 0.000 | disperses | disperses |
| R | matched E ~ 1.0e3 | w=0.9997, 999, 22.38 | w=0.9921, 999, 11.76 | 1.00 | 0.53 | 2612, 5403, 0.269 | 236, 333, 0.000 | disperses | disperses |
| R | matched E ~ 5.3e2 | w=0.9989, 525, 11.84 | w=0.9865, 532, 7.09 | 1.01 | 0.60 | 518, 1304, 0.153 | 72, 106, 0.000 | disperses | disperses |
| R | near-matched rate ~0.995 (bump 0.9950 a=0.79; pinch a=0.4384? rate .9932) | w=0.9950, 189, 4.38 | w=0.9932, 326, 5.02 | 1.73 | 1.15 | 0, 140, 0.000 | 34, 67, 0.000 | disperses | disperses |

Family-level (interpolated, from the family tables):

| model | rate | bump E | bump hw | pinch E (branch) | pinch hw | E ratio pinch/bump |
|---|---|---|---|---|---|---|
| O | 0.9500 | 1790.2 | 2.94 | 7976.6 (br 0) | 6.46 | 4.46 |
| O | 0.9700 | 1659.9 | 3.10 | 8208.5 (br 0) | 7.56 | 4.95 |
| O | 0.9800 | 1728.9 | 3.43 | 9037.5 (br 0) | 8.87 | 5.23 |
| O | 0.9900 | 2088.0 | 4.40 | 11512.7 (br 0) | 12.01 | 5.51 |
| O | 0.9950 | 2729.1 | 5.92 | 15462.7 (br 0) | 16.63 | 5.67 |
| O | 0.9980 | 4119.2 | 9.10 | 23706.0 (br 0) | 25.97 | 5.75 |
| O | 0.9990 | 5736.4 | 12.74 | 33202.7 (br 0) | 36.59 | 5.79 |
| R | 0.9950 | 186.4 | 4.32 | 1352.2 (br 0) | 15.49 | 7.25 |
| R | 0.9950 | 186.4 | 4.32 | 314.1 (br 1) | 4.85 | 1.69 |
| R | 0.9960 | 231.2 | 5.35 | 1542.5 (br 0) | 17.54 | 6.67 |
| R | 0.9960 | 231.2 | 5.35 | 310.1 (br 1) | 4.77 | 1.34 |
| R | 0.9980 | 367.3 | 8.35 | 2255.8 (br 0) | 25.30 | 6.14 |
| R | 0.9980 | 367.3 | 8.35 | 309.2 (br 1) | 4.62 | 0.84 |
| R | 0.9990 | 543.4 | 12.25 | 3241.6 (br 0) | 36.14 | 5.96 |
| R | 0.9990 | 543.4 | 12.25 | 316.2 (br 1) | 4.55 | 0.58 |

Reading: at matched rate the O pinch carries 2.2 to 5.8 times the bump's energy and 1.3 to 2.8 times its half-width (family level, 7 rates 0.95 to 0.999: 4.5 to 5.8; run points: 2.1 to 5.7). In R the pinch's lower branch carries 6.0 to 7.3 times the bump's energy at rates 0.995 to 0.999; its upper branch (rate 0.986 to 1) sits at 309 to 363 and crosses the bump's energy near rate 0.9975 (below the bump above that: ratio 0.84 at 0.998, 0.58 at 0.999). The R pinch cannot reach E < 307.9 while the R bump goes down to 145.7. At matched energy the O pinch is the longer-lived object at narrow size (E = 2e4: pinch held, bump T_death 2,008) and similar at E = 1e4 (both alive at 20,000, frac 0.80 pinch against 0.90 bump); the wide pairs (E = 1e4) both die (bump T_death 5,090, pinch 7,524). In R at E = 1000 the bump lives 5,403 against the pinch 333; at E = 533, 1,304 against 106. At matched rate in R the bump also outlives the pinch.

## 6. Verdict table (rule, measured number, verdict)

| Item | Rule (Block 2 / 6) | Measured | Verdict |
|---|---|---|---|
| F1 R bump traps below the floor | at least one R bump start "trapped" (2.1) with rate below 1; 1% rule | 0 of 8 R bump starts hold (7 family points plus the E = 1000 start). Best: amp 0.2 (rate 0.9996 at start) T_death 4,106 TU = 653 cycles, twin 2.1e-2 at 6,284; E = 1000 start T_death 5,403 = 860 cycles, twin T_death 1,563. frac(t=0) 0.71 to 0.92, late rates 0.9999 to 1.0003 (at the floor). Boundary scans: nothing within 1% of the collapse boundary survives past 147.5 TU (23 cycles). | **miss** (a lifetime 3.5 times the twin's but below 1000 cycles, and the resolution clause cannot be met, see section 0) |
| F2 energy minimum at finite amplitude along a trapped family | interior point below both neighbours and ends, families through folds | O bump: E min 1659.0 at rate 0.9711 (93% below the ends); O pinch: 7926.2 at 0.9558 (95% below). Held O starts (0.909, 0.86) lie on the branch beyond the minimum from the wide end, not at it. R bump: 145.65 at 0.99494, 0.1% below the ansatz-wall end (145.9), 12% below the fold value 165.0; R pinch: 307.87 at 0.99725 (15% below the high-rate end 362.7). No R family is trapped. | **split**: hit for O (trapped families); R minima exist in the tables but are not on a trapped family, so not scored; the R bump minimum is marginal |
| F3 pinch exists in O and R, higher energy than the bump at 3+ common rates | pinch "trapped" in O and R; E_pinch > E_bump at 3+ rates | O: pinch 0.86 held (1% rule passes, T_death 95,412 = 13,105 cycles); E_pinch/E_bump 4.5 to 5.8 at 7 rates (family), 2.2 to 5.7 (run points). R: 0 of 9 pinch starts hold (7 family points plus 2 matched-energy) (T_death 65 to 1,152 TU, 9 to 183 cycles); E ratio 6.0 to 7.3 on the lower pinch branch at 0.995 to 0.999, but 1.7 to 0.58 on the upper branch (pinch lower than bump above rate 0.9975). | **split**: hit in O (both parts); miss in R (not trapped), energy part hit on the lower R branch only |
| F4 energy per cycle nearly constant while E and rate drift | over a stretch with dE >= 1%, change in ln Epc <= 0.2 of the change in ln E | O pinch 0.86 (100k): dlnE -0.887, dlnEpc -0.978, ratio 1.10; O pinch 0.86 (20k) E change -1.8%, ratio 1.06; O pinch 1.01: -4.3%, 1.05; O bump 0.909 x1.01: -8.5%, 1.08; O bump x1.02 (100k): -2.6%, 1.07. R: nothing survives to measure. | **miss** (ratio 1.05 to 1.10, 5 times the 0.2 target; the drift is a change of E at nearly fixed rate) |
| O8 F4 misses for any lump losing more than a few percent | ratio dlnEpc/dlnE > 0.2 | same numbers as F4 (1.05 to 1.10); family level, ratio median 0.999 for all four families (fraction of family points with ratio <= 0.2: 0.02 O bump, 0 others). The reason given in Block 4 (|d ln w| much smaller than |d ln E|) holds: dlnw 0.002 to 0.09 against dlnE 0.03 to 0.89. | **hit** |
| O9 R: X < 1 identically in (f, p), only resolution can fail; pinch needs the transverse condition | Block 3 with the Block 4 correction | The X guard approaches 1 only as p -> infinity: R bump sigma 3 A 1.0, t(1 - X < 1e-7) = 7.400 (h/2), 7.291 (h/4): finite-time centre collapse that converges in h (sigma 2, 3, 5 and the shell agree to 0.1% to 0.2%), so not a resolution artifact. Pinch: failure is the transverse condition (3D ill-posed) at r about 2.3, t = 37.9125 / 37.925 / 37.925 at h = 0.05 / 0.025 / 0.0125, min(1 - X) = 0.046: converged in h, X not near 1. Guard tests of the correction work; no "unresolved" class occurred. | **split**: algebra (X < 1 for finite p) and the pinch correction hold; "what can fail is resolution" is a miss, the failures are finite-time p blow-up and the transverse condition |
| O12 3D small-amplitude R family unstable; a trapped R bump sits beyond an E, N minimum | charge-against-rate criterion | Small amplitude: (1 - rate) / F(0)^2 = 0.00996 constant to 0.1% over the six lowest-amplitude points (R bump; O bump 0.000997): rate below 1, quadratic in amplitude. The coefficient 3/16 in the prediction is not recovered (ratio 18.8; it refers to a uniform amplitude, the profile peak amplitude is not that). dN/drate > 0 and dE/drate > 0 from rate 1 to 0.99459 (charge grows toward rate 1: unstable side); signs flip after the fold (dE, dN < 0, rate 0.99471 to 0.99494) and flip back at the E minimum (0.99494). Starts beyond the fold (amp 0.90, 0.95, 1.0): T_death 102, 94, 90 TU: none holds. | **split**: instability sign and rate law hit; coefficient not comparable; no trapped R bump exists beyond the minimum |
| Aaron: optimal length and rate, narrow range | narrow: window < factor 2 in length and rate; optimal: interior lifetime at least 2 times both edges or loss rate at most half; too long: dies within 1000 cycles | O pinch: held rate 0.86 (and the 1% neighbours); 0.82 frac 0.504 and 0.91 frac 0.785 at T = 20,000; 0.94 dies at 13,666 (2,063 cycles); window about 0.82 to 0.94 in rate (factor 1.15) and hw 6 to 10 (factor 1.7). Interior 0.86 loses 3% where the edges lose 21% and 50%. O bump: held only at 0.909 and amplitude +1% to +5%; 0.86 dies at 4,083, 0.94 at 8,279. Too long (rate 0.965 to 0.995): bump T_death 285, 418, 91 cycles; pinch 71, 621, 1,205 cycles (the last exceeds 1000). R: no trapped state. "Too fast" not tested (no push). | **split**: supported for O (narrow, interior optimum, longer objects die within 1000 cycles in 5 of 6 cases); not applicable for R |
| Aaron: single bump not stable enough | supported if it leaves its site or dies within 1000 cycles; contradicted if it stays within one unit and frac >= 0.99 for 32,000 cycles | O bump 0.909: does not die in 20,000 TU (frac 0.597, T_90 1,542 cycles); with amplitude x1.02 frac 0.9716 at T = 100,000 (14,437 cycles). Radial runs cannot see the bump leaving its site. R bump: dies within 90 to 4,106 TU (14 to 653 cycles). | **not decided** in O (neither side of the rule is met, radial upper bound); supported in R |
| Aaron: the pinch (no pass or fail) | report numbers | O pinch exists and is the longest-lived object found (T_death 95,412, 13,105 cycles; E 19,821 at rate 0.86; 4.5 to 5.8 times the bump at equal rate; hw 7.1). R pinch: 9 to 183 cycles. Fails the transverse condition at amplitude 1.72 (sigma 3 Gaussian). | reported |
| Spearman, rule 2.2 (floor or ceiling) | rho <= -0.9 ceiling, >= +0.9 floor | O bump +0.857 (no monotone relation); O pinch +0.931 (floor); R bump lower branch +1.000 (n = 32, floor; upper branch n = 5 not scored); R pinch lower branch +1.000 (n = 59, floor), upper branch -0.987 (n = 46, ceiling), whole curve -0.269 (no monotone relation) | **split**: tighter is slower on the floor branches; the R pinch upper branch (above the fold) is tighter-faster |
| Standards 6, 9: literature | see `docs/S5_TRACKF_BRIEF.md` | not compared in this package (no literature work in WP-D). Block 1 F rows have no scholar annotation filled in. | not tested |

## 7. Not run, deviations, open issues

Not run: R starts at 1000 cycles beyond the 6,600 TU extension cap (none was alive); R pinch start through the R family beyond the partial table; shell starts with inward velocity; Gaussian pinch at a second sigma; push tests; literature comparison (standard 9). The boundary scan used T = 1000 as specified: the high-A "not trip" side was always dispersal, never "alive at cap".
Deviations: section 0 (frac definition, resolution clause, class from instantaneous frac, superseded run, guess wrapper, diagnostics after the 1% result).
Open: (1) rule 2.1 needs a retained-fraction definition (frac over frac at t = 0) and a rate clause that fits a 0.005-wide window before the R result can be scored as "held" at all; I did not change it; (2) the R bump collapse thresholds for the sigma 3 and shell cases are only 1e-2 deep common to two grids; (3) the 100,000 TU runs of O have no twin (O twin dies by T about 100); (4) O pinch 0.86 and O bump x1.02 are alive after 100,000 TU only in the sense frac 0.4 (pinch dead at 95,412) and 0.97 (bump); the bump could be extended.
