# Track F follow-up diagnostics (DF1 to DF4) and added table text

## Added after review (follow-up to the reviewer's recomputation; numbers only, no new verdicts)

All production rows above stand. Diagnostics live in `diagnostics/` (runs, `diag_metrics.json`, `items.json`, `rule21_columns.md/json`, `DIAGNOSTICS.md`).

### A1. Rule 2.1 two ways, every lifetime row (B, C and E runs)

Column A is the rule by the letter: cycle-averaged frac at 1,000 integrated cycles (the 1,000th row of the per-period table; for runs shorter than 1,000 cycles the last row, marked by "cycles used"), the zero-crossing rate at that cycle (median of 21 periods), the twin at the same time; the flags are (frac >= 0.90, twin <= 0.5, rate below 1 by more than 0.0314). Column B is the retained-fraction reading frac/frac(0) with the same rate and twin.

Two classifier defects (in `evolve.classify`, not edited): (1) T_death short-circuit: any run with a T_death, however late, returns "disperses" before the 1,000-cycle test is made (O pinch 0.94: T_death 13,666 TU = 2,063 cycles, O bump 0.99 scale: 19,608, O pinch 0.995: 8,035). (2) The 1,000-cycle time is 1000 x 2 pi / (rate over the last 100 time units), the end-of-run rate, not the integrated count: O pinch 0.94 gives 6,258 against 6,652 integrated (late rate 1.0040 after dispersal); O bump 0.99 scale 6,280 against 6,882; O bump 0.909 6,558 against 6,903.

| run | cycles used | frac(0) | A: frac cycle-avg at cycle N | A: rate at cycle N | A: twin at same t | A: 2.1 by the letter (frac, twin, rate) | B: retained frac/frac(0) | B: zero-crossing rate | B: retained >= 0.90 and twin <= 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| B_O_bump_w0.82 | 1000 | 1.000 | 0.2906 | 0.8581 | 1.83e-04 | no (n,y,y) | 0.2906 | 0.8581 | no |
| B_O_bump_w0.86 | 1000 | 1.000 | 0.4429 | 0.9015 | 5.51e-05 | no (n,y,y) | 0.4429 | 0.9015 | no |
| B_O_bump_w0.91 | 1000 | 0.997 | 0.9599 | 0.9121 | 2.61e-05 | yes (y,y,y) | 0.9623 | 0.9121 | yes |
| B_O_bump_w0.94 | 1000 | 0.987 | 0.8276 | 0.9643 | 3.52e-05 | no (n,y,y) | 0.8384 | 0.9643 | no |
| B_O_bump_w0.965 | 1000 | 0.967 | 0.0004 | 1.0020 | 7.21e-05 | no (n,y,n) | 0.0004 | 1.0020 | no |
| B_O_bump_w0.98 | 1000 | 0.949 | 0.0006 | 1.0010 | 2.11e-04 | no (n,y,n) | 0.0006 | 1.0010 | no |
| B_O_bump_w0.995 | 1000 | 0.926 | 0.0102 | 1.0013 | 5.12e-03 | no (n,y,n) | 0.0110 | 1.0013 | no |
| B_O_pinch_a1.828 | 1000 | 1.000 | 0.5099 | 0.8423 | 4.97e-06 | no (n,y,y) | 0.5099 | 0.8423 | no |
| B_O_pinch_w0.86 | 1000 | 1.000 | 0.9818 | 0.8613 | 1.83e-07 | yes (y,y,y) | 0.9818 | 0.8613 | yes |
| B_O_pinch_w0.86_T100k | 1000 | 1.000 | 0.9818 | 0.8613 | - | no (y,n,y) | 0.9818 | 0.8613 | no |
| B_O_pinch_w0.91 | 1000 | 1.000 | 0.7956 | 0.9459 | 6.36e-08 | no (n,y,y) | 0.7956 | 0.9459 | no |
| B_O_pinch_w0.94 | 1000 | 1.000 | 0.9643 | 0.9487 | 1.52e-07 | yes (y,y,y) | 0.9644 | 0.9487 | yes |
| B_O_pinch_w0.965 | 1000 | 1.000 | 0.0001 | 1.0039 | 2.11e-06 | no (n,y,n) | 0.0001 | 1.0039 | no |
| B_O_pinch_w0.98 | 1000 | 1.000 | 0.0024 | 1.0041 | 8.77e-06 | no (n,y,n) | 0.0024 | 1.0041 | no |
| B_O_pinch_w0.995 | 1000 | 1.000 | 0.5274 | 0.9550 | 5.35e-04 | no (n,y,y) | 0.5275 | 0.9550 | no |
| B_R_bump_a0.2 | 317 | 0.915 | 0.8979 | 1.0001 | 2.23e-01 | no (n,y,n) | 0.9818 | 1.0001 | yes |
| B_R_bump_a0.2_T6600 | 1000 | 0.915 | 0.1374 | 1.0001 | 2.06e-02 | no (n,y,n) | 0.1503 | 1.0001 | no |
| B_R_bump_a0.33 | 317 | 0.910 | 0.1548 | 1.0003 | 2.85e-02 | no (n,y,n) | 0.1701 | 1.0003 | no |
| B_R_bump_a0.55 | 317 | 0.895 | 0.0046 | 1.0002 | 1.88e-03 | no (n,y,n) | 0.0052 | 1.0002 | no |
| B_R_bump_a0.79 | 317 | 0.851 | 0.0004 | 1.0001 | 2.03e-04 | no (n,y,n) | 0.0004 | 1.0001 | no |
| B_R_bump_a0.90 | 317 | 0.803 | 0.0001 | 0.9998 | 9.16e-05 | no (n,y,n) | 0.0002 | 0.9998 | no |
| B_R_bump_a0.95 | 317 | 0.766 | 0.0001 | 0.9997 | 6.89e-05 | no (n,y,n) | 0.0001 | 0.9997 | no |
| B_R_bump_a1.0 | 317 | 0.714 | 0.0001 | 0.9997 | 5.67e-05 | no (n,y,n) | 0.0001 | 0.9997 | no |
| B_R_pinch_a0.0384 | 317 | 1.000 | 0.0465 | 1.0003 | 3.52e-03 | no (n,y,n) | 0.0465 | 1.0003 | no |
| B_R_pinch_a0.0916 | 317 | 1.000 | 0.0003 | 1.0010 | 5.36e-05 | no (n,y,n) | 0.0003 | 1.0010 | no |
| B_R_pinch_a0.2 | 318 | 0.999 | 0.0000 | 1.0402 | 1.57e-06 | no (n,y,n) | 0.0000 | 1.0402 | no |
| B_R_pinch_a0.36 | 319 | 0.994 | 0.0000 | 1.0264 | 2.55e-07 | no (n,y,n) | 0.0000 | 1.0264 | no |
| B_R_pinch_a0.4384 | 319 | 0.980 | 0.0000 | 1.0241 | 2.18e-07 | no (n,y,n) | 0.0000 | 1.0241 | no |
| B_R_pinch_a0.4861 | 317 | 0.945 | 0.0000 | 1.0094 | 4.27e-07 | no (n,y,n) | 0.0000 | 1.0094 | no |
| B_R_pinch_a0.51 | 316 | 0.884 | 0.0000 | 1.0016 | 1.42e-06 | no (n,y,n) | 0.0000 | 1.0016 | no |
| C_O_bump_w0.91_x0.98 | 1000 | 0.997 | 0.8187 | 0.9269 | 1.98e-04 | no (n,y,y) | 0.8208 | 0.9269 | no |
| C_O_bump_w0.91_x0.99 | 1000 | 0.997 | 0.9003 | 0.9180 | 2.69e-05 | yes (y,y,y) | 0.9026 | 0.9180 | yes |
| C_O_bump_w0.91_x1.01 | 1000 | 0.997 | 0.9881 | 0.9089 | 2.53e-05 | yes (y,y,y) | 0.9906 | 0.9089 | yes |
| C_O_bump_w0.91_x1.02 | 1000 | 0.997 | 0.9968 | 0.9071 | 1.98e-04 | yes (y,y,y) | 0.9993 | 0.9071 | yes |
| C_O_bump_w0.91_x1.02_T100k | 1000 | 0.997 | 0.9968 | 0.9071 | - | no (y,n,y) | 0.9993 | 0.9071 | no |
| C_O_bump_w0.91_x1.05 | 1000 | 0.997 | 0.9776 | 0.9050 | 1.98e-04 | yes (y,y,y) | 0.9801 | 0.9050 | yes |
| C_O_pinch_w0.86_x0.99 | 1000 | 1.000 | 0.9920 | 0.8617 | 1.79e-07 | yes (y,y,y) | 0.9920 | 0.8617 | yes |
| C_O_pinch_w0.86_x1.01 | 1000 | 1.000 | 0.9688 | 0.8610 | 1.85e-07 | yes (y,y,y) | 0.9688 | 0.8610 | yes |
| E_O_bump_E10k_w0.8531 | 1000 | 1.000 | 0.9080 | 0.8583 | 6.73e-04 | yes (y,y,y) | 0.9080 | 0.8583 | yes |
| E_O_bump_E10k_w0.99968 | 1000 | 0.917 | 0.3560 | 0.9364 | 3.36e-01 | no (n,y,y) | 0.3880 | 0.9364 | no |
| E_O_bump_E20k_w0.8312 | 1000 | 1.000 | 0.4579 | 0.8581 | 1.41e-03 | no (n,y,y) | 0.4579 | 0.8581 | no |
| E_O_pinch_E10k_w0.912 | 1000 | 1.000 | 0.8094 | 0.9460 | 3.50e-05 | no (n,y,y) | 0.8094 | 0.9460 | no |
| E_O_pinch_E10k_w0.9854 | 1000 | 1.000 | 0.8051 | 0.9533 | 2.49e-03 | no (n,y,y) | 0.8053 | 0.9533 | no |
| E_R_bump_E1000_a0.17715 | 317 | 0.915 | 0.9115 | 0.9999 | 3.28e-01 | no (y,y,n) | 0.9961 | 0.9999 | yes |
| E_R_bump_E1000_a0.17715_T6600 | 1000 | 0.915 | 0.3099 | 1.0001 | 4.34e-02 | no (n,y,n) | 0.3386 | 1.0001 | no |
| E_R_pinch_E1000_a0.10892 | 317 | 0.999 | 0.0001 | 0.9985 | 2.28e-05 | no (n,y,n) | 0.0001 | 0.9985 | no |
| E_R_pinch_E533_a0.26158 | 319 | 0.998 | 0.0000 | 1.0298 | 6.03e-07 | no (n,y,n) | 0.0000 | 1.0298 | no |

By the letter at 1,000 integrated cycles the following meet rule 2.1: O bump 0.909, O pinch 0.86, **O pinch 0.94 (0.9643, rate 0.9487, twin 1.5e-7)**, O bump 0.909 at 0.99, 1.01, 1.02, 1.05 times amplitude, O pinch 0.86 at 0.99 and 1.01, and also E_O_bump_E10k_w0.8531 (0.9080, a matched-energy run). So the statement "only two starts meet rule 2.1" is replaced by "three family starts (O bump 0.909, O pinch 0.86, O pinch 0.94)", plus the amplitude-scaled neighbours and the matched-energy bump 0.8531. By the retained reading B_R_bump_a0.2 (T = 2000, 317 cycles) and E_R_bump_E1000_a0.17715 (T = 2000) look held, but both die before 1,000 cycles (T6600 rows).

### A2. O bump 0.909, 1% rule wording

At 0.99 times amplitude: frac 0.9003 at the threshold (marginal), T_90 6,880 (t at 1,000 integrated cycles 6,882, so T_90 falls at 999 cycles); at 0.98 clear fail (0.8187). Replaces "fails as written".

### A3. Loss-rate table (checked from my runs; "not a tuned peak" is removed)

Loss rate = -d ln(frac)/dt of the instantaneous 3 hw frac series, fitted between two times. My values for t 2000 to 10000 are within 8% of the reviewer's list except the 1.02 and 1.05 rows (the reviewer's "about 0" matches 4.6e-8; 1.05 is 2.0e-6 here against 2.2e-6).

| scale | frac at T=20,000 (instantaneous) | loss rate -d ln frac/dt, t 2000 to 10000 | t 5000 to 15000 | t 0 to 20000 | reviewer's |
|---|---|---|---|---|---|
| 0.98 | 0.0009 | 4.51e-05 | 3.95e-05 | 3.50e-04 | 4.2e-5 |
| 0.99 | 0.0026 | 3.11e-05 | 4.28e-05 | 2.97e-04 | 2.9e-5 |
| 1.00 | 0.5970 | 9.85e-06 | 3.03e-05 | 2.57e-05 | 9.2e-6 |
| 1.01 | 0.9144 | 1.63e-06 | 2.84e-06 | 4.35e-06 | 1.5e-6 |
| 1.02 | 0.9957 | 4.56e-08 | 6.92e-08 | 8.98e-08 | about 0 |
| 1.05 | 0.9654 | 2.01e-06 | 1.25e-06 | 1.63e-06 | 2.2e-6 |

Session 2 context: `docs/SESSION_02_REPORT.md` line 34 (the narrow dip at rate 0.9065) and lines 78 to 79 (the Session 1 primary on its wall): the baseline start is at that wall, so the rise of lifetime with amplitude above it is consistent with that record.

### A4. frac(0) below 0.90

R bump: 5 of 8 distinct starts (amp 0.55, 0.79, 0.90, 0.95, 1.0: 0.895, 0.851, 0.803, 0.766, 0.714; the others 0.910 to 0.915). R pinch: 1 of 9 (amp 0.51: 0.884; the others 0.945 to 0.9997). Section 0 point 1 applies to those, not to all R starts.

### A5. Collapse-time thresholds

Each collapse time is the first crossing of min(1 - X) below a stated depth, common to the two finest grids: sigma 2: 3.822 at 1e-1; sigma 3: 12.847 at 1e-2 (A = 0.8500) and, at the first high-side point A = 1.0, 7.291 at 1e-7 with h/2 = 7.400 (**h/2 and h/4 differ by 1.5% at depth 1e-7**); sigma 5: 42.003 at 1e-1; shell: 36.131 at 1e-2 (finest two grids only). The 0.1% to 0.2% agreement quoted before holds for the shallow thresholds only.

### A6. Labels

The O pinch 0.86 100,000 TU run (`B_O_pinch_w0.86_T100k`) was added after the 20,000 TU result showed it alive. The "three diagnostics" after the 1% result are four runs (O bump 0.98, 1.02, 1.05 at T = 20,000, and 1.02 at T = 100,000); their twins ran to T = 2,000 only (none for the 100,000 run).

### A7. F2

The O energy minima (bump rate 0.9711, pinch 0.9558) sit at rates where starts disperse (O bump 0.965 and O pinch 0.965 die at 285 and 71 cycles); the held states (0.909, 0.86) are beyond the minimum, on the narrower side. The R pinch upper-branch energy crosses the R bump energy at rate 0.99733 (previously quoted as 0.9975).

### A8. O12 small-amplitude constant

Definition: c = (1 - omega)/F(0)^2 with F(0) the peak amplitude of the radial profile F(r) in f = F(r) cos(omega t) (for the bump F(0) = F_peak, the column checked), omega the family's rate, six lowest-amplitude points of `family_R_bump.json`:

| F(0) (peak, = F_peak) | rate omega | c = (1 - omega)/F(0)^2 |
|---|---|---|
| 0.07086 | 0.99995000 | 0.009957 |
| 0.07654 | 0.99994168 | 0.009956 |
| 0.07874 | 0.99993829 | 0.009955 |
| 0.08169 | 0.99993358 | 0.009954 |
| 0.08569 | 0.99992692 | 0.009953 |
| 0.09119 | 0.99991726 | 0.009951 |

Limit c = 0.009957; 3/16 = 0.1875; ratio 3/16 over c = 18.83. O bump: 0.000997, 0.000997, 0.000997, 0.000997, 0.000997, 0.000997.

### A9. Follow-up runs (DF1 to DF4), sponge 400/800 unless stated; T_death for DF1 is of the nonlinear run (stop-at-death), twin run to the stated T

DF1 (lifetime scaling; R bump family starts at hw 28.6, 37.0, 45.0):

| start | hw | frac(0) | frac cycle-avg at cycle 1000 (or last) | retained = frac/frac(0) | zero-crossing rate at cycle 1000 | cycle at which rate (21-period median) exceeds 1 | T_90 | T_death | T_death/hw^2 | twin T_death | twin T_death/hw^2 | run/twin | twin frac at same t | nonlinear share (run, t 50-150) | run ended |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DF1 amp 0.13887 | 28.62 | 0.9158 | 0.8295 | 0.9057 | 1.0001 | 680 | 4769.5 | 9286.0 | 11.34 | 2549.5 | 3.11 | 3.64 | 2.94e-01 | +0.00121 | T_end 9286; 1000 cycles used |
| DF1 amp 0.107627 | 36.97 | 0.9162 | 0.9139 | 0.9974 | 0.9999 | 1269 | 8769.0 | 16308.5 | 11.93 | 4255.0 | 3.11 | 3.83 | 2.62e-01 | +0.00073 | T_end 16308.5; 1000 cycles used |
| DF1 amp 0.0885 | 44.99 | 0.9165 | 0.9165 | 1.0000 | 0.9999 | 2020 | 14012.5 | 25148.0 | 12.42 | 6304.0 | 3.11 | 3.99 | 4.60e-01 | +0.00049 | T_end 25148; 1000 cycles used |

Production R bump starts with twins (T_death/hw^2): a0.2 hw 19.80: - / 3.12; a0.33 hw 11.84: 9.29 / 3.14; a0.55 hw 6.83: 8.06 / 3.19; a0.79 hw 4.38: 7.34 / 4.65; a0.90 hw 3.61: 7.87 / 6.33; a0.95 hw 3.30: 8.59 / 7.35; a1.0 hw 3.02: 9.81 / 8.71; B_R_bump_a0.2_T6600 hw 19.80 T_death 4106.5 (10.48 hw^2) twin 1223.0 (3.12 hw^2); E_R_bump_E1000_a0.17715_T6600 hw 22.38 T_death 5403.0 (10.78 hw^2) twin 1562.5 (3.12 hw^2).

DF2 (1% neighbours and box, hw 37):

| start | hw | frac(0) | frac cycle-avg at cycle 1000 (or last) | retained = frac/frac(0) | zero-crossing rate at cycle 1000 | cycle at which rate (21-period median) exceeds 1 | T_90 | T_death | T_death/hw^2 | twin T_death | twin T_death/hw^2 | run/twin | twin frac at same t | nonlinear share (run, t 50-150) | run ended |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hw37 x0.99 (400/800) | 36.97 | 0.9162 | 0.8662 | 0.9454 | 1.0001 | 578 | 4960.5 | > 6600 (T_end) | - | 4255.0 | 3.11 | - | 2.62e-01 | +0.00072 | T_end 6600; 1000 cycles used |
| hw37 x1.00 (400/800, DF1 run) | 36.97 | 0.9162 | 0.9139 | 0.9974 | 0.9999 | 1269 | 8769.0 | 16308.5 | 11.93 | 4255.0 | 3.11 | 3.83 | 2.62e-01 | +0.00073 | T_end 16308.5; 1000 cycles used |
| hw37 x1.01 (400/800) | 36.97 | 0.9162 | 0.9187 | 1.0027 | 0.9883 | - | - | guard trip at 4433.5 | - | 4255.0 | 3.11 | - | 4.35e-01 | +0.00074 | T_end 4433.5 (GUARD TRIP, class unresolved); 704 cycles used |
| hw37 x1.00, sponge 800/1600 | 36.97 | 0.9162 | 0.9138 | 0.9973 | 0.9999 | - | - | > 6600 (T_end) | - | 4255.0 | 3.11 | - | 2.62e-01 | +0.00073 | T_end 6600; 1000 cycles used |

DF3 (R pinch, hw about 45, small-amplitude branch, amp 0.008031, interpolated from the family table):

| start | hw | frac(0) | frac cycle-avg at cycle 1000 (or last) | retained = frac/frac(0) | zero-crossing rate at cycle 1000 | cycle at which rate (21-period median) exceeds 1 | T_90 | T_death | T_death/hw^2 | twin T_death | twin T_death/hw^2 | run/twin | twin frac at same t | nonlinear share (run, t 50-150) | run ended |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DF3 R pinch amp 0.008031 (hw ~45), sponge 400/800, T 6600 | 45.00 | 0.9997 | 0.6010 | 0.6011 | 1.0002 | 402 | 5145.0 | > 6600 (T_end) | - | 2301.5 | 1.14 | - | 2.13e-02 | +0.00409 | T_end 6600; 1000 cycles used |

DF4 (O pinch 0.94 and its 1% neighbours, T = 20,000, twin to 2,000 for the neighbours):

| start | hw | frac(0) | frac cycle-avg at cycle 1000 (or last) | retained = frac/frac(0) | zero-crossing rate at cycle 1000 | cycle at which rate (21-period median) exceeds 1 | T_90 | T_death | T_death/hw^2 | twin T_death | twin T_death/hw^2 | run/twin | twin frac at same t | nonlinear share (run, t 50-150) | run ended |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| O pinch 0.94 x0.99 | 6.21 | 1.0000 | 0.9785 | 0.9786 | 0.9525 | 1472 | 9788.0 | 9831.5 | 255.33 | 63.0 | 1.64 | 156.06 | 5.94e-05 | -0.09694 | T_end 20000; 1000 cycles used |
| O pinch 0.94 x1.00 (production, twin to 20000) | 6.21 | 1.0000 | 0.9643 | 0.9644 | 0.9487 | 2050 | 13619.5 | 13665.5 | 354.91 | 63.0 | 1.64 | 216.91 | 1.52e-07 | -0.10361 | T_end 20000; 1000 cycles used |
| O pinch 0.94 x1.01 | 6.21 | 1.0000 | 0.9498 | 0.9498 | 0.9484 | 1573 | 10473.5 | 10522.0 | 273.27 | 63.0 | 1.64 | 167.02 | 5.94e-05 | -0.10681 | T_end 20000; 1000 cycles used |
