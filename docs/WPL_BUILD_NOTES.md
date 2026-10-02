# WP-L build notes (Session 3): L1 and L2

Numerics worker, 2026-10-02. Builds the lone-lump break test: L1 (radial driver for the spherical disturbances a, b, c) and L2 (3D initial data for d and e, and the boosted-lump gate). No production scan was run. Units Set-B-like (m = 1, g4 = 0.30, g6 = 0.09 beta).

Code: `swirl/family/wpl.py` (L1), `swirl/engine/boost.py` (L2; hooked into `initial.build_initial`, `lumps.check_lumps` and `run.py` as the init entry `"wpl"`). Scripts and records: `outputs/session3/wpL/` (`l1_build.py`, `l2_gate.py`, `l2_t0.py`, and the JSON they write). Tests: `tests/family/test_wpl.py`, `tests/engine/test_wpl_initial.py`.

## 1. Starts and the settled state

| start | beta | profile omega | hw | rN | t_s | settled omega | drift over 200 | N (period, ball rN) |
|---|---|---|---|---|---|---|---|---|
| dip | 4/3 | 0.9065045 | 3.393 | 14.29 | 1004.60 | 0.9068848 | 5.8e-8 | 3028.88 |
| primary | 4/3 | 0.9091493 | 3.344 | 14.39 | 1001.64 | 0.9096527 | 4.4e-5 (see below) | 2897.12 |
| b5 | 5 | 0.9755982 | 6.623 | 27.57 | 1004.69 | 0.9756010 | 1.6e-8 | 6403.43 |

- **Settling time.** The profile start (phi = Phi, phi_t = 0, as Session 2) is evolved to T_SETTLE = 1000 and on to the next maximum of phi(0, t); that step is t_s (phase 0). The Session 2 long runs show omega flat to 1e-6 from t of about 300 at the dip and at beta = 5, so 1000 is "several hundred and more". The primary is not in a steady state at any time: it loses energy at about 4e-6 per unit time and dies at 10,630, so its omega drifts slowly whatever t_s is; the same t_s = 1000 is used and the control carries the drift.
- **Phase 90** is the step nearest the next downward zero of phi(0) (k90 = 139, 139 and 129 steps after t_s). A velocity kick is applied there (phi_t is largest); a field kick, the removal and the packet launch at phase 0. Measured phases at the kick: 359.5 and 90.0 degrees.
- rN = max(3 hw, hw + 4.6/kappa) (WP-A r_lump): the ball that holds the tail to 1e-4 of E.
- Each box scale settles its own copy from t = 0 (box doubling compares two independent settles).

## 2. L1: what was built

- **Driver** `wpl.run_one(spec)`: from the settled state, optional disturbance at t_d, absorber on, checkpoint/resume (`ckpt`, bit-identical), `linear=True` for the linear twin (g4 = g6 = 0 from the same disturbed state), `scale` for the box (2 = doubled), process pool `wpl.scan(specs, workers)` (one thread per worker, resumable, results named by `spec_name`).
- **The RK4 kernel** is radial._advance with per-step records added; it is bit-identical to `Radial.step` (test).
- **Disturbances** (module docstring has the formulas):
  - (a) field: phi -> phi (1 + eps w), w = 1 to 2 hw, C-infinity taper to 0 at 3 hw, phase 0. (a) velocity: the same on phi_t, phase 90.
  - (b) removal: (phi, phi_t) -> (1 - s c) (phi, phi_t), c = 1 to 1 hw, taper to 0 at 1.5 hw, phase 0. Distinct from (a): both components are scaled (independent of the phase) and only the core is cut, so the profile is dented rather than rescaled. s is the fraction by which phi and phi_t are scaled in the core, not a fraction of N. The ball r < 1.5 hw holds most of N, so Delta N / N is close to -(2 s - s^2) x 0.76 (s = 0.2 removes 28% of N; table below).
  - (c) ingoing Gaussian packet built with Omega = sqrt(m^2 - d_r^2) by FFT: the exact ingoing solution of the linear u-equation. Slow: k0 = 0.3, sigma = 20 (k0 sigma = 6). Third harmonic: k0 = k3 = sqrt(9 omega_s^2 - 1) (2.530, 2.539, 2.751), sigma = 5. Launch R0 = rN + 10 + 4 sigma. The slow packet needs a larger box: scale 2.25 at beta = 4/3, 1.25 at beta = 5 (`packet_scale`). Scan variable: efrac = E_packet / E(r < rN).
- **N estimators.** N_inst (instantaneous) and N per period (= (1/2 pi) times the time integral of int w phi_t^2 over one period between upward zero crossings of phi(0), the Session 2 estimator <int phi_t^2>/omega), for the whole box, the ball rN and the ball 3 hw. The t = 0 table uses N_inst at the disturbance phase; the runs report the period N against the control.
- **E + absorbed:** the sponge dissipation 4 pi dr sum sigma v^2 is integrated with the RK4 stage weights.
- **Crossing times** are the Schmitt-trigger crossings of WP-A refined by the zero of the cubic through the four samples around them. The running totals (absorbed, C) are accumulated step by step, so the records do not depend on where a run is split into chunks (a kick, a checkpoint).
- **Per run:** frac (E(r < 3 hw)/its value just before t_d), E(r < 3 hw), E(r < rN), omega per period, N per period, windowed loss rate (W = 500 from t_s + 250), E_h + absorbed before and after t_d, the kick record (pre and post), regime (omega/m after the kick, nonlinear share just after the kick and its mean). `wpl.compare(run, control)` gives the broken-outright test with the control's value, Delta N / N, Delta E / E and the omega shift in windows, and the loss rates side by side.

## 3. L1 checks

Every value is the worst over the record.

| Check | Target | Measured | Verdict |
|---|---|---|---|
| Kernel against radial.Radial.step, 2000 steps | identical | identical u and v | pass |
| N estimator on an exact standing mode of the 4th-order operator (linear, no sponge, 9 periods) | 1e-8 | N 2.4e-8, omega 3.1e-10 | N: pass with note (2.4x; section 9) |
| epsilon = 0 against the control (field, velocity at phase 90, removal, packet with efrac = 0) | bit for bit | raw series, crossing records and final fields identical (frac is normalised at t_d, so for the phase-90 kick it is compared through E(r < 3 hw)) | pass |
| t = 0 Delta E, quadrature (Simpson, 4th-order derivative) against the run at t_d (E_h), dip and b5, 5 disturbances each | 1e-4 relative | worst 6.4e-6 (k3 packet; field kicks 7e-10, removal 1.9e-6) | pass |
| t = 0 Delta N, same | 1e-4 relative | worst 1.5e-8 (removal); packets: |Delta N run - quad| / N below 1e-9 | pass |
| E + absorbed after the disturbance (field +0.1, T = 300) | 1e-5 (Session 2 level) | 6e-9; demos (T = 2000) worst 1.6e-7 (removal 0.2) | pass |
| Linear response: odd part of Delta N / N (window t_d + 50 to 250, against the control) at eps = 0.01 over eps = 0.005 | 2, to 1e-3 relative | 1.99977 (1.2e-4); window 250 to 390: 1.99949 (2.5e-4); even part / odd part at eps = 0.01: 1.8e-3 | pass |
| Box doubling, dip, field +0.1, T = 2000, R_box 200 against 400 | 1e-4 on frac | 9.6e-6 (at t - t_d = 1455) | pass |
| Packet energy: quadrature against 2 pi^1.5 sigma A^2 (m^2 + k0^2 + 1/(2 sigma^2)) | 1e-9 | 1.3e-15; E_h (4th-order SBP) against it: 1.3e-6 (k3), 3e-11 (slow) | pass |
| Free packet: energy-centroid speed against the exact P/E of the spectrum | 1e-3 | 4.1e-5 to 7.1e-5 | pass |
| Radial whole-box N against the 3D A9 run (Session 2), t >= 300 | 1e-4 | 3.4e-5 | pass |
| Same, t = 100 and 200 | — | 1.2e-3 and 1.2e-4: not an estimator error. The radial box (sponge from 100) still holds the start-up radiation, 1.3e-3 of N outside rN at t = 100, which the 3D box (sponge from 23) has already absorbed | note |
| N estimator jitter, per-period N in the ball rN about a linear trend, t = 500 to 1000 | — | rms 5.4e-7 (dip), 4.6e-6 (primary), 4.5e-7 (b5); max 1.2e-6, 1.0e-5, 9.1e-7 | — |
| Checkpoint resume | identical | identical; a checkpoint of another run is refused | pass |
| Broken-outright logic on synthetic series | — | as defined | pass |


**pytest** (worktree, NUMBA_NUM_THREADS = 2; the new tests plus the touched engine and family tests: test_checkpoint_resume, test_lumps, test_lump_floor, test_lumps_k, the snapshot guards and k3 resume, test_radial, test_wpa_run): 84 passed, 1 xfailed in 225 s. The xfail is WP-A's strict record of the beta = 0.75 box miss. The snapshot and WP-C tests rewrite their check records under outputs/session2/; those were restored and not committed.

## 4. t = 0 table (no PDE after settling)

`outputs/session3/wpL/t0_table.json`. Delta N / N is N_inst in the ball rN at the disturbance phase; Delta E / E is over E(r < rN).

**(a) field (phase 0) and (a) velocity (phase 90).**

| eps | dip field dN/N | dip field dE/E | dip vel dN/N | dip vel dE/E | primary field dN/N | primary field dE/E | primary vel dN/N | primary vel dE/E | b5 field dN/N | b5 field dE/E | b5 vel dN/N | b5 vel dE/E |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -0.3 | -0.5065 | -0.4532 | -0.5068 | -0.5068 | -0.5060 | -0.4530 | -0.5063 | -0.5063 | -0.5068 | -0.4933 | -0.5069 | -0.5069 |
| -0.2 | -0.3575 | -0.3156 | -0.3577 | -0.3577 | -0.3571 | -0.3152 | -0.3574 | -0.3573 | -0.3577 | -0.3471 | -0.3578 | -0.3578 |
| -0.1 | -0.1887 | -0.1662 | -0.1888 | -0.1888 | -0.1884 | -0.1657 | -0.1886 | -0.1886 | -0.1888 | -0.1831 | -0.1888 | -0.1888 |
| -0.05 | -0.0968 | -0.0857 | -0.0969 | -0.0969 | -0.0967 | -0.0853 | -0.0968 | -0.0968 | -0.0969 | -0.0940 | -0.0969 | -0.0969 |
| -0.02 | -0.0393 | -0.0350 | -0.0393 | -0.0393 | -0.0393 | -0.0348 | -0.0393 | -0.0393 | -0.0393 | -0.0382 | -0.0393 | -0.0393 |
| -0.01 | -0.0198 | -0.0176 | -0.0198 | -0.0198 | -0.0197 | -0.0175 | -0.0198 | -0.0197 | -0.0198 | -0.0192 | -0.0198 | -0.0198 |
| +0.01 | +0.0200 | +0.0179 | +0.0200 | +0.0200 | +0.0199 | +0.0178 | +0.0199 | +0.0199 | +0.0200 | +0.0194 | +0.0200 | +0.0200 |
| +0.02 | +0.0401 | +0.0360 | +0.0401 | +0.0401 | +0.0401 | +0.0358 | +0.0401 | +0.0401 | +0.0401 | +0.0391 | +0.0401 | +0.0401 |
| +0.05 | +0.1018 | +0.0922 | +0.1018 | +0.1018 | +0.1016 | +0.0915 | +0.1017 | +0.1017 | +0.1018 | +0.0994 | +0.1018 | +0.1018 |
| +0.1 | +0.2085 | +0.1928 | +0.2086 | +0.2086 | +0.2082 | +0.1908 | +0.2084 | +0.2084 | +0.2086 | +0.2045 | +0.2086 | +0.2086 |
| +0.2 | +0.4368 | +0.4282 | +0.4370 | +0.4370 | +0.4363 | +0.4216 | +0.4366 | +0.4366 | +0.4371 | +0.4344 | +0.4371 | +0.4371 |
| +0.3 | +0.6849 | +0.7303 | +0.6853 | +0.6853 | +0.6841 | +0.7148 | +0.6846 | +0.6846 | +0.6854 | +0.6958 | +0.6855 | +0.6854 |

**(b) removal (phase 0, core r < 1 hw, taper to 1.5 hw).**

| s | dip dN/N | dip dE/E | primary dN/N | primary dE/E | b5 dN/N | b5 dE/E |
|---|---|---|---|---|---|---|
| 0.05 | -0.0760 | -0.0665 | -0.0751 | -0.0655 | -0.0768 | -0.0744 |
| 0.1 | -0.1482 | -0.1278 | -0.1465 | -0.1260 | -0.1498 | -0.1446 |
| 0.2 | -0.2813 | -0.2387 | -0.2782 | -0.2358 | -0.2844 | -0.2736 |
| 0.3 | -0.3995 | -0.3379 | -0.3950 | -0.3340 | -0.4038 | -0.3883 |
| 0.5 | -0.5906 | -0.5052 | -0.5841 | -0.4990 | -0.5968 | -0.5758 |

Reference values at t_s: N_inst(ball rN) 2977.7 / 2845.6 / 6380.6 and N per period (ball rN) 3028.9 / 2897.1 / 6403.4; E(r < rN) 2952.4 / 2831.8 / 6355.2 (dip / primary / b5).

- (a) is close to Delta N / N = (1 + eps)^2 - 1 at all three starts (the ball holds almost all of N). At phase 90 the field is nearly zero, so the velocity kick's Delta E / E equals its Delta N / N to 4 digits; at phase 0 the field kick's Delta E is smaller than Delta N for eps < 0.2 at beta = 4/3 (the quartic term is negative) and larger above.
- (c) at t = 0: the packet sits outside the ball, so Delta N / N is below 1e-7 by construction (5e-8 slow, 8e-14 k3, measured); its size is efrac = E_p / E(r < rN). What N the lump loses or gains is read after the packet has passed (section 5).
- For orientation, not a prediction: with the Session 2 N/N_min (1.83, 1.73, 1.97 for dip, primary, b5), a Delta N / N of -0.45, -0.42 and -0.49 would put N at N_min (profile N; the period estimator reads 1.3%, 1.1% and 0.3% higher at the three starts).

**How the t = 0 numbers relate to what the lump keeps.** The kick's N is the instantaneous N; the lump then sheds part of it. At the dip, field kicks: Delta N / N at t = 0 is 1.986 eps; averaged over t_d + 50 to 250 against the control it is 1.908 eps (linearity.json); the first whole period after the kick agrees with t = 0 to 1e-3 for the dip field kick +0.1 only (0.2085 against 0.2087). Elsewhere it differs by 1% to 30% (first-period Delta N / N over the t = 0 value, minus 1: velocity +0.1: +0.31 dip, +0.29 primary, +0.04 b5; field +0.1: -0.01 primary, +0.13 b5; removal 0.2: +0.06 dip, +0.05 primary, -0.01 b5; `demos_summary.json`), so the t = 0 instantaneous N is a guide to size, not the N the lump carries in its first period. The removal amplitude s is the fraction by which phi and phi_t are scaled in the core, not a fraction of N: s = 0.2 removes 28% of N (table above). The omega shift is -0.117 eps (dip), so a 1% field kick moves omega by -0.00117 (Session 2's estimate was about -0.0013).

## 5. Demo runs (one per disturbance type per start, T = 2000)

`outputs/session3/wpL/demos_summary.json` (each against its control, with the free-packet passage time for packets and the linear twin at T = 500). Field and velocity eps = +0.1, removal s = 0.2, packets efrac = 0.05.

Windows are times after t_d; packets are read from the free packet's passage time t_pass (the time after launch at which E(r < rN + 5) of the packet alone falls below 1e-4 of E_p: 817 slow and 83 k3 at beta = 4/3, 966 and 110 at beta = 5). Gamma: loss rate of E(r < 3 hw) in the last of the three windows t_s + 250, + 750, + 1250 (each 500 long), that is t_s + 1250 to 1750. These rates are still falling, so they are transient values, not settled loss rates (dip field +0.1: 4.8e-5, 3.5e-5, 2.7e-5 in the three windows).

| run | t0 dN/N | dN/N vs control (50-250 / 250-750) | d omega (50-250) | min frac in 2000 (control) | broken | Gamma, t_s + 1250 to 1750, transient (control) | omega/m after | E + abs. after |
|---|---|---|---|---|---|---|---|---|
| dip field +0.1 | +0.2085 | +0.1926 / +0.1686 | -0.0101 | 1.000 (1.000) | no | 2.7e-5 (6e-9) | 0.895 | 3.2e-8 |
| dip velocity +0.1 | +0.2086 | +0.2052 / +0.1781 | -0.0106 | 1.105 (0.999) | no | 2.9e-5 (6e-9) | 0.876 | 3.4e-8 |
| dip removal 0.2 | -0.2813 | -0.2740 / -0.2916 | +0.0225 | 0.659 (1.000) | no | 6.2e-5 (6e-9) | 0.959 | 1.5e-7 |
| dip slow packet 0.05 | 0 | -0.0016 (t_pass to +200) | +0.0001 | 0.997 (1.000) | no | 1.6e-8 (5e-9) | 0.907 | 1.7e-8 |
| dip k3 packet 0.05 | 0 | -0.0031 | +0.0002 | 0.997 (1.000) | no | 3.6e-8 (6e-9) | 0.907 | 3.8e-8 |
| primary field +0.1 | +0.2082 | +0.1965 / +0.1840 | -0.0106 | 1.000 (0.992) | no | 1.8e-5 (4.2e-6) | 0.898 | 2.7e-8 |
| primary velocity +0.1 | +0.2084 | +0.2114 / +0.1960 | -0.0113 | 1.137 (0.991) | no | 2.1e-5 (4.2e-6) | 0.880 | 2.9e-8 |
| primary removal 0.2 | -0.2782 | -0.2697 / -0.2863 | +0.0230 | 0.667 (0.992) | no | 5.5e-5 (4.2e-6) | 0.960 | 1.6e-7 |
| primary slow packet 0.05 | 0 | -0.0010 | +0.0001 | 0.991 (0.992) | no | 4.4e-6 (4.2e-6) | 0.910 | 2.0e-8 |
| primary k3 packet 0.05 | 0 | -0.0070 | +0.0004 | 0.983 (0.992) | no | 5.7e-6 (4.2e-6) | 0.910 | 4.0e-8 |
| b5 field +0.1 | +0.2086 | +0.2055 / +0.2054 | -0.0025 | 1.000 (1.000) | no | 8e-9 (8e-10) | 0.962 | 7.8e-9 |
| b5 velocity +0.1 | +0.2086 | +0.2099 / +0.2098 | -0.0026 | 1.206 (1.000) | no | 8e-9 (8e-10) | 0.972 | 7.9e-9 |
| b5 removal 0.2 | -0.2844 | -0.2846 / -0.2854 | +0.0054 | 0.718 (1.000) | no | -2e-9 (8e-10) | 0.984 | 8.9e-9 |
| b5 slow packet 0.05 | 0 | +0.0013 | -0.00002 | 0.9995 (1.000) | no | 5e-8 (9e-10) | 0.976 | 7.7e-9 |
| b5 k3 packet 0.05 | 0 | -0.0057 | +0.0001 | 0.994 (1.000) | no | -1e-8 (8e-10) | 0.976 | 7.8e-8 |

- Regime: omega/m 0.876 to 0.984 after the disturbance; the nonlinear share (absolute, just after the disturbance) is 0.41 to 0.42 after a field kick at beta = 4/3, 0.18 at beta = 5, 0.08 to 0.23 after a removal, about 0 just after a velocity kick (phase 90, the field is near zero there), and 0.35 / 0.14 for the packets (lump unchanged at launch). Controls: 0.359, 0.350, 0.146.
- Free packets: the energy-centroid speed matches P/E to 4.1e-5 to 7.1e-5; the linear focus gives max |phi(0, t)| = 0.52 to 0.54 A (slow) and 5.0 to 5.5 A (k3).
- Every run has its linear twin (T = 500, `runs/*_lin.json`).

## 6. L2: what was built

`swirl/engine/boost.py`, init entry `{"wpl": {source, window, M, lumps: [{center, phase, v, quad_eps, random}]}}`; guards are the snapshot guards for the same source, window and centres (`check_wpl`).

- **Harmonic decomposition** of the settled source over one period: M = 32 phases (exact landing by a partial RK4 step, as `snapshot.phase_states`), window (26, 30) applied, DFT over the phase. Harmonics 0 to 16; odd ones carry the lump (|a_n|/Phi0: 0.49, 0.033, 0.0066, 0.0011, 2.0e-4, 3.7e-5, 8e-6, 3e-6 for n = 1 to 15), even ones are at 2e-5 to 2e-6 (the slow drift over one period). The first build used M = 16 and aliased n = 9 onto 7: 6.7e-4 Phi0 in phi and 7.8e-3 omega Phi0 in phi_t between samples. M = 32 brings that to 1.9e-5 and 1.6e-5.
- **Boost** along x by evaluating each harmonic at x = gamma xi (trigonometric interpolation matrix on the cube) with the phase theta_c - omega gamma v xi; phi_t' = gamma (phi_t - v phi_x).
- **(d) quadrupolar stretch** phi(S x), S = diag(sqrt(1 + eps), sqrt(1 + eps), 1/(1 + eps)) (the initial.py convention), by separable spectral interpolation of the settled snapshot. det S = 1, so Delta N = 0 and only the gradient energy changes: Delta E = sum_a (S_a^2 - 1) G_a = 3 eps^2 G + O(eps^3). This disturbance cannot be stated as a change in N; it is a pure shape change of size Delta E / E = 0.20 eps^2.
- **(d) random multiplier** (1 + a xi) on phi and phi_t, xi a seeded smooth Gaussian field (correlation length ell = 3, about hw), with its N-weighted mean removed and its N-weighted rms set to 1. Then Delta N / N = a^2 exactly: the uniform part (an amplitude kick, which gave Delta N / N = 3.3% at a = 0.01 in the first build) is removed, and a is the rms relative modulation. It gives the lump a small momentum (|P| = 0.02 to 0.11 at a = 0.02 to 0.1, v of order 4e-5 at most).
- **(e) two-lump data** (`two_lump_entry`): lump A at x = -d/2 moving +v, lump B at +d/2 moving -v, phases 0 and phi at their centres at t' = 0. The fields add; with window (26, 30) the cubes overlap for d < 60.
- **`track_run` and `gate_analysis`**: a 3D run of one lump with its energy centroid tracked in a ball (R = 10, taper 2), the ball energy, momentum and second moments, phi at the centre every step; the gate numbers below.

**t = 0 against quadrature** (`outputs/session3/wpL/l2_t0_testbig.json`, `l2_t0_prod.json`; the beta = 4/3 primary source of WP-E, dx = 0.8 and 0.55; E0 is the rest energy averaged over the 32 phases, 2838.32 and 2838.38; it varies by 2e-5 over the phase):

| quantity | target | dx = 0.8 | dx = 0.55 |
|---|---|---|---|
| boosted E / (gamma E0) - 1, v = 0.05, 0.1, 0.2, theta = 0 and 90 | 1e-4 | 5.0e-6 to 5.2e-5 | 4.6e-6 to 4.0e-5 |
| boosted P_x / (gamma v E0) - 1, same | 3e-4 | 1.9e-5 to 1.4e-4 | 7e-6 to 1.9e-4 |
| boosted P_y, P_z | 0 | 1e-12 | 1e-12 |
| stretch Delta E against the closed form, eps = +-0.05, +-0.1, +-0.2 | 1e-3 | 2.5e-5 to 7.3e-5 | 4e-10 to 6.7e-6 |
| stretch Delta N / N | 0 | 7e-8 or less | 3e-8 or less |
| random multiplier Delta N / N against a^2 | 1e-8 | 1e-12 | 1e-12 |
| placement with a sub-grid shift: E on the grid / E on the cube - 1 | 1e-6 | -7.6e-8 | — |
| two lumps: E_A + E_B against 2 gamma E0 | 1e-4 | 1e-5 to 5.2e-5 | 7e-6 to 4.0e-5 |

**(d) t = 0 values at the chosen amplitudes (primary source, dx = 0.55).** Stretch: Delta N / N = 0; Delta E / E = 4.8e-4, 1.80e-3, 6.42e-3 at eps = 0.05, 0.1, 0.2 and 5.5e-4, 2.35e-3, 1.11e-2 at -0.05, -0.1, -0.2. Random multiplier (ell = 3): Delta N / N = a^2 = 4e-4, 2.5e-3, 1e-2 at a = 0.02, 0.05, 0.1; Delta E / E = 4.0e-4 / -1.1e-4 (a = 0.02, seeds 1 / 2), 2.8e-3 / 1.6e-3 (0.05), 1.16e-2 / 9.0e-3 (0.1). The same seed gives a different field on another grid (the realisation depends on the cube size); compare seeds only on one grid.

**(e) two-lump cross term at t = 0** (E_two - E_A - E_B over E_A + E_B; this is the instantaneous energy of the overlap, not the period-averaged interaction): d = 20 (6 hw): +1.4e-3 (in phase), +1.6e-5 (90), -1.4e-3 (180) at v = 0.05, and -1.5e-4, +1.0e-5, +1.4e-4 at v = 0.2 (the boosted phase varies across each lump, so "in phase at the centres" is not in phase in the overlap); d = 30: below 7.5e-5; d = 40: below 1e-6.

## 7. The gate (boosted single lump in 3D)

Set-up (`outputs/session3/wpL/l2_gate.py`, results `gate.json`, tracks `gate/track_*.json`): WP-E test source src_testbig (beta = 4/3, profile omega 0.909149, settled omega 0.909540, dx = 0.8, dt = 0.08), its grid (N = 112, L = 89.6, default sponge), window (26, 30), M = 32. v = 0 at x = 0 to T = 240; v = 0.05 from x = -6 to T = 240; v = 0.2 from x = -12 to T = 120. Measured from t = 20, over the span both runs cover.

**Speeds proposed for the collisions: 0.05 and 0.2.** In Session 2 the lumps that came out of mergers left at 0.10 to 0.25 (E1, E2, E3, E7). v = 0.05 per lump (closing 0.1) is below what the mutual attraction gives a pair near contact, so the tail force and the relative phase act during the approach; v = 0.2 per lump (closing 0.4) is above it, so the collision is kinetic. Kinetic energy per lump is (gamma - 1) E0 = 3.6 and 58.6. Xue et al. 2025 (PRIOR_ART) used closing speeds of 0.01 to 0.04, slower than both.

| quantity | target | v = 0.05 | v = 0.2 | verdict |
|---|---|---|---|---|
| speed (slope of the centroid, t = 20 to T) | 1% | 0.0500015 (3.0e-5) | 0.2000080 (4.0e-5) | pass |
| E(t = 0) / (gamma E0) - 1 (E0: rest mean over the phases) | 1e-4 | 5.2e-5 | 2.1e-5 | pass |
| E(t = 0) / (gamma E_rest run(t = 0)) - 1 | 1e-4 | 4.2e-5 | 1.1e-5 | pass |
| omega_lab / omega_rest against 1/gamma | 1e-4 | 0.998746 against 0.998749 (-3.2e-6) | 0.979804 against 0.979796 (8.1e-6) | pass |
| shape: ball-energy drift minus the rest run's, same span | 1e-3 | -2.5e-5 | -1.6e-5 | pass |
| shape: drift of I_xx / I_yy over the run | — | -4.6e-6 | 2.2e-5 | held |
| contraction: (I_xx / I_yy) / rest | (1/gamma^2 = 0.99750, 0.96000, leading order only) | 0.99786 | 0.96587 | note |
| E + absorbed over the run | 1e-5 (Session 2 level, B1: 1.4e-5) | 2.2e-5 (rest run also 2.2e-5) | 1.4e-5 (rest over the same span 1.1e-5) | pass with note (2.2x); it is the same at v = 0, so it belongs to this grid (dx = 0.8, dt = 0.08) and the cropped start, not to the boost |

**The gate passes at both speeds.** The contraction of the energy second moment is not exactly 1/gamma^2: the lab energy density is gamma^2 (T00 + 2 v T0x + v^2 Txx) and the v^2 Txx term changes the moment at the same order; it is reported, not tested. The gate was run on the primary (omega 0.9091) test source at dx = 0.8, the only settled 3D source available; collision runs on another start or at dx = 0.55 should repeat it on their own source (about 4 minutes).

## 8. Proposed scan grid and cost

Spec lists: `outputs/session3/wpL/scan_specs_T2000.json` (219 runs) and `scan_specs_T30000.json` (84 runs); run with `l1_build.py scan --specs <file> --workers 2` (resumable).

| part | per start | values |
|---|---|---|
| (a) field, (a) velocity, T = 2000 | 21 each | eps = -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, -0.02, -0.01, -0.005, +0.005, +0.01, +0.02, +0.05, +0.1, +0.2, +0.3, +0.5, +0.75, +1.0 |
| (b) removal, T = 2000 | 13 | s = 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0 |
| (c) packets, T = 2000 | 8 x 2 | efrac = 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2; slow (k0 = 0.3, sigma = 20) and k3 (sigma = 5) |
| controls, T = 2000 | 2 | scale 1 and the slow-packet scale |
| degradation, T = 30,000 | 8 + 8 + 4 + 6 + 2 controls | eps = +-0.01, +-0.02, +-0.05, +-0.1 (field and velocity); s = 0.02, 0.05, 0.1, 0.2; efrac = 0.05, 0.2, 0.5 (both packets) |

Then, per start and disturbance, bisection between the last unbroken and the first broken amplitude (about 4 runs each, at T = 2000), and 3 doubled-box runs of selected degradation points at T = 30,000.

**Measured cost per run** (single thread, wall, with two workers and another numerics worker on the Mac): beta = 4/3 at scale 1: 4.0e-3 s per time unit (8 s for T = 2000, 120 s for 30,000); at scale 2.25 (slow packet): 8.8e-3; beta = 5 at scale 1: 7.6e-3 (15 s, 228 s); at scale 1.25: 9.5e-3.

| | runs | core-hours | Mac, 2 workers |
|---|---|---|---|
| T = 2000 grid | 219 | 0.69 | 0.35 h |
| T = 30,000 grid | 84 | 4.0 | 2.0 h |
| twins (T = 500) of the T = 2000 grid | 219 | 0.17 | 0.09 h |
| bisection (about 60 runs, T = 2000) | 60 | 0.2 | 0.1 h |
| doubled-box checks (3, T = 30,000) | 3 | 0.25 | 0.13 h |
| total | | about 5.3 | about 2.7 h |

No rented compute is needed. The 30,000 runs checkpoint every 1000 time units and resume; the batch is over 30 minutes, so it should run detached under caffeinate with a log.

## 9. Deviations, open issues

**Deviations from the brief.**
1. The velocity kick is applied at phase 90, k90 steps (1.6 to 1.7 time units) after t_s, not at t_s: at t_s (a maximum of phi(0)) phi_t is nearly zero and the kick would do almost nothing. The control is the same trajectory, so the comparison is unaffected. Any kick can be put at either phase (`phase`).
2. The t = 0 Delta N is the instantaneous N_inst at the disturbance phase. The period-averaged N needs evolution; the runs give it against the control (first whole period after the kick, and windows). They agree to 1e-3 in the first period only for the dip field kick +0.1; for the other demos the first period is 1% to 30% off (section 4), and afterwards the lump sheds part of the change.
3. The radial-against-3D N check is read from t = 300 on (3.4e-5); at t = 100 the two whole boxes hold different start-up radiation (1.2e-3, explained in section 3).
4. The random multiplier is centred and normalised with the N density (Delta N / N = a^2). A plain unit-rms multiplier made a 3.3% change in N at a = 0.01, so it was mostly an amplitude kick.
5. The gate ran on the existing WP-E test source (the primary, omega 0.9091, dx = 0.8), not on the dip start or at dx = 0.55. No settled 3D source exists for the dip or beta = 5 starts (section 6 has the (d) values for the primary only).
6. The slow packet needs a box of scale 2.25 at beta = 4/3 (R_box = 450) and 1.25 at beta = 5, so its runs have their own settled state and control.
7. Harmonics: M = 32 instead of a first M = 16 (aliasing, section 6).

**Open issues.**
1. **3D sources for L3.** (d) and (e) for the dip start and for beta = 5 need settled snapshot sources (`snapshot.make_source`). The dip start can use the production grid of the primary (N = 160, L = 88, dx = 0.55; about 45 min under load, resumable). beta = 5 needs a larger window and box: hw = 6.6 and the profile reaches 1e-6 Phi0 near r = 60, so about N = 224, L = 123 at dx = 0.55 (the A10 grid was N = 216, L = 118.8), with the window crop loss measured anew. The gate should be repeated on whichever source the collisions use.
2. **Checkpoints are keyed on parameters, not on code.** A result regenerated after a code change resumes from the old final checkpoint and returns the old state (this happened once here; the demos were rerun after deleting their checkpoints). When the code changes, delete `checkpoints/session3/wpL/runs/`.
3. **E + absorbed in the 3D gate** is 2.2e-5 (2.2x the Session 2 level), the same at v = 0.
4. **Bisection near thresholds** is not automated: the runner adds specs between the last unbroken and the first broken amplitude.
5. **N on the exact standing mode: 2.4e-8 against 1e-8** (within 3x, a pass with a note). Linear crossing times gave 4.9e-8; cubic refinement brought it to 2.4e-8 (attempt 1). An Euler-Maclaurin correction of the trapezoid C at the crossing made it worse, 1.2e-7 (attempt 2), and was dropped. The remaining cause is not identified. It is 20 times below the lump's own per-period N jitter (5e-7), so it does not limit any measurement here.
6. **Two bugs the tests found and that are fixed:** the absorbed energy and the C integrals were summed per chunk, so a run split at a kick or a checkpoint differed from an unsplit run in the last bits (the epsilon = 0 velocity kick and the resume test). Both are now accumulated step by step.
7. The settled primary start is not stationary: it loses about 4e-6 of its energy per unit time from the start (Session 1: it falls below 0.90 at 10,630), so its "degraded" measure is always relative to a decaying control.

**Seen in the demos (single runs, not results of the scan).**
- No demo broke outright. Minimum frac over 2000: field +0.1 1.000, velocity +0.1 1.10 to 1.21 (energy added), removal 0.2: 0.659 (dip), 0.667 (primary), 0.718 (b5); the removal's own energy cut is 0.24 to 0.27, so most of that drop is the cut itself.
- Loss rate over t_s + 1250 to 1750 (W = 500; still falling, transient) against the control: dip, field +0.1: 2.7e-5 against 6e-9 (control, within error); dip, removal 0.2: 6.2e-5; primary, field +0.1: 1.8e-5 against 4.2e-6; b5, all kicks: 1e-8 or below.
- Packets of 5% of E(r < rN) change N after they pass by: slow, -0.15% (dip), -0.10% (primary), +0.13% (b5); k3, -0.32% (dip), -0.70% (primary), -0.57% (b5).
- Linear twins disperse: frac at t_d + 100 is 0.017 to 0.025 at beta = 4/3 and 0.38 to 0.58 at beta = 5, 0.04 to 0.05 at +300.
