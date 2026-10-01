# WP-C build notes: pair diagnostics, phase control, pair-run design

Session 2, WP-C build (numerics worker). The derivation is in `docs/WPC_DERIVATION.md`. Code: `swirl/engine/lumps.py` (new) and `swirl/engine/run.py` (an optional `lumps` block). Tests: `tests/engine/test_lumps.py`. Outputs and scripts: `outputs/session2/wpC/`. The pair batch was not run.

## 1. Status in one paragraph

- **Built and checked.** Diagnostics, initial data and guards are built and checked against analytic or independent values (section 2). The derivation confirms U = -2 pi A^2 cos(dtheta) e^{-kappa d}/d exactly at leading order.
- **The gate does not pass as written.** In the radial single-lump phase control, theta = 45, 135 and 180 pass. **theta = 90 fails the pass criterion in `SESSION_02_PREDICTIONS.md`**, and prediction C0b misses at 90.
- **Cause.** Three cheap diagnostics trace the miss to the energy of the start slice of the single-frequency ansatz: the theta = 90 start has 8.4% less energy and settles on a different family member. It is not a phase-specific instability.
- **What it means for the batch.** Under plan section 4, the pair runs wait for the orchestrator's decision. The proposed design never uses a theta = 90 start, because it assigns the pair phases as -/+ dtheta/2. Every start it does use passes the criterion (section 3).

## 2. Check table

Measured values are recorded in `outputs/session2/wpC/checks/*.json`. "Worst" is the worst value over the whole record of that check.

| # | Check (test) | Independent value | Target | Measured (worst) | Verdict |
|---|---|---|---|---|---|
| 1 | Centroid of an off-lattice lump, ball started 0.54 off (`test_centroid_off_lattice`) | the lump centre (symmetry) | 1e-4 | 4.9e-6 after 4 tracking samples (8.8e-3, 1.4e-4, 6.5e-6, 4.9e-6) | pass |
| 2a | Moving lump, E(0) (`test_moving_lump_momentum`, v = 0.02) | 2D Gauss-Legendre quadrature | 1e-6 | 1.9e-8 | pass |
| 2b | Same, P(0) | quadrature | 1e-5 | 3.9e-9 | pass |
| 2c | Same, P conservation over T = 30 | exact | 1e-4 | 5.6e-6 | pass |
| 2d | Same, centroid velocity against P/E (dX_E/dt = P/E exactly) | identity | 2e-3 | -1.6e-3 | pass, close to target |
| 2e | Same, L_z about a parallel axis 3.0 away = -b P_x | exact | 1e-5 | 4.0e-10 | pass |
| 2f | Same, P/(E v) - 1 of the boosted single-frequency ansatz | not 0 for a non-periodic ansatz; small-v formula v(G/3 + omega^2 Q/2) | reported | +0.054 (the small-v formula matches the quadrature to 7.4e-5) | reported, see note A |
| 3 | Angular momentum of a spinning l = 1 Gaussian, three axes (`test_angular_momentum_spinning`) | omega a^2 pi^{3/2} R^5/2 | 1e-8 | 2.2e-16; L about the perpendicular axis 1.6e-15 | pass |
| 4a | Relative-phase estimator on synthetic signals with harmonics and a 10% amplitude ramp | set value | 1e-3 rad | 1.08e-4 rad (first version: 1.33e-3, fixed by a Hann-weighted projection) | pass after fix 1 |
| 4b | Anti-phase pair in the PDE (`test_relative_phase_pde`) | pi exactly (phi(-x) = -phi(x)) | 1e-8 | 0.0 | pass |
| 4c | Pair at -45/+45, d = 20, against two single-lump runs at -45 and +45 | singles | 5e-3 rad | 3.6e-3 rad (includes the predicted interaction, ~3e-3) | pass; see note B |
| 5 | Lump count from the period-averaged density of the superposition (`test_lump_count`) | geometry | d = 2 gives 1; d = 16 gives 2 at +-8; anti-phase d = 6 gives 2 | 1; 2 at +-7.993; 2 (saddle 0.233). In phase, d = 6: 2 maxima 1.65 apart, saddle 0.989, 1 distinct. d = 8: 2 (saddle 0.497) | pass; see note C |
| 6a | Pair (d = 10, theta = 0/60) energy at t = 0 (`test_pair_t0_against_quadrature`) | 2D quadrature | 1e-7 | 2.4e-12 | pass |
| 6b | Same, overlap energy E_pair - E_1 - E_2 = 203.1 | quadrature | 1e-4 | 3.7e-10 | pass |
| 6c | Same, ball E, int phi_t^2, int phi^2 per lump (smooth ball, mid-plane split) | quadrature with the same weights | 1e-6 | 5.6e-7 (worst of 6) | pass, close to target |
| 7 | Half-space momentum flux at early times, d = 16 in phase, N = 96 (`test_halfspace_force_d16`) | F_asym = 0.7720; F_sup = 0.7633 | 10% | pair minus 2 x single control: 0.8385 (+8.6% vs F_asym, +9.9% vs F_sup); pair only: 0.7578 (-1.8%) | pass, close to target; see note D |
| 8 | Ball N for a uniform field, whole-period average (`test_ball_N_uniform_field`, m = 1.113, 0.93) | N m/E = 1 exactly | 1e-4 | 1.8e-6 | pass; see note E |
| 9 | Checkpoint resume with the new diagnostics, two chunkings (`test_resume_bit_identical_with_lumps`) | bit-identical | exact | identical fields and JSON | pass |
| 10 | Guards: tail in the sponge, profile at the face, phase count (`test_lump_guards`) | — | raise | raise | pass |
| 11 | Whole suite (`.venv/bin/python -m pytest`, this branch) | — | all pass | `71 passed, 1 xfailed in 329.46s` (the xfail is the strict F-001 record) | pass |
| — | Averaging cadence (scratch check) | N m/E = 1 | 1e-4 | sampling every 10 steps at dt = 0.04 (17 samples per period): 6.4e-6 | the configs sample every 10 |

Notes.

- **A.** P = E v holds for exactly periodic solutions. The boosted single-frequency ansatz is not one, so it moves at P/E = 1.054 v. The identity that the engine must satisfy is dX_E/dt = P/E (check 2d).
- **B.** The first version asserted |dtheta - 90 deg| < 0.02 rad and measured 0.045. The 0.02 was a guess. The offset is the start-phase dependent settling of the ansatz (section 3), so the test now compares against single-lump runs. Raw offset of the ±45 pair from 90 degrees: 0.045 rad.
- **C.** The first version used count_min_sep = 2. That silently merged the two maxima 1.65 apart at in-phase d = 6. min_sep is now 1.0 (it only removes duplicate maxima on neighbouring grid points), and `count_distinct` joins maxima whose saddle ratio is >= 0.9. The first version also asserted lump positions at +-d/2 for anti-phase d = 6. The maxima are at +-3.61, because the node on the mid-plane pushes the energy out. That is physics, so the assertion is now made only for the separated case.
- **D.** The single-lump control removes the momentum of the shed radiation that crosses the plane. The two lumps radiate coherently, and for the mirror-symmetric pair d_x phi = 0 on the plane, so the radiation share of the pair's flux is not 2 x single. It lies between 0 and 2 x single, which is 0 to 0.08 here, about 10% of F at d = 16 and early times. The production analysis therefore compares the sliding force with F(d(t), dtheta(t)) after the shedding burst (section 6).
- **E.** N and E in the ball both deviate from A^2 m W/2 by 1.66e-5. The ball follows a centroid that wanders in a uniform field, and at dx = 1.25 the grid sum of the ball weight depends on the ball's position. The ratio removes the volume.

Code checks that needed a fix (one attempt each, all fixed):

- the phase estimator (4a);
- the lump-count de-duplication (note C);
- the time stamp of period averages. They are now placed at the window centre `t_center`. Stamped at the window end, they lag by about W/2 = 10 time units, which made a d = 16 test pair look 25% slower than its momentum. Found in a smoke run before any test was written.

## 3. Item 5: single-lump phase control in the radial solver

Script `outputs/session2/wpC/phase_control_radial.py`, result `phase_control_radial.json`. The radial solver is Session 1's: dr = 0.05, dt = 0.0125, R_box = 200, sponge from 100, sigma_max = 1. Start: phi = Phi cos(theta), phi_t = -omega Phi sin(theta), T = 1000. frac = E(r < 3 hw)/E0 of the same run. omega is the median core frequency over [T - 50, T]. The rows at +-22.5, -45 and +-67.5 are diagnostic D1 (`phase_control_diagnostics.py`).

| theta | E0 | frac 300 | 500 | 1000 | omega 300 | 500 | 1000 | max dfrac at T = 300/500/1000 | max domega | min frac, t > 50 | dead | criterion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2843.8 | 0.99560 | 0.99521 | 0.99415 | 0.90952 | 0.90956 | 0.90967 | — | — | 0.99328 | no | reference |
| 22.5 | 2861.2 | 0.99368 | 0.99333 | 0.99274 | 0.90923 | 0.90926 | 0.90935 | 1.9e-3 | 3.2e-4 | 0.99190 | no | pass |
| -22.5 | 2861.2 | 0.99592 | 0.99576 | 0.99422 | 0.90911 | 0.90914 | 0.90922 | 5.5e-4 | 4.5e-4 | 0.99376 | no | pass |
| 45 | 2878.8 | 0.99467 | 0.99440 | 0.99437 | 0.90875 | 0.90877 | 0.90883 | 9.3e-4 | 8.4e-4 | 0.99351 | no | pass |
| -45 | 2878.8 | 0.99665 | 0.99622 | 0.99471 | 0.90867 | 0.90869 | 0.90875 | 1.1e-3 | 9.2e-4 | 0.99471 | no | pass |
| 67.5 | 2742.1 | 0.99435 | 0.99216 | 0.98571 | 0.91215 | 0.91230 | 0.91271 | 8.4e-3 | 3.0e-3 | 0.98554 | no | pass (8.4e-3 of 1e-2 at T = 1000) |
| -67.5 | 2742.1 | 0.99393 | 0.99161 | 0.98565 | 0.91215 | 0.91229 | 0.91270 | 8.5e-3 | 3.0e-3 | 0.98565 | no | pass (8.5e-3 of 1e-2) |
| **90** | 2605.2 | 0.98896 | 0.98435 | 0.96957 | 0.91611 | 0.91650 | 0.91760 | **6.6e-3 / 1.09e-2 / 2.46e-2** | **6.6e-3 to 7.9e-3** | 0.96904 | no | **fail** |
| 135 | 2878.8 | 0.99665 | 0.99622 | 0.99471 | 0.90867 | 0.90869 | 0.90875 | 1.1e-3 | 9.2e-4 | 0.99471 | no | pass (equals -45 exactly: sign flip) |
| 180 | 2843.8 | 0.99560 | 0.99521 | 0.99415 | 0.90952 | 0.90956 | 0.90967 | 2.9e-15 | 6.8e-7 | 0.99328 | no | pass |

**Against the predictions on record.**

- **C0a (180 equals 0 to rounding): hit.** 2.9e-15.
- **C0b (45, 90 and 135 pass, with differences of order 1e-3 or less).**
  - 45: hit, 9.3e-4.
  - 135: hit, 1.1e-3.
  - **90: miss.** dfrac is 2.5e-2 at T = 1000, 25 times the predicted size. It is 1.09 times the criterion's 0.01 at T = 500 and 2.46 times at T = 1000. omega is off by 0.0066 to 0.0079 against 0.005, which is 1.3 to 1.6 times. No death.
- **The criterion in `SESSION_02_PREDICTIONS.md` fails at theta = 90.** CLAUDE.md's general rule would call anything within 3 times the tolerance a pass with a note. The predictions file fixes this criterion and says "a larger difference is a finding and the pair runs wait", so I report it as a fail.

**Investigation of the miss.**

- **Expected:** all starts settle onto the same lump, within 1e-3.
- **Measured:** the theta = 90 lump ends at omega 0.9176 with frac 0.970, against 0.9097 and 0.994.
- **Candidate causes, ranked.**
  1. **The slice energy of the ansatz depends on theta.** E0 runs from 2878.8 at 45 to 2605.2 = omega N at 90 (D3, `phase_control_diagnostics.json`). The theta = 90 slice is pure kinetic and lacks the harmonic content the true breather has at phi = 0. The lump settles onto the family member whose energy it has.
  2. A phase-specific instability of the theta = 90 start.
  3. A solver artefact of phi = 0 data. This is unlikely: the solver is linear in its treatment of phi_t, and theta = 180 reproduces theta = 0 to 3e-15.
- **Diagnostics run.**
  - **D1:** all the phases a symmetric pair design uses pass (table).
  - **D2:** a theta = 0 start from the family profile at the theta = 90 run's settled omega (0.9157; N = 2595, 9.5% below the primary) tracks the theta = 90 run. omega differs by 3.1e-4, 3.3e-4 and 3.9e-4 at T = 300, 500 and 1000. The loss rate over t = 200 to 1000 is 3.0e-5 against 2.9e-5 (the theta = 0 primary: 3.1e-6). frac agrees within 1.9e-3.
  - **D3:** the slice-energy table above.
- **What it rules in:** cause 1. The theta = 90 start is a lighter oscillon (9.5% less N) that radiates 10 times faster.
- **What it rules out:** a phase-specific instability. The 90-degree lump behaves exactly like the family member with its N.
- **What it means for pairs.** Lumps started at different theta are different lumps. Starting a dtheta = 90 pair as theta = (0, 90) would put a 9.5% N difference into the initial data, unrelated to the interaction. The design therefore uses theta_{1,2} = -/+ dtheta/2. The two lumps are then time-reversed images with equal slice energy. The lumps of different dtheta still differ: settled omega 0.9095 (0), 0.9092 (+-22.5), 0.9087 (+-45), 0.9122 (+-67.5). The post-processing predicts each run's force with the family A and kappa at the lumps' measured omega.
- **Expensive proposal, not run.** Start pairs from true quasi-breathers: take a settled 3D single lump at two times separated by dtheta/omega and superpose spectrally shifted copies. This removes the ansatz mismatch entirely.

## 4. Proposed pair-run design

| Item | Choice | Reason |
|---|---|---|
| Box, grid | L = 88, N = 160, dx = 0.55 | Session 1: dx = 0.547 reproduced dx = 0.365 to 1.6e-5 in frac. L/2 - 12 = 32 >= the 30 needed for Phi < 1e-6 Phi0 at the face, for lumps at +-12 (guard `radial_checks`) |
| dt | 0.04 | Same as Session 1 runs 1 and 5r at this dx |
| Sponge | default spherical (gamma 2, width 12, cubic), r_in = 32 | For lumps at +-12 the tail at the sponge is Phi(20) = 1.0e-4 Phi0 (guard `check_lumps`, tolerance 1e-3). At L = 70 it would be 7.7e-3 Phi0 and the guard rejects it |
| Lumps | at (-+d/2, 0, 0), theta_{1,2} = -/+ dtheta/2; dtheta = 180 uses (0, 180) | No theta = 90 start (section 3) |
| Separations | d = 16, 20, 24 (4.8, 6.0, 7.2 hw) | Little overlap: Phi(d/2)/Phi0 = 0.037, 0.013, 0.0046. The tail formula holds to 1.1%, 0.14% and 0.02%. F spans a factor 44, which gives kappa. Shifts at T are 0.03 to 30 times d for in phase |
| Run length | T = 500 for d = 20 and 24; T = 300 for d = 16 | At d = 16 the in-phase pair merges by ~126 and the 90-degree pair reaches dtheta ~ 224 by 300 |
| Diagnostics | R_ball = 10 (3 hw), taper 2, split width 1; every 10 steps (17 per period; N averaging accurate to 6e-6); count window every 25, 3 periods; planes x = 0, +-4; L about z and about x through the origin | |
| Centroid noise floor | from `noise_floor.py` on this grid: off-centre lump at x = 12, T = 80: drift 1.3e-6, residual rms 2.9e-7 (max 5.5e-7), fitted acceleration -2.2e-9 | The smallest nonzero predicted pair acceleration (d = 24, 45 degrees) is 4.4e-6, 2000 times above the fitted control acceleration. The smallest shift to resolve, d = 24 at 90 degrees and T = 300, is 3.6e-3, 10^4 times the noise rms |
| Off-centre control | `single_x12_t00.0`: one lump at the largest offset (x = +12), T = 500, theta = 0 | Its centroid drift is the sponge-asymmetry force. Pass if |drift| at T = 500 < 0.01, which is 3% of the smallest in-phase shift at that time (0.296 at d = 24, T = 300) |
| 3D phase controls | `single_x12_t{00.0, 22.5, 45.0, 67.5}`, T = 500 | Same starts as the pairs, in 3D; compare with the radial table |
| Linear twins | `twin_d{16,20,24}_p000`, in phase, g4 = g6 = 0, T = 300 | **I agree with one per separation.** The linear evolution of (theta_1, theta_2) data is the superposition of single-lump twins, so any phase pair is a linear combination of the in-phase and anti-phase twins. Their quadratic diagnostics differ only by cross terms, which are what the in-phase twin shows. T = 300, because Session 1's twins revive from the sponge after ~250 (F-005) |
| Resolution check | `check_res_d20_p000_N224`: d = 20, in phase, N = 224 (dx = 0.393), T = 300 | Same grid ratio as Session 1's run 4 (1.4x finer) |
| Box check | `check_box_d24_p000_L132`: d = 24, in phase, L = 132, N = 240 (same dx), T = 300; sponge r_in = 54 | Largest offset; it tests the sponge on the slowest-moving pair |

**Predicted numbers per pair run.** F(d0) and a = F/E are the leading-order law at the profile's A and kappa. The shift is per lump, from the point model with N exchange (`WPC_DERIVATION.md` sections 5 and 6). Positive means toward the partner. "Bounded" is at T_end only.

| run | d | dtheta | T | F(d0) | a = F/E | point model at T = 100: shift, d, dtheta | at T_end: shift, d, dtheta | predicted outcome |
|---|---|---|---|---|---|---|---|---|
| pair_d16_p000 | 16 | 0 | 300 | 0.772 | 2.76e-4 | 1.89, 12.23, 0 | merged by t = 126 | merged |
| pair_d16_p045 | 16 | 45 | 300 | 0.546 | 1.95e-4 | 1.10, 13.79, 64 | -8.32, 32.6, 501 | separated (attracts first) |
| pair_d16_p090 | 16 | 90 | 300 | 0 | 0 | -0.081, 16.16, 110 | -2.83, 21.7, 224 | separated |
| pair_d16_p135 | 16 | 135 | 300 | -0.546 | -1.95e-4 | -0.88, 17.75, 147 | -4.93, 25.9, 195 | separated |
| pair_d16_p180 | 16 | 180 | 300 | -0.772 | -2.76e-4 | -1.15, 18.30, 180 | -5.74, 27.5, 180 | separated |
| pair_d20_p000 | 20 | 0 | 500 | 0.1137 | 4.06e-5 | 0.21, 19.58, 0 | merged by t = 346 | merged |
| pair_d20_p045 | 20 | 45 | 500 | 0.0804 | 2.87e-5 | 0.146, 19.71, 47 | 1.48, 17.0, 214 | bounded at T (turning) |
| pair_d20_p090 | 20 | 90 | 500 | 0 | 0 | -0.0018, 20.00, 93 | -0.886, 21.8, 160 | separated |
| pair_d20_p135 | 20 | 135 | 500 | -0.0804 | -2.87e-5 | -0.141, 20.28, 137 | -2.64, 25.3, 170 | separated |
| pair_d20_p180 | 20 | 180 | 500 | -0.1137 | -4.06e-5 | -0.197, 20.39, 180 | -3.20, 26.4, 180 | separated |
| pair_d24_p000 | 24 | 0 | 500 | 0.01759 | 6.28e-6 | 0.0315, 23.94, 0 | 0.903, 22.2, 0 | bounded at T (approaching; merges ~904) |
| pair_d24_p045 | 24 | 45 | 500 | 0.01244 | 4.44e-6 | 0.0222, 23.96, 45 | 0.59, 22.8, 55 | bounded at T (approaching) |
| pair_d24_p090 | 24 | 90 | 500 | 0 | 0 | -4.5e-5, 24.00, 90 | -0.028, 24.06, 102 | bounded at T |
| pair_d24_p135 | 24 | 135 | 500 | -0.01244 | -4.44e-6 | -0.022, 24.04, 135 | -0.52, 25.05, 143 | separated |
| pair_d24_p180 | 24 | 180 | 500 | -0.01759 | -6.28e-6 | -0.031, 24.06, 180 | -0.71, 25.4, 180 | separated |

Caveats for the orchestrator's Block 2.

- **The start is not exactly at the nominal phase.** The -/+ starts settle with a phase offset: the ±45 pair at d = 20 reads 92.3 degrees (check 4c). A 2.3-degree offset at nominal 90 gives F = -0.040 F0 from the start. Compare with F(d(t), dtheta(t)) using the measured phase (the post-processing does).
- **The lump at each dtheta has its own omega.** The settled omega is 0.9087 to 0.9123. Corrected after review: A and kappa move together and F changes by -3.5%, -1.0% and +1.7% at d = 16, 20, 24 for omega = 0.9122 (the first version said up to 18%, counting only e^{-kappa d}; `WPC_DERIVATION.md` section 3). The analysis uses F at the measured omega.
- **The point model loses validity late in some runs.** It is linear in the N exchange and assumes constant A and E. It is outside its validity once |N_1 - N0| > ~10% N. That happens at d = 16 for 45 and 90 degrees late in the run.
- **Below d ≈ 12 nothing is predicted.** Merger dynamics there are outside the derivation.

## 5. Costs on this Mac

Measured with `bench_pair.py` (`bench_pair.json`). Conditions: 2 threads per run (NUMBA_NUM_THREADS = 2, FFT workers 2). The Session 2 WP-B 192^3 run (2 threads) was running throughout, and the load average was 3.9 to 5.7 from other workers.

| Grid | Diagnostics | steps/s | Note |
|---|---|---|---|
| N = 160, L = 88 | lumps every 5 steps | 5.9 | first measurement |
| N = 160, L = 88 | lumps every 10 steps (the configs) | 7.1 | 19% overhead over no lumps (8.4) |
| N = 160 | off-centre single, every 10, T = 80 (noise run) | 7.3 | |
| N = 224, L = 88 | lumps every 10 | 2.66 (3.03 without lumps) | 60-step bench, same load |
| N = 240, L = 132 | lumps every 10 | 2.30 (2.46 without lumps) | 60-step bench, same load |

| Runs | Count | Steps each | Hours each | Slot-hours |
|---|---|---|---|---|
| pairs d = 20, 24 (T = 500) | 10 | 12,500 | 0.49 | 4.9 |
| pairs d = 16 (T = 300) | 5 | 7,500 | 0.29 | 1.5 |
| single-lump controls (T = 500) | 4 | 12,500 | 0.49 | 2.0 |
| linear twins (T = 300) | 3 | 7,500 | 0.29 | 0.9 |
| resolution check N = 224 (T = 300) | 1 | 7,500 | 0.78 | 0.8 |
| box check N = 240, L = 132 (T = 300) | 1 | 7,500 | 0.90 | 0.9 |
| **Total** | 24 | | | **~11.0 slot-hours** |

- **Wall time.** With one free 2-thread slot (while WP-B runs) about 11 h; with two slots about 5.5 h.
- **Memory.** About 0.6 GB per N = 160 run, about 2 GB for N = 240.
- **Disk.** Checkpoints about 100 MB per N = 160 run (fields plus the lump-count accumulator), in `checkpoints/`, never committed. Result JSON about 3 to 4 MB per T = 500 run, about 80 MB for the batch.
- **Renting.** F-006 says AWS buys parallel slots, not speed. The ~11 slot-hours fit 3 c7i.4xlarge slots in about 4 h. That is the orchestrator's call.

## 6. Exact runner commands

Configs are already generated in `outputs/session2/wpC/configs/` (24 files; regenerate with `.venv/bin/python outputs/session2/wpC/make_configs.py`). From the repo root `/Users/aaronramos/Desktop/SWIRL`:

One run (resumable; re-running continues from its checkpoint; result written atomically with full parameters):

    outputs/session2/wpC/run_one.sh <name>

which executes

    NUMBA_NUM_THREADS=2 caffeinate -i .venv/bin/python -m swirl.engine.run outputs/session2/wpC/configs/<name>.json \
        --out outputs/session2/wpC/results/<name>.json --checkpoint-dir checkpoints/session2/wpC/<name> --verbose \
        >> outputs/session2/wpC/logs/<name>.log 2>&1

The batch: one detached runner per free 2-thread slot. Each list puts the gate runs first. Completed runs are skipped.

    nohup outputs/session2/wpC/run_batch.sh outputs/session2/wpC/batch_slotA.txt > outputs/session2/wpC/logs/slotA.log 2>&1 &
    nohup outputs/session2/wpC/run_batch.sh outputs/session2/wpC/batch_slotB.txt > outputs/session2/wpC/logs/slotB.log 2>&1 &

Post-processing, version 2 (after review; any time; it works on partial results):

    .venv/bin/python outputs/session2/wpC/gate_eval.py          # gate G1, G2 (cloud worker's script)
    .venv/bin/python outputs/session2/wpC/postprocess.py

It uses the stored series only and never puts the predicted force inside a fit: predictions are evaluated along the measured d(t), dtheta(t) and omega(t) and compared. It writes:

- `analysis/criteria.json`: one entry per row C1 to C13 of Block 2 with Block 2a: measured values, tolerance, hit or miss per item, or "not evaluable" with the reason; `all_misses` lists every miss; `evaluator_choices` lists every definition that Block 2/2a does not fix (for example the 0.25 F0 cut below which a force ratio is undefined, the prominence of an interior extremum for C10, the |L|/(E d0) threshold for C13).
- `analysis/summary.json` and one PNG per run.

Measures (Block 2a):

- (i) sliding half-space force, 40-unit windows centred at t >= 100 while two distinct lumps remain, against the window mean of F(d(t), dtheta(t)) at the measured omega; the slope of P(x<0) - P_total/2 is reported beside it, because the -/+ theta pairs carry a nonzero total momentum (the N current carries momentum omega x N flux; P_x = -18.95 at t = 0 for the d = 16, 90-degree pair on the test grid);
- (ii) d(t) against the point model integrated with the measured dtheta(t), and a second model with its own phase dynamics;
- (iii) at d = 24, the early slope with the single-lump contributions subtracted (lump 2 from the +12 single at its phase; lump 1 from the mirror image of the +12 single, which stands in for -theta except at 0 and 180);
- the early half-space slope over [10, 60], reported everywhere, a criterion only at d = 16;
- start values (N, E, d) from the first window centred at t >= 15;
- N flow per lump and antisymmetric, early and sliding, against 2 pi A^2 sin(measured dtheta) e^{-kappa d}/d, direction stated as leader-to-lagger or not;
- outcome (merged, separated, bounded) and the C10 bonded test as written; contact time (d <= 2 hw) and retained energy of completed mergers;
- C12 max |d(t) difference| against the partner run; C11 twins; C13 extremes of L and P; settled omega per lump.

`--test-record-end-as-T` treats a short record's end as the design T; for pipeline testing only (it is recorded in criteria.json).

## 7. Disclosure: test-sized pair runs made during the build

These were used to validate the pipeline. They are not the batch and not committed as results.

- **`test_halfspace_force_d16`** (N = 96, T = 30): check 7.
- **`test_relative_phase_pde`** (N = 96, d = 20, T = 24).
- **A scratch run of 4 configs at N = 96 (dx = 0.92), T = 70.** The configs were pair_d16 at 0, 90 and 180 and single_x12. Its numbers:
  - early force against F(d(t), dtheta(t)): ratio 1.004 (in phase) and 1.041 (anti-phase);
  - the ±45 pair at d = 16: early repulsion F = -0.065, consistent with its measured phase offset plus the pendulum drift;
  - single-lump drift at x = 12: -1.7e-4 over T = 70 on that coarse grid.

  The orchestrator may want to know these before writing Block 2.

## 8. Open issues, misses, deviations

**Misses (results, not fixed).**

- **Phase control at theta = 90 fails the criterion as written**, and C0b misses at 90 (section 3). Under plan section 4 the pair runs wait for the orchestrator's decision.
- **The brief's A, E and N belong to omega = 0.910, not the profile used (0.909149).** Using them would lower F by 3.5%.
- **The plan's point-model limit of 2.5 hw is too close.** At 2.5 hw the tail formula is 2.5 times too large. Valid to 10% needs about 3.6 hw (d ≈ 12), and to 1% about 4.8 hw (`WPC_DERIVATION.md` section 6). This matters for WP-D.

**Targets passed close to the limit.**

- Check 2d: 1.6e-3 of 2e-3.
- Check 6c: 5.6e-7 of 1e-6.
- Check 7: 8.6% of 10%.
- Phase control at ±67.5: 8.5e-3 of 1e-2 at T = 1000.

**Open.**

- **Early-time force is contaminated by coherent radiation.** The radiation momentum through the plane from the initial shedding is about 10% of F at d = 16 in the first 60 time units, and the incoherent single-lump control does not remove it (note D). The analysis compares the sliding force at the current separation and phase, and will show whether the contamination persists.
- **The ball centroid of an overlapping lump is biased toward its partner.** The overlap energy is real field energy. The bias is 0.028 at d = 16 (t = 0) and decays as e^{-kappa d}. It changes a measured shift by about 1% as d shrinks.
- **Not built.** The phase-averaged multipole shape diagnostic (still the Session 1 open issue).

**Found while testing postprocess v2 (test-sized data only: N = 96, dx = 0.92, T <= 150, scratch; not the batch).** These are properties of the criteria as written, likely to matter on the real runs. The criteria were not changed.

- **C2 at d = 16, dtheta = 0.** The only measure-(i) windows centred at t >= 100 run up to the contact (predicted near 126), where d falls to about 8. There the tail formula overpredicts by up to 2.5 times (`WPC_DERIVATION.md` section 6). On test data the judged worst ratio was 0.18. The analysis also reports, unjudged, the worst ratio over windows with d >= 12 (none existed for that run).
- **C2 at 45 and 135.** dtheta drifts toward 180, so F_pred passes through zero. Windows just above the 0.25 F0 cut amplify any absolute error: on test data d16_45 gave 1.48 while its residual stayed small.
- **C5b at d = 24.** Over the early window [15, 60] the +45 and -45 lumps settle at different rates. On test data dN/dt was 0.054 for the +45 lump and 0.0005 for the -45 lump, which is larger than the interaction rate of 0.038, and the antisymmetric rate came out in the wrong direction. The analysis subtracts each lump's own single-lump N drift where both singles exist. Only +theta singles are configured. Proposal: `single_x12_tm22.5`, `tm45.0`, `tm67.5` on the Mac, T = 150 (about 0.3 h each).
- **C11 "no centroid attraction".** The ball centroids of a dispersing linear field are not lump positions. On test data the in-phase twin's centroids closed by 3.8 (counted as attraction) and the anti-phase twin's separated by 18.
- **Total momentum of the -/+ theta pairs is not zero.** P_x = -18.95 at t = 0 for d = 16, 90 degrees. The overlap carries momentum: the period-averaged momentum density is omega times the N current, and the energy it moves from leader to lagger shifts the energy centroid at P/E. Measure (i) also reports the slope of P(x<0) - P_total/2.
- **L_z in the -/+ theta pairs is 4e-5 to 9e-5, not 1e-13.** Relative to E d0 that is 1e-9, which is inside the C13 evaluator threshold of 1e-8. Cause: the engine's minimum-image lever arm is -L/2 on the box-face plane, which has no mirror partner, so p_x there is not cancelled. Proposal for a later engine change: zero lever weight on that plane.
- **The sign slip "F = -dU/dd" also appears in Block 2 of `SESSION_02_PREDICTIONS.md`** (orchestrator's file, not edited).

**Deviations.**

- The box check is 1.5 times the box (L = 132 against 88), not the 2 times of standard 4 (also listed in Block 2a).
- Only in-phase linear twins were configured for the rented batch. One anti-phase twin, `twin_d16_p180` (d = 16, T = 300), was added after review and runs on the Mac. Linearity makes every phase pair a combination of the in-phase and anti-phase twins, so these two at d = 16 cover the phase dependence there; the other separations have the in-phase twin only.

- The radial phase option is a new script under `outputs/session2/wpC/` that drives `swirl.family.radial.Radial`, as the task allowed. `swirl/family/` was not edited.
- Test check records go to `outputs/session2/wpC/checks/`, not `outputs/engine/`, to stay inside the allowed paths.
- `run.py` adds the `lumps` key only when the config has it, so old configs and checkpoints keep identical params.

**Parking lot.**

- Quasi-breather pair initial data (section 3).
- A next-order (polarization) calculation of the force for 12 < d < 16.
- The phase-pendulum prediction is new to this project. Before anything is called new, scholar should check the Q-ball charge-transfer literature (Battye & Sutcliffe 2000, Axenides et al. 2000) for the same dynamics.

## 9. Independent recomputation of the pair-law verdict (reviewer, 2026-10-02; recorded by the orchestrator)

The reviewer read the 24 result JSONs with its own code (postprocess.py not imported). Its verdict: the measured pair law agrees with the derived one in sign, decay constant, cosine dependence, N flow and phase drift, **with stated exceptions**. It agrees with every hit and every miss of `analysis/criteria.json` as written, and adds the following.

**Numbers recomputed.**
- Sign (C1): 12 of 12.
- Early-acceleration ratio (quadratic fit of d(t) over [20, 90] at d = 16 and [20, 150] at d = 20, 24, against the derived force at the measured d, dtheta, omega): 0.972 to 1.069 over the 12 runs with a non-zero predicted force; residual at 90 degrees -0.004 to -0.008 F0.
- kappa: 0.4163 (scatter about 0.003; sub-selections 0.4156 to 0.4199), amplitude 20,744 against 2 pi A^2 = 20,200 (+2.7%). The script's 0.4172 agrees.
- 90-degree pairs: N flows leader to lagger at all three separations; rate over prediction 1.02 to 1.09 (d = 16), 1.04 to 1.12 (d = 20), 0.99 to 1.08 (d = 24) on windows from t = 60. Phase at the end 215.4, 159.6, 106.5 against the pendulum on the measured d(t): 196, 152.3, 103.5.
- Contact: 126.8 and 347.8 (d <= 2 hw); first window with one distinct lump at 135 and 360.
- Total momentum of the +-theta pairs is physical: P_x(0) = -omega I(d) sin(dtheta) with I the overlap integral of the true profile, ratio 1.0000 to the stored value in all nine runs (for example -18.9455 at d = 16, 90 degrees).
- Resolution check: genuinely different runs (N = 224 against 160; ball energy differs by 3e-3, max|phi| by 3.5e-2); d(t) agrees to 1.62e-6. Box check 4.8e-4, to T = 300 only.
- Cross-machine (Mac against AWS, five d = 16 pairs, identical code): d(t) to 2.2e-8 in the merging run and 7e-14 otherwise; per-lump N to 1.5e-4 in the merger burst and 4.6e-11 otherwise.

**The exceptions.**
1. **Close range.** Below d of about 12 the tail law overpredicts, as the superposition integral says. C2's misses for d16_0 and d20_0 are judged on two windows (d_min 7.2 and 8.4) and one window (d_min 6.8). Against the full superposition force the measured force is 1.00 to 1.04 for windows with d_min >= 10. Block 2b flagged d16_0 before the batch; d20_0 and C4 were not flagged.
2. **A real excess in the 45-degree pair at d = 16.** For t = 135 to 195, where the force is largest, the measured force is 1.3 to 1.5 times the law, and the force changes sign at dtheta = 85 degrees, not 90. The N asymmetry reaches 28% of N there (the derivation's validity limit is 15%) and d_min = 11.7. Unequal tails explain 5 to 8%; the cause is unidentified (non-stationary lumps is the leading guess, untested). The C6 outcome hit hides a 38% shortfall of the point model's shift for this run (-5.13 measured against -8.3).
3. **C4 at d = 16** is mostly the windows: the pooled slope 1.206 / intercept 0.152 comes from two near-contact in-phase windows and from the d16_45 run. The cosine law holds inside runs (the 90-degree run alone: slope 1.04, intercept -0.003 over cos from -0.39 to -1; d = 20 and 24 pass). With the centroid-based force the same fit gives 0.93 / -0.07.
4. **C5b at d = 24** is mechanical: the early window [15, 60] holds 1.7 units of N exchange against about 1.6 of settling difference between the +45 and -45 lumps. With the +45 single-lump run subtracted, lump 1 loses N at the predicted rate (-1.57 against -1.53 at t = 60; -17.1 against -16.0 at t = 450). The -45 single-lump run was not yet available to the reviewer.
5. **A +2 to 6% systematic excess** in force and N flow at d = 20 and 24.

**Other points.**
- The merged pair is one lump but not a settled one: its ball energy is still falling by about 0.07% per unit time at the end (0.858 at t = 159, 0.762 at t = 280 for d = 16), omega wobbles by 0.003 around 0.885. "Retained 0.76" is energy in the ball around the merged lump over its start value, read once while still falling; box energy over E0 is 0.84 to 0.85.
- C11: the ball-energy half hits (0.022 to 0.045 at T = 100 against 0.05); the "no centroid attraction" half cannot be measured by ball centroids of a dispersing field (the anti-phase twin closes by the same amount as the in-phase one). A clean no-interaction test is still missing.
- C13: L_z of 1e-5 to 5e-5 in the AWS +-theta runs is the lever-arm artefact; the Mac runs made after the fix give 1e-12. Three Mac runs (p045, p090, p135) used the fixed engine.
- Loose or marginal hits: C5b at d = 20 (27.7% against 30%); C7 retained energy (wide band, read while falling); C12 (tolerances 1e4 to 1e5 times the observed difference); C10 is definitional and the runs are short; the restated C5a tests the cosine law at the measured phase, not that the force vanishes at exactly 90.
