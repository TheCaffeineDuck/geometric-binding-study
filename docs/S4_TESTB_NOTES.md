# Session 4, Test B: build notes (numerics)

Date 2026-11-09. Plan: `SESSION_04_PLAN.md` section 3 (B1 to B5), section 4. Branch note: the
harness confined this worker to its own worktree (`.claude/worktrees/agent-a33f0c07ac8e4cac7`,
branch `worktree-agent-a33f0c07ac8e4cac7`), which started at the same commit as `s4-testB`
(7391c2d, no local changes there). The work is committed on that branch; `s4-testB` can
fast-forward to it.

Code: `swirl/session4/testB/` (`dispersion.py`, `run.py`, `analysis.py`). Tests:
`tests/session4/test_testB.py` (measured values in `outputs/session4/testB/test_values.json`).
Outputs: `outputs/session4/testB/`.

## 1. B1: the dispersion relation from the code

### 1.1 Linearisation and V''(0)

All three solvers evolve phi_tt = (spatial operator) phi - V'(phi) (+ damping on phi_t in the
absorber/filter). The implemented V' and its derivative at 0:

| Solver | V' in the code | V''(0) |
|---|---|---|
| `swirl/family/radial.py::_accel` | `p * (m2 - p2 * (g4/6 - g6/120 * p2))`, applied as `r * vp` to u = r phi | m^2 |
| `swirl/engine/evolver.py::_stage` | `p * (m2 + p2 * (a6 * p2 - a4))`, a4 = g4/6, a6 = g6/120 | m^2 |
| `legacy/engine/evolver.py::SexticEvolver.compute_rhs` | `m_sq*phi - (g4/6) phi^3 + (g6/120) phi^5` | m^2 |

So every linear mode obeys a'' = -Omega^2 a with Omega^2(k) = m^2 + K2(k), K2 the symbol of minus
the discrete Laplacian.

### 1.2 Semi-discrete symbols

- **Radial** (u = r phi, 4th-order central stencil (-u_{i+2} + 16u_{i+1} - 30u_i + 16u_{i-1} - u_{i-2})/(12 dr^2),
  odd reflections at r = 0 and r = R_box). On u_i = sin(k r_i) (an exact discrete eigenvector,
  because both reflections map sin onto itself for k = n pi / R_box):
  K2(k) = (30 - 32 cos th + 2 cos 2th)/(12 dr^2) = (1 - cos th)(7 - cos th)/(3 dr^2), th = k dr,
  = k^2 (1 - th^4/90 + ...). Maximum 16/(3 dr^2) at th = pi (0.540 of pi^2/dr^2).
  dK2/dk = sin th (8 - 2 cos th)/(3 dr); group velocity v_g = (dK2/dk)/(2 Omega) -> k/Omega.
- **Engine v2 `Evolver`** (real FFT, multiplication by -|K|^2, Nyquist included):
  K2 = |k|^2 exactly for |k_i| <= pi/dx. Continuum dispersion up to the grid cut-off.
- **Legacy `SexticEvolver`** (complex FFT): K2 = |k|^2 + sigma dx^4 |k|^4. The "Kreiss-Oliger"
  term `sigma*dx^4*ifftn(K4*phi_hat)` multiplies phi, not phi_t: it is a conservative stiffness
  (no damping), absent from the legacy energy (LESSONS 7). It vanishes as k^4 at k -> 0.

### 1.3 RK4 correction

Classical RK4 on zeta' = -i Omega zeta (zeta = Omega a + i a') multiplies zeta by R(-i th) per
step, th = Omega dt, R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24:

omega_num = arg R(i th)/dt = Omega (1 - th^4/120 + ...),  |R|^2 = 1 - th^6/72 + th^8/576 (exact).

### 1.4 Numbers at the dt and grids actually used

| Solver, grid | dt | omega_th = arg R(i m dt)/dt | omega_th - m | Omega_max (grid) | Omega_max dt |
|---|---|---|---|---|---|
| Radial, dr 0.05 (Sessions 1-3, Test B) | 0.0125 | 0.99999999980 | -2.03e-10 | 46.20 | 0.577 |
| Radial, dr 0.025 (Test B dr/2) | 0.00625 | 0.999999999987 | -1.27e-11 | 92.38 | 0.577 |
| Engine v2, dx 0.55 (WP-E / L3 production) | 0.04 | 0.9999999787 | -2.13e-8 | 9.94 (corner) | 0.398 |
| Engine v2, dx 0.55, default dt = dx/8 | 0.06875 | 0.99999981 | -1.86e-7 | 9.94 | 0.684 |
| Legacy SexticEvolver, N 64, L 50 (dx 0.781), sigma 0.01 | 0.05 | 0.99999995 | -5.20e-8 | 4.26 (axis Nyquist; 4.14 without sigma) | |

Threshold: omega_th = m in all three solvers, below m only by the RK4 phase error, at most
2.1e-8 (engine v2 production) and 2.0e-10 for the radial Test B runs. B-P1 target (1e-6): met.
The plan's stop condition (threshold not m) does not fire.

Group velocity near threshold (all three, k dx << 1): v_g = k/omega = sqrt(1 - m^2/omega^2),
about sqrt(2 (omega - m)): 0.0998 at omega = 1.005, 0.0245 at 1.0003. At the radiation lines
(radial, dr 0.05): 3 omega = 2.727, k3 = 2.5374, v_g = 0.9303 (the stencil changes K2 there by
-2.9e-6 relative); 5 omega = 4.546, v_g = 0.975.

High k: the radial stencil under-represents K2 (0.54 of k^2 at Nyquist), so grid-scale waves
are slower than continuum waves; RK4 damps them (|R|^2 = 1 - 4.9e-4 per step at the radial
Nyquist mode). Engine v2 is exact to Nyquist. SexticEvolver raises omega(k): the stiffness adds
sigma (k dx)^2 relative to k^2, 0.099 at the axis Nyquist and 0.296 at the corner (omega 4.26
against 4.14 on the axis at dx 0.781); at the 3 omega radiation wavenumber it lowers k by
0.029 (1.2%) at fixed omega.

### 1.5 Where damping could move the k -> 0 limit

- Radial sponge -sigma(r) u_t: sigma = 0 for r < R_s, so the interior relation is unchanged.
  Inside the sponge the local relation is omega^2 + i sigma omega = m^2 + k^2: for omega < m,
  k^2 = omega^2 - m^2 + i sigma omega has a real part of k, so a sub-threshold oscillation that
  reaches the sponge can lose energy there. It reaches it only through the evanescent tail,
  A e^(-kappa R_s)/R_s about 1e-37 at R_s = 200 for the primary: no effect. The reflecting wall
  quantises k = n pi / R_box; the lowest box mode is omega = sqrt(1 + (pi/400)^2) = 1 + 3.1e-5.
- Engine v2 absorber gamma(x) phi_t: position-space, zero in the interior; same remark.
  Filter nu (|k|/k_max)^p phi_t (p = 8): Re omega shifts by -nu(k)^2/(8 Omega), zero at k -> 0.
  Not used in production (only in `outputs/engine/kfilter_damping.json`, a validation run).
- Legacy: no absorber; the stiffness vanishes as k^4.
- Periodic boxes (both 3D engines) contain the exact k = 0 mode, whose measured frequency is
  the RK4 threshold value to machine precision (below).

### 1.6 Numerical confirmation

Each solver evolves single linear eigenmodes; the frequency is the phase rate of
zeta = Omega a + i a' (a = projection onto the mode). Measured vs derived (relative):

| Solver | Modes | Worst |omega_meas/omega_pred - 1| |
|---|---|---|
| Radial (R 40, dr 0.05, dt 0.0125, T 200, sponge off) | n = 1 (k dr 0.0039), 100, 400 (k dr = pi/2), 799 (k dr 3.14); n = 1 with g4, g6 on at eps 1e-6 | 2.0e-14 |
| Engine v2 (N 16, dx 0.55, dt 0.04) | k = 0, (1,0,0), (3,2,1), axis Nyquist, corner | 8.9e-16 |
| Legacy (N 16, dx 0.781, dt 0.05, sigma 0.01) | k = 0, (1,0,0), axis Nyquist, corner | 2.2e-16 |

The radial |R|^2 per step also matches (to 1e-12, e.g. 4.936e-4 at Nyquist). The nonlinear
radial run at small amplitude gives the same frequency as the linear one (V''(0) = m^2 from the
implemented potential). The legacy module is loaded by path with bytecode writing off; the test
checks that nothing is written under `legacy/`.

## 2. Existing far-field series

None usable. Radial runs of Sessions 1 to 3 stored phi(0, t) only (`run_radial.py`,
`wpa_run.py`, `wpl.py`). 3D runs (`swirl/engine/run.py`) store phi at the centre ("point"). The
only off-centre record is F-027's diagnostic: phi at one point per axis at r = 25 in the 3D dip
distortion runs (T = 2000, dx 0.55), under r = 30 and at one radius, so it cannot serve B4.

## 3. Probe recording

`run.read_probes` reads the solver state after each block of 16 RK4 steps (0.2 time units):
phi = u/r, phi_t = v/r, phi_r = (u_r - u/r)/r with the 4th-order central u_r; at r = 0 the
solver's own limit (4 f_1 - f_2)/3 and phi_r = 0. Nothing is written to u or v.

- Non-interference: final (u, v) after T = 40 is bit-identical between the driver with probes,
  one `Radial.step` call and Session 1's 40-step chunking (test).
- Accuracy on an analytic even field: errors 5.9e-8 (phi, at r = 0), 1.2e-7 (phi_t), 4.8e-10 (phi_r).
- Flux against the solver's energy: an outgoing linear packet (k 2.5); the time integral of
  4 pi r^2 (-phi_t phi_r) at r = 20 from the 0.2-spaced probes equals the drop of E(r < 20)
  (4th-order density, Simpson) to 2.7e-5.
- Resume from a checkpoint is bit-identical (test).

Storage: phi float64; phi_t and phi_r float32 (relative 6e-8; used only for flux, whose
criteria are at the 1e-1 level). A T = 20,000 run writes about 16 MB of series (`*_series.npz`):
b1_primary, b2_dip, b1_linear about 16 MB each, b1_box2 about 4 MB, b1_half_dr about 1.7 MB,
about 54 MB in all. The smoke series is 1.7 MB. These are results, not field arrays; committing
the production ones is the orchestrator's call (JSON and plots are small).

## 4. Run driver

`python -m swirl.session4.testB.run <name> ...` or `all` (the five production runs in order).
Grid: dr 0.05, dt 0.0125, sponge sigma = 1 ((r - R_s)/(R_box - R_s))^2 with R_s = 200,
R_box = 400 (the Session 2 box2 configuration that carried the dip to T = 300,000); all probes
(r <= 120) are in the undamped interior.

| Run | Start | Change | T | Expected wall |
|---|---|---|---|---|
| b1_primary | Session 1 table profile, omega 0.909149, phi_t = 0 | | 20,000 | 105 s |
| b2_dip | solve_profile(0.9065045072253043), phi_t = 0 | | 20,000 | 105 s |
| b1_linear | as b1_primary | g4 = g6 = 0 | 20,000 | 105 s |
| b1_box2 | as b1_primary | R_s 400, R_box 800 | 5,000 | 55 s |
| b1_half_dr | as b1_primary | dr 0.025 and dt 0.00625 (dt halved too, as Session 1's resolution check) | 2,000 | 45 s |

Measured: 5.1 ms of wall per time unit at R_box = 400 (smoke, one process, Test A running in
parallel). The whole list is about 7 minutes; analysis about 10 s per run. Checkpoints every
2000 time units in `checkpoints/session4/testB/` (gitignored); rerunning a name resumes. Every
output carries the full parameters (`params` in the npz and the run JSON). Writes are atomic.

## 5. Analysis and verdict

`python -m swirl.session4.testB.analysis <name> [--box2 <name>] [--compare a,b]`. Definitions are
in the module docstring. Epochs are [0, 5000], [5000, 10000], ... (one epoch [0, T] if
T < 5000), each with four 2000-unit Hann windows at 50% overlap.

Two methods I settled on from the smoke run, before any production run:

1. **Loss rate weighting.** Spectral fluxes are Hann^2-weighted time averages. A uniform
   endpoint rate over an epoch that contains the early shedding does not match them: on the
   smoke run the uniform -dE_in(10)/dt was 0.01114 and the time-averaged flux 0.01101, against
   0.00946 for the spectral flux. The loss rate is therefore taken with the same Hann^2 window
   weights (integration by parts on E_in(t)); with it, total spectral flux and
   -dE_in(r)/dt agree to 1e-5 to 2e-5 at every probe from r = 10 to 120. Uniform endpoint rates
   are kept in the output (`uniform_endpoint_rates`).
2. **Leakage floor.** A pure-line resynthesis of the near-threshold peak underestimated its
   leakage by about 200 times, because that content is a chirp (the shed transient). A
   bin-by-bin weighting overestimated by 2 to 230 times. The floor uses pure-line resynthesis
   for the harmonic lines and, for the band [omega_cut, omega_cut + 0.5], a non-negative fit
   by pure tones with the pipeline's own response. On synthetic signals the estimate is 0.83 to
   6.8 times the true leakage (chirp 0.90, free Klein-Gordon transient 0.83 and 4.6, pure tones
   1.9 at 1.005 and 6.8 at 1.0003). It is biased high, which makes criterion 1 easier to pass
   and the "present" clause harder, by at most about 7.

My readings of B4 where the text leaves a choice (implemented in `verdict_epoch` / `verdict`):

- Criteria 1 and 3 must hold at every probe with r >= 30 (30, 45, 60, 80, 120).
- Lines with omega in [omega_cut, omega_th) count as "above threshold" (only omega < omega_cut is
  sub-threshold, matching criterion 1).
- A "sub-threshold line present" for criterion 2 is a detected line with omega < omega_cut and
  power above max(leakage, evanescent floor) at a far probe; lines are grouped across probes
  within 2 bins; the exponent is a log-log fit over the probes where it is detected; "consistent
  with e^(-kappa r)/r" means the fitted kappa (log(a r) against r) is within 30% of
  sqrt(1 - omega^2). If it is detected at one probe only, the band power at the next far probe is
  an upper limit and the bound exponent must be steeper than -1.3.
- Criterion 3 uses method 1 (cross spectrum); method 2 is reported beside it.
- "Present", clause A, needs the doubled box. b1_box2 covers only [0, 5000], so its single epoch
  confirms lines in any epoch of b1_primary. B-run 2 has no doubled box, so a strong 1/r line
  there can only give "Undetermined" (flagged `present_a_unconfirmed_by_box2`).
- "Present", clause B: at >= 2 far probes the line flux is below loss/3 and the sub-threshold
  line flux is within a factor 3 of the deficit.
- Order: "present" if either clause holds, else "only above" if all three criteria hold, else
  "undetermined". Overall over epochs: "present" if any epoch is, "only above" if every epoch
  is, else "undetermined".
- Consequence worth knowing: a bound line at a frequency other than the fundamental, strong
  enough to exceed 10 times the floors, fails criterion 1 (its power is not covered by the
  fundamental's evanescent floor) even though criterion 2 calls it bound, so the verdict is
  "Undetermined" (synthetic test 4).

Tests of the verdict on synthetic series in the driver's format: clean (only above threshold,
flux = loss: ratio 0.99999); an outgoing 1/r line at 0.95 reproduced in a "box2" ->
"present"; the same without box2 -> "undetermined"; a bound line at 0.95 -> c2 passes, c1
fails, "undetermined"; loss 10 times the line flux -> "undetermined"; loss 10 times the line
flux with a sub-threshold line carrying the deficit -> "present" (clause B).

## 6. Smoke run: b1_smoke (B-run 1 to T = 2000)

Wall 10.2 s run plus 2.3 s analysis. Regime: omega/m 0.90947 (zero crossings, t 50 to 150),
nonlinear share -0.316 at t = 0 (signed) and -0.130 on average. E0 = 2843.77 (Session 1:
2843.77); frac(2000) = 0.989797 (Session 1, R_s 100: 0.989803). E_h never rises (max step rise
2.4e-10 relative).

- Core (r = 0, [0, 2000]): omega_core 0.90965; odd harmonics only, amplitudes 6.73 (omega),
  0.457 (3 omega), 0.0916, 0.0149, 2.7e-3, 4.9e-4, 9.1e-5, 1.7e-5, 3.2e-6 (17 omega); 19 omega to
  25 omega appear folded about the probe Nyquist frequency and are labelled as aliases. No even
  harmonic and no sideband above the detection contrast at r = 0.
- Far probes (r = 30 to 120): 3 omega at 2.7289, a_n = r a = 0.01471 to 0.01461; 5 omega at
  4.5482, a_n = 2.34e-4 (62.6 to 62.9 times weaker in amplitude than 3 omega, 0.07% of the flux);
  7 omega a_n 3.2e-6; 9 omega 1.1e-8. Near-threshold unidentified line from 1.0011 (r = 30) to
  1.0082 (r = 120), amplitude 2.3e-5 to 3.0e-5: the shed transient. In 400-unit windows it
  follows the free Klein-Gordon local frequency t/sqrt(t^2 - r^2) from above (r = 120:
  1.135, 1.048, 1.021, 1.011, 1.007 at t = 200 to 1000, against 1.25, 1.048, 1.021, 1.011, 1.007).
  Fundamental at r = 30: 7.37e-6 against 7.32e-6 predicted by A e^(-kappa r)/r; at r = 45:
  9.67e-9 against 9.6e-9; not detected beyond.
- Sub-threshold power against floors (power): r = 30: 2.71e-11 against 2.68e-11 (evanescent;
  ratio 1.01); r = 45: 1.78e-15 against 7.7e-15 (leakage; 0.23); r = 60: 6.5e-16 against 3.0e-15
  (0.21); r = 80: 4.5e-16 against 1.7e-15 (0.26); r = 120: 2.8e-16 against 2.3e-15 (0.12).
  Relative to total power: 2.2e-4 at r = 30, 2e-8 to 4e-8 beyond.
- Flux against loss: line flux above threshold 0.00943 (r = 30) to 0.00930 (r = 120), method 2
  equal to 1e-4; loss rate -dE_in(3 hw)/dt 0.00946 (Hann^2-weighted; uniform endpoint 0.01113);
  absorbed rate 0.00929. Ratio 0.996 to 0.983.
- Verdict on this one epoch: "Emission only above threshold" (c1, c2, c3 true). This is a smoke
  check of the pipeline, not the Test B result.
- Also seen: a broad continuum near omega 1.5 to 2 at power 1e-15 to 1e-13 at every r >= 10
  (below the line contrast), and an unidentified line at 2.6379 (3 omega - 0.091) at r = 15,
  amplitude 1.4e-7. Both are for the production epochs to settle (transient or breathing sideband).

## 7. Open issues and parking lot

- The doubled box covers [0, 5000] only, so "present" can be confirmed for later epochs only
  through a line that already shows in the first epoch. A box2 run to 20,000 costs about 210 s.
- B-run 2 has no doubled box in the plan; a strong 1/r sub-threshold line there could only read
  "Undetermined".
- The leakage floor is biased high by up to about 7 (section 5).
- Superseded by section 8: the fixed line bands of width 3 + ceil(n drift / 2 bins).

## 8. Fixes after the production runs (reviewer's defects, 2026-11-09)

The production runs (s4-testB, a68b0ab) were analysed with the code above. The reviewer found
the following, fixed on top of a68b0ab; the six analyses were rerun with the same options from
the same series files (code for the runs unchanged).

1. **Line bands overlapped or left gaps.** Bands of +-(3 + n drift) bins about each peak overlapped
   when the in-epoch drift split a line into sub-peaks (b1_primary [15000, 20000], r = 30: the
   3 omega bands [896,903], [901,908], [898,919] counted 10 bins twice; line flux 0.1116 against
   0.0646 in their union) and left gaps elsewhere (epoch 2, r = 45 and 60: 28% of the flux outside
   every band). Now (`line_bands`): bands are grown strongest first and never share a bin; a
   band covers its hump down to max(1e-4 of its peak, the local background: the median 10 to 40
   bins away), crossing valleys only while they stay above 1e-2 of its peak, so the sub-peaks of a
   drift-split line belong to it while a weak neighbour behind a deep valley stays separate
   (the 2.6255 and 2.8158 lines at the dip keep their own bands); touching bands with a shallow
   junction merge. Line flux is the band sum, so the line fluxes add up exactly to the in-band
   flux; the rest is reported (`flux_outside_bands_m1`). Labels use the band centroid, with
   omega_core over its in-epoch per-window range (2 bins about n omega_core when the drift is 0).
   Method 2 uses the centroid frequency.
2. **Doubled box epochs.** `verdict()` compared only box2 epoch 0. The doubled boxes now run to
   20,000, so each epoch is confirmed against the box2 epoch with the same [t0, t1]
   (`box2_epoch_matched` in each epoch row; all four matched for both runs).
3. **Near-threshold content** (peaks from 1.0004 to 1.0077, within about 3 bins of omega_th) is the
   spectrum edge plus the shed transient, not a line: peaks or centroids in
   [omega_cut, omega_th + 3 bins] are labelled "near-threshold band". They stay in the
   above-threshold flux (they are above threshold). `flux_above_harmonics_m1` (the n omega lines
   only) is reported beside `flux_above_m1`.
4. **Test** `test_drifting_and_split_lines_flux_adds_up`: 3 omega drifting by 0.04 (13 bins) and 5
   omega by 0.067 in the epoch, two tones 4 bins apart, and a near-threshold chirp. The bands are
   disjoint, each drifting line is one line, the two tones are one line with two merged peaks, and
   the summed line flux equals the total flux to 1.3e-5 at r = 30 and 120 (target 1e-3).
5. **Absorbed energy in the radial runs is not an independent check.** `run.py` stores
   `absorbed = E_h(0) - E_h(t)` by construction, so "E + absorbed conserved" is a tautology
   there. The real check is that E_h, which the semi-discrete system conserves exactly without
   the sponge, never rises with the sponge on (the sponge only removes energy). Measured, largest
   rise between consecutive 1.0-spaced samples, relative to E_h(0): b1_primary 2.39e-10,
   b2_dip 2.44e-10, b1_box2 2.39e-10, b2_box2 2.45e-10, b1_half_dr 1.51e-11 (dt halved: about 16x
   smaller, as an O(dt^4) RK4 error), b1_linear never rises (-8.8e-12). The rises are RK4 error on
   the nonlinear core, about 7e-7 in absolute energy per sample, against absorbed totals of 1117
   (primary) and 3.5 (dip) over 20,000.

Flux over loss rate (method 1, above-threshold bands), as-run -> fixed:

| Run | Epoch | r = 30 | r = 120 | Verdict (as-run -> fixed) |
|---|---|---|---|---|
| b1_primary | [0, 5000] | 0.996 -> 0.996 | 0.980 -> 0.980 | only above -> only above |
| b1_primary | [5000, 10000] | 0.994 -> 0.993 | 0.971 -> 0.971 | only above -> only above |
| b1_primary | [10000, 15000] | 0.926 -> 0.992 | 1.149 -> 0.977 | only above -> only above |
| b1_primary | [15000, 20000] | 1.657 -> 0.971 | 1.647 -> 1.003 | only above -> only above |
| b2_dip | [0, 5000] | 1.004 -> 1.005 | 1.197 -> 1.401 | only above -> only above |
| b2_dip | [5000, 10000] | 0.997 -> 0.997 | 0.997 -> 0.997 | only above -> only above |
| b2_dip | [10000, 15000] | 0.997 -> 0.997 | 0.997 -> 0.997 | only above -> only above |
| b2_dip | [15000, 20000] | 0.997 -> 0.997 | 0.997 -> 0.996 | only above -> only above |

The dip's first epoch at r = 120 rises because the near-threshold band now holds its whole hump
(flux 3.7e-6 against 1.8e-6 in the old +-3 bins), against a loss rate of only 9.2e-6: the start
transient passing r = 120 is comparable to the dip's steady emission. With the harmonic lines
only the ratio is 0.997 there; with them only, every epoch of both runs lies in 0.971 to 1.003.
Flux outside every band: at most 2.4e-4 of the total at r >= 30 for the primary; for the dip
1.6e-5 except epoch [0, 5000], where it reaches 16% at r = 120 (the broadband transient).
No verdict changed: b1_primary, b2_dip, b1_box2, b2_box2 and b1_half_dr "Emission only above
threshold" in every epoch; b1_linear "Undetermined" in every epoch (no lump, so criterion 3 has
no meaning there).
