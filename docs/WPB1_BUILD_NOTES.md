# WP-B1 build notes: the string solver for Track C

Session 5, 2026-11-10. Worker `numerics` (model claude-opus-5-5 as I see it; effort setting not visible to me). Code in `swirl/string/`, tests in `tests/string/`, results in `outputs/string/`. Derivations below are mine and were written without looking at the reviewer's notes.

Labels used: **derived** (algebra, no run), **measured** (from a run), **calibrated** (a setting chosen by a run).

## 1. Model and units

u(x, t) sideways displacement; long-wave speed 1, smallest scale 1, limit of stretch 1. Stretch d: u_{n+1} - u_n on the grid, u_x in the smooth string. Laws: G1 F = d/(1 - d^2), V = -(1/2) ln(1 - d^2); G2 F = d + beta d^3, V = d^2/2 + beta d^4/4; linear twin F = d, V = d^2/2. In each law V' = F, F odd, F'(0) = 1.

## 2. Equations of motion and energies (derived)

**C-grid.** L = sum_n v_n^2/2 - sum_n V(u_{n+1} - u_n). Euler-Lagrange: u_n'' = F(d_{n+1/2}) - F(d_{n-1/2}). E = sum v_n^2/2 + sum V(d). Linear: omega = 2 |sin(k/2)|, band top 2.

**C-smooth, continuum.** L = int [u_t^2/2 + u_xt^2/2 - V(u_x)] dx. Varying u: -u_tt + u_xxtt + (F(u_x))_x = 0, i.e. u_tt - u_xxtt = (F(u_x))_x. E = int [u_t^2/2 + u_xt^2/2 + V(u_x)] dx; the u_xt^2/2 term is the energy of the u_xxtt inertia term (standard 3). dE/dt = int [u_t (u_tt - u_xxtt) + F(u_x) u_xt] dx after one integration by parts; the second term is -int u_t (F)_x, so dE/dt = 0 on solutions. Linear: omega^2 = k^2/(1 + k^2), ceiling 1 approached as k -> infinity, never reached. With w = u_x: (1 - d_xx) w_tt = d_xx F(w).

**C-smooth, scheme 'fd' (staggered finite differences, spacing h).** Nodes u_j at x_j = j h, stretch d_{j+1/2} = (u_{j+1} - u_j)/h on bonds. Discrete Lagrangian
L_h = h sum_j v_j^2/2 + h sum_bonds [((v_{j+1} - v_j)/h)^2/2 - V(d_{j+1/2})].
Euler-Lagrange: h (I - D2) a = F(d_{j+1/2}) - F(d_{j-1/2}), i.e. (I - D2) a_j = (F_{j+1/2} - F_{j-1/2})/h, D2 the 3-point Laplacian; a cyclic tridiagonal solve per force evaluation. Exactly conserved semi-discrete energy:
E_h = h sum v_j^2/2 + (1/2h) sum (v_{j+1} - v_j)^2 + h sum V(d_{j+1/2}).
Linear: omega^2 = s^2/(1 + s^2), s = (2/h) sin(k h/2). **Band top of the scheme** (k = pi/h): omega_top = 2/sqrt(4 + h^2): 0.992278 (h = 1/4), 0.998053 (1/8), 0.999512 (1/16). Rates in (omega_top(h), 1) are carried by the continuum (as very short waves) and not by the scheme; a state whose rate sits there, or whose width follows h, is a property of the h-grid.

**C-smooth, scheme 'spectral' (Fourier, spacing h, ring).** D the spectral derivative with the Nyquist mode zeroed (keeps D real and antisymmetric). L_h = h sum [v^2/2 + (Dv)^2/2 - V(Du)] gives (I + D^T D) a = -D^T F(Du), i.e. (I - D^2) a = D F(Du); E_h = h sum [v^2/2 + (Dv)^2/2 + V(Du)], exactly conserved. Linear: omega^2 = k^2/(1 + k^2) exactly for |k| < pi/h; band top omega_top = (pi/h)/sqrt(1 + (pi/h)^2): 0.996849, 0.999209, 0.999802. The Nyquist mode of u is a decoupled zero-rate mode (no force acts on it). The G1 guard is checked at the points only.

**Why two schemes.** They share the continuum limit and differ at the grid scale: 'fd' has O(h^2) dispersion error (measured: -4.5e-3 at the band top for h = 1/4, -2.9e-4 at h = 1/16); 'spectral' has none below pi/h, but aliasing in F(Du) and Gibbs ringing at a jump in stretch. A result that differs between them is a grid-scale result. 'fd' is numba (fast); 'spectral' is numpy (about 10 times slower per point-step).

## 3. Time stepping (derived, then measured)

Fourth-order Yoshida triple jump of velocity Verlet (w1 = 1/(2 - 2^(1/3)), w0 = -2^(1/3) w1), FSAL: 3 force evaluations per step. Symplectic for the closed ring, so the energy error is bounded and O(dt^4), not drifting (check 2: ratio 16.0 between dt and dt/2 in every case). The mass operator is constant, so the splitting is exact (kick with M^{-1} force, drift u and d). Step rule: dt omega_max <= about 0.1, omega_max = c sqrt(max F'(d)) (c = 2 grid, 1 smooth). In C-smooth the linear band is below 1 at every h, so dt is set by the peak stretch, not by h.

## 4. Absorber (calibrated)

Ring of length L; sponge sigma(x) = sigma_max ((|x| - x_s)/L_s)^2 on |x| > x_s = L/2 - L_s, the two ends meeting inside it. **Production default (revised after review): L_s = 300, sigma_max = 0.05** (`SPONGE_LEN`, `SPONGE_SIGMA` in `driver.py`; was 100 / 0.3).

Why it was revised: reflection at the onset of the ramp scales with its curvature 2 sigma_max / L_s^2. The test is a grid band-edge packet (k = 0.9 pi, group velocity 0.156) moving into the sponge, comparing the interior energy of a string and one twice as long (test in check 4). Reflected fraction:

| L_s / sigma_max | Reflected fraction |
|---|---|
| 100 / 0.3 (old) | 3.5e-4 |
| 200 / 0.1 | 2.7e-5 |
| 200 / 0.05 | 1.4e-5 |
| 300 / 0.1 | 1.2e-5 |
| 300 / 0.05 (chosen) | 6.0e-6 |
| 400 / 0.05 | 3.4e-6 |

The reviewer measured 8.4e-4 for the old sponge with a different construction. Long waves still lose e^-20 of their energy on one traverse of the new sponge (exp(-2 int sigma dx), int over 2 L_s = 2 sigma_max L_s / 3 = 10). Cost: the ring is 600 units longer, and every step with damping now does 4 force evaluations instead of 3 (see below). The driver refuses a job whose interior half-length L/2 - L_s is below max(W, profile_range).

**Stale acceleration (fixed after review).** The damping half-step changes d, so the carried (FSAL) acceleration must be recomputed after it. Without the recomputation the energy balance in the sponge was first order in dt. Balance |E + absorbed - E0| / E0 for fully absorbed runs at dt = 0.02:

| Run | Before | After |
|---|---|---|
| Grid, reviewer's measurement | 2.65e-4 | — |
| fd linear (my check-5 runs) | 2.0e-5 | 1.7e-10 |
| fd G1 (my check-5 runs) | 8.1e-6 | 4.7e-10 |
| Grid linear | — | 2.6e-10 |
| Grid G1 | — | 9.8e-9 |

The statement below about a splitting error of about (sigma dt)^2 is withdrawn: that error was this bug.

**Matched (distortionless) sponge.** The first version damped the velocity only. It failed check 4 on the interior field: difference-field energy 4.5e-6 of E0 against the 1e-6 target, at any sponge length tried up to 400. Cause (derived and confirmed): waves with omega below about (sigma_max / L_s^2)^(1/3) (about 0.01, wavelengths longer than the sponge) are reflected by a velocity-only damper, whose impedance at low frequency is sqrt(1 + i sigma/omega). Fix: carry the stretch d as its own variable and damp velocity and stretch at the same rate:
M v_t = force - B v, d_t = (stretch of v) - sigma d, B = sigma + D^T sigma D ('fd', spectral; diagonal sigma on the grid).
For the long-wave limit (v_t = d_x, d_t = v_x) equal damping of both fields is Heaviside's distortionless line: the impedance is unchanged at every frequency, so long waves enter without reflection. Both terms only remove energy: dE/dt = -v.Bv - h sum sigma_b F(d) d <= 0 (F(d) d >= 0 for every law). Implemented as an exact (grid) or Crank-Nicolson ('fd': cyclic tridiagonal; spectral: fixed-point iteration) dissipation step of dt/2 before and after each symplectic step (Strang); the removed energy is accumulated exactly in `absorbed`, and every record reports the balance (E + absorbed - E0)/E0. That balance is limited by the splitting, about (sigma dt)^2 per unit of absorbed energy (measured 2e-5 of E0 for a fully dispersed linear run at dt = 0.02; 8e-6 for G1). It does not touch the window, where sigma = 0. Inside the sponge d is no longer the stretch of u, so u there is a diagnostic only. The B term damps every wavenumber at rate about sigma; the velocity-only damping of a C-smooth short wave would be sigma/(1 + k^2).

**Short waves in C-smooth.** The group velocity is (1 + k^2)^(-3/2) in the continuum. In 'fd' it falls further near the band top: 3.0e-5 (h = 1/4), 3.8e-6 (1/8), 4.8e-7 (1/16) at 0.99 of pi/h. Time to leave a window of half-length 10 and to reach a sponge starting at 120: k = 1: 28 and 339; k = 2: 112 and 1,342; k = 3: 316 and 3,795; k = 5: 1,326 and 15,909; k = 10: 10,150 and 121,804. Waves with k >= 3 do not reach any absorber in a run shorter than about 4,000. They are not reflected; they stay near the source and, where k >= 1, leave the window slowly. This is linear physics (the twin shows it too).

**Unbounded stretch in C-smooth G1 (Block 4, item 6).** The energy bounds int V(u_x) dx, not sup |u_x|. In 'fd' one bond carries h V(d) <= E, so d <= sqrt(1 - exp(-2E/h)), which tends to 1 as h -> 0. A narrow spike can reach the pole at finite energy, and finer h makes it easier. A guard trip there is a finding to record, not something to avoid.

**The C-grid G1 long-wave speed exceeds 1.** The local speed of long stretched waves is sqrt(F'(d)): 1.82 at d = 0.6. Check 4 stops its comparison before the double string's own sponge can echo back at that speed.

## 5. Guard (check 6)

Every force evaluation (each of the 3 stages of every step) checks |d| < 1 (and not NaN) on every bond. On failure the step is undone and `GuardTrip` carries the time, the stage, the bond and its position, the stretch, the energy and absorbed energy of the last good state, and that state (u, v, d). The driver writes the state to `checkpoints/string/<name>_guard_state.npz` and puts the diagnosis in the result JSON. After review: the driver also reruns the last stretch automatically, from the state at the start of the sample chunk in which the guard tripped, at dt/2 up to t_fail + max(10 dt, 1). It records the new trip time. A trip that disappears or moves by more than dt is labelled "numerical (moves with dt)"; otherwise "persists at dt/2". Nothing is clipped. In the exact dynamics G1 cannot reach |d| = 1 at finite energy on the grid. A trip there is therefore an integration statement: either the energy forces 1 - d^2 below double precision (the test start: energy 25 in one bond, 1 - d^2 >= 2e-22), or dt is too large for the local stiffness.

## 6. Exact periodic states and Floquet multipliers (tools a and b)

Unknowns u(0) with v(0) = 0. Equations: u(T/4) = 0 at every point. Derived: F odd and the scheme time-reversible give u(-t) = u(t) and u(T/4 + s) = -u(T/4 - s), hence u(t + T/2) = -u(t) and period T. The quarter-period map is integrated by the same Yoshida scheme with T = 4 nq dt, so the state is an exact periodic orbit of the discrete map, O(dt^4) from the flow's orbit. Newton uses the exact tangent map (dense) with step halving. The uniform shift is not a null direction (it survives to T/4), so no pinning is needed on the ring. Floquet: the monodromy of the discrete map over one period, 2N x 2N, symplectic by construction. Two exact symmetries (time shift; uniform shift on a ring) put two Jordan pairs at +1, split by round-off to about sqrt(eps |M|) (up to 1e-5 at peak stretch above 0.98). So the result reports the raw largest modulus, the largest modulus excluding the four multipliers nearest +1, the distances from 1 of those four, and the distance from 1 of the nearest kept multiplier, so that a genuine multiplier within 3e-6 of 1 stays visible. It also reports the parity of the leading eigenvector under the reflection about the object's centre. Family continuation in rate uses a secant predictor and an adaptive step. Each point reports rate, peak stretch, peak displacement, E, E per cycle 2 pi E / rate, L_half and rms width of the cycle-averaged energy, the nonlinear share (cycle average of (E - E_quadratic)/E), and the tail decay rate fitted to the component at the rate (Block 4 item 5), beside O1 and the envelope fit. It also reports the period-return error, the Floquet numbers, and the profile.

Tool b runs the same Newton on 'fd' at spacing h from a Gaussian guess at the O4 lone-oscillator rate (or a given rate). It reports what happens: convergence, the residual history, and for a converged state the diagnostics above plus the conditioning of the quarter-map Jacobian (sigma_min/sigma_max; near zero means the state is not isolated).

## 7. Single-harmonic profile of C-smooth with G1: jump profiles (derived; Block 4 item 2)

With w = u_x: (1 - d_xx) w_tt = d_xx F(w). Keep one harmonic, w = a(x) cos(omega t). The first cosine coefficient of G1 is exact in closed form, f(a) = (2/a)((1 - a^2)^(-1/2) - 1) = a + 3a^3/4 + .... Projection gives

(f(a) - omega^2 a)'' = -omega^2 a.

Write y = g(a) = f(a) - omega^2 a. Then y'' = -omega^2 a, with first integral y'^2/2 + Phi(a) = C, where Phi(a) = omega^2 (a g(a) - int_0^a g) = omega^2 int_0^a s g'(s) ds (even in a), and int_0^a f = 2 ln(2/(1 + sqrt(1 - a^2))). Integrating over the line gives int a dx = 0 exactly: a localized state has no net stretch.

Shape of g for omega > 1: g'(0) = 1 - omega^2 < 0. g falls to a minimum at the fold A* (f'(A*) = omega^2) and rises to +infinity as a -> 1. It is zero again at A_z (f(A_z) = omega^2 A_z), with A* < A_z. Tail: y ~ exp(-kappa |x|), kappa = omega/sqrt(omega^2 - 1) (O1).

**No continuous localized profile exists.**
- (i) On the lower branch alone (|a| < A*), g(a)/a < 0, so y'' = c y with c > 0. Multiplying by y and integrating, a solution decaying at both ends would need -int y'^2 = int c y^2, so it is zero.
- (ii) A continuous profile that reaches the upper branch must pass a = A*, where y takes its least value g(A*). There y'^2 = 2(C - Phi(A*)), and C = 0 from the tail while Phi(A*) < 0, so y' != 0 at a minimum of the reachable y. That is impossible.

**Jump profiles exist.** In the weak form y and y' are continuous; a switches branch at fixed y. Inside the outer jump C = Phi(a_top), and in the tail C = 0. At the outer jump y > 0 and y' < 0 (y is concave while a > 0), so the tail has the opposite phase (a_out < 0). The matching condition is Phi(a_top) - Phi(a_s) + Phi(a_out(a_s)) = 0, with g(a_out) = g(a_s) on the lower branch. For each rate this gives a one-parameter family (a_top, or equivalently the jump distance x_s), limited above where the opposite branch runs out (g(a_s) <= -g(A*)).
- **S-bend-like** (a even, u odd): plateau from a_top at x = 0 down to a_s at x_s, then a jump to the tail.
- **Bump-like** (a odd, u even): a jumps from -A_z to +A_z at x = 0 (y = 0 there), rises to a_top, falls to a_s, then jumps to the tail.

Constructed by quadrature in `swirl/string/jump.py`, with a series for Phi below |a| = 1e-2 (the closed form cancels there). Checked by test: first integral to 4e-7 (inner) and 2e-5 (tail) of |Phi(a_top)|; y continuous across the jumps to 1e-16; net stretch zero to the trapezoid error of the jumps; tail rate equal to kappa to 1e-8. Measured family numbers, A* and A_z: rate 1.2: 0.393 and 0.626; rate 1.5: 0.567 and 0.824. Widest S-bend-like profile at rate 1.2: x_s = 0.225. The driver start `{"type": "jump", "obj", "omega", "x_s"}` samples a(x) where the scheme keeps the stretch, removes the sampling's small net stretch, and reports `plateau_cells` = 2 x_s / h (below 1: the scheme cannot represent the plateau).

## 8. Analytic expectations before the PDE (standard 6; derived)

- **C-grid, rotating wave** (patterns from the plan; F1 the first cosine coefficient):
  - bump A(-1/2, 1, -1/2): omega^2 = 2 F1(1.5A)/A; G2: 3 + (81/16) beta A^2.
  - S-bend A(-1, 1): omega^2 = (F1(A) + F1(2A))/A; G2: 3 + (27/4) beta A^2.
  - At small amplitude G1 is G2 with beta = 1 (F = d + d^3 + ...).
  - These two-site patterns are poor near the band top, where the exact states widen (tail exp(-kappa n), 2 cosh(kappa/2) = omega, O1). The exact solver uses continuation there.
- **C-smooth, lone oscillator** (O4): rate 2 pi / (4 int_0^a dw / sqrt(ln((1 - w^2)/(1 - a^2)))); 1.111 at a = 0.5, 1.423 at a = 0.8.
- **C-smooth linear twin** (O3): orchestrator's prediction on record.
- **Pure-quartic limit of G2** (SPS abstract): growth rate 0.15 "in units of the mode frequency".

## 9. Diagnostics (driver, per record; Block 2.1 and Block 4)

- **Windows and frac.** Fixed window |x - x0| <= W, with W = max(10, 5 L_half) from the t = 0 energy density (instantaneous, since the start is at rest). The co-moving window follows the energy centroid (two centroid iterations inside the window, minimal image on the ring). frac(t) = record-averaged window energy / window energy at t = 0. One record = record_periods nominal periods; the nominal period is the start's rate, or a 60-unit pre-run's zero crossings.
- **Lifetimes.** T_90 and T_death as in Block 2.1, "> cap" when alive at the cap.
- **Rates.** From zero crossings and the spectral peak of the core coordinate. Block 4: bump = displacement at the centre; S-bend = stretch across the centre (grid u_c - u_{c-1}; 'fd' (u_{c+1} - u_{c-1})/2h). Also reported: rates of the stretch at the bond of initial peak stretch and of the projection on the start.
- **Width.** L_half and rms width of the record-averaged window energy.
- **Energy accounting.** Peak displacement and stretch in the window, energy balance.
- **Jump measure (C-smooth).** Mean wavenumber of the stretch in the window. On the record's peak-stretch sample: the largest jump between neighbouring cells, its position, the total variation of the stretch, and the number of cells the largest jump spans.
- **Run-level outputs.** Time-averaged displacement, stretch and rms displacement against distance, and the amplitude of u at the nominal rate against distance with its tail fit inside the window. Probe series (u at chosen distances, the core coordinates) go to `<name>_series.npz`. Optional field snapshots go to `checkpoints/string/` (not committed).
- **Push.** v = -lambda du/dx at a turning point (u_t = -c u_x for rigid motion, so lambda is a nominal speed). Reported: the measured speed (centroid fit after the first 10% of the run), the energy lost per unit distance from the co-moving window, whether it broke (co-moving frac < 0.5), and frac when it crosses dist_target. Measured speeds are 3 to 6% of lambda (the kick projects weakly on translation).

**Changes after review (2026-11-10), in the driver and diag:**
- **Rates and cycles.** rate_early and rate_late are the medians of the records' zero-crossing rates over the first and last 10% of records. The probe-window rates (the first probe_max samples, a transient) are renamed `*_first_window`. cycles_at_end, T_90_cycles and T_death_cycles integrate the record rates over time.
- **Tail fit.** The component of u is taken over the last 20% of the run (`tail_fit_from` = 0.8), Hann-tapered, at a bank of 81 rates within +-10% of the record rate at the start of that interval. The bank rate nearest the rate measured over the interval is fitted. Outputs: `tail_kappa_at_late_rate`, `tail_kappa_O1_at_late_rate`, `tail_fit_rate`, and the profile `amp_u_at_late_rate`.
  - Asserted: trapped grid S-bend at rate 2.5, -1.9e-5 against O1.
  - Asserted: exact C-smooth S-bend (h = 1/8, rate 1.423), -1.3e-3 against O1.
  - Reported: the held jump state from rate 1.2. Its rate drifts from 1.16 to 1.07, and its component at the late rate falls like O1 (kappa about 2.8) only from x = 1 to 1.5, then levels to a floor. Fitted kappa 0.61 against O1 2.85.
- **T_death baseline.** Interpolated at t = 50.
- **Wavenumbers.** `k_first_moment_stretch` and `k_rms_stretch` replace `k_mean_stretch`, which was the rms.
- **Defaults and refusals.**
  - Every start gets seeded noise 1e-8 (seed 0) unless start.noise = 0; the amplitude and seed are in start_info.
  - Push jobs run the twin by default.
  - A G2 job without beta is an error.
  - The interior-length refusal above.
- **New start type.** "smooth_orbit": an exact 'fd' state from tool b, embedded in the long ring.

## 10. Checks: measured against target

Full run after review: `pytest tests/string -q`: 67 passed in 512 s (first run: 57 passed in 233 s) (the slow tests are in that time: check 3, 4 runs of about 26 s each plus a 4 s diagnostic; the spectral closed-ring energy runs, about 17 s each). Every measured value is also in `outputs/string/checks/`.

| Check | Target | Measured (worst over the record) | Verdict |
|---|---|---|---|
| 1. Dispersion vs scheme, 6 to 7 k per case, grid and fd/spectral at h = 1/4, 1/8, 1/16 | 1e-6 rel | 1.9e-9 (fd h = 1/16) | hit |
| 1. fd vs continuum (reported) | none | -4.5e-3 at band top h = 1/4, -1.15e-3 (1/8), -2.9e-4 (1/16): O(h^2); spectral 0 | as derived |
| 2a. dE/dt from the EOM, random smooth data, 3 media x 3 laws | round-off | 7.8e-15 / power; negative control without the u_xt term 1.6 to 1.7 | hit |
| 2b. Closed ring, 1,000 time units, G1 peak stretch 0.8, G2 beta = 1, twin | 1e-8 rel | 3.6e-9 (grid G1, dt 0.004); fd and spectral G1 1.3e-9 and 1.2e-9 (dt 0.008) | hit |
| 2b. Order | dt^4 | ratio dt : dt/2 = 16.0 in all 9 cases; no drift (first and last 10% of run equal) | hit |
| 3. S-bend, exact start, Flach-Gorbach setup (G2 beta 1, Omega 4.5) | multiplier within 1e-6 of 1; amp, centre 1%, frac >= 0.99 at 32,000 cycles | multiplier 1 + 8e-15; amp -1.7e-5; centre drift 5e-10 sites; frac 0.99999988 at 31,995 cycles | hit |
| 3. S-bend, rotating-wave start (as in the SPS abstract) | same | frac 0.9971 (vs t = 0), 0.9984 (vs 100 cycles); centre 1.3e-9; amp -2.9% vs 100 cycles (relaxes from 1.5986 toward the exact state's 1.5435) | miss on amp (1% rule); rest hit |
| 3. Bump, exact start | multiplier > 1 + 1e-3; leaves site by >= 1 with co-moving frac >= 0.9 | multiplier 2.062, antisymmetric; centroid moves 0 -> -0.5 within the first 50 cycles and stays at -0.50 +- 0.01 to 32,112 cycles (converted to an S-bend half a site over); max distance 0.52; co-moving frac >= 0.998 (first run: +0.5, 0.52) | unstable: hit; moves >= 1 site: miss (Block 5 reading: shifts 0.5 toward the S-bend position and stays) |
| 3. Bump, rotating-wave start | same | after review (700-site ring, production sponge): leaves its site by 1 at cycle 97, settles at -2.50 (an S-bend position 2.5 sites away) and stays to 32,019 cycles; co-moving frac >= 0.997. First run (300-site ring, old sponge): 0.265 -> 0.50, max distance 0.52 | moves: hit now, miss before; the outcome depends on the perturbation |
| 3. Bump, exact, Omega = 2.3 (report) | as above | multiplier 1.073; on the 700- and 800-site rings the centroid hops 0 <-> -1 repeatedly over 3,000 cycles (max distance 0.9999992 and 0.9999995), frac 1.000; on the 400-site ring with sponge 100 / 0.3 (saved as data) it travels steadily to 7 sites, passing 1 site at cycle 501, frac 1.000 | moves 1 site: miss on two rings, hit on the third; whether it travels depends on the perturbation |
| 3. Pure-quartic growth rate (report) | SPS 0.15 | angular reading 0.147 / 0.153 / 0.155 at Omega 10 / 20 / 40; ordinary reading 0.925 / 0.962 / 0.972 | angular reading agrees within 3% |
| 4. Window energy, string vs twice as long | 1e-6 E0 | after review (sponge 300 / 0.05, acceleration fix): 1.2e-9 (grid G1) worst; linear 1.5e-17 to 2.2e-13. Before: 5.2e-8 | hit |
| 4. Interior difference-field energy | 1e-6 E0 | after review: 6.4e-11 (grid G1) worst. Before: 3.7e-7 | hit |
| 4. Absorbing case (C-smooth G1, sigma 1.5, replaces sigma 0.5 where only 6% was absorbed) | absorbed >= 0.5; 1e-6 | absorbed 0.566; window 3.7e-10; field 5.4e-14 | hit |
| 4. Band-edge packet, grid k = 0.9 pi, reflected fraction | 1e-5 | 6.0e-6 (production 300 / 0.05); old 100 / 0.3: 3.5e-4 | hit |
| 4. Energy balance, fully absorbed, dt 0.02 | 1e-6 | 9.8e-9 (grid G1) worst; 1.7e-10 to 4.9e-10 otherwise. Before the fix: 2.65e-4 grid (reviewer), 2.0e-5 fd | hit |
| 4. Energy never rises under damping | <= symplectic error | largest rise between records 3.4e-9 E0; damping increments >= -1.5e-15 E0 (round-off) | hit |
| 5. fd linear, h = 1/4, 1/8, 1/16 | convergence | pointwise order 2.04 / 2.03 / 2.03 / 2.02 at t = 50 / 100 / 200 / 300; frac order 3.5 to 4.3; rate order 2.07 | hit |
| 5. fd G1, sigma 0.5, peak stretch 0.8 | report | frac order 3.0 (t = 100), 2.4 (200), 2.05 (300; values 0.698 / 0.782 / 0.802); pointwise order 1.63 (t = 50), 1.11 (100), 0.06 (200), 0.55 (300): no convergence after t ~ 100; rates show no order (0.965 / 0.971 / 0.949); mean stretch wavenumber 8.7 / 11.6 / 21.7 grows like 1/h | no order for pointwise and rate: stated plainly |
| 6. Guard | trips with diagnosis | start at d = 1.05 trips at t = 0; dynamic trips in grid, fd, spectral; state left at last good (max d 0.992 / 0.840 / 0.909); no false trip on an exact S-bend at peak stretch 0.935 over 50 periods; driver rerun at dt/2 trips at the same time (shift 0, "persists at dt/2") for the energy-800 start | hit |

The worst values over the record: check 1, 1.9e-9; check 2a, 7.8e-15; check 2b, 3.6e-9; check 4, 3.7e-7.

Numbers that miss their target or rule:
- (a) rotating-wave S-bend start, amplitude rule 1%: 2.9%;
- (b) bump "moves >= 1 site" at Omega = 4.5 from the exact start: 0.52. After review the rotating-wave start meets it (2.52 sites), where the first run did not (0.52);
- (c) the same rule at Omega = 2.3: 0.9999995 (it hops 0 <-> -1);
- (d) check 5 G1 pointwise convergence after t ~ 100: none.

(a) is the start's distance from the exact state, which it sheds; (b) and (c) are physics, reported as Block 5 asks; (d) is the expected outcome of the phase-mixing picture (O5) and the jump structure (O10).

## 10b. First results from the tools (measured)

- **C-grid G1 families** (`outputs/string/family_grid_G1_*.json`, ring N = 184).
  - S-bend: rate 2.010 to 4.316, peak stretch 0.162 to 0.993, E 0.134 to 2.449, L_half 2.79 to 0.515.
  - Bump: rate 2.010 to 4.916, peak stretch 0.161 to 0.991.
  - Largest Floquet multiplier: S-bend 1 + 6e-14 excluding the 4 symmetry multipliers at +1 (raw up to 1 + 1e-5, the split Jordan pairs), stable at every point. Bump 1.0001 at 2.036, 1.053 at 2.138, 1.47 at 2.30, 4.8 at 2.72, 20.8 at 3.32, 605 at 4.92, always antisymmetric.
  - Tail at the rate against O1 within 2e-5.
- **C-smooth exact states, tool b** (`outputs/string/smooth_orbits_G1_*.json`). Newton converges in 20 of 36 attempts (S-bend 11, bump 9; none from sigma = 1 except S-bend at h = 1/8, a = 0.5) (sigma 0.25 to 1, guess peak stretch 0.5 and 0.8, rate from O4). Every converged state has a largest stretch jump spanning one cell, at every h.
  - Some reproduce across h: S-bend from sigma 0.5 at 0.8, L_half 0.1736 / 0.1515 / 0.1511 and E 0.3650 / 0.3618 / 0.3610.
  - Others follow h: S-bend from sigma 0.25 at 0.5, L_half 0.195 / 0.081 / 0.072.
  - Floquet multipliers, largest excluding the 4 at +1: bump-type 1.6 to 5.9 (all unstable); S-bend-type 1.000 to 1.65 (stable to round-off in 3 cases, all at h <= 1/8).
  - Jacobian sigma_min/sigma_max 0.01 to 0.09: isolated, but which state is found depends on the guess.
- **Push.** C-grid G1 S-bend: at rate 2.05 and 2.1 it moves for every lambda from 0.05 to 1 (speed 0.0015 to 0.056, loss per unit distance growing with speed); at 2.3 it stays pinned for lambda <= 0.2 and moves at lambda >= 0.5 (speed 0.039 to 0.053).
- **Cost on one core** (`outputs/string/cost_table.json`, wall time per 1,000 cycles with all diagnostics):

  | Run | Wall time per 1,000 cycles |
  |---|---|
  | C-grid G1 S-bend, N = 800 (interior 100 + sponge 300 each side), dt 0.01, rate 2.5 | 5.9 s |
  | fd G1, L = 840 (interior 120), dt 0.02, rate taken 1.1, h = 1/4 | 89 s |
  | fd, same, h = 1/8 | 175 s |
  | fd, same, h = 1/16 | 345 s |
  | spectral, same, h = 1/8 | 1,138 s |

  These are after review. The first table (old sponge, 3 force evaluations per step) read 2.3 / 30 / 59 / 115 / 481 s. Measured while a second process ran. Cost scales with N and with 1/dt; dt is set by the peak stretch, not by h.
- **Floquet near the band top (after review).** At rate 2.01 both families have, besides the 4 symmetry multipliers (distances 3e-11 to 2.8e-6 from 1), a pair at phase +-3.2e-6 rad on the unit circle (|lambda| - 1 = 2e-15). That is a very slow internal mode, now visible as intended (`floquet_nearest_kept_dist_from_1`). It is not an instability. Family numbers are otherwise unchanged (no sponge in the orbit solver).

## 11. How to run

`NUMBA_NUM_THREADS=1 caffeinate -i /Users/aaronramos/Desktop/SWIRL/.venv/bin/python -m swirl.string.driver <jobs.json>` from the repository root. Job kinds and keys are in the docstring of `swirl/string/driver.py`; an example is `swirl/string/jobs/wpb1_first.json`. Results go to `outputs/string/<name>.json` (full parameters inside, atomic writes) and `<name>_series.npz`. Runs checkpoint to `checkpoints/string/<name>.npz` every `ckpt_every_s` (300) seconds of wall time and resume when rerun with identical parameters (tested: identical records after a cut).
