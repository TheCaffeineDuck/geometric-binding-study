# WP-B2 build notes: radial solver for equations O and R, single bump and pinch

Session 5, Track F. Code in `swirl/radial5/`, tests in `tests/radial5/`, results in `outputs/radial5/`. Labels: **measured** (from a run), **derived** (algebra here), **calibrated** (a threshold chosen by me).

## 1. Model and reductions (derived)

Field u(x, t) with three components. Units: wave speed 1, gap 1, limit 1. S = sum_i |d_i u|^2, X = |u_t|^2 - S.

- O: L_O = (1/2)|u_t|^2 - (1/2)S - U(|u|), with U(f) = f^2/2 - g4 f^4/4 + g6 f^6/6.
- R: L_R = 1 - sqrt(1 - X) - |u|^2/2, for X < 1. The full square root is used everywhere, including the start profiles.

The two objects share one radial profile f(r, t), and in both |u| = |f|.
- **Single bump**, u = f e: d_i u = f_r x_i/r e, so S = f_r^2.
- **Pinch**, u = f x/r: d_i u_j = f_r x_i x_j/r^2 + (f/r)(delta_ij - x_i x_j/r^2). The two parts are orthogonal, and the transverse projector has trace 2, so S = f_r^2 + 2 f^2/r^2.

Write s2 = 0 for the bump and s2 = 2 for the pinch. The action is the integral of 4 pi r^2 L dr dt.

### Equations of motion and energies

**O, both objects:**

    f_tt = f_rr + (2/r) f_r - s2 f/r^2 - f + g4 f^3 - g6 f^5
    E_O  = 4 pi int r^2 [ f_t^2/2 + (f_r^2 + s2 f^2/r^2)/2 + f^2/2 - g4 f^4/4 + g6 f^6/6 ] dr

**R, both objects.** Write gamma = (1 - X)^(-1/2) and p = gamma f_t, the momentum density conjugate to f.

    d_t(gamma f_t) = r^-2 d_r(r^2 gamma f_r) - s2 gamma f/r^2 - f
    E_R = 4 pi int r^2 [ gamma (1 + S) - 1 + f^2/2 ] dr
        = 4 pi int r^2 [ sqrt((1 + S)(1 + p^2)) - 1 + f^2/2 ] dr

The second line uses 1 - X = (1 + S)/(1 + p^2), which follows from p^2 = f_t^2/(1 - f_t^2 + S).
- Second-order form. Use d_t gamma = gamma^3 (f_t f_tt - f_r f_rt - s2 f f_t/r^2). The coefficient of f_tt is gamma^3 (1 + S) > 0, so f_tt is explicit.
- Hamilton's equations:

      f_t = p sqrt((1 + S)/(1 + p^2)) = p/q,   q = sqrt((1 + p^2)/(1 + S)) = gamma
      p_t = r^-2 (r^2 q f_r)_r - s2 q f/r^2 - f

- Small amplitude. 1 - sqrt(1 - X) = X/2 + X^2/8 + ..., so R reduces to Klein-Gordon with mass 1 and speed 1.
- The quartic gradient energy in static E_R is sqrt(1 + S) - 1 = S/2 - S^2/8 + ...: negative. The term retired on 2026-10-01 (LESSONS 6) had L containing -kappa X^2, giving energy +kappa |grad|^4. The sign is opposite.

### Map between the plan's couplings and `swirl/family/radial.py`

radial.py's potential is m^2 phi^2/2 - g4f phi^4/24 + g6f phi^6/720, so:
- g4 = g4f/6 and g6 = g6f/120;
- beta_file = g6f/g4f^2 = (10/3) g6/g4^2.

The Session 1 point, g4f = 0.30 and g6f = 0.12 (beta 4/3), is g4 = 0.05 and g6 = 0.001. The code passes 0.30/6 and 0.12/120, so the arithmetic is bit-identical to radial.py.

### Regularity at r = 0

- Bump: f is even in r, f_r(0) = 0.
- Pinch: f is odd, f = a r + b r^3 + ... Then f(0) = 0, S(0) = 3a^2 is finite, and s2 f/r^2 is regular.
- In u = r f (the O code): u is odd for the bump and even for the pinch (u = a r^2 + ...), with u(0) = 0 in both.

### Well-posedness of R (derived; checked numerically)

**Radial equation.** Write a = f_t, b = f_r, W = s2 f^2/r^2.
- The principal symbol has L_aa = gamma + a^2 gamma^3 > 0 and L_ab^2 - L_aa L_bb = gamma^4 (1 + W) > 0.
- So the radial equation is strictly hyperbolic whenever X < 1. No further condition appears.
- The characteristic speeds are v = [-a b gamma ± sqrt(1 + W)] / (gamma (1 + b^2 + W)).
- Over 123,233 random admissible states: no non-hyperbolic case, and max |v| = 1 - 8e-11. The speeds never exceed 1.

**Three-component theory: the scholar's DF-5. I agree.** I derived it independently.
- The principal symbol is (1 - X)(xi.xi) delta^ab + (du^a.xi)(du^b.xi). It has a light cone twice and the cone G(xi) = (1 - X) xi.xi + sum_a (du^a.xi)^2 once.
- For the pinch at a point on the x axis, du_t = f_t e_x, du_x = f_r e_x and du_y,z = (f/r) e_y,z. Write k = (k_r, k_perp). The quadratic in omega has real roots for all k iff two conditions hold:
  - the k_r^2 coefficient (1 - X)(1 + 2 f^2/r^2) >= 0;
  - the k_perp^2 coefficient (1 + S)(1 + f_r^2 + (f/r)^2 - f_t^2) >= 0.
- So the pinch needs **f_t^2 < 1 + f_r^2 + (f/r)^2** in addition to X < 1.
- In (f, p) this is exactly **1 - X > (f/r)^2**. It can fail for finite p.
- For the bump, the two transverse polarisations travel on the light cone, so X < 1 suffices.

### Formulation chosen for R, and what the X < 1 guard means in it

R uses the first-order form in (f, p).
- 1 - X = (1 + S)/(1 + p^2) > 0 for every finite state, because S >= 0 and p is real. **X < 1 cannot be violated by finite data during a run.**
- The condition acts only once, at the start: converting f_t to p needs f_t^2 < 1 + S. A start outside this domain trips the guard at t = 0.

What can fail during a run, and is watched:
- p growing without bound (1 - X -> 0 at a point). This is the analogue of the Eggers-Hoppe finite-time singularity of the massless equation.
- Loss of resolution: a steepening front.
- For the pinch, angular hyperbolicity: 1 - X > (f/r)^2.
- Non-finite values.

The kernel tests 1 - X = (1 + S)/(1 + p^2) > 0 and never X < 1. At |p| ~ 1e8, X itself rounds to 1.0 in double precision (test_check6 measures min(1 - X) = 8.5e-18).

## 2. Discretisation (derived; the checks are in section 4)

- **O** (`solver_o.py`) uses the kernel of `swirl/family/radial.py`:
  - u = r f on nodes r_i = i h;
  - the 4th-order 5-point D2;
  - odd (bump) or even (pinch) reflection at r = 0, and a reflecting wall at R;
  - the term -s2 u/r^2;
  - RK4 and the quadratic sponge sigma_max ((r - R_s)/(R - R_s))^2 on v.
  - The discrete energy E_h = 4 pi h sum [v^2/2 - u D2u/2 + (s2/2) u^2/r^2 + r^2 U(u/r)] is exactly conserved by the semi-discrete system without the sponge.
  - Core coordinate: f(0) for the bump; f_r(0) = (16 u_1 - u_2)/(12 h^2) for the pinch.
- **R** (`solver_r.py`) is a semi-discrete Hamiltonian:
  - cell centres r_i = (i + 1/2) h, with weights w_i = r_i^2 h (midpoint rule; spectrally accurate for the even integrand);
  - D is the 4th-order central first derivative, with ghost cells from the parity of f and an odd wall at R;
  - H_h = 4 pi sum w_i [sqrt((1 + S_i)(1 + p_i^2)) - 1 + f_i^2/2];
  - p_t = -(1/w) D^T(w q Df) - s2 q f/r^2 - f - sigma p, and f_t = p/q;
  - H_h is exactly conserved without the sponge, and the sponge can only remove energy;
  - kg=True swaps in the Klein-Gordon density: the twin, on the same grid and operators.
  - A known property: the centred D annihilates the grid mode, which then sits at omega = 1 with no group velocity. The resolution guard watches for energy reaching it.
- **R guard**:
  - Per RK4 stage at every node:
    - finite values and 1 - X > 0;
    - q < q_max = 1e4 (calibrated);
    - for the pinch, 1 - X - (f/r)^2 > 0.
  - Per record (0.5 time units): the high-k share of the (f, p) spectral power, over r < R_sponge with a cos^2 taper, at most 1e-6 (calibrated). Resolved production starts measure 1e-12 or less. The taper keeps the wall out: an untapered measure saw the wall's cut of a wide profile tail.
  - A trip raises GuardTrip with time, radius, X, q, the pinch margin, max|p|, max|f_r|, the energy so far and the last state (saved as .npz when a checkpoint directory is given). Nothing is clipped.
  - Every R run also logs, over every stage, min(1 - X), max|p| and the pinch margin, with their radii: block extremes every 5 time units and the global extremes with time and radius (`guard_extremes` in the run JSON).
- **Start profiles** (`profiles.py`):
  - Ansatz f = F(r) cos(omega t). The cycle-averaged full Lagrangian uses the midpoint rule in theta with K = 96 nodes (K = 192 changes E by 6e-14).
  - The Euler-Lagrange equation is (r^2 A F')' = r^2 (Ubar - omega^2 F B) + s2 A F, with A = <2 cos^2 P'(X)>, B = <2 sin^2 P'(X)> and Ubar = <U'(F cos) cos>.
  - Solved by bisection shooting from below (the smallest-amplitude bracket), then the exact linear tail is attached: e^{-kappa r}/r for the bump, the k1 shape for the pinch.
  - Observables: N = dLbar/domega, E = omega N - Lbar, energy per cycle 2 pi E/omega.
  - Solver-precision checks: Derrick 3 Lbar + int T A = 0, and dE/domega = omega dN/domega.
  - O bump starts in the driver use `swirl/family/profile.py` (the existing family tool). The generic solver reproduces it to 2e-13 in E and N.

## 3. Prediction rows O8 and O9 (Block 3): derived independently

- **O9.** The algebra agrees:
  - H = sqrt((1 + p^2)(1 + S)) - 1 + f^2/2;
  - X < 1 identically for finite p;
  - the quartic gradient energy is -S^2/8, the opposite sign to the retired term.

  **The implication does not hold**, that only resolution can fail. Two things can fail at finite p:
  - p -> infinity at a point. Measured below: a large R bump drives 1 - X to 1e-8 at the centre.
  - The pinch's angular hyperbolicity, 1 - X > (f/r)^2. Measured below: it trips at t = 11.69, converged in h.
- **O8.** The identity dE = omega dN along a family holds. Measured: dE/domega = omega dN/domega to 3e-11 to 1.3e-9 (test_profiles).

  The identity alone does not imply the conclusion. For drift along a family, d ln(E per cycle)/d ln E = 1 - d ln omega/d ln E. F4 needs this ratio to be at most 0.2, so it needs d ln omega/d ln E >= 0.8.

  Measured on my family tables, d ln omega/d ln E is:
  - O bump: -0.085 at omega 0.92 and -0.041 at 0.85, so the ratio is 1.04 to 1.09;
  - O pinch: -0.10 to -0.14 at omega 0.90 to 0.92;
  - R families: below 0.011 everywhere, so the ratio is 0.99 to 1.00.

  The exception is close to the energy minimum, where dE/domega -> 0 and the ratio diverges. So I **agree with O8's conclusion** (F4 misses for a lump that drifts along its family), for this reason rather than the one stated.

  The first runs agree, though they are not families. Over t = 60 to 1000:
  - R bump: Delta ln E = -2.03 and Delta ln(E per cycle) = -2.03;
  - R pinch: -5.57 and -5.57;
  - the O runs change by less than 1% (-0.003), so F4 is "not testable" there.

## 4. The six checks (measured; tests in `tests/radial5/`)

| # | Check | Target | Measured | Verdict |
|---|---|---|---|---|
| 1a | O bump step for step against `swirl/family/radial.py`, 8000 steps, sponge on | round-off (1e-12) | 0.0 (bit-identical) | hit |
| 1b | Start: omega 0.909149280617113, Phi(0), hw | Session 1: 0.9091, 6.89, 3.344038184 | 6.890549, hw 3.344038184 | hit |
| 1c | E at t = 0: discrete E_h against the analytic start energy | 1e-6 | 2843.828051 against 2843.828062, 3.8e-9 | hit |
| 1d | E at t = 0 against Session 1's E0 2843.7719 | 1e-4 | 2.0e-5. Session 1's diagnostic was np.gradient (O(dr^2), F-003) | hit, with a note |
| 1e | Rate, median over t = 975 to 1025 | 0.9096747 (Session 1 radial), 1e-4 | 0.9096711, 4.0e-6 | hit |
| 1f | frac(1000) | 0.994146, 5e-4 | 0.994127, -1.9e-5 | hit |
| 2 | Linear box modes, R = 40, n = 2 and 5, omega against sqrt(k^2 + 1); pinch mode j1 | 1e-6 | O bump 4.0e-10; O pinch 4.6e-10; R bump 1.3e-8; R pinch 4.35e-7 (worst) | hit (R pinch within target, 1000 times the O value) |
| 2' | R Klein-Gordon twin against O linear, Gaussian, core to T = 50 | 1e-5 | 3.4e-6 | hit |
| 3a | Closed box, T = 100, dt 0.0125, production starts | 1e-8 | O bump 8.4e-10, O pinch 5.1e-10, R bump 4.3e-10, R pinch 4.5e-10 | hit |
| 3b | Scaling per dt halving (0.025 / 0.0125 / 0.00625) | shown | ratios 29 to 35 in all four, the dt^5 law (Session 1 found 28) | shown |
| 3c | dE_h/dt along the kernel's right-hand side by complex step, random smooth data | round-off | 3.8e-17 to 6.6e-15; with one term dropped from E, 7.3e-3 to 0.75 | hit |
| 4 | h and dt / 1.5, T = 300: E0 / rate / frac(100) / frac(300) | 1e-6 / 1e-4 / 1e-4 / 1e-4 | worst: 1.5e-8 / 9.0e-8 / 7.6e-8 / 5.0e-7 | hit |
| 5 | Sponge 100 to 200 against 200 to 400, T = 1000, frac, all four and their twins | 1e-4 | see section 4.1 | see 4.1 |
| 6 | R guard (see check 6 in section 1 and the tests) | trips with a diagnosis | f_t = 1.5 at S ~ 0 trips at t = 0. A pinch start with X = 0.886 but margin -0.105 trips (start check), and in the kernel at stage 1. f_t = 1 - 1e-10 trips on q = 7.1e4. A width-0.06 Gaussian trips on resolution at t = 0.5. Guard on and off give bit-identical runs. In 400 random states with |p| up to 1e8, 1 - X > 0 holds and X from Hamilton's f_t equals the identity to 2.5e-12. 140 of 200 random pinch states have margin <= 0 | hit |

### 4.1 Check 5, box doubling (measured)

Max |frac_L - frac_2L| over t <= 1000. Target 1e-4. The production starts are those of section 6.

| Combination | Run | Twin |
|---|---|---|
| O bump | 7.0e-7 | 1.09e-5 |
| O pinch | 1.0e-7 | 6.7e-6 |
| R bump | **1.03e-4** (t = 955) | **1.20e-4** (t = 1000) |
| R pinch | 9.6e-5 (t = 920) | 4.9e-5 |

- O, and R pinch: hit.
- **R bump: pass with a note** (1.2 times the target), and R pinch is close to the target.

Diagnostic run (R bump twin, boxes 1×, 2× and 3×):
- 1× against 2×: 1.20e-4;
- 2× against 3×: 2.7e-7;
- 1× against 3×: 1.20e-4, of which only 3.9e-6 is present by t = 500.

So the small box is the one that differs. Slow, long waves near omega = 1 return from the 100-to-200 sponge after t ~ 500. The R bump at omega 0.995 is wide (kappa 0.1) and puts its energy into exactly such waves. The difference is 4% of what is left (frac 0.0028), so it does not change the verdict "disperses". This is the same kind of effect as F-008 (3.3e-4).

Recommendation: wide R starts (kappa < ~0.15, or hw > ~4) run in the 200-to-400 box. This doubles the cost.

## 5. Families (measured; `outputs/radial5/family_*.json`)

Single-frequency families. The O couplings are the Session 1 values (beta 4/3).

**Expectation (small amplitude, written before solving).** F = (eps/sqrt(lam)) R_l(eps r), with eps^2 = 1 - omega^2.
- lam = 3 g4/4 = 0.0375 for O and 3/8 for R.
- R_0(0) = 4.3374 for the bump. For the pinch, R_1'(0) = 3.7782, with peak 2.7098.
- E ~ N ~ I_l/(2 lam eps), with I_0 = 18.90 and I_1 = 110.13. So the pinch's energy is 5.83 times the bump's at the same rate, at leading order.
- dE/domega > 0 near omega = 1.
- In R the ansatz needs omega F < 1, which leading order puts at omega ~ 0.990 for the bump and 0.974 for the pinch.

| Family | Rate range solved | Ends because | Energy minimum | E at the min | dE/domega, dN/domega | NLS ratios (F, hw, E) at the top |
|---|---|---|---|---|---|---|
| O bump | 0.8018 to 0.995 | shooting precision (thin wall; the analytic end is 0.760) | interior, omega 0.970 | 1657.4 (Session 1: 1657.1 at 0.9692) | both signs | 0.98, 1.05, 1.08 at 0.995 |
| O pinch | 0.8751 to 0.995 | first non-undershoot reaches the runaway value F_out (physical end or solver limit: open) | interior, omega 0.955 | 7928.0 | both signs | 0.998, 1.02, 1.05 |
| R bump | 0.99455 to 0.99995 | omega F(0) -> 1: every F(0) up to 1/omega undershoots | none (E falls to the end) | 161.4 at the end | > 0 throughout | 1.0005, 0.999, 0.998 |
| R pinch | 0.98651 to 0.99995 | the first non-undershoot reaches omega F = 1 | none | 524.0 at the end | > 0 throughout | 1.0001, 1.000, 0.999 |

- **R in this ansatz is only the unstable-sign branch next to omega = 1.** The ansatz cannot reach |f_t| = 1. The exact homogeneous relativistic oscillator reaches omega A = pi/2, so **the end of the R families is an end of the ansatz, not shown to be an end of the equation.**
- **Pinch against bump at the same rate.** The pinch is higher in both models:
  - O at omega 0.95: pinch 7972 against bump 1787, a factor of 4.5;
  - R at omega 0.9953: pinch 1403 against bump 200, a factor of 7.0. Leading order gives 5.83.
- Max X over a cycle along R: up to 0.776 (bump) and 0.670 (pinch) at the ends.

## 6. First evolutions to T = 1000 against the twin (measured; `outputs/radial5/runs/first_*.json`)

| Start | frac(100), (500), (1000) | Twin frac(100), (1000) | T_90, T_death | Rate (settled 50 to 150; last 100; spectral) | Holds against the twin? |
|---|---|---|---|---|---|
| O bump, omega 0.909149 | 0.99607, 0.99521, 0.99413 | 0.0198, 2.7e-5 | none, none | 0.90949; 0.90966; 0.90965 | yes |
| O pinch, omega 0.95 | 0.99969, 0.99860, 0.99737 | 0.0421, 7.5e-6 | none, none | 0.94667; 0.94749; 0.94827 | yes |
| R bump, omega 0.995 | 0.613, 0.021, 0.0028 | 0.225, 0.0015 | 0.5*, 137.5 | 1.0051; 0.9994; 1.0003 | no: disperses, slower than the twin |
| R pinch, omega 0.99 | 0.9985, 0.023, 0.00073 | 0.680, 1.3e-4 | 161, 232 | 1.0091; 0.9956; 0.9916 | no: holds about 160, then disperses |

\* The R bump at omega 0.995 is wide (kappa 0.0999). Only 0.849 of its energy lies inside 3 hw at t = 0, so the Session 2 T_90 (first t with frac < 0.90) is the first record. The definition is badly posed for wide starts; the cycle-averaged T_90 is 7.8. This is reported, not replaced.

**Exploration beyond the family** (`outputs/radial5/explore_guard.json`, `explore_guard.py`): R Gaussian starts, production grid.
- **Bumps:** every one with A >= 1 (sigma 1.5 or 3) trips the guard within a few time units:
  - max|p| grows at the centre, and 1 - X falls to 7.5e-7 (A = 1, sigma = 3);
  - in grid-convergence runs (h = 0.05, 0.025, 0.0125), the trip time for A = 1, sigma = 3 is 7.5, 7.5 and 7.34, with max|p| 1.3e3, 8.2e3 and 1.1e4 growing as h falls;
  - for A = 2, sigma = 3 the trip time is 4.0, 4.5 and 5.5, with max|p| 35, 61 and 326.
  - This is consistent with a finite-time singularity at the centre, 1 - X -> 0, as in the massless Born-Infeld result. Not shown; a finer study is a proposal.
- **Pinch, A = 2, sigma = 4:** trips the angular hyperbolicity condition at t = 11.6875, r = 3.78 to 3.79, identical at all three h.
- **Smaller pinches** (A = 1 and 2 at sigma 2; A = 1 at sigma 4) disperse by t = 60 to 85 with rate about 1.

## 7. Tools for the production runner

- `python -m swirl.radial5.run_evolve5 --model {O,R} --obj {bump,pinch} --start profile:<omega> | gauss:<A>,<sigma> --T <T>`
  - Writes `outputs/radial5/runs/<tag>.json`: run and twin, full parameters, frac series every 2.5 time units, a per-period table (rate, E_in, energy per cycle), far probe f(r = 50) every 0.5, the last 200 time units of the core, high-k shares, nonlinear share, and the R guard extremes.
  - Checkpoints every 500 time units under `checkpoints/radial5/`. Rerunning the same command resumes, bit-identical (tested).
  - The twin runs to min(T, 2000) by default (`--T-twin`).
- `python -m swirl.radial5.run_family5 [O_bump ...]`: family tables.
- `python -m swirl.radial5.cost5`: the cost table.
- `python -m swirl.radial5.explore_guard`: the exploration above.

**Cost** (measured, one core, production grid h 0.05, dt 0.0125, R_box 200): wall seconds per 10,000 time units.

| Combination | Run | Twin |
|---|---|---|
| O bump | 43 | 46 |
| O pinch | 51 | 51 |
| R bump | 143 | 114 |
| R pinch | 130 | 103 |

A 10^5 R run takes about 25 minutes. Two processes give about 50 R runs of 10^4 per hour.

## 8. Deviations

- **E_in quadrature.** E_in uses a 4th-order density (O) or the exact discrete density (R) instead of Session 2's np.gradient plus trapezoid, which has an O(dr^2) error of about 1e-4 (F-003). The region (3 hw), the normalisation (E_total at t = 0), T_90 and T_death are unchanged.
  - Effect on the Session 1 point: E0 changes by 2.0e-5 and frac(1000) by 1.9e-5.
- **frac_cyc and T_90_cyc** (Block 2.1, cycle-averaged) are stored next to the Session 2 instantaneous values.
- **The check 5 test asserts below 3e-4** (CLAUDE.md: within 3 times the target is a pass with a note). It prints the verdict against the 1e-4 target.
- **Trip times in the exploration** are those of the guard (0.5 to 7.5 for the bumps). T_end in those records is the last full record before the trip.

## 9. Open issues

1. The R families exist only for omega > 0.9945 (bump) and > 0.9865 (pinch), where the ansatz holds, and on the dE/domega > 0 sign only. Starts that are long-lived, if any exist, are outside the single-frequency tool. A two-harmonic balance, or relaxation in the PDE, is a proposal.
2. Large R bumps approach 1 - X = 0 at the centre in a few time units, and the large R pinch loses angular hyperbolicity at a converged time. This bears on F1: the guard is not idle.
3. The Session 2 T_90 is badly posed for wide starts whose initial frac is below 0.90 (R bump, omega >= 0.995).
4. The end of the O-pinch family at omega 0.875 may be a solver limit (the runaway value), not a physical end.
5. The O-bump family solver stops at 0.802, short of the thin-wall end 0.760 (shooting precision). Session 1's tables cover the rest.

## 10. Follow-up to the review (2026-11-10 evening)

The reviewer accepted the solver, the energies and the guard, and set six conditions for R. Old values are kept where something changed. The first-pass family tables are in `outputs/radial5/old/`.

### 10.1 Families through the folds (condition 1): done for O bump, O pinch and R bump; R pinch see 10.7

`profiles.solve` scans in omega and bisects on the amplitude from below. It therefore stops at the first fold in omega and reports that fold as the end.

`profiles.continue_family` replaces it.
- It follows the family as a curve in (ln eps, ln x), with x = F(0) (bump) or F'(0) (pinch).
- Predictor: the secant. Corrector: `solve_point`, which holds omega or x fixed (whichever the tangent favours) and searches outward from the prediction to the nearest undershoot/overshoot change, then bisects to round-off.
- Passing a fold therefore needs no special treatment.

| Family | Old (scan in omega) | New (continuation) | Why it ends (new) |
|---|---|---|---|
| O bump | 0.8018–0.995; E min 1657.41 at 0.970 | 0.8006–0.99995, 52 points. No fold in omega; fold in F(0) at omega 0.8759 (F0 max 7.131). E min 1658.96 at 0.9711 (sampled; Session 1: 1657.1 at 0.9692) | solver limit: shooting precision (thin wall). Analytic thin-wall end 0.760 |
| O pinch | 0.8751–0.995; "ends" at the runaway value | 0.7968–0.99995, 78 points. No fold in omega; fold in F'(0) at omega 0.8880 (x max 2.223). E min 7926.2 at 0.9558 | solver limit: shooting precision (thin wall). **The old end at 0.875 was an artefact** of scanning omega from below past the F'(0) fold |
| R bump | 0.99455–0.99995; E falls to the end, 161.4 | 0.994586–0.99995, 36 points. **Fold in omega** at 0.994586 (F0 0.871; the old bisected end 0.9945451 is the fold, the reviewer has 0.994519). The upper branch runs to F0 = 1.005 | **ansatz wall**: omega F0 = 0.99988 at the last point |

The R bump has an energy minimum just before the wall:
- E = 145.654 at F0 = 1.0016, omega 0.994939;
- E = 145.863 at the end;
- E = 164.97 at the omega fold;
- the reviewer has 145.6 at F0 = 1.00.

Signs of dE/domega and dN/domega along the curves (central differences along the curve). They agree in sign everywhere, consistent with dE = omega dN:
- **O bump:** (+, +) from 0.99995 to 0.9711, then (−, −) from 0.9652 to 0.8006.
- **O pinch:** (+, +) from 0.99995 to 0.9613, then (−, −) from 0.9558 to 0.7968.
- **R bump:**
  - (+, +) on the lower branch, 0.99995 to the fold;
  - **(−, −) on the upper branch**, 0.99471 to 0.99494;
  - (+, +) at the last point, past the energy minimum.

Solver precision along the new curves:
- median |dE/domega − omega dN/domega| / max: 2.1e-6 (O bump), 4.2e-7 (O pinch), 4.4e-7 (R bump); larger only at folds;
- max |Derrick|: 9.5e-12, 1.5e-11 and 8.6e-7.

Every family row now records which parameter was held fixed (`fixed`).

### 10.2 Past the ansatz wall (condition 2): not done, parked after two attempts

Method: Newton–Krylov shooting for symmetric periodic orbits of R in (f, p) in a closed box (`swirl/radial5/periodic.py`).
- Unknowns: f0 without the pinned centre cell, and the period T.
- Equations: f(T/4) = 0 at every cell.

Two attempts, both from the single-frequency seed at F0 = 0.9, seed residual 3.7e-2:
- **Attempt 1** (box 80, h 0.1): NoConvergence at the first point. The iterate left the profile shape.
- **Attempt 2** (box 39.68, sized so 3 omega sits mid-gap between box modes; deeper Krylov solve): the residual fell to 5.3e-3 and stalled for 40 Newton steps. A spike grew next to the pinned cell (f1 = 1.03 > c = 0.9).

Diagnosis:
- pinning one cell is a poor amplitude condition;
- the shooting Jacobian is ill-conditioned through near-resonant box modes;
- the first cells of the R operator are second order (10.6), right where the pin sits.

Proposals for the parking lot:
- harmonic balance with 3 to 5 odd harmonics, collocated in time;
- or Newton shooting with an integral amplitude condition and a spectral grid near r = 0.

No Floquet rates were obtained.

### 10.3 Starts, bisection driver, end classes (condition 3): done

- `--scale` on any profile start.
- `amp:<x>`: the profile with a given central amplitude, which reaches branches past a fold. Example: amp 0.95 gives omega 0.994593 on the upper R-bump branch.
- `gauss:A,sigma,v0`: f_t = v0 times the shape.
- `shell:A,r0,sigma,v_in`: a Gaussian shell made regular by its mirror image, moving inward at v_in.
- `evolve.classify` gives one of: held (rule 2.1), held but not by stiffening, alive at cap, disperses, singular, 3D ill-posed, unresolved, invalid start.
- `evolve.refine_trip` reruns a resolution or momentum trip at h/2 and h/4 and compares the first times at which the per-stage min(1 − X) crosses 1e-1 … 1e-8.
  - Agreement within 2% at the deepest common threshold → singular, with t*.
  - Otherwise → unresolved.
  - Measured: R bump Gaussian A 1, sigma 3 has t(1 − X < 1e-7) = 7.40 at h 0.025 and 7.29 at h 0.0125, so singular, centre collapse, t* = 7.29. The reviewer has about 7.25.
- `python -m swirl.radial5.boundary5` bisects the amplitude between not-tripping and tripping. Every step is written atomically, with class, T_death, trip time and crossing times.
- The R-bump Gaussian, sigma 3, in a T = 60 smoke run brackets the boundary between A = 0.75 (disperses) and A = 1.0 (trips).

### 10.4 Regime numbers (condition 4): done; one definition everywhere

**Definition:** nonlinear share = 1 − <E_quad>/<E>, cycle-averaged, with E_quad = ∫ 4π r² (f_t²/2 + S/2 + f²/2) in the Lagrangian variables (f, f_t). For R, f_t = p/q.
- Profiles: the cycle average of the ansatz.
- Runs: per measured period inside r_lump. This is the `nonlinear_share_cyc_rlump` column of the per-period table, plus the first-period and t = 50 to 150 medians in `regime`.

**Old:** −4.4e-5 for the R bump at omega 0.995. That was instantaneous, at t = 0 where p = 0, and measured with p.

**New**, first evolutions, first period / t 50 to 150, against the profile table:

| Start | New, first period | New, t 50 to 150 | Profile table |
|---|---|---|---|
| R bump | +0.029 | +0.006 | +0.036 |
| R pinch | +0.056 | +0.016 | +0.069 |
| O bump | −0.129 | −0.129 | about −0.134 |
| O pinch | −0.088 | −0.089 | −0.085 |

**X_max and q_max** are now per-stage extremes: block rows every 5 time units, plus the global value. For the first R runs: X_max 0.5045 and q_max 1.421 (bump); 0.357 and 1.247 (pinch). The 10-unit instantaneous samples are gone.

### 10.5 Twin and box defaults (condition 5): done

- The twin runs to the same T as the run.
- R defaults to the sponge 200 to 400 (`evolve.GRID_R`); O keeps 100 to 200.
- Check 5 now doubles the production box: R 400 against 800.
- The first R evolutions were rerun. Old to new:
  - R bump frac(1000): 0.002785 to 0.002818; twin frac(1000): 0.001543 to 0.001423;
  - R pinch frac(1000): 0.000728 to 0.000718.
  - T_90 and T_death are unchanged (R bump 0.5* and 137.5; R pinch 161 and 232).
  - End class: disperses, both.
  - O runs: unchanged numbers, end class "alive at cap" (1000 time units = 145 and 151 cycles, short of the 1000 cycles of rule 2.1).

**Cost**, wall seconds per 10,000 time units, one core. Measured while two other jobs ran, so these are upper bounds; the quiet-machine O values earlier were 43 to 51.

| Combination | Run | Twin |
|---|---|---|
| O bump | 28 | 28 |
| O pinch | 37 | 37 |
| R bump (400 box) | 239 | 181 |
| R pinch (400 box) | 241 | 193 |

### 10.6 Euler–Lagrange test and the order in the first cells (condition 6): done

`tests/radial5/test_el_kernel.py` sets f and f_t to analytic fields with max X = 0.75 (gamma up to 2.0). It compares the kernel's p_t with r⁻²(r² γ f_r)_r − s2 γ f/r² − f written out with exact derivatives.
- **Bulk** (r in [1, 8]), at h = 0.1 / 0.05 / 0.025:
  - R bump: 6.7e-6 / 4.2e-7 / 2.7e-8 (order 4.0);
  - R pinch: 4.5e-3 / 3.6e-4 / 2.4e-5 (order 3.65, 3.94);
  - O bump: order 4.0; O pinch: order 3.95 to 3.99.
- **Sensitivity:** dropping gamma from the pinch's s2 f/r² term (in kernel and energy alike) changes p_t by 7.0 times max|p_t|. Energy conservation cannot see this; this test does.

**True order in the first cells:**
- **R bump: second order.**
  - Linear Laplacian, cell 0 (r = h/2): 7.8e-2 / 2.0e-2 / 5.0e-3, against |∇²f| = 3.
  - Nonlinear test at h = 0.05: cell 0 error 5.9e-2 of max|p_t|; observed order 1.2 to 1.8, tending to 2.
  - Cause: dividing D(r² q D f) by w = r² h at r ~ h turns the O(h⁴) truncation into O(h²).
- **R pinch:** third order at cell 0, two to three in cells 1 to 3.
- **O (u = r f):** fourth to fifth order at the first nodes; no loss.

**Consequence:** the R bump's centre value has a local error of about 1% at h = 0.05. A centre collapse should be judged by the h-refined crossing times (10.3), never by one grid. A closure near r = 0 that keeps fourth order is a proposal.

### 10.7 R pinch through the fold

FILLED BELOW
