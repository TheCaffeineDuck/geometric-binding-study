# WP-C derivation: the tail-overlap interaction of two equal-frequency lumps

Written by the numerics worker for Session 2, WP-C item 1, before reading `PRIOR_ART.md` section 4.2. The comparison with 4.2 is in section 9 and was added after sections 1 to 8 were written. The numbers come from `outputs/session2/wpC/derive_pair_law.py`, which writes `pair_law_predictions.json`. No PDE was run for this document.

## 0. Setup and the numbers used

Field equation and energy (Set-B-like units):

    phi_tt - lap phi + V'(phi) = 0,   V = m^2 phi^2/2 - g4 phi^4/24 + g6 phi^6/720
    E = int [phi_t^2/2 + |grad phi|^2/2 + V]
    momentum density  p_i = -phi_t d_i phi   (it is also the energy flux)
    stress            T_ij = d_i phi d_j phi + delta_ij Lag,   Lag = phi_t^2/2 - |grad phi|^2/2 - V
    d_t p_i + d_j T_ij = -(phi_tt - lap phi + V') d_i phi = 0 on solutions

Lumps: phi_i = Phi(|x - x_i|) cos(omega t + theta_i). The profile solves the period-averaged (single-frequency) equation

    lap Phi - kappa^2 Phi = -S(Phi),   S = (g4/8) Phi^3 - (g6/192) Phi^5,   kappa^2 = m^2 - omega^2,

and far out it is Phi = A e^{-kappa r}/r. Lump 1 sits at x = -d/2 and lump 2 at +d/2, and dtheta = theta_2 - theta_1.

**Regime (standard 1).** omega/m = 0.909. Nonlinear share of the period-averaged profile energy (E = 2801.7): quartic <-g4 phi^4/24> = -19.5% of E, sextic <g6 phi^6/720> = +6.3% (checked from the profile table; time averages <cos^4> = 3/8, <cos^6> = 5/16). On the t = 0 slice of a theta = 0 start (E0 = 2843.8) the same terms are -51.3% and +19.7%; Session 1's 0.35 was (|E4| + |E6|) over the sum of absolute terms, a third definition.

Values from the profile actually used for initial data (`outputs/step2/profiles/profile_beta1.3333_w0.9091.json`, omega = 0.909149): A = 56.700, kappa = 0.41647, E = 2801.7 (the period-averaged energy), N = 2865.5, hw = 3.344, Phi(0) = 6.8905.

- **The brief's numbers belong to a different profile.** Its A = 55.7, E = 2767 and N = 2828 are the family row at omega = 0.910 (A_resc 30.494 / sqrt(0.3) = 55.67), not the profile used here. Using them would lower F by 3.5%.
- **"E" has two meanings, 1.5% apart.** The profile E is the period average. The energy of the t = 0 slice depends on the start phase, from 2843.8 at theta = 0 to 2605.2 at theta = 90 (section 7).

## 1. The force from the momentum flux through the mid-plane

The momentum of the half-space x < 0 changes only through the plane x = 0:

    F_1 = d/dt int_{x<0} p_x = - int_{x=0} T_xx dA = int_{x=0} [ -(d_x phi)^2/2 + |grad_perp phi|^2/2 - phi_t^2/2 + V ] dA

Two facts reduce this to the overlap of the tails.

- **The self terms average to zero.** For one ansatz lump, the residual of the field equation contains only 3 omega and 5 omega harmonics, because the profile equation is the projection onto cos(omega t). So <residual * d_i phi> = 0 and <d_j T_ij> = 0. A plane with the lump on one side therefore carries zero period-averaged self flux. The instantaneous self flux oscillates at 2 omega with an amplitude of the same order as the interaction, so every measurement must average over whole periods.
- **V is quadratic at the mid-plane at leading order.** The nonlinear cross terms there are O(Phi(d/2)^4).

Insert phi = f_1 c_1 + f_2 c_2, with c_i = cos(omega t + theta_i) and s_i = sin(omega t + theta_i). Use <c_1 c_2> = <s_1 s_2> = cos(dtheta)/2. On the plane, f_1 = f_2 = g(r) with r^2 = (d/2)^2 + rho^2. Also d_x f_1 = -d_x f_2 = g' (d/2)/r, and grad_perp f_1 = grad_perp f_2 = g' rho/r. The period-averaged cross term is then

    <F_1> = (cos dtheta / 2) int 2 pi rho drho [ g'^2 + kappa^2 g^2 ]          (the m^2 and omega^2 terms combine into kappa^2)
          = pi cos dtheta int_{d/2}^inf r dr [ g'^2 + kappa^2 g^2 ]

With g = A e^{-kappa r}/r, the integrand is A^2 e^{-2 kappa r} (2 kappa^2/r + 2 kappa/r^2 + 1/r^3), which is exactly -d/dr [ A^2 (kappa e^{-2 kappa r}/r + e^{-2 kappa r}/(2 r^2)) ]. So

    F(d, dtheta) = 2 pi A^2 cos(dtheta) e^{-kappa d} (kappa/d + 1/d^2)     (force on lump 1 toward lump 2; > 0 attracts)

and F = +dU/dd with

    U(d, dtheta) = -2 pi A^2 cos(dtheta) e^{-kappa d} / d.

Sign convention (corrected after review; the first version wrote "F = -dU/dd" here): F is defined positive for attraction, i.e. toward decreasing d. The generalized force along increasing separation is -dU/dd = -F, so F = +dU/dd. For dtheta = 0, U rises toward 0 as d grows, dU/dd > 0, and the lumps attract. The relative coordinate obeys d'' = -2F/E (section 4), which is the same statement.

**This confirms the brief's U, coefficient and sign.** In-phase lumps attract and anti-phase lumps repel.

There is a numerical check that does not use the asymptotic tail. The script computes the full period-averaged plane integral of -T_xx for the superposed profiles (`F_sup`), with the true Phi(r) and the full nonlinear V and with the self terms subtracted. The ratio F_sup/F_asym at dtheta = 0 is:

| d | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|
| F_sup / F_asym | 0.346 | 0.736 | 0.907 | 0.968 | 0.9888 | 0.9960 | 0.9986 | 0.9998 |

The ratio goes to 1 as the tails separate, so the coefficient 2 pi A^2 is confirmed numerically.

## 2. What is held fixed, and what U means when N is not fixed

U is the overlap term of the period-averaged free energy E - omega N, at fixed omega, fixed tails (A, kappa) and fixed phases. This can be checked directly. The cross term of <E - omega N> for the superposition is

    (cos dtheta/2) int [grad f_1 . grad f_2 + kappa^2 f_1 f_2] + <V_NL cross>.

Integrate by parts on each side of the mid-plane, moving the derivative onto the lump whose core is on that side. The bulk terms give int f_2 S_1, and the first-order nonlinear cross term -(cos dtheta/2) f_2 S_1 cancels them exactly. Only the surface term on the mid-plane survives:

    (E - omega N)_cross = (cos dtheta/2) int_{x=0} [f_2 d_x f_1 - f_1 d_x f_2] dA = -2 pi A^2 cos(dtheta) e^{-kappa d}/d = U.

Three consequences follow.

- **Fixed N.** First-order changes of E at fixed N equal first-order changes of E - omega N at fixed omega, since dE = omega dN along the family (Legendre). So U is also the interaction energy of two lumps that each keep their N. The force (section 1) is the same whichever is held fixed. It is a property of the instantaneous field.
- **The t = 0 superposition energy is not U.** The overlap also adds N: N_cross = omega cos(dtheta) int f_1 f_2 ≈ 2 pi A^2 omega cos(dtheta) e^{-kappa d}/kappa. The energy excess of the superposed data is U + omega N_cross, and omega N_cross / |U| = omega^2 d/kappa, which is 32 at d = 16. The excess is N that the lumps carry and keep, not binding. The t = 0 test (`pair_t0_against_quadrature`) measures the overlap energy of one t = 0 slice: E_cross = 203.1 at d = 10 and dtheta = 60, against U = -15.7. It agrees with the quadrature to 4e-10. It is a single slice, not a period average, and d = 10 is inside the core region, so it is not a quantitative test of U + omega N_cross. It shows the size of the effect, and it must not be read as U.
- **When N moves.** For dtheta other than 0 and 180, N flows between the lumps (section 5), so N_i are dynamical. In the averaged-Lagrangian description, N_i is the momentum conjugate to theta_i, and the pair is Hamiltonian in (d, P; theta_i, N_i) with H = sum_i E_i(N_i) + U(d, theta_2 - theta_1). U(d) then gives the force at the current amplitudes, and the amplitudes follow N_i.

## 3. Corrections

- **No series in 1/(kappa d).** For Yukawa tails the plane integral is exact. Written as F = 2 pi A^2 kappa cos(dtheta) e^{-kappa d}/d [1 + 1/(kappa d)], the bracket is exact. The 1/(kappa d) term is 15% at d = 16, 12% at d = 20 and 10% at d = 24. Taking F ≈ kappa |U| would drop it.
- **Tail shape.** Phi/(A e^{-kappa r}/r) - 1 stays below 1% for r > 6.0, 3% for r > 5.0 and 10% for r > 3.9. The tail is A e^{-kappa r}/r up to O(e^{-3 kappa r}) nonlinear corrections. There are no power-law corrections, because the omega component obeys the linear Klein-Gordon equation outside the core.
- **Nonlinearity and second order in the overlap.** At the mid-plane the nonlinear source is (g4/8) Phi(d/2)^2/kappa^2 of the linear operator: 0.0138 at r = 8 (d = 16), 0.0017 at r = 10, 2e-4 at r = 12. F_sup/F_asym - 1 (table above) estimates this correction: -1.1% at d = 16, -0.14% at d = 20, -0.02% at d = 24. It is repulsive at every phase. The estimate is incomplete at O(e^{-2 kappa d}), because each core also polarizes in the other's tail.
- **Next-order force at dtheta = 90.** The leading force vanishes there. The superposition integral leaves a small repulsion: -9.6e-4 at d = 16 (0.12% of F(0)), -1.8e-5 at d = 20, -4e-7 at d = 24.
- **Radiation.** The third harmonic radiates (3 omega > m). A coherent cross-radiation force scales as B^2/d with radiated amplitude B ≈ 0.015, about 1e-4, without exponential decay. That is negligible against F(24) = 0.018, but it would dominate beyond d ≈ 35. The initial shedding of the single-frequency start (0.03% of E) crosses the mid-plane in the first ~50 time units. It contaminates the early half-space momentum slope by up to about 10% at d = 16 (section 8, check 7).
- **The lumps are not at the profile omega.** The start phase sets the settled omega (section 7): 0.9095 at theta = 0 and 0.9123 at theta = +-67.5. A and kappa both follow omega and their effects on F largely cancel. Corrected after review (the first version said "up to 18%", counting only e^{-kappa d}). For omega = 0.9122 against the profile's 0.909149, with A(omega) from a cubic spline of the family table: A^2 falls 12.2%, e^{-kappa d} rises 11.4%, 14.4% and 17.5% at d = 16, 20, 24, and (kappa/d + 1/d^2) falls 1.4%. Net change of F: -3.5% (d = 16), -1.0% (d = 20), +1.7% (d = 24). For the theta = 90 start's omega = 0.9161 (not used in the pairs): -7.2%, -1.3%, +5.0%. The reviewer's estimate of a net 3% at d = 24 (A^2 -10.5%, exponential +15%) is the same size; my numbers use the spline, linear interpolation of the 0.005-spaced table gives the smaller A^2 drop. Predictions per run still use the measured omega.

## 4. From force to centroid acceleration

The inertia of a lump is its energy. For an exactly periodic solution, a boost gives P = gamma E v. The general statement used for measurement is exact: d/dt int x e d^3x = int p d^3x, so the energy centroid moves at P/E whatever the field. The engine test checks this to 1.6e-3. Hence

    a = F/E,   E = 2801.7  (the 1.5% ambiguity in E is below every other uncertainty here)

The lumps accelerate toward each other (dtheta < 90) or apart (dtheta > 90). The separation obeys d'' = -2F(d, dtheta)/E. Under constant force each lump shifts by a T^2/2. The point model integrates d'' = -2F/E and stops when d reaches 2 hw.

## 5. dtheta = 90 degrees: N flow and phase drift

With phi = Re[Psi e^{i omega t}] and a slowly varying complex amplitude Psi = R e^{i chi}, the period-averaged Lagrangian is

    <Lag> = [ R_t^2 + R^2 (omega + chi_t)^2 - |grad R|^2 - R^2 |grad chi|^2 - m^2 R^2 ]/4 - <V_NL>(R).

It is invariant under chi -> chi + const. The Noether current gives

    n = R^2 (omega + chi_t)/2,   j = -R^2 grad chi/2 = -Im(Psi* grad Psi)/2,   d_t n + div j = 0,

and int n = N = omega int Phi^2/2 for one lump, which matches the family's N. For the pair, Psi = f_1 e^{i theta_1} + f_2 e^{i theta_2}, and the flux through the mid-plane into lump 2's half is

    dN_2/dt = -(1/2) sin(dtheta) int_{x=0} [f_1 d_x f_2 - f_2 d_x f_1] dA = -2 pi A^2 sin(dtheta) e^{-kappa d}/d.

This equals -dU/dtheta_2, as Hamilton's equation with N_2 conjugate to theta_2 requires.

- **Direction.** At dtheta = +90 (lump 2 leads in phase), N flows from the leading lump to the lagging one at the full rate C_N = 2 pi A^2 e^{-kappa d}/d. The plan's prediction ("leading force vanishes, N flows from one lump to the other") is confirmed. The direction is from leader to lagger.
- **Phase drift.** Each lump's phase advances at its own frequency, d theta_i/dt = omega_i(N_i) - omega. On the stable branch d omega/dN = -2.228e-5 (dN/d omega = -44,878 from the beta = 4/3 family at omega = 0.909149). With N_1 = N0 + n and N_2 = N0 - n:

      n' = C_N sin(dtheta),   dtheta' = 2 |d omega/dN| n   =>   dtheta'' = K sin(dtheta),   K = 2 C_N |d omega/dN|.

  This is a pendulum with its stable point at dtheta = 180 and its unstable point at dtheta = 0. A pair started at 90 does not stay at 90. N flows, the lagging lump gains N and slows further, dtheta grows toward 180 on the time scale 1/sqrt(K), and the pair then oscillates about anti-phase between 90 and 270. N swings by up to sqrt(C_N/|d omega/dN|). Because cos(dtheta) < 0 once dtheta > 90, **the 90-degree pair develops a repulsion** as its phase runs away. Starts at 45 and 135 also drift toward 180, so a 45-degree pair that first attracts turns repulsive.
- **Validity.** The model is linear in n, so it needs |n| << N. The N swing is 269 (9% of N) at d = 16 and 42 at d = 24. It is not valid for d <= 14, where the swing exceeds 15% of N. The table's d <= 14 rows of this model are shown only to mark that limit. It also uses the single-frequency family's d omega/dN, which may differ from the true breather's by a few percent.

| d | C_N | dN_2/dt with true Phi | K | 1/sqrt K | period of small oscillation about 180 | max N swing from 90 | 90-degree start at T = 100: dtheta, N_1 - N0, shift per lump (negative = apart) | T = 300 | T = 500 |
|---|---|---|---|---|---|---|---|---|---|
| 14 (invalid) | 4.237 | -4.207 | 1.89e-4 | 72.8 | 457 | 436 | 141, 359, -0.50 | — | — |
| 16 | 1.612 | -1.608 | 7.18e-5 | 118.0 | 741 | 269 | 110, 157, -0.081 | 224, 234, -2.83 | 342, 229, -6.92 |
| 18 | 0.6229 | -0.6223 | 2.78e-5 | 189.8 | 1193 | 167 | 98, 62, -0.012 | 155, 145, -0.78 | 231, 145, -3.03 |
| 20 | 0.2437 | -0.2437 | 1.09e-5 | 303.4 | 1906 | 105 | 93, 24.4, -0.0018 | 118, 69.6, -0.143 | 160, 92.1, -0.886 |
| 24 | 0.03839 | -0.03839 | 1.71e-6 | 764.5 | 4804 | 42 | 90, 3.84, -4.5e-5 | 94, 11.5, -0.0036 | 102, 19.0, -0.028 |

dtheta is the unwrapped phase in degrees. The fact that the superposition (true Phi) and the asymptotic C_N agree to 0.2% at d >= 16 confirms the coefficient of the N current.

## 6. Predicted table, dtheta = 0 (force, acceleration, centroid shift per lump)

F_asym is from section 1, a = F_asym/E. "Constant force" is a T^2/2. The point model integrates d'' = -2F/E from rest and stops at d = 2 hw = 6.7, where it says "merged by t".

| d | d/hw | Phi(d/2)/Phi0 | U(d, 0) | F_asym | F_sup/F_asym | F_sup at 90 | F_sup at 180 | a | const. force T = 100 | 300 | 500 | point model T = 100 / 300 / 500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 2.99 | 0.199 | -31.38 | 16.21 | 0.736 | -0.468 | -14.67 | 5.78e-3 | 28.9 | 260 | 723 | merged by 21 |
| 12 | 3.59 | 0.112 | -11.37 | 5.682 | 0.907 | -0.058 | -5.499 | 2.03e-3 | 10.1 | 91.3 | 254 | merged by 40 |
| 14 | 4.19 | 0.0635 | -4.237 | 2.067 | 0.968 | -7.3e-3 | -2.044 | 7.38e-4 | 3.69 | 33.2 | 92.2 | merged by 73 |
| 16 | 4.78 | 0.0367 | -1.612 | 0.772 | 0.9888 | -9.6e-4 | -0.769 | 2.76e-4 | 1.38 | 12.4 | 34.4 | 1.89 / merged by 126 |
| 18 | 5.38 | 0.0215 | -0.6229 | 0.294 | 0.9960 | -1.3e-4 | -0.2936 | 1.05e-4 | 0.525 | 4.72 | 13.1 | 0.576 / merged by 210 |
| 20 | 5.98 | 0.0128 | -0.2437 | 0.1137 | 0.9986 | -1.8e-5 | -0.1136 | 4.06e-5 | 0.203 | 1.83 | 5.07 | 0.210 / 2.92 / merged by 346 |
| 24 | 7.18 | 0.0046 | -0.03839 | 0.01759 | 0.9998 | -3.9e-7 | -0.01759 | 6.28e-6 | 0.0314 | 0.283 | 0.785 | 0.0315 / 0.296 / 0.903 |

At other phases the leading force scales with cos(dtheta). The constant-force shift scales the same way. The point-model shifts for the proposed separations, with the phase held fixed and with the N exchange of section 5 (shift per lump; negative = apart):

| d | dtheta | F_asym | point model, fixed phase, T = 100 / 300 / 500 | with N exchange |
|---|---|---|---|---|
| 16 | 0 | 0.772 | 1.89 / merged by 126 | same |
| 16 | 45 | 0.546 | 1.19 / merged by 149 | 1.10 / -8.3 / -25.9 |
| 16 | 90 | 0 | 0 | -0.081 / -2.83 / -6.92 |
| 16 | 135 | -0.546 | -0.85 / -4.60 / -8.63 | -0.88 / -4.93 / -9.31 |
| 16 | 180 | -0.772 | -1.15 / -5.74 / -10.5 | same |
| 20 | 0 | 0.1137 | 0.210 / 2.92 / merged by 346 | same |
| 20 | 45 | 0.0804 | 0.147 / 1.69 / merged by 411 | 0.146 / 1.50 / 1.48 |
| 20 | 90 | 0 | 0 | -0.0018 / -0.143 / -0.886 |
| 20 | 135 | -0.0804 | -0.140 / -1.09 / -2.49 | -0.141 / -1.13 / -2.64 |
| 20 | 180 | -0.1137 | -0.197 / -1.46 / -3.20 | same |
| 24 | 0 | 0.01759 | 0.0315 / 0.296 / 0.903 | same |
| 24 | 45 | 0.01244 | 0.0223 / 0.206 / 0.610 | 0.0222 / 0.204 / 0.590 |
| 24 | 90 | 0 | 0 | -4.5e-5 / -3.6e-3 / -0.028 |
| 24 | 135 | -0.01244 | -0.0221 / -0.194 / -0.513 | -0.0221 / -0.196 / -0.524 |
| 24 | 180 | -0.01759 | -0.0312 / -0.271 / -0.705 | same |

**Where the tail approximation stops being valid.** Within 1% of the full superposition integral needs d >= 16 (4.8 hw). Within 10% needs d >= 12 (3.6 hw). At the plan's point-model limit of 2.5 hw (d = 8.4), the asymptotic force overestimates the superposition force by a factor of about 2.5 (0.35 at d = 8). The mid-plane there is inside the core (Phi(4.2) = 0.33 Phi0, where the nonlinear source is about equal to the linear operator). I would put the point-model limit at about 3.6 hw for 10% accuracy. WP-D should flag balances below d ≈ 12, not 8.4. The superposition integral itself is incomplete at that range, because each core polarizes in the other's tail.

## 7. The start phase changes the lump (input from item 5)

The single-frequency ansatz is not the true quasi-breather, so the energy of its t = 0 slice depends on theta:

| theta (deg) | 0 | 22.5 | 45 | 67.5 | 90 |
|---|---|---|---|---|---|
| E0 slice | 2843.8 | 2861.3 | 2878.8 | 2742.1 | 2605.2 (= omega N) |

E0 is even in theta and in 180 - theta. In the radial runs, each start settles onto a different member of the family. The settled omega is 0.9095 at theta = 0, 0.9093 at 22.5, 0.9087 at 45, 0.9122 at 67.5 and 0.9161 at 90. This is why the theta = 90 phase control misses (`WPC_BUILD_NOTES.md`). For the pair it matters twice:

- Lumps started at different theta are unequal. N then flows because of the unequal start, not only because of the interaction.
- The settled omega sets kappa and A.

The pair design therefore assigns phases symmetrically, theta_{1,2} = -/+ dtheta/2. The two lumps are then time-reversed images with equal slice energy, and no theta = 90 start is used.

## 8. Summary for the prediction file (Block 2)

- **Force law.** F = 2 pi A^2 cos(dtheta) e^{-kappa d}(kappa/d + 1/d^2), with A = 56.70 and kappa = 0.41647 (or the values at the measured omega). Sign: attractive for dtheta < 90. Decay constant kappa, including the exact 1/d^2 term. cos(dtheta) dependence. Valid to 1% for d >= 16.
- **Centroid.** a = F/E, E = 2801.7. Predicted shifts per lump are in the table of section 6.
- **dtheta = 90.** Zero leading force. N flows from the leading to the lagging lump at 2 pi A^2 e^{-kappa d}/d (1.61 per unit time at d = 16, 0.244 at 20, 0.038 at 24). dtheta drifts toward 180 with K = 2 C_N |d omega/dN|, after which the pair repels.
- **Not predicted.** Merger dynamics below d ≈ 12. Whether a bounded state exists. The size of polarization (O(e^{-2 kappa d})) effects.

## 9. Comparison with PRIOR_ART.md section 4.2 (read after sections 1 to 8)

Section 4.2 uses the same method (mid-plane momentum flux, after Bowcock et al. 2008).

- **Agreement.**
  - 4.2's mid-plane integrand, -(1/2) cos(delta) f^2 [kappa^2 + (kappa + 1/r)^2], is mine: g'^2 = g^2 (kappa + 1/r)^2.
  - The sign (attraction in phase) agrees.
  - The coefficient C = 2 pi A^2, with A the coefficient of e^{-kappa r}/r, agrees.
- **One difference: 4.2 drops the 1/d^2 term in F.** It gives F "about 2 pi C^2 kappa cos(delta) e^{-kappa d}/d" for kappa d >> 1. The plane integral can be done exactly: F = 2 pi A^2 cos(delta) e^{-kappa d}(kappa/d + 1/d^2). This equals +dU/dd exactly (F positive for attraction; see section 1) for U = -2 pi A^2 cos(delta) e^{-kappa d}/d. So the U of 4.2 and of the brief is exact at leading order in the overlap, and only its stated F is approximate. The dropped 1/(kappa d) is 15% at d = 16, 12% at d = 20 and 10% at d = 24, which matters at the plan's force-law precision. (4.2's 2 pi C^2 reads as 2 pi C with C = A^2 in its own notation. Its last line, U about -2 pi C^2 ..., uses C = A, so the notation is mixed but the content is the same.)
- **What is held fixed.** 4.2 says an energy comparison at fixed N is not meaningful because the superposed ansatz changes N with separation, and recommends centroid acceleration. I agree on the measurement, and section 2 makes the point exact:
  - U is the overlap term of E - omega N at fixed omega, and equals the fixed-N interaction energy to first order.
  - The superposition's raw E excess is U + omega N_cross, about 30 times |U| at d = 16.
- **N exchange.** 4.2 records charge transfer at intermediate phases for complex Q-balls (Battye & Sutcliffe 2000), none at alpha = pi, and no source for real 3D oscillons. Section 5 gives, in the averaged-Lagrangian approximation:
  - the rate 2 pi A^2 sin(dtheta) e^{-kappa d}/d;
  - the direction, from the leading to the lagging lump;
  - the zeros at 0 and pi, consistent with Battye & Sutcliffe;
  - the phase pendulum that drives dtheta toward pi. The oscillation it predicts about pi is reminiscent of Axenides et al.'s breathing at the difference of internal frequencies. I have not checked that it is the same mechanism.
- **Nothing in 4.2 contradicts sections 1 to 8.**
