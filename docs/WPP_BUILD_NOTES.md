# WP-P build notes (Session 3, P1; the P2 driver built, not run)

Numerics worker, 2026-10-02. Code: `swirl/arrange/unequal.py` (the extended model), `swirl/arrange/orbit_table.py` (item 4), `swirl/arrange/p2scan.py` (the P2 driver), `swirl/arrange/beat_check.py` (a diagnostic). Tests: `tests/arrange/test_unequal.py` (13 tests). Outputs: `outputs/session3/wpP/`. No PDE was run. Beta = 4/3 only; no reason to add another beta turned up.

## 0. In brief

- **The model now takes lumps of any N on the beta = 4/3 stable branch** (omega 0.85 to 0.945, N 1842 to 10530, N/N_min 1.13 to 6.4), with the coupling, inertia, frequency and size of every lump following its own N, and **moving lumps** with relativistic inertia, time-dilated phase, and the Lorentz boost phase of each tail at the other lump. The boost phase is what carries the overlap momentum and the overlap angular momentum: the Session 2 model had neither in its dynamics.
- **It reduces to the Session 2 law exactly** (frozen family: 2e-13 on trajectories; full family at equal lumps: 3.9e-8, the fit error of the coupling table).
- **The unequal law** (section 2) is a sum of two exponentials: each lump's core sits in the other's tail, weighted by a coupling charge Q_i(kappa_j). Checked against a direct quadrature of the averaged Lagrangian to 6e-5.
- **Orbits cannot outrun the instabilities.** In every rigidly rotating balance in the valid range (3.6 to 8 hw), the fastest unstable mode e-folds in 0.1 to 10% of the orbital period; where a phase-dominated mode exists it e-folds in 4 to 14%. For a pair, (phase growth rate / orbital angular frequency)^2 = |domega/dN| E d/(kappa + 1/d), independent of the coupling strength (2.1 at 4.8 hw). Circular orbits under the tail law are radially unstable for kappa d > (1 + sqrt 5)/2 (d > 3.9 = 1.16 hw) and lie above the dissociation energy for kappa d > 1. These are model statements, not 3D results.
- **Rotation shifts phase steps.** A rigidly rotating ring presents every neighbour with the same boost phase, so its effective step is the lab step plus or minus W v d cos(pi/K) (5 to 60 degrees for in-phase rings at circular speed, up to 82 for stepped ones). Every rotating balance found has an effective step below 90 degrees (attraction is needed for the centripetal force) and has unstable modes; near 90 degrees the unstable modes are position-dominated rather than phase-dominated. So FND-01's exclusion carries over in the effective step, with the orbital instability added.
- **Unequal lumps beat.** Their relative phase runs at omega_1 - omega_2; the first-order force averages out and a small residual attraction remains (1e-4 to 0.25 of the in-phase force in the model, section 5.3). The analytic estimate of that residual misses by 1.3 to 5 times.
- **Regression against the Session 2 3D pairs**: all 15 outcome classes agree; both in-phase contact times within 1%; the 45-degree pair at d = 16, which the Session 2 model missed by 3.7 length units (F-013), is followed to 0.85. Two runs exceed my target by 1.3 and 1.8 times (section 4).
- **P2 driver**: 1822 runs plus one far-apart control each; smoke test of 24 runs: 1.7 s each, about 26 minutes on 2 workers for the whole scan.

## 1. Files

| file | what |
|---|---|
| `swirl/arrange/unequal.py` | family table builder (profiles, Q matrix), `Family` (Chebyshev fits in omega, complex-safe), Hamiltonian, complex-step gradient, `Model` (integration, momenta from velocities, conserved quantities), rotating-ring balance and rotating-frame linearization |
| `swirl/arrange/orbit_table.py` | item 4 table |
| `swirl/arrange/p2scan.py` | P2 driver: run list, setup, outcome classes, bonded rule and clause-5 variants |
| `swirl/arrange/beat_check.py` | beat-averaged force at held positions |
| `tests/arrange/test_unequal.py` | the checks of section 4 |
| `outputs/session3/wpP/family_coupling_beta1.3333.json` | 39 solved profiles, omega 0.850 to 0.945 step 0.0025, with Q_i(kappa_j) |
| `outputs/session3/wpP/checks/*.json` | measured check values |
| `outputs/session3/wpP/instability_vs_orbit.{json,md}` | item 4 |
| `outputs/session3/wpP/beat_average_check.json` | section 5.3 |
| `outputs/session3/wpP/p2_smoke/` | 24 smoke runs and `summary.json` |

## 2. Unequal lumps: the leading-order interaction

### 2.1 Derivation

Ansatz phi = Re sum_i F_i(x - X_i) e^{i theta_i(t)}, each F_i the stationary profile at its own frequency Omega_i (kappa_i^2 = m^2 - Omega_i^2), lab phases theta_i with theta_i' close to Omega_i. The averaged Lagrangian density is (|psi_t|^2 - |grad psi|^2 - m^2 |psi|^2)/4 - <V_NL>. Fast averaging (dropping terms at Omega_1 + Omega_2) needs only |Omega_1 - Omega_2| << Omega_1 + Omega_2, always true here. The cross term, first order in the overlap, at rest:

    L_12 = (cos D/2) int [theta_1' theta_2' F_1 F_2 - grad F_1.grad F_2 - m^2 F_1 F_2 + S_1 F_2 + S_2 F_1],   D = theta_1 - theta_2

with S_i = (g4/8) F_i^3 - (g6/192) F_i^5 the lump's own source ((lap - kappa_i^2) F_i = -S_i). Two exact steps:

- Integrating by parts over all space with each profile equation and averaging the two ways gives int grad F_1.grad F_2 = -(kappa_1^2 + kappa_2^2)/2 O + (1/2) int (S_1 F_2 + S_2 F_1), with O = int F_1 F_2.
- Green's identity with the two profile equations gives O = int (S_1 F_2 - S_2 F_1)/(kappa_1^2 - kappa_2^2).

On shell (theta_i' = Omega_i) the O terms combine to -(Omega_1 - Omega_2)^2 O/2, and with the identity:

    L_12 = cos D [Omega_1 int S_1 F_2 + Omega_2 int S_2 F_1] / (2 (Omega_1 + Omega_2))

The addition theorem for e^{-kappa r}/r (the spherical mean over a sphere of radius r < d is e^{-kappa d}/d sinh(kappa r)/(kappa r)) gives int S_1 F_2 = 4 pi Q_1(kappa_2) A_2 e^{-kappa_2 d}/d, with the **coupling charge**

    Q_i(k) = (1/4 pi) int S_i(r) sinh(k r)/(k r) d^3x,   Q_i(kappa_i) = A_i (the tail amplitude; checked to 4.9e-8 on all 39 profiles)

The pair Hamiltonian at first order is H_12 = -L_12 at theta' = Omega(N) (the first-order Legendre rule):

    U_12 = -(2 pi / d) [ w_1 Q_1(kappa_2) A_2 e^{-kappa_2 d} + w_2 Q_2(kappa_1) A_1 e^{-kappa_1 d} ] cos D,   w_i = Omega_i/(Omega_1 + Omega_2)

- **Force** on lump 1 toward lump 2: F = dU/dd. Each exponential carries its own (kappa_j/d + 1/d^2); the longer tail (the smaller lump, smaller kappa) dominates at large d.
- **N flow**: N_1' = -dU/dtheta_1, proportional to sin D. N still flows from the phase leader to the lagger.
- **Phase**: theta_i' = dH/dN_i = Omega_i(N_i)/gamma_i + dU/dN_i. The relative phase D runs at Omega_1 - Omega_2 (the beat) plus coupling terms. For |Omega_1 - Omega_2| well above the pendulum rate sqrt K, D rotates and the cos D force averages out at first order (section 5.3).
- **Equal limit**: Q = A and w = 1/2 give U = -2 pi A^2 cos D e^{-kappa d}/d, the WP-C law exactly.

### 2.2 What is held fixed, and where it is valid

- Each lump stays on the family at its own instantaneous N: A_i, kappa_i, E_i, hw_i and Q_i follow N_i (adiabatic profiles). Total N is conserved. Nothing else is held.
- E(N) is the exact integral of omega dN of the fit (dE/dN = omega to 9e-16), so phi' = Omega/gamma is consistent with the energy.
- **Valid**: first order in the overlap, so beyond about 3.6 hw (10%) and 4.8 hw (1%), as in WPC section 6; omega 0.85 to 0.945 (runs leaving it stop as "left the family"). The long-lived range at 4/3 is about omega 0.857 to 0.9075 (N/N_min 1.8 to 5.2, alive at 30,000 in WP-A); smaller lumps (N/N_min about 1.3, omega 0.931) die within 10^4.
- **Not included**: O(e^{-2 kappa d}) terms (core polarization, and the response of one core to the other's off-frequency tail, which is where any frequency pulling or locking beyond the pendulum would come from); radiation; lump lifetimes; three-body overlap. The superposition-corrected law (Model B) is not extended to unequal lumps.
- **Size of the unequal structure**: a single-exponential law with geometric-mean amplitude and mean kappa is off by 3 to 10% at N ratio 1.5 and 7 to 20% at ratio 2 (d = 16 to 24).

## 3. Moving lumps

- **Single lump**: H_i = sqrt(P_i^2 + E(N_i)^2). So X' = P/(gamma E), energy gamma E, phase rate dH/dN = Omega/gamma (exact to 4e-16 in the test). At the speeds that matter (circular speeds 0.003 to 0.16; 0.02 to 0.07 at 4.8 to 6 hw): gamma - 1 = 2e-4 to 5e-3. The time-dilation shift Omega (1 - 1/gamma) is 1.1e-3 per unit time at v = 0.05. It is common to equal lumps on a symmetric orbit; it is not negligible against the pendulum rate (0.0085 at 4.8 hw) when lumps move at different speeds, and it reduces the in-phase pair's phase growth by 9% at 3.6 hw through the coupling of N to the kinetic energy.
- **Boost phase**: a lump moving with V has the phase -Omega gamma V.(x - X) relative to its centre. Each core therefore sees the other's tail with a shifted phase, delta_j = Omega_j P_j.r/E_j. This term is first order in V and carries the momentum and angular momentum of the overlap:
  - two resting lumps at phase difference dtheta have canonical momentum -omega 2 pi A^2 e^{-kappa d} sin(dtheta), the FND-11 value (3.9e-8);
  - a resting stepped ring has canonical angular momentum equal to the WPD overlap formula J = omega sum I_ij (x_i x x_j) (3.9e-8; the asymptotic-tail value, 19 to 27% above the true-profile J at d = 12 to 16).
- **The second-order part of the boost phase** (a miss of my first form, investigated). With both shifts kept in full ("resummed": cos(D - delta_j) at core i), the model's in-phase contact time at d = 16 came out 7.4% early against 3D, against 1.0% for the Session 2 model.
  - Diagnostic: an axisymmetric quadrature of the averaged Lagrangian of two exactly boosted profiles.
  - A uniformly moving pair reproduces L0(gamma d)/gamma, the Lorentz value, to 2e-11.
  - For head-on motion (+v, -v) the superposition's reduction of the cross term is 0.69 of the resummed cos form at v = 0.02, 0.04 and 0.06.
  - **Adopted form (default)**: the co-moving part of the two shifts exact, the counter-moving part to first order only. It is exact for boosted pairs and exact at first order in V, and it leaves out the counter-moving O(v^2) term. The superposition says that term is 0.69 of the resummed one; the 3D contact times at d = 16 and 20 agree best with leaving it out (-0.8% and -0.5%, against -7.4% and -3.3% resummed). The resummed form is kept as `boost="resummed"`. Rigid rotation has no counter-moving part, so the item 4 table does not depend on this choice.
- **Lorentz contraction** of the tails is left out: kappa d (gamma - 1) = 5e-3 at v = 0.04 and d = 16.
- **Pending L2 comparison** (needs WP-L's boosted 3D lump). The model predicts, for a lone settled lump boosted to v: centroid speed v, energy gamma E0, phase rate at the moving centre Omega/gamma. At a fixed lab point the frequency is gamma Omega. For the profile lump (Omega 0.909149, E0 2801.68): v = 0.02 gives E = 2802.244 and rate 0.908967; v = 0.05 gives 2805.192 and 0.908012; v = 0.1 gives 2815.798 and 0.904592. The comparison to make: the tracked centroid speed against v; total energy (plus absorbed) against gamma E0 of the settled lump; the frequency of phi at the tracked centre against omega_settled/gamma; and the tail phase gradient -Omega gamma v along the motion, which is the model's boost phase. Report each as a relative difference; the model expects them to 1e-3 or better at v up to 0.1, limited by the settling of the 3D lump.

## 4. Validation (targets, measured worst over the record)

| check | target | measured | status |
|---|---|---|---|
| Q_i(kappa_i) = A_i, 39 profiles | 1e-6 | 4.9e-8 | pass |
| new profiles against the Session 1 family file (N, E, A; 20 shared omegas) | 1e-6 | 2.5e-9 | pass |
| fits against fresh profiles at 5 midpoints: N, E, A, q | 1e-7, 1e-7, 1e-6, 1e-6 | 1.4e-11, 1.4e-11, 1.9e-10, 3.2e-8 | pass |
| dE/dN = omega | 1e-12 | 8.9e-16 | pass |
| domega/dN at the profile lump against a fresh central difference | 1e-6 | 5.5e-8 (Session 2's spline value differs by 3.6e-5) | pass |
| **unequal law reduces to the equal law**: frozen family against model.py, 3 moving exchanging lumps, T = 300 | 1e-9 | 2.2e-13 (H 2.5e-13) | pass |
| full family at equal lumps: U, F, dN_2/dt against `pair_law_predictions.json`, d = 8 to 24 | 1e-6 | 3.9e-8 | pass |
| unequal law against direct quadrature (N ratio 1, 1.5, 2; d = 16, 20, 24) | 5e-3 | 6.3e-5 | pass |
| **conservation**, unequal moving 4-ring, T = 1500, rtol 1e-12, N exchange 10%: H (relative to initial interaction plus kinetic terms) | 1e-8 | 8.5e-13 | pass |
| same: sum N; P; L (absolute; scales 177, 5900) | 1e-12; 1e-9; 1e-7 | 7.1e-15; 1.7e-13; 8.4e-10 | pass |
| **circular orbit**, phases frozen, in phase; N ratio 1 and 2; d = 16, 20: period against the relativistic two-body value | 1e-6 | 1.4e-9; d drift over one period 3.1e-7 (the orbit is unstable, 14 to 16 e-folds per period) | pass |
| **time reversal**, unequal moving 4-ring, T = 300 there and back: X, P, phi, omega | 1e-6, 1e-6, 1e-8, 1e-11 | 2.5e-12, 5.5e-13, 3.1e-13, 3.8e-15 | pass |
| single moving lump: speed, gamma E0, Omega/gamma | 1e-12 | 6.7e-16 | pass |
| overlap momentum at rest against FND-11 (first order) | 1e-7 | 3.9e-8 | pass |
| overlap angular momentum of resting stepped rings against WPD's formula | 1e-6 | 3.9e-8 | pass |
| linearization against the analytic pair (frozen law): phase rate, radial rate, Omega_orb | 1e-6 | 3.8e-9 | pass |
| boost phase: quadrature Lorentz check; model unchanged under a boost | 1e-4; 1e-9 | 2.2e-11; 0 | pass |
| **regression, 15 Session 2 3D pairs**: outcome class | 15/15 | 15/15 | pass |
| same: in-phase contact time | 10% | 0.8% | pass |
| same: d(t) every 25 time units, max error over max(0.5, 10% of the 3D displacement) | 1.0 | **1.80** (d = 20, 45 degrees); 1.28 (d = 16, 90 degrees) | pass with a note (under 3 times); the Session 2 model gives 4.39 and 2.06 there |

The regression in more detail (max |d - d_3D| over the limit; Session 2 model, extended, extended without boost phase): d16_45: 4.39, 1.00, 8.7. d20_45: 2.06, 1.80, 2.04. d16_90: 0.69, 1.28, 0.64. Every other run is 0.4 or less for all three.
- The extended model follows the 45-degree pair at d = 16 (F-013, a 1.3 to 1.5 times force excess in 3D once the lumps became unequal) to within the limit: d(300) = 26.97 against 26.17 in 3D and 32.64 in the Session 2 model.
- Both new pieces are needed for that. The N-dependent coupling without the boost phase does worse (8.7), and the frozen law with the boost phase is the Session 2 model plus the overlap momentum.
- At 90 degrees, d = 16, the extended model separates more slowly than both 3D and the Session 2 model (0.64 against 0.35). Not diagnosed.

## 5. Item 4: phase instability against the orbit

`outputs/session3/wpP/instability_vs_orbit.md` has the full table (6 spacings, the pair and rings of 3 to 6, in phase and stepped m = 1 rotating with or against the phase wave). Equal lumps at the profile omega. Modes come from the rotating-frame linearization of the full model. "Phase" means position share below 0.5; "position" means 0.5 or above; T_orb is the circular-orbit period.

### 5.1 Pair, in phase

| spacing (hw) | d | v | T_orb | phase e-fold | phase e-fold / T_orb | radial e-fold / T_orb | analytic phase e-fold (static pendulum) |
|---|---|---|---|---|---|---|---|
| 3.6 | 12.04 | 0.109 | 347 | 47.5 | 0.137 | 0.089 | 44.9 |
| 4.2 | 14.04 | 0.071 | 621 | 74.8 | 0.121 | 0.080 | 73.6 |
| 4.8 | 16.05 | 0.046 | 1086 | 120.0 | 0.110 | 0.073 | 119.5 |
| 6.0 | 20.06 | 0.020 | 3172 | 308 | 0.097 | 0.063 | 307 |
| 7.2 | 24.08 | 0.0085 | 8857 | 778 | 0.088 | 0.056 | 778 |
| 8.0 | 26.75 | 0.0049 | 17257 | 1432 | 0.083 | 0.052 | 1432 |

Analytic, frozen law and nonrelativistic: (lambda_phase/Omega_orb)^2 = |domega/dN| E d/(kappa + 1/d), and lambda_radial = Omega_orb sqrt(kappa d + (kappa + 2/d)/(kappa + 1/d) - 3). The phase mode e-folds 7 to 12 times per orbit, and the radial mode 11 to 19 times.

### 5.2 Rings

- In phase, K = 3 to 6, 3.6 to 8 hw: fastest e-fold 0.03 to 0.10 T_orb; phase-dominated modes 0.04 to 0.10 T_orb where they separate from position modes.
- Rotation shifts the effective step by 5 to 60 degrees for in-phase rings at circular speed (0.6 to 82 degrees over all balances).
- Stepped m = 1, rotating with the phase wave (effective step grows):
  - K = 3, 4: no balance at any spacing (steps 120 and 90 become repulsive);
  - K = 5, 6: balance, no phase-dominated unstable mode; position-dominated modes at 0.026 to 0.041 T_orb.
- Stepped m = 1, rotating against the phase wave (effective step shrinks):
  - K = 4 balances only at very slow rotation, with the effective step just below 90 degrees (T_orb 2,400 to 330,000); position mode at 0.001 to 0.017 T_orb;
  - K = 5, 6: phase modes at 0.05 to 0.08 T_orb where present.
- **Every rotating balance in the valid range is unstable, with its fastest e-fold between 0.1% and 10% of an orbital period.**

### 5.3 Unequal pairs: beat against orbit

| N ratio | beat period | T_orb (frozen, in phase) at 4.8 / 6.0 / 7.2 hw | beat / T_orb at 4.8 / 6.0 / 7.2 hw |
|---|---|---|---|
| 1.1 | 1068 | 1085 / 3207 / 9056 | 0.98 / 0.33 / 0.12 |
| 1.5 | 276 | 1043 / 3149 / 9057 | 0.26 / 0.088 / 0.030 |
| 2.0 | 174 | 968 / 2949 / 8521 | 0.18 / 0.059 / 0.020 |

- **Beat-averaged force at held positions** (`beat_average_check.json`): <F>/F0 = 0.25 (ratio 1.1, d = 16, where the phase is not yet rotating: beat/sqrt K = 0.7), 0.004 to 0.1 at beat/sqrt K of 2 to 11, and -1e-4 at ratio 2, d = 24. It is attractive in 11 of 12 cases.
- **A miss of my own estimate.** I estimated the residual as K/(2 beat^2), the time a fast pendulum spends near in-phase. That overestimates the model's average by 1.3 to 5 times at beat/sqrt K above 1.5 (and gets the sign of the one negative case wrong). Candidate cause: the first-order frequency modulation from the N dependence of the coupling (dU/dN_i in phi_i'), which shifts the time-weighting at first order in the coupling. Not separated further.
- The residual is second order in the overlap, the order of the terms the model leaves out (WPC section 3: polarization about 1% of F at d = 16). The model cannot decide whether unequal lumps bind on beat-averaged forces.

## 6. P2 scan driver

- **Size**: 1822 runs, each with its far-apart static control (d = 45, 13.5 hw).
  - (i) Static unequal: 250 pairs (N ratio 1 to 2.95 in 10 steps, 5 phase differences, 3.6 to 7.2 hw); 312 rings (K = 3 to 6; one big lump, graded, alternating sizes; ratio 1.1, 1.5, 2, 2.95; in phase, stepped, alternating; 3.6, 4.8, 6 hw).
  - (ii) Orbits: 672 equal (K = 2 to 6; in phase, stepped with or against the phase wave, alternating; speed 0.6 to 1.4 of the circular speed; 3.6 to 8 hw); 252 unequal pairs (ratio 1.1, 1.5, 2; initial dtheta 0, 90); 336 alternating-size rings (K = 4, 6; ratio 1.5, 2; in phase, stepped).
- **Starts**:
  - Lumps at rest or at the set lab velocity. Canonical momenta are solved so that X' equals that velocity; resting lumps carry the overlap momentum.
  - Generic perturbation as in Session 2: sigma 1e-3 in positions and in phases, seed 20261002 + run number. Without it, symmetric orbits never leave their symmetric subspace.
  - The reference speed is the rigid-rotation speed of the pattern. Where the pattern has no balance, the in-phase one is used (labelled). For unequal pairs it is the frozen-phase in-phase two-body orbit.
- **Run length**: 5 in-phase contact times of equal lumps at the same K and d, and at least 3 reference orbital periods, capped at T = 10,000. The lone-lump T_90 of the profile lump is 10,630. Orbits at 7.2 and 8 hw get less than one period.
- **Outcome classes**: merged (hw_i + hw_j); flew apart (all nearest-neighbour distances above 1.5 times start, or all pairs beyond max(3 d0, d0 + 40)); left the family; bounded for the run. Each comes with the event time, the closest approach in hw, and the flag for closest approach below 3.6 hw.
- **The corrected bonded rule** (bonded.py unchanged) runs on every point-model series against its control, with the in-phase contact time of equal lumps as the unit. Two clause-5 variants (section 7) are recorded next to it.
- **Smoke test** (24 runs spread over the list):
  - outcomes: 12 flew apart, 8 merged, 2 bounded for the run, 2 left the family; 11 flagged below 3.6 hw;
  - bonded verdicts: all 24 "not bonded";
  - cost: 1.7 s per run with its control, 25 s wall on 2 workers; **the full scan is about 3,200 CPU seconds, about 26 minutes on 2 workers** (allow up to an hour for the spread of run lengths).
  - The scan resumes: existing result files are skipped.

## 7. Proposal: the bonded rule's phase clause for unequal and rotating groups (bonded.py not changed)

- **The failure.** Clause 5 holds each lab neighbour step to 0.25 degrees.
  - Unequal lumps: the step runs at the beat omega_i - omega_j (0.006 to 0.036 rad per unit time for N ratios 1.1 to 2), so clause 5 fails by construction within a few time units.
  - Moving lumps: the lab step also picks up the boost phase, so a uniformly moving group's lab steps differ from its rest-frame steps by Omega gamma v.r, and they change whenever the velocities change.
- **Proposal 1: hold the effective step.** For moving groups, apply clause 5 to the effective (boost-covariant) step, D_eff = theta_j - theta_i + (delta_i + delta_j)/2, with delta = Omega gamma V.r. This is the phase relation in the frame moving with the pair along its line of centres. Reasons: it is what the interaction depends on; it is unchanged by a boost of the whole group; for a rigidly rotating ring it is constant; and for a resting group it is the raw step. In 3D it needs the tracked centroid velocities, which the tracker gives. Recorded by the driver as `clause5_variant_effective_steps`.
- **Proposal 2: lock or beat.** For unequal frequencies, replace "steps held" by two numbers per neighbour edge: the mean drift rate of the effective step (a linear fit), and the largest excursion about that fit (`clause5_variant_phase_lock`).
  - **(a) Locked**: drift rate within its noise of zero, excursion within the tolerance. This is clause 5 (on effective steps) for a group whose frequencies have pulled together. It is the reading I recommend for "a bound group with a phase pattern".
  - **(b) Beating**: drift rate equal to the measured frequency difference of the two lumps and constant, excursion within tolerance. With clause 4 applied to beat-averaged N, this would admit a group held by beat-averaged forces. In this model such forces are second order in the overlap, below the model's accuracy, so (b) should not support a claim without 3D.
- The noise floors of the drift rate and the detrended excursion need measuring on the far-apart control, as E14 did for clause 5.

## 8. Misses, open issues

1. **Regression d(t) at d = 20, 45 degrees (1.80 times target) and d = 16, 90 degrees (1.28).** Passes with a note. At d20_45 every model variant misses by the same amount (Session 2 model 2.06), so it is not new. d16_90 is worse than the Session 2 model: the extended model separates more slowly from t = 75 on. Candidates, ranked:
   - the counter-moving O(v^2) boost term left out (section 3);
   - the true-profile overlap momentum, 19 to 27% below the asymptotic one the model uses;
   - the measured coupling 4.6% above 2 pi A^2.
   
   Not separated.
2. **The counter-moving O(v^2) boost-phase term**: the superposition gives 0.69 of the resummed form, 3D contact times prefer none. The default leaves it out. The pending L2 check and any 3D collision run (WP-L e) would decide it.
3. **The beat-average estimate** misses by 1.3 to 5 times (section 5.3).
4. **L2 comparison pending** (section 3).
5. The asymptotic tail is used for the overlap momentum and angular momentum. WPD found the true-profile J 19 to 27% smaller at d = 12 to 16, so the model's velocity–phase coupling may be that much too strong at 3.6 to 4.8 hw.
6. Two smoke runs left the family range (a lump's omega passed 0.945, N below 1842): rings at 4.2 to 6 hw with strong N flow. These are recorded as their own outcome. Extending the table toward N_min would not help, because such lumps die within 10^4 (WP-A).

## 9. Deviations and choices

- The fitting and state variable for each lump is omega, not N. N(omega) is smooth while omega(N) folds at N_min, and Chebyshev fits in N reached only 1e-5. (phi, N) stay the canonical pair, and the equations are written for omega through N'(omega).
- Gradients are taken by complex step of the coded Hamiltonian: no hand derivatives.
- The boost-phase form (section 3) is a judgment, documented with its alternative.
- The far-apart control is at d = 45, not Session 2's 9 hw. At 30, in-phase rings merged within the 10,000-unit runs.
- The interaction unit for unequal runs is the in-phase contact time of equal lumps at the same K and d.
- The regression target (d error within max(0.5, 10% of the 3D displacement), contact within 10%) is mine; the task gave none.
- Model and effort: Opus 5.5 (claude-opus-5-5) visible; the effort level is not visible to me as a named setting.

## 10. Against the literature (standard 9), not novelty claims (standard 10)

| statement | literature | source |
|---|---|---|
| circular orbits in a Yukawa potential are unstable for kappa r > (1 + sqrt 5)/2 | classical mechanics textbook result | not checked in `PRIOR_ART.md`; for the scholar to source |
| phase sector has negative effective mass; orbiting soliton pairs unstable | same | Malomed 1998 (PRIOR_ART S6.1, S6.5) |
| rotating stepped clusters held by attraction plus angular momentum | the model now carries the overlap J; every rotating balance found is unstable at valid spacings | Desyatnikov & Kivshar 2002 (PRIOR_ART S6.4, S6.5) |
| unequal-frequency lumps: relative phase beats, first-order force averages out | WP-S brief due this session | PRIOR_ART section pending (WP-S) |

No claim that anything here is new.
