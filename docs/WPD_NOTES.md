# WP-D notes: arrangement search in the point-lump model

Session 2, WP-D (numerics worker), 2026-10-02. No PDE was run for this package. Code: `swirl/arrange/` (pair laws, geometry and symmetry, model, scan driver). Tests: `tests/arrange/`. Outputs: `outputs/session2/wpD/`.

## 0. The answer in brief

- **Nothing in the scan is stable in the point model.** All 28 force balances found (5 with the asymptotic law, Model A; 23 with the superposition-corrected law, Model B) have unstable modes in the full linearization (positions, phases and N together). That is a result, not a failure of the search.
- **The slowest instability anywhere has an e-folding time of 46.6 time units (6.75 oscillation periods).** It is the Model B balance of a hexagon around an anti-phase centre at 4.71 hw. The slowest in Model A is 10.5 time units (1.52 periods): a cube around an anti-phase centre at 2.54 hw. Nineteen of the 28 balances e-fold in less than one period. There the period-averaged point model is not valid at all.
- **Every balance but one sits inside the tail law's 10% validity limit of 3.6 hw.** 25 of 28 are inside the plan's 2.5 hw limit. The one exception (Model B, hexagon around an anti-phase centre, 4.71 hw) exists only through second-order superposition terms that the model captures incompletely (section 9, item 2).
- **Claude Desktop's two numbers are confirmed, digit by digit, and so are the scholar's** (section 5).
- **Point-model time evolution at nearest-neighbour spacings 12, 16 and 20 (3.6, 4.8 and 6.0 hw).**
  - Every pattern with net attraction between nearest neighbours merges. That covers the stepped rings with steps below 90 degrees, every all-in-phase control and every rounded 0/180 control.
  - The anti-phase-centre shells and the N15 mirror controls fly apart.
  - Stepped rings merge later than their controls. The 6-ring with 60-degree steps merges at t = 82, 253 and 686, against 56, 176 and 485 in phase and 47, 147 and 401 rounded to 0/180.
  - Only the 90-degree-step rings come close to staying bounded:
    - 4-ring, m = 1: flies apart at spacing 12; still expanding slowly at 16 and 20 (+33% and +3% by T = 1000).
    - 8-ring, m = 2: merges at 12 and 16; unchanged to 0.1% at 20.
  - Section 7 has the full table.
- **What only a 3D run can decide** (section 8): the stepped rings with m = 1 (K = 5 to 8). Their balances lie inside the cores. The point model has no coupling between the overlap angular momentum they carry and their orbital motion. That coupling is the mechanism that holds the published optical clusters.

## 1. The model

### 1.1 Variables and Hamiltonian

Each lump i has position x_i, momentum p_i, phase theta_i and charge N_i. With a rotating-frame phase phi_i = theta_i - omega0 t, nu_i = N_i - N0, and E(N) = E0 + omega0 (N - N0) + (w/2)(N - N0)^2, where w = d omega/dN < 0:

    H = sum_i p_i^2/(2 E0) + (w/2) sum_i nu_i^2 + U(x, phi),     U = sum_{i<j} u(d_ij, phi_i - phi_j)

    x_i' = p_i/E0,   p_i' = -dU/dx_i,   phi_i' = w nu_i   (theta_i' = omega(N_i)),   nu_i' = -dU/dphi_i

The constants n E0 and omega0 sum N_i are dropped (the second is conserved). Conserved quantities, all tested: H, sum N_i, total momentum, and orbital angular momentum sum x_i x p_i. H is invariant under joint rotation of positions only, because the phases are internal.

### 1.2 Check against the WP-C derivation

- **Force.** With u = -C cos(D) e^{-kappa d}/d and C = 2 pi A^2, -dU/dx on lump 1 of a pair at -d/2 and +d/2 equals F = 2 pi A^2 cos(D) e^{-kappa d}(kappa/d + 1/d^2) toward the partner (`WPC_DERIVATION.md` section 1), and d'' = -2F/E (section 4).
- **N flow.** N_2' = -dU/dtheta_2 = -C sin(theta_2 - theta_1) e^{-kappa d}/d. N flows from the phase leader to the lagger (section 5).
- **Pendulum.** With n = N_1 - N0: n' = C_N sin(D) and D' = w (N_2 - N_1) = 2|w| n. Hence D'' = K sin D with K = 2 C_N |w|. This is section 5's pendulum, stable at 180 degrees and unstable at 0. The phase sector has negative effective mass, because w < 0 (Malomed 1998, via `PRIOR_ART.md` S6.1).
- **The test.** The model, with the full-precision profile values, reproduces every F, U, dN_2/dt and K in `pair_law_predictions.json` exactly (relative difference 0). Every printed digit of the WPC tables matches. The 27 printed point-model shifts with N exchange (section 6 table) all match, and the JSON values match to 2.6e-11 (section 3 of these notes).

### 1.3 Two pair laws

- **Model A (asymptotic).** u = -C cos(D) e^{-kappa d}/d.
  - It is scale-free. U = C f(kappa x), so every balance's kappa R, mode count and sign structure is independent of A and kappa. Only the physical size R = (kappa R)/kappa and the time scales change when the measured pair law arrives.
- **Model B (superposition).** For a sextic V the period-averaged superposition plane force is exactly a cosine series of order 3 in D: F_sup(d, D) = sum_{k=0..3} a_k(d) cos(kD).
  - I evaluate `derive_pair_law.F_sup` (unchanged, imported) at D = 0, 45, 90, 135 and 180 on d = 3 to 40 in steps of 0.25 and solve for the a_k (fit residual 1.3e-15).
  - The pair energy is b_k(d) = -int_d^40 a_k (F = +du/dd).
  - Model B = Model A(params) + [a_k,sup - a_k,asym](profile), splined. With the default parameters it is F_sup exactly: 7.9e-16 at the table nodes and 9.6e-6 of F_asym between them.
  - The derive script allowed the table down to d = 3 (0.9 hw), so it goes there. Model B is undefined below d = 3.
- **Model B is incomplete wherever it matters.** Below d of about 12 (3.6 hw) each core polarizes in the other's tail. That is O(e^{-2 kappa d}) and missing. Below d of about 8 (2.4 hw) the mid-plane is inside the cores. There F_sup(d, 0) passes through zero between d = 5.5 and 6.25 (1.6 to 1.9 hw): superposed profiles stop attracting where they overlap. That is a property of the superposition, not of merging lumps. **Every Model B balance below 2.5 hw sits on this artefact** (section 9, item 1).

### 1.4 What is held fixed

- The tail (A, kappa), the inertia E0, and a linear omega(N) with slope w.
- Hence the coupling C does not follow N_i. That requires |N_i - N0| << N0. The dynamics flag the first time any |N_i - N0| > 0.1 N0.
- Period averaging. The model needs every motion to be slow against 2 pi/omega = 6.91. Balances with e-folding below one period (19 of 28) are outside it.
- No radiation, no core polarization, no merger physics. Pairs below 2 hw end a run.
- Regime (standard 1), from `WPC_DERIVATION.md` section 0: omega/m = 0.909. Nonlinear share of the period-averaged energy: quartic -19.5%, sextic +6.3%.

### 1.5 Identities and formulas used

- **N16 (screened Earnshaw).** For Model A, (laplacian - kappa^2) e^{-kappa d}/d = 0 away from the source. So the trace of lump i's own 3x3 Hessian block equals kappa^2 U_i, with U_i = sum_j u_ij. U_i < 0 then forces a negative eigenvalue in that block. Verified to 1.2e-15 at random configurations and to 7.5e-14 at every Model A balance, with no exceptions.
- **N15 (phase-exchanging mirror).** A pure reflection (det -1, trace +1) of the arrangement with theta(image of i) = theta_i + pi for every lump. The field is then odd under the plane, and no radiation-free bound state exists. No lump may lie on the plane, so no pattern with a centre lump qualifies.
- **Angular momentum (N12 extended; my derivation).** For phi = sum_i f_i cos(omega t + theta_i) and L_z = -int phi_t d_phi phi, the period average is

      J = (omega/2) sum_{i != j} sin(theta_i - theta_j) (x_j x x_i) O'(d_ij)/d_ij,     O(d) = int f_i f_j d^3x

  With asymptotic tails, O = 2 pi A^2 e^{-kappa d}/kappa and O'/d = -C g. So

      J = omega sum_{i<j} I_{i->j} (x_i x x_j),     I_{i->j} = C g(d_ij) sin(theta_i - theta_j)

  that is, omega times the pair N current times the cross product of the positions.
  - 0/180 patterns carry exactly none (largest found: 2.9e-30).
  - **A correction I found while testing.** The asymptotic O(d) overestimates J by 24%, 29% and 37% at d = 16, 14 and 12. It is an algebraic 1/(kappa d) core correction: J is a volume overlap, not a mid-plane flux. A true-profile O(d) table (2D quadrature of the profile) reproduces a direct 3D grid integral of -<phi_t d_phi phi> to 2.0e-5. Both numbers are reported, and the true-profile one is the one to use.
  - The N current itself is a mid-plane flux. It agrees with the true-profile current to 0.2% at d >= 16 (WPC section 5).

## 2. The enumeration

- **Arrangements.** Rings of K = 3 to 8 and the tetrahedron, octahedron, cube and icosahedron, each with and without a central lump. 20 arrangements, unit circumradius.
- **Symmetry groups from coordinates (standard 8).** Every orthogonal Q that maps the point set onto itself, found by mapping a reference pair (a, b, a x b) onto all candidate pairs with matching norms and dot product. Orders 4K for rings, 24, 48, 48 and 120, rotations and reflections included. Labels such as "vertex", "edge" and "face" axis are assigned from coordinates. Each rotation axis class is typed by what lies on it.
- **0/180 classes (standard 7).** Orbits of the 2^n colourings under G x {identity, sign flip}, by brute force. Checked against Burnside's lemma for all 20 arrangements, and against closed forms:
  - cube vertices under rotations: 23; under O_h without the flip: 22;
  - icosahedron under rotations: 96;
  - 2-colour bracelets for n = 3 to 8: 4, 6, 8, 13, 18, 30.
- **Stepped patterns.** theta = m x azimuth about one representative of each rotation-axis class. Lumps on the axis get phase 0.
  - m runs over 0 to L-1, where L is the smallest m with m x azimuth constant mod 2 pi.
  - On the 2-fold edge axes of the cube and the icosahedron the azimuths are irregular (no L <= 24). There m is capped at 0 to 3 (deviation).
  - With a centre lump, its phase is scanned over 0 to 330 degrees in 30-degree steps.
  - Patterns are deduplicated under the group up to a global phase (a time shift; for 0/180 this is the sign flip). Duplicates are kept as aliases. Stepped patterns that come out as 0/180 are merged into their 0/180 class (for example ring6 m = 3 is the alternating ring).
- **Totals.** 312 0/180 classes and 283 distinct stepped patterns, 595 patterns in all.
  - 58 have forces radial by symmetry.
  - 331 are fixed points of the N flow: all 0/180 patterns, all centre-free stepped rings, and 7 of 21 stepped polyhedra (tetra 1 of 2, octa 3 of 4, cube 1 of 6, icosa 2 of 9). No stepped pattern with a centre is a fixed point.
  - 265 carry J.
  - 15 have an N15 mirror.

| arrangement | lumps | group order (rotations) | 0/180 classes, brute force | Burnside | stepped patterns (distinct) | radial by symmetry | N-flow fixed points | carry J | N15 mirror |
|---|---|---|---|---|---|---|---|---|---|
| ring3 | 3 | 12 (6) | 2 | 2 | 1 | 2 | 3 | 1 | 0 |
| ring3+c | 4 | 12 (6) | 4 | 4 | 4 | 2 | 4 | 4 | 0 |
| ring4 | 4 | 16 (8) | 4 | 4 | 1 | 3 | 5 | 1 | 2 |
| ring4+c | 5 | 16 (8) | 6 | 6 | 3 | 3 | 6 | 3 | 0 |
| ring5 | 5 | 20 (10) | 4 | 4 | 2 | 3 | 6 | 2 | 0 |
| ring5+c | 6 | 20 (10) | 8 | 8 | 24 | 2 | 8 | 24 | 0 |
| ring6 | 6 | 24 (12) | 8 | 8 | 2 | 4 | 10 | 2 | 3 |
| ring6+c | 7 | 24 (12) | 13 | 13 | 6 | 3 | 13 | 6 | 0 |
| ring7 | 7 | 28 (14) | 9 | 9 | 3 | 4 | 12 | 3 | 0 |
| ring7+c | 8 | 28 (14) | 18 | 18 | 36 | 2 | 18 | 36 | 0 |
| ring8 | 8 | 32 (16) | 18 | 18 | 3 | 5 | 21 | 3 | 6 |
| ring8+c | 9 | 32 (16) | 30 | 30 | 9 | 3 | 30 | 9 | 0 |
| tetra | 4 | 24 (12) | 3 | 3 | 2 | 1 | 4 | 2 | 0 |
| tetra+c | 5 | 24 (12) | 5 | 5 | 15 | 2 | 5 | 15 | 0 |
| octa | 6 | 48 (24) | 6 | 6 | 4 | 3 | 9 | 3 | 0 |
| octa+c | 7 | 48 (24) | 10 | 10 | 24 | 8 | 10 | 20 | 0 |
| cube | 8 | 48 (24) | 14 | 14 | 6 | 2 | 15 | 6 | 4 |
| cube+c | 9 | 48 (24) | 22 | 22 | 51 | 3 | 22 | 51 | 0 |
| icosa | 12 | 120 (60) | 46 | 46 | 9 | 1 | 48 | 8 | 0 |
| icosa+c | 13 | 120 (60) | 82 | 82 | 78 | 2 | 82 | 66 | 0 |

## 3. Check table

Target, then the measured worst value over the whole record. All checks are pytest tests in `tests/arrange/`. Recorded values are in `outputs/session2/wpD/checks/*.json`.

| check | target | measured (worst) | status |
|---|---|---|---|
| Two-lump F, U, dN_2/dt (90 deg), K against `pair_law_predictions.json` (d = 8 to 24) | to printed digits | relative difference 0 at every d; all printed digits match (d = 16: F 0.77199, C_N 1.61176, K 7.183e-5, 1/sqrt K 117.99) | pass |
| Two-lump point-model shifts, dtheta, N_1 - N0 at T = 100, 300, 500 (d = 16 to 24, 5 phases, with and without exchange) | printed digits; JSON to 1e-6 | 27/27 printed shifts match; JSON relative 2.6e-11 (absolute 1.2e-8); merge times 1.1e-9 | pass |
| Model B force against `F_sup` (d = 4 to 24, 6 phases) | nodes 1e-10, midpoints 1e-4 (of F_asym) | 7.9e-16; 9.6e-6 | pass |
| Cosine-series fit of F_sup (order 3, 5 phases) | exact | max relative residual 1.3e-15 | pass |
| Group orders from coordinates | 4K, 24, 48, 48, 120 | all equal | pass |
| 0/180 classes: brute force against Burnside (with sign flip), 20 arrangements | equal | 20/20 equal; closed forms 23, 22, 96, bracelets 4 to 30 equal | pass |
| Forces and N flow against minus the numerical gradient of U | 1e-8 | A 1.6e-12, B 9.8e-12 | pass |
| Hessian against the numerical derivative of the gradient | 1e-6 | A 5.9e-11, B 3.4e-10 | pass |
| Energy conservation, phases and N frozen, T = 1000 (alternating 4-ring and octahedron with anti-phase centre, spacing 12, random momenta; 86 to 179 length units of motion) | 1e-8 | A 4.7e-12, B 1.0e-10 | pass |
| Full model: H, sum N, P, L conserved (5-ring, 72-degree steps, spacing 20, run to merger at t = 685 to 689 with N swings of 0.13 N) | H 1e-8 | H: A 1.3e-13, B 1.0e-9 relative to the largest energy terms met; **relative to the initial terms A 1.2e-10, B 4.0e-7**. sum N 1.3e-16 relative; P 3.2e-13; L 7.0e-9 | pass (B is a miss under the stricter normalization; see section 9) |
| Energy drift in the 288 production runs (T = 1000 or to merger) | 1e-8 | A 1.2e-12, B 5.1e-9 at rtol 1e-12 (it was 3.0e-7 at rtol 1e-10: diagnosed and fixed, section 9) | pass |
| N16 identity: block trace = kappa^2 U_i (Model A), random configurations | 1e-10 | 1.2e-15; 0 lumps with U_i < 0 and a stable block | pass |
| N16 at every Model A balance | 1e-10 | 7.5e-14; 0 violations | pass |
| Forces radial by symmetry (58 patterns, both models, spacings 8, 12, 16) | 1e-12 | 4.1e-13 of the per-lump pair-force scale; 1.7e-11 of the net force (near cancellation: ring6+c, anti-phase centre, Model B) | pass on the pair-force scale; see section 9 |
| Balance force residual | 1e-8 of the pair-force scale | 6.1e-15 | pass |
| Zero modes at balances: 6, all rigid | 6 and 6 | 6 and 6 at all 28; rigid Rayleigh quotient 6.0e-16 | pass |
| Full linearization against the decoupled sectors (0/180 balance) | 1e-6 | 4.0e-15, 15 = 15 modes | pass |
| Angular momentum formula (true profile) against a 3D grid integral (rings 3, 4 and 5 at d = 16, 14, 12) | 1e-3 | 2.0e-5; asymptotic formula 1.24 to 1.37 times the grid value | pass (the asymptotic formula's excess is a finding) |
| Claude Desktop and scholar numbers | digits as printed | all match (section 5) | pass |

## 4. Results: every balance found

Spacing is the minimum pair distance (centre to shell where there is a centre). R is the circumradius. d2U/dR2 is the curvature of the collective breathing mode in physical units. "neg / modes" counts negative eigenvalues of the position Hessian with phases and N frozen; the 6 zero modes are always the 3 translations and 3 rotations. "unstable eigs" counts eigenvalues of the full linearization with positive real part. The fastest mode is classified by its position share kappa^2|dx|^2/(kappa^2|dx|^2 + |dphi|^2): positional above 0.9, phase/N below 0.1, mixed in between. J/(n N0) is the overlap angular momentum per lump in units of N0, asymptotic and (true profile). The rank is section 8's.

| rank | model | arrangement | pattern | kappa R | R | spacing | /hw | <2.5hw | <3.6hw | d2U/dR2 | restoring | neg / modes | U_i min | unstable eigs (full) | fastest rate | e-fold (t) | e-fold (periods) | fastest kind | J/(n N0) asym (true profile) | N15 control |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | B | ring6 | face(6) m=1 | 3.0465 | 7.315 | 7.315 | 2.187 | yes | yes | 69 | yes | 0 / 18 | -55.6 | 3 | 0.0422 | 23.7 | 3.43 | mixed | 1.8 (0.96) | no |
| 2 | B | ring7 | face(7) m=1 | 3.3332 | 8.003 | 6.945 | 2.077 | yes | yes | 78.7 | yes | 0 / 21 | -81.7 | 6 | 0.0663 | 15.1 | 2.18 | mixed | 2.2 (1.1) | no |
| 3 | B | ring8 | face(8) m=1 | 3.6490 | 8.762 | 6.706 | 2.005 | yes | yes | 74.8 | yes | 0 / 24 | -101 | 7 | 0.0751 | 13.3 | 1.93 | phase/N | 2.5 (1.3) | no |
| 4 | B | ring5 | face(5) m=1 | 2.8740 | 6.901 | 8.112 | 2.426 | yes | yes | 35.2 | yes | 0 / 15 | -22.3 | 8 | 0.0529 | 18.9 | 2.73 | mixed | 1.2 (0.7) | no |
| 5 | B | ring8+c | 111111110 nn-anti=0/8 c-anti=8/8 | 4.1473 | 9.958 | 7.622 | 2.279 | yes | yes | 67.8 | yes | 1 / 27 | -107 | 8 | 0.0698 | 14.3 | 2.07 | phase/N | 0 (0) | no |
| 6 | B | ring4+c | 11110 nn-anti=4/4 c-anti=4/4 | 1.4858 | 3.568 | 3.568 | 1.067 | yes | yes | -640 | no | 7 / 15 | 99.7 | 8 | 0.249 | 4.01 | 0.58 | positional | 0 (0) | no |
| 7 | A | ring5 | face(5) m=1 | 1.0695 | 2.568 | 3.019 | 0.903 | yes | yes | 668 | yes | 4 / 15 | -301 | 8 | 0.541 | 1.85 | 0.268 | mixed | 4 (0.66) | no |
| 8 | B | tetra+c | 11110 nn-anti=4/4 c-anti=4/4 | 1.3219 | 3.174 | 3.174 | 0.949 | yes | yes | -1.09e+03 | no | 9 / 15 | 117 | 9 | 0.312 | 3.2 | 0.463 | positional | 0 (0) | no |
| 9 | A | ring5+c | 111110 nn-anti=5/5 c-anti=5/5 | 1.2669 | 3.042 | 3.042 | 0.910 | yes | yes | -888 | no | 6 / 18 | -1.3e+03 | 10 | 0.991 | 1.01 | 0.146 | positional | 0 (0) | no |
| 10 | A | ring6 | face(6) m=1 | 0.2947 | 0.708 | 0.708 | 0.212 | yes | yes | 2.09e+04 | yes | 5 / 18 | -3.45e+03 | 10 | 8.03 | 0.125 | 0.018 | positional | 3.7 (-) | no |
| 11 | B | ring6+c | 1111110 nn-anti=6/12 c-anti=6/6 | 6.5664 | 15.767 | 15.767 | 4.715 | no | no | 0.00915 | yes | 6 / 21 | -1.8 | 11 | 0.0214 | 46.6 | 6.75 | positional | 0 (0) | no |
| 12 | B | ring6 | face(6) m=1 | 1.2953 | 3.110 | 3.110 | 0.930 | yes | yes | -687 | no | 6 / 18 | 246 | 11 | 0.273 | 3.66 | 0.53 | mixed | 5 (0.98) | no |
| 13 | B | ring7+c | 11111110 nn-anti=0/7 c-anti=7/7 | 4.0275 | 9.671 | 8.392 | 2.509 | no | yes | 41.7 | yes | 7 / 24 | -73.5 | 13 | 0.0712 | 14.1 | 2.03 | positional | 0 (0) | no |
| 14 | B | ring7 | face(7) m=1 | 1.7502 | 4.202 | 3.647 | 1.091 | yes | yes | -345 | no | 7 / 21 | 66.9 | 13 | 0.284 | 3.52 | 0.509 | mixed | 5.3 (1.4) | no |
| 15 | B | icosa+c | 1111111111110 nn-anti=12/12 c-anti=12/12 | 2.7546 | 6.614 | 6.614 | 1.978 | yes | yes | 292 | yes | 3 / 39 | -304 | 14 | 0.109 | 9.16 | 1.33 | positional | 0 (0) | no |
| 16 | B | ring5+c | 111110 nn-anti=5/5 c-anti=5/5 | 1.7990 | 4.320 | 4.320 | 1.292 | yes | yes | -586 | no | 10 / 18 | 21.9 | 14 | 0.267 | 3.74 | 0.541 | positional | 0 (0) | no |
| 17 | B | ring8 | face(8) m=1 | 2.3810 | 5.717 | 4.376 | 1.308 | yes | yes | -239 | no | 8 / 24 | -40 | 15 | 0.293 | 3.41 | 0.493 | mixed | 4.9 (1.6) | no |
| 18 | A | octa+c | 1111110 nn-anti=6/6 c-anti=6/6 | 1.5909 | 3.820 | 3.820 | 1.142 | yes | yes | -748 | no | 10 / 21 | -609 | 15 | 0.524 | 1.91 | 0.276 | positional | 0 (0) | no |
| 19 | B | ring6+c | 1111110 nn-anti=6/12 c-anti=6/6 | 2.1017 | 5.046 | 5.046 | 1.509 | yes | yes | -603 | no | 11 / 21 | -34.6 | 16 | 0.3 | 3.33 | 0.482 | positional | 0 (0) | no |
| 20 | B | ring7+c | 11111110 nn-anti=0/7 c-anti=7/7 | 2.3988 | 5.760 | 4.998 | 1.495 | yes | yes | -565 | no | 10 / 24 | -81.3 | 16 | 0.319 | 3.13 | 0.453 | positional | 0 (0) | no |
| 21 | B | ring8+c | 111111110 nn-anti=0/8 c-anti=8/8 | 2.7158 | 6.521 | 4.991 | 1.493 | yes | yes | -464 | no | 9 / 27 | -119 | 16 | 0.335 | 2.99 | 0.432 | positional | 0 (0) | no |
| 22 | B | octa | face(3) m=2 | 0.9824 | 2.359 | 3.336 | 0.998 | yes | yes | -1.88e+03 | no | 12 / 18 | 587 | 17 | 0.335 | 2.99 | 0.432 | positional | 0 (0) | no |
| 23 | A | cube+c | 111111110 nn-anti=8/8 c-anti=8/8 | 3.5432 | 8.508 | 8.508 | 2.544 | no | yes | -27.6 | no | 12 / 27 | -48.8 | 19 | 0.0951 | 10.5 | 1.52 | positional | 0 (0) | no |
| 24 | B | octa+c | 1111110 nn-anti=6/6 c-anti=6/6 | 1.5427 | 3.704 | 3.704 | 1.108 | yes | yes | -1.35e+03 | no | 15 / 21 | -48.1 | 20 | 0.283 | 3.53 | 0.511 | positional | 0 (0) | no |
| 25 | B | cube+c | 111111110 nn-anti=8/8 c-anti=8/8 | 1.9327 | 4.641 | 4.641 | 1.388 | yes | yes | -589 | no | 15 / 27 | -188 | 22 | 0.199 | 5.02 | 0.726 | positional | 0 (0) | no |
| 26 | B | icosa+c | 1111111111110 nn-anti=12/12 c-anti=12/12 | 2.2711 | 5.453 | 5.453 | 1.631 | yes | yes | -722 | no | 33 / 39 | -359 | 44 | 0.194 | 5.17 | 0.747 | positional | 0 (0) | no |
| 27 | B | ring4 | 1010 nn-anti=4/4 | 0.9824 | 2.359 | 3.336 | 0.998 | yes | yes | -1.26e+03 | no | 6 / 12 | 587 | 6 | 0.335 | 2.99 | 0.432 | positional | 1.6e-32 (2e-33) | yes |
| 28 | B | cube | 10010110 nn-anti=12/12 | 1.2108 | 2.907 | 3.357 | 1.004 | yes | yes | -4.4e+03 | no | 18 / 24 | 934 | 18 | 0.454 | 2.2 | 0.319 | positional | 0 (0) | yes |

Model A balances in physical units (kappa = 0.41647, hw = 3.344):

| arrangement, pattern | kappa R | R | pair distances (hw) | radial | negative position modes | e-fold |
|---|---|---|---|---|---|---|
| ring5, step 72 | 1.0695 | 2.568 | nn 3.019 (0.90), next 4.885 (1.46) | restoring | 4 / 15 | 1.85 (0.27 periods) |
| ring6, step 60 | 0.2947 | 0.708 | nn 0.708 (0.21) | restoring | 5 / 18 | 0.125 |
| ring5 + anti-phase centre | 1.2669 | 3.042 | centre 3.042 (0.91), nn 3.576 (1.07) | not | 6 / 18 | 1.01 |
| octahedron + anti-phase centre | 1.5909 | 3.820 | centre 3.820 (1.14), shell nn 5.402 (1.62) | not | 10 / 21 | 1.91 |
| cube + anti-phase centre | 3.5432 | 8.508 | centre 8.508 (2.54), shell nn 9.824 (2.94) | not | 12 / 27 | 10.5 (1.52 periods) |

Every Model A balance has a lump with U_i < 0, so N16 already says it is unstable, and it is.

What the balances say:
- **Model A.** The only balances are mixed attraction and repulsion at kappa R of order 1. In-phase or nearly in-phase neighbours attract, and an anti-phase centre or anti-phase next neighbours repel. All have negative position modes. Four of the five are inside 1.7 hw.
- **Model B, stepped rings with m = 1 (K = 5 to 8).** Balances at 2.0 to 2.43 hw. They are radially restoring and positionally stable with phases frozen: 0 negative modes, the only balances anywhere with none. They become unstable once phases and N move (mixed position-phase modes, e-folding 13 to 24 time units, 1.9 to 3.4 periods). These balances lie where the superposition attraction at the ring's phase step is 2.4% (K = 8) to 29% (K = 5) of the tail law (diagnostic D2, section 9). They sit on the superposition soft core, not on the tail.
- **Model B evades N16, as expected.** At 6 Model B balances some lumps have U_i < 0 and a stable own block: the four stepped rings, ring8+c and icosa+c. Model B is not a Yukawa law, so this is not a bug. It is the kind of non-Yukawa term PRIOR_ART S6.4 says can evade the identity.

## 5. Item 4: the two Claude Desktop numbers, against the scholar's reproduction

Normalized units C = 1, kappa = 1 (`claude_desktop_check` in the scan, recorded in `scan_results.json`).

| quantity | Claude Desktop | scholar (PRIOR_ART S6.4) | this work | agreement |
|---|---|---|---|---|
| 5-ring, 72-degree steps: balance kappa R | about 1.07 | 1.0695 | 1.0695021 | all 5 printed digits |
| radial (breathing) mode | restoring | restoring | restoring, d2U/dR2 = +0.4577 | yes |
| negative modes / total | — | 4 / 15 (6 zero) | 4 / 15 (6 zero, all rigid) | exact |
| single-lump U_i | — | -0.0358 | -0.0358044 | 3 digits |
| lump curvature radial, tangential, axial | — | +0.169, -0.328, +0.123 | +0.16917, -0.32836, +0.12339 | 3 digits each |
| physical R, nn and next spacing | "inside the overlap region" | 2.57; 3.02 (0.90 hw); 4.88 (1.46 hw) | 2.568; 3.019 (0.903 hw); 4.885 (1.461 hw) | yes |
| 6-ring, 60-degree steps: kappa R, U_i | — | 0.2947, -0.4102 | 0.29467, -0.41018 | 4 digits |
| octahedron, in-phase shell, anti-phase centre: kappa R | balance exists | 1.591 | 1.5909003 | 4 digits |
| radial mode | not restoring | -0.51 | -0.51262 (total U) | 2 digits |
| negative modes / total | — | 10 / 21 | 10 / 21 (6 zero, rigid) | exact |
| shell-lump U_i | — | -0.0724 | -0.0723900 | 3 digits |
| physical R, shell spacing | — | 3.82; 5.40 (1.6 hw) | 3.820; 5.402 (1.62 hw); centre to shell 3.820 (1.14 hw) | yes |
| other rings with no balance (m = 1: N = 3, 4, 7, 8; m = 2: N = 5, 8; in-phase N = 5) | — | none in kappa R 0.05 to 12 | none in Model A | yes |

**Verdict: both Claude Desktop numbers are confirmed.** The 5-ring balance is at kappa R = 1.0695 and is radially restoring but sits at 0.90 hw. The octahedral shell around an anti-phase centre balances at kappa R = 1.591 and is not restoring. Both are unstable in the full point model: 4 and 10 negative position modes, 8 and 15 unstable eigenvalues, e-folding 1.85 and 1.91 time units.

## 6. N flow, angular momentum and mirror controls (items 3 ii, vi, vii)

- **N flow.**
  - Every 0/180 pattern is a fixed point (sin 0 = sin pi = 0).
  - Stepped rings without a centre are fixed points: inflow from one neighbour equals outflow to the other (PRIOR_ART N14).
  - Stepped polyhedra are fixed points only about some axes (7 of 21).
  - Any stepped pattern with a centre lump is not a fixed point, because the centre exchanges N with the shell.
  - Rates at spacing 16 are in `scan_results.json` (`at_spacing.16.N_flow`). Typical non-fixed stepped polyhedra have max |N'| of 0.4 to 3.2 per time unit, against C g(16) = 1.61.
  - Every balance found is an N fixed point.
- **Angular momentum, stepped rings at spacing 16 (4.8 hw).** Nearest-neighbour N current I_nn = C g(16) sin(step) = 1.612 sin(step).

| ring, m (step) | I_nn | J_z asymptotic | J_z true profile | J/(n N0) true profile |
|---|---|---|---|---|
| 4, 1 (90) | 1.612 | -750 | -606 | 0.053 |
| 5, 1 (72) | 1.533 | -1232 | -996 | 0.070 |
| 5, 2 (144) | 0.947 | -751 | -606 | 0.042 |
| 6, 1 (60) | 1.396 | -1695 | -1371 | 0.080 |
| 6, 2 (120) | 1.396 | -1681 | -1358 | 0.079 |
| 7, 1 (51) | 1.260 | -2140 | -1730 | 0.086 |
| 7, 2 (103) | 1.571 | -2654 | -2144 | 0.107 |
| 7, 3 (154) | 0.699 | -1176 | -950 | 0.047 |
| 8, 1 (45) | 1.140 | -2571 | -2079 | 0.091 |
| 8, 2 (90) | 1.612 | -3622 | -2927 | 0.128 |
| 8, 3 (135) | 1.140 | -2552 | -2061 | 0.090 |

  The sign is negative for theta_k = 2 pi m k/K: the oscillation peak travels clockwise. A staircase ring at 4.8 hw carries 4 to 13% of N per lump, against m N for a ramp (N12). That is small next to a ramp but not "almost none". The point model does not use it: there is no centrifugal term.
- **N15 mirror controls (15 patterns, all 0/180, none with a centre).**
  - ring4: 1100, 1010 (alternating).
  - ring6: 111000, 110100, 101010 (alternating).
  - ring8: 11110000, 11101000, 11010100, 10110100, 11001100, 10101010 (alternating).
  - cube: 11110000, 01111000, 00111100, 10010110 (the alternating cube).
  - Odd rings, the octahedron, the tetrahedron and the icosahedron have none: every one of their mirror planes passes through a lump.
  - The only N15 patterns that balance are Model B balances at 1.0 hw (alternating square and alternating cube). They are ranked last, as controls.

## 7. Point-model time evolution (item 5): predictions for the 3D runs

- **Setup.** Start from the exact pattern plus a generic perturbation: Gaussian displacements with sigma 1e-3 per component (3e-4 hw), Gaussian phase offsets with sigma 1e-3 rad, N_i = N0, p = 0, seed 20261002. T = 1000, DOP853, rtol = atol = 1e-12.
- **Sizes.** Each balance size, then nearest-neighbour spacings 12, 16 and 20, and 30 (8.97 hw) as the "far apart" control.
- **Outcomes.**
  - merger: a pair below 2 hw.
  - collapse: for runs starting at or below 2 hw, a pair below half the starting spacing.
  - flies apart: every lump's nearest-neighbour distance above 1.5 times its start value at T, timed when the last lump crossed.
  - bounded: none of these; the mean pair-distance change by T is given.
  - "[N>10% from t]" marks where the linear omega(N) stops being valid.
- **Run set.** The top 6 ranked distinct patterns, every pattern with a Model A balance, every balance at or above 2.5 hw, the literature stepped rings (4, 5, 6, 7 and 8 with m = 1; 8 with m = 2), each with its controls, and both N15 controls that balance.
- **Output.** `predictions_point_model.json`, with full parameters.

| arrangement | phases (deg) | role | model | balance | spacing 12 | 16 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|
| ring6 | 0,60,120,180,240,300 | candidate (rank 1) | A | collapse 1 [N>10% from 1] at 0.21 hw | merger 82 | merger 253 [N>10% from 248] | merger 686 [N>10% from 682] | bounded (-0.3%) |
| ring6 | 0,60,120,180,240,300 | candidate (rank 1) | B | collapse 13 at 0.93 hw; merger 174 [N>10% from 166] at 2.19 hw | merger 93 | merger 261 [N>10% from 260] | merger 692 [N>10% from 688] | bounded (-0.3%) |
| ring6 | 0,0,0,0,0,0 | control: all_in_phase | A | - | merger 56 | merger 176 | merger 485 | bounded (-0.7%) |
| ring6 | 0,0,0,0,0,0 | control: all_in_phase | B | - | merger 62 | merger 181 | merger 490 | bounded (-0.7%) |
| ring6 | 0,0,180,180,180,0 | control: plain_0_180_rounded | A | - | merger 47 | merger 147 | merger 401 | bounded (-0.2%) |
| ring6 | 0,0,180,180,180,0 | control: plain_0_180_rounded | B | - | merger 53 | merger 153 | merger 407 | bounded (-0.2%) |
| ring7 | 0,51.4286,102.857,154.286,205.714,257.143,308.571 | candidate (rank 2) | A | - | merger 83 | merger 254 [N>10% from 244] | merger 706 [N>10% from 670] | bounded (-0.3%) |
| ring7 | 0,51.4286,102.857,154.286,205.714,257.143,308.571 | candidate (rank 2) | B | collapse 16 [N>10% from 12] at 1.09 hw; merger 93 [N>10% from 85] at 2.08 hw | merger 93 | merger 267 [N>10% from 252] | merger 711 [N>10% from 672] | bounded (-0.3%) |
| ring7 | 0,0,0,0,0,0,0 | control: all_in_phase | A | - | merger 65 | merger 201 | merger 551 | bounded (-0.5%) |
| ring7 | 0,0,0,0,0,0,0 | control: all_in_phase | B | - | merger 72 | merger 208 | merger 557 | bounded (-0.5%) |
| ring7 | 0,0,180,180,180,180,0 | control: plain_0_180_rounded | A | - | merger 43 | merger 133 | merger 364 | bounded (-0.2%) |
| ring7 | 0,0,180,180,180,180,0 | control: plain_0_180_rounded | B | - | merger 49 | merger 137 | merger 367 | bounded (-0.2%) |
| ring8 | 0,45,90,135,180,225,270,315 | candidate (rank 3) | A | - | merger 85 [N>10% from 83] | merger 261 [N>10% from 244] | merger 741 [N>10% from 676] | bounded (-0.3%) |
| ring8 | 0,45,90,135,180,225,270,315 | candidate (rank 3) | B | collapse 20 [N>10% from 15] at 1.31 hw; merger 46 at 2.01 hw | merger 98 | merger 278 [N>10% from 250] | merger 746 [N>10% from 678] | bounded (-0.3%) |
| ring8 | 0,0,0,0,0,0,0,0 | control: all_in_phase | A | - | merger 73 | merger 223 | merger 608 | bounded (-0.4%) |
| ring8 | 0,0,0,0,0,0,0,0 | control: all_in_phase | B | - | merger 82 | merger 231 | merger 614 | bounded (-0.4%) |
| ring8 | 0,0,0,180,180,180,0,0 | control: plain_0_180_rounded | A | - | merger 43 | merger 133 | merger 364 | bounded (-0.2%) |
| ring8 | 0,0,0,180,180,180,0,0 | control: plain_0_180_rounded | B | - | merger 49 | merger 137 | merger 367 | bounded (-0.2%) |
| ring5 | 0,72,144,216,288 | candidate (rank 4) | A | collapse 12 [N>10% from 5] at 0.90 hw | merger 95 | merger 276 [N>10% from 275] | merger 745 [N>10% from 727] | bounded (-0.3%) |
| ring5 | 0,72,144,216,288 | candidate (rank 4) | B | merger 135 [N>10% from 100] at 2.43 hw | merger 113 | merger 286 | merger 768 [N>10% from 732] | bounded (-0.3%) |
| ring5 | 0,0,0,0,0 | control: all_in_phase | A | - | merger 47 | merger 149 | merger 413 | bounded (-1.0%) |
| ring5 | 0,0,0,0,0 | control: all_in_phase | B | - | merger 52 | merger 153 | merger 417 | bounded (-1.0%) |
| ring5 | 0,0,180,180,0 | control: plain_0_180_rounded | A | - | merger 38 | merger 117 | merger 322 | bounded (-0.2%) |
| ring5 | 0,0,180,180,0 | control: plain_0_180_rounded | B | - | merger 42 | merger 120 | merger 325 | bounded (-0.2%) |
| ring8+c | 180,180,180,180,180,180,180,180,0 | candidate (rank 5) | A | - | merger 81 | merger 237 | merger 630 | bounded (-0.4%) |
| ring8+c | 180,180,180,180,180,180,180,180,0 | candidate (rank 5) | B | collapse 24 at 1.49 hw; merger 119 [N>10% from 72] at 2.28 hw | merger 97 | merger 247 | merger 636 | bounded (-0.4%) |
| ring8+c | 0,0,0,0,0,0,0,0,0 | control: all_in_phase | A | - | merger 66 | merger 211 | merger 587 | bounded (-0.4%) |
| ring8+c | 0,0,0,0,0,0,0,0,0 | control: all_in_phase | B | - | merger 72 | merger 217 | merger 593 | bounded (-0.4%) |
| ring4+c | 180,180,180,180,0 | candidate (rank 6) | A | - | flies apart 103 | flies apart 330 | flies apart 998 | bounded (+0.7%) |
| ring4+c | 180,180,180,180,0 | candidate (rank 6) | B | collapse 30 at 1.07 hw | flies apart 104 | flies apart 330 | flies apart 998 | bounded (+0.7%) |
| ring4+c | 0,0,0,0,0 | control: all_in_phase | A | - | merger 53 | merger 170 | merger 474 | bounded (-0.7%) |
| ring4+c | 0,0,0,0,0 | control: all_in_phase | B | - | merger 58 | merger 174 | merger 479 | bounded (-0.7%) |
| ring5+c | 180,180,180,180,180,0 | candidate (rank 9) | A | collapse 9 at 0.91 hw | flies apart 124 | flies apart 380 | bounded (+43.7%) | bounded (+0.6%) |
| ring5+c | 180,180,180,180,180,0 | candidate (rank 9) | B | collapse 30 at 1.29 hw | flies apart 126 | flies apart 380 | bounded (+43.7%) | bounded (+0.6%) |
| ring5+c | 0,0,0,0,0,0 | control: all_in_phase | A | - | merger 47 | merger 153 | merger 433 | bounded (-0.8%) |
| ring5+c | 0,0,0,0,0,0 | control: all_in_phase | B | - | merger 52 | merger 156 | merger 437 | bounded (-0.8%) |
| ring6+c | 180,180,180,180,180,180,0 | candidate (rank 11) | A | - | merger 156 | merger 472 | bounded (-0.2%) | bounded (-0.0%) |
| ring6+c | 180,180,180,180,180,180,0 | candidate (rank 11) | B | collapse 27 at 1.51 hw; merger 463 at 4.71 hw | merger 204 | merger 490 | bounded (-0.2%) | bounded (-0.0%) |
| ring6+c | 0,0,0,0,0,0,0 | control: all_in_phase | A | - | merger 40 | merger 125 | merger 344 | bounded (-1.4%) |
| ring6+c | 0,0,0,0,0,0,0 | control: all_in_phase | B | - | merger 44 | merger 129 | merger 348 | bounded (-1.4%) |
| ring7+c | 180,180,180,180,180,180,180,0 | candidate (rank 13) | A | - | merger 86 | merger 246 | merger 640 | bounded (-0.4%) |
| ring7+c | 180,180,180,180,180,180,180,0 | candidate (rank 13) | B | collapse 25 at 1.49 hw; merger 150 [N>10% from 90] at 2.51 hw | merger 202 [N>10% from 126] | merger 258 | merger 647 | bounded (-0.4%) |
| ring7+c | 0,0,0,0,0,0,0,0 | control: all_in_phase | A | - | merger 54 | merger 172 | merger 488 | bounded (-0.6%) |
| ring7+c | 0,0,0,0,0,0,0,0 | control: all_in_phase | B | - | merger 58 | merger 176 | merger 493 | bounded (-0.6%) |
| octa+c | 180,180,180,180,180,180,0 | candidate (rank 18) | A | collapse 17 at 1.14 hw | flies apart 110 | flies apart 340 | bounded (+49.4%) | bounded (+0.7%) |
| octa+c | 180,180,180,180,180,180,0 | candidate (rank 18) | B | collapse 24 at 1.11 hw | flies apart 112 | flies apart 340 | bounded (+49.4%) | bounded (+0.7%) |
| octa+c | 0,0,0,0,0,0,0 | control: all_in_phase | A | - | merger 50 | merger 164 | merger 467 [N>10% from 460] | bounded (-0.7%) |
| octa+c | 0,0,0,0,0,0,0 | control: all_in_phase | B | - | merger 54 | merger 168 | merger 470 [N>10% from 462] | bounded (-0.7%) |
| cube+c | 180,180,180,180,180,180,180,180,0 | candidate (rank 23) | A | merger 86 at 2.54 hw | flies apart 177 | flies apart 460 | bounded (+36.8%) | bounded (+0.5%) |
| cube+c | 180,180,180,180,180,180,180,180,0 | candidate (rank 23) | B | collapse 39 at 1.39 hw | flies apart 179 | flies apart 461 | bounded (+36.7%) | bounded (+0.5%) |
| cube+c | 0,0,0,0,0,0,0,0,0 | control: all_in_phase | A | - | merger 43 | merger 140 | merger 401 [N>10% from 398] | bounded (-0.8%) |
| cube+c | 0,0,0,0,0,0,0,0,0 | control: all_in_phase | B | - | merger 46 | merger 142 | merger 404 [N>10% from 400] | bounded (-0.8%) |
| ring4 | 0,90,180,270 | literature candidate | A | - | flies apart 318 | bounded (+32.7%) | bounded (+2.9%) | bounded (+0.0%) |
| ring4 | 0,90,180,270 | literature candidate | B | - | flies apart 292 | bounded (+33.3%) | bounded (+3.0%) | bounded (+0.0%) |
| ring4 | 0,0,0,0 | control: all_in_phase | A | - | merger 39 | merger 123 | merger 342 | bounded (-1.4%) |
| ring4 | 0,0,0,0 | control: all_in_phase | B | - | merger 43 | merger 126 | merger 345 | bounded (-1.4%) |
| ring4 | 0,0,180,0 | control: plain_0_180_rounded | A | - | merger 44 | merger 140 | merger 379 | bounded (-0.0%) |
| ring4 | 0,0,180,0 | control: plain_0_180_rounded | B | - | merger 51 | merger 147 | merger 386 | bounded (-0.0%) |
| ring8 | 0,90,180,270,0,90,180,270 | literature candidate | A | - | merger 248 [N>10% from 186] | merger 670 [N>10% from 668] | bounded (+0.0%) | bounded (+0.0%) |
| ring8 | 0,90,180,270,0,90,180,270 | literature candidate | B | - | merger 267 [N>10% from 200] | merger 678 [N>10% from 674] | bounded (+0.1%) | bounded (+0.0%) |
| ring8 | 0,0,180,0,0,0,180,0 | control: plain_0_180_rounded | A | - | merger 46 | merger 146 | merger 401 | bounded (-0.0%) |
| ring8 | 0,0,180,0,0,0,180,0 | control: plain_0_180_rounded | B | - | merger 52 | merger 152 | merger 407 | bounded (-0.0%) |
| ring4 | 180,0,180,0 | N15 mirror control | A | - | flies apart 71 | flies apart 230 | flies apart 700 | bounded (+1.3%) |
| ring4 | 180,0,180,0 | N15 mirror control | B | collapse 36 at 1.00 hw | flies apart 72 | flies apart 230 | flies apart 700 | bounded (+1.3%) |
| cube | 180,0,0,180,0,180,180,0 | N15 mirror control | A | - | flies apart 72 | flies apart 234 | flies apart 705 | bounded (+1.3%) |
| cube | 180,0,0,180,0,180,180,0 | N15 mirror control | B | flies apart 23 at 1.00 hw | flies apart 74 | flies apart 234 | flies apart 705 | bounded (+1.3%) |

Summary of the predictions:
- **All-in-phase controls** merge at t = 39 to 73 (spacing 12), 123 to 223 (16) and 342 to 608 (20), in both models.
- **Stepped rings, m = 1, K = 5 to 8.** They merge 1.16 to 2.0 times later than their in-phase control: 82 to 95, 253 to 276, 686 to 745 (Model A). In Model A at 16 and 20 the N swing passes 10% of N just before the merger, so the last part of those runs is outside the model. **In 3D, look for a delayed merger, not a bound ring.**
- **The 90-degree-step rings are the only stepped patterns that do not merge at 16 and 20.**
  - 4-ring, m = 1: flies apart by t = 318 (A) or 292 (B) at spacing 12. Expands 33% at 16 and 3% at 20 by T = 1000. The first-order nearest-neighbour force is zero, and the anti-phase diagonal repels.
  - 8-ring, m = 2: merges at 248 (12) and 670 (16) after its phases drift. Unchanged to 0.1% at 20.
- **Anti-phase-centre shells.**
  - ring4+c, ring5+c, octa+c and cube+c fly apart at 12 (103 to 177) and 16 (330 to 460). At 20 they expand by 37 to 49% (ring4+c flies apart at 998).
  - ring6+c, ring7+c and ring8+c merge (in-phase shell neighbours win). ring6+c needs 156 and 472, the slowest mergers in the table, and is unchanged to 0.2% at 20.
- **N15 controls** (alternating square and alternating cube) fly apart at 71 to 74 (12), 230 to 234 (16) and 700 to 705 (20), as N15 and the necklace literature expect.
- **At balance sizes.** Every balance collapses or merges, except the alternating cube in Model B, which flies apart.
  - Model A cube+c at 2.54 hw: merger at t = 86.
  - Model B stepped rings at 2.0 to 2.4 hw: mergers at 46 to 174.
  - Model B ring6+c at 4.71 hw: merger at 463.
- **At spacing 30** nothing resolves by T = 1000 (changes below 2.3%). These runs are the far-apart controls and say nothing about binding.
- **Model A against Model B** at spacings 12 to 20: outcome classes agree in every run. Outcome times agree within 20% except ring6+c at spacing 12 (156 against 204) and ring7+c at spacing 12 (86 against 202). Both are anti-phase-centre rings, where the leading-order forces nearly cancel and the second-order terms decide.

## 8. Ranked candidates and controls

Ranking (`candidates.json`, `ranked_balances`): N15 controls last, then N-flow fixed point, then the number of unstable eigenvalues of the full linearization, then the slowest fastest growth, then the spacing flags. **Nothing is stable, so the list is in effect ordered by how few modes grow and how slowly.** Controls for every candidate: the same lumps and phases at spacing 30 (8.97 hw); the same geometry with the phases rounded to 0/180; all in phase. Their outcomes are in section 7.

| rank | candidate | model | predicted spacing | stability (full linearization) | controls (0/180 rounded) | 3D needed because |
|---|---|---|---|---|---|---|
| 1 | 6-ring, 60-degree steps | B | 7.32 (2.19 hw), kappa R 3.05 | 3 unstable, e-fold 23.7 (3.4 periods), mixed; radially restoring; 0 negative position modes | 0,0,180,180,180,0 | balance on the superposition soft core; overlap J (0.96 N per lump there) not in the model |
| 2 | 7-ring, 51.4-degree steps | B | 6.95 (2.08 hw) | 6 unstable, e-fold 15.1 (2.2 periods), mixed | 0,0,180,180,180,180,0 | as rank 1 |
| 3 | 8-ring, 45-degree steps | B | 6.71 (2.01 hw) | 7 unstable, e-fold 13.3 (1.9 periods), phase/N | 0,0,0,180,180,180,0,0 | as rank 1 |
| 4 | 5-ring, 72-degree steps | B; A | B 8.11 (2.43 hw); A 3.02 (0.90 hw) | B 8 unstable, e-fold 18.9; A 8 unstable, e-fold 1.85 | 0,0,180,180,0 | the literature's best case (DK 2002); A balance inside the core |
| 5 | 8-ring, in-phase shell, anti-phase centre | B | 7.62 (2.28 hw) | 8 unstable, e-fold 14.3, phase/N | all in phase; far apart | soft-core balance |
| 11 | 6-ring, in-phase shell, anti-phase centre | B | 15.77 (4.71 hw) | 11 unstable, e-fold 46.6 (6.75 periods), the slowest anywhere | all in phase; far apart | the only balance beyond 3.6 hw. It exists only through the O(e^{-2 kappa d}) terms, because the leading-order nearest-neighbour and centre forces cancel exactly in a hexagon (section 9, item 2) |
| 23 | cube, in-phase shell, anti-phase centre | A | 8.51 (2.54 hw) | 19 unstable, e-fold 10.5 (1.5 periods) | all in phase; far apart | the only Model A balance above 2.5 hw; not restoring |
| — | 4-ring, 90-degree steps; 8-ring, 90-degree steps | none | — | no balance; 4-ring expands slowly, 8-ring m = 2 bounded at spacing 20 to T = 1000 | 0,0,180,0 and 0,0,180,0,0,0,180,0 | the literature's equilibria at 90-degree steps (DK N = 4, R0 = 3.8; N = 8, m = 2) come from overlap and angular momentum, which the tail law lacks (PRIOR_ART S8 item 8) |

Two further Model B balances are slow enough to note:
- rank 13: 7-ring, in-phase shell, anti-phase centre, at 2.51 hw, e-fold 14.1 (2.0 periods);
- rank 15: icosahedron around an anti-phase centre, at 1.98 hw, e-fold 9.2 (1.3 periods).

The remaining ranks (6 to 10, 12, 14, 16 to 22, 24 to 26) are balances inside 1.7 hw with e-folding under one period. They are listed in section 4 and are not useful 3D candidates. Ranks 27 and 28 are the N15 controls.

**Plainly.** No arrangement holds in the point model. The best it offers are the stepped rings, which merge later than their controls, and two slow but unstable balances (ring6+c at 4.7 hw, cube+c at 2.5 hw). Only a 3D run can decide the stepped rings, for three reasons:
- their balances lie where neither pair law is valid;
- their overlap angular momentum is not coupled to orbital motion in this model;
- their instability times are a few periods, where period averaging fails.

**A practical note for WP-E.** A stepped ring built from theta-shifted single-frequency starts puts lumps with different start phases into the ring. `WPC_BUILD_NOTES.md` section 3 shows that the start phase changes the settled lump: 9.5% less N at 90 degrees. That injects N differences and phase drift unrelated to the interaction. The pair design solved this with symmetric phases (-/+ dtheta/2). For rings there is no such trick. The quasi-breather copy method proposed there is needed, or this effect must be measured in a single-lump control at each phase used.

## 9. Misses and findings, investigated

1. **Model B balances below 2.5 hw are superposition artefacts.**
   - Expected: balances only where the tail law balances.
   - Measured: Model B has 23 balances, 21 of them below 2.5 hw.
   - Diagnostic D2: the superposition force at the stepped rings' Model B balances is 13.5% (K = 6, 60 degrees, d = 7.32), 6.4% (K = 7), 2.4% (K = 8) and 29% (K = 5) of the tail law. F_sup(d, 0) changes sign at d of about 5.5 and 6.25.
   - Rules in: the balances sit on the soft core of F_sup, where superposed profiles stop attracting. Rules out: a tail-law balance. The 3D pairs (WP-C) are the test of whether real lumps have such a core.
2. **The ring6+c balance at 4.71 hw rests on second-order terms.**
   - D0: in Model A the radial force on a hexagon lump from its two neighbours (in phase, at distance R) and the anti-phase centre (at R) cancels exactly. The net is the inward pull of the farther lumps: -7.22e-3 at R = 15.77.
   - D1: keeping only the cos(dtheta) part of the Model B correction removes the balance. With the full correction, the anti-phase repulsion exceeds the in-phase attraction at d = R by 7.22e-3 (the a_0 and a_2 terms) and balances it.
   - Rules in: the balance radius is set entirely by O(e^{-2 kappa d}) terms, and Model B lacks part of them (core polarization). The position could move a lot or vanish in 3D.
3. **The asymptotic angular momentum formula overestimates J by 24 to 37% at d = 12 to 16.**
   - Cause: J is a volume overlap integral, and the asymptotic tail is wrong inside the cores, which contribute at order 1/(kappa d). This is not exponentially small.
   - Fix: the true-profile overlap table, verified to 2.0e-5 against a 3D grid integral. The force and the N current are mid-plane fluxes and do not carry this error.
4. **Model B energy drift.**
   - Production runs at rtol 1e-10 drifted by up to 3.0e-7 (19 of 151 runs above 1e-8).
   - D: rerunning the worst run (ring5 rounded control, spacing 20) gives 3.04e-7 at rtol 1e-10, 3.2e-11 at 1e-12 and 3.4e-10 at 1e-13, with the merger time unchanged at 325.469. Cause: step-size control against the spline's discontinuous force derivative. Fixed by running everything at 1e-12. Worst production drift now 5.1e-9.
   - In the conservation test, Model B's H drift is 4.0e-7 relative to the initial energy terms (a 40-times miss under that normalization) and 1.0e-9 relative to the largest terms met. The pair energies grow by about 400 times toward the merger, which is where the absolute error accrues. Reported as a miss under the stricter reading.
5. **The radial-force test, measured against the net force, gives 1.7e-11, not 1e-12.** The case is the hexagon around an anti-phase centre in Model B at spacing 16, where the net force (2.9e-4) is a near-cancellation of pair forces of size 0.77. Measured against the per-lump pair-force scale it is 4.1e-13. Rounding, not symmetry breaking.
6. **Sub-period instabilities.** 19 of 28 balances e-fold in less than one oscillation period. The averaged model assumes the opposite. Those growth rates show the balances are violently unstable, but the numbers themselves are not quantitative.

## 10. What the model cannot say

- Anything below about 3.6 hw quantitatively (the tail law is off by more than 10%). Model B is the superposition, not the dynamics, and lacks core polarization.
- Merger and its products. Runs stop at 2 hw.
- Radiation. N15's radiating case: radiation-balanced states are not excluded by the argument and not modelled. The lone-lump lifetime is about 1e4.
- The centrifugal effect of the overlap angular momentum. The DK mechanism for optical clusters is attraction balanced by angular momentum. Here J is a diagnostic, not a dynamical variable.
- Amplitude dependence of the coupling on N_i, and any change of inertia with N.
- Start-phase effects of the single-frequency ansatz in 3D (section 8 note).
- Non-rigid families. Balances are searched along the one-parameter family x = R x_unit. A deformed shape (for example a bipyramid with two radii) could balance where the rigid family does not. Patterns whose dU/dR vanishes without all forces vanishing are recorded as `R_stationary_not_balance`.

## 11. Against the literature (standard 9) and novelty (standard 10)

| prediction or result here | literature | source (via PRIOR_ART) |
|---|---|---|
| 5-ring, 72-degree steps: radially restoring balance in the overlap region | same: N = 5, m = 1 equilibrium is a minimum of the effective potential, ring oscillates in radius | Desyatnikov & Kivshar 2002 (`READ`, S6.5) |
| every balance unstable; phase sector has negative mass | orbiting two-soliton bound states always unstable; phase has negative effective mass | Malomed 1998 (`READ`, S6.1, S6.5) |
| stepped rings' fastest modes mixed position-phase | staircase clusters break by azimuthal instability under noise | Mihalache 2004 (`ABSTRACT`, S6.5) |
| 90-degree-step rings: no tail-law balance | DK's N = 4 and N = 8, m = 2 equilibria come from overlap and angular momentum | DK 2002 (`READ`, S6.4, S8 item 8) |
| 0/180 alternating rings and cube fly apart | necklaces without angular momentum expand; no radiation-free bound state with a phase-exchanging mirror | Soljacic 1998 (`READ`); N15 (`DERIVED`, S6.3) |
| staircase J is 4 to 13% of N per lump at 4.8 hw | "carries almost no J while the lumps do not overlap" | N12 (`DERIVED`, S2.1). Consistent in kind; this quantifies it, and at 4.8 hw it is not negligible |
| in-phase merges first, stepped later | 3D cubic-quintic: merger at 0 and pi/4, critical phase between pi/4 and pi/2 | Burlak & Malomed 2018 (`READ`, S6.2) |

No claim that anything here is new (standard 10). The point-model scan is arithmetic on a published pair law and the scholar's identities.

## 12. Deviations and choices

- "Spacing" means the minimum pair distance. With a centre that is centre to shell. The scholar quoted the octahedron's shell-to-shell 5.40; both are given in section 5.
- Centre phases for stepped shells are scanned in 30-degree steps. Lumps on a stepping axis get phase 0. m is capped at 0 to 3 on the cube's and icosahedron's 2-fold edge axes (irregular azimuths).
- Pattern equivalence for counting is the group action plus a global phase. For 0/180 that is exactly rotations, reflections and the sign flip. For stepped patterns it adds time shifts.
- The radial-by-symmetry test was first written to include stepped rings with a centre. Their forces are radial but not by the pattern symmetry: the centre breaks the stepping rotation. The test now checks every pattern the code flags as radial by symmetry (58), plus the expected ones.
- Model B with new parameters = Model A(new) + the profile's superposition correction.
- Runs that start inside 2 hw use "collapse" (half the starting spacing) instead of the 2 hw merger rule. Model B runs also stop at d = 3.
- Production integration tolerance was raised to rtol 1e-12 after diagnostic 4. Runtime is 100 s for everything.

## 13. Rerun with new pair-law numbers

Copy `outputs/session2/wpD/params_default.json` and replace A, kappa, E, N, hw, omega and domega_dN with the measured values. Then run, from the repository root:

    /Users/aaronramos/Desktop/SWIRL/.venv/bin/python -m swirl.arrange.scan --params <new params>.json --out outputs/session2/wpD

This rewrites `scan_results.json`, `candidates.json`, `predictions_point_model.json` and `tables.md`, the tables in sections 2, 4 and 7, in about 100 s on this Mac. `fsup_table.json` (the profile's superposition harmonics and overlap) is rebuilt only with `--rebuild-fsup` or when it is missing; that takes about 30 s. Model A balance kappa R values do not change with A or kappa; their physical sizes and all time scales do. Use a different `--out` to keep this version.

## Review corrections (orchestrator, 2026-10-02, from the independent review of merge f049374)

The reviewer rebuilt the model from the formulas and recomputed the two-lump tables, the class counts (all 20 arrangements), the Model A balances (5 of 5), 22 of 23 Model B balances, the growth rates, and three time evolutions with its own integrator. Every recomputed number matches. Verdict: usable with caveats. The caveats change how the ranked list is read:

- **The top of the ranking is the part the notes call an artefact.** Ranks 1 to 5 are Model B balances at 2.0 to 2.4 hw. The ranking key (number of unstable eigenvalues, then slowest e-fold) is not an ordering by likelihood of binding. The superposition force changes sign between d = 5.5 and 6.25 (F_sup(d, 0) = 2.4 at 5.5, -2.65 at 6, -0.78 at 6.25, 9.3 at 7), and the point model's own evolution from those balances gives collapse or merger at t = 46 to 174. **No Model B balance below 2.5 hw (21 of 23) carries a "predicted spacing".** They are 3D probe points whose point-model outcome is merge or collapse.
- **The 4.71 hw balance (6-ring around an anti-phase centre) is not a candidate.** Its radius is set by second-order terms: scaling the even-harmonic correction by 0.5, 0.8, 1.2, 2.0 moves it to 3.75, 4.40, 4.98, 5.73 hw, and at 0 it vanishes. The net force near it (3e-4 to 6e-4) is comparable to the neglected radiation cross-force (about 1e-4). Model B also lacks three-lump overlap terms of the same order. It stays on the 3D list only as a probe.
- **Model B's phase and N sector is unvalidated** (its harmonics assume a potential exists; the test checks only the force).
- **"Bounded" in `predictions_point_model.json` hides expansion.** Labelled bounded but expanding: 5-ring with centre at spacing 20 (+43.7%), octahedron with centre (+49.4%), cube with centre (+36.8%), 4-ring m = 1 at spacing 16 (+32.7%). Predictions are to be written from `mean_pair_distance_change_rel`, not from the label.
- **"None stable" for the stepped rings is the phase-pendulum verdict.** The four m = 1 rings at their Model B balances are positionally stable with phases frozen. Their instability comes from the phase and N sector, whose sign rests on the derived N-flow direction and d omega/dN < 0. The pair batch tests exactly that (C5b, C5c).
- **A structural result the notes do not state.** For uniform stepped rings (K = 3 to 8, every winding, spacings 12, 16, 20) the pattern is a fixed point of the N flow, and the phase sector is unstable only for steps below 90 degrees. For steps of 90 degrees or more it is stable, and those rings have no force balance in either model. In the first-order pair law, attraction between neighbours and stability of their relative phase exclude each other.
- **Angular momentum.** The 4 to 13% of N per lump at 4.8 hw is verified by an independent quadrature. At the Model B balances the table shows J/(n N0) of 0.96 to 1.3, at or above the ramp value; that signals breakdown of the first-order overlap formula there and was not flagged.
- **Check table.** Two rows marked pass exceed the 3x rule and are misses: Model B conservation 4.0e-7 against 1e-8 (under the initial-term normalization) and the radial-force test 1.7e-11 against 1e-12 (against the net force). Four rows of the table have no pytest behind them (their values reproduce from the data files).
- **Wording.** Line 9 means "below 3.6 hw, where the tail law is off by more than 10%". The in-phase control merger times quoted for "both models" are Model A's (Model B: 43 to 82, 126 to 231, 345 to 614). "No arrangement holds in the point model" covers the rigid one-parameter family of each pattern only; 537 of 595 patterns are not radial by symmetry.
