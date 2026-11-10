# Session 5, Track F: literature brief (WP-A items 4 to 6)

Date: 2026-11-10. Author: `scholar` (Track F half). Written only in `docs/`. Nothing committed. `PRIOR_ART.md` and `SESSION_05_PREDICTIONS.md` were not touched; the orchestrator folds this file in. Search numbers are S5F-1 onward; the orchestrator renumbers.

**Evidence tags** (as at the head of `PRIOR_ART.md`, with the exact meaning used here).

- `READ`: I extracted the full text of the arXiv PDF locally (macOS PDFKit) and read the passage cited. `READ (part)` means I read the sections named and not the rest.
- `ABSTRACT`: I saw only the abstract (arXiv abstract page or search-result abstract text).
- `SECONDHAND`: another source says the paper says this, or the only access was a summary written by the web-fetch tool.
- `DERIVED`: my own algebra from the plan's equations or from numbers in a `READ` source. Not a literature claim. Re-derive before relying on it. Items are numbered DF-1 onward and listed in section 6. DF-1, DF-4, DF-5 and DF-7 were also checked numerically (a short script, scratch only; results are quoted in the items).
- "no source found": I searched (see the log) and found nothing.

Notation. S = sum over i of |d_i u|^2, X = |u_t|^2 - S (no 1/2), as in the plan. Amin's variable is X_A = -(1/2) d_mu phi d^mu phi, which in our units is X_A = X/2 (DF-2). L_O and L_R as in the plan.

---

## Gate (answer first)

**YES. The equation-R production runs may go ahead.** The plan's stop condition (section 8: "L_R is not well posed for this use, or the October lesson applies to the full form") is not met. Reasons, in three lines.

1. L_R, with the full square root, is hyperbolic, subluminal and has positive energy density for X < 1. This is Babichev, Mukhanov, Vikman (READ), Adams et al. (READ; their eq. 7 is this Lagrangian's series), Eperon, Reall, Sbierski (READ), and my own algebra (DF-1, DF-4).
2. Three of the four clauses of the October lesson are reversed for L_R (sign of the quartic term, propagation speed, energy bound). The fourth, the static no-go, still holds for L_R, but it only rules out static solutions, not oscillons (DF-6).
3. Amin's existence sign condition is satisfied: Delta = xi_2 = +1/2 > 0 for L_R (Amin eq. 24, READ; map in DF-2).

**Conditions to carry into the run set-up.** None of these stops the start.

- **G1. The guard "X < 1" is not the whole well-posedness condition for the pinch.** For the single bump (one fixed direction) X < 1 is exactly the hyperbolicity condition. For any configuration with more than one field direction active, X < 1 is necessary but not sufficient (DF-5). For the pinch u = f r-hat the extra condition is f_t^2 < 1 + f_r^2 + (f/r)^2. The radial solver cannot see its violation, because the mode that goes bad is the angular one. Add this to the WP-B2 guard and log it. Verified with an independent determinant calculation (DF-5); the reviewer should re-derive it.
- **G2. A guard trip at X -> 1 is a documented physical outcome, not necessarily a bug.** For the massless Born-Infeld/Nambu-Goto equation, a regular graph at t = 0 reaches 1 - X = 0 with a curvature singularity in finite time (Eggers and Hoppe, READ), including for radially symmetric membranes. For the equation with a mass term no existence or blow-up result was found. So log min(1 - X), max |p| (p the conjugate momentum), and where and when, and call a trip a finding with a diagnosis, as the plan's stop condition already says.
- **G3. F1 should be recorded as resting on existence only.** Amin's paper gives existence of small-amplitude k-oscillons and, in the same paper, a long-wavelength instability in three space dimensions (his d >= 3) for small amplitude. It does not give trapping or lifetimes for DBI. Pre-register: small-amplitude starts of R are expected to collapse or disperse by Amin's own criterion; a trapped state, if any, is at large amplitude, where no source was found.
- **G4. The X^2 truncation stays forbidden.** It reintroduces loss of hyperbolicity (for a static gradient with 2/3 < S < 2) and a static energy that is negative for S > 4 (DF-7).
- **G5. (Optional, cheap.)** The single bump in the three-component theory has two transverse polarizations that propagate at exactly the speed of light on top of the oscillating background (DF-5). Their parametric stability is not covered by Amin (single field). No source found.

---

## Three most important findings

1. **L_R is the DBI Lagrangian of Amin's paper with f = 1 and V = phi^2/2.** Existence sign condition Delta = 1/2 > 0 holds. But Amin's own stability result says the small-amplitude 3D family is unstable (long-wavelength, same criterion as the canonical model). I found no published stable 3D k-oscillon family. The one statement about large-amplitude DBI is "rather intriguing dynamics", with the details left to an unpublished Pearce and Amin paper (section 1.6).
2. **The October lesson does not transfer.** Its retired term is L = X_A - 4 kappa X_A^2 (xi_2 = -4 kappa < 0). L_R has xi_2 = +1/2 > 0. I rebuilt the lesson's own numbers (the -3 kappa phi_t^4 energy term and the speed sqrt((1 + 12 kappa b^2)/(1 + 4 kappa b^2))) from that Lagrangian, so the identification is exact (DF-3).
3. **Plan-relevant correction to O9 and to WP-B2 check 6.** O9's algebra is right (DF-1, to 1e-14). But "X cannot trip the guard for finite p" understates the risk (G2), and X < 1 is not sufficient for hyperbolicity of the pinch (G1).

---

## 1. Item 4: a Born-Infeld (DBI) scalar with a mass term

### 1.1 L_R and its place in the literature

- L_R = 1 - sqrt(1 - X) - |u|^2/2 is the Dirac-Born-Infeld Lagrangian of Amin's paper with f = 1 and a quadratic potential. Amin writes it as L = f^-1(phi) [1 - sqrt(1 - 2 f(phi) X_A)] - V(phi), says it gives xi_2 = 1/2, and says he analysed large-amplitude oscillons in it with V = phi^2/2, results not presented ([Amin 2013](https://arxiv.org/abs/1303.1102), sec. 2 and 5, `READ`).
- In 1+1 dimensions, 1 - sqrt(1 - X) with X = u_t^2 - u_x^2 and no potential is the Born-Infeld scalar, the gauge-fixed Nambu-Goto string ([Eperon, Reall, Sbierski](https://arxiv.org/abs/1802.03020), sec. 3.2-3.3, `READ`; their c = +1 theory).
- The same expansion appears as the textbook example of a brane fluctuation: L = -f^4 sqrt(1 - (dy)^2) = f^4(-1 + (dy)^2/2 + (dy)^4/8 + ...), which they note has the "correct" (positive) quartic sign ([Adams, Arkani-Hamed, Dubovsky, Nicolis, Rattazzi](https://arxiv.org/abs/hep-th/0602178), eq. 7, `READ`). Our L_R has the series X/2 + X^2/8 + ..., the same.

### 1.2 Hyperbolic, positive energy, for X < 1

- **General k-essence criterion.** For L(X_A, phi) the field equation has principal part G^{mu nu} d_mu d_nu phi with G^{mu nu} = L_X g^{mu nu} + L_XX d^mu phi d^nu phi (their 2.4). It is hyperbolic (G Lorentzian) iff 1 + 2 X_A L_XX / L_X > 0 (their 2.5). The null energy condition is L_X >= 0; violating it means an energy unbounded below. They add that L_X must nowhere vanish or become infinite, since those points are singularities of the emergent geometry, and that a well-posed Cauchy problem does not guarantee a global solution ([Babichev, Mukhanov, Vikman 2007](https://arxiv.org/abs/0708.0561), sec. 2, 4, `READ (part)`).
- **No superluminality** for a single derivatively coupled scalar in a general background needs L to be convex in X: L'' >= 0 ([Adams et al.](https://arxiv.org/abs/hep-th/0602178), p. 11, `READ`).
- **For L_R**: L_X = 1/sqrt(1 - X) > 0, L_XX > 0, and 1 + 2 X_A L_XX/L_X = 1/(1 - X) > 0 for X < 1. All three conditions hold exactly when X < 1. L_X diverges as X -> 1, which Babichev et al. flag as the singular point of the emergent geometry (`DERIVED`, DF-2, from their criteria).
- **Energy.** With p the momentum conjugate to u, H = sqrt((1 + p^2)(1 + S)) - 1 + |u|^2/2, and 1 - X = (1 + S)/(1 + p^2). So H >= |u|^2/2 >= 0, and X < 1 is equivalent to p finite (`DERIVED`, DF-1; the same as O9). The same holds for the three-component field with p a vector and S the full sum.
- **Local Cauchy problem.** Hyperbolicity gives local well-posedness by the standard quasilinear theory (Kato; Hughes, Kato, Marsden; not read, no arXiv text). The mass term is a lower-order term and does not change the principal part. I did not find a source that states this for the massive Born-Infeld scalar, so "local well-posed for X < 1" is `DERIVED` plus standard theory, not a quoted result.

### 1.3 Propagation speeds along and across the gradient

Single field. Background u_t = a, gradient of size b along a unit vector n. Define X = a^2 - b^2. `DERIVED` (DF-4); checked numerically against the closed forms (agreement to the 7 digits printed).

| Direction of the wave vector | Speed | Notes |
|---|---|---|
| across the gradient (k perpendicular to n) | c_perp^2 = (1 - X)/(1 + b^2) = 1 - a^2/(1 + b^2) | at most 1 |
| along the gradient (k parallel to n) | v = (-a b +/- sqrt(1 - X))/(1 + b^2) | magnitude at most 1; equals 1 only at X = 0 |
| gradient-free (b = 0) | c^2 = 1 - a^2 = 1 - X | the DBI speed limit; also Amin eq. 25, c_s^2 = (1 + 2 X_A T_XX/T_X)^-1 = 1 - 4 xi_2 X_A + ..., `READ` |
| static (a = 0) | c_perp = 1, c_par = 1/sqrt(1 + b^2) | subluminal along the gradient |
| retired term, static, for comparison | c_par^2 = (1 + 12 kappa b^2)/(1 + 4 kappa b^2) | exactly the speed in LESSONS item 6; above 1 |

The characteristic cone is the null cone of the induced (Nambu-Goto) metric. Eperon et al. state the c = +1 theory is subluminal in the sense that the null cone of g lies on or inside that of the Minkowski metric (their 3.12, `READ`), consistent with the table.

Vector-valued field (the pinch, and the bump with its two extra components): see DF-5 and G1. In the three-component theory two polarizations propagate on the light cone and one on a DBI-type cone, and X < 1 does not guarantee that the DBI-type cone is real.

### 1.4 Can X reach 1, or smoothness be lost in finite time from smooth data

Without a potential (the massless Born-Infeld equation):

- Exceptional, no shocks. In the 1+1 case a wavepacket moves without distortion and never forms a shock (the property called "exceptional"; Taniuti 1959 via [Eperon et al.](https://arxiv.org/abs/1802.03020) `READ`). For the Born-Infeld electrodynamics the same word means no shocks form, and "completely exceptional" is Boillat's term; a 3D result of Christodoulou is reported for the scalar (Chaplygin) case ([Kadlecova](https://arxiv.org/abs/2107.12249), intro, `SECONDHAND`, which cites Boillat and Christodoulou, neither read).
- Simple waves do not steepen. Both the canonical and the DBI scalar are free of caustics for planar simple waves, where generic k-essence is not ([Mukohyama, Namba, Watanabe 2016](https://arxiv.org/abs/1605.06418), `ABSTRACT`; against [Babichev 2016](https://arxiv.org/abs/1602.00735), `ABSTRACT`, who argued generic k-essence forms caustics).
- But smooth data can still fail in finite time by another route. In the 1+1 massless equation z_tt(1 + z_x^2) - z_xx(1 - z_t^2) = 2 z_t z_x z_xt, "can develop singularities in finite time, starting from finite initial data". The singularity is a swallowtail: the graph is regular at t = 0 and has a curvature singularity at t = 1 in their explicit example (their eqs. 31 and 32), and for t > 1 is no longer a graph. At the singular point L^2 = 1 - z_t^2 + z_x^2 = 0, which in our variables is X = 1 ([Eggers, Hoppe 2008](https://arxiv.org/abs/0812.1790), `READ`). They state the same singular solution applies to radially symmetric membranes and to higher dimensions (their sec. III).
- Eperon et al. describe the same thing as two strong wavepackets meeting and the string folding back, so Phi would have to become multivalued; for small amplitude the packets pass through each other with a time delay, for large amplitude a singularity forms, and the maximal globally hyperbolic development is then not unique in the subluminal theory ([Eperon et al.](https://arxiv.org/abs/1802.03020), sec. 1 and 3.4, `READ (part)`).
- Small data are fine: global existence for small data of the timelike minimal-surface (Born-Infeld) equation in Minkowski space ([Lindblad 2002](https://arxiv.org/abs/math/0210056), `ABSTRACT`). Small 1+1 solutions decay inside a smaller region of the light cone ([Alejo, Munoz 2017](https://arxiv.org/abs/1707.02595), `ABSTRACT`). Explicit self-similar blow-up solutions of the 1+1 equation, with a nonlinear stability result ([Yan 2018](https://arxiv.org/abs/1807.04227), `ABSTRACT`).

With a potential:

- For Born-Infeld scalars with a potential that multiplies the square root, V(T) sqrt(1 + (dT)^2), and runaway V, caustics (multivalued regions) form generically, in 1+1 and in 3+1 ([Felder, Kofman, Starobinsky 2002](https://arxiv.org/abs/hep-th/0208019), `ABSTRACT` plus the first two sections read). That is a different structure from L_R (an additive mass term, no runaway), so it is context, not a prediction for us.
- For the DBI scalar plus an additive quadratic potential, as in L_R: **no source found** on well-posedness, global existence or blow-up (searches S5F-7, S5F-18).

What follows for the runs. The singularity mechanism is amplitude-driven (strong, colliding packets; X -> 1 needs |u_t| of order the Born-Infeld speed, which in our units is field amplitude of order 1/omega). An oscillon at large amplitude is a standing superposition of left- and right-moving pieces, so G2 is a live possibility in the large-amplitude part of the family. Not a prediction; no source applies.

### 1.5 Amin's existence condition ([arXiv:1303.1102](https://arxiv.org/abs/1303.1102), PRD 87, 123505, `READ`)

- Lagrangian L = T(X_A, phi) - V(phi), X_A = -(1/2) eta^{ab} d_a phi d_b phi (mostly plus signature, so X_A = (1/2)(phi_t^2 - |grad phi|^2)), T = X_A + xi_2 X_A^2 + xi_3 phi X_A^2 + ..., V = phi^2/2 + lambda_3 phi^3/3 + lambda_4 phi^4/4 + ... (eqs. 1 to 3).
- Small-amplitude expansion phi = eps phi_1 + ..., tau = sqrt(1 - eps^2) t, rho = eps r. The profile equation is f'' + ((d - 1)/rho) f' - f + (3/4) Delta f^3 = 0 (eq. 18) with **Delta = xi_2 - lambda_4 + (10/9) lambda_3^2** (eq. 19). Oscillons exist iff Delta > 0 (eq. 24, boxed). For a quadratic potential, xi_2 > 0 suffices, which he states is equivalent to c_s < 1 (eq. 25).
- 1D: f = sqrt(8/(3 Delta)) sech(rho). 3D: numerical, f_0 = 3.06699 sqrt(8/(3 Delta)) (eq. 32). No simulations tables for non-canonical models are given.

**Map to L_R** (`DERIVED`, DF-2): X_A = X/2, T = 1 - sqrt(1 - 2 X_A) = X_A + X_A^2/2 + X_A^3/2 + (5/8) X_A^4 + ..., so xi_2 = 1/2, lambda_3 = lambda_4 = 0, m = 1. **Delta = 1/2 > 0: the sign condition is satisfied.** At leading order in 3D (using S(0) = 4.33738768 and the integral of S^2 = 18.897 for the canonical 3D profile, from Fodor et al., `READ` in the Session 1 brief): phi(0) = 7.08 eps and E = 25.2/eps + O(eps), with omega^2 = 1 - eps^2. So the amplitude goes to zero and the energy diverges as eps goes to zero. Context: valid only where the single-frequency small-amplitude expansion is, eps below about 0.25 in the canonical case.

For the single bump u = f e in the three-component theory, S = f_r^2 and |u_t|^2 = f_t^2, so L_R reduces exactly to the single-field DBI Lagrangian. Amin's single-field result applies to that invariant sector.

### 1.6 What Amin found for stability and lifetime, and later work

- **Stability criterion** (eqs. 53 to 56): small-amplitude oscillons are stable against long-wavelength perturbations iff dN/d eps > 0, with N = eps^{2-d} times the integral of f^2 d^d rho. So stable for d = 1, 2 and **unstable for d >= 3**, "confirmed by numerical simulations" with no data shown. The typeset eq. 56 ends in "proportional to eps^{d-2}", which contradicts the first part of the same line and eq. 54; the text and the first equality give eps^{2-d}. Treat that as a typo.
- He says large amplitudes may remove the 3D instability (as in Amin and Shirokoff, where a large sextic term does it), and that "no general stability condition exists" for large amplitude. The sextic term is what turns N(eps) around there ([Amin, Shirokoff 2010](https://arxiv.org/abs/1002.3380), `READ` in the Session 1 brief).
- **Lifetime**: no number. The radiating tail is expected to be exponentially small, decay rate of order exp(-1/eps), citing Segur-Kruskal and Fodor et al. He computes radiation only in an expanding background (1+1).
- **DBI at large amplitude**: "rather intriguing dynamics appearing at large amplitudes", reference [53] Pearce and Amin "in progress"; acknowledgment thanks M. Pearce for simulations of large-amplitude DBI oscillons (sec. 5 and 6, `READ`). **I found no published Pearce and Amin paper** (S5F-1, S5F-9, S5F-16).
- Later work found:
  - [Sang, Huang 2021](https://arxiv.org/abs/2012.14697) (PLB 823, 136781), 3+1D lattice, DBI kinetic term, axion-monodromy potential, formation during preheating. Reports oscillon-like overdensities with space-averaged Lorentz factor up to about 3.9 and says formation is suppressed by the non-canonical term; gives no lifetimes, amplitudes or frequencies. `SECONDHAND` (fetch-tool summary of the ar5iv text; the PDF exceeds the tool's size limit).
  - [Sakstein, Trodden 2018](https://arxiv.org/abs/1809.07724) (PRD 98, 123512): oscillon-type quasi-breathers sustained by the quartic galileon, only C^1. `ABSTRACT`. Different structure.
  - [van Dissel, Sfakianakis](https://arxiv.org/abs/2010.07789) (PRD 106, 096018) say non-canonical oscillons have been considered only for single fields (their ref. 8 is Amin) (sec. II, `READ (part)`).
  - [Diaz-Alonso, Rubiera-Garcia 2007](https://arxiv.org/abs/0705.0112) (PLB 653, 445): some k-field models with kinetic self-coupling do have static, finite-energy, spherically symmetric solutions, under conditions on the model (vanishing vacuum energy, positive-definite energy density). `ABSTRACT`. This does not contradict DF-6 (L_R is not shown to be in that class).
  - [Endlich et al. 2010](https://arxiv.org/abs/1002.4873) (JHEP 05 (2011) 73): no stable static solitons in a broad class of derivatively coupled theories including some k-essence models. `ABSTRACT`.
- No source found for any measured lifetime of a DBI-plus-mass oscillon in 3+1 (S5F-1, S5F-9, S5F-11, S5F-16).

### 1.7 Energy minimum, stable branch, saturating term (3D)

- Canonical sextic case: the family has an energy and N minimum at finite amplitude, which is the edge of the stable branch (details and numbers in `PRIOR_ART.md` 2.2 to 2.4, `READ`). Examples: [Fodor et al. 2008](https://arxiv.org/abs/0802.3525), E = 6.29908/eps + 264.262 eps for the phi^4 double well; [Amin, Shirokoff](https://arxiv.org/abs/1002.3380), N minimum near alpha^2 = 0.025; [Li, Yamaguchi, Zhang 2025](https://arxiv.org/abs/2507.13276), beta = 4/3, omega_death = 0.97, N = 80.3. All from the Session 1 brief; not re-opened this round.
- k-oscillon case: the small-amplitude end gives N proportional to 1/eps and E = 25.2/eps for L_R (section 1.5), so E rises without bound toward small amplitude. Whether E turns up again at large amplitude, i.e. whether the kinetic nonlinearity supplies its own saturation, **no source found**. Amin says large amplitude "requires a case by case analysis" and gives none for DBI. The expansion has no small parameter here (Delta = 1/2 is order one, unlike the large-g limit of Amin and Shirokoff), so the interior minimum, if it exists, sits at order-one amplitude where only numerics can decide. This is why F1 and F2 for R rest on the runs, not on theory.

---

## 2. Item 5: does the October lesson apply to the full square-root L_R

**Answer: No.** Three clauses are reversed; the fourth (static no-go) holds but does not constrain oscillons.

The retired term. LESSONS item 6 gives W = phi'^2/2 + 3 kappa phi'^4 - V, a term -3 kappa phi_t^4 in H, and a speed sqrt((1 + 12 kappa phi'^2)/(1 + 4 kappa phi'^2)). I rebuilt all three from L = X_A - 4 kappa X_A^2 - V (DF-3). So the retired term is xi_2 = -4 kappa < 0 in Amin's notation.

| Clause | Retired term (LESSONS item 6) | L_R, full square root | Source, tag |
|---|---|---|---|
| (a) Static regular solution | none exists | **none exists either** | `DERIVED` DF-6: the quantity H_l = V - (1 - 1/sqrt(1 + phi'^2)) satisfies dH_l/dr = (2/r) phi'^2/sqrt(1 + phi'^2) >= 0 with H_l(0) = V(phi_0) >= 0 and H_l(infinity) = 0, so phi_0 = 0. Derrick bracket is also a sum of non-negative terms. Endlich et al. (`ABSTRACT`) are consistent. **Harmless for oscillons**: Amin's k-oscillons are time-periodic and need no static balance (`READ`). |
| (b) Sign of the quartic terms | +kappa grad^4 in the static energy (beats Derrick); -3 kappa phi_t^4 in H | xi_2 = +1/2: static gradient energy S/2 - S^2/8 + ... (the quartic is negative, so Derrick is made worse, not better); H in velocity variables = u_t^2/2 + (3/8) u_t^4 + ... (positive). In momentum variables the quartic is -p^4/8, but sqrt(1 + p^2) - 1 is non-negative, so the sign of a truncated coefficient is not the criterion | `DERIVED` DF-2, DF-3; sign is the "correct" one in [Adams et al.](https://arxiv.org/abs/hep-th/0602178) eq. 7 (`READ`); Amin: xi_2 > 0 gives Delta > 0 (`READ`) |
| (c) Propagation speed | superluminal along gradients | subluminal everywhere in X < 1; at most 1 (table in 1.3) | Adams et al. p. 11 (L'' >= 0, `READ`); Eperon et al. (3.12) (`READ`); DF-4 |
| (d) Energy bounded below | no (-3 kappa phi_t^4) | yes: H >= |u|^2/2 >= 0 | `DERIVED` DF-1; NEC L_X > 0 ([Babichev et al.](https://arxiv.org/abs/0708.0561), `READ`) |

**O9, checked against sources.** The conjugate-momentum form H = sqrt((1 + p^2)(1 + S)) - 1 + |u|^2/2 and the statement "the X^2 term has the opposite sign to the retired one" are both confirmed (DF-1: Legendre transform checked numerically, maximum error 7e-15 over 10,000 random states; DF-3). No source contradicts them. Two cautions, from G1 and G2: p can diverge in finite time in the massless limit, so "X stays below 1" is not a reason to drop the guard; and for the three-component field X < 1 does not give hyperbolicity.

**What truncating at X^2 would do** (`DERIVED`, DF-7). L_T = X/2 + X^2/8 - |u|^2/2.

- For a static gradient S, hyperbolicity needs (1 - 3S/2)/(1 - S/2) > 0, which fails for 2/3 < S < 2 (the equation changes type) and for S > 2 the coefficient of the kinetic term is negative (the null energy condition is lost).
- Static energy density S/2 - S^2/8 is negative for S > 4: unbounded below.
- So the truncated model has both pathologies of the retired term, at a gradient |grad u| of order one in units of the limit. Amin uses the X^2 truncation legitimately, only as a leading-order small-amplitude expansion (`READ`). The plan is right to forbid it for production.

---

## 3. Item 6: vector oscillons

### 3.1 Their model ([Zhang, Jain, Amin 2022](https://arxiv.org/abs/2111.08700), PRD 105, 096037, `READ`; and the first author's thesis [arXiv:2401.00043](https://arxiv.org/abs/2401.00043), sec. 3.5, `READ (part)`)

- Real massive vector (Proca) field W_mu: L = -(1/4) X_{mu nu}X^{mu nu} - V(W_mu W^mu), V = m^2 W^2/2 - (lambda/4)(W^2)^2 + (gamma/6)(W^2)^3, with lambda, gamma > 0 and gamma = lambda^2/m^2 in the simulations (paper, eqs. 1 and 2, footnote 1). The thesis writes it with lambda = -1 and h = 1 (eq. 3.81 and sec. 3.5.5). In the plan's convention beta = (10/3) g6/g4^2 (`PRIOR_ART.md` 1.1, with g4 = lambda, g6 = gamma, m = 1) this is **beta = 10/3**, not 4/3 (`DERIVED` arithmetic). Their numbers are context, not predictions for our beta = 4/3.
- Nonrelativistic limit: three spatial components behave like Schroedinger fields with the constraint W_0 solved; the nonlinear potential contains a term proportional to the spin density squared, S.S (paper eq. 8). This is what separates directional from spinning.
- The component equations decouple in the free nonrelativistic limit ([Adshead, Lozanov 2021](https://arxiv.org/abs/2101.07265), `ABSTRACT`).

### 3.2 The three configurations (all spherically symmetric in energy; only the hedgehog is also in the field)

| Configuration | Form of W (small amplitude) | Total spin | Maps to ours |
|---|---|---|---|
| Directional | W = f(r) cos(omega t) z-hat, fixed direction | 0 | **single bump** (u = f e). Profile eq. v'' + 2v'/r - v + (3/4)|lambda| v^3 = 0 (thesis 3.96) |
| Spinning | W = f(r)[cos(omega t) x-hat + sin(omega t) y-hat], circular polarization | N times hbar (N = particle number) | a bump with a second sideways component **out of step by a quarter period**; same profile in both. Not an S-bend. Plan's reading "spinning = candidate for the twist" fits the out-of-step part only |
| Hedgehog | W = f(r) r-hat cos(omega t), f(0) = 0 | zero | **pinch** (u = f r-hat). Profile eq. v'' + 2v'/r - (1 + 2/r^2) v + (3/4)|lambda| v^3 = 0 (thesis 3.98): the same cubic coefficient as directional, plus the l = 1 term |

### 3.3 Energies

- **At fixed particle number** (nonrelativistic, small and moderate amplitude): directional < spinning, E_d < E_s (paper eq. 20, `READ`). For gamma = 0, E_d = (4/9) E_s at equal N and (2/3) E_s at equal omega (paper footnote 3). Hedgehog is the highest: the paper says it "includes significant relativistic corrections towards its center, and is also higher in energy" than the other two, and defers details to future work. No number in the paper.
- The thesis gives numbers: leading-order energies 5.01/eps, 18.9/eps, 73.4/eps for directional, spinning, hedgehog (sec. 3.5.2), and argues the ordering at fixed N from dE/dN = omega: "directional and hedgehog oscillons are the lowest- and highest-energy state respectively, which is not surprising since hedgehog profiles have nodes". It says the conclusion was checked with subleading profiles. `READ (part)`.
- **Check of the printed numbers** (`DERIVED`, DF-8). With the printed profile equations and the energy formula E = (1/2 eps) times the integral of W-dot^2 + W^2, I get E_s = 18.9/eps (agrees) and E_d = 12.6/eps, not 5.01/eps. The ratio 73.4/12.6 = 5.8 equals the ratio of the two profile integrals, which I can reproduce from the printed equations. The printed 5.01 is either a typo or defined differently. The ordering d < s < h is not affected. I would use "hedgehog about 6 times the directional energy at the same omega, small amplitude, that model" and not the 14.7 that the printed numbers give.

### 3.4 Lifetimes (Proca model, beta = 10/3)

| Object | Lifetime | Conditions | Source |
|---|---|---|---|
| Directional | no significant energy loss over about 1e5 /m (about 1.6e4 cycles at omega = 0.975); lifetime bound alpha_d >= 1e5 | 2+1D reduction (cylindrical symmetry), absorbing boundary, omega about 0.975 m, E about 164 m/lambda from a Gaussian start | paper sec. VI and App. C, D, `READ` |
| Spinning | no decay for at least about 1e3 /m (about 155 cycles) | full 3+1D, absorbing boundary | same |
| Hedgehog | about 1e6 /m (about 1.6e5 cycles at omega 0.976 to 0.986) | **1+1D radial reduction**; initial W_i = (x_i/r) C sin(pi r/2R) exp(-r^2/R^2), C = 1/sqrt|lambda|, R = 9/m; decay rate has a dip | thesis sec. 3.5.5 and Fig. 3.15 (text and axis labels only; figure not examined), `READ (part)` |

The hedgehog number is from a spherically symmetric run, so it cannot see the object losing its roundness: an upper bound with respect to non-radial instabilities (the same caveat as the plan's section 7). The directional run is 2+1D and the spinning run is 3+1D but short. These lifetimes are not comparable with each other as measured, and none is a prediction for our potential.

### 3.5 Hedgehog references the paper cites, and other multi-component results

- The paper's references for the hedgehog (its refs. 50, 72 to 76): Adshead and Lozanov (self-gravitating vector dark matter, `ABSTRACT`: hedgehog vector solitons, lowest hedgehog halo about three times less gravitationally bound than the ground state; gravity-bound, nonrelativistic), Loginov 2015 (PRD 91, 105028, complex vector Q-ball, not opened), [Brito, Cardoso, Herdeiro, Radu 2015](https://arxiv.org/abs/1508.05395) (Proca stars, `ABSTRACT`; complex field coupled to gravity), and three Proca-star numerical papers (not opened). All are complex vector fields with gravity or a U(1) charge, not real fields with a sextic potential.
- Real **scalar** multi-component sextic oscillons: [van Dissel, Sfakianakis](https://arxiv.org/abs/2010.07789) (`READ (part)`): two or N fields each with the same quadratic, quartic and sextic terms and a quartic cross-coupling, fields oscillating in or out of phase; they extend the Vakhitov-Kolokolov criterion, find lifetimes like the single-field case, and run 1+1D and 3+1D. No l = 1 hedgehog.
- **No source found** for a hedgehog (l = 1, spherically symmetric, three-component) oscillon in a real three-scalar model with a phi^4-phi^6 potential, or in any scalar non-canonical multi-component model (S5F-12, S5F-14, S5F-15).

### 3.6 What differs from L_O and L_R, and what carries over

Differences.

1. Proca has a constraint (W_0 is not dynamical), three polarizations with the longitudinal one special, and a Lorentz-invariant potential V(W_mu W^mu) that includes -W_0^2. The hedgehog is purely longitudinal (curl-free), exactly the part the constraint changes at relativistic order, which is why the paper finds large relativistic corrections near its centre. Three real scalars have no constraint.
2. Our potential is V(|u|^2) with the same quadratic, attractive quartic, repulsive sextic structure; the nonrelativistic limit of the two models has the same kinetic term (three decoupled Schroedinger fields) and the same kind of quartic terms. Their beta is 10/3 and ours 4/3.
3. L_R has a non-canonical kinetic term; nothing in the vector papers does.

What carries over (nonrelativistic, small to moderate amplitude).

- **Energy ordering at fixed N.** The hedgehog's nonrelativistic kinetic energy is f_r^2 + 2f^2/r^2 against f_r^2 for the same profile, with the same local quartic, so its energy exceeds the bump's at equal N (`DERIVED`, DF-8). Directional vs spinning: in a three-scalar model the time-averaged quartic energy is (3/8)|A|^4 for linear and (1/4)|A|^4 for circular polarization (A the complex amplitude), ratio 3:2, so the energy ratio 4/9 for the pure-quartic case matches the paper's footnote (`DERIVED`, DF-8).
- **Energy ratio at the same omega, hedgehog to bump, about 5.8** at small amplitude, from the thesis profile integrals. For L_R the leading cubic coefficient comes from the u_t^4 part of X^2, which is the same for any linearly polarized pattern, so the ratio should hold there too at small amplitude (`DERIVED` and unverified; it needs the thesis profile integrals to be right).
- **Existence needs an attractive cubic** for all three (thesis sec. 3.5.1): for L_R that is Delta > 0.
- Lifetimes, amplitudes and the large-amplitude ordering do not carry over.

---

## 4. A source for F2 (energy along a 3D family has a minimum at finite amplitude)

- **Canonical:** Fodor, Forgacs, Horvath, Mezei 2008 (E = 6.29908/eps + 264.262 eps, double well; the energy minimum is near omega_c = 0.9659, which is also where stability begins), [Amin, Shirokoff 2010](https://arxiv.org/abs/1002.3380), [Hertzberg 2010](https://arxiv.org/abs/1003.3459), [Mukaida et al.](https://arxiv.org/abs/1612.07750), [Ibe et al.](https://arxiv.org/abs/1901.06130), [Li, Yamaguchi, Zhang 2025](https://arxiv.org/abs/2507.13276), all `READ` in the Session 1 brief, `PRIOR_ART.md` 2.1 to 2.4. The relation dE = omega dN ties the energy minimum to dN/d omega = 0, the edge of the stable branch. For our O, beta = 4/3: omega at the energy minimum 0.97, N = 80.3 (Li 2025).
- **Non-canonical (L_R):** only the small-amplitude end is sourced (E proportional to 1/eps, from [Amin 2013](https://arxiv.org/abs/1303.1102), `READ`, with DF-2). A minimum at finite amplitude: **no source found**. See 1.7.

---

## 5. Predictions on record against the sources

| # | Plan says | What the literature expects | Source, tag | Plan vs literature |
|---|---|---|---|---|
| F1 | R traps the single bump below the floor; basis Amin, "scholar to confirm the sign condition" | Small-amplitude k-oscillons exist if Delta > 0; for L_R Delta = xi_2 = 1/2 > 0. In 3D the same paper's criterion says the small-amplitude family is **unstable** (dN/d eps < 0). No source for a stable large-amplitude DBI family; the one remark is unpublished | [Amin 2013](https://arxiv.org/abs/1303.1102) `READ`; DF-2 | **Agrees on existence, rests on nothing for trapping.** The cited basis supports a quasi-breather, not a trapped state. By Amin's own criterion a small-amplitude start is expected to collapse or disperse |
| F2 | Energy along a 3D family has an interior minimum | Yes for the canonical sextic and double-well models (sources in section 4). For R only the small-amplitude end (E rises as 1/eps) is sourced | `PRIOR_ART.md` 2.1 to 2.4 `READ`; Amin `READ`; DF-8 | **Agrees for O. Rests on nothing found for R.** |
| F3 | Pinch exists in O and R with higher energy than the bump at the same rate; no lifetime prediction | In the real Proca model with the same potential structure, hedgehog has the highest energy: stated in the paper, with numbers and an argument at fixed N in the thesis (about 5.8 times directional at the same omega, small amplitude). Hedgehog radial lifetime about 1e6/m at beta = 10/3 | [Zhang, Jain, Amin](https://arxiv.org/abs/2111.08700), [thesis](https://arxiv.org/abs/2401.00043) `READ (part)`; DF-8 | **Agrees in direction for O (DF-8 makes it a nonrelativistic inequality at equal N). For R: no source found; DF-8 suggests the same.** Their model is Proca, not ours; no source found for a hedgehog oscillon in a three-scalar or non-canonical model. Lifetime numbers do not carry over |
| F4 | Energy per cycle stays nearly constant while E and rate drift slowly; "new measurement" | **No source found** for it as stated. The sources say the opposite along a radiating family: N is what is lost (decay to N_min), dE = omega dN, so E/omega tracks N. From the beta = 4/3 numbers N(0.94) = 96.2, N(0.85) = 525, ln N changes 16.9 times faster than ln omega; I get change of ln(E/omega) to change of ln E of about 1.06, far above the rule's 0.2 (`DERIVED` DF-9, error from the figure reading E/(mN) about +/-0.03). F4 can hold only near the N minimum, where dN/d omega is about 0 | Mukaida eq. 2.26, Ibe eq. 21, Li 2025 (`READ`, via `PRIOR_ART.md`); Ibe Fig. 6 (N decays to about 80) | **Rests on nothing found, and the sourced relations point against it** away from omega_death. Consistent with O8 |
| O8 | F4 misses for any lump losing more than a few percent, since dE = omega dN | Agrees: dE = omega dN is in two independent `READ` sources | as for F4; project data (Session 4: change in ln E = -0.52 with change in ln omega = +0.04) | **Agrees** |
| O9 | In (f, p), X < 1 identically for finite p; what fails is resolution; H = sqrt((1 + p^2)(1 + S)) - 1 + f^2/2; quartic gradient energy has the opposite sign to the retired term | Algebra and sign confirmed (DF-1, DF-3). Caution: the massless equation reaches 1 - X = 0 in finite time from smooth data, which in (f, p) is p -> infinity at a point; and X < 1 is not sufficient for hyperbolicity with more than one field direction (DF-5) | [Eggers, Hoppe](https://arxiv.org/abs/0812.1790) `READ`; [Eperon et al.](https://arxiv.org/abs/1802.03020) `READ (part)`; DF-1, DF-3, DF-5 | **Agrees with the algebra. Disagrees with the implication that the guard is idle**: see G1, G2 |

---

## 6. DERIVED notes

Each is my own algebra, not a literature claim. DF-1, DF-4, DF-5, DF-7 were also checked numerically with a short script (scratch, not project code: `/private/tmp/claude-501/-Users-aaronramos-Desktop-SWIRL/a9be1e91-ebcc-49bf-ac1c-002c1f72b41f/scratchpad/tools/check_alg.py` and `check_alg2.py`). The reviewer should re-derive independently.

**DF-1. Energy.** L_R = 1 - sqrt(1 - X) - V, X = |u_t|^2 - S. Momentum p = u_t/sqrt(1 - X). Then 1 - X = (1 + S)/(1 + |p|^2), |u_t|^2 = |p|^2(1 + S)/(1 + |p|^2), and H = p.u_t - L = (1 + S)/sqrt(1 - X) - 1 + V = sqrt((1 + |p|^2)(1 + S)) - 1 + V. Hence H >= V >= 0, and X < 1 iff p is finite. Numerical check of the Legendre transform at 10,000 random states: maximum error 7e-15. Pinch: S = f_r^2 + 2f^2/r^2, p = p_r r-hat.

**DF-2. Map to Amin.** X_A = X/2. 1 - sqrt(1 - 2X_A) = X_A + X_A^2/2 + X_A^3/2 + (5/8) X_A^4 + ... So xi_2 = 1/2, lambda_3 = lambda_4 = 0, Delta = 1/2. T_X = (1 - 2X_A)^(-1/2), T_XX = (1 - 2X_A)^(-3/2), 1 + 2X_A T_XX/T_X = 1/(1 - 2X_A) = 1/(1 - X). c_s^2 = 1 - X. Amin's constants are consistent: 3.06699 times sqrt(2) = 4.33738 = S(0). Leading 3D numbers for L_R (Delta = 1/2): f_0 = S(0) sqrt(4/(3 Delta)) = 7.083, integral f^2 d^3 rho = (4/(3 Delta)) 18.897 = 50.39, E = (1/(2 eps)) times that = 25.2/eps. These use the canonical 3D constants (S(0), 18.897 from Fodor et al., Session 1 brief) and the leading-order energy without the O(eps) correction.

**DF-3. The retired term.** L = X_A - 4 kappa X_A^2 - V gives: static radial W = phi'^2/2 + 3 kappa phi'^4 - V (the lesson's W); homogeneous H = phi_t^2/2 - 3 kappa phi_t^4 (the lesson's term); speed along a static gradient b: c^2 = 1 + 2X_A L_XX/L_X = (1 + 12 kappa b^2)/(1 + 4 kappa b^2) (the lesson's expression). So the retired term is xi_2 = -4 kappa. L_R is xi_2 = +1/2.

**DF-4. Speeds.** Principal symbol of the single-field equation: (1 - X) eta^{mu nu} d_mu d_nu phi + d^mu phi d^nu phi d_mu d_nu phi = lower order (signature + - - -). Plane wave on background (a, b n): (1 + b^2) omega^2 + 2 a b omega k_par - (1 - a^2) k_par^2 - (1 - a^2 + b^2) k_perp^2 = 0. Gives the table in 1.3. Bound: (|a| b + sqrt(1 - X))^2 <= (a^2 + (1 - X))(b^2 + 1) = (1 + b^2)^2, so |v| <= 1, with equality only at X = 0. Numerical check of the closed forms against the quadratic: agreement to the 7 digits printed, at four test backgrounds.

**DF-5. Three-component field.** Equation: (1 - X) box u^a + d^mu u^a d^nu u^b d_mu d_nu u^b = -(1 - X)^{3/2} V_a. Principal symbol P^{ab} = (1 - X)(xi.xi) delta^{ab} + (d u^a . xi)(d u^b . xi). Eigenvalues: (1 - X)(xi.xi), twice (two polarizations at the speed of light), and (1 - X)(xi.xi) + sum_a (d u^a . xi)^2, once. The last defines a cone G^{mu nu} xi_mu xi_nu = 0 with G^{mu nu} = (1 - X) eta^{mu nu} + sum_a d^mu u^a d^nu u^a. With M^a_i = d_i u^a, w = M k, S = ||M||^2: the quadratic in omega has real roots iff D(k) = (u_t . w)^2 + (1 + S)[(1 - X)|k|^2 - |w|^2] >= 0 for all k. For one field direction D = (1 - X)[(1 + S)|k|^2 - (k.grad f)^2] >= (1 - X)|k|^2 > 0: X < 1 suffices. For several directions it does not: u_t = (0, sqrt 1.2, 0), u_x = (sqrt 0.5, 0, 0) has X = 0.70 < 1 and D = -0.30 < 0; an independent determinant calculation gives roots omega = +/- 0.365 i. Among 1,415 random three-component states with X < 1, 281 (20 percent) had a non-hyperbolic direction (random states, not necessarily reachable). Pinch (frame r, theta, phi; q = f/r): D_r = (1 + 2q^2)(1 - X) >= 0 (same as the radial reduction, so the radial solver sees X < 1 only), and **D_theta = (1 + S)(1 + f_r^2 + q^2 - f_t^2)**, which is negative in the window 1 + f_r^2 + q^2 < f_t^2 < 1 + f_r^2 + 2q^2 even though X < 1. Example f_t = 1.2, f_r = 0.3, q = 0.5: X = 0.85, D_theta = -0.159. The window needs |f_t| > 1, i.e. large amplitude. Single bump transverse modes: effective metric conformal to Minkowski, so exactly luminal; with an oscillating coefficient P_X(t) (parametric resonance is possible; no source).

**DF-6. Static no-go for L_R.** Static energy density sqrt(1 + S) - 1 + V with V = |u|^2/2. Derrick: scaling u(x) -> u(x/lambda) gives dE/d lambda at 1 equal to the integral of g(S) + 3 integral V, with g(S) = 3(s - 1) - S/s, s = sqrt(1 + S), equal to (2s - 1)(s - 1)/s >= 0. Both terms non-negative, so no nonzero static finite-energy solution in 3D, for any number of components. Radial scalar: see clause (a). The October argument is the same with the sign of dW/dr flipped, and V(phi_0) >= 0 against W(infinity) = 0 again forces phi_0 = 0.

**DF-7. Truncation at X^2.** L_T = X_A + X_A^2/2 (Amin variables), T_X = 1 + X_A, T_XX = 1. Hyperbolic iff (1 + 3X_A)/(1 + X_A) > 0; static gradient X_A = -S/2 gives the windows in section 2. Static energy -L = S/2 - S^2/8. Checked at nine values of S.

**DF-8. Hedgehog vs bump, directional vs spinning, in a three-scalar model.** Real u = Re(A e^{-i omega t}): time average of |u|^4 = (1/4)(|A|^4 + (1/2)|A.A|^2). Linear polarization (A.A = |A|^2): (3/8)|A|^4. Circular (A.A = 0): (1/4)|A|^4. Ratio 3:2, equal to the ratio of the (3/4) and 1 cubic coefficients in the thesis profile equations (3.96, 3.97) and to the footnote of the paper. In the nonrelativistic pure-quartic problem E at the stationary point scales as 1/(g^2 N) (derived by scaling the profile at fixed N), so E_d/E_s = (2/3)^2 = 4/9, as in the paper. The hedgehog u = f r-hat is linearly polarized at every point (|u|^4 averages to (3/8) f^4 as for the bump) with gradient energy f_r^2 + 2f^2/r^2 >= f_r^2 for the same f, so at equal N its energy is at least the bump's. The thesis ratio 73.4/12.6 = 5.8 equals the ratio of profile integrals for the two printed equations (110.1 against 18.9 for unit cubic coefficient). The printed 5.01 for directional is inconsistent with this; see 3.3.

**DF-9. F4 against the family relations.** E/omega = N (E/(omega N)). Along the beta = 4/3 family N(0.85) = 525, N(0.94) = 96.2, E/(mN) about 0.90 at 0.85 (Li) and about 0.97 at 0.94 (interpolated, +/-0.03): d ln N/d ln omega = -16.9; change in ln E = -1.62; change in ln(E/omega) = -1.72; ratio 1.06. The rule in `SESSION_05_PREDICTIONS.md` 2.4 needs at most 0.2. The ratio is small only where dN/d omega is small, near omega_death.

---

## 7. Corrections to the plan's literature column and to the predictions file

1. **F1 basis.** Amin 2013 supports existence of a small-amplitude quasi-breather for L_R (Delta = 1/2 > 0). It does not support trapping, and its stability criterion says the 3D small-amplitude family is unstable. Rewrite the basis as "existence only".
2. **F3 basis.** The hedgehog energy statement in the paper is one sentence with no data; the numbers are in the first author's thesis (arXiv:2401.00043, sec. 3.5.2), one of which looks misprinted (3.3). Their potential is beta = 10/3 in the plan's convention.
3. **Plan section 4, "valid for X < 1".** Sufficient for the bump; not for the pinch (G1, DF-5). WP-B2 check 6 should name the pinch condition.
4. **Amin's X** is (1/2)(d phi)^2 (mostly plus); the plan's X is twice that, and Amin's DBI "2 f X" is our X for f = 1.
5. **LESSONS item 6.** Add that the static no-go clause holds for L_R too (DF-6) and is not a reason to doubt R; the other three clauses do not transfer.
6. **O9, second sentence.** "The X guard cannot trip for finite p": true as algebra, but p can diverge in finite time in the massless equation (G2). Keep the guard.
7. **F4.** The plan names it "new measurement". The sourced relations predict it fails away from omega_death (F4 row, DF-9). The orchestrator already says this in O8.
8. **Model and effort.** The CLAUDE.md table lists `scholar` as Sonnet 5.5, effort xhigh. Here the model reports as `claude-sonnet-5-5` and the effort setting shown to me is the number 30. I cannot map that to the label "xhigh". Flag if the table is meant strictly.

---

## 8. Novelty check for statements this brief might lead to

| Statement | Searches | What they returned | Wording supported |
|---|---|---|---|
| "DBI-plus-mass oscillons: measured lifetimes in 3+1" | S5F-1, 2, 9, 11, 16 | Amin 2013 (existence, no lifetimes; large-amplitude DBI "in progress"); Sang and Huang 2021 (formation in preheating, no lifetimes); nothing else | "No published lifetime measurement of a DBI-plus-mass oscillon was found; Amin 2013 refers to unpublished large-amplitude simulations." Not "first". |
| "Hedgehog oscillon in a three-component real scalar model" | S5F-12, 13, 14, 15 | Proca hedgehog (thesis); gravity-bound Proca stars; van Dissel and Sfakianakis (no l = 1) | "No source found for a scalar-triplet hedgehog oscillon." Not "new": the Proca hedgehog is the closest relative. |
| "For a vector-valued Born-Infeld scalar, X < 1 does not guarantee hyperbolicity" | S5F-17, 18 | Single-field k-essence and Born-Infeld only (Babichev et al., Eperon et al., Mukohyama et al.); no multi-field characteristic analysis found | "Derived here (DF-5), not found in the sources read." Do not call it new without a search of the multi-field k-essence literature. |
| "Finite-time X -> 1 singularities with a mass term" | S5F-7, 18 | massless results only (Eggers and Hoppe; Eperon et al.; Yan) | "No source found for the massive case." |

---

## 9. Search log (S5F-n; web search unless stated; 2026-11-10)

1. k-oscillons DBI large amplitude Pearce Amin non-canonical lifetime: Amin pdf, Zhang et al. 2020, galileon oscillons; no Pearce-Amin paper.
2. oscillons in scalar field with non-canonical kinetic term, Born-Infeld, mass term, simulation: Sang and Huang 2012.14697.
3. Zhang Jain Amin polarized/vector oscillons hedgehog: 2111.08700; Jain and Amin 2109.04892.
4. Born-Infeld "completely exceptional" scalar shock formation Nambu-Goto: Kadlecova 2107.12249 and BI electrodynamics papers; Christodoulou only second-hand.
5. Lindblad minimal surface equation Minkowskian: math/0210056; Brendle, Allen-Andersson-Isenberg, Ai-Ifrim-Tataru as citing work; not opened.
6. Eggers Hoppe singularity formation timelike extremal hypersurfaces: 0812.1790 (arXiv title "M-brane singularity formation"; journal PLB 680, 274, from the search result); Eggers, Hoppe, Hynek, Suramlishvili 2015 not opened.
7. Born-Infeld scalar 1+1 blow-up finite time smooth data: Eperon et al.; Yan; Alejo and Munoz; Felder et al.
8. Born-Infeld scalar potential soliton Derrick no static solutions: Diaz-Alonso and Rubiera-Garcia; Endlich et al.; BPS solitons in DBI (1402.0019, not opened).
9. oscillons P(X, phi) k-essence 3+1 lattice "k-oscillon" (extended): no k-essence oscillon paper beyond Amin.
10. oscillons Born-Infeld/DBI breathers 1+1 kink-antikink noncanonical: DBI solitons (BPS, kinks), no oscillons.
11. "K-oscillons" Amin 2013 cited, follow-up: Sakstein and Trodden; Zhang et al. 2020 citing it.
12. hedgehog oscillon multi-component scalar O(3) l = 1 phi^6: two-field phi^6 (1907.07145), van Dissel and Sfakianakis, review 2411.16604; no hedgehog.
13. "Symmetric multifield oscillons" 2010.07789: found, read (part).
14. Loginov nontopological solitons self-interacting complex vector field: 2015 paper and "electromagnetic hedgehogs" (2405.01335); not opened.
15. hedgehog oscillon real vector field / O(3) triplet (extended): led to the thesis 2401.00043 (hedgehog section found by text search after extraction).
16. DBI scalar oscillon lifetime large amplitude 3+1 2022 to 2025 (extended): nothing new.
17. multi-field k-essence hyperbolicity characteristics: 2008.07546, 1605.06418, 1602.00735, 2003.13730, 2412.21169; none multi-field Born-Infeld.
18. Born-Infeld scalar with mass term Klein-Gordon well-posedness blow-up: 1807.04227, 1707.02595; nothing for the massive scalar.

Fetches that were read in full or in part (arXiv PDF, text extracted locally): 1303.1102, 2111.08700, PRD 105, 096037 (accepted manuscript), 2401.00043 (thesis, hedgehog and vector-oscillon sections), 0708.0561, hep-th/0602178, 1802.03020, 0812.1790, hep-th/0208019 (intro), 2107.12249 (intro), 2010.07789 (model section). Abstract pages only: 1002.4873, 1107.4370, 0705.0112, 1602.00735, 1605.06418, 1807.04227, 1707.02595, math/0210056, 1809.07724, 2101.07265, 1508.05395, 2008.07546, 2006.09053.

---

## 10. Not obtained, not read

- Pearce and Amin, large-amplitude DBI oscillons: not found; may not exist in print.
- Boillat 1970, Whitham 1974, Barbashov and Chernikov 1966 and 1967, Taniuti 1959, Christodoulou (scalar Born-Infeld, 3D): no arXiv text; claims about them are `SECONDHAND` through Eperon et al. and Kadlecova.
- Kato and Hughes-Kato-Marsden (local well-posedness of quasilinear hyperbolic systems): not read.
- Lindblad 2004, Brendle 2002, Allen-Andersson-Isenberg: abstract of Lindblad only.
- Yan 2018 and Alejo and Munoz 2017: abstracts only; I did not check what smoothness "blow-up" refers to there.
- Felder, Kofman, Starobinsky: only the introduction and abstract; the 3+1 section was not read.
- Sang and Huang 2021: fetch-tool summary only (PDF over the 10 MB limit); amplitude, lifetime and frequency are not reported there per that summary, not verified against the text.
- Loginov 2015, Sanchis-Gual et al. 2017 and 2019, Bustillo et al. 2021, Jain and Amin 2109.04892 (abstract only): not opened.
- Thesis Fig. 3.15 (the hedgehog decay-rate figure) not examined; the numbers 1e6/m and omega 0.976 to 0.986 are from the text and axis labels. The printed energies 5.01, 18.9, 73.4 per eps: see 3.3.
- Fodor et al. 2008, Amin and Shirokoff, Hertzberg, Li et al., Mukaida et al., Ibe et al.: relied on the Session 1 brief; not re-opened.
- Multi-field Born-Infeld or k-essence hyperbolicity analysis: no source found.
- d'Avenia and collaborators on Klein-Gordon-Born-Infeld systems (surfaced in a search snippet): not opened; may concern coupling to Born-Infeld electrodynamics rather than the scalar.
- Whether F2 holds for L_R, whether the 3D k-oscillon family has a stable branch, and lifetimes of the pinch in L_O or L_R: no source found; these are what the runs measure.
