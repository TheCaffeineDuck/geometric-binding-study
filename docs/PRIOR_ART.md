# Prior art

Maintained by the `scholar` worker. One section per brief, newest last. The orchestrator reads the brief before the parameter pick.

**Evidence tags.** Every claim about the literature carries a link and one tag.

- `READ`: I opened the source (full text, extracted locally) and it says this. A tag like `READ (figure)` means I read a number off a plot, with the reading error stated.
- `ABSTRACT`: I saw only the abstract.
- `SECONDHAND`: another source says the paper says this.
- `DERIVED`: my own algebra from the plan's equations or from numbers in a `READ` source. Not a literature claim. Re-derive before relying on it.
- "no source found": I searched and found nothing.

Numbers are quoted with their parameters. A number from a different potential, dimension or amplitude regime is context, not a prediction for this project, and says so.

---

## Session 1 / Steps 1-2 brief

Scope: WP1L. Also covers the Step 3 and Step 4 predictions on record. Model and notation as in `SWIRL_PLAN_2026-10-01.md` section 3. Date: 2026-10-01. No git commands were run; only `docs/` was written.

### 0. Headline findings

1. **The stability sign in the plan is right, and it applies to real-scalar oscillons.** The stable branch is dN/domega < 0 (lower omega, larger amplitude, N rising as omega falls). Leading-order derivation for real fields: [Amin & Shirokoff 2010](https://arxiv.org/abs/1002.3380) `READ`. Used for real oscillons with lattice confirmation: [Ibe et al. 2019](https://arxiv.org/abs/1901.06130) `READ`, [Zhang et al. 2020](https://arxiv.org/abs/2004.01202) `READ`, [Li, Yamaguchi, Zhang 2025](https://arxiv.org/abs/2507.13276) `READ`. It is a criterion for the radiation-free approximation. The real oscillon still radiates and walks toward N_min, where it dies ("energetic death").
2. **The plan's Set B prior-estimate table is consistent with independent literature.** The single-frequency family is a one-parameter universal curve (section 1.3, `DERIVED`). Built from three independent numbers in Li et al. 2025 (beta = 4/3), it reproduces the table's N at the minimum to 1.5%, the energy relation dE = omega dN to about 2% per segment, and the omega = 0.99 row to the size of the expected sextic correction (section 3). Step 1 should reproduce the table to a few percent. The one soft spot: the table's minimum is at about 0.922 by sampling; the literature puts the N-minimum at alpha^2/alpha_c^2 of about 0.14, which is omega of about 0.933 at beta = 0.611.
3. **The plan's "other branch does not stay localized" is only half right.** In the 3D cubic-quintic NLS (the non-relativistic limit of this model), a perturbed unstable-branch soliton disperses if the perturbation lowers N and relaxes onto the stable branch, shedding radiation, if it raises N: [Carles, Klein, Sparber 2021](https://arxiv.org/abs/2012.11637) `READ`. Pure phi^4 collapses instead ([Hertzberg 2010](https://arxiv.org/abs/1003.3459) `READ`). So the radial check on the unstable side can legitimately return either outcome. Say so before running.
4. **A published, directly comparable point exists at beta = 4/3** (m = 1, g4 = 6, g6 = 48): N_min = 80.3 at omega = 0.97, N(0.94) = 96.2, N(0.85) = 525, lifetimes about 1e4 to 1e5, radiation rate Gamma = |dE/dt|/E between about 1e-7 (at dips) and 4e-4 (at peaks). Three papers agree (section 2.4). I recommend adding beta = 4/3 as a Step 1 validation row. There is no published 3D data at beta = 0.70 or below 1.
5. **Radiation rates at generic omega can break the Step 2 "Go" target by themselves.** At beta = 4/3, Gamma peaks near 1e-4 to 4e-4. Over T = 1000 that is 10% to 33% energy loss with no failure of localization. Only near the rate dips does the retention exceed 90% comfortably. A miss of the 90% target near a radiation peak is expected physics, not a no-go (section 7).
6. **Step 3 sign: the plan agrees with every primary calculation and simulation I read, except one verbal statement.** In-phase pairs attract and merge, anti-phase repel (Q-balls in 1D/2D/3D, non-relativistic solitons, 3+1D real oscillons). One 2026 paper states the opposite sign in words while its own formula and my re-derivation say in-phase attracts (section 4.2). An uncited Wikipedia sentence claims the reverse; the only primary source I found with that sign is a granular-media model, a different system. I could not trace the sentence itself.
7. **The plan's profile equation drops the third-harmonic back-reaction, which changes the effective sextic coefficient by a factor (1 + 1/(8 beta))** at small amplitude (12.5% at beta = 1, 18% at 0.70, 20% at 0.611; `DERIVED` from [Fodor et al. 2008](https://arxiv.org/abs/0802.3525) `READ` eq. 25 and [Mukaida et al. 2017](https://arxiv.org/abs/1612.07750) `READ` eq. 2.17). Expect PDE-versus-profile mismatches of that order before any other effect.
8. **The plan's suggested window E/(mN) = 0.90-0.98 selects large-amplitude, flat-top-leaning profiles.** The literature stability criterion is not E < mN. Li et al. find E/(mN) of about 1.0 at the N-minimum and about 0.9 at omega = 0.85 for beta = 4/3. Large amplitude is where single-frequency profiles are least reliable and where lattice-noise-seeded 3D fragmentation was seen (in sine-Gordon; section 6).

### 1. Conventions and dictionary

#### 1.1 Parameter dictionary (m = 1; beta = g6 m^2 / g4^2)

| Source | Potential | Map to beta | Value used |
|---|---|---|---|
| Plan | m^2 phi^2/2 - g4 phi^4/24 + g6 phi^6/720 | definition | Set B 0.611 |
| Amin & Shirokoff 2010 | phi^2/2 - lambda phi^4/4 + g phi^6/6 (rescaled so lambda = 1) | g_AS = 3 beta/10 | g = 1 gives 10/3 |
| Mukaida 2017, Ibe 2019 | -g4 phi^4/4 + g6 phi^6/6 (their g4, g6) | beta = (10/3) g6/g4^2 | g4 = 1, g6 = 0.4 gives 4/3 |
| Li et al. 2025 | -lambda phi^4 + g6 phi^6 | beta = 1.25 g6/lambda^2 | m^2 g6/lambda^2 = 16/15 gives 4/3 |
| Kasuya et al. 2003 | -a phi^4/4 + b phi^6/(6 m^2), a = 0.1, b = 0.005 | beta = (10/3) b/a^2 | 1.67 (2D lattice) |
| Fodor et al. 2009 (2D) | phi^2/2 - phi^4/4 + phi^6/6 | as A&S | 10/3 (2D) |
| Complex phi^6 (Battye-Sutcliffe; Canillas Martinez et al. 2025) | \|phi\|^2 - \|phi\|^4 + beta_t \|phi\|^6 | beta = 5 beta_t / 2 | beta_t >= 1/4 is beta >= 5/8 |

The three beta = 4/3 papers (Mukaida, Ibe, Li) are the same potential. I checked the mapping two ways: g4 = 6 lambda, g6 = 720 g6_Li gives 4/3, and Ibe's g4 = -3!, g6 = 0.4 x 5! gives 4/3. Their adiabatic invariant I is the plan's N: I = (1/omega) integral of the period-averaged phi_t^2 equals (omega/2) integral of Phi^2 for phi = Phi cos(omega t) (Ibe eq. 23, Li eq. 2.16 `READ`).

Conversion to the plan's rescaled N: g4 N / m = 24 (lambda I) with lambda = g4/24. For Ibe/Mukaida units (g4_theirs = 1), g4 N = 6 Q.

#### 1.2 Vacuum structure (`DERIVED`)

- V' = 0 has nonzero roots iff beta <= 5/6. V(phi_v) >= 0 iff beta >= 5/8. So for 5/8 <= beta <= 5/6 the potential has a false vacuum at phi_v^2 = (10 / (beta g4)) (1 + sqrt(1 - 1.2 beta)), m = 1. At beta = 5/8 it is degenerate with phi = 0.
- The complex phi^6 paper finds a false vacuum for beta_t in (1/4, 1/3) ([Canillas Martinez et al. 2025](https://arxiv.org/abs/2509.03192) `READ`), which is beta in (5/8, 5/6). Same window.
- Set B (g4 = 0.30, g6 = 0.055, beta = 0.611): V is negative for 8.34 < |phi| < 9.70, with minimum about -0.9. The plan's prior-table rows with Phi(0) = 9.7 (omega = 0.837) and 10.5 (omega = 0.707) put the core at or past that window.
- beta = 0.70 with g4 = 0.3: false vacuum at |phi| = 8.2 with V = 3.7. Rescaled: g4 phi_v^2 = 20.0.
- Legacy Gaussian check: the first inflection point of V (V'' = 0) in Set B is at |phi| = 2.74. The legacy amplitude 0.5 is 5.5 times below it.

#### 1.3 The single-frequency family is universal (`DERIVED`)

The plan's profile equation, with Phi = a Psi(rho), r = L rho, a^2 = 20/(beta g4), L^2 = 0.3 beta, becomes

    Psi'' + (2/rho) Psi' - alpha^2 Psi + (3/4) Psi^3 - (5/8) Psi^5 = 0,    alpha^2 = 0.3 beta (1 - omega^2).

This is exactly eq. (28) of [Amin & Shirokoff 2010](https://arxiv.org/abs/1002.3380) `READ`, with their g = 0.3 beta. Consequences:

- Every beta gives the same curve N_AS(alpha^2) = integral of Psi^2 d^3rho, reparametrized. N follows from g4 N / m = 1.6432 omega sqrt(beta) N_AS(alpha^2). This is a free Step 1 consistency test: results for beta = 0.611, 0.625, 0.70, 0.85, 1.0 must collapse onto one curve.
- Localized solutions exist only for alpha^2 <= alpha_c^2 = 27/160 = 0.16875 (A&S print "alpha_c = 27/160", but their eq. 16 maximum gives alpha_c^2 = 27/160; I checked). Equivalent: omega >= omega_flat with omega_flat^2 = m^2 (1 - 9/(16 beta)). Values: beta = 0.611: 0.282; 0.625: 0.316; 0.70: 0.443; 0.85: 0.582; 1.0: 0.661; 4/3: 0.760. The family has no lower omega limit for beta < 9/16.
- The complex Q-ball threshold is omega_min^2 = m^2 (1 - 5/(8 beta)) (from Coleman's U/f^2 rule as summarized in [Battye & Sutcliffe 2000](https://arxiv.org/abs/hep-th/0003252) `READ`, and the closed form in [Canillas Martinez et al. 2025](https://arxiv.org/abs/2509.03192) `READ`: omega_min^2 = 1 - 1/(4 beta_t)). It reaches zero exactly at beta = 5/8. For the real oscillating field the single-frequency flat-top limit is 9/(16 beta), because cos^4 and cos^6 average to less than 1. So "bubble-like at 5/8" is the complex-field statement. The real-field single-frequency family reaches its thin-wall end at beta = 9/16 = 0.5625.

Universal-curve points (alpha^2, N_AS, Psi0 = Phi(0) sqrt(beta g4/20)). Rows marked P are derived from the plan's Set B table. Rows marked L are derived from the numbers in [Li et al. 2025](https://arxiv.org/abs/2507.13276) `READ`, lambda I = 24.04 at omega = 0.94, 131.32 at omega = 0.85, 20.07 at omega_death = 0.97, and amplitudes read from their Fig. 1 (error about 0.02 in sqrt(lambda) psi(0)).

| alpha^2 | N_AS | Psi0 | from |
|---|---|---|---|
| 0.00365 | 447 | 0.297 | P, omega = 0.99 |
| 0.0130 | 288 | 0.536 | P, omega = 0.964 |
| 0.0236 | 262 | about 0.68 | L, omega = 0.97 (N-minimum) |
| 0.0275 | 269 | 0.737 | P, omega = 0.922 |
| 0.0466 | 324 | 0.885 | L, omega = 0.94 |
| 0.0549 | 372 | 0.929 | P, omega = 0.837 |
| 0.0917 | 924 | 1.005 | P, omega = 0.707 |
| 0.111 | 1954 | 1.012 | L, omega = 0.85 |
| 0.16875 | diverges | 0.949 | flat-top limit (A&S: Phi_c^2 = 9/10) |

P and L rows interleave monotonically. A&S note that in 3D the flat-top limit is approached from above in amplitude, consistent with Psi0 peaking near 1.01 and falling to 0.949. The small-alpha limit N_AS = 25.2/alpha (from the NLS ground state, `DERIVED`) gives 417 at alpha^2 = 0.00365, 7% below the P row, as expected from the sextic correction.

Internal check of the plan's table (`DERIVED`): the Q-ball relation dE/dN = omega ([Mukaida 2017](https://arxiv.org/abs/1612.07750) `READ` eq. 2.26; [Ibe 2019](https://arxiv.org/abs/1901.06130) `READ` eq. 21) gives, between table rows, Delta E / Delta N = 0.982 (omega 0.99 to 0.964), 0.952 (0.964 to 0.922), 0.870 (0.922 to 0.837), 0.758 (0.837 to 0.707). All lie within about 2% of the mid-segment omega (0.977, 0.943, 0.880, 0.772), which is the expected size for a nonlinear N(omega).

### 2. Existence, stability, minimum size (Step 1)

#### 2.1 Small-amplitude theory and the NLS peak

- Small-amplitude quasi-breathers in D spatial dimensions: phi = eps phi_1 + ..., with omega^2 = 1 - eps^2 (verified from the paper's own numbers: eps = 0.1478787 for omega = 0.9890055), spatial scale eps x, and leading profile p1 = S / sqrt(lambda_F), where S solves Delta S - S + S^3 = 0 and lambda_F = (5/6) g2^2 - (3/4) g3. Existence needs lambda_F > 0. No small-amplitude quasi-breathers exist for D >= 4 ([Fodor et al. 2008](https://arxiv.org/abs/0802.3525) `READ`, eqs. 19-21, 40-43).
- For our potential (g2 = 0, g3 = -g4/6 in their U' = phi + sum g_k phi^k), lambda_F = g4/8 (m = 1). This matches the plan's Phi^3 coefficient.
- 3D ground state: S(0) = 4.33738768 (Table I, same paper `READ`). The plan's check 1 (4.3374, tolerance 1e-3) is correct. The plan's scaling identity (check 2) matches their virial identities, eqs. 41-43 `READ`.
- 3D energy: E = 6.29908/eps + 264.262 eps for the phi^4 double well (lambda_F = 3/2), eq. 54 `READ`. The leading coefficient gives integral S^2 = 18.897 (`DERIVED`: 2 lambda_F times 6.29908). The 1/eps divergence as eps -> 0 means E has a minimum at finite omega in D = 3. The small-eps estimate eps_m = 0.154 is stated by the authors to be too large to trust.
- Leading-order dictionary for the plan's model (`DERIVED`): Phi(0) = 4.33739 sqrt(8 (m^2 - omega^2) / g4); g4 N / m = 75.59 omega / sqrt(1 - omega^2); E = N / omega at this order. Validity: eps = sqrt(1 - omega^2) below about 0.25 ([Fodor 2008](https://arxiv.org/abs/0802.3525) `READ`, sec. IV A). The plan's omega = 0.964 row (eps = 0.266) is outside that range.
- Third-harmonic back-reaction. Fodor et al. eq. 25 with g2 = 0 gives the effective quintic coefficient (lambda_F^2/24) + (5/8) g5, with g5 = g6/120. Relative to the plan's single-frequency coefficient g6/192 this is a factor (1 + g4^2/(8 g6)) = (1 + 1/(8 beta)) (`DERIVED`). [Mukaida et al.](https://arxiv.org/abs/1612.07750) `READ` eq. 2.17 has the same extra term, of size g4^2/(128 m^2) relative to 5 g6/24. Equivalent statement: the amplitude equation sees beta_eff = beta + 1/8. Note the numerical profiles in Ibe 2019 (sec. III B) and Li 2025 (eq. 2.20) use the single-frequency equation and agree with each other, so published beta = 4/3 numbers do not include this correction.
- Even with g5 = 0, the third harmonic supplies a repulsive quintic term lambda_F^2/24 (`DERIVED` reading of eq. 25). This is consistent with pure phi^4 double-well oscillons having a stable branch at moderate amplitude.

#### 2.2 The stability criterion

| Source | System | Statement | Tag |
|---|---|---|---|
| [Amin & Shirokoff 2010](https://arxiv.org/abs/1002.3380) | real field, phi^4-phi^6, 1+1 and 3+1, leading order in 1/g | long-wavelength linear stability (localized perturbations, radiation ignored) holds iff dN/d(alpha^2) > 0, alpha^2 = g(1 - omega^2); proof via Vakhitov-Kolokolov; 3+1 uses the radial operator; in 3+1 small-amplitude oscillons are unstable, large-amplitude ones robust | READ |
| [Zhang et al. 2020](https://arxiv.org/abs/2004.01202) | real field, general | N = (omega/2) integral 4 pi r^2 Phi^2 dr; need dN/domega < 0; dN/domega = 0 at omega_crit | READ (eq. 3.7-3.8) |
| [Mukaida et al. 2017](https://arxiv.org/abs/1612.07750) | real field via NR effective theory | stable iff d omega / dQ < 0 (eq. 2.29); the branch violating it decays immediately in their simulations | READ |
| [Li et al. 2025](https://arxiv.org/abs/2507.13276) | real phi^4-phi^6, 3D | omega_death defined by dE/domega = 0; between omega_death and m the oscillon is unstable; same as VK | READ |
| [Canillas Martinez et al. 2025](https://arxiv.org/abs/2509.03192) | complex phi^6 Q-ball, 1+1 | stable iff omega dQ/domega < 0 | READ |
| [Carles, Klein, Sparber 2021](https://arxiv.org/abs/2012.11637) | 3D cubic-quintic NLS | stable iff dM/d(omega_NLS) > 0; U-shaped M; unstable near the threshold, stable on the dense side; omega_NLS in (0, 3/16), N-minimum at omega_crit about 0.026 | READ |
| [Hertzberg 2010](https://arxiv.org/abs/1003.3459) | real, 3D | oscillon stable (unstable) to collapse if d M_osc / d amplitude is positive (negative); cites A&S and Fodor et al. 2008 for this; beyond a critical amplitude no collapse instability in 3D | READ |

Sign translation. The plan's dN/domega < 0 equals A&S's dN/d(alpha^2) > 0 (alpha^2 grows as omega falls) equals Mukaida's d omega/dQ < 0 equals CKS's dM/d omega_NLS > 0. All agree.

Does it apply to real scalars? Yes at the level of "approximate adiabatic invariant", with these qualifications.
- Real-field VK is derived only at leading order in 1/g with radiation dropped (A&S) or through an approximate U(1) (Mukaida).
- Checked on the lattice at beta = 4/3: the oscillon decays rapidly once its charge falls to Q_cr = 10^1.9 (about 80), at about 1e4 to 2e4 ([Ibe 2019](https://arxiv.org/abs/1901.06130) `READ`, Fig. 6).
- In the phi^4 double well, the same picture holds with omega_c about 0.967 in units of m ([Fodor 2008](https://arxiv.org/abs/0802.3525) `READ`, sec. IV A): stable below, one unstable mode above, minimum energy near omega_c.
- Short-wavelength (Floquet) instabilities exist beyond VK ([A&S](https://arxiv.org/abs/1002.3380) `READ`: narrowest bands at k near sqrt(3); [Zhang et al.](https://arxiv.org/abs/2004.01202) `READ`: found analytically, fate unclear). Parity-violating phi^4 quasi-breathers are linearly unstable but nonlinearly stabilized ([Cyncynates & Giurgica-Tiron 2021](https://arxiv.org/abs/2104.02069) `READ`).
- Non-radial (l >= 1) modes: [Cyncynates & Giurgica-Tiron](https://arxiv.org/abs/2104.02069) `READ` (app. C) find higher l generally more stable, low l case-dependent. [Adib, Gleiser, Almeida 2002](https://arxiv.org/abs/hep-th/0203072) `ABSTRACT`: 2+1D phi^4 oscillons linearly stable to small radial and angular perturbations. [Hindmarsh & Salmi 2006](https://arxiv.org/abs/hep-th/0606016) `ABSTRACT`: elliptical perturbations of 2D oscillons decay. [van Dissel, Pujolas, Sfakianakis 2023](https://arxiv.org/abs/2303.16072) `READ`: sine-Gordon states with radial nodes fragment in 3D; nodeless states do not. No source found for non-radial stability of the sextic family in 3D.

#### 2.3 The sextic term, and pure phi^4

- Pure attractive phi^4 in 3D: the small-amplitude branch has collapse instability. [Hertzberg 2010](https://arxiv.org/abs/1003.3459) `READ` finds exponentially growing modes at k of order eps in d = 3 (not d = 1) for -lambda phi^4, and shows an example: a profile 3% above the preferred amplitude at eps = 0.05 collapses by t about 150. NR argument: an attractive delta-function pair potential is unbounded for d >= 2, collapse time naively 1/(lambda eps^2).
- Mathematical analogue: focusing cubic NLS has solitons in every dimension, is orbitally unstable in d = 3, and blows up in finite time ([Carles, Klein, Sparber 2021](https://arxiv.org/abs/2012.11637) `READ`, sec. 2.2.1). A defocusing quintic term prevents finite-time blow-up in d = 2, 3 (their Prop. 1.1, as stated in the intro `READ`).
- Role of the sextic: Kasuya et al. found no I-balls for a repulsive quartic (negative a) and added the phi^6 term to keep the vacuum stable; their simulations used a = 0.1, b = 0.005 ([Kasuya, Kawasaki, Takahashi 2003](https://arxiv.org/abs/hep-ph/0209358) `READ`). Hertzberg states that beyond a critical amplitude no collapse instability exists in 3D (`READ`, section VIII). In the phi^4-phi^6 model the sextic term creates the dense stable branch and the flat-top limit (A&S, CKS).
- Answer for pure attractive phi^4 without a sextic term in 3D: the small-amplitude branch collapses (Hertzberg `READ`), and the potential is unbounded below, so there is no bounded-potential model without the sextic. The third harmonic adds a small repulsive quintic term (section 2.1), which may support moderate-amplitude states in the double-well case; no source tested pure symmetric -phi^4 beyond Hertzberg's small-amplitude run. With the plan's sextic, a stable branch exists for every beta > 0 in the single-frequency approximation (`DERIVED`: the family ends at alpha_c^2 or at omega -> 0).

#### 2.4 Minimum N, omega and energy: published numbers

All at (m = 1) unless stated; `READ` unless stated.

| Potential | Quantity | Value | Source |
|---|---|---|---|
| phi^4-phi^6, beta = 4/3, 3D radial | N at omega_death | lambda N = 20.07, so N = 80.3 for lambda = 1/4 (the Ibe/Mukaida g4 = 1 normalization); g4 N = 481.7 | [Li 2025](https://arxiv.org/abs/2507.13276) |
| same | omega_death | 0.97 | Li 2025; Ibe Fig. 2 `READ (figure)` about 0.97 |
| same | Q_cr | 10^1.9 about 80 (Ibe), 10^1.91 (Mukaida) | [Ibe](https://arxiv.org/abs/1901.06130), [Mukaida](https://arxiv.org/abs/1612.07750) |
| same | N at omega = 0.94, 0.85 | 96.2, 525 (g4 N = 577, 3152) | Li 2025, Fig. 1 legend |
| same | E/(mN) | about 1.0 at omega_death, about 0.9 at 0.85 | Li Fig. 2 `READ (figure)`, +/- 0.03 |
| same | profile | Gaussian-like for omega >= about 0.9; tanh flat-top below; radius (1/e) 3.6 at omega = 0.945, 7.3 at 0.83 | Li Fig. 1 `READ (figure)`; Ibe: deviates from Gaussian at 0.85 |
| same | validity of NR theory | fails for Q above 10^2.5 (omega below about 0.87) | [Ibe](https://arxiv.org/abs/1901.06130) |
| phi^4 double well, 3D | omega_c (energy minimum, stability edge) | 0.9659 (from numerics cited there); stable below 0.967 | [Fodor 2008](https://arxiv.org/abs/0802.3525) |
| same | plateau energy | about 43 m/lambda (Copeland), about 50 (Gleiser), 41.56 asymptote and 42.7 at destabilization (Salmi-Hindmarsh; their units) | [Copeland](https://arxiv.org/abs/hep-ph/9503217), [Gleiser](https://arxiv.org/abs/hep-ph/9308279), [Salmi-Hindmarsh](https://arxiv.org/abs/1201.1934) |
| A&S large-g, 3D | N-minimum position | alpha^2 about 0.025 `READ (figure)`, N_AS about 270 | [A&S Fig. 4](https://arxiv.org/abs/1002.3380) |
| 3D cubic-quintic NLS | N-minimum | omega_crit about 0.026 of omega_max = 3/16, ratio 0.139 | [CKS 2021](https://arxiv.org/abs/2012.11637) |

Normalization caution: Ibe's footnote 5 says their Q0 and psi0 differ from the figure in Mukaida et al. by a factor of two, yet both texts quote the same Q_cr (10^1.9 and 10^1.91). Li's lambda I_death = 20.07 gives 80.3 for g4_Ibe = 1, which matches Ibe's lattice number (decay at Q about 80). I use Li and Ibe, and treat Mukaida's figure normalization as unverified.

Position of the N-minimum is therefore universal: alpha^2 / alpha_c^2 = 0.139 (CKS), 0.140 (Li), about 0.15 (A&S figure). So omega_Nmin = sqrt(1 - 0.0788/beta) (`DERIVED`):

| beta | 0.611 | 0.625 | 0.70 | 0.85 | 1.0 | 4/3 |
|---|---|---|---|---|---|---|
| omega at N-min | 0.933 | 0.935 | 0.942 | 0.953 | 0.960 | 0.970 |
| g4 N_min / m | 314 | 318 | 339 | 378 | 413 | 482 |
| N_min for g4 = 0.3 | 1046 | 1059 | 1130 | 1259 | 1376 | 1606 |

(Using N_AS,min = 261.7 from Li. Expected accuracy 2-3%. The plan's Set B table has 1062 at omega = 0.922, within 1.5% of 1046.) These are expectations for WP1A, not targets to tune toward.

#### 2.5 Large amplitude, flat-top and near-degenerate behaviour

- Flat-top oscillons exist in the real phi^4-phi^6 model: as alpha approaches alpha_c the core amplitude tends to the plateau value Phi_c (Phi_c^2 = 9/10 in A&S's rescaled units) and the width diverges (A&S). In 3D the width is non-monotonic in amplitude and the flat-top limit is approached from above ([A&S](https://arxiv.org/abs/1002.3380) `READ`). A&S's analytic control is the large-g (large-beta) regime. A&S report that flat-tops are more robust against collapse in 3+1D than small-amplitude oscillons, and Hertzberg adds that beyond a critical amplitude no collapse instability exists in 3D (citing A&S and Fodor et al. 2008; [Hertzberg](https://arxiv.org/abs/1003.3459) `READ`). A&S's claim that radiation stays small for flat-tops is an argument, not a computation (their sec. 4.3).
- "Near beta = 5/8 the lumps become bubbles": no source found for real 3D oscillons. Related reads: complex phi^6 thin-wall Q-balls are bound states of a Q-kink and Q-antikink; the false-vacuum bubble is usually short-lived and is stabilized only temporarily, by Goldstone modes ([Canillas Martinez et al. 2025](https://arxiv.org/abs/2509.03192) `READ`, 1+1, complex). 1+1D real false-vacuum oscillons can pass over a sphaleron barrier to the true vacuum if they carry enough energy ([Campos, Manton, Mohammadi 2026](https://arxiv.org/abs/2605.12633) `READ`).
- Large amplitude, many harmonics: the single-frequency ansatz keeps energy to a few percent at central amplitudes up to a few times the potential scale but degrades for decay rates ([Zhang et al. 2020](https://arxiv.org/abs/2004.01202) `READ`, sec. 8). Large-amplitude oscillons in a deformed sine-Gordon model (and by their argument phi^6) decay in staggered steps with bursts of higher harmonics ([Dorey, Romanczukiewicz, Shnir 2019](https://arxiv.org/abs/1910.04128) `ABSTRACT`, 1D).

#### 2.6 Lifetimes and early radiation (units 1/m, all spherical unless noted)

| System | Lifetime | Source |
|---|---|---|
| phi^4 double well, 3D | 1e3-1e4; longest Gaussian bubble 7.4e3 (R0 = 2.86); tanh 4.4e3 | [Copeland et al. 1995](https://arxiv.org/abs/hep-ph/9503217) `READ` |
| same | 6000-7000 (units with threshold sqrt 2) | [Fodor 2006](https://arxiv.org/abs/hep-th/0609023) `READ` |
| same | decay at about 7200; E - E0 falls as t^(-0.65), omega* - omega as t^(-0.20) | [Salmi & Hindmarsh 2012](https://arxiv.org/abs/1201.1934) `READ` |
| same | about 6000 numerics, 5900 predicted | [Cyncynates & Giurgica-Tiron 2021](https://arxiv.org/abs/2104.02069) `READ` |
| same | fine-tuned initial data: lifetime grows without limit near 125 resonances | [Honda & Choptuik 2002](https://arxiv.org/abs/hep-ph/0110065) `READ`; not supported by the classical decay-rate calculation of [Zhang et al.](https://arxiv.org/abs/2004.01202) `READ` |
| sine-Gordon, 3D | about 1e3; two nodeless states, 7e2-9.5e2 | Zhang; [van Dissel et al.](https://arxiv.org/abs/2303.16072) `READ` |
| phi^4-phi^6, beta = 4/3 | Gamma = \|dE/dt\|/E: peak about 1e-4 (Q = 10^2.0), 4e-4 (10^2.4); dips to about 1e-6 to 1e-7 near Q = 10^2.14 and 10^2.56; lifetime about 1e4 for Q about 10^2.1; Mukaida runs to T = 1e5 without death (R_ini = 10) | [Ibe 2019](https://arxiv.org/abs/1901.06130) `READ (figure)`; [Mukaida 2017](https://arxiv.org/abs/1612.07750) `READ` |
| phi^4-phi^6, beta = 1.67, 2D lattice, from noise | O(1e3) | [Kasuya et al. 2003](https://arxiv.org/abs/hep-ph/0209358) `READ` |
| flattened potentials | 1e8-1e9 | [Zhang](https://arxiv.org/abs/2004.01202), [Olle, Pujolas, Rompineve 2020](https://arxiv.org/abs/1906.06352) `READ` |
| cosine, 3D | 1e3 | Zhang; Olle |

Mechanism: the decay is slow because the radiating harmonics (3 omega, 5 omega for symmetric potentials) sit at wavenumbers far above the inverse core size. Decay rates are non-monotonic in omega with dips where the leading harmonic's source vanishes ([Zhang 2020](https://arxiv.org/abs/2004.01202) `READ`; [Ibe 2019](https://arxiv.org/abs/1901.06130) `READ`; [Cyncynates & Giurgica-Tiron](https://arxiv.org/abs/2104.02069) `READ`). At the dips the 5 omega channel sets the rate (Zhang; Mukaida spectrum `READ`). For small amplitude, dE/dt = -A_D eps^(1-D) exp(-B_D/eps) ([Fodor et al. 2009](https://arxiv.org/abs/0903.0953) `READ`; eq. 1, 45). The calculation overestimates measured decay rates at large amplitude ([Zhang](https://arxiv.org/abs/2004.01202) `READ`).

Early radiation. No source found for a universal early-shedding fraction. Examples: phi^4 double-well bubbles just above the threshold energy (E0 at least about 60 m/lambda) settle to a plateau of about 43-50 m/lambda, i.e. they shed at least roughly 17-28% ([Gleiser 1994](https://arxiv.org/abs/hep-ph/9308279), [Copeland 1995](https://arxiv.org/abs/hep-ph/9503217) `READ`; my arithmetic on their numbers); a large fraction is radiated right after bubble collapse, with an abrupt initial drop in a Gaussian-started 3D run ([Honda & Choptuik](https://arxiv.org/abs/hep-ph/0110065), [Salmi & Hindmarsh](https://arxiv.org/abs/1201.1934) `READ`, no number given). Profile-matched initial data shed little: in Ibe's beta = 4/3 run at omega_ini = 0.910 the energy and charge curves are flat on a log axis from t = 100 (about 140 and 130) until the slow decline (`READ (figure)`, no more precise than 10%). A profile with phi_t = 0 is not the quasi-breather: it relaxes onto an oscillon with higher harmonics added ([van Dissel et al.](https://arxiv.org/abs/2303.16072) `READ`). The size of that initial shedding for the sextic model: no source found; my expectation is a few percent to 10%.

Collision merger loss: two in-phase oscillons merge with E_final = 0.7 (E1 + E2) in a 3+1D axion-like model with omega about 0.82 ([Amin & Mou 2020](https://arxiv.org/abs/2009.11337) `READ`). Context only: different potential.

#### 2.7 Q-ball background

- Existence: omega in (omega_-, omega_+), with omega_-^2 = 2 U(f0)/f0^2 at the minimum of U/f^2; large Q is stable via the thin-wall limit; for small Q the result depends on the potential ([Battye & Sutcliffe 2000](https://arxiv.org/abs/hep-th/0003252) `READ`, which reports Coleman's results). Coleman 1985, Lee & Pang 1992, Friedberg-Lee-Sirlin 1976 and Vakhitov-Kolokolov 1973 have no arXiv text; not read.
- Oscillons are the real-field projection of the Q-ball: the NR effective theory has an approximate U(1) and the profile follows from the bounce equation ([Mukaida et al. 2017](https://arxiv.org/abs/1612.07750) `READ`). Energetic death is the end of the stable branch.
- 3D cubic-quintic NLS stable branch: perturbations of a stable ground state relax back (L-infinity norm settles within t = 15, error 1e-4) ([CKS 2021](https://arxiv.org/abs/2012.11637) `READ`).

### 3. Expected numbers for the plan's model

| Quantity | Value | Parameters | Basis | Note |
|---|---|---|---|---|
| NLS peak, 3D | 4.33738768 | Delta S - S + S^3 = 0 | Fodor 2008 Table I `READ` | plan's check 1 is correct |
| Peak in our units, small amplitude | Phi(0) = 4.33739 sqrt(8 (m^2 - omega^2)/g4) | omega near m | `DERIVED` | Set B omega = 0.99: 3.16 vs plan 3.1 (-2%) |
| Small-amplitude N, E | g4 N = 75.59 omega/eps, E = N/omega | eps = sqrt(1 - omega^2) < 0.25 | `DERIVED` from eq. 54 | Set B omega = 0.99: N = 1768, E = 1786 vs plan 1895, 1912 (+7%) |
| Expected sextic correction at the core | (beta/3) eps^2 S(0)^2 = 7.6% | Set B, omega = 0.99 | `DERIVED` | same size as the +7% above |
| Third-harmonic factor on sextic | 1 + 1/(8 beta) | m = 1, g2 = 0 | `DERIVED` (Fodor eq. 25; Mukaida eq. 2.17) | 0.611: 1.20; 0.70: 1.18; 1.0: 1.125; 4/3: 1.09 |
| alpha_c^2 | 27/160 | single-frequency | A&S eq. 16 `READ` | omega_flat in section 1.3 |
| N-minimum, universal | alpha^2/alpha_c^2 = 0.14 | single-frequency | CKS, Li, A&S | table in 2.4 |
| beta = 4/3 point | N_min 80.3, omega 0.97 | g4 = 6, g6 = 48, m = 1 | Li 2025 `READ` | recommended validation row |
| E/(mN) at N-min | about 1.0 (beta = 4/3); plan 1.034 (0.611) | | Li Fig. 2 `READ (figure)` | different beta |
| Lifetime, stable branch | 1e4-1e5 | beta = 4/3, radial | Ibe, Mukaida `READ` | none below beta = 1 |
| Decay rate Gamma | 1e-7 to 4e-4 | beta = 4/3 | Ibe Fig. 4-5 `READ (figure)` | peaks 1e-4 to 4e-4 |
| Collapse of pure phi^4 in 3D | yes (small-amplitude branch) | | Hertzberg `READ` | |

### 4. Interactions (Steps 3 and 4)

#### 4.1 Sources

| Source | System | Result | Tag |
|---|---|---|---|
| [Battye & Sutcliffe 2000](https://arxiv.org/abs/hep-th/0003252) | complex Q-balls, U = f^2 (1 + (1 - f^2)^2) (beta_complex = 5/4), 1D, 2D, 3D (100^3 grids) | alpha = 0 attract and coalesce (charge deficit leaves as fission products); alpha = pi repel with no charge transfer; other phases give charge transfer; in 3D small Q-balls behave as in 1D | READ |
| [Axenides et al. 2000](https://arxiv.org/abs/hep-ph/9910388) | Q-balls, 1D and 2D | pairs generically perform breather-type oscillations at the difference of internal frequencies; slow equal-charge 2D collisions scatter at right angles then oscillate bound | READ |
| [Bowcock, Foster, Sutcliffe 2008](https://arxiv.org/abs/0809.3895) | Q-balls, 1+1, general potential | momentum-flux force F = -4 cos(theta) A_omega^2 omega'^2 exp(-2 a omega'), omega' = sqrt(1 - omega^2); F < 0 (attractive) in phase, F > 0 (repulsive) at theta = pi; A_omega = 2 omega' in the models studied, so F proportional to omega'^4; valid only for equal frequencies, exactly in or out of phase | READ |
| [Amin & Mocz 2019](https://arxiv.org/abs/1902.07261) | NR solitons, saturating attractive nonlinearity, 3D lattice, with self-gravity and expansion | relative phase about pi: repel or bounce; about 0: merge, burst of waves; also binaries and 3-body; heuristic explanation via refractive index | READ |
| [Cotner 2016](https://arxiv.org/abs/1608.00547) | NR boson stars, Gaussian ansatz, with gravity | effective potential: in phase attractive and out of phase repulsive in the weak self-interaction regime; switched in the strong-repulsive regime. Repulsive self-interaction and gravity included; context only | READ |
| [Palenzuela et al. 2007](https://arxiv.org/abs/gr-qc/0612067) | boson stars, 3D, gravity | phase opposition: stars repel and bounce; in phase: merge | READ (partial) |
| [Hindmarsh & Salmi 2008](https://arxiv.org/abs/0712.0614) | oscillons, phi^4 double well, 2D | off-axis collisions: attractive scattering; below a critical velocity they merge; energy leaks as spiral waves; the relative phase appears to control the energy transfer in oscillon-domain-wall crossings | READ |
| [Amin & Mou 2020](https://arxiv.org/abs/2009.11337) | real oscillons, axion-like, 3+1D lattice, omega = 0.82 | identical in-phase, separation 15, v = 0.01c: merge, E_f about 0.7 (E1 + E2), omega_f = 0.76; head-on mergers up to v = 0.1c; exactly pi out of phase they bounce off; small phase differences merge with a small final velocity (cites [Hertzberg et al. 2020](https://arxiv.org/abs/2001.07476) `SECONDHAND`); ultra-relativistic: pass through | READ |
| [Xue et al. 2025](https://arxiv.org/abs/2510.01597) | real oscillons, generalized monodromy potential, 3+1D, 128^3 or 256^3, closing speed 0.01-0.04c | identical copies: all in-phase collisions merge, out-of-phase (field times -1) bounce; mismatched oscillons: outcome effectively stochastic because frequencies differ and phase drifts | READ |
| [Campos, Manton, Mohammadi 2026](https://arxiv.org/abs/2605.12633) | small-amplitude oscillons around a false vacuum, 1+1 | force F = 4 A^2 eps^4 cos(delta) exp(-2 a eps) (eq. IV.13); windows of reflection, crossing, merging, vacuum decay; see 4.2 for a sign problem | READ |
| [Blaschke et al. 2025](https://arxiv.org/abs/2502.09136) | 1+1, oscillon from Q-ball (RG expansion) | an excited (amplitude-modulated) oscillon is a bound state of two fundamental oscillons | READ (intro, conclusion) |
| [Canillas Martinez et al. 2025](https://arxiv.org/abs/2509.03192) | complex phi^6, 1+1 | Q-ball and anti-Q-ball collisions in thin-wall regime are chaotic, with ephemeral bubbles and charged oscillons | READ (abstract, sec. 1-3) |
| [Tsimring & Aranson 1997](https://arxiv.org/abs/patt-sol/9703009) | vibrated granular layer, phenomenological amplitude equation | numerically found bound states of oppositely phased oscillons; also a stable like-phased bound state at larger separation; chains | READ |

Not the same physics. The Wikipedia [Oscillon](https://en.wikipedia.org/wiki/Oscillon) article states, with no citation attached, that oscillons of opposite phase attract and form bonded pairs and chains while like phases repel (`READ` the passage; the page cites no source for it). The only primary source I found with that sign is the granular-media model above, which is a parametrically driven pattern-forming system, not a Klein-Gordon field. Do not import it.

#### 4.2 The sign conflict and my re-derivation

[Campos, Manton, Mohammadi 2026](https://arxiv.org/abs/2605.12633) `READ` writes, after eq. IV.13, that the force is attractive for cos(delta) < 0 and repulsive for cos(delta) > 0. Their formula has F proportional to +cos(delta), computed as the rate of change of momentum in the half-line to the left of an intermediate point. For that definition a positive F pulls the left oscillon toward the right one, i.e. attraction in phase. [Bowcock et al. 2008](https://arxiv.org/abs/0809.3895) `READ` use the same momentum-flux method and find in-phase attraction. I could not reconcile their verbal sign with their own formula. I cannot see eq. IV.2's sign convention cleanly in the extracted text. Treat their words as unreliable until Step 3 measures it.

Re-derivation (`DERIVED`, momentum flux through the midplane; the method of Bowcock et al.). Take two tails f_i (static profiles, equal omega, kappa = sqrt(m^2 - omega^2)) with fields f_1 cos(omega t + theta_1) + f_2 cos(omega t + theta_2). The period-averaged cross term in T^xx at the midplane is

    <T^xx>_cross = (1/2) cos(delta) [ -kappa^2 f_1 f_2 + d_x f_1 d_x f_2 - grad_perp f_1 . grad_perp f_2 ].

- 1D: d_x f_1 d_x f_2 = -kappa^2 f_1 f_2, giving -kappa^2 cos(delta) f_1 f_2. With the Fodor tails this is -4 A^2 eps^4 cos(delta) e^(-2 eps a), matching their magnitude. The force on the left body is -T^xx, so F > 0 toward the other oscillon for cos(delta) > 0: attraction in phase.
- 3D, tails f = C e^(-kappa r)/r: at the midplane, <T^xx>_cross = -(1/2) cos(delta) f^2 [kappa^2 + (kappa + 1/r)^2]. Integrating over the plane for kappa d >> 1: F about 2 pi C^2 kappa cos(delta) e^(-kappa d)/d, i.e. U(d, delta) about -2 pi C^2 cos(delta) e^(-kappa d)/d.

So the plan's form U = -C cos(delta) exp(-kappa d)/d is the right shape with C = 2 pi A^2, where A is the coefficient of e^(-kappa r)/r in the tail, attractive in phase. I found no published 3D tail-force formula for oscillons or Q-balls: "no source found". The 1D papers have the same exponential with an extra kappa^2 relative to A_1D^2; the 3D version has no extra suppression relative to the e^(-kappa r)/r amplitude. The factor can be confirmed in Step 3 from centroid acceleration.

Caveats for the cosine law: it holds only for equal omega. Unequal frequencies make the phase drift and give breathing at the difference frequency ([Bowcock et al.](https://arxiv.org/abs/0809.3895), [Axenides et al.](https://arxiv.org/abs/hep-ph/9910388) `READ`), and intermediate phases give charge transfer for Q-balls ([Battye & Sutcliffe](https://arxiv.org/abs/hep-th/0003252) `READ`). At fixed N an energy comparison is not meaningful because the superposed ansatz changes N with separation (Bowcock et al.). Centroid acceleration, as the plan says, is the correct measurement.

#### 4.3 Bound states and clusters

- Two-body bound states: breathing pairs of Q-balls (1D, 2D) and modulated oscillons as two-oscillon bound states (1+1). No static pair at fixed separation appears in any source.
- Multi-body static arrays, polyhedra, chains, "molecules" of scalar-field oscillons: no source found (searches in section 8). The only chains/molecules found are in granular media (above) and orbiting binaries under gravity ([Amin & Mocz 2019](https://arxiv.org/abs/1902.07261) `READ`).
- Charge/N exchange: documented for complex Q-balls with intermediate relative phase; no source found for real 3D oscillons.
- Merging sub-critical lumps (Step 4a): Q-ball merger and fission ([Battye & Sutcliffe](https://arxiv.org/abs/hep-th/0003252) `READ`); the Q-ball literature calls growth by fusion of small Q-balls "solitosynthesis" (`SECONDHAND`: search snippets for [arXiv:1202.0873](https://arxiv.org/abs/1202.0873), not opened). For real oscillons the merger sheds a large energy fraction (30% in [Amin & Mou](https://arxiv.org/abs/2009.11337) `READ`). A lump below N_min has no equilibrium partner at all (Mukaida: no solution below Q_cr; Ibe: rapid decay), so "one alone" is expected to disperse.

### 5. The plan's predictions against the sources

| # | Prediction on record | Verdict | Evidence |
|---|---|---|---|
| 1a | Step 1: profiles with dN/domega < 0 stay localized | **Agrees**, with the finite-lifetime caveat | VK sign in 2.2; Ibe lattice; stable-branch lifetimes 1e4+ at beta = 4/3. They still radiate and drift to N_min. "Stays localized" means to T = 500 only if N is comfortably above N_min |
| 1b | profiles on the other branch do not stay localized | **Partly agrees** | Decay by dispersal or collapse for pure phi^4 (Fodor 2008, Hertzberg). In the cubic-quintic NLS it depends on the sign of the perturbation: lower N disperses, higher N settles on the stable branch with N lost to radiation (CKS 2021, 3D, t up to 600). Mukaida reports that configurations near the unstable branch decay immediately, without giving perturbation details. No source for this scenario in the relativistic sextic model |
| 1c | legacy Gaussian disperses | **Agrees** | Linear KG decays in about 5 (Copeland); oscillon needs initial energy above plateau and core amplitude above the inflection point (Copeland): legacy 0.5 vs inflection 2.74; N_legacy about 12 vs N_min about 1e3 (Set B); no solution below Q_cr (Mukaida) |
| 2 | Section 3 prior table (Set B) | **Agrees**; details | N at the minimum within 1.5% of the universal-curve estimate; dE = omega dN to about 2%; omega = 0.99 row consistent with small-amplitude theory plus the expected sextic correction. Minimum placed at about 0.922 by sampling vs expected 0.933. This brief supports the table to 2-3%, so a Step 1 difference above 2% (WP1A check 4) deserves scrutiny of the solver first |
| 2b | "single-frequency numbers may be off by tens of percent" | **Agrees** | Third-harmonic factor 1.2; NR breakdown below omega = 0.87 at beta = 4/3 (Ibe); Zhang sec. 8 |
| 2c | E/(mN) > 1 near threshold, < 1 deep | **Agrees** qualitatively | E positive for 3D NLS solitons (CKS); beta = 4/3 E/(mN) about 1.0 at death, about 0.9 at 0.85 (Li, figure) |
| 3 | Step 2 beta = 0.70 keeps phi = 0 the true vacuum | **Agrees** (V >= 0), with a caveat | secondary false vacuum exists for beta < 5/6 (section 1.2); large lumps' interiors sit there |
| 3b | "at exactly 5/8 the vacua are degenerate and large lumps behave like bubbles" | **Partly agrees** | complex Q-ball threshold omega_min -> 0 at 5/8 (Battye-Sutcliffe, Canillas Martinez); the real single-frequency flat-top limit is at 9/16 and depends on omega, not on the static vacuum degeneracy. No source for real 3D |
| 3c | profile at E/(mN) of 0.90-0.98 | **No source found**; hazard | stability criterion in the literature is VK, not E < mN (Mukaida's eq. 2.29 comes from an energy argument comparing Q + dQ with Q plus dQ extra quanta of energy omega each; Ibe footnote 6 states it as E(Q1 + Q2) < E(Q1) + E(Q2)). At beta = 4/3 the window is omega about 0.85-0.96, spanning the Gaussian to flat-top crossover and the NR breakdown |
| 4 | Step 2 Go: >= 90% energy in 3 half-widths at T = 1000 | **No source found for this metric**; hazard | Gamma at beta = 4/3 up to 4e-4 gives 33% loss in 1000; only near dips is retention above 90% reliable |
| 4b | Step 2 No-go: no member survives for any beta >= 5/8 | **Unlikely given sources** (radial) | 3D radial oscillons live 1e4-1e5 at beta = 4/3 (Ibe, Mukaida, Li); no published 3D non-radial or beta < 1 test |
| 4c | Control: g4 = g6 = 0 falls below 10% | **Agrees** | linear KG packet leaves in about 5 (Copeland). With periodic boundaries radiation re-enters the box (Hindmarsh-Salmi, Amin-Mou), consistent with the legacy revival of max\|phi\| to 24% by t = 500 (LESSONS item 1) |
| 5 | Step 3: in phase attract and merge, anti-phase repel | **Agrees** | 4.1 table; Xue 2025 and Amin-Mou 2020 are real-field 3+1D. Exceptions: the 2026 verbal sign; granular oscillons (other physics) |
| 5b | U = -C cos(dphi) exp(-kappa d)/d | **Agrees in shape, no 3D source** | 1D force exp(-kappa d) cos(delta) (Bowcock, Campos); 3D derived in 4.2 |
| 5c | "exchange of N between the lumps" | **No source for real oscillons**; Q-ball charge transfer documented | Battye-Sutcliffe |
| 6 | Step 4a plausible, gives one merged lump | **Agrees** | mergers in Q-ball and oscillon literature; energy loss about 30% on merging (Amin-Mou). Sub-critical lumps cannot exist alone |
| 7 | Step 4b unlikely for static polyhedra | **Agrees; no direct source** | in-phase neighbors attract (merger); out-of-phase neighbors repel; pairs breathe; no static multi-oscillon arrays reported |

### 6. Known pitfalls in simulating oscillons

**Boundaries**
- Outgoing-wave (Sommerfeld) conditions are only approximate for massive fields and reflect, contaminating the interior ([Honda & Choptuik 2002](https://arxiv.org/abs/hep-ph/0110065) `READ`).
- Periodic boxes recirculate radiation: part of it returns to the centre, where it can perturb the oscillon or pump energy back in and lengthen the apparent lifetime ([Hindmarsh & Salmi 2006](https://arxiv.org/abs/hep-th/0606016) `READ`; box size must exceed t_max or absorbers be used: [Amin & Mou 2020](https://arxiv.org/abs/2009.11337) `READ`). This is the legacy periodic-wrap lesson.
- Absorbers used: second-order absorbing condition at r_max = 60 ([Zhang](https://arxiv.org/abs/2004.01202)), 50-80 ([Olle et al.](https://arxiv.org/abs/1906.06352)), L = 64 ([Ibe](https://arxiv.org/abs/1901.06130)), core-to-edge 100 ([Salmi & Hindmarsh](https://arxiv.org/abs/1201.1934)), coordinate "freezing" plus dissipation ([Honda & Choptuik](https://arxiv.org/abs/hep-ph/0110065)), adiabatic damping (Gleiser-Sornborger, `SECONDHAND` via Honda & Choptuik). Validation used by Salmi & Hindmarsh: compare total lattice energy with the energy in a shell of radius 5 r0; a reflected burst would show as a rise in the shell energy `READ`. This is compatible with the plan's twice-box check.
- Free low-momentum quanta with frequency just above m appear when an initial profile settles, and may linger ([Zhang 2020](https://arxiv.org/abs/2004.01202) `READ`, footnote 15). Slow quanta are the hard case for any absorber.

**Initial profiles**
- Single-frequency profile with phi_t = 0 is not the quasi-breather. It latches onto an oscillon with higher harmonics, and to observe a given state it is better to initialize at omega somewhat below the target ([van Dissel et al.](https://arxiv.org/abs/2303.16072) `READ`).
- Gaussians settle onto the same attractor; the attractor is independent of the initial shape if the field is inhomogeneous enough, but the early lifetime and the final plateau depend on r0 with fine structure ([Olle et al.](https://arxiv.org/abs/1906.06352), [Honda & Choptuik](https://arxiv.org/abs/hep-ph/0110065) `READ`).
- Small-amplitude-expansion initial data: first and second order fail to give the expected state at eps = 0.0555 in 3D; improvement with order is not monotonic ([Fodor 2008](https://arxiv.org/abs/0802.3525) `READ`).
- Profiles for Q below Q_cr die quickly; profiles that start below the attractor charge relax to it only above Q_cr ([Mukaida](https://arxiv.org/abs/1612.07750) `READ`, R_ini = 5 case). Start comfortably above N_min.

**Resolution and scheme**
- Used in published radial runs: dr = 0.025, dt = dr/5, second-order Verlet ([Zhang](https://arxiv.org/abs/2004.01202)); dr = 0.05, dt = dr/5 ([Olle](https://arxiv.org/abs/1906.06352)); dx = 0.01, dt = 0.002 ([Salmi-Hindmarsh](https://arxiv.org/abs/1201.1934)); L = 64 with N = 1024 and dt = 0.01, fourth-order symplectic and fourth-order differences ([Ibe](https://arxiv.org/abs/1901.06130)); compactified grid with 2^12-2^13 points, about a week on a PC for T = 7000 ([Fodor 2006](https://arxiv.org/abs/hep-th/0609023)). Convergence was checked by changing box, step and averaging windows in each.
- The plan's grid criterion uses the 3 omega radiation wavenumber. At the radiation dips the 5 omega channel dominates ([Zhang](https://arxiv.org/abs/2004.01202), [Mukaida](https://arxiv.org/abs/1612.07750) `READ`), and these dips are exactly the long-lived states. k_5 = sqrt(25 omega^2 - 1) is about 4.1 at omega = 0.85. Resolution must resolve it.
- 3D lattices: numerical noise with the symmetry of the cubic lattice seeds fragmentation along lattice diagonals; the break-up of large-amplitude (central amplitude near 8 pi in sine-Gordon) and nodal states happens even with no added noise ([van Dissel et al.](https://arxiv.org/abs/2303.16072) `READ`). For large core amplitude, V'' swings widely and Floquet bands extend to high k (k_max roughly omega Phi0 in sine-Gordon). Run spherical-symmetry data in 3D as the baseline and distinguish numerical from physical breakup by resolution.
- Lifetimes are limited by machine precision near resonances, 1 part in 1e14 ([Honda & Choptuik](https://arxiv.org/abs/hep-ph/0110065) `READ`); not an issue for non-tuned data.

**Lifetime and localization measures**
- Energy inside a radius of order 10 is nearly independent of the radius chosen ([Zhang](https://arxiv.org/abs/2004.01202), r_max/2 with r_max = 60). Shell at 5 r0 ([Salmi-Hindmarsh](https://arxiv.org/abs/1201.1934)). Mass defined as energy inside the radius that held 90% of the initial energy, with a state requiring omega to vary less than 10% and mass less than 20% ([van Dissel et al.](https://arxiv.org/abs/2303.16072) `READ`). Central amplitude falling below a threshold: lifetime insensitive to the precise rule because the final decay is quick ([Fodor 2006](https://arxiv.org/abs/hep-th/0609023) `READ`).
- Late-time decay is a slow power law and the frequency drifts up as t^(-0.2) toward a value above m that is never reached ([Salmi & Hindmarsh](https://arxiv.org/abs/1201.1934) `READ`). The amplitude carries a growing beat, so averaging over a window is required: T_avl = 10 ([Mukaida](https://arxiv.org/abs/1612.07750)), 100 ([Ibe](https://arxiv.org/abs/1901.06130)), 200 ([Zhang](https://arxiv.org/abs/2004.01202)). Measure omega from zero crossings or maxima at the origin, averaged.
- Large-amplitude decay can be staircase-like (1D evidence: Dorey et al. `ABSTRACT`); a threshold on E_loc may trigger on a step.
- Radial simulations overestimate lifetimes relative to 3D lattice for fragmenting states ([van Dissel](https://arxiv.org/abs/2303.16072) `READ`); quasi-breather radiation estimates overshoot measured lifetimes by factors of a few ([Saffin & Tranberg 2007](https://arxiv.org/abs/hep-th/0610191) `READ`, partial).

### 7. Implications for Session 1

1. **Add a beta = 4/3 validation row to WP1A** (g4 = 6, g6 = 48, m = 1; or any g4 with beta = 4/3). Targets from three papers: N_min = 80.3 (g4 N = 481.7) at omega_death = 0.97; N(0.94) = 96.2; N(0.85) = 525. Plus the universal-curve collapse and alpha_c^2 = 27/160. This costs nothing and tests the solver against the literature instead of only against the plan's own table.
2. **Expected family minima per beta** are in the table in 2.4 (g4 N_min = 314, 318, 339, 378, 413 for beta = 0.611, 0.625, 0.70, 0.85, 1.0). Differences above 3% are findings.
3. **The unstable-branch radial run has two legitimate outcomes.** Record the sign of the effective perturbation (single-frequency data with phi_t = 0 is not the quasi-breather) and whether the lump disperses or relaxes to the stable partner. Do not call the second a contradiction of the stability criterion.
4. **Parameter pick.** The window E/(mN) = 0.90-0.98 is not a literature criterion and lands in the Gaussian-to-flat-top crossover where the NR approximation fails (omega about 0.85-0.96 at beta = 4/3). Two choices to weigh: (a) beta = 4/3 has three independent radial references and lifetimes of 1e4+, but is the same potential as the published work, so the Step 2 gate adds only the 3D, non-radial question; (b) beta = 0.70 has no published 3D point, a false vacuum at g4 phi_v^2 = 20, and a thin N-minimum margin (at omega = 0.90, N is only about 1.1 N_min by interpolating the universal curve; low confidence). Whichever is chosen, start with N at least 1.3 to 1.5 times N_min. In Ibe's run, N = 130 against N_cr = 80 (ratio 1.6) lived to about 1e4 to 2e4.
5. **Define the Go/No-go reading in advance for radiation.** Predict Gamma(omega) from the radial run (WP1A) and compare the 3D E_loc curve to exp(-Gamma t) before calling a miss. A fall of 10-30% over T = 1000 near a Gamma peak is expected for beta = 4/3 and would otherwise register as a miss. If the target must be met at a generic omega, expect to choose near a dip; the dips for beta below 1 are not published.
6. **Grid and absorber:** resolve the 5 omega radiation, not only the 3 omega; keep E_loc radius at 3-5 half-widths or more than 10; keep the total-versus-shell energy consistency check; start with spherically symmetric initial data and no added noise, then the distortion run. Expect cubic-lattice-seeded high-k growth at large core amplitude.
7. **Third-harmonic mismatch:** the PDE will differ from the single-frequency profile at the level of beta_eff = beta + 1/8. If the radial check "contradicts the stability prediction", this may be the cause: try initial omega somewhat below the target (van Dissel) before concluding.
8. **Step 3:** expect in-phase attraction and merger with about 30% energy radiated; use equal-omega pairs, exactly 0 and pi relative phase, centroid acceleration, short times. Predicted force at large separation: F about 2 pi A^2 kappa cos(delta) exp(-kappa d)/d (derived, section 4.2); A is the coefficient of exp(-kappa r)/r.
9. **Step 4a:** expect lumps below N_min to disperse alone; sub-critical lumps merge with substantial radiation, so the merged N will be below the sum; choose k with total N well above N_min. **Step 4b:** no published static multi-oscillon arrangement; the expectation on record stands.

### 8. Novelty-check log

Searches run (web search, 2026-10-01) and what they returned. Nothing in this brief is claimed as new.

| # | Search | Result |
|---|---|---|
| 1 | Fodor/Forgacs/Horvath/Lukacs small-amplitude quasi-breathers | arXiv:0802.3525, read |
| 2 | Amin and Shirokoff flat-top oscillons | arXiv:1002.3380, read |
| 3 | Zhang/Amin/Copeland/Saffin/Lozanov classical decay rates | arXiv:2004.01202, read |
| 4 | Kasuya/Kawasaki/Takahashi I-balls | hep-ph/0209358, read (an earlier automated summary of this paper was wrong and was discarded) |
| 5 | Cyncynates and Giurgica-Tiron structure of the oscillon | arXiv:2104.02069, read |
| 6 | focusing cubic NLS 3D unstable / blow-up / VK | led to Carles-Klein-Sparber arXiv:2012.11637 (read); other hits (arXiv:1007.4025, 0902.1643) not opened |
| 7 | pure phi^4 attractive collapse, sextic stabilization | Hertzberg arXiv:1003.3459 (read) |
| 8 | oscillons phi^6 degenerate vacua, thin wall, bubble | arXiv:2509.03192, 2603.28298 (abstract), 1910.04128 (abstract) |
| 9 | Ibe et al. decay of I-ball | arXiv:1901.06130 (read) |
| 10 | Amin inflaton fragmentation | arXiv:1006.3075 (parameter section read) |
| 11 | oscillon-oscillon interaction, relative phase, bound states, molecule | 2605.12633 (read), Wikipedia (uncited) |
| 12 | oscillon mergers, in/out of phase, Yukawa tail | 2510.01597, 2009.11337 (read) |
| 13 | Q-ball interaction relative phase effective potential | hep-th/0003252, 0809.3895 (read) |
| 14 | origin of "opposite phase attracts, like repels" | granular layer, patt-sol/9703009 (read) |
| 15 | bound states of Q-balls, multi-Q-ball clusters | no static cluster found; Q-ball polarization, excited Q-balls (not opened) |
| 16 | multi-soliton NLS bound states with phase pi/2 | generic soliton-optics hits; one snippet claims stationary two-soliton bound states exist for phase differences 0, pi, +/- pi/2 in reduced models with oscillatory tails; `SECONDHAND`, not traced; relevant to Step 4b, worth a follow-up |
| 17 | oscillon arrangement on polyhedron vertices, regular lattice | no relevant result |
| 18 | oscillons below critical charge merging above critical; energetic death merger | no direct result; Campos et al. and Amin-Mou on merging |
| 19 | solitosynthesis | arXiv:1202.0873 and others (not opened) |
| 20 | recent (2024-2026) 3+1D sextic oscillon lifetime / critical frequency | Li-Yamaguchi-Zhang arXiv:2507.13276 (read); 2026 papers 2605.12633, 2603.28298, 2607.15624 (1+1D, abstract) |

No claim here is flagged as new. The beta-collapse "universal curve" is a re-expression of A&S's eq. 28. The 3D tail-force formula in 4.2 is my derivation using a published method; I found no 3D published formula.

### 9. Not obtained, not read, open

- Not on arXiv, not read: Coleman 1985, Lee & Pang 1992, Friedberg-Lee-Sirlin 1976, Vakhitov-Kolokolov 1973, Bogolyubsky-Makhankov 1976.
- Found but not read, or read only in part: Gleiser-Sicilia general theory of oscillon dynamics (arXiv:0910.5922, downloaded, not read; Zhang et al. footnote 8 `READ` says its radiation assumption is incorrect); Gleiser-Krackow arXiv:1906.04070; Kawasaki-Takahashi adiabatic invariance (2015); Saffin & Tranberg arXiv:hep-th/0610191 (abstract and a skim); Palenzuela et al. (abstract and conclusions); Eby et al. arXiv:1608.06911 (axion stars with gravity, context only); Amin-Mocz and Cotner for the full effective potential at separation.
- Open for the project: (i) PDE-versus-single-frequency mismatch for the sextic model, no published number below beta = 1; (ii) radiation dips for beta < 1; (iii) non-radial stability of the sextic family in 3D; (iv) 3D tail force law and its numerical coefficient; (v) real-oscillon N exchange; (vi) static multi-oscillon arrangements.
- Extraction note: text was extracted from the arXiv PDFs; garbled symbols (for example lost square-root signs) were checked against the mapping consistency tests in section 1.1 before use. Figures were read by eye from rendered pages with the errors stated.

---

## Addendum A (2026-10-01, Session 1): literature checks for F-001 and F-002

Scope: two targeted checks requested for the failure ledger (`docs/FAILURES.md`, entries F-001 and F-002). Not a new full brief. Same evidence tags as above. No code run, no git. Text of every `READ` source was extracted locally from the arXiv PDF; where an extracted formula was garbled I say so. `PROJECT DATA` means a number taken from this repo's own output files, not from the literature.

### A1. Absorbing layers for a massive Klein-Gordon field at low k (for F-001)

Question: is poor absorption of slow, near-gap waves (k about 0.5, group speed about 0.5, omega just above m) by a gamma(r) phi_t sponge a known limitation?

| Source | System | What it says | Tag |
|---|---|---|---|
| [Honda thesis 2000](https://arxiv.org/abs/hep-ph/0009104), sec. on outer boundary | massive and nonlinear KG, spherical | The massive KG equation has no closed-form outgoing condition because the outgoing radiation speed is not unique (omega^2 = k^2 + m^2). Sponge-filter schemes impose an outgoing condition only approximately and can back-scatter. The adiabatic-damping term gamma(x) phi_t damps all radiation, not only outgoing | READ |
| [Honda & Choptuik 2002](https://arxiv.org/abs/hep-ph/0110065), introduction | oscillons, phi^4 | For massive fields the Sommerfeld condition is only approximate and generically gives significant reflection at r_max | READ |
| [Gleiser & Sornborger 2000](https://arxiv.org/abs/patt-sol/9909002), sec. II B, III | 3D oscillons, phi^4 double well, radial | Same sponge form as the engine: phi_tt + gamma(rho) phi_t - lap phi + V' = 0. Damping starts at rho_0 of about 20 R_0 (R_0 = oscillon scale 2.5 to 2.7); box 1024 points, physical size 204.8, dx = 0.2; eta = 0.005, gamma = eta^2 (rho - rho_0)^2 as extracted. Shell energy (r < 5 R_0) agrees with an expanding-grid run to better than 1e-4 relative over the oscillon lifetime. They report that long-wavelength modes are damped less efficiently than short ones and that this, with the reflecting end of the grid, spoils agreement for larger R_0 once the oscillon decays; the stated cure is a larger grid and a larger rho_0 | READ |
| [Li, Yamaguchi, Zhang 2025](https://arxiv.org/abs/2507.13276), App. B | real phi^4-phi^6, beta = 4/3, 3D radial | Same form: gamma = 0 for r <= r_eta = 64, gamma = eta (r - r_eta)^2 beyond, eta = 0.007. Checked against a second-order absorbing condition and against a box larger than the run time, same lifetime | READ |
| [Salmi & Hindmarsh 2012](https://arxiv.org/abs/1201.1934), introduction, sec. II, III and the convex-potential section | phi^4, sine-Gordon, convex potentials | Absorbing boundary built for a massive field, with the core-to-boundary distance fixed at 100. Reflection test: total energy against energy in a shell of 5 r_0; a reflected burst would show as a rise in the shell energy. The boundary update assumes a linearizable potential; where V'' varies strongly the absorption is only partial | READ |
| [Zhang et al. 2020](https://arxiv.org/abs/2004.01202), sec. 6 and footnote 15 | real scalar, radial | Second-order absorbing condition at r_max = 60 for oscillon widths below about 10. At large amplitude a very low momentum mode with frequency (1 + eps) m appears and is read as free particles made while the initial data settle | READ |
| [Antoine & Zhao 2022](https://arxiv.org/abs/2101.02528), sec. 2.1, 3 | nonlinear KG, PML, 1D | For the linear equation the PML is perfectly matching. The in-layer damping of a mode is modulated by g(s) = -sqrt(s^2+1)/(s+alpha); the authors note there are always s with Re(s) arbitrarily small for which g(s) approaches the imaginary axis, i.e. no damping. | READ |
| [Mani 2010](https://web.stanford.edu/group/ctr/ResBriefs/2010/11_mani.pdf), sec. 1 to 2.3 | compressible acoustics, damping sponge, non-dispersive (context only) | A sponge is ineffective when its strength is not small against the incident frequency (reflection tends to the no-sponge value as omega goes to 0); a higher strength causes reflection from the sponge itself; design requirements are set by the low-wavenumber content | READ |
| [Komech & Kopylova 2010](https://arxiv.org/abs/1003.3799), introduction | free 3D KG | Free 3D Klein-Gordon solutions decay as t^(-3/2) in L-infinity (attributed there to Morawetz and Strauss; I did not open that paper: SECONDHAND for the original) | READ (intro) |

Not found. (i) A published reflection coefficient versus k for a gamma(r) phi_t sponge on a massive field near the gap: no source found. (ii) A published criterion for gamma_max against m: no source found. (iii) Any published sponge for this model with interior radius below 50: every radial setup I read uses 50 to 100 between core and absorber (section 6, and the rows above). The Antoine-Lorin-Tang 2017 review of absorbing conditions for KG ([Molecular Physics 115, 1861](https://www.tandfonline.com/doi/full/10.1080/00268976.2017.1290834)) is paywalled and was not read.

`DERIVED` notes N1 to N4 (my algebra; re-derive before relying on them; named N so they are not confused with the orchestrator's review diagnostics D1 to D3 quoted in F-002).
- N1. For phi_tt + gamma phi_t - lap phi + m^2 phi = 0 with uniform gamma, a plane wave e^(ikx - i omega t) has omega = -i gamma/2 +/- sqrt(k^2 + m^2 - gamma^2/4). The mode is critically damped at gamma = 2 sqrt(k^2 + m^2), so at k = 0 at gamma = 2m. The engine default gamma_max = 2 with m = 1 sits at that value for the slowest modes and is overdamped (slow root decays at gamma/2 - sqrt(gamma^2/4 - k^2 - m^2), less than gamma/2) for gamma > 2 sqrt(k^2 + m^2). For a steady wave at real omega the local wavenumber inside the sponge is k_eff^2 = omega^2 - m^2 + i gamma omega; for omega^2 = m^2 + 0.36 (k = 0.6) and gamma = 2 this is k_eff about 1.17 + 1.0 i, so a wave with k = 0.6 meets a large change of wavenumber near the sponge peak. The same plane-wave reading of Antoine & Zhao eq. 2.5 at s = i omega gives a PML attenuation factor v_g times the integral of sigma, which goes to zero with v_g as omega tends to m. Whether the project's reflection is dominated by this mechanism is not established.
- N2. Onset times in F-001 (first 1e-3 error at t = 35.5, 48.2, 58.4, 73.6 for r_in = 13, 16.1, 19.25, 22.4), read as an out-and-back path r_in + (r_in - 10) to r = 10, give v = 0.45, 0.46, 0.49, 0.47, i.e. k = v / sqrt(1 - v^2) of 0.50 to 0.56, nearly constant across r_in. So one band of wavenumbers near k = 0.5 sets the first reflected signal at every r_in.
- N3. `PROJECT DATA` (outputs/engine/absorber_design_scan.json, 1D scan, gamma_max 2, width 12, cubic, m = 1; fraction of packet energy found back in the interior): 0.174 (k0 = 0.2), 0.069 (0.4), 0.0021 (0.7), 1.9e-5 (1.0), 1.9e-6 (1.5). The scan file's own note says below k0 of about 0.5 the number is not a reflection coefficient. For gamma 4, width 14 against gamma 2, width 14 the returned fraction at k0 = 0.7 is 0.0029 against 0.0011 (2.6 times worse), at 0.4 it is 0.077 against 0.056. So the knee of the 1D data (k0 between 0.4 and 0.7) coincides with the band N2 identifies.
- N4. Compared with Gleiser-Sornborger (taking the extracted 'eta^2' form at face value: gamma at the far end = 0.005^2 x about 150^2, about 0.6, layer about 150 wide, m = sqrt(2) in their units), the engine default is about 5 times stronger relative to m and about 12 times narrower. The in-repo radial solver (sigma_max = 1, quadratic ramp, width 100, `swirl/family/run_radial.py`) is much closer to the published form than the 3D engine default.

### A2. Lifetime and death at low sextic coupling (for F-002)

Mapping used (from section 1.1 above): A&S g_AS = 3 beta / 10; (lambda/g)^2 in Amin-Easther-Finkel = 1 / g_AS = 10 / (3 beta).

| Source | System | What it says | Tag |
|---|---|---|---|
| [Mukaida et al. 2017](https://arxiv.org/abs/1612.07750), [Ibe et al. 2019](https://arxiv.org/abs/1901.06130), [Li et al. 2025](https://arxiv.org/abs/2507.13276) | beta = 4/3 only (g4 = 1, g6 = 0.4 in their units) | Only one value of g6 is simulated in each. Decay is the slow loss of charge to the critical value Q_cr, then rapid death | READ |
| [Kasuya et al. 2003](https://arxiv.org/abs/hep-ph/0209358) | beta about 1.67 (a = 0.1, b = 0.005), lattice, from noise | One parameter set; lifetime of order 1e3; no dependence on b studied | READ |
| [Amin, Easther, Finkel 2010](https://arxiv.org/abs/1009.2505), sec. 1, 4.3 | 3D lattice, A&S model | Fiducial (lambda/g)^2 = 0.1, also 0.2, i.e. beta about 33 and 17. They restrict the analytic treatment to (lambda/g)^2 much less than 1 and warn that outside it solutions need not be well described by the single-frequency profile | READ |
| [Amin & Shirokoff 2010](https://arxiv.org/abs/1002.3380), sec. 4 | analytic, g_AS much greater than 1 | Corrections to the single-frequency profile are strongly suppressed only for g much greater than 1; their example g of about 5 gives a denominator near 100. Our beta of 0.611 to 1.0 is g_AS of 0.18 to 0.30 | READ |
| [Zhang et al. 2020](https://arxiv.org/abs/2004.01202), sec. 7.5, 8 | phi^6 potential, radial | Decay rate is non-monotonic with multiple dips; the single-frequency decay-rate calculation over-estimates the numerical rate at large central amplitude and is not trustworthy beyond a few times the potential scale, while the single-frequency energy stays reasonable. Section 7.5 gives no numerical parameters | READ |
| [Olle, Pujolas, Rompineve 2020](https://arxiv.org/abs/1906.06352), conclusions, footnote 8 | 3D radial, flattened and cosine-type potentials | Heuristic ("seems to be") condition for very long-lived oscillons: V'' not very negative anywhere, i.e. V'' positive everywhere or abs(V'') much smaller than V''(0). Established for their benchmark potentials, not for phi^4-phi^6 | READ |
| [Nagy & Takacs 2021](https://arxiv.org/abs/2105.01089), sec. IV A | sine-Gordon in D spatial dimensions, quasi-breather initial data | In D = 3 the oscillon radiates, its frequency rises and it collapses suddenly at the frequency of the energy minimum (omega_c = 0.9395 for D = 3). Quasi-breather initial data of high precision show almost no initial transient | READ |
| [Fodor et al. 2008](https://arxiv.org/abs/0802.3525), [van Dissel et al. 2023](https://arxiv.org/abs/2303.16072), [Zhang 2020 fn 15](https://arxiv.org/abs/2004.01202) | various | phi_t = 0 single-frequency data are not the quasi-breather and settle with radiation and free low-momentum quanta; small-amplitude-expansion initial data are not reliably accurate | READ (see section 6 above) |
| [Drogosz, Sfakianakis, Slawinska, Wereszczynski 2026](https://arxiv.org/abs/2607.23616), sec. II, III | 3D radial, real scalar with sextic term, oscillating about vacua at K = +/-1 with a possible symmetric local minimum at K = 0; Gaussian initial data. Different potential from the plan | Lifetime against sextic coupling chi_6 (in m_H^2): from about 6e3 at 0 to above 2e5 by about 0.2 to 0.4; slowly falling to 0.6; a more intricate multi-peak lifetime-versus-R_0 structure around the onset of the extra minimum at 2/3 that simplifies at degeneracy at 1. In a two-field extension the lifetime falls once the symmetric minimum appears but stays much longer than for chi_6 = 0. Non-monotonic in the sextic coupling; context only | READ |
| [Canillas Martinez et al. 2025](https://arxiv.org/abs/2509.03192), abstract and sec. 1 | complex phi^6, 1+1 | A bubble of the false vacuum is usually short-lived and is only temporarily stabilized by Goldstone waves. Context only (complex field, 1+1D) | READ (abstract, intro) |

No source found for: any 3D oscillon lifetime or decay-rate measurement of the real phi^4-phi^6 model at beta below 4/3 (the published values are 4/3, about 1.67 in 2D, and about 17 to 33); lifetime shrinking monotonically as beta falls; the position of Gamma dips for beta below 1.

`DERIVED` notes N5 to N9 (my algebra; re-derive before relying on them).
- N5. V'' = m^2 - g4 phi^2 / 2 + g6 phi^4 / 24 has its minimum at phi^2 = 6 g4 / g6, giving V''_min / m^2 = 1 - 3 / (2 beta). Values: beta 0.611, 0.625, 0.70, 0.85, 1.0, 4/3, 1.5 give -1.455, -1.40, -1.143, -0.765, -0.50, -0.125, 0. So |V''_min| / V''(0) = 3 / (2 beta) - 1 is 1.46, 1.14, 0.76, 0.50, 0.125 for beta 0.611, 0.70, 0.85, 1.0, 4/3. It orders the F-002 outcomes (long-lived at 4/3, alive to T = 3000 at 1.0, survivors at 0.85, none at 0.70 and 0.611) in the direction the Olle et al. heuristic points, but that heuristic was not tested on this model.
- N6. A false vacuum exists for beta < 5/6 = 0.8333 (section 1.2). In the universal variable Psi (section 1.3) it sits at Psi_v^2 = (1 + sqrt(1 - 1.2 beta)) / 2, which is 0.5 at beta = 5/6 and 0.70 at beta = 0.70, while the profile core reaches Psi_0 of about 0.9 to 1.0 on the stable branch. I first suggested (before the independent review) that the F-002 outcomes straddling 5/6 might be a false-vacuum threshold effect. **Tested and not supported:** the orchestrator's matched-N/N_min scan (F-002, D3) has death times 216.5 / 308.5 / 482.5 at beta = 0.611 / 0.70 / 0.80 for N/N_min = 1.3, then survivors at 0.85, so the lifetime is smooth and monotone in beta with no step at 5/6. The fact (a false vacuum exists below 5/6) stands; the hypothesis that it sets the lifetime does not.
- N7. In the single-frequency, source-only picture the 3 omega source is S_3 = -g4 Phi^3 / 24 + g6 Phi^5 / 384, which changes sign at Psi^2 = 0.8 independent of beta, so a dip (a zero of its Fourier transform at k_3 = sqrt(9 omega^2 - 1)) needs a core with Psi_0 above 0.894. At fixed alpha^2 the profile is beta-independent and (k_3 L)^2 = 2.4 beta - 9 alpha^2 (L^2 = 0.3 beta), so the dip position in alpha^2 depends on beta. Using the two measured dips (beta = 4/3, omega = 0.909, alpha^2 = 0.0695; beta = 0.85, omega = 0.779, alpha^2 = 0.1003) and a linear interpolation of (k_3 L)^2 against alpha^2, the beta = 0.70 dip lands near alpha^2 = 0.110, omega = 0.69, just below the scanned omega range 0.71 to 0.92. Two points, a linear guess and an assumption that both dips belong to one family: low confidence. The point is only that the absence of dips in 0.71 to 0.92 does not show there are none.
- N8. The universal N-minimum estimate (section 2.4) gives omega at the N-minimum of 0.933 (beta = 0.611) and 0.942 (beta = 0.70). The F-002 entry reports omega_c of 0.925 and 0.937 at death, within 1 percent.
- N9. The third-harmonic factor 1 + 1/(8 beta) (section 2.1) is 1.205 at beta = 0.611 and 1.094 at beta = 4/3, a 10 percent change, and the A&S correction prefactor 1/(24 g_AS) = 1/(7.2 beta) changes by 2.2 times over the same range. The measured initial shedding at N/N_min = 1.8 (F-002, D3) falls from 34 percent to 0.03 percent, a factor of about 1000. So these small-amplitude factors point the right way but are too weak a lever to account for the size of the change on their own; a steeper dependence is needed (for example the exponential suppression of radiation at k_3, whose scale (k_3 L)^2 = 2.4 beta - 9 alpha^2 of N7 is about 0.9^2 at beta = 0.611 and 1.6^2 at 4/3 for alpha^2 of about 0.065, DERIVED roughly) or other harmonic content. No source gives the dependence.

### A3. Search log for this addendum (numbering continues from section 8)

| # | Search | Result |
|---|---|---|
| 21 | PML Klein-Gordon massive field low frequency cutoff reflection | Antoine & Zhao arXiv:2101.02528 (read). EM waveguide PML results not opened |
| 22 | sponge damping slow waves dispersive KG oscillon reflection | Honda thesis hep-ph/0009104 (read); fluid-dynamics sponge papers, one read (Mani) |
| 23 | Gleiser Sornborger adiabatic damping | patt-sol/9909002 (read) |
| 24 | nonreflecting boundary conditions KG near cutoff | high-order NRBC dispersive-wave papers, not opened |
| 25 | PML evanescent cutoff waveguide | electromagnetics; snippets only, not used |
| 26 | sponge gamma phi_t KG critical damping overdamped | no KG-specific source found |
| 27 | review of absorbing conditions for relativistic waves | Antoine, Lorin, Tang 2017 (paywalled, not read) |
| 28 | free KG 3D decay t^(-3/2) | Komech & Kopylova arXiv:1003.3799 (introduction read) |
| 29 | oscillon lifetime against sextic coupling | arXiv:2607.23616 (read). A review (arXiv:2411.16604) has a sentence "lifetime increases with the phi^6 coupling"; I opened it: it concerns composite Q-balls, not oscillons. Discarded. A search-engine summary attributing a lifetime-versus-epsilon statement to sextic strength could not be traced to any source I read. Discarded |
| 30 | oscillon false vacuum metastable lifetime | 2605.12633 (abstract), 2509.03192 (read), 2603.28298 (abstract) |
| 31 | collapse instability staccato decay oscillons | Nagy & Takacs arXiv:2105.01089 (read) |
| 32 | phi^4-phi^6 lifetime scan, critical charge versus g6 | only g6 = 0.4 in Mukaida and Ibe; Amin-Easther-Finkel arXiv:1009.2505 (read, beta of 17 to 33); arXiv:2406.00108 (alpha-attractor, not sextic, irrelevant) |
| 33 | oscillon V'' negative, large amplitude, decay | Zhang 2020 (read), Olle et al. footnote 8 (read) |
| 34 | Gleiser & Sicilia lifetime vs potential | arXiv:0804.0791, quartic polynomial potentials per its abstract snippet; not opened |
| 35 | oscillon sextic extra local minimum, lifetime | Drogosz et al. arXiv:2607.23616 (read) |
| 36 | KG damping layer critical damping | no relevant source found |
| 37 | energy conservation reported in oscillon simulations (cached papers) | Copeland 1995, Gleiser & Sornborger, Honda & Choptuik give numbers (A4); Salmi-Hindmarsh, Zhang, Ibe, Li, Olle: none stated |
| 38 | symplectic integrators, modified Hamiltonian, energy drift | Hairer, lecture 3 (read, A4) |
| 39 | oscillon lifetime by integrating decay rate along adiabatic evolution; dips and attractors | Ibe eq. 55, Li eq. 2.30, Mukaida, Zhang sec. 3.1, 6, 8 (all read, A5) |
| 40 | oscillon decay rate time dependence, frequency drift to critical frequency | Salmi & Hindmarsh, Nagy & Takacs, van Dissel, Cyncynates & Giurgica-Tiron (read, A5); no source for sextic beta below 4/3 |

Claims of novelty: none made. N1, N5 to N7 and N9 are suggestions to test. If any of them is later called new, the searches above (22, 26, 29, 30, 32, 35, 36) are the documented attempt and found nothing that states it.

### A4. Energy conservation of radial oscillon solvers (for F-003; minimal check)

| Source | System | What it says | Tag |
|---|---|---|---|
| [Copeland et al. 1995](https://arxiv.org/abs/hep-ph/9503217) | phi^4 double well, radial | Energy conserved to better than one part in 1e3; a high-resolution run (d_rho = 0.01, d_tau = 0.005) gave one part in 1e5 | READ |
| [Gleiser & Sornborger 2000](https://arxiv.org/abs/patt-sol/9909002), sec. II A | phi^4, 4th-order symplectic splitting, dt = 0.01, 650000 steps | Energy change delta E / E = 4e-4 over 6500 time units, stated as compatible with the expected error | READ |
| [Honda & Choptuik 2002](https://arxiv.org/abs/hep-ph/0110065), appendix and fig. 15 | phi^4, radial, compactified coordinates | Bubble plus radiated energy conserved to a few tenths of a percent over a quarter million iterations; they note energy conservation says little about reflection at the boundary | READ |
| [Hairer, lecture 3 on backward error analysis](https://www.unige.ch/~hairer/poly_geoint/week3.pdf) | symplectic integrators, ODEs | A symplectic method applied to a Hamiltonian system conserves a modified Hamiltonian (a power series in the step) and the original energy only to O(h^p) over long times | READ (sections 1 to 3) |

No source found for the energy-conservation figure of Salmi-Hindmarsh, Zhang et al., Ibe et al., Li et al. or Olle et al.: none of the cached texts states one (search 37).

`DERIVED` note N10. The semi-discrete radial scheme is a finite-dimensional Hamiltonian system whose own energy E_h is constant along the exact flow, so E_h tests the time integrator only (RK4 adds an O(dt^4) drift). A physical-density energy built with a different gradient stencil and quadrature differs from E_h by a functional error of O(dr^2) that changes as the field spreads, so its drift is a property of the diagnostic and does not fall when dt is refined. This is the orchestrator's stated cause for F-003; I add only that it is the expected behaviour.

### A5. Time dependence of the radiation rate and lifetime estimates (for F-004)

Question: is it documented that the radiation rate Gamma changes along the evolution, rising as omega drifts up toward the death frequency and leaving dips as omega drifts, and are lifetime estimates from an early Gamma known to overestimate?

| Source | System | What it says | Tag |
|---|---|---|---|
| [Ibe et al. 2019](https://arxiv.org/abs/1901.06130), sec. III A, IV | real phi^4-phi^6, beta = 4/3, radial | Decay rate Gamma is a function of the instantaneous charge Q (fig. 4), with dips at the zeros of the 3 omega source. Lifetime from eq. 55: tau = integral from Q_cr to Q_i of omega dQ / (E Gamma), an integral along the family, not 1/Gamma at the start. Simulations started at three different omega_ini follow the same analytic Gamma(Q) curve (fig. 5); the lump decays rapidly when Q reaches Q_cr of about 80 (fig. 6) | READ |
| [Li, Yamaguchi, Zhang 2025](https://arxiv.org/abs/2507.13276), eq. 2.29 and 2.30, fig. 3 | same model, beta = 4/3, radial | Gamma = (1/E) dE/dt tabulated against E; lifetime estimated by tau(E_ini) = minus the integral from E_ini to E_death of dE / (E Gamma); simulations from omega_ini = 0.85, 0.90, 0.91 compared with the semi-analytic curve; E_death marks energetic death | READ |
| [Mukaida et al. 2017](https://arxiv.org/abs/1612.07750), abstract and sec. 2.3, 3 | same model | There are approximate attractor behaviours during the evolution (charge lingers near zeros of the 4-point contribution) that enhance the lifetime by many orders of magnitude; the charge evolution has many bumps; after reaching the last attractor just above Q_cr it decays suddenly | READ |
| [Zhang et al. 2020](https://arxiv.org/abs/2004.01202), sec. 3.1, 6, 8 | real scalar, radial | The oscillon passes adiabatically through a unique set of configurations with slowly increasing omega until it collapses at omega_crit; the numerical Gamma along omega(t) is compared with the analytic Gamma(omega); the calculation's over-estimate in the dips prevents accurate lifetime estimates where dips exist | READ |
| [Cyncynates & Giurgica-Tiron 2021](https://arxiv.org/abs/2104.02069), introduction and sec. III C | real scalar, 3+1D, physical quasibreather | Dips ("islands of longevity") migrate to higher frequency as the potential parameter varies, and lifetime grows until the dip passes the frequency of energetic death ("longevity cliff"); at high omega the bulk decouples from radiation, so radiated power falls exponentially near the end of a long life, but energetic death forces a rapid dissipation before the threshold is reached | READ (introduction and III C) |
| [van Dissel et al. 2023](https://arxiv.org/abs/2303.16072) | sine-Gordon, radial | Nodeless states probe the oscillon configurations adiabatically up to energetic death; they describe energetic death as apparently generic for oscillons in power-series potentials; adiabatic evolution can stop early when initial data perturb the state | READ |
| [Nagy & Takacs 2021](https://arxiv.org/abs/2105.01089), sec. IV A | sine-Gordon in D dimensions | Radiation raises the frequency but cannot raise it beyond omega_c; the final decay is a sudden collapse at omega_c (0.9395 for D = 3) | READ |
| [Salmi & Hindmarsh 2012](https://arxiv.org/abs/1201.1934), sec. III | 2D phi^4 | The frequency approaches its critical value as a power law, omega_star - omega ~ t^(-0.18) (omega_star = 0.981), and the energy loss rate flattens out with time (E - E_0 ~ t^(-0.15)). Here the rate falls with time, the opposite sign to F-004 | READ |

No source found for: Gamma at beta = 0.85 or any value below 4/3; a statement that "1/Gamma at the start overestimates the lifetime" as a rule. The published practice is the path integral above. No source found for 3D versus radial comparisons of sextic oscillons (non-radial stability is treated in Cyncynates & Giurgica-Tiron appendix C, which I did not read in full).

`DERIVED` note N11. Taking the F-004 radial windows (Gamma 5.2e-5 for t 200 to 500, 3.0e-4 for 500 to 1000, 2.9e-3 for 1000 to 2000; death at 1145.5), 1/Gamma is 1.9e4, 3.3e3 and 3.5e2: 17 times, 2.9 times and 0.3 times the measured radial death time, depending on the window. Since N falls at the rate Gamma, the drift of omega along the family is slow where Gamma is small and fast where it is large, so a lump leaves a dip faster the larger the dip's Gamma: at beta = 4/3 the dip rate is about 3e-6 and the beta = 0.85 dip rate is about 5e-5 to 6e-5 (17 to 20 times larger), a plausible reason the same constant-Gamma prediction held for the primary run and failed for the secondary. Not derived from sources; re-derive before use.

Correction to section 7, item 5 of the first brief. It advised comparing the 3D E_loc curve with exp(-Gamma t) for a Gamma read from the radial run. The published practice (Ibe eq. 55, Li eq. 2.30) integrates along the evolution with Gamma as a function of the instantaneous omega or energy, and a constant Gamma is only safe where omega stays inside a dip for the whole run. Use Gamma(omega(t)) with the radial family, not a single Gamma.


---

## Session 2 brief

Scope: WP-S of `SESSION_02_PLAN.md`. Written 2026-10-01 for the orchestrator before WP-D. Model claude-sonnet-5-5. Effort: a "reasoning_effort" value of 30 is visible in my context; the mapping from that number to a named level is not visible. No git commands, no simulations, nothing written outside `docs/` (scratch files stayed in the session scratchpad).

Tags as in the Session 1 brief: `READ`, `ABSTRACT`, `SECONDHAND`, `DERIVED`, "no source found". Numbers carry their parameters. `PROJECT DATA` means a number from this repo.

How sources were read. The fetch tool saved the PDF of each paper it could not convert; I extracted the text locally (PDFKit) and searched it. `READ` means I read the passage in that extracted text. Two cautions. (i) The fetch tool's own summaries of PDFs were wrong at least once (it said arXiv:2509.18656 shows an azimuthal instability of spinning Q-balls; the text does not), so no claim below rests on a summary alone. (ii) Where only an HTML page could be fetched, I say so and tag the body statements `ABSTRACT`. Section numbers (S0 to S10) are new; `DERIVED` notes continue as N12 to N18. Notes N1 to N11 are in the Session 1 brief and Addendum A.

### S0. Headline findings

1. **Claim 5a (rotating optical clusters held by stepped phases) is confirmed, with limits.** A ring of N >= 4 solitons with phase step 2 pi m/N, 0 < step <= pi/2, held by attraction balanced by angular momentum: [Desyatnikov & Kivshar 2002](https://arxiv.org/abs/nlin/0112039) `READ`. Every example is a complex envelope with conserved power and angular momentum, rotating, and metastable (a few revolutions at most; tens to thousands of diffraction lengths). None is static, and none is a real field (S1, S5).
2. **Claim 5b (some potential shapes give far longer oscillon lifetimes) is confirmed.** Up to 1e14 cycles ([Olle, Pujolas, Rompineve 2021](https://arxiv.org/abs/2012.13409) `READ`) and in excess of 1e17 years ([Cyncynates & Giurgica-Tiron 2021](https://arxiv.org/abs/2104.02069) `READ`), in monodromy and "frustrated quadratic" potentials, not in the sextic family. For phi^4-phi^6 only beta = 4/3 has numbers (1e4 to 1e5) (S3, S5).
3. **Lifetime against beta: no source for this model.** Two published mechanisms point different ways. Geometric decoupling (A&S, `READ`) says radiation falls as beta rises at fixed amplitude parameter, so a rise. The "longevity cliff" (Cyncynates & Giurgica-Tiron, `READ`) says lifetime rises while a radiation dip migrates to higher frequency, then falls abruptly when the dip passes the energetic-death frequency. Session 1's two measured dips migrate upward with beta; a two-point extrapolation puts the crossing at beta of about 1.6 to 2.1 (`DERIVED`, low confidence, N17). So prediction A2 (monotone, no fall) is neither supported nor excluded; CD's "rise then fall" has a published mechanism, but not the stated cause ("stable branch narrows") (S3.4, S7).
4. **Novelty search: no source found for geometric binding of real-field oscillons in 3D.** The nearest published arrangements each need something this model lacks: gravity (boson-star chains), a second light real field (flat-space Friedberg-Lee-Sirlin chains, unstable beyond a dipole), complex-field angular momentum (optics), oscillatory tails (dissipative and fourth-order-dispersion solitons), or a trap (BEC). Search record in S9 (S4).
5. **An exact no-go for radiation-free patterns with an antisymmetry plane.** For a real field with an even potential, if the configuration is odd under a reflection (phase 0 on one side, phase 180 on the mirror side), the field vanishes on that plane and the momentum flux through it is positive at every instant: the two halves are pushed apart, for any potential. Without radiation that forbids any bound state; with radiation, the momentum leaving through the far surface enters the balance and this argument alone does not settle it. A published complex-field version for stationary (radiation-free) states: [Cunha, Herdeiro, Radu, Shnir 2022](https://arxiv.org/abs/2210.01833) `READ`. Alternating 0/180 rings, dipoles and chains therefore have no static bound state; optical necklace beams expand for this reason ([Soljacic, Sears, Segev 1998](http://www.mit.edu/~soljacic/necklace.pdf) `READ`) (S6.3, N15).
6. **A 3D tail-force formula exists** ([Malomed 1998](https://arxiv.org/abs/patt-sol/9809009) `READ`): U = -4 pi |A|^2 cos(psi) exp(-kappa R)/R for solitons with tail A exp(-kappa r)/r in cubic-quintic NLS. Mapped to a real field it gives the same prefactor 2 pi as the Session 1 derivation (`DERIVED`, N13). The N-transfer rate has the same prefactor with sin in place of cos (`DERIVED`, N14).
7. **At 90 degrees the leading force vanishes but the pair repels, with large N transfer**, in the 3D cubic-quintic NLS (the non-relativistic limit of our model): [Burlak & Malomed 2018](https://arxiv.org/abs/1805.09173) `READ`; pair K = 0.062 each, d = 10, energy asymmetry up to 0.76 at 90 degrees (S6.2).
8. **CD's two WP-D numbers reproduce, and both sit deep in the overlap region.** The five-lump ring with 72 degree steps balances at kappa R = 1.0695 and its collective radial mode is restoring; the octahedral shell around an anti-phase centre balances at kappa R = 1.591 and is not restoring (`DERIVED`, N16). For the Session 1 primary lump (kappa = 0.4166, half-width 3.344) these are neighbour spacings of 0.90 and 1.6 half-widths. In the static point model the ring still has four negative normal modes of 15, and a screened-Earnshaw identity (N16) shows any balance with net attractive single-lump energy is unstable in at least one direction.
9. **A spinning real scalar oscillon: no source found.** Spinning real vector oscillons exist (`READ`); 3D cubic-quintic NLS has completely stable vortex tori (`ABSTRACT`); complex Q-balls spin with J = nQ but small-J versions radiate (`READ`); rotating Bose stars are unstable for attractive self-interaction (`ABSTRACT`). In a real field J is exactly conserved (rotation symmetry) and equals m N in the non-relativistic limit (`DERIVED`, N12) (S2).
10. **Ring clusters in optics sit at a radially restoring balance but are metastable in the azimuthal direction.** An excited N = 5 cluster oscillates about its equilibrium radius (READ); noise breaks staircase-phase clusters by azimuthal instability (`ABSTRACT`; `READ` for the 3D two-colour case, thousands of diffraction lengths) (S6.5).

### S1. Bound arrangements of several solitons, oscillons and Q-balls

Rows list: what holds it, whether it is static / rotating / breathing / quasi-stable, dimension, lifetime. "Complex" means a field with a conserved charge; our model is one real field.

#### S1.1 Optics and nonlinear Schrodinger-type systems

| System | Source, tag | Dim | What holds it | Behaviour | Lifetime |
|---|---|---|---|---|---|
| Necklace beams, cubic NLS | [Soljacic, Sears, Segev, PRL 81, 4851 (1998)](http://www.mit.edu/~soljacic/necklace.pdf) `READ` | 2 transverse + propagation | Ring of "pearls" with phase pi between neighbours; neighbours repel, net force on each pearl is radially outward | Quasi-stationary, slowly expanding ring; expansion slower for more pearls; stable if the azimuthal period is at most the ring width | "Very large distances" (the number was lost in extraction); the review ([nlin/0501026](https://arxiv.org/abs/nlin/0501026) `READ`) says several tenths (sic) of diffraction lengths |
| Rotating soliton clusters, saturable Kerr (s = 0.5) | [Desyatnikov & Kivshar, PRL 88, 053901 (2002)](https://arxiv.org/abs/nlin/0112039) `READ` | 2 transverse + propagation | Staircase phase theta = 2 pi m/N around a ring; attraction (needs 0 < theta <= pi/2, so N >= 4m) balanced by angular momentum. N = 3: no bound state. m = 0: collapse and fusion. theta > pi/2: expands | Rotating, stationary at a minimum of the effective potential. N = 5, m = 1: R0 = 3 (single-soliton Gaussian width a = 1.623), angular velocity 0.157 measured vs 0.154 from M/I; an excited ring oscillates in radius with period 22. N = 4, m = 1: R0 = 3.8, angular velocity 0.042. N = 8, m = 2: R0 = 5.5, 0.092 | "Generally metastable, symmetry-breaking instability"; N = 4 shown to 60 diffraction lengths (about 0.4 revolution at 0.042, my arithmetic) |
| Same, vector (second incoherent component) | same paper `READ` | same | Incoherent attraction between components balances coherent repulsion and the centrifugal term; traps 2 and 3 solitons too | Rotating, with internal oscillations | A four-soliton vector cluster rotated by about 11.25 pi at z = 225 diffraction lengths (review `READ`) |
| 2D cubic-quintic necklaces carrying angular momentum | [Mihalache et al., PRE 68, 046612 (2003)](https://www.osti.gov/etdeweb/biblio/20754799) `ABSTRACT` | 2 transverse + propagation | Competing cubic (focusing) and quintic (defocusing) nonlinearity; interaction-potential equilibrium | Quasi-stable, persistent radius oscillations; survive added noise | "Very long evolution" (no number in the abstract) |
| 3D cubic-quintic clusters (light bullets) | [Mihalache et al., J. Opt. B 6, S333 (2004)](https://www.osti.gov/etdeweb/biblio/20508036) `ABSTRACT` | 3 (2 transverse + time) | Staircase or ramp phase imprints angular momentum; N = 4, 5, 6 | Robust rotating clusters without noise; with noise they break into single solitons or merge into a vortex torus; staircase phase gives azimuthal instability, ramp phase fuses into a torus | Not retrievable (abstract only) |
| Same, as described by later authors | [Burlak & Malomed 2018](https://arxiv.org/abs/1805.09173) `READ` (introduction, secondhand for the cluster paper) | 3 | n/a | "Slow merger or expansion of the cluster"; in free space the cubic-quintic NLS gives no persistent bound states of 3D solitons, either molecules or necklaces | n/a |
| 3D two-colour light-bullet molecules, chi(2) + defocusing chi(3) | [Crasovan et al., PRE 67, 046610 (2003)](https://arxiv.org/abs/nlin/0301029) `READ` | 3 | Staircase phase M = 1, competing nonlinearities weaken the soliton-soliton force. N = 5 (R0 = 16) and N = 6 (R0 = 12), noise variance 0.1 added | Rotating (0.0027 rad per unit z for N = 4, 0.0014 for N = 5); M = 0 fuses in 100 to 150 units; M = 2, 3 with overlapping solitons expand; N = 6, M = 2, R0 > 22 changes radius moderately over more than 600 diffraction lengths | Thousands of diffraction lengths, then quasi-periodic shrink and expand and decay into unequal fragments; a staircase-phase cluster ends by splitting in two, a ramp-phase one develops into a vortex torus |
| Necklace with a flat-top confinement, two components, cubic-quintic | [Zezyulin 2026](https://arxiv.org/abs/2608.11885) `READ` (introduction and abstract) | 2 transverse + propagation | Repulsion of out-of-phase neighbours in one component balanced by the boundary of a flat-top region in the other; the same paper lists the other known strategies (phase imprint with angular momentum, quadratic, nonlocal, cubic-quintic, traps, vector coupling) | Metastable | About 100 diffraction lengths with noise |
| Nonlocal media | search snippets of [arXiv:0803.4398](https://arxiv.org/abs/0803.4398) and Buccoliero et al., PRL 98, 053901 (2007) (titles only) `SECONDHAND` | 2 | Nonlocal response stabilises dipole solitons; the snippet says higher-order multipoles and necklaces stayed unstable | Rotating dipoles stable | not read |
| Dissipative cavity-soliton clusters | review sec. 8.1 ([nlin/0501026](https://arxiv.org/abs/nlin/0501026) `READ`) | 2 | Oscillating effective potential from oscillatory tails, discrete equilibrium distances | Static or rotating | n/a |

#### S1.2 Oscillatory-tail bound states: the open lead from search 16, traced

The "stationary two-soliton bound states at phase differences 0, pi, +/- pi/2" lead traces to systems whose single-soliton tails oscillate.

- **Mechanism.** Overlap of oscillating tails gives an interaction potential U(T, phi) = U0 exp(i phi - chi T) cos(omega T) cos(Omega T) in a third-order-dispersion model ([Sakaguchi, Skryabin, Malomed 2018](https://arxiv.org/abs/1805.02190) `READ`, Eq. 5 and 6). Its minima are the equilibrium separations T = (pi/2 omega)(1 + 2m) and (pi/2 Omega)(1 + 2n); both static and oscillating pairs were found by simulation. The same potential in 3D with oscillation wavenumber q: [Malomed 1998](https://arxiv.org/abs/patt-sol/9809009) `READ` Eq. 29 and 31 (S6.1).
- **Primary sources.** Conservative case with fourth-order dispersion: families of two- and multi-soliton stationary bound states exist where single solitons have radiationless oscillating tails, and a stability criterion is derived for stationary multisoliton states in Hamiltonian systems ([Buryak & Akhmediev, PRE 51, 3572 (1995)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.51.3572) `ABSTRACT`; the abstract does not list phases). Dissipative case: small gain and loss make the pulse tails oscillate, which allows multi-pulse bound states that the abstract calls only weakly stable ([Malomed, PRA 44, 6954 (1991)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.44.6954) `ABSTRACT`). The phase classification (0 and pi pairs are saddles, +/- pi/2 pairs are stable spirals, in the anomalous-dispersion regime of the complex Ginzburg-Landau equation) is in the 2026 review by [Cui, Lu, Zhang, Mao, Malomed](https://arxiv.org/abs/2604.28090) `READ` (section on stationary points, citing Akhmediev, Ankiewicz, Soto-Crespo, PRL 79, 4047 (1997), not opened). Malomed 1998 `READ` adds that the effective mass of the phase degree of freedom is negative, which is why 0 and pi states are saddles and pi/2 states are not; the pi/2 state has an extremely weak instability that makes it a slowly unwinding spiral with R and psi oscillating in a narrow range.
- **Dimension and lifetime.** 1D temporal solitons in fibres and lasers; the 2D and 3D versions are proposed only (Malomed 1998 suggests lattices and triangle or tetrahedron "covalent molecules" for dissipative systems). Dissipative states are fixed by system parameters, not by initial data (review `READ`).
- **Status for us.** Needs tails with Im kappa not equal to 0. Our lumps (omega < m) have monotone exp(-kappa r)/r tails. The outgoing 3 omega radiation field does oscillate, but no source treats it as a binding agent: "no source found" (search 61).

#### S1.3 Bose-Einstein condensates

| System | Source, tag | Dim | What holds it | Behaviour, lifetime |
|---|---|---|---|---|
| 7Li soliton trains | [Strecker et al. 2002](https://arxiv.org/abs/cond-mat/0204532) `READ` | quasi-1D trap | Alternating phase makes neighbours repel; the axial trap holds the train | Many oscillation cycles in the trap without spreading |
| Soliton molecule from an in-phase pair | [Al Khawaja & Stoof 2011](https://arxiv.org/abs/1012.3762) `READ` (abstract, introduction) | 1D | Exact two-soliton interaction energy has the form of a Morse potential and depends on separation and relative phase; in-phase pairs with small separation and velocity always bind. The cosine law at large distance is credited to Karpman & Solov'ev (perturbation), Gordon (exact), Anderson & Lisak (variational) | Bound pair; 1D cubic NLS, no 3D analogue |
| Rotating two-soliton "molecule" of 3D solitons | [Burlak & Malomed 2018](https://arxiv.org/abs/1805.09173) `READ` | 3D with a 2D harmonic trap (V0 = 1e-4) | Trap plus opposite transverse kicks (velocity 0.3), separation 10, phase pi/2 | Rotation period about 150; three revolutions shown to t = 450; symmetry between the two solitons breaks and partly restores. Without the trap the pair rotates 180 degrees and separates |

#### S1.4 Q-balls and boson stars (complex fields)

| System | Source, tag | Dim | What holds it | Behaviour, lifetime |
|---|---|---|---|---|
| Charge-swapping Q-balls | [Copeland, Saffin, Zhou 2014](https://arxiv.org/abs/1409.3232) `READ` | field simulations 2+1D shown; complex, potential V = m^2 phi^2/2 (1 + K ln(phi^2/2M^2)), K = -0.1, M = 10 m | A Q-ball and an anti-Q-ball with cores overlapping, separation d at most about 2 sigma (sigma = 3.16/m); total charge zero; initial relative phase irrelevant | Single blob whose energy density stays like one Q-ball while the charges swap at a frequency below m; lives at least O(1e4) oscillation periods. Not separated lumps |
| Same, reinterpreted | [Alonso-Izquierdo et al. 2025](https://arxiv.org/abs/2504.17382) `ABSTRACT` (the full PDF exceeds the fetch limit; body statements came from an HTML summary and are unverified) | 1+1, complex phi^6 | A real oscillon carrying an excitation in the imaginary direction, not a molecule of Q-balls; a two-variable collective model (relative phase, amplitude) captures the swap | long-lived; lifetimes of charge-swapping Q-balls mapped by [Xie, Saffin, Zhou](https://arxiv.org/abs/2101.06988) `ABSTRACT` (four phases: relaxation, plateau, fast decay, oscillon plateau) |
| Flat-space dipoles and chains, Friedberg-Lee-Sirlin | [Jaramillo & Zhou 2024](https://arxiv.org/abs/2411.08985) `READ` | 3+1D, flat limit | Complex field plus a real field; attraction from a light or massless real mediator, repulsion from the pi relative phase; static, axisymmetric, 2, 4 and 6 lumps along the axis (massless mediator, omega = 0.9) | Evolved chains decay to simpler states before t = 2000; for a massive mediator (mu = 0.1 and 0.125 tested) the dipole survives but the four- and six-lump chains do not. The authors state no other non-rotating flat-space Q-ball chains were known |
| Boson-star dipoles and chains | [Cunha et al. 2022](https://arxiv.org/abs/2210.01833) `READ`; [Herdeiro et al. 2021](https://arxiv.org/abs/2101.06442) `READ` (abstract; sextic complex potential plus gravity); [Shnir 2021](https://arxiv.org/abs/2101.07552) `READ` | 3+1D axisymmetric, with gravity | Long-range gravity balances short-range pi-phase scalar repulsion; free-field dipoles are unstable and a Coleman-type self-interaction stabilises them (statement in Jaramillo & Zhou's introduction, `READ`) | Static solutions; no flat-space limit |
| Rotating chains | [Gervalle 2022](https://arxiv.org/abs/2206.03982) `ABSTRACT` | 3+1D with gravity | Rotation; the flat limit reduces to chains of Q-balls | Stationary |
| No-go for static pairs | Cunha et al. 2022 sec. 5.3 `READ` | flat, stationary single-frequency complex field | Odd-parity static two-Q-ball states do not exist for any self-interaction potential; rotation (dipole-dipole interaction of spinning constituents) evades it; even parity not excluded | Exact for stationary states |

#### S1.5 Oscillons: collisions and bound states

- Real oscillons in 3+1D: in-phase pairs merge (energy loss about 30%), pi out of phase bounce, mismatched pairs are stochastic: [Amin & Mou 2020](https://arxiv.org/abs/2009.11337), [Xue et al. 2025](https://arxiv.org/abs/2510.01597), `READ` in the Session 1 brief section 4.1. 2D: [Hindmarsh & Salmi 2008](https://arxiv.org/abs/0712.0614) `READ`. 1+1D: forces exponentially decaying and modulated by relative phase ([Campos, Manton, Mohammadi 2026](https://arxiv.org/abs/2605.12633) `READ`; v1 only, the verbal sign sentence is unchanged from the version noted in section 4.2 of the Session 1 brief). No static pair, ring or polyhedron of oscillons found at any dimension.
- A modulated oscillon is a bound state of two unmodulated oscillons ([Blaschke et al. 2025](https://arxiv.org/abs/2502.09136) `READ`, 1+1D); [Unified theory of oscillons and modes](https://arxiv.org/abs/2606.22680) `ABSTRACT` adds wobblerons (oscillon-kink bound states).
- sine-Gordon breather pairs: synchronised pairs attract, antisynchronised repel, both exponentially with separation ([Nishida et al., PRE 80, 036603 (2009)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.80.036603) `ABSTRACT`, 1+1D).
- Pattern-forming systems (not Klein-Gordon): bound structures of oscillons of equal and opposite polarity in a Swift-Hohenberg model ([Crawford & Riecke 1998](https://arxiv.org/abs/patt-sol/9804005) `ABSTRACT`), dipoles, tetramers and chains in vibrated granular layers ([Tsimring & Aranson 1997](https://arxiv.org/abs/patt-sol/9703009) `READ`, Session 1). The mechanism is the oscillatory spatial tail of that class of system. Do not import it.
- Axion-clump collisions in the non-relativistic gravity-plus-attraction limit: merger for any phase difference except pi, which keeps the clumps apart by destructive interference; a phase of pi/2 gives the merged clump a recoil ([Hertzberg, Li, Schiappacasse 2020](https://arxiv.org/abs/2005.02405) `READ`, section 3.5; head-on, relative velocity set by the initial conditions). Context only.

#### S1.6 Which mechanisms need something our model lacks

| Mechanism | What it needs | Our model | Carries over? |
|---|---|---|---|
| Staircase phase with angular momentum (centrifugal balance) | Conserved N and J; complex envelope | J is exactly conserved (rotation symmetry); N is only an adiabatic invariant (Mukaida, Ibe `READ` in Session 1); the lump radiates | Plausible in principle, never tested for a real field. The relativistic complex case is known to be fragile: relativistic corrections give exponentially growing modes for rotating 2+1D lumps with quartic self-interaction ([Galushkina et al. 2025](https://journals.aps.org/prd/abstract/10.1103/pdrb-sh7n) `ABSTRACT`) |
| Oscillatory tails | Im kappa not equal to 0 | Monotone tails for omega < m | No |
| Nonlocality | Nonlocal response | None | No |
| Competing nonlinearity (cubic-quintic) | Attractive quartic plus repulsive sextic | Present (the non-relativistic limit of the model, S2.3) | Yes, and the free-space 3D cubic-quintic results (no persistent bound states, slow merger or expansion) are the closest published analogue |
| Incoherent attraction (second component, or detuned frequency) | A second field, or lumps of different omega | One field; lumps of different omega exist on the stable branch | Open: untested for a real Klein-Gordon field; optical vector clusters rely on it (DK 2002, Zezyulin 2026, `READ`) |
| Long-range phase-independent attraction | Gravity, or a light second field | None | No (boson-star and FLS chains) |
| Trap | External potential | None | No (BEC trains, rotating molecule) |
| Phase pi repulsion plus attraction from another source | Both | Only the phase-dependent tail force, both signs from cos(delta) | Fails the radiation-free no-go for any pattern with an antisymmetry plane (S6.3) |

### S2. Angular momentum: spinning lumps and stepped-phase patterns

#### S2.1 What a stepped phase carries (`DERIVED` N12)

`DERIVED` note N12 (my algebra; re-derive before relying on it). For a real field with a rotation-invariant potential, J_z = -integral of phi_t (x d_y - y d_x) phi is exactly conserved. Write phi = (psi e^(-i m t) + c.c.)/sqrt(2m), the reduction behind the approximate U(1) of Mukaida and Ibe (`READ`, Session 1 brief section 2.7). Period-averaged, J_z = Im integral of psi* d_phi psi and N = integral |psi|^2 (hbar = 1). A lump with azimuthal twist, psi = f(r, theta) e^(i m_t phi), has J_z = m_t N exactly in that limit. In a real field this is a pattern rotating as cos(omega t - m_t phi), so it is a legitimate object for the real equation; whether it is a stable one is the question.

For a ring of N_l separate lumps with uniform phase step 2 pi m_t/N_l:
- A pure **staircase** (each lump uniform in phase) carries almost no J while the lumps do not overlap. In [Desyatnikov & Kivshar 2002](https://arxiv.org/abs/nlin/0112039) `READ`, Eq. 3, M is proportional to A^2 times a sum of exp(Y_nk) |r_n x r_k| sin(theta_nk), with Y_nk = -|r_n - r_k|^2/(4 a^2): every term is an overlap term. Their equilibria sit where that overlap is large (N = 5, m = 1: R0 = 3 with a = 1.623, so the nearest-neighbour factor exp(Y) is 0.31, `DERIVED` arithmetic).
- A **ramp** (a phase gradient across each lump, tangential wavenumber m_t/R) gives each lump orbital momentum, and J = m_t N from the lumps' own motion. This is the case the centrifugal-barrier picture describes. Optical results differ between the two (S1.1).
- Patterns with all phases 0 or 180 are real, so J = 0 exactly in this limit.
- Mirror images exist (m_t to -m_t); the field sign flip shifts every phase by 180 degrees.

#### S2.2 What is known

| Object | Source, tag | Finding |
|---|---|---|
| Spinning Q-balls, complex, 3+1D | [Volkov & Wohnert 2002](https://arxiv.org/abs/hep-th/0205157) `ABSTRACT`; [Almumin et al. 2024](https://arxiv.org/abs/2302.11589) `READ` (introduction) | Exist; first explicit spinning solitons in 3+1D Minkowski space. Angular momentum is quantised, J = n Q, and pushes the field away from the centre (the Q-ball stability paper [arXiv:2509.18656](https://arxiv.org/abs/2509.18656) `READ`, introduction, cites spinning Q-balls this way but does not test their stability). A toroidal energy density is stated in a search snippet only (`SECONDHAND`) |
| Slowly rotating Q-balls | Almumin et al. 2024 `READ` (sections 4, 5) | Small J is possible only as a metastable, radiating state: a perturbative rotating solution needs infinitely many (L, M) modes, high-M modes are radiation; the authors work under the assumption that the state is classically unstable and estimate a lifetime; no number retrieved |
| Spinning boson stars, gravity | [Sanchis-Gual et al. 2019](https://arxiv.org/abs/1907.12565) `READ` (abstract) | In the complex scalar case a non-axisymmetric instability always develops and ejects the angular momentum (or collapses an existing star); no such instability for Proca stars. [Delgado, Herdeiro, Radu 2020](https://arxiv.org/abs/2005.05982) `READ` (introduction): self-interactions might mitigate it |
| Rotating Bose stars, non-relativistic, gravity | [Dmitriev et al. 2021](https://arxiv.org/abs/2104.00962) `ABSTRACT` | Unstable at every l not equal to 0 for attractive or negligible self-interaction, with lifetimes of order the inverse binding energy; stable for l = 1 with strong enough repulsive self-interaction. [Hertzberg & Schiappacasse 2018](https://arxiv.org/abs/1804.07255) `READ`: minimum-energy clumps at fixed N and fixed angular momentum exist, with a maximum mass that grows with L (variational, not a dynamical test) |
| 3D cubic-quintic NLS (non-relativistic limit of our model, S2.3) | [Mihalache et al. 2002](https://arxiv.org/abs/nlin/0201029) `ABSTRACT`; [Burlak & Malomed 2018](https://arxiv.org/abs/1805.09173) `READ` (introduction) | "Completely stable" spinning solitons (vortex tori) with S = 1 exist in 3D, in a window of the propagation constant; in 2D, S = 1, 2, 3 are partly stable and S > 3 only in extremely narrow windows |
| Rotating lumps, 2+1D, relativistic, complex, quartic | [Galushkina et al. 2025](https://journals.aps.org/prd/abstract/10.1103/pdrb-sh7n) `ABSTRACT` | Relativistic corrections break a conformal symmetry of the non-relativistic limit and give exponentially growing modes |
| Real vector field, 3+1D | [Zhang, Jain, Amin 2022](https://arxiv.org/abs/2111.08700) `READ` | "Spinning" (circularly polarised) oscillons exist, with intrinsic spin equal to hbar times the particle number and no orbital angular momentum; with absorbing boundaries no decay for at least about 1e3/m. A different object from a scalar lump |
| Real scalar, spherical oscillon plus angular perturbations | [Cyncynates & Giurgica-Tiron 2021](https://arxiv.org/abs/2104.02069) `READ` (appendix C.3) | Higher angular number l is generally more stable; low l is case-dependent. Not a spinning state |
| Real scalar spinning oscillon | no source found (searches 50, 51, 52, 75, 76) | |

#### S2.3 Why the non-relativistic limit matters here

The profile equation of the plan, with Phi = a Psi(rho), becomes Psi'' + 2 Psi'/rho - alpha^2 Psi + (3/4) Psi^3 - (5/8) Psi^5 = 0 (Session 1 brief section 1.3, `DERIVED`): cubic focusing plus quintic defocusing, the cubic-quintic NLS. So the 3D cubic-quintic results in S1 and S2.2 are the closest published analogue, valid as omega approaches m. Session 1's primary has omega = 0.909, so kappa^2/m^2 = 0.17 and relativistic corrections are not small; the relativistic complex case is the one reported to be fragile (Galushkina et al., `ABSTRACT`).

### S3. Oscillon lifetime against potential shape

#### S3.1 Numbers, with potentials and conditions (units 1/m unless stated)

| Source, tag | Potential and parameters | Lifetime | Method and note |
|---|---|---|---|
| Ibe et al. 2019, Mukaida et al. 2017, Li et al. 2025 (Session 1 brief section 2.6, `READ`) | phi^4-phi^6, beta = 4/3, 3D radial | about 1e4 to 1e5; Mukaida ran T = 1e5 without death from R_ini = 10; Gamma from about 1e-7 (dips) to 4e-4 (peaks) | Radial |
| Kasuya, Kawasaki, Takahashi 2003 (Session 1, `READ`) | phi^4-phi^6, beta about 1.67, 2D lattice, from noise | O(1e3) | One parameter set |
| [Amin & Shirokoff 2010](https://arxiv.org/abs/1002.3380) `READ` (sec. 4.3, 6) | phi^4-phi^6, large g (large beta), flat-top oscillons, 3+1D | Argument only: at fixed alpha the profile in phi(t, x) is g-independent, so its Fourier transform narrows with g and the amplitude at the radiating wavenumber k_r of about sqrt(8) m can be made as small as wanted. Their lifetime formula is for loss due to expansion, T of order g exp(alpha^2/(gH)); it does not apply in flat space | Analytic; the paper says numerical checks are still needed |
| [Amin, Easther, Finkel 2010](https://arxiv.org/abs/1009.2505) `READ` | same model, (lambda/g)^2 = 0.1 fiducial (beta about 33) and 0.2 (beta about 17), expanding background, 3D lattice | Oscillons persist over several Hubble times (H^-1 of order 1e3/m); "stable on timescales longer than the post-inflationary Hubble time". No lifetime-against-g scan | Lattice |
| [Drogosz et al. 2026](https://arxiv.org/abs/2607.23616) `READ` (Addendum A2, section A2) and `ABSTRACT` | Higgs-like real field with a sextic term chi_6, 3D radial; different potential | about 6e3 at chi_6 = 0 to above 2e5 by about 0.2 to 0.4, slowly falling to 0.6, a multi-peak structure near the onset of an extra minimum; the abstract says several orders of magnitude for the physical mass ratio | Context only, non-monotone in the sextic coupling |
| [Olle, Pujolas, Rompineve 2020](https://arxiv.org/abs/1906.06352) `READ` | monodromy-type family with exponent p (the 2021 paper's Eq. 14, V = m^2 F^2 ((1 + phi^2/F^2)^p - 1)/(2p), says it is the same family), 3D radial | generic potentials about 1e3; p = 1/2: 3.4e8; p = -1/2: above 6.5e8 (not seen to decay in 1200 CPU hours) | Lattice and radial |
| [Olle, Pujolas, Rompineve 2021](https://arxiv.org/abs/2012.13409) `READ` | same family; "relax and fast-forward" (integrate Gamma(omega) along the attractor) | p to -infinity: about 1e5; p = -5, -4, -2.5, -2: estimated 8.45e5, 1.39e6, 4.74e6, 1.13e7 against explicit evolution 9e5, 1.5e6, 5e6, 1.2e7 (errors 6.1, 7.3, 5.2, 5.8 per cent); p = 0.95: about 1e14, grows exponentially as p to 1 | Exact eternal oscillons for V of order phi^2 log phi in 1+1D and 3+1D |
| [Cyncynates & Giurgica-Tiron 2021](https://arxiv.org/abs/2104.02069) `READ` | periodic potentials with parity, 3D; and a "frustrated quadratic" (mass m at small phi, smaller m_f at large phi) | Most of parameter space 1e2 to 1e4 oscillations; tuning two Fourier coefficients reaches 1e14 cycles; each added coefficient adds a longer island (log10 of mT about 0 to 4, 4 to 9, 9 to 14); frustrated quadratic: log10(mT) = 28 (m_f/m)^2 - 11 (three coefficients), 39 (m_f/m)^2 - 21 (four); the abstract states more than 1e17 years | Single-non-perturbative-harmonic quasibreather estimate; radiated power compared with simulation for selected potentials (their Fig. 11); the longest lifetimes cannot be simulated |
| Zhang et al. 2020, Fodor 2006, Copeland 1995 (Session 1 brief section 2.6, `READ`) | flattened potentials; cosine; phi^4 double well | 1e8 to 1e9; 1e3; 1e3 to 1e4 | Radial |
| [Antusch, Cefala, Torrenti 2019](https://arxiv.org/abs/1907.00611) `ABSTRACT` | hilltop potentials, 3+1D lattice, expanding background | 4 to 5 e-folds; stronger breathing means shorter life | Context only |

For the phi^4-phi^6 family the only 3D or radial lifetime numbers I found are at beta = 4/3 (and the 2D noise-seeded one at 1.67). Nothing between 1.67 and 17, and nothing below 4/3, in this model: no source found (searches 29, 32 of the Session 1 log, and 70).

#### S3.2 Shape conditions

[Olle et al. 2021](https://arxiv.org/abs/2012.13409) `READ` (section 2): two properties of V boost lifetime, flattening and positivity of V''; positivity of V'' matters more, and a small negative |V''|/m^2 at the core is tolerated. [Olle et al. 2020](https://arxiv.org/abs/1906.06352) `READ` (footnote 8, Addendum A2) states the condition as a heuristic. For our family V''_min/m^2 = 1 - 3/(2 beta) (N5), so V'' is non-negative from beta = 1.5. The binding potential Delta V = m^2 phi^2/2 - V = g4 phi^4/24 - g6 phi^6/720 is positive only for phi^2 < 30 g4/g6 (`DERIVED`), which is phi < 8.66 for the Session 1 parameters (g4 = 0.30, g6 = 0.12); the primary lump's core amplitude is 6.89, inside that window. Above it V rises back above the quadratic, so "flattening" exists only over a finite amplitude range. Neither source tests this model.

#### S3.3 The longevity cliff

[Cyncynates & Giurgica-Tiron 2021](https://arxiv.org/abs/2104.02069) `READ` (introduction, section III, Fig. 9). Three effects: destructive interference (a radiation channel's source has a zero, giving a dip in radiated power); geometric decoupling (the lump grows as 1/sqrt(m^2 - omega^2), which decouples a smooth bulk from radiation of wavelength 2 pi/(n omega) and cuts radiated power exponentially toward the end of life); energetic death (the lump must dissipate when its energy stops falling with omega, in three or more dimensions). Their words for the landscape: an island of longevity starts when a dip emerges at low frequency; as the dip migrates to higher frequency geometric decoupling enhances it and lifetime rises, until the dip moves beyond the frequency of energetic death, which produces a longevity cliff (Fig. 9 caption describes the same for a two-parameter periodic potential). Their lifetime axis is a potential parameter, here V_1, V_2; the beta axis of our family is the analogue.

#### S3.4 What this implies for lifetime against beta (`DERIVED` N17)

`DERIVED` note N17 (low confidence). Session 1 measured two rate dips: (beta, omega, alpha^2) = (0.85, 0.779, 0.1003) and (4/3, 0.909, 0.0695) (`SESSION_01_REPORT.md` section 2, Addendum A2 N7). Both move to higher omega as beta rises, as the cliff picture needs. The energetic-death frequency has alpha^2 about 0.0236 at every beta (N8, universal-curve position of the N minimum, Li 2025 `READ`), equivalently omega_death = sqrt(1 - 0.0788/beta): 0.9692 at 4/3. A straight line through the two dips in alpha^2 reaches 0.0236 at beta = 2.05; a straight line through them in omega (slope 0.269 per unit beta) meets omega_death at beta = 1.58. Range 1.6 to 2.1. Assumptions: both dips are the same zero of the 3 omega source; migration is linear; no new dip appears at lower omega (CGT report new islands appear as parameters are added, but not as one parameter varies).

What the sources say about A2 to A5, together:
- Rise with beta: geometric decoupling, an exponential suppression of radiation as the lump grows relative to the radiation wavelength; for our family (k_3 L)^2 = 2.4 beta - 9 alpha^2 (N7). A&S's argument (`READ`) and Olle's V'' heuristic both point this way for beta of 1.5 and above.
- Fall: the cliff, for a profile tuned to a dip. A max-over-omega "best profile" can hide it if the non-dip lifetime has already caught up. Nothing in the literature tells which happens at beta of 1.6 to 2.1 in this model.
- "Branch narrows": in the universal variable alpha^2 the stable branch is fixed (0.0236 to 0.16875, N8 and Session 1 section 1.3). In omega it narrows as beta rises, from [0.760, 0.970] at 4/3 to [0.848, 0.980] at 2 and [0.942, 0.992] at 5 (`DERIVED` from sqrt(1 - 9/(16 beta)) and sqrt(1 - 0.0788/beta)). No source ties lifetime to branch width.
- Permanence: energetic death is generic for power-series potentials (van Dissel et al. 2023, `READ`, Session 1). Eternal oscillons exist only for non-polynomial potentials such as phi^2 log phi (Olle 2021 `READ`). The sextic family is a power series, so A5 agrees.

### S4. Novelty search for geometric binding of oscillons in 3D

Target: static or quasi-static arrangements of several distinct oscillons in a real scalar field in 3D (rings, polyhedra, chains, lattices), held by a phase pattern. No novelty claim is made here. The record is in S9 (searches 41 to 82). Outcome:

- **Found for real scalar fields in 3D:** nothing. No ring, polyhedron, chain or lattice of oscillons, no measured pair force against separation and phase, no report of a bound pair. Pair collisions in 3D and 3+1D are covered by Amin & Mou, Xue et al. (S1.5).
- **Found for real fields in 1D and 2D:** forces between pairs of oscillons and breathers with exponential distance dependence and phase dependence, and bound structures in pattern-forming (non-Klein-Gordon) systems.
- **Found for related fields:** the arrangements of S1: optical clusters (complex NLS, stepped phase), Q-ball and boson-star chains (complex, with a second field or gravity), BEC trains (trapped). Each needs a mechanism that a single real Klein-Gordon field does not provide (S1.6).
- **Limits of the search.** The tool is web search, not a full-text or citation query of arXiv and INSPIRE. "No source found" means these queries returned nothing, not that nothing exists. Searches for the oscillon cases returned the same few papers repeatedly (Campos et al., Amin & Mou, Xue et al., Blaschke et al.), which is some evidence that the field has little here.
- **A caution that cuts both ways.** Because the non-relativistic limit of the model is the cubic-quintic NLS, a result in this project may be a known cubic-quintic result in disguise (clusters, merging, symmetry breaking at 90 degrees). What would be new is the real-field, omega/m of about 0.9 regime, radiation included. That is a statement about where to search next, not a claim.

### S5. The two claims from Claude Desktop

**(a) "Optics has rotating soliton clusters held by stepped phases." Confirmed, with these qualifications.**
- Introduced in [Desyatnikov & Kivshar, PRL 88, 053901 (2002)](https://arxiv.org/abs/nlin/0112039) `READ`; extended to 3D spatiotemporal clusters ([Crasovan et al. 2003](https://arxiv.org/abs/nlin/0301029) `READ`; [Mihalache et al. 2004](https://www.osti.gov/etdeweb/biblio/20508036) `ABSTRACT`) and to vector clusters.
- The phase step is 2 pi m/N with N >= 4m and step <= pi/2; N = 3 has none; m = 0 fuses; larger steps expand.
- What holds them is attraction between neighbours balanced by angular momentum (centrifugal effect), which needs a conserved power and a conserved angular momentum. In optics that is a complex envelope.
- They are rotating and metastable, not static, and they last tens (2D saturable) to thousands (3D two-colour, competing quadratic and cubic) of diffraction lengths, which is at most a few revolutions (S1.1).
- Under noise, a staircase phase leads to azimuthal instability and a ramp phase to fusion into a vortex torus (`ABSTRACT`, `READ`).
- In free-space 3D cubic-quintic NLS, later authors say no persistent bound states of 3D solitons form (Burlak & Malomed 2018, `READ`, introduction). That is in tension with the abstract's word "robust" for the 2004 cluster paper; I could not read the paper to see the durations.

**(b) "Some potential shapes give far longer oscillon lifetimes." Confirmed.** Flattened potentials and potentials with V'' positive at large field: 3e8 to above 6.5e8 (Olle et al. 2020), 1e14 cycles (Olle et al. 2021), above 1e17 years from frustrated quadratics (Cyncynates & Giurgica-Tiron), against 1e3 to 1e4 for generic shapes. Table in S3.1. Two cautions. These are different potential families, and the longest numbers come from radial or quasibreather estimates rather than full lattice runs. The phi^4-phi^6 family has numbers only at beta = 4/3 (1e4 to 1e5).

### S6. Interaction law, tail force, and stability of point-lump equilibria

#### S6.1 A 3D tail-force formula: found

[Malomed, PRE 58, 7928 (1998)](https://arxiv.org/abs/patt-sol/9809009) `READ` derives the interaction potential of two far-separated 2D or 3D solitons from the overlap term of the full Hamiltonian, without an artificial overlap radius. Model: i v_t + (1/2) laplacian v + |v|^2 v - alpha |v|^4 v = 0 (the conservative cubic-quintic case; complex Ginzburg-Landau terms optional), H = integral of [|grad V|^2/2 - |V|^4/2 + alpha |V|^6/3 - omega |V|^2], tail V of about A r^(-1) exp(-kappa r) with kappa = sqrt(-2 omega). Result for identical 3D solitons with phase difference psi:

- U_3(R, psi) = -4 pi |A|^2 cos(psi) R^(-1) exp(-kappa R) cos(q R), Eq. 31; q = Im kappa, zero in a conservative model. The 2D form, Eq. 29, has R^(-1/2) and an extra (-1)^s for spin s.
- Conditions: identical solitons (the same omega and the same |A|); lowest order in 1/R only ("an effective particle potential can be consistently defined only at lowest order"); the auxiliary radius must satisfy kappa^-1 << rho << R, so R is many decay lengths.
- Fixed quantities: the profile and omega (H contains -omega N, so U is the interaction part of the grand potential at fixed omega). To first order in the overlap this equals the interaction energy at fixed N (a Legendre-transform property, `DERIVED`), so the force is the same at leading order.
- Consequences stated: in-phase attracts, anti-phase repels, a stationary orbiting pair always sits at a maximum of the effective energy and is unstable, and the phase degree of freedom has negative effective mass.

`DERIVED` note N13 (cross-check with the Session 1 derivation). Take m = 1 and a real-field tail phi = A cos(omega t - theta) exp(-kappa r)/r. With phi = sqrt(2) Re(psi e^(-i t)) the NLS amplitude is |psi|^2 = A^2/2 times the tail, so |A_Malomed|^2 = A^2/2 and U_3 = -2 pi A^2 cos(psi) exp(-kappa R)/R. That is the Session 1 result (section 4.2, momentum flux through the mid-plane), reached by an independent route. The match is exact at omega to m. The Session 1 derivation is exact for the Klein-Gordon equation at any omega because kappa^2 = m^2 - omega^2 enters only through the period-averaged terms, so WP-C should use it and treat the NLS mapping as the check. Neither handles the overlap region.

#### S6.2 Intermediate relative phase and N transfer

- **3D cubic-quintic NLS**, [Burlak & Malomed 2018](https://arxiv.org/abs/1805.09173) `READ`: two identical solitons, K = 0.062 each (tail exp(-sqrt(K) r), so kappa = 0.249 and the decay length is 4.0; d = 10 gives kappa d = 2.5), phase difference phi, placed along a diagonal. Outcomes: phi = 0, merger completed by t = 150, into one nearly fundamental soliton with a residual 10 per cent oscillation of the central density (1D gives a breather instead); phi = pi/4, merger also; phi = pi/2, repulsion with symmetry breaking: two separating solitons with different energies, measure Delta = |E1 - E2|/(E1 + E2) reaching about 0.76 at phi = pi/2; phi = pi, they rebound and Delta stays 0. They place a critical phase between pi/4 and pi/2. The paper notes the lowest-order effective potential is zero at pi/2 (their reference to Malomed 1998), so the repulsion is a higher-order effect, and explains the asymmetry as a mismatch between the "amplitude centre" and the "phase centre" once the solitons acquire opposite velocities. Direction of the transfer is not stated.
- **Q-balls**: [Battye & Sutcliffe 2000](https://arxiv.org/abs/hep-th/0003252) `READ` (Session 1 brief): charge transfer at phases other than 0 and pi, no transfer at pi. Axenides et al. 2000 and Bowcock et al. 2008 (Session 1, `READ`): breathing at the difference of the internal frequencies for unequal omega; the cosine law only for equal frequencies and exactly in or out of phase.
- **Real oscillons**: N exchange: no source found (Session 1 brief section 4.3, and searches 47, 59 to 63). The 1+1D complex-field work on charge swapping is the nearest ([Alonso-Izquierdo et al. 2025](https://arxiv.org/abs/2504.17382) `ABSTRACT`).

`DERIVED` note N14 (my algebra; re-derive). In the non-relativistic limit with m = 1, the number current is j = Im(psi* grad psi). For two tails f1, f2 (real, f1 decaying away from lump 1 and f2 away from lump 2) with phases theta1, theta2 and psi = f1 e^(i theta1) + f2 e^(i theta2), the cross term at the mid-plane is j_x = sin(theta2 - theta1)(f1 f2' - f2 f1') = 2 sin(theta2 - theta1)(kappa + 1/r) f1 f2. Integrating over the plane for kappa d >> 1 with f = A_psi exp(-kappa r)/r gives a flux of 4 pi A_psi^2 sin(theta2 - theta1) exp(-kappa d)/d. Result:
- dN_1/dt = -|U_0(d)| sin(theta2 - theta1), with |U_0| = 4 pi A_psi^2 exp(-kappa d)/d the size of the cosine potential in S6.1 (hbar = 1; N counts quanta).
- Direction, with phi = A cos(omega t - theta): N flows toward the lump with the larger theta, the lump whose oscillation peaks later.
- The prefactor equals the potential depth, so the transfer time is N/(|U_0| |sin delta|), the same exponential as the force with sin in place of cos.
- For a ring with uniform step the inflow from one neighbour equals the outflow to the other, so the net transfer to each lump vanishes. That is Desyatnikov & Kivshar's zero-energy-exchange condition, Eq. 4 (theta = 2 pi m/N).
The result matches the Hamiltonian relation dN/dt = derivative of the interaction energy with respect to the phase. No source gives the rate or direction; the 3D pair result above (Delta = 0.76 at 90 degrees) is a non-perturbative case.

#### S6.3 An exact no-go for radiation-free patterns with an antisymmetry plane

[Cunha, Herdeiro, Radu, Shnir 2022](https://arxiv.org/abs/2210.01833) `READ` (section 5.3). For a complex scalar with any self-interaction U(|Phi|^2) in flat space, stationary Phi = e^(-i omega t) phi(rho, z): integrating the z-component of momentum conservation over the half space z >= 0 gives that the integral of T^z_z over the plane z = 0 must vanish. For an odd-parity field phi vanishes on the plane, and the integrand reduces to phi_1^2 (the first z-derivative squared), whose integral is positive unless phi_1 = 0, and then every higher odd coefficient vanishes too. Conclusion: no static odd-parity pair of Q-balls exists in flat space, for any potential. The authors state that it does not apply to rotating configurations and does not exclude even-parity solutions.

`DERIVED` note N15 (real-field version; my algebra, re-derive before relying on it). For a real field the argument needs no stationarity. Momentum balance for the half space z > 0: d/dt of its z-momentum = (integral over the plane z = 0 of T^zz) minus (z-momentum flux leaving through the far surface), with T^zz = phi_z^2/2 + phi_t^2/2 - (grad_perp phi)^2/2 - V. If the configuration is odd under reflection through the plane (phi(-z, t) = -phi(z, t), exact for V even, which holds here), then phi = phi_t = grad_perp phi = 0 on the plane and V(0) = 0, so T^zz = phi_z^2/2 >= 0 at every time. The plane term pushes the right half away from the left half at every instant, for any potential, any time dependence, and any number of lumps on each side.
- **Radiation-free configurations** (strictly periodic or stationary): the far-surface flux is zero, so the plane term must vanish, which forces phi_z = 0 on the plane and then phi = 0 everywhere. No such bound configuration exists. This is the real-field, time-dependent version of the Cunha et al. result.
- **Radiating configurations** (all oscillons): the far-surface flux is not zero, and the balance becomes plane term = outgoing flux, both non-negative. The argument does not forbid that. Both are of the size of the radiated power (Gamma about 1e-6 to 4e-4 of E per unit time at beta = 4/3, Session 1 brief section 2.6) rather than of the tail force, and the plane term contains the radiation field's own stress. I have not worked out whether a radiation-balanced pair exists; no source found (search 61). Treat the no-go as exact in the radiation-free limit and as the leading effect otherwise.
- **What it covers**: anti-phase pairs, alternating 0/180 rings, 0/180 chains, any polyhedron pattern with a mirror plane that exchanges phases. **What it does not**: patterns whose mirror planes map phases to themselves (for example an octahedron with an anti-phase centre), stepped-phase patterns (a mirror reverses the step, which is a time reversal, so no instantaneous sign constraint), or any pattern with other forces. The plane integral of T^zz is a measurable diagnostic.

#### S6.4 A screened Earnshaw identity and what the point model gives

`DERIVED` note N16 (my algebra; I found no source stating it: search 71). Take N lumps with pair energy u_ij = -C cos(delta_ij) g(d_ij), g(d) = exp(-kappa d)/d, which is the lowest-order law of S6.1 for equal omega. Since (laplacian - kappa^2) g = -4 pi delta^3, the energy U_i(x) of lump i as a function of its own position (the others fixed, phases fixed) obeys laplacian U_i = kappa^2 U_i away from the other lumps. At a force-free point the Hessian of U_i has trace kappa^2 U_i, so stability under displacement of lump i requires U_i >= 0, that is, net repulsive interaction energy. If U_i < 0 at a balance, at least one displacement direction lowers the energy. Scope: pair law only, equal kappa, fixed phases and amplitudes, no rotation, point lumps (d much larger than the lump size). Non-Yukawa terms (centrifugal terms, core overlap, saturation) are exactly what can evade it.

Numbers from the pair law with C = 1 (`DERIVED`, kappa = 1, forces from -grad of u, scan kappa R from 0.05 to 12; arithmetic check, not a simulation):

| Arrangement | Static pair-law balance | Collective radial mode | Single-lump U_i | Normal modes of the whole cluster |
|---|---|---|---|---|
| 5-lump ring, phase step 72 degrees | kappa R = 1.0695 | restoring | -0.0358 (negative) | 4 negative modes of 15 (6 zero modes are translations and rotations); single-lump curvature radial +0.169, tangential -0.328, axial +0.123 |
| 6-lump ring, step 60 degrees | kappa R = 0.2947 | restoring | -0.4102 | not computed |
| Rings with m = 1: N = 3, 4, 7, 8; with m = 2: N = 5, 8; in-phase N = 5 | none in the scanned range | | | |
| Octahedral shell of 6 in-phase lumps around an anti-phase centre | kappa R = 1.591 | not restoring (second derivative -0.51) | shell lump -0.0724 (negative) | 10 negative modes of 21 |

Claude Desktop's two numbers therefore reproduce: the 5-ring balance is kappa R = 1.07 and radially restoring, and the octahedral-shell balance is not restoring. Two further results. (i) For the Session 1 primary (omega = 0.9091, kappa = 0.4166, half-width 3.344 from the Session 1 report), the 5-ring balance has R = 2.57, nearest-neighbour spacing 3.02 (0.90 half-widths) and next-neighbour 4.88 (1.46); the octahedron has R = 3.82 and nearest-neighbour spacing 5.40 (1.6). Both are inside the plan's 2.5 half-width flag, where the point model is not valid. (ii) Restoring in the collective radial mode is not stability: the 5-ring has four negative normal modes in this static model. The identity forecast this: for the octahedron it holds at every kappa R (algebra: U_shell/C = g(sqrt2 R)(-1.172 a - 2)/(a + 1) - 0.5 g(2R)/(a + 1), with a = kappa R, negative for all a at the force balance).

Optical clusters are not contradicted by this. Their equilibria are not pure pair-law balances: Desyatnikov & Kivshar's N = 4, m = 1 ring has first-order neighbour force cos(90 degrees) = 0 and still an equilibrium at R0 = 3.8, which their paper attributes to attraction balanced by angular momentum; the first-order tail law supplies neither. A point model with the tail law will not find equilibria of that kind.

#### S6.5 Do optical rings sit at a stable balance or only a slowly expanding one?

- **Radially: a restoring minimum.** Desyatnikov & Kivshar `READ`: the N = 5, m = 1 equilibrium is a minimum of the effective potential; an excited ring oscillates in radius with period 22 about it; the equilibrium radius and angular velocity match the effective-potential estimates (0.157 measured, 0.154 estimated).
- **Azimuthally and in noise: metastable.** Clusters are "generally metastable" with a symmetry-breaking instability (DK 2002 `READ`); staircase-phase clusters break by azimuthal instability under noise (Mihalache 2004 `ABSTRACT`); the 3D two-colour ones split into two solitons after thousands of diffraction lengths (Crasovan 2003 `READ`).
- **Without angular momentum: slow expansion.** Necklace beams expand slowly (Soljacic et al. 1998 `READ`); Zezyulin 2026 (`READ`, introduction) says necklaces in general do not propagate stationarily, from fusion of the solitons to unbounded expansion of the ring.
- **Two bodies: unstable.** An orbiting bound state of two solitons is always unstable in the conservative 2D and 3D models (Malomed 1998 `READ`). Rings of four or more solitons are a many-body effect not covered by that argument.
- **Static chains: need something extra.** Flat-space Q-ball chains in the two-field model are unstable beyond a dipole (Jaramillo & Zhou `READ`).

### S7. Predictions on record against the sources

| Item | Prediction on record | What the sources say | Verdict |
|---|---|---|---|
| WP-A, path integral | Estimate from integrating dE/(E Gamma) along the family reproduces the beta = 0.85 death (1146) and the beta = 4/3 crossing (10,630) | This is published practice (Ibe eq. 55, Li eq. 2.30, `READ`, Session 1 A5). Benchmark of the same idea against explicit evolution: 5 to 7 per cent errors for monodromy potentials at lifetimes of 1e6 to 1e7 ([Olle 2021](https://arxiv.org/abs/2012.13409) `READ`, Table 1). Zhang 2020 (`READ`): the calculation overestimates inside dips | Agrees with practice. Both validation points sit on rate dips (omega = 0.909 and 0.779), the case Zhang flags, so expect the largest errors there |
| WP-A, CD guess | Lifetime rises for some way above 4/3, then falls as the stable branch narrows | Cliff picture (CGT `READ`): rise then abrupt fall when a dip passes omega_death. Our dips migrate up; crossing at beta of about 1.6 to 2.1 (N17, `DERIVED`). In omega the branch does narrow with beta; no source links that to lifetime | Shape has a published mechanism; the stated cause does not |
| WP-A, "beta list, fine omega scan" | Scan omega finely to find dips | Dips are narrow and set the lifetime (Ibe, Mukaida, Zhang, CGT, `READ`) | Agrees |
| WP-A, 3D confirmation of any beta that wins | One full 3D run | A&S: flat-tops more robust to long-wavelength collapse in 3D; a narrow-band short-wavelength instability near k = sqrt(3) m "needs to be resolved rigorously" (A&S, `READ`, conclusion); sine-Gordon states fragment in 3D when large or nodal (van Dissel, Session 1, `READ`) | Agrees; warranted |
| A1 | Path-integral reproduces the two validation lifetimes to within 30 per cent, using a Gamma(E) map that excludes the validation runs | As in the first row | Plausible, no source for 30 per cent on dips |
| A2 | Best-profile lifetime rises monotonically over beta = 0.75 to 5.0, no fall | Geometric decoupling and A&S's argument support a rise; the cliff gives a mechanism for a fall near 1.6 to 2.1; no lifetime-against-beta source for this model (searches 29, 32, 70) | Not supported or excluded. A2 is an extrapolation |
| A2-CD | Rise then fall | See above | See above |
| A3 | beta = 0.75: T_death 350 to 600; 1.0: 3,000 to 12,000; 1.15: T_90 3,000 to 15,000 | No source below beta = 4/3 | Rests on nothing found |
| A4 | beta >= 1.6: alive at T = 30,000 with frac above 0.90 | beta = 4/3 lives 1e4 to 1e5 with Mukaida's run to 1e5 without death (READ); crossing 0.90 at 10,630 in our radial run. Rests on A2; the cliff would cut it if it falls between 1.6 and 2 | Plausible; same caveat as A2 |
| A5 | No beta is permanent | Energetic death generic for power-series potentials (van Dissel 2023, `READ`); eternal oscillons only in non-polynomial potentials (Olle 2021 `READ`) | Agrees |
| B1 | 3D tracks radial to 1e-3 to T = 5000 | No source for non-radial stability of the sextic family; non-spherical perturbation: higher l more stable, low l case-dependent (CGT appendix C.3 `READ`); nodeless states do not fragment (van Dissel `READ`) | Consistent; no direct source |
| B2 | No growth of non-spherical structure | same | Consistent; no direct source |
| C0a | theta = 180 equals theta = 0 | Exact symmetry of V (phi to -phi) | Agrees |
| C0b | theta = 45, 90, 135 pass with differences of order 1e-3 or less | A phi_t = 0 single-frequency start is not the quasi-breather (van Dissel `READ`); pair literature uses phase-shifted copies (Amin & Mou, Xue et al., `READ`). `DERIVED` N18: a theta-shifted single-frequency start leaves the 3 omega component's phase unshifted (it should be 3 theta), so its mismatch is first order in the 3 omega amplitude ratio eps and its energy cost second order | Consistent; size depends on eps, not given in any source |
| WP-C | In phase attracts and merges, anti-phase repels (every primary source agrees) | 3D cubic-quintic NLS: merger at 0 and pi/4, rebound at pi (Burlak & Malomed `READ`); real oscillons: Amin & Mou, Xue et al. (`READ`); the plane stress at pi is non-negative at every instant (N15). Real-oscillon merger loses about 30% of the energy (Amin & Mou) | Agrees. 45 degrees is expected to merge, not only 0 |
| WP-C | At 90 degrees the leading force vanishes and N flows from one lump to the other (CD, no source) | First-order force is zero at pi/2 (Malomed 1998, via Burlak & Malomed). But the pair repels with a large asymmetric N split (Delta about 0.76; K = 0.062, d = 10, kappa d = 2.5) | Agrees on the N flow; the "leading force vanishes" part is true only at first order. Expect repulsion. Direction rule: N14 |
| WP-C | U = -2 pi A^2 cos(dphi) exp(-kappa d)/d, check it and state what is held fixed | Malomed 1998 Eq. 31 (identical form once the NLS tail coefficient squared is A^2/2, N13); fixed omega, equal solitons, lowest order, d many decay lengths | Agrees. Plan's form confirmed by a second independent derivation and a published formula |
| WP-C | Control: single lump at each phase matches phase 0 | See C0b | See C0b |
| WP-D | Treat lumps as points; flag balances closer than about 2.5 half-widths | Validity of the point law needs d many decay lengths (Malomed 1998); no source gives 2.5 half-widths | The flag is reasonable and unsourced. Both of CD's numbers fall below it (0.90 and 1.6 half-widths) |
| WP-D, CD number 1 | 5-lump ring, 72 degree steps, balances near kappa R = 1.07, radially restoring, inside overlap | Reproduced with the pair law (N16): kappa R = 1.0695, collective radial mode restoring, spacing 0.90 half-widths. Same pattern published in 2D saturable Kerr: N = 5, m = 1 equilibrium in an overlap region with a restoring radius (DK 2002 `READ`; nearest-neighbour spacing 3.53 against a = 1.623, which is 1.85 half-widths at half maximum of their Gaussian amplitude, 1.18 a each; `DERIVED` arithmetic, and not the project's half-width definition) | Agrees. Caveat: four negative normal modes in the static point model |
| WP-D, CD number 2 | Octahedral shell around an opposite-phase centre has a balance radius that is not restoring | Reproduced (N16): kappa R = 1.591, collective radial second derivative -0.51; shell-lump U_i < 0 at every kappa R, so unstable by the identity | Agrees |
| WP-D, patterns | 0/180 patterns up to symmetry plus stepped phases around an axis | Alternating 0/180 with a mirror plane: no radiation-free bound state (N15) and published necklace expansion (Soljacic `READ`). Stepped rings: literature candidates N >= 4m, step <= 90 degrees (DK `READ`) | 0/180 patterns with a phase-exchanging mirror plane should be controls, not candidates |
| WP-D, angular momentum | Note which patterns carry J | N12 (`DERIVED`): 0/180 patterns carry none; staircase rings only through overlap; ramps carry m_t N | Consistent |
| Plan, geometric binding | "No published work on it was found" | Still true for real fields (S4). Published arrangements exist for complex fields with extra ingredients (S1) | Agrees, with the S1 list as context |

`DERIVED` note N18 (for C0b; my algebra). The quasi-breather at phase theta is Phi_1(r) cos(omega t - theta) + Phi_3(r) cos(3(omega t - theta)) + higher odd harmonics. The plan's start (phi = Phi cos theta, phi_t = -omega Phi sin theta, with Phi the single-frequency profile) keeps only the first term. The omitted piece has the same size at every theta, but its phase relative to the fundamental is 3 theta: the omitted (field, velocity) pair is (Phi_3 cos 3 theta, 3 omega Phi_3 sin 3 theta). It is purely field at theta = 0 and 180 degrees, purely velocity at 90, and a mixture at 45 and 135. So the injected transient is not identical across theta even though the amplitude is; it is first order in eps = Phi_3/Phi_1 in the field and second order in the energy. No source gives eps for this model; a harmonic fit to the Session 1 radial run would. **Correction (Session 2, F-009): the energy of the slice is first order in eps, not second; the start-phase dependence is the 2, 4 and 6 times phase oscillation of the ansatz energy (N22). See Addendum C3.**

### S8. Implications for WP-C, WP-D, WP-E (and one for WP-A)

Each item names its source. Section references are to this brief.

1. **WP-C, what the formula is.** Use the Session 1 momentum-flux law as the primary check and Malomed 1998 Eq. 31 as the independent one; they agree to the prefactor 2 pi at omega to m (N13, S6.1). Measured values of A and kappa per profile are in `outputs/family/family_beta*.json`. State that U is a grand-potential term at fixed omega, equal to the fixed-N energy at first order.
2. **WP-C, a free diagnostic.** For the anti-phase pair the integral of T^zz over the mid-plane must be non-negative at every sampled time (N15; exact, because it uses only the plane's odd symmetry). A negative value would point to an engine or diagnostic error. The plane integral includes the radiation field, so compare it with the outgoing flux when judging a balance.
3. **WP-C, 90 and 135 degrees: expect repulsion, not zero force.** The 3D cubic-quintic result at pi/2 is repulsion with an N split of about 0.76 at kappa d = 2.5 (Burlak & Malomed `READ`). At the larger separations the plan intends (first-order regime, kappa d of 4 or more) the cosine force is small at 90 degrees, so measure it against the sin-driven N transfer (N14): dN_1/dt = -|U_0(d)| sin(theta2 - theta1), N flowing to the later-peaking lump. A ratio of the measured transfer rate to |U_0| near sin(delta) would test the law.
4. **WP-C, merger thresholds.** In the 3D cubic-quintic NLS pi/4 merges and pi/2 does not (`READ`); expect the 45 degree pair to merge, with the critical phase between 45 and 90 degrees. A real-oscillon merger loses about 30% of the energy (Amin & Mou `READ`).
5. **WP-C, the phase control.** The theta-shifted start has a 3 omega phase error of order eps (N18); take eps from the radial profile files before reading small differences as significant.
6. **WP-D, demote the phase-exchanging mirror patterns.** Alternating 0/180 rings, dipoles and chains have no radiation-free bound state (N15); keep one as a control that should expand (necklace beams do, Soljacic `READ`). A radiation-balanced state is not excluded by the argument, only unsupported by any source.
7. **WP-D, screen with the identity.** For every candidate compute U_i at each force balance (N16). Any U_i < 0 means the point model is unstable in at least one direction; rotation, overlap or a second ingredient has to be what rescues it. The 5-ring and octahedron both fail.
8. **WP-D, the point model will miss the published equilibria.** Desyatnikov & Kivshar's N = 4 ring (step 90 degrees, first-order neighbour force zero) is bound at R0 = 3.8; their N = 8, m = 2 ring (step 90 degrees) at R0 = 5.5. Those come from overlap and angular momentum, not the tail law (S6.4). The 3D tests (WP-E) have to cover stepped rings directly, not only the ones the point model balances.
9. **WP-D, every number inside the overlap flag.** The spacings of CD's two cases are 0.90 and 1.6 half-widths for the primary lump. No literature validates the pair law there; the 3D cubic-quintic pairs at kappa d = 2.5 merge or repel and do not follow the first-order law.
10. **WP-D, J bookkeeping.** Zero for 0/180; overlap-only for staircases; m_t N for ramps (N12). Compute J and the moment of inertia from the field and compare the angular velocity J/I with the observed rotation (Desyatnikov & Kivshar agree to 2%).
11. **WP-E, candidate list with literature support.** Stepped rings with m = 1 and N = 5 or 6 (step 72 or 60 degrees), and N = 8 with m = 2; N = 3 is excluded (no bound state in DK `READ`); N = 4 is marginal (first-order force zero; in the 2-colour 3D case the effective potential had no minimum for N = 4, yet a launched N = 4 cluster stayed robust, Crasovan `READ`). Phase step at most 90 degrees.
12. **WP-E, controls from the literature.** (a) Staircase phase against ramp phase on identical geometry (Crasovan, Mihalache: staircase splits in two, ramp fuses into a torus). (b) m = 0 (all in phase): expect fusion in about 100 to 150 propagation units in the optical analogue. (c) step above 90 degrees: expect expansion. (d) the same lumps far apart. (e) the 0/180 ring (expands).
13. **WP-E, what "bonded" should mean in time.** Published clusters last at most a few revolutions (S1.1, my arithmetic: DK N = 4, 0.4 revolution in 60 diffraction lengths; four-soliton vector cluster 5.6 revolutions in 225). Define the criterion against the rotation period, which J/I gives.
14. **WP-E, an open arm the literature suggests, not a recommendation.** Optical clusters are stabilised by an incoherent second component; for one real field the analogue would be a lump of different omega. No source tests it. Put it in the parking lot unless the first arms fail.
15. **WP-E, measurements.** Per-lump N, lump-to-lump distances, J and the radius oscillation period of an excited ring (DK: period 22 for N = 5), and the lump-by-lump phase.
16. **WP-A.** Record omega_dip(beta) and omega_death(beta) from the scan and look for the crossing near beta of 1.6 to 2.1 (N17). The published benchmark for the path-integral estimate is 5 to 7 per cent (Olle 2021, smooth cases without the dips that matter here); the plan's 30 per cent is looser, and a miss beyond it on a dip would be worth recording, not rescuing.

### S9. Search log (numbering continues from section 8 and Addendum A3; web search, 2026-10-01)

| # | Search | Result |
|---|---|---|
| 41 | Desyatnikov Kivshar rotating optical soliton clusters | nlin/0112039 (read); review nlin/0501026 (read, sec. 6.2, 6.3, 8.1) |
| 42 | necklace beams self-trapped ring angular momentum Soljacic Segev | Soljacic, Sears, Segev 1998 (read); PRE 62, 2810 and PRL 86, 420 (abstract or snippet only); Zezyulin 2026 (read); arXiv:0910.2926 on necklace arrays with modulated nonlinearity (lattice, skimmed only) |
| 43 | stationary two-soliton bound states, phase pi/2, oscillatory tails, fourth-order dispersion (search 16 lead) | Buryak & Akhmediev 1995 (abstract); Sakaguchi et al. 2018 (read); Cui et al. 2026 review (read); Malomed 1998 (read); Malomed 1991 (abstract) |
| 44 | Recipes for oscillon longevity | arXiv:2012.13409 (read) |
| 45 | spinning oscillon real scalar field angular momentum | Zhang, Jain, Amin 2022 (read; vector field) |
| 46 | spinning Q-balls stability angular momentum azimuthal instability Volkov Wohnert | arXiv:2509.18656 (read; no stability test; the fetch summary was wrong); Volkov & Wohnert (abstract) |
| 47 | Excited oscillons and charge-swapping | arXiv:2504.17382 (abstract; the PDF is over the fetch limit) |
| 48 | Collision dynamics of false-vacuum oscillons | arXiv:2605.12633 (re-checked; v1 only) |
| 49 | multi-Q-ball bound states, molecules, clusters, dimers | Copeland et al. 2014 (read); 2509.03192 (earlier) |
| 50 | oscillons with angular momentum, non-spherical, 3+1D real scalar | vector oscillons only |
| 51 | rotating oscillon, azimuthal quantum number, 2+1D | nothing relevant |
| 52 | spinning axion stars, angular momentum, real scalar, non-relativistic EFT | Hertzberg & Schiappacasse (read); Delgado et al. (read, introduction); Almumin et al. (read); Dmitriev et al., Sanchis-Gual et al., Galushkina et al. (abstracts) |
| 53 | Mihalache et al. soliton clusters 3D cubic-quintic (two queries) | OSTI abstracts for PRE 68, 046612 and J. Opt. B 6, S333; Crasovan et al. (read); Burlak & Malomed (read) |
| 54 | Malomed, potential of interaction between two- and three-dimensional solitons | patt-sol/9809009 (read) |
| 55 | Copeland, Saffin, Zhou charge-swapping Q-balls (id) | arXiv:1409.3232 (read) |
| 56 | Hertzberg, Li, Schiappacasse axion clump merger | arXiv:2005.02405 (read, sec. 3.5) |
| 57 | Al Khawaja & Stoof soliton molecules | arXiv:1012.3762 (read, abstract and introduction) |
| 58 | bright matter-wave soliton train alternating phase | Strecker et al. 2002 (read); Khaykovich et al. and Al Khawaja et al. (snippets only) |
| 59 | force between two oscillons in 3D, relative phase, Yukawa tail | no 3D oscillon formula; a snippet of a 1+1D unequal-oscillon force in Campos et al., not opened further |
| 60 | interaction potential between Q-balls, tails, relative phase, 3D | Battye & Sutcliffe, Bowcock et al. (earlier brief) |
| 61 | radiation-mediated interaction between oscillons, third-harmonic outgoing waves | nothing |
| 62 | oscillon-oscillon collision, bound state, phi^4, relative phase | kink-antikink and merger results; Amin & Mou, Xue et al. (earlier) |
| 63 | axion star molecule, soliton binary, relative phase, fuzzy dark matter | nothing on bound arrangements |
| 64 | ring or necklace of oscillons with phase twist and angular momentum | optical necklaces; nothing for oscillons |
| 65 | boson stars, multi-centre, ring, chains | Cunha et al., Herdeiro et al., Gervalle, Jaramillo & Zhou, Shnir (read or abstract as tagged in S1.4) |
| 66 | oscillon cluster, tetrahedral, octahedral, vertices, phase pattern | nothing (granular and unrelated hits) |
| 67 | 2D soliton necklaces, nonlocal | snippets only (S1.1) |
| 68 | Unified theory of oscillons and modes | abstract (wobblerons) |
| 69 | bound states and clusters of oscillons, 2025 and 2026 | the same papers as 47, 48, 68; Floquet-mode papers (not opened) |
| 70 | oscillon lifetime against sextic coupling, 3D lattice | Drogosz et al. (abstract and earlier read); Antusch et al. (abstract); nothing for beta between 4/3 and 17 |
| 71 | Earnshaw theorem for Yukawa screened potentials | no source for the screened statement; only the textbook fact that the Yukawa function solves the modified Helmholtz equation |
| 72 | Afanasjev, Chu, Malomed bound states | PRE 56, 6020 (1997) found by title only; PRE 57, 1088 not found |
| 73 | Stratmann, Pagel, Mitschke temporal soliton molecules | snippets only |
| 74 | Mihalache et al., stable spinning optical solitons in three dimensions | arXiv:nlin/0201029 (abstract) |
| 75 | non-spherical oscillon l = 1, dipole oscillon, excited, real scalar 3+1D | nothing |
| 76 | oscillon vortex, spinning, rotating, NR limit, cubic-quintic, relativistic corrections | nothing for real scalars |
| 77 | real Klein-Gordon time-periodic multi-lump, phase-locked, polygon | nothing |
| 78 | oscillon chain, ring, lattice, repulsive out-of-phase, attractive in-phase | the Wikipedia statement already in the Session 1 brief; nothing new |
| 79 | breather clusters, bound states of breathers, sine-Gordon 2D, 3D | Nishida et al. 2009 (abstract); Pelinovsky & Sakovich 2012 on discrete Klein-Gordon lattices (abstract; lattice-pinned, not read further) |
| 80 | 3D vortices and rotating solitons in ultralight dark matter | Brax & Valageas (abstract; non-relativistic, repulsive quartic, complex) |
| 81 | charge-swapping Q-balls and their lifetimes | Xie, Saffin, Zhou (abstract) |
| 82 | flat-space Q-ball chains, Friedberg-Lee-Sirlin; Q-ball dipole equilibria | Jaramillo & Zhou (read) |

No claim of novelty is made. If one is made later about geometric binding of real-field oscillons, the supporting searches are 17, 64, 66, 69, 77, 78 and the arrangement searches 49, 65, 82 (which found the complex-field cases only).

### S10. Not obtained

- Full text: Buryak & Akhmediev 1995, Malomed 1991, Akhmediev, Ankiewicz, Soto-Crespo 1997 (paywalled or not on arXiv); Soljacic & Segev PRE 62, 2810 and PRL 86, 420; Mihalache et al. 2003 and 2004 (OSTI abstracts only, so cluster durations in 3D cubic-quintic are unknown to me); Volkov & Wohnert full text; Gordon 1983, Karpman & Solov'ev, Anderson & Lisak (credited secondhand in Al Khawaja & Stoof); Stratmann et al. 2005; Khaykovich et al. 2002; the EPJC 2026 paper on chains of rotating boson stars with quartic or sextic self-interaction; the full text of arXiv:2504.17382 (over the fetch size limit).
- Not found: any lifetime for phi^4-phi^6 oscillons between beta = 1.67 and 17, or below 4/3; any source for the force, N transfer or relative-phase drift between real oscillons in 3D; any real-scalar spinning oscillon; any statement of the screened Earnshaw identity; Afanasjev, Chu, Malomed PRE 57, 1088.
- Not checked beyond abstracts: the claim that the Dmitriev et al., Sanchis-Gual et al. and Galushkina et al. results carry over to our regime (all are gravity-bound or 2+1D complex).
- The point-pair results in S6.4 use C = 1 and kappa = 1 and scan kappa R from 0.05 to 12. They are not a replacement for WP-D.
- The fetch tool's summaries were unreliable once; every claim marked `READ` was checked against locally extracted text. The HTML-only read (arXiv:2504.17382 body) is tagged accordingly. A Nature page and a Springer page redirected to login endpoints; I did not follow them.

---

## Addendum B (2026-10-01, Session 2): lifetime estimates built from a measured Gamma(E) map (for F-007)

Scope: the literature check for the A1 miss recorded in `FAILURES.md` F-007. Model claude-sonnet-5-5; a "reasoning_effort" value of 30 is visible in my context. Sources were re-read for this addendum from PDFs saved by the fetch tool and converted locally (PDFKit); every `READ` below was checked against that text, not against a fetch summary. Figures were not examined; where a statement rests on a figure I say so. Search numbers continue from S9 (83 to 87). Derived notes continue as N19 to N21.

### B1. What the sources say (only the points that bear on F-007)

| Source | System | What it says | Tag |
|---|---|---|---|
| [Ibe et al. 2019](https://arxiv.org/abs/1901.06130), eq. 54 and 55, sec. IV, Fig. 5 | real phi^4-phi^6, beta = 4/3, radial | Gamma(Q) is computed analytically for each family member from the time-averaged Poynting flux of the radiation field (eq. 54), then integrated for the lifetime (eq. 55). Simulations from omega_ini = 0.850, 0.858 and 0.908 are compared with the analytic curve: good agreement for Q0 up to about 10^2.5, deviation above (the mu much less than m approximation fails). The simulation averaging window is Tave = 100, stated to be much longer than 2 pi/omega (about 10) and much shorter than 1/Gamma (about 1e4); simulations stop at t = 1e5, so rates below about 1e-7 cannot be shown. The lifetime for Q0 of about 10^2.1 is said to be about 1e4, "expected from" Gamma; no lifetime error figure is stated | READ (text; figure not examined) |
| [Li, Yamaguchi, Zhang 2025](https://arxiv.org/abs/2507.13276), sec. 2.3, eq. 2.28 to 2.30, Fig. 3 | same model, beta = 4/3, radial | Gamma_xi is semi-analytic on family members (eq. 2.28, 2.29), lifetime from eq. 2.30. Simulations are run from three omega_ini (0.85, 0.90, 0.91) instead of one very long run; the later part of one curve continues into the start of the next. The path has stepped platforms ("discrete energy levels") where the lump stays a long time, and it falls quickly between them; the platforms are the dips of Gamma. Averaging window Tave = 100. No numerical error for eq. 2.30 is stated in sec. 2.3 | READ (text; Fig. 3 not examined) |
| [Olle, Pujolas, Rompineve 2021](https://arxiv.org/abs/2012.13409), sec. 3.1, 3.2, App. B, C, Table 1, Fig. 8 to 10 | monodromy family, radial | "Relax and fast-forward": sample cosh initial data in (A, sigma), evolve each for t_relax = 1e4 per m, read Gamma from the outgoing flux at a fixed radius averaged over about 1e2/omega (App. B), and join the points into Gamma(omega). Whether a start relaxes to an oscillon at all depends strongly on (A, sigma). The relaxed states cluster in parameter space (Fig. 8). The method needs t_relax much smaller than the lifetime, and the check that relaxation is complete must be made per initial condition (the configuration "barely moves" after a fast phase, App. C); where the lifetime is too long to see a trajectory, it is reconstructed from several starts. Benchmark against explicit evolution, Table 1: p = -5, -4, -2.5, -2, lifetimes 9e5 to 1.2e7, errors 6.11, 7.33, 5.20, 5.83 per cent. For large negative p the dip in Gamma(omega) is shallow (sec. 3.2) | READ |
| [Zhang et al. 2020](https://arxiv.org/abs/2004.01202), introduction (summary of results; Fig. 2 discussion), sec. 7, 7.5, 8 | real scalar, radial | Dips, when present, account for most of the lifetime. The analytic Gamma over-estimates the measured rate at large amplitude, and the over-estimate in the dips prevents accurate lifetime estimates where dips exist (sec. 8); no factor is given. For the phi^4-phi^6 potential the analytic dips shift against the numerics (sec. 7.5) | READ |
| [Zhang 2020b](https://arxiv.org/abs/2011.11720), sec. 2.1 | real scalar, radial | Lifetime is the area under dt/dE = 1/(Gamma E) against E. An oscillon that starts below omega_dip rises in omega by radiating and spends most of its lifetime in the dip | READ |
| [Cyncynates & Giurgica-Tiron 2021](https://arxiv.org/abs/2104.02069), sec. IV B, IV C, Fig. 14 | periodic potentials, 3D | phi^4: explicit lifetime about 6000 per m against the physical-quasibreather prediction 5900 per m. QCD-axion oscillon (lifetime about 400 per m): radiated power from the quasibreather and from simulation disagree within a factor of a few, because the orthogonal deformation is non-perturbative; the dip near omega = 0.6 m, where the third harmonic is confined, carries most of the lifetime | READ |
| [Mukaida et al. 2017](https://arxiv.org/abs/1612.07750), sec. 2 (toy model, Fig. 2, 3), sec. 3 (Fig. 4) | real phi^4-phi^6, beta = 4/3 | Which attractor point the lump reaches depends on the starting state: from R_ini = 10 it reaches an attractor and is too long-lived to follow to the end; from R_ini = 7 it starts below that attractor and reaches another; from R_ini = 5 it starts below that one, reaches Q_cr and decays suddenly | READ |
| [van Dissel et al. 2023](https://arxiv.org/abs/2303.16072), sec. III (introduction, III A, III B) | sine-Gordon, radial | Single-frequency starts at different omega, relaxed for a short fixed time and then measured, generically latch onto an oscillon with higher harmonics, but not all such starts relax to what they call a state. A nodeless start probes all the oscillon configurations up to energetic death; footnote 6 says this is strictly true only for single-frequency starts, and more general initial data can excite instabilities and leave the adiabatic curve. For states with nodes more than one initialization is needed | READ |

### B2. The two questions put for F-007

1. Have published lifetime estimates from a path integral or a decay rate been seen to fail by a factor of 3 to 10 at rate dips? No source found. The published agreements are 2 per cent (phi^4, explicit 6000 against 5900, Cyncynates & Giurgica-Tiron) and 5 to 7 per cent (Olle 2021 Table 1, shallow-dip cases), and Ibe and Li state no lifetime error. Zhang 2020 says the estimate cannot be accurate where dips exist and gives no number; the only quantified dip-region disagreement I found is the factor of a few in radiated power for the QCD-axion oscillon (CGT Fig. 14 caption). All of these compare an analytic Gamma with simulation. Their error is a Gamma that is too high (Zhang), so a lifetime that is too short. The WP-A error is a map Gamma that is too low across a gap, so a lifetime that is too long. The mechanism and the sign differ.
2. Have published works noted that some states on the adiabatic path cannot be reached from single-frequency initial data? No source found for that statement. The nearest points: not every single-frequency start relaxes to an oscillon (van Dissel, III); relaxed states cluster in parameter space and the outcome depends strongly on the start (Olle, 3.1 and Fig. 8); different starts land on different attractor points and starts below an attractor skip it (Mukaida). The one-way nature of the path (omega rises as the lump radiates, so a start can only sample states at or above its settled omega) follows from the picture in Zhang 2020b and Li 2025 (`DERIVED`, N20 below).

### B3. Correction to S7 row A1 and S8 item 16 of my Session 2 brief

S7 row A1 said the path-integral plan "agrees with practice", with the largest errors expected on the dips because of Zhang 2020. S8 item 16 said a miss beyond 30 per cent on a dip would be worth recording, not rescuing. Two things were wrong or missing:

- The integral is published practice (Ibe eq. 55, Li eq. 2.30, Zhang 2020b sec. 2.1). The construction of Gamma(E) is not: in Ibe, Li and Zhang it is semi-analytic on a dense family, which has no coverage gaps. The only published map built from short runs, Olle 2021, relaxes each start for 1e4 per m, samples a two-parameter family and checks relaxation per start. I used Olle only as an accuracy benchmark (5 to 7 per cent) and did not point out that its relaxation condition and clustering (N19, Fig. 8) apply to the WP-A construction.
- I cited Zhang's over-estimate as the reason to expect trouble on dips. That caveat concerns an analytic Gamma, and it has the opposite sign to the miss that happened. What the brief got right was the location (dips) and the warning that a miss beyond 30 per cent should be recorded; it did not predict the cause or the size.

### B4. Derived notes

`DERIVED` note N19 (relaxation time against lifetime). Olle 2021 relaxes each start for t_relax = 1e4 per m (App. C) and states the method needs t_relax much smaller than the lifetime (sec. 3.1). For its Table 1 cases the ratio t_relax/tau (explicit lifetimes) is 0.011, 0.0067, 0.0020, 0.00083 (p = -5, -4, -2.5, -2). The WP-A transient exclusion is t_s = 150 + 2 R_lump, 183.6 at beta = 0.85 and 209.0 at beta = 4/3, against the Session 1 measured 1145.5 and 10,630.5: ratios 0.160 and 0.0197. That is 14 to 190 times Olle's ratios at beta = 0.85 and 1.8 to 24 times at beta = 4/3. Context only: t_s is a window-exclusion rule on single-frequency starts, not a relaxation time on cosh starts, and no source gives a threshold for the ratio.

`DERIVED` note N20 (where the lifetime sits at beta = 4/3, from the F-007 numbers). The gap segment 2838 to 2718 is 4.3 per cent in ln E. It holds 29,646 of the 32,797 predicted (90 per cent) and about 7100 of the 10,630 measured (67 per cent). The whole T_90 path 2838 to 2566 is 10.1 per cent in ln E, so the first 43 per cent of the path in ln E carries two thirds of the time. Along a path on which omega rises and E falls, a start samples only energies at or below its settled energy, so a segment on the low-energy wall of a dip is sampled only by runs that sit in or just above that dip and then leave it, which is slow for exactly the deepest dips (a map run of 2000 time units leaves a dip with Gamma of 1.8e-5 after losing under 4 per cent of its energy, DERIVED: 2000 x 1.8e-5).

`DERIVED` note N21 (window length against decay time). Ibe's averaging window is much shorter than 1/Gamma. The WP-A window W = 300 equals 0.30 of 1/Gamma at Gamma = 1e-3 and 0.99 of it at Gamma = 3.3e-3. The validation run's own windowed Gamma at beta = 0.85 is 9.7e-4 in the window that starts at t = 783.6 (`validation.json`, D1), and F-004 D1 gives 2.9e-3 for t from 1000 to 2000. So the window condition the published papers state is broken in the fast part of the beta = 0.85 path.

### B5. Search log (numbering continues from S9)

| # | Search | Result |
|---|---|---|
| 83 | oscillon lifetime estimate integrating decay rate along adiabatic evolution, compared with simulation, dips, error | Li 2025, Zhang 2020, Mukaida 2017, Olle 2021, Ibe 2019, CGT 2021, Zhang review arXiv:1911.03340 (all but the review already read; none gives a lifetime error on a dip; the review was not opened beyond extraction) |
| 84 | single-frequency initial data cannot populate all points of the attractor path; relaxation; sampling initial data | van Dissel 2023, Olle 2021, CGT 2021 (read); arXiv:1210.2227, 2010.07789 (titles only, not opened) |
| 85 | oscillon stepped platforms, energy levels, decay-rate dips, sextic, numerical measurement from relaxed data | Li 2025 sec. 2.3 (read); no new source |
| 86 | "fast-forward" oscillon lifetime, Olle method applied 2024 to 2025 | Olle 2021 (read); Olle's UAB thesis (2022, not opened); arXiv:2505.20233 (AdS oscillon lifetime resonances, abstract snippet only); arXiv:2512.17193 (snippet only); no paper found that applies the method to the sextic family |
| 87 | dips narrow in frequency, trajectories escape the dip, lifetime sensitivity to initial frequency | Zhang 2020b sec. 2.1 (read); arXiv:2511.03501 and 2101.06988 (titles only, not opened) |

No claim of novelty is made. For F-007: no source found for a lifetime estimate from a simulation-built Gamma(E) map failing at a dip, and no source found for a coverage criterion (a node-gap threshold) for such a map.

### B6. Not read, open

- Olle's thesis (UAB, 2022) may discuss limits of relax and fast-forward; found, not opened.
- Figures in Ibe 2019 (Fig. 4, 5) and Li 2025 (Fig. 3) were not examined; I did not compare dip depths or widths with the project's maps. Ibe's analytic dips at beta = 4/3 were read by eye in Session 1 at about 1e-6 to 1e-7 (section 2.6). The WP-A start at omega = 0.9065 has a measured Gamma at or below the 2e-8 floor; whether that is the same dip, and whether it is deeper than the published analytic dip, is not checked, and nothing here should be read as a novelty claim.

---

## Addendum C (2026-10-01, Session 2): phase of the start, and phase dynamics of a pair (for F-009)

Scope: the literature check for the C0b miss in `FAILURES.md` F-009. Model claude-sonnet-5-5; a "reasoning_effort" value of 30 is visible in my context (the mapping to a named level is not). Sources were re-read for this addendum from PDFs saved by the fetch tool and converted locally (PDFKit); every `READ` was checked against that text, and the equations of Battye & Sutcliffe (3.7, 3.8) were checked on the rendered page. Search numbers continue from B5 (88 to 98). Derived notes continue as N22 to N24. No git commands, no simulations; the arithmetic in N22 and N23 uses two files already in the repo.

### C1. What the sources say

| Source | System | What it says (only the points that bear on F-009) | Tag |
|---|---|---|---|
| [Battye & Sutcliffe 2000](https://arxiv.org/abs/hep-th/0003252), sec. 3, eqs. 3.2, 3.3, 3.6 to 3.8 | complex Q-balls, 1D, 2D, 3D | Relative phase alpha is put on one Q-ball by e^(i alpha), exact by the U(1) symmetry; the ansatz's total charge depends on alpha through the overlap, so the energy functional cannot be used to compare phases. Mechanical model (fixed positions, fixed profile): L = (M/2)(theta1_dot^2 + theta2_dot^2) - epsilon^2 cos(theta1 - theta2) + const (3.7); theta1_ddot + theta2_ddot = 0, theta1_ddot - theta2_ddot = (2 epsilon^2/M) sin(theta1 - theta2) (3.8). For every alpha in (0, pi) the relative phase accelerates positively (from 0 toward pi), the first (higher-frequency, phase-ahead) Q-ball loses charge to the second, the initial rate is greatest at alpha = pi/2; in-phase and anti-phase are symmetric solutions. Limits: profile held fixed, positions held fixed; including the profile's frequency dependence would give a fourth-order equation. Cites an effective Hamiltonian for Q-ball dynamics "in preparation" (ref. 11) and Ahn, MacKay, Sepulchre (ref. 18) for relative-phase dynamics of breathers | READ |
| [Hertzberg, Li, Schiappacasse 2020](https://arxiv.org/abs/2005.02405), sec. 3.5, eq. 3.25 | non-relativistic complex wave function, 3D | relative phase delta enters as e^(-i delta) on one clump; delta = pi gives destructive interference at the plane and a void (effective repulsion); any delta other than pi merges | READ |
| [Amin & Mou 2020](https://arxiv.org/abs/2009.11337), sec. 4, App. A.3 (A.8, A.10 to A.13) | real oscillons, tanh potential, 3+1D | single oscillon: phi = M f(r) cos(omega t) with f from the period-averaged action (A.10, A.11); pair: two boosted copies of equal phase and frequency (A.13); "we limited ourselves to two oscillons of same phases"; other phases cited to Hertzberg, Schiappacasse, Yanagida ([arXiv:2001.07476](https://arxiv.org/abs/2001.07476), not opened) | READ |
| [Xue et al. 2025](https://arxiv.org/abs/2510.01597), secs. III and V | real oscillons extracted from lattice runs, 3+1D | oscillons carry full harmonic content; the pi case is the field and its time derivative times -1; for mismatched oscillons the relative phase "cannot easily be set in advance"; a Galilean boost of the extracted data gives a velocity that depends on the oscillon's phase | READ |
| [Campos, Manton, Mohammadi 2026](https://arxiv.org/abs/2605.12633), eq. V.1 and sec. V B (v1 only) | real, 1+1D, Fodor expansion truncated at third order | relative phase delta enters as a shift of the time argument of the whole oscillon (harmonics included); the Fodor expansion is not an exact solution; the deviation for two-oscillon evolution depends on delta, largest at delta = pi, with more radiation near pi | READ |
| [Gálvez Ghersi & Braden 2024](https://arxiv.org/abs/2303.04750), eq. 3, Fig. 4, Fig. 16 caption, sec. VI | exact sine-Gordon breather at initial phase theta_0, used as a start in a D-dimensional deformed equation and in a deformed potential | for D = 1 the initial energy is independent of theta_0 and the oscillon properties are nearly independent of theta_0; with a dimensional deformation the settled frequency has cusps centred on theta_0 = pi/2 (softer for slower transitions); fitted relations hold for theta_0 other than pi/2 | READ |
| [van Dissel et al. 2023](https://arxiv.org/abs/2303.16072), sec. III | sine-Gordon, radial | start phi(r, 0) = Phi(r), phi_t = 0 (one phase); higher harmonics non-negligible; energy of the ansatz against the quasi-breather compared (Fig. 8, difference small, the text does not say period average or slice); quasi-breather starts gave slightly altered lifetimes | READ |
| [Malomed 1998](https://arxiv.org/abs/patt-sol/9809009), concluding remarks | 2D and 3D solitons, conservative and dissipative | effective mass of the phase degree of freedom is negative (attributed to Afanasjev, Chu, Malomed, PRE 57, 1088, not obtained); a stationary orbiting state with d^2E/dpsi^2 > 0 is therefore unstable in psi; no pendulum equation, no N exchange | READ |
| [Raghavan et al. 1999](https://arxiv.org/abs/cond-mat/9706220), eqs. 2.6 to 2.8, 4.1, 4.7, 4.10 to 4.12 | two-mode bosonic Josephson junction | z_dot = -sqrt(1 - z^2) sin(phi); H = Lambda z^2/2 - sqrt(1 - z^2) cos(phi); "non-rigid pendulum"; phi_ddot = -[Lambda sin(phi) + sin(2 phi)/2] + O(z^2) (4.11); small-oscillation inverse periods about 0 and pi in the ratio sqrt((1 - Lambda)/(1 + Lambda)) for Lambda < 1 (4.7, 4.10); Lambda > 0 (repulsive) treated | READ |
| [Parker et al. 2008](https://arxiv.org/abs/0712.3002) | 3D Gross-Pitaevskii, attractive, bright solitary waves in a waveguide, collisions | population transferred varies as sin(delta phi) at the highest speed, peak at pi/2; skewed at lower speed; a two-mode Josephson-like model agrees qualitatively and overestimates the amplitude | READ |
| [Ahn, MacKay, Sepulchre 2001](https://link.springer.com/chapter/10.1007/978-94-017-2452-4_9) | weakly coupled oscillators, breathers | effective Hamiltonian on a manifold of periodic orbits, independent of the overall phase; critical points at fixed action are exact periodic solutions | ABSTRACT (search summary; the Springer page redirects to a login) |
| [Axenides et al. 2000](https://arxiv.org/abs/hep-ph/9910388), sec. III | Q-balls, 1D | two Q-balls with internal frequencies omega_+ and omega_- (the case studied has omega_- = -omega_+): interaction energy proportional to cos[(omega_+ - omega_-) t], breathing at the frequency difference | READ |
| [Copeland, Saffin, Zhou 2014](https://arxiv.org/abs/1409.3232) | Q-ball and anti-Q-ball, overlapping cores | charge swapping from a slow drift of the relative phase caused by a small frequency difference; existence independent of the initial relative phase | READ |
| [Al Khawaja 2010](https://arxiv.org/abs/1004.3000), introduction and secs. II to III | 1D exact two-soliton solution | the soliton-soliton force, exponential in the separation and depending on phase (in phase attracts), is credited to Karpman & Solov'ev, Gordon, Anderson & Lisak; the relative phase of two solitons of different amplitude drifts in time and sets the force | READ (partial); the credits are SECONDHAND |

### C2. The two questions put for F-009

1. **Is the phase dependence of the single-frequency slice energy documented, and how do pair studies build the phase?** The first: no source found. The second: complex fields use an exact U(1) rotation; real-field studies that vary the phase use a time shift of full-harmonic data (Campos et al., Gálvez Ghersi & Braden), a sign flip (Xue et al.), or have only the in-phase case (Amin & Mou). No source found for a real-field pair built by rotating a single-frequency ansatz at an intermediate phase. Related and READ: Campos et al. report a phase-dependent deviation from the field equation for truncated-expansion pairs; Gálvez Ghersi & Braden find phase structure (cusps at pi/2) for a start that is an exact solution only in D = 1; Battye & Sutcliffe note the charge of an ansatz pair depends on the phase.
2. **Is the pendulum, or the negative mass of the phase, already published, with the same stable point?** For Q-balls (Battye & Sutcliffe, fixed-profile model), solitons (Malomed, negative mass) and condensates (Raghavan, non-rigid pendulum), yes in structure; for Q-balls the relative phase is driven from 0 toward pi, and the direction of the charge flow (phase-leading to lagging) is the project's. For the stable point at pi: implied for Q-balls, implied by the negative mass for solitons, derived by a sign flip for the condensate model (N24). For real scalar oscillons in any dimension: no source found.

### C3. Correction to N18, S7 row C0b and S8 item 5

N18 (S7) said that the theta-shifted single-frequency start leaves the 3 omega component's phase unshifted (it should be 3 theta), so that the mismatch is "first order in eps in the field and second order in the energy". The first part stands. The energy claim is wrong, and the verdict "Consistent" in S7 row C0b understated the effect. Reasons:

- The energy of the slice of the ansatz Phi cos(theta), -omega Phi sin(theta) is a function of the fundamental alone (N22). It oscillates with theta at 2, 4 and 6 times the phase, with amplitudes set by the 3 omega and 5 omega residual of the single-frequency equation, which is first order, and it equals the period-averaged energy only on average over theta.
- The quasi-breather is a stationary point of the period-averaged action (equivalently of E - omega N at fixed omega), not of E on phase space. The cross term between the fundamental and the omitted harmonics in E is therefore first order in eps. Only the period-averaged energy of the ansatz differs from the quasi-breather's at second order (consistent with the small difference in van Dissel et al., Fig. 8; DERIVED, re-derive before relying on it).
- Size for the project's primary (beta = 4/3, omega = 0.909149, Set-B-like units): E_bar = 2801.68; slice energy +1.5% at theta = 0, +2.8% at 45, -2.1% at 67.5, -7.0% at 90; peak to peak 9.8%. The 3 omega source at the core is 11% of the linear term. S8 item 5 asked for eps from the radial profile files; N22 gives the quantity that matters.

### C4. Derived notes

`DERIVED` note N22 (slice energy of the single-frequency ansatz; arithmetic on `outputs/step2/profiles/profile_beta1.3333_w0.9091.json`, re-derive). Let phi = Phi(r) cos s with Phi from the profile equation. Then dE/dt = integral of phi_t R, where R = phi_tt - lap phi + V'(phi) is the residual. The fundamental part of R vanishes by the profile equation, leaving R = R3 cos 3s + R5 cos 5s with R3 = -g4 Phi^3/24 + g6 Phi^5/384 and R5 = g6 Phi^5/1920 (checked numerically against the projection of V_NL' on cos 3s and cos 5s at three radii). With I_j = integral of Phi R_j d^3x and s = omega t, integrating over time gives E(s) = E_bar + a2 cos 2s + a4 cos 4s + a6 cos 6s with a2 = -I3/4, a4 = (I3 - I5)/8, a6 = I5/12. For the primary: I3 = -407.20, I5 = +210.13, so a2 = +101.80, a4 = -77.17, a6 = +17.51; E_bar = 2801.683 (the profile's E, 2801.684). The start at phase theta is the ansatz at s = theta (E is even in s, so the sign does not matter). E(0) = 2843.83, E(22.5) = 2861.28, E(45) = 2878.85, E(67.5) = 2742.08, E(90) = 2605.21 = omega N. Direct evaluation from the profile reproduces the numerics worker's D3 table to all printed digits. a4 and a6 agree with the Fourier decomposition of the direct evaluation to 1e-14, a2 to 1e-5.

`DERIVED` note N23 (slice energy against settled omega; `PROJECT DATA`: `outputs/family/family_beta1.3333.json` rows at beta = 4/3 and `outputs/session2/wpC/phase_control_radial.json`). Fit E(omega) of the single-frequency family over omega 0.88 to 0.95 with a quartic (dE/domega = -40,741 at omega = 0.9091; fit E(0.909149) = 2801.27 against the profile's 2801.68). The omega with E(omega) = E0(theta) - (E_fit(omega_0) - E_bar) is 0.90811, 0.90769, 0.90728, 0.91062, 0.91428 for theta = 0, 22.5, 45, 67.5, 90. Measured settled omega at T = 300: 0.90952, 0.90923, 0.90875, 0.91215, 0.91611. Differences +1.41e-3, +1.54e-3, +1.47e-3, +1.53e-3, +1.83e-3 (the negative-theta runs differ from these by at most 1.2e-4). The measured spread over phases is 7.4e-3 and the spread of the differences 0.4e-3. A constant offset of about 1.5e-3 is about 61 in energy (2.2% of E): shed energy, the difference between the single-frequency family's E(omega) and the true quasi-breather's, or shape-mode energy; not separable here. Limits: five points, one beta, no shedding term, a fit of the family rather than the true quasi-breather family.

`DERIVED` note N24 (the pendulum in the bosonic Josephson junction variables). Raghavan et al. eq. 4.11, phi_ddot = -[Lambda sin(phi) + sin(2 phi)/2] + O(z^2), holds as written for either sign of Lambda. Linearising: about phi = 0, phi_ddot = -(1 + Lambda) phi, unstable for Lambda < -1; about phi = pi, with delta = phi - pi, delta_ddot = (Lambda - 1) delta, stable for Lambda < 1, which includes every negative Lambda. For an attractive nonlinearity (frequency falling as N rises, d omega/dN < 0, which is the stable branch here) Lambda is negative, so the zero mode is a saddle and pi is stable. In the project's variables, with the coupling taken independent of N_i (the sin 2phi term dropped), the equation is delta'' = K sin(delta) with K = 2 C_N |d omega/dN|, unstable at 0 and stable at pi. The sign structure is the same as Battye & Sutcliffe's eq. 3.8. I have not matched the constants (Raghavan's Lambda is U N_T/2K in a double-well; theirs and the project's N_i are normalised differently).

### C5. Search log (numbering continues from B5)

| # | Search | Result |
|---|---|---|
| 88 | Ahn MacKay Sepulchre "Dynamics of relative phases in generalised multibreathers" | Springer chapter and Nonlinear Dynamics 25, 157 (2001); abstract only (search summary); page redirects to a login |
| 89 | relative phase dynamics two interacting solitons, pendulum, amplitude exchange, negative effective mass of the phase | arXiv:1901.11243 (positive and negative mass solitons, title only), Al Khawaja 2010 (read); no pendulum for the relative phase |
| 90 | Josephson-like oscillation, two bright solitons, relative phase, population exchange, attractive | Parker et al. 2008 (read); others are spin-orbit-coupled or dark-bright BEC papers (titles only) |
| 91 | Battye Mackay Sutcliffe effective Hamiltonian Q-ball dynamics, relative phase, charge transfer | Battye & Sutcliffe 2000 (read); the "effective Hamiltonian for Q-ball dynamics" paper: no source found |
| 92 | Raghavan Smerzi Fantoni Shenoy coherent oscillations, pi oscillations, self-trapping | arXiv:cond-mat/9706220 (read) |
| 93 | Stability and dynamics of two-soliton molecules, relative phase | Al Khawaja 2010, arXiv:1004.3000 (read, introduction and secs. II to III) |
| 94 | Karpman Solov'ev Gordon soliton interaction, relative phase and amplitude difference equations | titles and abstracts only (Karpman & Solov'ev, Physica D 3, 487; Gordon 1983); equations not obtained |
| 95 | Afanasjev Chu Malomed bound states, effective mass of the phase negative | PRE 56, 6020 (title only); PRE 57, 1088 not obtained |
| 96 | oscillon pair relative phase, effective Hamiltonian, number exchange, pendulum, phase locking, real scalar field | nothing relevant (multifield oscillons arXiv:2010.07789 and a classical EFT for large oscillons arXiv:2208.04334, titles only) |
| 97 | oscillon initial conditions, arbitrary oscillation phase, time-shifted quasi-breather against single-frequency ansatz, energy depends on phase | van Dissel 2023 (read), Gálvez Ghersi & Braden 2024 (read); Fodor et al. hep-th/0609023 and others (titles only) |
| 98 | oscillon collisions, intermediate relative phase pi/2, real scalar, phase-shifted copy | Xue 2025, Amin & Mou 2020, Campos et al. 2026 (read); the only intermediate phases found are Campos et al.'s (1+1D) |

No claim of novelty is made. For F-009: no source found for the phase dependence of the slice energy of a single-frequency start; no source found for N exchange, a relative-phase pendulum or a stable anti-phase state between real oscillons in any dimension.

### C6. Not read, open

- Afanasjev, Chu, Malomed, PRE 57, 1088 (1998), the source Malomed 1998 gives for the negative mass of the phase; Karpman & Solov'ev (1981) and Gordon (1983), the 1D NLS equations for relative phase and amplitude difference; Ahn, MacKay, Sepulchre (2001) beyond the abstract; Hertzberg, Schiappacasse, Yanagida (arXiv:2001.07476), which Amin & Mou cite for other relative phases; the unpublished Battye, Mackay, Sutcliffe paper.
- Alonso-Izquierdo et al. 2025 ([arXiv:2504.17382](https://arxiv.org/abs/2504.17382), ABSTRACT in S1.4) describe a two-variable collective model for charge swapping; I did not check whether it contains a pendulum.
- The cusp structure in Gálvez Ghersi & Braden (Fig. 4) is a different set-up (a breather start in a deformed equation) and is context only; the text I read does not explain it.
- No source reports the harmonic amplitude ratio eps for the sextic family. N22 gives the 3 omega source at the core (11% of the linear term), which is not eps.
- The comparison of Battye & Sutcliffe's pendulum constants (M, epsilon^2) with the project's (|d omega/dN|, C_N) was not done.
