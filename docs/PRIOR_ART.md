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

