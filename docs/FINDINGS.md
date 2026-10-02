# Findings

One entry per established result. Each gives the statement, the evidence, its limits, and its status against the literature: **new** (no source found, with the search reference in `docs/PRIOR_ART.md`; web search only, so "no source found" is not proof of absence), **published** (source named), or **not checked**.

Started 2026-10-02 (Session 3, WP0) from the Session 1 and Session 2 reports. Entries are added at the end of each session. Paths are under `outputs/session2/` unless stated. Since 2026-10-02 `outputs/` in the main checkout is a link to an external drive; the files at these paths are there.

## A. Binding and arrangements

### FND-01. Equal swirls of a single real field do not bind geometrically at first order
- **Statement.** In the pair law, neighbours attract only when their relative phase is below 90 degrees, and that is exactly where the relative phase is unstable (dtheta'' = K sin dtheta, K > 0). At 90 degrees a uniform stepped ring's phase mode is neutral; above 90 the phase is stable and neighbours repel. So attraction and phase stability exclude each other, and no uniform stepped ring has a stable balance.
- **Evidence.** `docs/WPC_DERIVATION.md`; `docs/WPD_NOTES.md` with its review corrections; `wpD/`, `wpD_measured/`; Session 2 report, Verdict and section 6.
- **Limits.** Derived for equal lumps and uniform stepped rings in the first-order tail law, which is valid beyond about 3.6 hw. **It does not cover second-order forces, unequal lumps, orbital motion, or a second field.** The one second-order probe (E11) did not hold either. The 90-degree rings do not hold their pattern in 3D (FND-04). Not shown: that nothing can bind.
- **Status.** The pair structure is published (Malomed 1998; Battye & Sutcliffe 2000; Desyatnikov & Kivshar 2002; nearest on rings, Dong, Liu & Malomed 2025). The ring statement for real-field lumps is this project's derivation; no source found (Addendum F item 5a).

### FND-02. No arrangement of distinct swirls stayed bonded in the 3D tests at beta = 4/3
- **Statement.** 14 arrangements plus E11's second runs: 13 of 14 merged, flew apart, or were never distinct lumps; E11 merged in its second run (first pair within 2 hw at t = 699.4; 720.1 on a finer grid). E14 is the far-apart control.
- **Evidence.** `wpE/analysis/criteria_E.json`, `wpE/results/`.
- **Limits.** One beta, one lump size (hw 3.34), one grid (dx 0.55; E11 also 0.393). Only E1 has a linear twin. Resolution and box checks cover E3 only, up to contact. The Session 2 bonded criterion was flawed; it is replaced in `swirl/engine/bonded.py` (Session 3 WP0), under which E11's first run is "not bonded" and E14 "no interaction".
- **Status.** New (searches 17, 64, 66, 69, 77, 78, 111, 118). Bound 3D clusters of stepped-phase lumps exist for complex fields (Crasovan 2003; Dong, Liu & Malomed 2025), held by angular momentum.

### FND-03. Point-lump scan: no stable balance among 595 patterns
- **Statement.** 28 force balances among 595 patterns (312 classes of 0/180 patterns, 283 stepped); none stable once phases and N move. 27 of 28 have lumps closer than 3.6 hw; 19 grow faster than one oscillation.
- **Evidence.** `wpD/`, `wpD_measured/`; `docs/WPD_NOTES.md`.
- **Limits.** The model is valid beyond about 3.6 hw. Each pattern is scanned along its one-parameter family of sizes; non-radial equilibria are not excluded. Model B's phase and N sector is not validated.
- **Status.** New (searches 17, 64, 66, 77, 78, 111, 116).

### FND-04. Rings with 90-degree steps exchange N early in 3D
- **Statement.** E6 (four lumps) reaches a 34% N spread by t = 200 and flies apart; E10 (eight lumps) reaches 52% by t = 150 and its eight lumps become four. The point model treats these rings as a fixed point of the N flow.
- **Evidence.** `wpE/analysis/`; F-019.
- **Limits.** Single runs; cause not diagnosed.
- **Status.** Not checked.

### FND-05. Mirror-plane no-go (N15): derived, not tested directly
- **Statement.** For a real field with an even potential, a configuration that is odd under a mirror plane is pushed apart across that plane at every instant; without radiation no such state can be bound.
- **Evidence.** `docs/PRIOR_ART.md`, Session 2 brief, N15. Consistent with the alternating ring E5, which flew apart.
- **Limits.** **The plane stress was not computed in any run, so the statement was not tested directly.**
- **Status.** Published for stationary complex fields (Cunha et al. 2022). For real-field oscillons, derived here; no source found.

### FND-06. Claude Desktop's two balance points hold in the point model, not in 3D
- **Statement.** A 5-ring with 72-degree steps balances at kappa R = 1.0695, radially restoring, at a spacing of 0.90 hw. An octahedral shell around an opposite-phase centre balances at kappa R = 1.591, not restoring. In 3D neither is a set of distinct lumps (E4: 2 distinct lumps in the first window; E8: one lump from t = 25).
- **Evidence.** `wpD/predictions_point_model.json`; `wpE/analysis/`.
- **Limits.** Both balances lie in the overlap region, where the model is not valid.
- **Status.** Not checked.

## B. Lone swirls

### FND-07. A lone swirl at beta = 4/3 exists in 3D and keeps its energy to T = 5000
- **Statement.** The Session 1 primary (omega 0.9091) keeps frac 0.994 at T = 1000 and 0.9755 at T = 5000, tracking the radial solver to 7.3e-5. A 2% perturbation with no symmetry (Session 1 run 5) ends within 1e-4 of the unperturbed run, with no growth of quadrupole or octupole moments to T = 1000. Radial T_90 is 10,630.
- **Evidence.** `outputs/step2/results/run1_primary.json`, `run5_primary_generic.json`, `outputs/step2/diagnostics/`; `wpB/compare_radial.json`.
- **Limits.** Runs other than run 5 keep exact cubic symmetry and cannot show non-spherical growth. Run 5 excludes growth in l = 1 to 3 faster than about 5e-3 per time unit, from seeds of order 1e-3, to T = 1000. Its a2 bump at t = 200 and the linear twin's revival are undiagnosed.
- **Status.** Radial existence is published (Mukaida 2017; Ibe 2019; Li, Yamaguchi & Zhang 2025). No source found for a non-radial stability test of the sextic family.

### FND-08. Lifetime of a lone swirl across beta (radial)
- **Statement.** 100 long runs to death or T = 30,000, seven to 300,000. At beta of 1.0 and below every start died within 30,000 (the longest at 29,404). At 1.15 the best start was alive at 30,000. From 4/3 to 5, no start chosen for a low loss rate died within 300,000. Starts that did die above 4/3: the smallest lumps at 4/3 (N/N_min 1.3 and below) and three very large ones (N/N_min 20 to 34 at beta 1.6, 2 and 3, dying at 580 to 1,491). At beta 3 and 5 long life is broad: 10 of 11 and 10 of 10 starts alive at 30,000 with frac above 0.96.
- **Evidence.** `wpA/lifetime_table.json`, `wpA/lifetime_table_v2.json`, `wpA/lifetime_vs_beta_v2.png`; F-010, F-012.
- **Limits.** Each row is the best of the starts tried, and lifetime depends on the start as strongly as on beta. Above 4/3 the runs are capped: **the ordering is not measured, and a fall in lifetime above 4/3 is neither shown nor excluded**. T_90 at matched size is not monotone in beta (F-012). Permanence is not shown. No doubled box for the 300,000 runs at beta 1.6, 2, 3; no resolution check of any 300,000 run.
- **Status.** New as a map for this potential (searches 20, 29, 32, 35, 70, 83 to 87, 99 to 101, 105, 106, 114, 119). Published lifetimes exist at beta = 4/3 only. The "longevity cliff" near beta 1.6 to 2.1 (N17) is not seen.

### FND-09. A lone swirl at beta = 4/3, omega = 0.9065 is very long-lived (not new)
- **Statement.** The start at omega 0.906505 (N/N_min 1.83) keeps frac 0.9946 to T = 300,000 (about 43,000 oscillations), with a loss rate of 3.3e-9 rising to 1.7e-8. In 3D it tracks the radial curve to 7.0e-5 over T = 3000. The low rate sits in a dip about 1.7% wide in N: the start at N/N_min 1.80 falls below 0.90 at 29,396, and the Session 1 primary (omega 0.0026 higher) at 10,630.
- **Evidence.** `wpA/long_beta1.3333_w0.906505_T300k.json`, `wpA/box2_beta1.3333_w0.906505_T300k.json`, `wpA/confirm3d/a9_comparison.json`; F-011.
- **Limits.** 3D covers 3000 time units with exact cubic symmetry; **no generic-perturbation 3D test exists for this start**. The rate rose fourfold, so a finite lifetime is expected.
- **Status.** Published: a lump of this size at beta = 4/3 alive beyond 3e5 is visible in Li, Yamaguchi & Zhang 2025 (figure reading). No source found for a measured loss rate below about 5e-9, a retained fraction at 300,000, or a 3D test (Addendum F item 2).

### FND-10. A lone swirl at beta = 5 shows no measurable decay after the first 30,000 time units
- **Statement.** The start at omega 0.9756 (N/N_min 1.97, hw 6.62) keeps frac 0.998 at 300,000 (0.1% lost before 30,000; rate zero within error after); the start at 0.9912 keeps 0.969. In 3D it matches its radial run to 2.4e-5 over T = 1000.
- **Evidence.** `wpA/long_beta5.0000_w0.975598_T300k.json`, `wpA/long_beta5.0000_w0.991190_T300k.json`, `wpA/box_check_beta5_long.json`, `wpA/confirm3d_beta5/a10_comparison.json`.
- **Limits.** 3D is 155 oscillations with exact cubic symmetry. The doubled box is checked only to 30,000, for one start. No resolution check of the 300,000 run. Permanence is not shown.
- **Status.** New in the sense of FND-08 (Addendum F item 1).

## C. Pairs

### FND-11. Two swirls follow the derived tail law
- **Statement.** At 3.6 to 7 hw, while the lumps stay nearly equal, the force follows U = -2 pi A^2 cos(dtheta) exp(-kappa d)/d to about 3 to 7%; the measured kappa is 0.4172 against 0.4165 derived. In phase the pair attracts and merges; in anti-phase it repels; in between, N flows from the phase leader to the lagger, the relative phase runs toward 180 degrees and the pair ends up repelling. No pair turns around twice. Pairs not at 0 or 180 carry momentum from the start, P_x = -omega I(d) sin(dtheta).
- **Evidence.** `wpC/analysis/criteria.json`, `wpC/results/`; `docs/WPC_DERIVATION.md`; `docs/WPC_BUILD_NOTES.md` section 9.
- **Limits.** The law fails inside 3.6 hw, as derived. A 2 to 6% excess of force and N flow at d = 20 and 24 is unexplained. The 45-degree pair at d = 16 shows 1.3 to 1.5 times the law once the lumps are unequal, also unexplained (F-013). No clean test that linear lumps do not interact (F-015).
- **Status.** The tail law and the charge exchange are published for complex solitons and Q-balls (Malomed 1998; Battye & Sutcliffe 2000); a phase-dependent exponential force for 1+1D oscillons (Campos et al. 2026, abstract only). No source found for a measured force, decay constant, N exchange or phase drift between real 3D oscillons (Addendum F item 3).

### FND-12. Opposite phases repel in this model
- **Statement.** Anti-phase pairs repel (shifts -5.45, -3.14, -0.69 at d = 16, 20, 24 against -5.7, -3.2, -0.71 derived), and so do 135-degree pairs. The uncited sentence on Wikipedia ("Oscillon") that opposite phases attract and form bonded pairs is contradicted.
- **Evidence.** `wpC/analysis/criteria.json`.
- **Limits.** One potential, one lump size.
- **Status.** Published: every primary source has in phase attracting and anti-phase repelling.

## D. Mergers

### FND-13. Mergers do not turn a ring into one bigger swirl (single runs)
- **Statement.** A merging 5-ring throws out two lumps back to back, each with 0.36 to 0.41 of the starting energy (E1, E2, E3); the 6-ring E7 leaves two of 0.45. A shell in deep overlap around an opposite-phase centre leaves one lump of about half the energy, still decaying (E8: 0.47 E0, omega 0.83, losing 0.23% per unit time; E9 about 0.45). E11 is a slow collapse driven by a swing of the centre-to-ring phase about 180 degrees (plus or minus 17 degrees, period about 350), ended by neighbours pairing off; the five lumps left fly apart.
- **Evidence.** `wpE/analysis/`, `wpE/results/`; `wpD_measured/e11_seed_scan.json`; F-017, F-018.
- **Limits.** **Single runs**, not resolution-checked after contact. Absorbed angular momentum is not recorded. That E11's swing starts from a frequency difference between centre and ring is an untested hypothesis. The Session 2 end-of-run labels counted debris (F-018); the lump finder has an absolute floor since Session 3 WP0, and with it E3, E7 and E4 read "merged" at the end. `E_retained_end` measures residence in the group sphere, not survival; the key `E_residence_end` now carries it.
- **Status.** New (searches 18, 62, 112, 113, 115). Closest: an in-phase ring of 3D quantum droplets fuses into a state elongated along the ring axis (Dong et al. 2025; complex field; no ejection described).

## E. Method results

### FND-14. The phase of a single-frequency start changes the lump
- **Statement.** The t = 0 energy of the start Phi cos(theta), -omega Phi sin(theta) depends on theta (2843.8 at 0, 2605.2 at 90 degrees; 9.5% in N), so starts at different phases are different lumps. The offset divided by dE/domega predicts the settled frequency shift (-0.0009, +0.0025, +0.0059 against -0.0008, +0.0026, +0.0066 observed at 45, 67.5, 90). Copies of one settled lump cut at different times remove the problem (N equal to 4e-6 at the start).
- **Evidence.** F-009; `wpC/` single-lump controls; `docs/WPE_BUILD_NOTES.md`.
- **Limits.** One beta, five phases.
- **Status.** A derived consequence of the ansatz; no source found (Addendum F item 4).

### FND-15. The path-integral lifetime estimate failed its validation
- **Statement.** Integrating dE/(E Gamma(E)) over a measured rate map predicted 11,540 against 1145.5 (beta 0.85) and 33,006 against 10,630 (beta 4/3). All 38 dying starts were predicted at or above their measured death.
- **Evidence.** F-007; `wpA/checks.json`.
- **Limits.** The run's own rates fed to the integral give +0.4% and +1.3%, so the integral is sound; the failure is in the map (unsampled energy ranges, interpolation across a rate null).
- **Status.** Published agreement is 2 to 7% away from deep dips (Ibe 2019; Li 2025); no source for a failure this size.

### FND-16. Stepped rings carry angular momentum from the start
- **Statement.** 4 to 13% of N per lump at 4.8 hw (7.2% for E1), within +0.5 to +3.4% of the overlap formula, conserved to 0.2 to 1.5% until merger or ejection. None rotates at J/I before contact.
- **Evidence.** `wpE/analysis/criteria_E.json` (E-rot); `docs/WPE_BUILD_NOTES.md`.
- **Limits.** Absorbed angular momentum is not recorded.
- **Status.** Not checked.
