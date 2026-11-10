# Session 5 predictions on record

Created in WP0 before any run. The orchestrator adds definitions and decision rules below before any run.

## Block 1. Copied from section 5 of the plan

| # | Prediction | Basis | Literature expects (scholar, WP-A) |
|---|---|---| --- |
| C1 | In C-grid both the single bump and the S-bend exist as trapped vibrations with rate above the ceiling. | Sievers and Takeno 1988; Flach and Willis, "Discrete breathers", arXiv:patt-sol/9704004. | Agrees for G2. Both patterns found by Newton at Omega_b = 4.5, phi2 = phi4 = 1 ([Flach and Gorbach 2005](https://arxiv.org/abs/cond-mat/0410026) READ); at beta = 0.1 exact breathers exist at every frequency above the band ([Cretegny et al.](https://arxiv.org/abs/cond-mat/9709204) READ); small-amplitude proof when 3 K4 - 4 K3^2 > 0 ([Sanchez-Rey et al.](https://arxiv.org/abs/nlin/0306025) READ, James proof SECONDHAND). Sievers and Takeno 1988 only proposes the mode ([abstract](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.61.970) ABSTRACT). G1: no source found; at small amplitude it is G2 with beta = 1. See PRIOR_ART.md, Session 5 brief Track C, sec. 6. |
| C2 | In C-grid with G2 the S-bend is stable and runs unchanged for more than 32,000 cycles. The single bump is unstable, and the instability makes it move, not die. | Sandusky, Page, Schmidt, Phys. Rev. B 46, 6161 (1992). | Agrees on the ordering and on "moves, not dies": [SPS abstract](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.46.6161) ABSTRACT; Page stable and Sievers-Takeno unstable (real Floquet multiplier) in four READ sources: [Flach and Gorbach 2005](https://arxiv.org/abs/cond-mat/0410026), [Duran et al.](https://arxiv.org/abs/2111.09167), [Cretegny et al.](https://arxiv.org/abs/cond-mat/9709204), [Sanchez-Rey et al.](https://arxiv.org/abs/nlin/0306025). "More than 32 000 mode oscillations" is in the SPS abstract for the even-parity mode; its lattice, potential and amplitude: no source found (docs/S5_SPS1992_PARAMS.md). Caution: a perturbed Sievers-Takeno mode can oscillate about the Page mode (half a site away, my reading), so rule 2.3 "moves by at least one unit" may score it as not moving. |
| C3 | The same ordering holds with the asymptote law G1. | Extension of C2. No source. | No source found for the log law (the plan is right). The ordering is not universal: it depends on bond versus site nonlinearity, on the symmetry of the potential (a cubic term spoils the Page mode) and on the hardening sign (dark breathers reverse it), see [Flach and Gorbach 2005](https://arxiv.org/abs/cond-mat/0410026), [Sanchez-Rey et al.](https://arxiv.org/abs/nlin/0306025), [Johansson and Jason](https://arxiv.org/abs/1412.0994), all READ. G1 is bond, symmetric, hardening, like G2, so analogy points the same way at small amplitude; near the pole no source found. |
| C4 | A trapped vibration in C-grid emits nothing at its own rate or its overtones, since all of them sit above the ceiling. Energy inside a fixed length stays constant to solver precision. | Flach and Willis; `scholar` to confirm. | Confirmed, agrees. The exact periodic solution has all harmonics above the band, so no radiation; quasi-periodic (perturbed) objects radiate slowly ([Flach and Willis, secs. VIII, XIX](https://arxiv.org/abs/patt-sol/9704004) READ; [Flach and Gorbach 2005](https://arxiv.org/abs/cond-mat/0410026) READ). A far probe sees evanescent lines at the rate and overtones falling as exp(-kappa L), not zero: set the leakage floor against that. |
| C5 | In C-grid a trapped vibration can be moved with small loss at low speed, and the loss grows with speed. | Sandusky, Page, Schmidt discuss a smoothly travelling version. `scholar` to confirm. | Rests on little. 1D FPU chains carry moving breathers that travel far ([Flach and Willis, sec. IX](https://arxiv.org/abs/patt-sol/9704004) READ); exact moving beta-FPU breathers have wings with resonance peaks in frequency ([Duran et al.](https://arxiv.org/abs/2111.09167) READ); motion generally radiates and the barrier concept is ill defined ([Johansson and Jason](https://arxiv.org/abs/1412.0994) READ); pinning in FPU "too small to be detected" ([Cretegny et al.](https://arxiv.org/abs/cond-mat/9709204) READ). The SPS "smoothly travelling version" is a different object and its abstract gives no loss. No source found for loss per distance against speed; the monotone rule in 2.3 has no support. |
| C6 | C-smooth: no prediction on whether a grid-independent trap exists. Published existence results rely on a grid. | Open question. | Agrees, no prediction; no source found on existence in a bounded-spectrum continuum. The non-existence results (Segur and Kruskal, Kichenassamy, Denzler) are for Klein-Gordon equations with an unbounded spectrum and do not cover C-smooth. The literature mechanism is the unbounded spectrum ([Flach and Kladko](https://arxiv.org/abs/cond-mat/9709049), [Chong and Kevrekidis](https://arxiv.org/abs/2408.15837), READ), so the premise should read "rely on a bounded linear spectrum" (lattice, periodic coefficients, or no linear band). C-smooth is the locally nonlinear Eringen rod and the improved Boussinesq equation ([Duruk et al. 2009](https://research.sabanciuniv.edu/id/eprint/8722/1/IMA_Duruk-Erbay-Erkip.pdf) READ). |
| F1 | Equation R traps the single bump below the floor. | Amin, "K-oscillons", arXiv:1303.1102. `scholar` to confirm the sign condition for L_R. | |
| F2 | In three dimensions the energy along a trapped family has a minimum at a finite amplitude. | Small-amplitude theory. `scholar` to source. | |
| F3 | The pinch exists in O and in R with higher energy than the single bump at the same rate. No prediction on its lifetime. | Zhang, Jain, Amin, arXiv:2111.08700. Their model is not ours. | |
| F4 | For a long-lived swirl, energy per cycle stays nearly constant while energy and rate drift slowly. | New measurement. Do not cite the debunked 50,000 times claim in the findings ledger. | |

Aaron's own expectations, recorded so the data can speak to them:

- A swirl has an optimal length and rate and the range is narrow. Too long, or moving too fast, and it becomes a travelling wave.
- The S-bend traps energy better than the single bump.
- The single bump is probably not stable enough to build a structure. He is open to being wrong.
- Swirls can merge, but they can also hold together through their geometry. Such a structure has a field that reaches further than a lone swirl's. Not tested here. Session 6.
- The pinch is the object he started from and he wants to see how it behaves.

## Block 2. Definitions and decision rules (orchestrator). Written 2026-11-10, before any solver existed and before any run

Every number below is a target (CLAUDE.md: within about 3 times is a pass with a note). Rules are applied as written; a rule that turns out to be badly posed is reported as such, not replaced after the data are seen.

### 2.1 The yardstick, the same for every object

- **Region.** Track C: a fixed window of half-length W = max(10, 5 times the object's initial half-energy half-length) centred on the starting centre. For an object that moves, the same window centred on the tracked energy centroid (co-moving) is reported beside the fixed one. Track F: the sphere of 3 half-widths, the Session 2 definition unchanged.
- **frac(t):** cycle-averaged energy in the region over its value at t = 0.
- **T_90:** first time frac falls below 0.90. **T_death:** first time the energy in the region falls below half its value at t = 50. Both as in Session 2. Quoted in time units and in cycles (time times measured rate over 2 pi).
- **Cap.** Every run states its cap. A run alive at the cap is "> cap". No extrapolation.
- **Rate:** from upward zero crossings of the core coordinate over whole periods, cross-checked against the spectrum peak. Ceiling: 2 in C-grid, 1 in C-smooth. Floor: 1 in Track F.
- **Length:** L_half, the half-length of the shortest window holding half of the cycle-averaged energy, and the rms energy width. Track F: hw as in Session 2.
- **Energy per cycle:** E divided by the number of cycles per unit time, 2 pi E / rate.
- **Regime check (standard 1):** the rate relative to the ceiling or floor, and the share of the energy that is not quadratic.
- **Trapped:** frac at least 0.90 at 1,000 cycles, the linear twin from the same start at most 0.50 at the same time, and the rate on the forbidden side by more than the spectral resolution. If the twin also holds more than 0.50: **"held, but not by stiffening"**, reported with both numbers.

### 2.2 Ceiling or floor: is a tighter vibration faster or slower

Along each family, Spearman rank correlation between rate and length over at least 8 points. rho at most -0.9: tighter is faster (ceiling signature). rho at least +0.9: tighter is slower (floor signature). Otherwise "no monotone relation", with the curve shown. For plucks that are not family members (C-smooth), the same rule on rate against start width at fixed displacement amplitude, and separately at fixed peak stretch.

### 2.3 Rules for the Track C predictions

- **C1 exists:** a time-periodic solution to solver precision with rate above 2, and "trapped" by 2.1.
- **C2, C3 stable:** largest Floquet multiplier within 1e-6 of the unit circle, and in direct evolution peak amplitude and centre within 1% and fixed-window frac at least 0.99 at 32,000 cycles. **Unstable and moves:** a Floquet multiplier above 1 + 1e-3, and in direct evolution the energy centroid leaves its start site by at least one grid unit while the co-moving frac is still at least 0.90. "Dies" would be co-moving frac below 0.50.
- **C4:** for the stable object started from the periodic solution, the drift of the fixed-window energy over 10,000 cycles is at most 10 times the closed-string conservation error of the same run, and no line at the rate or its overtones stands above the leakage floor at a far probe.
- **C5:** at the lowest of at least five push strengths the object crosses 100 grid units with co-moving frac at least 0.90, and the energy lost per unit distance does not decrease from one speed to the next. The break speed is the lowest measured speed at which co-moving frac falls below 0.50 within 100 units.
- **C6, C-smooth, applied to each start at h = 1/4, 1/8, 1/16:**
  - **grid-independent trap:** "trapped" by 2.1 at all three h, and frac, rate and L_half agree across the three within 5%, with the change from 1/8 to 1/16 smaller than from 1/4 to 1/8;
  - **artifact of the numerical spacing:** the width scales with h (log-log slope of L_half or rms width against h at least 0.5), or the state holds at one h and not at a finer one;
  - **held, but not by stiffening:** as in 2.1;
  - **selection:** from start widths 0.25, 0.5, 1 and 2 the final widths agree within 20%: "the medium selects a size". Final over start width within 0.67 to 1.5 for all four: "no selection, the state keeps the width it was given";
  - otherwise undetermined.

### 2.4 Rules for the Track F predictions

- **F1:** at least one start of equation R, single bump, is "trapped" by 2.1 with rate below 1. If every start disperses like its twin or trips the guard: miss.
- **F2:** along a trapped family an interior point has energy below both neighbours and below both ends by more than the family solver's error.
- **F3:** a pinch start (f = 0 at the centre) is "trapped" in O and in R; at three or more common rates its energy exceeds the single bump's.
- **F4:** over the stretch of a lifetime run in which E has changed by at least 1%, the change in ln(energy per cycle) is at most 0.2 of the change in ln E. If nothing changes by 1%: "not testable, nothing drifted".

### 2.5 Rules for Aaron's expectations

- **Optimal length and rate, narrow range.** Narrow: the trapped range spans less than a factor 2 in length and less than a factor 2 in rate. Optimal: an interior point of the window has a lifetime at least twice both edges, or, if lifetimes exceed the cap, a loss rate at most half. "Too long becomes a travelling wave": starts longer than the window's upper edge fall below frac 0.50 within 1,000 cycles. "Too fast becomes a travelling wave": a break speed below the long-wave speed 1 exists.
- **S-bend traps better than single bump.** Supported if the S-bend's T_death (co-moving), or its retained frac at the cap, exceeds the bump's at 4 or more of 5 points matched in energy, in the same medium. Contradicted if the reverse. "Not distinguishable" if both exceed the cap with loss rates within a factor 2.
- **Single bump not stable enough to build with.** Supported if the unperturbed bump leaves its site or dies within 1,000 cycles in the medium where it exists; contradicted if it stays within one unit and frac at least 0.99 for 32,000 cycles.
- **Structures and reach:** not tested, Session 6.
- **The pinch:** reported with its numbers (existence, rate, energy against the bump, lifetime); no pass or fail.

## Block 3. Orchestrator's own predictions, derived on paper, unchecked. Written 2026-11-10 before any run

These are mine, not the plan's and not the literature's. The reviewer checks the derivations; the scholar says whether they are known.

**Basis for C-smooth.** With w = u_x, the C-smooth equation is w_tt = -F(w) + K * F(w), where K * is convolution with (1/2) exp(-|x|) (apply the inverse of 1 - d_xx to the x-derivative of the equation). Every point is its own oscillator with restoring force F, coupled to the rest only through a smooth average. Linear check: omega^2 = 1 - 1/(1 + k^2) = k^2/(1 + k^2).

| # | Prediction | Basis | Literature |
|---|---|---|---|
| O1 | Tails. C-grid: the envelope of a trapped vibration falls by exp(-kappa) per site with alternating sign, 2 cosh(kappa/2) = rate. C-smooth: any tail at rate omega above 1 falls as exp(-kappa x) with kappa = omega / sqrt(omega^2 - 1), which is above 1 and does not alternate. Within 5% | Linear equations outside the core | Known (method), formulas DERIVED. The tail exponent follows from linear theory with the frequency as input, and the lattice tail is staggered ([Flach and Willis](https://arxiv.org/abs/patt-sol/9704004), [Flach and Gorbach 2005](https://arxiv.org/abs/cond-mat/0410026), [Flach and Kladko](https://arxiv.org/abs/cond-mat/9709049) eq. 6, all READ). 2 cosh(kappa/2) = Omega and kappa = omega/sqrt(omega^2 - 1) follow from the two dispersion relations (DERIVED, agree). Caveat: a forced nonlinear harmonic tail can fall more slowly than its free one; for pure power laws the tail is double exponential. |
| O2 | Surroundings. The time-averaged displacement and stretch of a lone bump or S-bend are zero to solver precision at every distance: both objects obey u(t + half a period) = -u(t). A lone object has no static far field in this model | Symmetry of F (odd) | Known for the lattice: a symmetric potential gives only odd harmonics and zero static deformation, "A_(2k,l) = 0 ... d_+ = 0" ([Flach and Gorbach 2005](https://arxiv.org/abs/cond-mat/0410026) READ), hence u(t + T/2) = -u(t). C-smooth: same argument for odd F (DERIVED). Asymmetric potentials give a static kink-shaped deformation. Holds for the exact periodic object, not a perturbed one. |
| O3 | C-smooth linear twin, rest Gaussian displacement pluck of width sigma (u = A exp(-x^2/2 sigma^2)), window half-length 10, t = 2000: retained energy fraction erfc(a) + (2a/sqrt pi) exp(-a^2) with a = 5.76 sigma. That is 0.88 at sigma 0.1, 0.25 at 0.25, 0.001 at 0.5. Within 0.05. So a narrow pluck is held by the linear medium alone | Group velocity (1 + k^2)^(-3/2); energy spectrum k^2 exp(-k^2 sigma^2); stationary phase | No source found for the numbers. The dispersion omega = c k/sqrt(L(ik)) of the nonlocal rod is READ ([Duruk et al. 2009](https://research.sabanciuniv.edu/id/eprint/8722/1/IMA_Duruk-Erbay-Erkip.pdf) eq. 2.4); with L = 1 + k^2 the group velocity is (1 + k^2)^(-3/2) (DERIVED). The retained-energy formula re-derived and agrees (0.87 at sigma 0.1; the 0.88 quoted is rounding). |
| O4 | C-smooth with G1, pluck with sigma at most 0.25: the rate at the point of peak stretch a is within 10% of the lone-oscillator rate of w_tt = -F(w), 2 pi / (4 integral_0^a dw / sqrt(ln((1 - w^2)/(1 - a^2)))), which is above 1 and rises with a | Coupling term is small for a narrow pluck | No source found. The formula is the period of w_tt = -w/(1 - w^2), correct as algebra (DERIVED). |
| O5 | C-smooth with G1: no selected size. Points of different stretch run at different rates, so the stretch profile winds up finer and finer (its mean wavenumber grows roughly linearly in time) until it reaches the numerical cutoff pi/h. Pointwise profiles stop agreeing across h after a time proportional to 1/h, while frac(t) in the window still agrees. The rule in 2.3 returns "no selection" | Phase mixing of uncoupled oscillators; no term in the equation smooths w | No source found for the claim. The mechanism is nonlinear phase mixing (Dawson 1959, SECONDHAND); the improved Boussinesq equation is an ODE in a Banach space with a smoothing coupling (proof of Theorem 2 in [Duruk et al. 2009](https://research.sabanciuniv.edu/id/eprint/8722/1/IMA_Duruk-Erbay-Erkip.pdf), READ). Not contradicted. |
| O6 | C-smooth: a corner in u (a jump in stretch) stays at the point where it was made, in the linear and the nonlinear medium | The only coupling is through the smooth kernel K | No source found. Consistent with the same structure (each point an ODE under a smooth drive, DERIVED); [Duruk et al. Theorem 3](https://research.sabanciuniv.edu/id/eprint/8722/1/IMA_Duruk-Erbay-Erkip.pdf) (READ): regularity in x is preserved, not improved; a jump in w is outside their class (s above 1/2). |
| O7 | C-smooth with G1, sigma = 0.5, peak stretch 0.8: frac at t = 2000 is at least 0.5 where the twin holds 0.001. Low confidence | The strongly stretched part runs above 1 and cannot hand its energy to travelling waves at first order | No source found. Nothing published to compare. |
| O8 | F4 misses in Track F for any lump that loses more than a few percent: along a family dE = omega dN, so energy per cycle falls about as fast as the energy | Session 4 primary: ln E changed by -0.52 while ln omega changed by +0.04 | Session 2 and 4 data |
| O9 | Equation R: in the variables (f, p) with p the momentum conjugate to f, X stays below 1 identically, so the X guard cannot trip for finite p; what can fail is resolution (a steepening front). Energy density sqrt((1 + p^2)(1 + S)) - 1 + f^2/2, positive. The quartic gradient energy has the opposite sign to the term retired on 2026-10-01 | My algebra from L_R; numerics and reviewer derive independently | scholar, WP-A items 4 and 5 |

## Block 4. Addendum after the reviewer's independent derivation. Written 2026-11-10, before any production run

Source: reviewer, `scratchpad/reviewer_S5_derivation/derivation.md` (equations derived independently; O1 to O9 checked; some verdicts rest on short single-resolution probes and are labelled so there). Blocks 1 to 3 stand as written and are scored as written. This block fixes definitions the reviewer found missing and records where Block 3 was wrong or incomplete, so the scoring cannot drift.

**Corrections to Block 3 on record (my errors, kept visible):**
- O1: the C-smooth tail law holds for the component at the rate only. Harmonic n decays with kappa_n = n omega / sqrt(n^2 omega^2 - 1), which tends to 1, so the far envelope falls slower. The 5% test uses the spectral component at the rate.
- O2: the symmetry is u -> -u with a half-period shift (odd harmonics only). It holds for family members, not for an arbitrary pluck. A Gaussian displacement hump in C-smooth has non-zero area, which is conserved and must leave as long waves. O2 is scored on family members and on jump-profile states only.
- O5: the reviewer's probes (t up to 150, one resolution) show winding for about 100 time units, then saturation of the mean wavenumber at 8 to 20, and the uncoupled picture failing by t = 5 to 10 for sigma of 0.25 and above. O5 stays on record as written and is expected to miss in part.
- O8: the conclusion stands; the reason is that |d ln omega| is much smaller than |d ln E| (Session 4: -0.56 against -0.52).
- O9: correct for the bump. For the pinch X < 1 is not sufficient: transverse waves need f_t^2 <= 1 + f_r^2 + f^2/r^2 (scholar DF-5 and reviewer agree independently). The guard tests 1 - X above a small positive number, and the transverse condition for the pinch.

**Definitions added:**
- **Objects in C-smooth.** Single bump: u even about the centre, u = A exp(-x^2 / 2 sigma^2) at rest. S-bend: u odd about the centre, u = A (x/sigma) exp(1/2 - x^2 / 2 sigma^2) at rest (peak displacement A). Neither is a family member. A third start is added: the **jump profile** of the reviewer's single-frequency analysis (a plateau of stretch amplitude near A0 for |x| < x_s, jumping to a small opposite-phase tail), in its bump-like and S-bend-like forms where they exist.
- **Core coordinate.** Single bump: displacement at the centre. S-bend: the stretch across the centre (grid: u_1 - u_0 for the pair of centre sites).
- **Amplitudes on the grid.** Patterns are scaled so the peak stretch is below 1; families are labelled by rate and peak stretch.
- **Twin time.** Rule 2.1 compares at 1,000 cycles of the nonlinear run (C-smooth ceiling period 2 pi, so about 6,283 time units when the rate is near 1). O3 is checked at its own t = 2000.
- **Jump measure for rule 2.3.** Added to the artifact test: the largest jump in stretch between neighbouring cells, the total variation of the stretch, and the number of cells the largest jump spans, each against h. A state whose transition is one cell wide at every h is reported as "a jump in stretch, not resolved by any spacing": neither grid-independent nor a plain artifact, with its position and height against h given.
- **Rule 2.3 when some starts are not trapped:** selection is judged only on trapped starts; fewer than three trapped starts gives "undetermined".

**Predictions added (reviewer's analysis, single-harmonic closure; mine to own once written here):**

| # | Prediction | Basis |
|---|---|---|
| O10 | C-smooth with G1 holds no smooth standing state above the ceiling. A held state has a jump in stretch at 0.04 to 0.3 units from its centre, one cell wide at every h, with rate converging at first order in h | Single-harmonic profile equation: concavity argument, fold at f'(A*) = omega^2 |
| O11 | A jump-profile start at rate 1.2 or 1.5 keeps frac at least 0.99 to 1,000 cycles at all three h, where a Gaussian start of sigma at least 1 disperses like its twin | Reviewer's probe: frac 0.9998 to t = 300 |
| O12 | The 3D small-amplitude family of equation R is unstable (rate 1 - (3/16) A^2; charge grows toward rate 1). A trapped R bump, if any, sits at an amplitude beyond a minimum of E and N | Charge-against-rate criterion; Amin's criterion (scholar) |

**Plan ambiguities resolved for this session (deviations noted for the report):**
- The plan's g4 and g6 are the Session 1 constants divided by 6 and 120 (0.05 and 0.001); beta = 4/3 is the Session 1 label.
- The truncation of R at X^2 has the opposite sign to the term retired on 2026-10-01; it fails for its own reason (hyperbolicity lost for static S of 2/3 and above). The full root is used, as planned.
- Equation R limits X, not stretch; G1 limits stretch, not speed (the long-wave speed in a stretched string exceeds 1). Both are stated in the report's limits.
