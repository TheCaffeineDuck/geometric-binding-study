# Session 5 report

Date: 2026-11-10. Plan: `docs/SESSION_05_PLAN.md`. Predictions and rules: `docs/SESSION_05_PREDICTIONS.md`. Status log: `docs/SESSION_05_STATUS.md`. Nothing was pushed. No cloud compute was used.

Every number below was measured in this session unless it is labelled calibrated (set by a fit or an input), derived (from a formula), or build (measured during the solver build, before production).

## 1. Summary

**Track F, the floor trap in three dimensions (round objects only).**

- **Equation R slows dispersal but did not trap anything.** Every R start died. Bump starts from R's own family live up to 4.0 times as long as the same start in the linear medium, and the lifetime grows with the square of the width (11 to 12 times the half-width squared at the wide end). A wide enough start therefore passes a fixed 1,000-cycle test without being trapped: the widest three (half-widths 29, 37, 45) kept 0.906, 0.997 and 1.000 of their starting share at 1,000 cycles and died at 9,286, 16,308 and 25,148 time units. Away from the family, a start either disperses in 10 to 23 cycles or, above a threshold amplitude, the equation breaks down in finite time. I score prediction F1 a miss over the range tested, half-widths 3 to 45.
- **The pinch exists in the old equation and holds:** at rate 0.86 it lasts 95,412 time units (13,105 cycles) and holds at 0.99 and 1.01 times its amplitude. It carries 4.5 to 5.8 times the energy of a bump at the same rate. A bump at 1.02 times the Session 1 amplitude was still at 0.972 when its run stopped at 100,000, so the two are not ranked.
- **In the floor medium a tighter vibration is slower,** the opposite of the string: on the small-amplitude branches the rate rises toward the floor as the object widens (rank correlation +1.000, 32 and 59 points).
- **The Session 1 baseline bump sits on the wall of the narrow dip found in Session 2.** At 0.99 times its amplitude it is at the limit of the hold rule (0.9003 against 0.90); at 0.98 it fails (0.819); at 1.02 it holds to 100,000 time units.
- **Energy per cycle is not constant** as an object decays; it falls 1.05 to 1.10 times as fast as the energy. F4 is a miss, as I predicted (O8).

**Track C, the string (ceiling trap).**

- **On a grid the ceiling trap works, and the S-bend is the object that holds.** With the asymptote spring law (G1), the S-bend ran unchanged at all six rates tested, from 2.05 to 4.2, for 32,000 cycles: the energy inside its window stayed at 0.99999994 or more of the start value, against 0.0014 to 0.0030 for the same start with the stiffening switched off.
- **A tighter vibration is faster on the string.** Along both grid families the rate rises as the length falls at every point (rank correlation -1.0, 21 and 24 points). The S-bend runs from rate 2.01 at half-length 2.79 to rate 4.32 at half-length 0.52.
- **The single bump is unstable and turns into an S-bend.** It does not die and it does not travel. It slides half a site, sheds 1.4% to 2.6% of its energy, and lands on the S-bend family within 0.03% to 1.6% in energy.
- **Without a grid, no smooth trapped vibration was found.** In the smooth medium every smooth start leaked. The only states that held are ones with a sharp corner in the string (a jump in stretch). The equation itself keeps such a corner fixed in place for all time, so this is a property of the medium and not of the numerical grid; the numerics converge on it at second order. These states are unstable at a rate that does not depend on the numerical spacing (0.007 to 0.009 per cycle), and two of the three that were run long lost hold near 3,000 to 4,000 cycles. One (rate 1.85) showed no loss to 5,000 cycles.
- **Misses.** Seven predictions missed or split on Track C: motion under a push (C5), the bump's instability (C3, split), and five of my own (O2, O5, O6, O10, O11). They are in section 6 at the same depth as the hits.

## 2. What was run

| Package | Worker | Outcome |
|---|---|---|
| WP0 housekeeping, predictions file | runner | Done before any run (722778d) |
| WP-A prior art | scholar (two, one per track) | Gate passed for the smooth medium and for equation R, with conditions |
| WP-B1 string solver | numerics | 67 tests; reviewer found 9 defects, all fixed (49b9dc9) |
| WP-B2 radial solver | numerics | Reviewer found 5 defects; fixed (dc6796d, 2e5d2fa); multi-harmonic R states parked after two attempts |
| WP-C Track C runs | runner | 3 h 20 min; then diagnostics D1 to D6 |
| WP-D Track F runs | runner | 1 h 15 min; families, lifetimes, 1% tests, boundary scans, pinch against bump (301dca5); then diagnostics DF1 to DF4 (6ac7f5e) |
| WP-E review | reviewer | Equations derived independently before production; Track C and Track F numbers recomputed; this text reviewed |

Models used on the string (units: long-wave speed 1, smallest scale 1, stretch limit 1):

- **C-grid:** masses one unit apart, u_n'' = F(d_n) - F(d_{n-1}), with d the stretch of a link. Travelling waves exist only up to rate 2 (the ceiling).
- **C-smooth:** u_tt - u_xxtt = d/dx F(u_x). Travelling waves exist only up to rate 1.
- **G1:** F(d) = d / (1 - d^2), force rising without bound as the stretch nears 1. **G2:** F = d + beta d^3, used only for the regression against published work.
- **Linear twin:** the same start with F(d) = d.

## 3. Track C: the seven questions

### C-Q1. Existence

*Prediction C1: in C-grid both objects exist as trapped vibrations above the ceiling. C6: no prediction for C-smooth.*

**C-grid: yes for both objects. C1 is a hit.**

- Exact periodic states were found by Newton iteration at 21 rates for the S-bend (2.01 to 4.316) and 24 for the bump (2.01 to 4.916). The worst error in returning to the start after one period is 1.1e-11.
- Rule 2.1 (energy in the window at least 0.90 after 1,000 cycles, twin at most 0.50, rate above the ceiling): met by the S-bend at 6 of 6 rates and by the bump at 5 of 6. The bump at rate 4.2 fails (0.747).
- Under the stricter Block 6 rule (the 0.99 and 1.01 amplitude starts must also hold), only three were tested: S-bend 2.5, S-bend 3.5 and bump 2.5. All three hold (0.9998 or better for the S-bends, 0.987 to 0.989 for the bump). The other nine are "held", not "trapped", in Block 6 terms.
- The bump rows meet the rule on energy, but the object in the window at the end is no longer a bump (C-Q3).

**C-smooth: no smooth trapped state; states with a jump in stretch hold.** See C-Q7.

### C-Q2. The window

*No numbered prediction. Aaron's expectation: the range is narrow.*

C-grid families, G1 (full tables: `outputs/session5/trackC/TABLES.md` table 1; plot `plots/family_curves.png`):

| Object | Rate | Peak stretch | Half-length | Energy | Energy per cycle |
|---|---|---|---|---|---|
| S-bend | 2.01 to 4.316 | 0.162 to 0.993 | 2.79 to 0.515 | 0.134 to 2.449 | 0.418 to 3.566 |
| Bump | 2.01 to 4.916 | 0.161 to 0.991 | 2.79 to 0.409 | 0.134 to 4.141 | 0.418 to 5.293 |

- **The window is not narrow.** The rate spans a factor 2.15 (S-bend) and 2.45 (bump); the half-length spans a factor 5.4 and 6.8.
- **The lower edge is the ceiling.** As the rate falls toward 2 the object grows without limit in length and shrinks in amplitude.
- **No upper edge was found.** With G1 the rate keeps rising as the peak stretch approaches 1. The families stop at peak stretch 0.99 because that is where I stopped them, not because the object ends there.
- **Energy and energy per cycle rise along the whole family.** There is no minimum and no preferred member picked out by energy.
- **"Too long" starts.** Staggered patterns under a Gaussian envelope, started at rest: envelope widths 3 and 6 keep 0.81 to 0.99 of their energy after 1,000 cycles; widths 12 and 24 fall to 0.02 to 0.67 and break into 2 to 6 lumps. Two width-24 runs at peak stretch 0.6 tripped the stretch guard and are not scored.

C-smooth: the window was not mapped. All jump states were built with the jump 0.25 from the centre, so their half-length (0.151) is an input, not a measurement. The three states that hold have rates 1.42, 1.46 and 1.85, energies 0.36, 0.39 and 0.62.

### C-Q3. Lifetime of the bump against the S-bend

*Prediction C2 (G2): S-bend stable beyond 32,000 cycles; bump unstable, and the instability makes it move, not die. C3: the same ordering with G1.*

**C2 (G2, regression against Sandusky, Page and Schmidt 1992), build measurement:** the S-bend exact start is unchanged at 32,000 cycles (energy fraction 0.99999988). The bump is unstable (largest Floquet multiplier 2.062 at rate 4.5, reproduced independently by the reviewer). Whether it travels depends on how it is disturbed: the exact start settles half a site away, and a start with a travelling disturbance moves 2.5 sites. **Hit on stability, split on "moves".** The full text of the 1992 paper was not obtained, so this is scored against its abstract and a later review.

**C3 (G1), production: split.**

C-grid lifetimes, cap 32,000 cycles (plot `plots/lifetimes_A2_frac.png`):

| Object | Rate | Energy fraction at cap | Twin | T_90 | Outcome |
|---|---|---|---|---|---|
| S-bend | 2.05, 2.2, 2.5, 3.0, 3.5, 4.2 | at least 0.99999994 | 0.0014 to 0.0030 | beyond cap | Unchanged: amplitude change at most 1.35e-3, centre shift at most 2.5e-7 |
| Bump | 2.05 | 0.9999 | 5.5e-4 | beyond cap | Slow drift of 0.997 sites; multiplier only 2.9e-4 above 1 |
| Bump | 2.2 | 0.9997 | 5e-6 | beyond cap | Slides toward the S-bend position (max shift 0.99) |
| Bump | 2.5 | 0.988 | 7e-4 | beyond cap | Slides half a site, becomes an S-bend |
| Bump | 3.0 | 0.974 | 6e-4 | beyond cap | Becomes an S-bend at rate 3.37 |
| Bump | 3.5 | 0.969 | 4e-4 | beyond cap | Becomes an S-bend at rate 4.07 |
| Bump | 4.2 | 0.747 | 4e-4 | 116 cycles | Sheds a quarter of its energy |

- **S-bend stable: hit, 6 of 6.** Floquet multipliers differ from 1 by 1e-14 to 6e-14 once the four that belong to exact symmetries are set aside. The multipliers at these six rates are interpolated from the family table, not computed at the rates themselves.
- **Bump unstable: hit.** Its largest multiplier rises from 1.0001 at rate 2.04 to 605 at rate 4.92.
- **"Moves by at least one site": miss, 0 of 6.** Diagnostic D3 shows what happens instead. At the end of 10,000-cycle reruns at rates 2.5, 3.0 and 3.5, the object is centred between two sites (within 0.002 to 0.04 of a site), its energy is 98.4% to 100% in the S-bend symmetry, and its (rate, energy) point lies on the S-bend family curve within 0.03%, 0.4% and 1.6%. At rate 2.2 it is still drifting.
- The 1.0-site threshold is missed by 0.3% at rate 2.05 and by 1.3% at 2.2.

### C-Q4. Leak

*Prediction C4: nothing is emitted at the rate or its overtones; energy in a fixed length is constant to solver precision.*

**Hit, with a criterion I must disclose as changed after the data.**

- S-bend at rate 2.5 for 10,000 cycles: the energy in the window drifts by 6.0e-12 on a closed ring and 5.8e-12 with absorbing ends. The solver's own conservation error is 2.8e-8, so the drift is 2e-4 of it (rule: at most 10 times).
- The absorbing ends took up 1.6e-15 of the energy in 10,000 cycles. This is the best bound on the leak, and it is comparable to the seeded noise.
- **Disclosure.** The runner's first spectrum test was "no line above 3 times the local noise floor". One line, the fifth overtone at 60 sites with absorbing ends, came out at 3.24 times. Under that test the spectrum part would have been a miss. The test was replaced after seeing the data by the Block 6 floor (below 1e-5 of the start energy counts as unresolved), which was on record before the runs; the lines are 5e-35 of the start energy or less. The reviewer adds that the far probes were stored in single precision, so their floor is rounding noise, and that the symmetry-forbidden even overtones show the same 1.5 to 2.0 times, which says a line at 2 to 3 times the floor is noise.
- Nothing below the ceiling: the largest far-probe amplitude in the travelling-wave band is 4.6e-10 (closed ring), which is the seeded noise.

### C-Q5. Motion

*Prediction C5: a trapped vibration can be moved with small loss at low speed, and the loss grows with speed.*

**Miss as written.** The rule required the lowest push to carry the object 100 sites with at least 0.9 of its energy, and the loss per unit distance not to fall as the push grows.

| Object, rate | What a push does |
|---|---|
| S-bend 2.1 | Travels at 5.6% of the push size (speed 0.0025 to 0.149) for 89 to 292 sites, keeping 0.966 to 1.000 of its energy. At push 10 it is destroyed (0.52 left after 2.8 sites). |
| S-bend 2.3 | Travels 10 to 20 sites, then stops (19.8 sites at push 1). |
| Bump 2.3 | Travels up to 9.4 sites, then stops. |
| S-bend 3.0 | Does not move (at most 0.24 sites), and loses energy to the kick. |

- The lowest push (0.05) at rate 2.1 covered 89.3 sites in the run length with no measurable loss (energy fraction 1.000). The rule asked for 100, so by the letter it fails; the run was 11% too short. This part of the rule was badly posed.
- Loss per unit distance at rate 2.1 is not monotone: 2.05e-7 at push 0.3, 9.0e-8 at push 1, 3.3e-6 at push 3.
- No speed at which the object breaks into travelling waves was measured. The fastest intact motion is 0.149. Larger pushes either destroy the object in place or trip the stretch guard at the kick; those trips are logged as failure entries.
- **Correction of a first report.** The runner's first tables said the S-bend at 2.3 and the bump at 2.3 were "pinned, no motion". Diagnostic D1 showed that was an artefact of fitting the speed late in a long run. The build's speed (0.0386 at push 0.5) reproduces exactly at the build's run length of 300; the object then stops by about 20 sites.
- **Reading.** The weakly trapped, long S-bend near the ceiling moves almost freely. Tighter, faster ones are held by the grid, and the tightest does not move at all.
- **C-smooth:** jump states cannot be moved. The equation has no term that transports a jump, and the pushed runs show centre shifts of at most 0.24.

### C-Q6. Surroundings

*Prediction O2 (mine): time-averaged displacement and stretch are zero to 1e-8 at every distance.*

**Miss by the threshold; the physical statement behind it is not contradicted for the S-bend.**

- S-bend 2.5: largest mean displacement 5.4e-9, mean stretch 1.1e-8. S-bend 3.5: 2.7e-8 and 5.4e-8. The threshold of 1e-8 equals the size of the seeded noise, so these sit at the floor. The threshold was badly posed.
- The oscillating field falls off as predicted: its size is 0.28 at the centre, 4e-4 at 5 sites, 4e-7 at 10 sites for the S-bend at 2.5, then the noise floor near 1e-9.
- Bump at 2.5 over 200 cycles: mean displacement 2.5e-4 and mean stretch 4.8e-4. Diagnostic D4: this appears about 40 to 50 time units after the bump starts to slide and does not follow the slide (correlation 0.16 to 0.21). It looks like radiation from the conversion to an S-bend, not a static field. Only one probe was usable, so this is indicative.
- **Baseline for Session 6:** a lone S-bend on the grid has no static field above 1e-7 at any distance out to 60 sites.
- C-smooth states with a jump: mean stretch 4e-5 to 1.1e-4 at the centre, falling to 1e-7 or less by 5 units.

### C-Q7. Grid or no grid

*Prediction C6: none. My predictions O5, O6, O7, O10, O11 bear on it.*

**Answer: no smooth trapped vibration survives without a grid. A vibration around a sharp corner does hold, for thousands of cycles, and the corner is a feature of the equation, not of the numerical grid.**

Every start was run at numerical spacings 1/4, 1/8 and 1/16 (plots `plots/smooth_frac_*.png`).

| Start | Energy fraction at about 1,000 cycles, spacing 1/4, 1/8, 1/16 | Twin | Class |
|---|---|---|---|
| Gaussian bump, peak stretch 0.8, widths 0.25, 0.5, 1, 2 (at 1/8) | 0.71, 0.52, 0.013, 3e-4 | at most 0.09 | Not trapped |
| Gaussian S-bend, peak stretch 0.8, widths 0.25, 0.5, 1, 2 (at 1/8) | 0.90, 0.69, 0.001, 1e-6 | at most 0.17 | Not trapped; width 0.25 holds only at spacing 1/4 (artefact) |
| Jump S-bend, rate 1.2 | 0.995, 0.809, 0.789 | 0.23 | Holds only at spacing 1/4 (artefact) |
| Jump S-bend, rate 1.5 | 1.000, 1.000, 1.000 (at 970 to 975 cycles) | 0.24 | Holds; jump not resolved by any spacing |
| Jump S-bend, rate 2.0 | 1.000, 1.000, 1.000 (at 920 to 924 cycles) | 0.28 | Holds; jump not resolved by any spacing |
| Exact periodic S-bend state, rate 1.4231 (calibrated input) | 1, 1, 1 | 0.24 | Holds; jump not resolved by any spacing |
| Jump bump, rates 1.2 and 1.5 | 0.93, 0.87, 0.86 and 0.95, 0.86, 0.90 | 0.74 to 0.81 | Held, but not by stiffening: the twin holds most of it |

Under rule 2.3 as written the count of grid-independent traps is **0 of 16 starts**: the jump is one numerical cell wide at every spacing, and a width that follows the spacing was to be reported as an artefact. I report it under the Block 4 class written for this case, "a jump in stretch, not resolved by any spacing", and not as an artefact, for these reasons.

- **Derivation (reviewer, independent).** In terms of the stretch w, the smooth-medium equation is w_tt = -F(w) + K * F(w), where K * means a weighted average over neighbouring points with weight (1/2) exp(-|x|). No spatial derivative acts on w. So each point of the string is its own oscillator, driven by a smooth average of its neighbours. A jump in w stays exactly where it is, forever. No jump can form from a smooth start, and none can move. The linear equation keeps jumps too.
- **What a converged jump looks like.** The scheme is this equation with a discrete average (checked to 5.6e-16). A one-cell jump at a fixed position at every spacing is what a correct solution with a true jump must look like.
- **Convergence (D5).** For the exact state at spacings 1/4, 1/8, 1/16, 1/32: energy 0.36501, 0.36181, 0.36101, 0.36081. Successive differences fall by a factor 4. The rate of the jump states (1.456, 1.461, 1.462) and the tail fall-off converge the same way. At spacing 1/64 the iteration landed on a state with energy 0.39087, 8% off the trend; this is unexplained and was not followed up.
- **Smooth starts cannot reach these states.** That follows from the derivation, and the data agree: 0 of 10 Gaussian starts is trapped at all three spacings. The 0.5 to 0.9 that narrow Gaussian starts retain is fine structure wound up to the numerical cutoff (final wavenumber at 0.5 to 0.76 of the cutoff, doubling when the spacing halves). That part is an artefact of the grid.
- **The jump states are unstable, and the instability is not a grid effect (D5, D6).** The exact state has one growing mode at every spacing, with growth 0.0090, 0.0067, 0.0069, 0.0080 per cycle at spacings 1/4 to 1/32. The mode is a wave of wavenumber about 3 that fills the box; it is not at the jump and not at the grid scale. In a box twice as long the growth is 15% to 20% lower (0.0056 to 0.0061). In the time runs the loss grows at 0.0072 to 0.0096 per cycle after staying near 1e-5 for 2,000 cycles, and the 90% time is 4,065 cycles at spacing 1/4 and 3,117 at 1/8. So the lifetime is set by how small the starting disturbance is, not by the object.
- **The exception.** The jump S-bend at rate 2.0 showed no growth: 0.9997 or better to 5,000 cycles at spacings 1/4 and 1/8, and to the 2,001 cycles reached at 1/16. No stability calculation was done for it.
- **Not shown.** Independence from the numerical scheme: the second scheme (spectral) crashed on the rate-2.0 start, and for the rate-1.5 start it gave 0.857 at 300 cycles against 1.000. The reviewer expects this from the overshoot a spectral method makes at a jump. It is one start at one spacing and it is open.
- **Part of the hold is not stiffening.** The twin keeps 0.18 to 0.29 of the S-bend forms, because in this medium short waves hardly travel at all (speed falls as the cube of the wavelength). For the bump forms the twin keeps 0.74 to 0.81.

*My predictions here.* O7 hit: at time 2,000 the Gaussian bump of width 0.5 keeps 0.61, 0.60, 0.59 against 9e-4 for the twin. O5 miss, O6 split, O10 split, O11 miss: section 6.

## 4. Track F: the five questions

Units: wave speed 1, floor (lowest rate of a travelling wave) 1, limit 1. Model O is the Session 1 equation at beta = 4/3. Model R is L = 1 - sqrt(1 - X) - |u|^2/2. All runs are round (one radial profile), grid spacing 0.05, absorbing shell at 100 to 200 (O) or 200 to 400 (R). Tables: `outputs/session5/trackF/TABLES.md`. Plots: `plot_families.png`, `plot_frac_vs_time.png`, `plot_boundary_scan.png`, `plot_bump_vs_pinch_E.png`.

**Regression.** The O bump reproduces Session 1 bit for bit (rate 0.9096711; energy fraction 0.994127 at time 1,000; build).

**Three places where the hold rule did not fit equation R** (found by the runner, reported as found, nothing changed):
1. The rule asks for 0.90 of the start energy inside three half-widths. R profiles have wide tails, so they start at 0.71 to 0.92 by that measure.
2. The rule asks for the rate to be below the floor by more than the frequency resolution (0.031). The whole R family lies within 0.0054 (bump) or 0.0135 (pinch) of the floor. Rates from zero crossings are quoted instead.
3. The classifier uses the instantaneous energy fraction, the rule says cycle-averaged. They differ in the third decimal.
4. Two defects in the classifier (reviewer): it calls a run "disperses" whenever a lifetime exists, even if the run met the rule at 1,000 cycles; and it counts 1,000 cycles using the end-of-run rate (for example 6,258 time units against 6,652 counted cycle by cycle).

The tables now carry two columns for every run: the rule by the letter (cycle-averaged share of the total start energy at 1,000 counted cycles), and a "retained" reading (share relative to the start, rate from zero crossings). Verdicts below say which is used. The rule's own text says "over its value at t = 0", which supports the retained reading.

### F-Q1. Does equation R trap a swirl below the floor?

*Prediction F1: R traps the single bump. The scholar's brief said this rested on existence arguments only.*

**Not in any test made: every start died. R does hold energy up to 4.0 times longer than the linear medium. F1 is a miss over the range tested.**

- **Periodic profiles exist** (single-harmonic approximation): a bump family at rates 0.99459 to 1 and a pinch family at 0.98652 to 1. Both turn back at their lowest rate.
- **None holds when evolved.** Production runs, half-widths 3.0 to 22.4. Lifetime is the time to fall to half the reference energy fraction:

| R start | Lifetime, time units (cycles) | Late rate |
|---|---|---|
| Bump, amplitude 0.2 | 4,106 (653) | 1.000 |
| Bump, 0.33 | 1,304 | 1.000 |
| Bump, 0.55 | 376 | 1.000 |
| Bump, 0.79 | 140 | 1.000 |
| Bump, 0.90, 0.95, 1.0 (past the turn) | 102, 94, 90 | 1.000 |
| Pinch, 0.038 | 1,152 (183) | 1.000 |
| Pinch, 0.092 | 412 | 1.002 |
| Pinch, 0.2 to 0.51 | 153 down to 65 | 0.998 to 1.033 |

- Every late rate is at or above the floor: what remains has become ordinary waves.
- A further production start at energy 1,000 (half-width 22.4) lasted 5,403 time units (860 cycles) and kept 0.34 of its starting share at 1,000 cycles, the best of the production set.
- **Wider starts (diagnostic DF1, run after the reviewer pointed out that the family extends to half-width 56 and only 22.4 had been run).** Absorbing shell moved out to 400 to 800; a run with the shell at 800 to 1,600 agrees to 1e-4.

| Half-width | Retained at 1,000 cycles (by the letter) | Twin | Rate passes above the floor at cycle | Lifetime | Lifetime / half-width squared | Twin lifetime / half-width squared | Run / twin |
|---|---|---|---|---|---|---|---|
| 28.6 | 0.906 (0.830) | 0.294 | 680 | 9,286 | 11.3 | 3.11 | 3.64 |
| 37.0 | 0.997 (0.914) | 0.262 | 1,269 | 16,308 | 11.9 | 3.11 | 3.83 |
| 45.0 | 1.000 (0.917) | 0.460 | 2,020 | 25,148 | 12.4 | 3.11 | 3.99 |

- **Reading.** The linear twin's lifetime is 3.11 to 3.19 times the half-width squared for half-widths 6.8 to 45: that is plain spreading. (The four narrowest starts, half-widths 3.0 to 4.4, have twin values 4.65 to 8.71.) The R run's is 8.1 at half-width 6.8, rising steadily to 12.4 at 45. So R delays spreading by a factor that grows slowly with width: 1.1 at the narrowest and largest-amplitude start (half-width 3.0), 3.99 at the widest. Each run ends the same way: the rate drifts up through the floor, then the object goes. The nonlinear share of the energy in these wide starts is 0.05% to 0.12%.
- **Against the rule.** The half-width 37 and 45 starts meet the energy and twin clauses of the hold rule at 1,000 cycles, by the letter and by the retained reading. They do so because they are wide: the horizon of 1,000 cycles is shorter than their spreading time. The twin at half-width 45 is itself at 0.46, close to the 0.50 limit.
- **The 1% test at half-width 37 (DF2) is not passed.** At 0.99 times the amplitude: 0.945 retained, 0.866 by the letter. At 1.01 times: the run tripped the resolution guard at the centre at t = 4,433 (704 cycles, X = 0.994); the refinement test could not classify it, so it is "unresolved".
- **The wide pinch (DF3, half-width 45):** 0.601 retained at 1,000 cycles, rate above the floor from cycle 402, twin 0.021.
- **Away from the families** (Gaussian and shell starts, amplitude bisected to 1%):

| Start | Threshold amplitude | Below | Above |
|---|---|---|---|
| Bump, width 2 | 0.927 | Disperses in 62 time units | Collapse at the centre at t = 3.82 |
| Bump, width 3 | 0.848 | Disperses in 64 | Collapse at t = 12.85 |
| Bump, width 5 | 0.613 | Disperses in 148 | Collapse at t = 42.0 |
| Shell at radius 10 | 0.214 | Disperses in 129 | Collapse at t = 36.1 |
| Pinch, width 3 | 1.72 | Disperses in 98 | Equation stops being well posed at t = 37.9 |

- **Collapse** means 1 - X heads to zero at the centre in finite time. The times at which 1 - X first passes a fixed shallow level agree across three grid spacings to 0.1% (width 5, level 0.1: 42.013, 42.006, 42.003; width 2, level 0.1; width 3 and the shell, level 0.01, the shell on two grids only). At a deep level (1e-7; width 3, amplitude 1) the two finer grids differ by 1.5% (7.400 against 7.291). So the approach to collapse is a property of the equation; the last stage is not resolved. The solver is second-order accurate in the first cells at the centre for the bump.
- **For the pinch** the failure is different: the condition for sideways waves to propagate fails at radius 2.3 while X is still well below 1 (1 - X = 0.046). The time agrees across three grids (37.9125, 37.925, 37.925). A round run cannot follow what happens next.
- **What was not tried:** periodic states with more than one harmonic (two attempts failed and the work is parked), starts with inward velocity, half-widths beyond 45 (the family goes to 56), and non-round shapes.
- **Guard trips** are logged as failure entries. The equation was not changed to avoid them.

**Why I score F1 a miss and not "undecided".** The reviewer's position after the production runs was that the miss was earned only up to half-width 22.4. DF1 and DF2 extend the range to 45: every start still dies, the best one fails the 1% test, and the pass at 1,000 cycles follows from width alone. What stays open is whether the delay factor keeps growing for still wider objects; it rose only from 3.64 to 3.99 while the width went from 29 to 45.

### F-Q2. Families: rate, size, energy, energy per cycle, and the energy minimum

*Prediction F2: the energy along a trapped family has a minimum at a finite amplitude.*

| Family | Rate | Half-width | Energy | Energy minimum | Turning point |
|---|---|---|---|---|---|
| O bump | 0.8006 to 1 | 2.94 to 56 | 1,659 to 81,196 | 1,659.0 at rate 0.9711 | In amplitude, at rate 0.876 |
| O pinch | 0.7968 to 1 | 6.02 to 163 | 7,926 to 184,019 | 7,926.2 at rate 0.9558 | In amplitude, at rate 0.888 |
| R bump | 0.99459 to 1 | 3.00 to 56 | 145.7 to 2,515 | 145.65 at rate 0.99494 | In rate, at 0.994586 |
| R pinch | 0.98652 to 1 | 4.48 to 163 | 307.9 to 14,674 | 307.87 at rate 0.99725 | In rate, at 0.986521 |

- **F2: hit for O with a qualifier, not scorable for R.** Both O families have an interior energy minimum (Session 1 found 1,657.1 for the bump). The qualifier: starts at the minimum do not hold (bump at 0.965 and 0.98 die in 285 and 418 cycles; pinch at 0.965 in 71). The objects that hold (bump 0.909, pinch 0.86) are at larger amplitude than the minimum. The R tables have minima too, but no R family is trapped, and the prediction was about trapped families.
- Energy per cycle follows the energy along every family (the rate changes by under 1.5% where it matters): its logarithmic slope against energy has median 0.999.
- The O families stop near rate 0.80 because the profile solver runs out of precision there. The R bump family stops where the single-harmonic approximation reaches its limit. The R pinch upper branch was stopped by hand at 104 points.

### F-Q3. Lifetime

*Session 2 definitions unchanged. Cap 20,000 time units for O, with two runs extended to 100,000.*

| Start | Energy fraction at 1,000 cycles | T_90 | Lifetime | Outcome |
|---|---|---|---|---|
| O bump 0.82 | 0.291 | 186 | 429 | Disperses |
| O bump 0.86 | 0.443 | 1,680 | 4,083 | Disperses |
| O bump 0.909 (Session 1 baseline) | 0.960 | 10,630 | beyond 20,000 (0.597 at the cap) | Held |
| O bump 0.94 | 0.828 | 1,985 | 8,279 | Disperses |
| O bump 0.965, 0.98, 0.995 | 0.0004 to 0.010 | 388 to 2,657 | 575 to 2,722 | Disperses |
| O pinch 0.82 | 0.510 | 233 | beyond 20,000 (0.504) | Alive at cap |
| O pinch 0.86 | 0.982 | none | 95,412 (13,105 cycles) | Held |
| O pinch 0.91 | 0.796 | 794 | beyond 20,000 (0.785) | Alive at cap |
| O pinch 0.94 | 0.964 | 13,620 | 13,666 (2,063 cycles) | Meets the rule at 1,000 cycles, gone by 2,063 |
| O pinch 0.965, 0.98, 0.995 | 0.0001 to 0.527 | 417 to 3,262 | 462 to 8,035 | Disperses |
| R, 17 production starts | see F-Q1 | | 65 to 5,403 | Disperses |
| R bump, three wide diagnostic starts | 0.830, 0.914, 0.917 | 4,770 to 14,013 | 9,286 to 25,148 | Disperses |

- Linear twins are below 2e-3 at the same times, except the smallest R bump (2e-2).
- **The 1% rule.** Three production starts meet the hold rule by the letter: O bump 0.909, O pinch 0.86, O pinch 0.94 (the runner first reported two; the classifier had marked the third as dispersing). A fourth, a bump at rate 0.853 from the matched-energy set, also meets it (0.908) and has no 1% test.
  - O pinch 0.86 passes: 0.992, 0.982, 0.969 at 0.99, 1.00, 1.01 times the amplitude.
  - O pinch 0.94 passes at 1,000 cycles (0.979, 0.964, 0.950; diagnostic DF4), but all three are gone by 1,472 to 2,063 cycles. The rule's horizon is what it passes.
  - O bump 0.909 is marginal: at 0.99 times the amplitude it is at 0.9003 against the limit 0.90, with T_90 at 999 cycles; at 1.01 it is at 0.988.
- **The baseline is on the wall of a narrow dip in the loss rate.** Loss rate per time unit (fit over times 2,000 to 10,000) at 0.98, 0.99, 1.00, 1.01, 1.02, 1.05 times the amplitude: 4.5e-5, 3.1e-5, 9.8e-6, 1.6e-6, 4.6e-8, 2.0e-6. Energy fraction at 20,000: 0.0009, 0.0026, 0.597, 0.914, 0.996, 0.965. The minimum is near 1.02, where the object holds to 100,000 time units (0.9716, 14,437 cycles), and the loss rises again at 1.05. This is the dip Session 2 reported (rate 0.9065 holding 0.9946 to 300,000, with the Session 1 primary "on the wall of the same dip"; `docs/SESSION_02_REPORT.md` lines 34 and 78 to 79). It is not new and does not conflict with Sessions 1 and 2. The 0.98, 1.02, 1.05 and both 100,000 runs were added after seeing first results and are diagnostics, not scored runs.
- **Radial lifetimes are upper bounds.** A round run cannot see an object lose its roundness.

### F-Q4. Bump against pinch: which has the lower energy at the same rate?

*Prediction F3: the pinch exists in O and R with higher energy than the bump at the same rate.*

**The bump has the lower energy, except on one branch of R. F3 is a split.**

- **O:** the pinch carries 4.5 to 5.8 times the bump's energy at rates 0.95 to 0.999 (family tables), and 2.2 to 4.2 times at rates 0.82 to 0.94 (run points). Its half-width is 1.3 to 2.8 times the bump's. Hit. The published paper (Zhang, Jain, Amin) says only that the pinch is higher in energy; the figure 5.8 is the scholar's own reading of profile integrals in a thesis by the same group, for the low-amplitude limit of a different model.
- **R, small-amplitude branch:** 6.0 to 7.3 times at rates 0.995 to 0.999. Hit on ordering.
- **R, upper branch of the pinch:** its energy stays near 309 to 363 and falls below the bump's above rate 0.99733 (ratio 0.84 at 0.998, 0.58 at 0.999). This contradicts the ordering there. Whether that branch is physical is open; it was stopped by hand.
- **"Exists in R":** as a profile, yes; as something that holds, no, for half-widths up to 20 in production and one diagnostic point at 45 (0.60 retained at 1,000 cycles).
- **Lifetime at matched energy.** O at energy 2e4: pinch held (0.972 at 20,000), bump gone by 2,008. O at 1e4: both alive (pinch 0.80, bump 0.90). R at energy 1,000: bump 5,403 against pinch 333.

### F-Q5. The pinch in the old equation

**It exists and it holds for more than 13,000 cycles at its best rate.**

- Family from rate 0.797 to 1, energy minimum 7,926 at rate 0.956.
- At rate 0.86 it is held for 95,412 time units (13,105 cycles) and passes the 1% rule.
- There is a window: rates 0.82 to 0.94 live past 13,000 time units, with 0.86 the best (it has lost 2% at 1,000 cycles where 0.82 has lost about half and 0.91 a fifth). At rates 0.965 and above it dies within 1,205 cycles. The window rests on seven sampled rates.
- Against the bump at the same rate 0.86: pinch held, bump gone by 4,083. The bump's own best amplitude (1.02 times the baseline, rate 0.909) was still at 0.972 at 100,000, so neither object is ranked above the other overall.
- The longest-lived point (0.86) is not the energy minimum (0.956).
- Limits: round only. The pinch has directions in which it could deform that this run cannot see.

### Energy per cycle (F4, O8)

*F4: for a long-lived swirl energy per cycle stays nearly constant while energy and rate drift. O8 (mine): F4 misses.*

**F4 miss, O8 hit. These are one observation, not two.** In held O runs the energy per cycle changes 0.92 to 1.10 times as fast as the energy (reviewer's values from 20-period averages at each end; the runner's first-to-last values are 1.05 to 1.10; rule for F4: at most 0.2). Energy per cycle is 2 pi times energy over rate by construction, so the ratio is 1 minus the relative drift of the rate, and the rate moves far less than the energy. In plain terms both energy and energy per cycle change by only 2% to 9% over 20,000 time units in these runs, which reads as "nearly constant"; the wording of F4 and the rule written for it disagree, and the rule is what was scored.

### Equation R as a set of equations (O9, O12)

- **O9, split.** Hit: X stays below 1 for any finite momentum, and the pinch needs the extra sideways condition. Miss: I said the only thing that could fail was resolution. What fails is the equation itself, at a time that does not move when the grid is refined (to 0.1% at shallow levels; see F-Q1 for the depth caveat). One run, the 1.01 neighbour at half-width 37, did end as "unresolved".
- **O12, split.** Hit: at small amplitude 1 - rate is proportional to the square of the centre amplitude, with constant 0.00996, steady to 0.1% over six points; and the small-amplitude branch has the sign that marks instability. Miss: my coefficient 3/16 was wrong by a factor 18.8. The scholar traced it: 3/16 is the one-dimensional value. Amin's three-dimensional expansion gives 0.0099666 for R, against 0.009957 measured, and 0.00099666 for O against 0.000997 measured. Miss: I said a trapped R bump, if any, would sit past the energy minimum. Starts there are the shortest-lived of the bump family (90 to 102 time units).

## 5. Is a tighter vibration faster or slower?

| Medium | Measured | Answer |
|---|---|---|
| C-grid, S-bend family | Rate 2.01 at half-length 2.79 up to 4.32 at 0.515; rank correlation -1.0 (n = 21) | **Tighter is faster** |
| C-grid, bump family | Rate 2.01 at 2.79 up to 4.92 at 0.409; rank correlation -1.0 (n = 24) | **Tighter is faster** |
| C-smooth, Gaussian starts at fixed displacement | Rank correlation -0.79 (core rate, n = 8) and -0.19 (rate at peak stretch, n = 11); rule needs -0.9 | No monotone relation shown |
| C-smooth, Gaussian starts at fixed peak stretch 0.8 | -0.62 to -0.76 (n = 8); -0.96 for the bump core rate but n = 7, below the rule's 8 | No monotone relation shown |
| C-smooth, jump states | Half-length is an input (0.151), so not testable | Not tested |
| O bump (floor) | Rank correlation of rate with half-width +0.857 (n = 52). Half-width has a minimum of 2.94 at rate 0.95 and grows again toward the thin-wall end | No monotone relation by the rule; **tighter is slower** from rate 1 down to 0.95 |
| O pinch (floor) | +0.931 (n = 78) | **Tighter is slower** |
| R bump, small-amplitude branch | +1.000 (n = 32); upper branch has 5 points, not scored | **Tighter is slower** |
| R pinch, small-amplitude branch | +1.000 (n = 59) | **Tighter is slower** |
| R pinch, upper branch | -0.987 (n = 46) | Tighter is faster on this branch; nothing on it holds |

So the two traps have opposite signatures, as the plan expected: under a ceiling a tighter vibration runs faster and moves further above the band; under a floor a tighter one runs slower and moves further below it.

In C-smooth the rate at a point is set mainly by the stretch at that point (prediction O4, hit: the measured rate is 3.4% to 9.2% below the lone-oscillator value for peak stretch 0.3 to 0.95). It is the size of the stretch, not the length of the object, that sets the rate there.

**Does the trap survive without a grid?** Stated with its numbers in C-Q7: no smooth start is trapped at all three spacings (0 of 10); states with a stationary jump in stretch hold to at least 2,000 cycles, converge at second order in the spacing, and are unstable at 0.007 to 0.009 per cycle, with one exception that held to 5,000 cycles.

## 6. Predictions: hits, splits and misses

### Track C scoreboard

| # | Prediction (short) | Measured | Verdict |
|---|---|---|---|
| C1 | Both objects exist as trapped vibrations in C-grid | 21 and 24 exact states; rule 2.1 met by 6 of 6 S-bends and 5 of 6 bumps | Hit |
| C2 | G2: S-bend unchanged past 32,000 cycles; bump unstable and moves | S-bend 0.99999988 at 32,000 cycles; bump multiplier 2.062; travel depends on the disturbance (build) | Hit; split on "moves" |
| C3 | Same ordering with G1 | S-bend stable 6 of 6; bump moves a full site at 0 of 6, becomes an S-bend | Split |
| C4 | No emission; window energy constant | Drift 6.0e-12 against conservation error 2.8e-8; absorbed 1.6e-15 | Hit, criterion changed after data |
| C5 | Moves with small loss; loss grows with speed | Only the rate-2.1 S-bend travels 100 sites; loss not monotone | Miss |
| C6 | No prediction | 0 of 16 by rule 2.3; 3 states hold around a jump | Recorded |
| O1 | Tail fall-off laws within 5% | C-grid: worst 8.1% (bump 4.2, a state that is shedding), others about 1e-6. C-smooth: 16 of 18 within 1.7%; two non-steady states far off | C-grid pass with note; C-smooth miss by the worst-case rule as written, hit for steady states |
| O2 | Mean field zero to 1e-8 | 5.4e-9 to 5.4e-8 (S-bend); 2.5e-4 (bump, converting) | Miss |
| O3 | Linear retention of a Gaussian in C-smooth | All six within 0.018 (0.864 against 0.882 the worst) | Hit |
| O4 | Rate at peak stretch within 10% of lone oscillator | 3.4%, 6.2%, 9.2%, 5.7% below | Hit (9.2% is close to the limit) |
| O5 | Winding to the cutoff, no selected size (five parts) | 1 of 5 parts by the runner's count, 2 of 5 by the reviewer's | Miss; two parts badly posed |
| O6 | A jump stays where it is made | All 9 held states: 0 cells. Worst non-held run: 6 cells | Split; measure changed after data |
| O7 | Width-0.5 pluck keeps at least 0.5 at time 2,000 | 0.61, 0.60, 0.59; twin 9e-4 | Hit |
| O10 | Jump states: fixed position, one cell, first-order convergence | Position 0.25 and one cell at every spacing; convergence is second order (ratio 3.87 to 4.06) | Split |
| O11 | Jump starts at 1.2 and 1.5 keep 0.99 at all spacings | 1 of 4 does (S-bend 1.5) | Miss |

### Splits and misses in detail

**C3, split.** Expected: bump unstable and moving at least one site. Measured: 0 of 6 move a full site. Cause, ranked: (1) the unstable direction of the bump is the slide toward the S-bend position, half a site away, where it stops (D3: lands on the S-bend family within 0.03% to 1.6% in energy); (2) at rates 3.0 and above the change is also one of shape, with the rate rising 12% to 16%; (3) the one-site threshold, missed by 0.3% at rate 2.05. Rules in: the bump is a way of making an S-bend. Rules out: the bump dying, and the bump as a mover at these amplitudes. Literature: the instability of the site-centred mode and its oscillation about the bond-centred one are published (Flach and Gorbach 2005; Flach and Willis 1994). The energy match and the shed fraction of the conversion were not found in print (searches 362, 363, 392, 402, 403). Caution: at rate 2.2 the two families differ by only 0.8% in energy, so the energy match does not discriminate there; the symmetry does. Ledger: F-045.

**C5, miss.** Expected: free motion with loss rising with speed. Measured: section C-Q5. Causes, ranked: (1) the grid holds tight objects in place, more strongly the tighter they are; (2) the 100-site rule was missed by run length at the one rate that moves freely (89.3 sites, no loss); (3) the "loss grows with speed" rule had no source and the dip at push 1 may be a resonance. Rules in: nearly loss-free motion for long, weakly trapped S-bends (loss 1.9e-11 per site at speed 0.0025). Rules out: free motion at all amplitudes. Literature: known for the cubic law, not for G1. Shelkan, Klopov and Hizhnyakov (arXiv:1812.03496) report kicked vibrations that stop after a finite distance, sooner at higher rate; Cuevas and others (arXiv:nlin/0202007) report speed proportional to the kick. No support for the monotone-loss rule was found. The scholar adds that the loss-per-distance figure excludes the launch and sits near the noise floor, so that clause is doubly weak. Ledger: F-038.

**O2, miss.** Expected zero to 1e-8. Measured 5.4e-9 to 5.4e-8 for S-bends. Cause: threshold set at the noise amplitude. Rules out a static far field above 1e-7 for a lone S-bend.

**O5, miss.** Expected: the stretch pattern winds steadily to the numerical cutoff, profiles stop agreeing between spacings while the energy fraction agrees, no selected size. Measured: winding saturates by about time 100 at 0.5 to 0.76 of the cutoff (fit quality 0.49); profiles do stop agreeing (0.36 rising to 0.54); the energy fraction differs between spacings by 0.027 in the median and 0.148 at most; selection could not be scored with fewer than three trapped starts. Two of the five parts were badly posed (a relative spread of numbers near zero; a test that cannot return its own predicted outcome).

**O6, split.** Expected: a jump never moves. Measured: zero motion for every held state. The detector follows the largest jump, and in runs that are falling apart the largest jump swaps between two jumps 0.25 apart. The measure was changed after the data from signed position to distance from centre; the verdict is a miss under both by the worst case (14 cells before, 6 after). The derivation in C-Q7 says a jump cannot move.

**O10, split.** Position and one-cell width: hold. First-order convergence: wrong; it is second order, because the jump sits on a cell edge at every spacing used. The exact state's rate is an input, so the runner's ratio of 0.099 for it is not a measurement; its energy ratio is 3.99.

**O11, miss.** One of four jump starts keeps 0.99 at all spacings. The rate-1.2 S-bend is the widest member near a fold of its family and wanders; the jump bumps are held mostly by the linear medium (twin 0.74 to 0.81). Rows were read at 920 to 975 cycles, not 1,000.

**O1 for C-smooth.** Miss by the rule as I wrote it (worst case of 18 fits). The two failures are the rate-1.2 jump S-bend at spacings 1/8 and 1/16, which is not steady and has a late rate of 1.07 and 1.02, where the formula blows up. The other 16 fits are within 1.7%, and for the held states the error falls as the square of the spacing (0.49%, 0.12%, 0.03%).

### Track F scoreboard

| # | Prediction (short) | Measured | Verdict |
|---|---|---|---|
| F1 | R traps the bump | 0 of 11 family starts survive (8 production, 3 diagnostic; half-widths 3 to 45; lifetimes 90 to 25,148); lifetime up to 4.0 times the twin's; the half-width 37 start fails the 1% test; scans find dispersal or collapse | Miss over the range tested |
| F2 | Energy minimum at finite amplitude along a trapped family | O bump 1,659.0 at 0.9711; O pinch 7,926.2 at 0.9558; the held objects are not at the minimum; R not trapped | Hit for O, qualified; not scorable for R |
| F3 | Pinch exists in O and R with higher energy than the bump | O: 4.5 to 5.8 times, held 13,105 cycles. R: 6.0 to 7.3 times on one branch, below the bump on the other; does not hold | Split |
| F4 | Energy per cycle nearly constant | Changes 0.92 to 1.10 times as fast as energy (limit 0.2) | Miss |
| O8 | F4 misses | Same numbers | Hit |
| O9 | X cannot reach 1; only resolution can fail | Algebra holds; failure is real collapse, times converged to 0.1% at shallow levels (1.5% at depth 1e-7) | Split |
| O12 | Small-amplitude R family unstable, rate 1 - (3/16) A^2; a trap, if any, past the minimum | Scaling holds with constant 0.009957, matching the published 3D value 0.0099666, not 3/16; starts past the turn die in 90 to 102 | Split |

### Track F misses in detail

**F1, miss over the range tested.** Expected: R traps the bump (basis: Amin's work on such kinetic terms; the brief warned it rested on existence only). Measured: section F-Q1. Causes, ranked: (1) the small-amplitude family is on the unstable side (O12), and its instability time grows with the square of the width, so wide members last long without being stable; (2) the stiffening in R only becomes strong near amplitude 0.9, where collapse takes over, and the scans find no stable window between dispersal and collapse; (3) the starts are single-harmonic approximations, least accurate at large amplitude; (4) the hold rule did not fit R, and its fixed horizon can be met by width alone. Diagnostics run: DF1 (lifetime against width), DF2 (1% neighbours and box), DF3 (wide pinch). Rules in: dispersal delayed by a factor up to 4.0; a hard upper limit on amplitude set by breakdown of the equation. Rules out: a trapped R bump reachable from the single-harmonic family at half-widths 3 to 45, or from a Gaussian or shell start at rest. Does not rule out: a multi-harmonic state, or a delay factor that keeps growing beyond half-width 45. Literature: no conflict. Amin (arXiv:1303.1102) covers existence for this kinetic term with a pure mass term, and his own stability criterion makes the three-dimensional small-amplitude family unstable; he gives no lifetime. One short segment of the R bump family (rates 0.99471 to 0.99494) satisfies his criterion and its starts still die in 90 to 102 time units; he limits the criterion to cases where one frequency is a good approximation, and on this family X reaches 1.000 over a cycle at its largest (not checked point by point on the segment). Ledger: F-052, F-056.

**F4, miss.** Expected constant energy per cycle. Measured ratio 1.05 to 1.10. Cause: along a family the change in energy equals rate times change in "amount of oscillation", and the rate hardly changes, so the two fall together. This was O8. Literature: partly known. The adiabatic invariant is described as approximately conserved (Kasuya, Kawasaki, Takahashi hep-ph/0209358; Kawasaki, Takahashi, Takeda arXiv:1508.01028; abstracts only). No source found states how energy over rate behaves along a decay. Ledger: F-053.

**O9 clause, miss.** Expected: failure only by resolution. Measured: collapse times converged across three grids. Cause: I treated the bounded X as protection; it bounds the speed of the field, not the steepening of the profile. Literature: finite-time breakdown is published for the massless equation (Eggers and Hoppe, arXiv:0812.1790). With a mass term, and for the pinch's sideways condition, no source was found. Ledger: F-054.

**O12 clauses.** The 3/16 coefficient: my error (one-dimensional value); the measurement confirms the published three-dimensional one. "Past the minimum": wrong; the upper branch is the short-lived one. Ledger: F-055, F-056.

**Baseline bump marginal on the 1% rule.** Not a numbered prediction, but Sessions 1 to 4 used this object. Reading in F-Q3: it is the Session 2 dip. Literature: lifetime windows and dips are known in other models (Honda and Choptuik hep-ph/0110065; Zhang and others arXiv:2004.01202); nothing found for this model's scan or for the pinch window. Ledger: F-062.

## 7. Aaron's expectations

| Expectation | Verdict | Deciding number |
|---|---|---|
| A swirl has an optimal length and rate | Split | By loss rate the S-bend family has an interior best point and the bump family does not. By energy there is no optimum: energy and energy per cycle rise along the whole family |
| The range is narrow | Split: contradicted on the grid, supported for the O pinch | Rate spans a factor 2.15 and length a factor 5.4 for the S-bend; no upper edge with G1. For the O pinch the long-lived window is rates 0.82 to 0.94, a factor 1.15, with a best point at 0.86: supported there |
| Too long and it becomes a travelling wave | Split | Envelope widths 12 and 24 fall to 0.02 to 0.67 in 1,000 cycles; of the starts longer than 1.44, 4 of 9 are below 0.5. Long exact states near the ceiling (half-length 2.8) hold |
| Moving too fast and it becomes a travelling wave | Not tested | No intact motion above speed 0.149 was achieved; harder pushes destroy the object in place or trip the guard |
| The S-bend traps energy better than the bump | Supported | At 5 of 5 matched energies the S-bend keeps 1.000 against 0.9997, 0.990, 0.980, 0.973, 0.969 for the bump. Bump values are interpolated across six runs. In C-smooth: jump S-bends hold, jump bumps do not hold by stiffening |
| The bump is probably not stable enough to build a structure | Split, leaning supported | Unstable at every rate (multiplier up to 605). It does not disperse: it becomes an S-bend within tens of cycles at rates 2.5 and above. By the rule on record: supported at 1 of 6 rates, contradicted at 2 of 6 |
| Structures and reach | Not tested | Session 6 |
| The pinch, the object he started from | Reported | In the old equation it exists, holds for 13,105 cycles at rate 0.86, passes the 1% test, and costs 4.5 to 5.8 times a bump's energy at the same rate. In equation R it does not hold (9 to 183 cycles). Round runs only |

## 8. Process notes, deviations and limits

**Analysis definitions changed after seeing data** (none changes a verdict except as stated): the C4 spectrum test (section C-Q4); the O6 position measure; the filter for "too long" starts (from "reached 1,000 cycles" to "at least 900 cycles and no guard trip"; 2 of 4 before, 4 of 9 after); guard trips labelled numerical were rerun once at a quarter of the time step, originals kept.

**Corrections made during the session:** the "pinned" wording for pushed objects (D1); four 1% runs for C-smooth that existed but were missing from the tables; cycle counts for jump rows that were labelled as 1,000.

**Deviations from the plan:**
- The full text of Sandusky, Page and Schmidt 1992 was not obtained. C2 is scored against the abstract and Flach and Gorbach's review.
- Two scholars were used for WP-A, one per track.
- Three kinds of start (scaled exact states, staggered envelopes) were built in the run script, not in the solver's driver. The reviewer checked their energies against the twins.
- Multi-harmonic states of equation R were attempted twice and parked.
- The plan was ambiguous on four points (normalisation of the old equation's couplings, the sign of the truncation, what quantity the limit in R applies to, and stretch against speed in G1). The readings taken are in Block 4 of the predictions file.
- The radial family tool hung once when a worker process died without a traceback; it was replaced by a serial tool. Cause not established.
- Result files are in the repository worktrees and on main. They were not copied to the external drive that the main checkout's `outputs/` folder links to, because that is outside this folder. Aaron to decide.

**Workers as seen** (CLAUDE.md asks for this on first delegation):

| Worker | Expected | Seen |
|---|---|---|
| numerics | Opus, high | Opus 5.5, high |
| runner | Sonnet 5.5, medium | Sonnet 5.5; effort shown as 5 |
| reviewer | Sonnet 5.5, xhigh | Sonnet 5.5; effort shown as 30 |
| scholar | Sonnet 5.5, xhigh | Sonnet 5.5; effort shown as 30 |

The effort labels do not match the table for runner, reviewer and scholar. I cannot tell from inside whether 30 is how xhigh is displayed.

**What the plan's limits mean for these results:**
- **One dimension.** The string results say nothing yet about a twist, a pinch or a structure. The S-bend's stability is against disturbances along the string only.
- **The grid.** The clean trap (C-Q1, C-Q3, C-Q4) needs the grid. Without it, what survives needs a sharp corner that the medium cannot create by itself from a smooth start. Whether nature allows a grid, or a corner, is not settled here.
- **Radial symmetry.** Every Track F lifetime is an upper bound: a round run cannot see an object lose its roundness. The pinch's breakdown in R is in a sideways direction that a round run can detect but not follow. What is shown for R is that round objects from the starts listed all die.
- **Units only.** No physical values are attached to any number.

**Standards check (CLAUDE.md, ten items).** Regime numbers (rate against ceiling, nonlinear share) are in the family tables. Stability is judged by energy in a window. Energies include every term (reviewer derivation). Absorbing ends were compared with a closed ring. Every nonlinear run has a twin, except the D2, D3 and D6 diagnostics, which are nonlinear only. Linear and exact values came first (O1, O3, O4). Items 7 and 8 (symmetry classes, structures) do not arise. Each prediction sits beside its literature expectation in the predictions file. No claim of novelty is made in this report.

## 9. Literature

Full entries, sources and the search log (362 to 404 for Track C; S5F-1 to S5F-18 and 405 to 430 for Track F) are in `docs/PRIOR_ART.md`. Failure entries F-038 to F-051 (Track C) and F-052 to F-062 (Track F) are in `docs/FAILURES.md`. No result here is claimed as new.

| Result | Published? | Key source |
|---|---|---|
| Bump unstable, S-bend stable, bump slides to the S-bend position | Known. The energy match of the conversion was not found | Flach and Gorbach 2005; Flach and Willis 1994; Kastner and Sepulchre |
| Pushed vibration travels a finite distance and stops; tight ones are pinned | Known for the cubic law, not found for G1 | Shelkan, Klopov, Hizhnyakov (arXiv:1812.03496) |
| The smooth-medium equation is an ordinary differential equation in a function space and keeps jumps | Known for the equation class; for the linear case the "jump stays, none is created" statement is in print. Not found for G1 | Constantin and Molinet 2002 (secondhand, via Geba and Lin); Erbay, Erbay, Erkip (arXiv:1805.07264); Emmrich and Weckner |
| A time-periodic localized state with a jump in this medium | Not found | Searches in PRIOR_ART, item (c) |
| A proof that no smooth trapped vibration exists in this medium | Not found | Same |
| Second-order convergence with the jump on a cell edge | Not found for jump profiles; second order is proved for smooth data | Erbay and others, Theorem 4.1 |
| Long staggered envelopes break up | Known for the cubic law (modulational instability, packets merge); the argument carries to G1 at small amplitude (derived by the scholar, not sourced) | Cretegny and others (arXiv:cond-mat/9709204) |
| Oscillatory instability that depends weakly on box length | Two known kinds, one that vanishes in a long box and one that persists. Two box sizes cannot tell them apart | Flach and Gorbach 2005; Marin and Aubry (secondhand) |

- **Not obtained:** Constantin and Molinet 2002; Marin and Aubry 1998; Johansson and Kivshar 1999; Sandusky, Page and Schmidt 1992 (full text); a direct source for convergence order with an edge-aligned jump.
- **Where the data disagree with a plan premise, not with the literature:** "the bump's instability makes it move" (the brief flagged this before the runs); "too long becomes a travelling wave" (the published outcome is break-up into merging lumps, which is what the envelope runs show: 2 to 6 lumps).
- **Caution on one paper:** Coclite and others 2026 (arXiv:2609.30041) show jumps dissolving and forming in a linear nonlocal model with a singular weight. The scholar reads this as not applying here, because the weight in C-smooth is bounded. That reading has not been checked by the reviewer.

**Track F.**

| Result | Published? | Key source |
|---|---|---|
| Small-amplitude R and O families: rate against amplitude | Known; measured constants match to 0.1% | Amin, arXiv:1303.1102 |
| R does not trap in three dimensions at small amplitude | Consistent with Amin's stability criterion; no published lifetime or large-amplitude result | Same |
| Finite-time collapse in R | Known for the massless equation; not found with a mass term | Eggers and Hoppe, arXiv:0812.1790 |
| Pinch loses the sideways propagation condition | Not found | Search log 405 to 429 |
| Pinch carries more energy than a bump | Known as an ordering. The ratio 5.8 in the low-amplitude limit is the scholar's derivation from a thesis (arXiv:2401.00043), not a published number; ours 5.79 in O, 5.96 in R at rate 0.999 | Zhang, Jain, Amin, arXiv:2111.08700 |
| Pinch lifetime window in O | Known in kind (the thesis shows a dip in lifetime for a round pinch run), not for this model | arXiv:2401.00043 |
| Energy per cycle not conserved along decay | Partly known: the abstracts say the invariant is "approximately conserved"; the relation between energy loss and its loss is from Ibe and Mukaida (Session 1 brief). No source states how energy over rate behaves along a decay | Kasuya and others; Kawasaki and others (abstracts) |
| Lifetime of an unstable small-amplitude object growing as width squared | Scaling consistent with Amin's slow-time analysis (scholar's derivation); no published lifetime | Search 430 |

- **Not obtained:** Pearce and Amin on large-amplitude objects with this kinetic term (cited as in progress, not found in print); full texts of the two adiabatic-invariant papers.
- **To confirm:** the O comparison with Amin uses a coupling value (0.05) read from the plan's convention; the reviewer has not checked it.

## 10. Proposals for Session 6

1. **The twist on the grid string.** A second direction of sideways motion on C-grid, with the S-bend as the starting object. This is the plan's first candidate and Track C gives it a stable object to start from.
2. **Two S-bends on one string.** Force, merging and the reach of the combined field, using the surroundings baseline of C-Q6 (no static field above 1e-7 for one S-bend).
3. **Settle the smooth medium.** (a) Stability calculation for the rate-2.0 jump state and a scan over jump position, which maps the window that was not mapped here. (b) Box-length scan of the growth rate, which fell 15% to 20% when the box doubled: does it vanish in a long box. (c) The spectral scheme with a treatment of the jump. (d) The 1/64 anomaly.
4. **A medium that can make its own corner, or does not need one.** A smooth medium with a fourth-order term, for comparison with the one used here, which cannot form a jump.
5. **Decide whether equation R is worth more work.** Two capped checks: the delay factor at half-widths 56 and beyond (does 3.99 keep rising), and one attempt at multi-harmonic periodic states with their stability rates. If neither shows a hold, retire R as a trap and record why.
6. **The pinch in the old equation, out of round.** It is the longest-lived object found, but only round runs exist. This is three-dimensional vector work, which the plan forbids this session.
7. **Rewrite the hold rule** before any further floor runs: retained fraction relative to the start; rate from zero crossings; a horizon that scales with the object's own spreading time (the twin's lifetime) so that width alone cannot pass it; and fix the two classifier defects.
8. **Re-base the old-equation baseline** at the bottom of the dip (1.02 times the Session 1 amplitude), since Sessions 1 to 4 used an object on its wall.

## 11. Parking lot

- Session 4 carry-over: the 2 times box rerun of the far-tail force.
- Full text of Sandusky, Page and Schmidt 1992.
- Multi-harmonic periodic states of equation R past the single-harmonic limit, with stability rates.
- A fourth-order closure at the centre for the R solver (it is second order in the first cells for the bump).
- Why the bump's largest multiplier at rate 2.05 is only 2.9e-4 above 1, and the slow one-site relocation there.
- Pushed objects: a direct measurement of the barrier the grid puts in the way of motion, against rate.
- Cause of the worker death in the radial family tool.
- Erratum carried from Session 4: commit 9bd140c says "1.5x box"; the correct figure is 1.33.
- Later: energy per cycle of a preferred swirl; lensing.
