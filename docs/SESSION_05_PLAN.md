# Session 5 plan

Date: 2026-11-10. Revision 2. It replaces the earlier draft of the same date.
Status: agreed with Aaron.
This file replaces the model decision in section 2 of `SESSION_02_PLAN.md` ("stay with the single real field"). Everything else in `CLAUDE.md` still applies.
Inputs: `SESSION_04_REPORT.md`, `PRIOR_ART.md`, `FAILURES.md`, `LESSONS_2026-10-01.md`.

## 1. What changed on 2026-11-10

Decisions Aaron made, in order:

| Topic | Decision |
|---|---|
| Model | The medium has a direction of motion at every point, not one number per point. |
| Wave speed | One speed for long waves. |
| Elastic limit | An asymptote. As the limit is approached the stiffness climbs toward infinity. |
| How a swirl is trapped | It is too short and too fast for the medium to carry. Not too slow. |
| The swirl's range | There is an optimal length and rate, and the range around it is narrow. Too long, or moving too fast, and it becomes a travelling wave. |
| Numbers | None inserted. Constants are used as units. Results are reported as relationships. |
| Pictures | Dropped. The math produces the shapes and the force distribution. |
| Yardstick | Lifetime, measured the same way for every object. |

Aaron's reasons, for the record. At scale, space behaves as a free elastic medium. A swirl deals with the medium at its limit. The speed of light and Planck's constant are the hard stops on what space can do. Nothing smaller exists for a swirl to radiate into. A particle never leaves its wave state: it is a configuration of waves moving through the medium, which is his answer to the ether experiments. Lensing by gravity is one route to the elasticity of space and other routes are open.

His guitar string picture, which this session tests directly: pluck a string at two points in opposite directions and it vibrates faster. Move the plucks toward each other and the pitch climbs. At the limit of the string's elasticity the wave exists only where it was plucked and does not travel. One such pluck makes no sound. Many together make a sound that reaches further.

### Two ways to trap a wave

- **Floor.** The medium cannot carry anything below a lowest note. A swirl that pulses slower than that is stuck. Sessions 1 to 4 ran this.
- **Ceiling.** The medium has a smallest scale, so it cannot carry anything above a highest note. Stiffening pushes a tight vibration above that note and it is stuck. This is the guitar string, and it is what Aaron means.

They differ in a way that can be measured: under a floor a tighter swirl pulses slower, above a ceiling a tighter swirl pulses faster.

### Shared terms

- **Single bump.** All the motion points one direction. One hump.
- **S-bend.** Two halves moving opposite ways with a still point between them.
- **Pinch.** The motion points inward and outward from a center. Exists only in three dimensions.
- **Twist.** An S-bend with a sideways motion added, out of step with the up-and-down motion. Not in this session.

### Readings made by Claude Desktop that Aaron can overrule

- "Free at scale" is taken to mean the ceiling medium has no floor. Long, gentle waves travel in it freely.
- The smallest scale is wanted smooth, with no fixed grid points, so that a moving swirl has no bumps to feel. The trap is established in the literature only for a grid. Whether it survives without a grid is the open question this session tests.

## 2. The session has two tracks

**Track C, the ceiling. Primary.** A one-dimensional string, as in Aaron's picture. If time runs short, Track C is finished first.

**Track F, the floor. Comparison.** The radial three-dimensional work from the earlier draft, trimmed. It keeps the pinch in the session and tests the floor trap with the asymptote limit.

## 3. Track C: the string

The string lies along x. Its displacement u(x, t) is sideways, in one direction.

Units: long-wave speed = 1, smallest scale = 1, limit of stretch = 1. No other constants.

Write d for the stretch: the difference in u between neighbouring points on a grid, or the slope du/dx in a continuous string. The restoring force is F(d).

Stiffening laws:

- **G1, the asymptote.** F(d) = d / (1 - d^2). Energy -(1/2) ln(1 - d^2). The stiffness climbs toward infinity as d goes to 1.
- **G2, the classic.** F(d) = d + beta d^3. Used only to reproduce published results, at the published value of beta.

Two media:

- **C-grid.** Points one unit apart, joined by springs with force F. Linear waves obey omega = 2 sin(k/2), so the ceiling is 2. This is the established case.
- **C-smooth.** A continuous string with the smallest scale built into its inertia and no fixed points: u_tt - u_xxtt = d/dx [ F(u_x) ]. Linear waves obey omega^2 = k^2 / (1 + k^2), so the ceiling is 1, approached as waves get shorter. `scholar` may propose a better smooth candidate. This one is tested at minimum.

Objects on the string:

- **Single bump:** centered on a point. On the grid the pattern is (..., 0, -1/2, 1, -1/2, 0, ...).
- **S-bend:** centered between two points. On the grid the pattern is (..., 0, -1, 1, 0, ...).

Controls: every run has a linear twin (F(d) = d). In C-smooth short waves travel slowly even with no stiffening. That is a linear effect, not a trap, and the twin separates the two.

### Questions for Track C

1. **Existence.** Does each medium hold a trapped vibration with rate above the ceiling, for each object.
2. **The window.** The allowed range of rate and length, from the ceiling up to the limit. Across it: rate, width, energy, energy per cycle. How narrow is it.
3. **Lifetime** of the single bump against the S-bend.
4. **Leak.** The radiation spectrum. Is anything emitted into long waves.
5. **Motion.** Push a trapped vibration at several speeds. Does it travel intact. Energy lost per unit distance. The speed at which it breaks into travelling waves.
6. **Surroundings.** Time-averaged displacement and stretch of the string against distance from the vibration. This is the baseline for the "sound" question in Session 6.
7. **Grid or no grid.** Does the trap survive in C-smooth. Every C-smooth result is repeated at numerical spacings of 1/4, 1/8 and 1/16 of the smallest scale. A trapped state whose width follows the numerical spacing is an artifact and is reported as one.

## 4. Track F: the floor, radial

The field is **u**(x, t), three components. Units: wave speed = 1, gap = 1, limit = 1.

Write S = sum over i of |d_i **u**|^2 and X = |d_t **u**|^2 - S.

- **O. The old equation, as the control.**
  L_O = (1/2)|d_t **u**|^2 - (1/2) S - (1/2)|**u**|^2 + (g4/4)|**u**|^4 - (g6/6)|**u**|^6, at the verified Session 1 setting (beta = 4/3).
- **R. Floor plus a limit on the rate of motion and stretch (Born-Infeld type).**
  L_R = 1 - sqrt(1 - X) - (1/2)|**u**|^2, valid for X < 1. Use the full square root. Do not use a truncated series. The X^2 truncation is the term retired on 2026-10-01.

Objects, both round, one radial profile f(r, t):

- **Single bump:** **u** = f times a fixed direction. S = (d_r f)^2. At the center d_r f = 0.
- **Pinch:** **u** = f times the outward radial direction. S = (d_r f)^2 + 2 f^2 / r^2. At the center f = 0.

For the single bump, equation O reduces exactly to the Session 1 equation. Regression target: omega = 0.909 at beta = 4/3.

### Questions for Track F

1. Does equation R trap a swirl below the floor.
2. For each trapped family: rate, size, energy, energy per cycle across the allowed amplitudes. Where the energy minimum is.
3. Lifetime, using the Session 2 definition unchanged.
4. Bump against pinch: which has the lower energy at the same rate.
5. The pinch in the old equation: does it exist and how long does it live.

## 5. Predictions on record

Copy these into the predictions file before any run. Tolerances are targets.

| # | Prediction | Basis |
|---|---|---|
| C1 | In C-grid both the single bump and the S-bend exist as trapped vibrations with rate above the ceiling. | Sievers and Takeno 1988; Flach and Willis, "Discrete breathers", arXiv:patt-sol/9704004. |
| C2 | In C-grid with G2 the S-bend is stable and runs unchanged for more than 32,000 cycles. The single bump is unstable, and the instability makes it move, not die. | Sandusky, Page, Schmidt, Phys. Rev. B 46, 6161 (1992). |
| C3 | The same ordering holds with the asymptote law G1. | Extension of C2. No source. |
| C4 | A trapped vibration in C-grid emits nothing at its own rate or its overtones, since all of them sit above the ceiling. Energy inside a fixed length stays constant to solver precision. | Flach and Willis; `scholar` to confirm. |
| C5 | In C-grid a trapped vibration can be moved with small loss at low speed, and the loss grows with speed. | Sandusky, Page, Schmidt discuss a smoothly travelling version. `scholar` to confirm. |
| C6 | C-smooth: no prediction on whether a grid-independent trap exists. Published existence results rely on a grid. | Open question. |
| F1 | Equation R traps the single bump below the floor. | Amin, "K-oscillons", arXiv:1303.1102. `scholar` to confirm the sign condition for L_R. |
| F2 | In three dimensions the energy along a trapped family has a minimum at a finite amplitude. | Small-amplitude theory. `scholar` to source. |
| F3 | The pinch exists in O and in R with higher energy than the single bump at the same rate. No prediction on its lifetime. | Zhang, Jain, Amin, arXiv:2111.08700. Their model is not ours. |
| F4 | For a long-lived swirl, energy per cycle stays nearly constant while energy and rate drift slowly. | New measurement. Do not cite the debunked 50,000 times claim in the findings ledger. |

Aaron's own expectations, recorded so the data can speak to them:

- A swirl has an optimal length and rate and the range is narrow. Too long, or moving too fast, and it becomes a travelling wave.
- The S-bend traps energy better than the single bump.
- The single bump is probably not stable enough to build a structure. He is open to being wrong.
- Swirls can merge, but they can also hold together through their geometry. Such a structure has a field that reaches further than a lone swirl's. Not tested here. Session 6.
- The pinch is the object he started from and he wants to see how it behaves.

## 6. Work packages

Order: WP0, then WP-A, WP-B1 and WP-B2 in parallel, then WP-C and WP-D in parallel, then WP-E, then WP-F.

### WP0. Housekeeping (`runner`)

- Add a short note to `docs/` that records the tables and terms in section 1.
- Move the Session 4 leftover (the 2 times box rerun of the far-tail force, old equation) to the parking lot. It is not run this session.
- Record that local commit 9bd140c says "1.5x box" in its message and the correct figure is 1.33 times. Do not rewrite history.
- Create `docs/SESSION_05_STATUS.md` and the predictions file.

### WP-A. Literature brief (`scholar`)

For Track C:

1. Intrinsic localized modes, also called discrete breathers. Existence conditions, the role of the grid, the single bump and S-bend patterns and their stability, how they move, and what changes in two and three dimensions. Start from Sievers and Takeno 1988; Sandusky, Page, Schmidt 1992; Flach and Willis (arXiv:patt-sol/9704004); Flach and Gorbach, Phys. Rep. 467 (2008); MacKay and Aubry 1994.
2. The grid question. Is any continuous medium with a highest note known to hold a trapped vibration above it. Candidates: inertia with a built-in length (the C-smooth form), nonlocal elasticity, strain-gradient elasticity. Also collect the non-existence results for continuous media.
3. Springs with a limit of stretch like G1.

For Track F:

4. Born-Infeld type scalar with a floor: is it well posed, and the existence condition in arXiv:1303.1102.
5. The October lesson on the X^2 term in `LESSONS_2026-10-01.md`. Does it apply to the full square-root form, yes or no, with the argument.
6. Vector oscillons, arXiv:2111.08700 and the hedgehog references it cites. Map to our terms: directional = single bump, hedgehog = pinch, spinning = a candidate for the twist.

Output: additions to `docs/PRIOR_ART.md`, and each prediction in section 5 annotated with what the literature expects and a source. Findings on C-smooth and on equation R gate their production runs.

### WP-B1. String solver (`numerics`, own worktree)

One solver for C-grid and C-smooth, either stiffening law.

Checks that define done, each one a test:

1. **Linear waves.** Measured dispersion matches omega = 2 sin(k/2) on the grid and omega^2 = k^2 / (1 + k^2) in the smooth medium.
2. **Energy.** The energy function includes every term in the equation of motion and is conserved on a closed string with no absorber.
3. **Regression.** C-grid with G2 at the published beta reproduces prediction C2: the S-bend runs unchanged past 32,000 cycles, the single bump goes unstable and moves.
4. **Ends.** Absorbing ends validated against a string twice as long.
5. **Numerical spacing.** C-smooth results hold at spacings of 1/4, 1/8 and 1/16 of the smallest scale.
6. **Guard.** With G1 every step checks |d| < 1. If it fails the run stops with a diagnosis. That is a finding, not a bug to tune away.

### WP-B2. Radial solver (`numerics`, second worktree)

Extend the existing radial solver to equations O and R and to both objects. `numerics` derives the four equations of motion and energies from the Lagrangian.

Checks, each one a test:

1. **Regression.** Equation O, single bump, beta = 4/3 reproduces omega = 0.909 and the Session 1 energy.
2. **Linear twin.** At small amplitude omega^2 = k^2 + 1. For the pinch the linear solution is the l = 1 radial mode.
3. **Energy** conserved in a closed box.
4. **Resolution.** Results hold on a grid 1.5 times finer.
5. **Boundary.** Absorbing boundary validated against a box twice the size. Twice, not 1.33 times.
6. **Guard for R.** Every step checks X < 1 and the well-posedness condition. A failure stops the run with a diagnosis.

### WP-C. Track C runs (`runner`)

Answer the seven questions in section 3, for both media, both objects, with G1. For each: the family across the window, lifetime runs at five or more points, the radiation spectrum, the push test at five or more speeds, and the time-averaged surroundings. Each nonlinear run has its linear twin.

### WP-D. Track F runs (`runner`)

Answer the five questions in section 4: families for the four combinations, lifetime at five or more points per trapped family, and the pinch in the old equation set against the single bump baseline from Sessions 1 and 2.

### WP-E. Review (`reviewer`)

- Before production runs: independent derivation of the string equations and energies, and of the four radial equations and energies.
- After WP-C and WP-D: recompute the headline numbers from the saved time series.
- Final review of the report text against the data. Session 4 needed 20 text corrections. Every verdict quotes its measured number next to the rule it was judged by.

### WP-F. Report (orchestrator)

`docs/SESSION_05_REPORT.md`, with:

- The seven Track C questions and the five Track F questions, each answered with measured numbers next to its prediction.
- A direct statement, with numbers, of whether a tighter vibration is faster (ceiling) or slower (floor) in each medium.
- The window for each trapped family, as tables and plots.
- A lifetime table for every object and medium.
- Aaron's expectations in section 5, each marked supported, contradicted, or not tested, with the number that decides it.
- Hits, split results and misses at the same depth. New entries in `docs/FAILURES.md`.
- What the limits in section 7 mean for each result.
- Proposals for Session 6, and a parking lot.

## 7. Limits of this session

- **Track C is one-dimensional, with one direction of sideways motion.** No twist, no pinch, no structures.
- **Track F is radial.** A radial run cannot see a round object losing its roundness. Radial lifetimes are upper bounds.
- **The grid.** A trap found only in C-grid is a trap that depends on a grid. Whether nature allows a grid is not settled here.
- **Units only.** No physical values are attached.
- **Local compute.** No cloud.

## 8. Stop conditions

- The hard stops in `CLAUDE.md`.
- If the WP-B1 regression fails after two fix attempts: stop Track C runs and report. Track F continues.
- If the WP-B2 regression fails after two fix attempts: stop Track F runs and report. Track C continues.
- If WP-A finds L_R is not well posed for this use, or that the October lesson applies to the full form: stop the R runs and report why.
- If a guard trips (|d| reaches 1, or X reaches 1): log it as a failure entry with a diagnosis. Do not change the equation to avoid it.
- Do not start three-dimensional vector work, the twist, structures, or interactions between objects.
- Do not push.

## 9. Parking lot carried in

- Session 4: the 2 times box rerun of the far-tail force at fine separation spacing (old equation).
- Session 6 candidates: the twist, as a second direction of sideways motion on the same string, with the timing varied; two and many trapped vibrations on one string and the reach of their combined field; the ceiling medium in three dimensions, including the pinch; a ceiling medium with a floor as well.
- Later: energy per cycle of the preferred swirl as a route to Planck's constant; lensing as a route to the stiffness of space.
