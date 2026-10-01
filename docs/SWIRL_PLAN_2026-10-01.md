# Swirl Theory restart: maths plan

Date: 2026-10-01 (revision 3: harness with five workers and pinned effort levels, failure handling, literature checks, rented compute)
Status: agreed with Aaron. Replaces STUDY_PLAN_2026-03-18.md.
Companion files: LESSONS_2026-10-01.md, CLAUDE.md (repo root)

## 1. The question

Swirl Theory rests on one claim: a single swirl is transient, and certain groups of swirls lock together and stay localized. The 2025-26 work did not test this. It ran at an amplitude where the nonlinear terms hold 0.06% of the energy, so the field was nearly free and nothing could lock. This plan tests the claim where the field can trap itself.

## 2. Decisions on record

| Topic | Decision |
|---|---|
| Path | The five steps in section 4, in order |
| Paper v9 | Shelved. Its closed-form energy formula is kept as an engine regression test |
| NEST k-essence | The X² term is retired. The Lorentz-covariant framing is kept |
| Scope of maths | Both: the cluster question and the theory-level results |
| Working folder | `~/Desktop/SWIRL` on Aaron's Mac. A fresh repo, with the legacy repo frozen under `legacy/` |
| GitHub | Nothing is pushed without Aaron's say |
| Workflow | Claude Desktop plans and interprets. Claude Code (desktop app) runs everything |
| Harness | One orchestrating session plus five workers, defined in `.claude/agents/` |
| Model routing | Fable orchestrates and judges (high). Opus writes numerics (high). Sonnet 5.5 reviews (xhigh), researches the literature (xhigh), does mechanical work (medium) and runs rented compute (high) |
| Effort levels | Starting points. Session 1 measures them and the report says what to change |
| Style of rules | Guidelines with reasons, plus a short list of hard stops. See CLAUDE.md |
| Failures | A missed target or failed prediction is a result. It gets the same depth of analysis as a hit and an entry in `docs/FAILURES.md` |
| Literature | Every prediction sits next to what published work expects. Nothing is called new without a documented search |
| Compute | Rented AWS compute in us-east-2, on-demand only, within a pre-approved working budget of 50 dollars of credits. The Mac handles anything short |

## 3. Model, notation, observables

Field equation (unchanged from the legacy work):

```
phi_tt - lap(phi) + m^2 phi - (g4/6) phi^3 + (g6/120) phi^5 = 0
V(phi) = (1/2) m^2 phi^2 - (g4/24) phi^4 + (g6/720) phi^6
```

**One real parameter.** Measure lengths and times in units of 1/m and the field in units of m/sqrt(g4). Only `beta = g6 m^2 / g4^2` remains. Energies scale as m/g4 times a function of beta.

**True-vacuum condition.** V >= 0 everywhere only if `beta >= 5/8`. Set B has beta = 0.611 and Set C has beta = 0.400, so in both, phi = 0 is not the lowest state of V. At amplitude 0.5 this never showed. At real oscillon amplitudes it matters.

**Single-frequency profile equation.** With phi = Phi(r) cos(omega t):

```
Phi'' + (2/r) Phi' = (m^2 - omega^2) Phi - (g4/8) Phi^3 + (g6/192) Phi^5
E = integral [ (1/4)(omega^2 + m^2) Phi^2 + (1/4) Phi'^2 - (g4/64) Phi^4 + (g6/2304) Phi^6 ] d3x
N = (omega/2) integral Phi^2 d3x
```

**Observables used in every step.**

| Symbol | Meaning | Why |
|---|---|---|
| omega | Core oscillation frequency, measured from the time series | A self-trapped swirl needs omega < m |
| N | Adiabatic invariant, (1/omega) times the integral of time-averaged phi_t² | Approximately conserved. Plays the role of particle number. Energies are compared at fixed N, never at fixed amplitude |
| E | Total energy, including every term in the equation of motion | Conservation check |
| E_loc(r_c, t) | Energy inside radius r_c of the lump | The stability observable. Binding energy alone says nothing about localization |

**Prior estimate (Claude Desktop, Set B, single-frequency approximation, unverified).** Step 1 confirms or refutes this.

| omega/m | Phi(0) | Half-width | N | E | E/(mN) |
|---|---|---|---|---|---|
| 0.990 | 3.1 | 4.2 | 1,895 | 1,912 | 1.009 |
| 0.964 | 5.6 | 2.5 | 1,187 | 1,217 | 1.026 |
| 0.922 | 7.7 | 2.1 | 1,062 | 1,098 | 1.034 |
| 0.837 | 9.7 | 2.1 | 1,332 | 1,333 | 1.000 |
| 0.707 | 10.5 | 2.8 | 2,796 | 2,443 | 0.874 |

The family has a minimum energy near 1,100 at omega about 0.92. The legacy packet had E = 13.5 and N about 12. At these amplitudes the nonlinear terms are large, so the single-frequency numbers may be off by tens of percent.

## 4. The five steps

### Step 1. Find the real swirl (maths, then a cheap radial check)

**Question.** For which beta and omega does a self-trapped lump exist, how big is it, and what is the smallest one?

**Work.**
- Solve the profile equation across omega for several beta, including Set B and values at or above 5/8.
- Per beta: minimum E and N on the family, the omega where dN/d(omega) changes sign (stability boundary), and the omega where E = mN.
- Tail constants: Phi tends to A exp(-kappa r)/r with kappa = sqrt(m² - omega²). Step 3 needs A and kappa.
- Check the approximation by time-evolving a few profiles in a spherically symmetric solver.

**Prediction on record.** Profiles on the branch with dN/d(omega) < 0 stay localized. Profiles on the other branch do not. The legacy Gaussian disperses.

**Status of the predictions in this plan.** They come from Claude Desktop's recollection of the oscillon literature and have not been checked against sources. The scholar's first brief does that check.

### Parameter selection for Step 2 (a suggestion, not a rule)

The orchestrator picks the production parameters from the Step 1 results. This is the suggested default. It may choose differently when the data argue for it, and it says why in the report.

- **beta:** 0.70. It keeps phi = 0 safely the true vacuum and is the nearest tested value to the legacy potential with margin. At exactly 5/8 the two vacua are degenerate and large lumps start to behave like bubbles.
- **Profile:** on the dN/d(omega) < 0 branch, with E/(mN) between about 0.90 and 0.98. That is bound against dispersal into free waves without being so deep that the lump is wide and expensive.
- **If the radial check contradicts the stability prediction:** pick the profile the radial check shows is longest-lived, and record the contradiction as a finding.
- **Grid:** at least 6 points across the half-width, and at least 6 per wavelength of the third-harmonic radiation, k = sqrt(9 omega² - m²).
- **Box:** the lump plus an absorbing layer, validated against a box twice the size.
- **Time step:** start near dx/8 and adjust until the energy standard is met.

### Step 2. Single-swirl gate (3D PDE)

**Question.** Does one real oscillon survive in full 3D, where non-spherical perturbations are allowed?

**Targets.** These are what a clear answer looks like. A result between them is reported as it is, with the curve, and not rerun to land on a threshold.
- Go: the lump keeps at least 90% of its energy inside 3 half-widths at T = 1000, with measured omega < m.
- Control: the same initial data with g4 = g6 = 0 falls below 10% in the same time. If the control does not fall, the diagnostic is broken.
- No-go: no member of the family survives for any beta >= 5/8. Then the real-scalar sextic model has no long-lived swirls and the plan stops for a rethink.

### Step 3. Two-body law (maths, then PDE pairs)

**Question.** What force do two real swirls exert on each other, as a function of distance and relative phase, at fixed N?

**Work.** Derive the interaction from tail overlap of two Step 1 profiles. Expected form: `U(d, dphi) = -C cos(dphi) exp(-kappa d)/d` with C > 0 fixed by the tail amplitude A.

**Prediction on record.** In-phase pairs attract and merge. Anti-phase pairs repel. This is the reverse of the legacy paper's rule.

**PDE check.** Pairs at several separations and phases. Measure the initial centroid acceleration and any exchange of N between the two lumps. Verdicts on three things: the sign, the decay constant kappa, and the cosine dependence.

### Step 4. The theory's test

**4a. Transient alone, stable together.** Take lumps that each sit below the minimum N from Step 1. Compare one alone against k of them placed close together in phase, with total N above the minimum. Controls: the same k lumps far apart, and the same k lumps with alternating phase. Observable: E_loc after a long run.

**4b. Does any geometry survive as a geometry?** Place real oscillons on polyhedra with phase patterns chosen from the Step 3 law. Measure whether the lump-to-lump distances stay bounded: no merger, no flying apart.

**Expectation on record.** 4a is plausible, but it produces one merged lump, not a cluster. 4b looks unlikely for static polyhedra given Step 3. The outcome is recorded either way.

**Decision point.** If 4b is negative, Aaron and Claude Desktop decide what locking mechanism to test next. Nothing is pre-approved.

### Step 5. Theory-level results

These follow from Lorentz covariance and are cheap once Step 2 has a real swirl.

- **Inertia.** A boosted swirl has E = gamma E0 and momentum gamma E0 v, so its rest energy is its inertial mass.
- **Clocks.** The internal frequency of a moving swirl is observed as omega/gamma.
- **Lengths.** The profile of a moving swirl contracts by gamma along the motion.
- **Minimum swirl.** The Step 1 minimum energy answers the theory document's open question about a nonlinearity threshold.
- **Not covered.** None of this addresses gravity or the aether-density mechanism. That needs its own formulation later.

## 5. Standards for every run

These come from LESSONS_2026-10-01.md.

1. **Regime check first:** report omega relative to m and the fraction of energy in the nonlinear terms before trusting any run.
2. **Stability means localization:** use E_loc, never binding energy alone.
3. **Complete energy:** the energy function includes every term in the equation of motion.
4. **Real absorbing boundaries:** validated against a box twice the size.
5. **Linear twin:** every nonlinear run has a control with g4 = g6 = 0.
6. **Analytic value first:** compute the t = 0 or closed-form answer before the PDE. If the PDE adds nothing, the result is not dynamics.
7. **Full symmetry group:** equivalence classes use rotations, reflections and the field sign flip.
8. **Coordinates, not labels:** compute structural quantities from vertex positions.
9. **Verdict targets before running:** and each robustness test states what failure would look like.
10. **Literature next to every prediction:** with a source, tagged by whether the source was read.
11. **No claim of novelty without a documented search.**
12. **Misses get equal depth:** expected, measured, ranked causes, diagnostics, what it rules in and out.
13. **No paper drafting until the map is complete.**

## 6. Session map

| Session | Content | Ends with |
|---|---|---|
| 1 | Repo setup, Step 1, engine v2, lessons audit, benchmark, parameter pick, Step 2 gate | `docs/SESSION_01_REPORT.md` with a go or no-go |
| 2 | Step 3 derivation and pair runs | Force law verdict |
| 3 | Step 4a and 4b | Verdict on the central claim |
| 4 | Step 5 | Theory-level results |

## Appendix A. Session 1 work packages

Dependencies: WP0 first. Then 1A, 1B, 1C and 1L in parallel. 1D needs 1B. WP2 needs 1A and 1L. WP3 needs 1A, 1B and 1D. Tolerances below are targets (see CLAUDE.md for how to treat them).

### WP0. Setup (runner)

- `git init` on branch `main` and commit the kit as it stands.
- Bring in the legacy repo, https://github.com/TheCaffeineDuck/geometric-binding-study. Tag commit 8a62dd2 as `frozen-v9-2026-04`. Place that commit's tree, unchanged, under `legacy/`. Leave no remote named `origin`.
- Create `.venv` at the repo root with numpy, scipy, numba, matplotlib and pytest.
- Write `docs/ENVIRONMENT.md`: chip, performance-core count, RAM, Python and package versions.

### WP1A. Oscillon family (numerics)

Code in `swirl/family/`, tests in `tests/family/`, results in `outputs/family/`.

- Solve the profile equation of section 3 for the nodeless localized solution, across omega/m from near 1 down to where solutions stop, for beta = 0.611 (Set B), 0.625, 0.70, 0.85 and 1.0. Work in the rescaled one-parameter form. Report in rescaled units and in Set B units.
- Per solution: Phi(0), half-width, N, E, E/(mN), tail constants A and kappa.
- Per beta: minimum E and minimum N, the omega where dN/d(omega) changes sign, the omega where E = mN.
- Radial time-evolution check: a spherically symmetric solver for the full field equation with an absorbing outer boundary. For beta = 0.611 and 0.70, evolve three profiles to T = 500: one on each side of the dN/d(omega) sign change and one near minimum E. Also evolve the legacy Gaussian (phi0 = 0.5, R = 2.5, Set B) as the control. Record the measured core frequency and the fraction of energy inside 3 half-widths over time.

Checks:
1. With g6 = 0 the rescaled peak equals the known 3D cubic ground-state value 4.3374, target 1e-3.
2. Each profile satisfies the scaling identity: the integral of (1/2)Phi'² plus 3 times the integral of G(Phi) is zero, target 1e-6 relative, where G' is the right-hand side of the profile equation.
3. Two parameter sets with equal beta give the same rescaled family, target 1e-6.
4. Set B results compared with the prior-estimate table in section 3. Differences above 2% are findings, not something to tune toward.
5. The radial solver conserves energy with the absorber off, target 1e-5 relative, and the legacy Gaussian control disperses.

### WP1B. Engine v2 (numerics)

Code in `swirl/engine/`, tests in `tests/engine/`. Start from `legacy/engine/evolver.py`; its spectral Laplacian and RK4 are sound. Single-run speed matters, because Step 2 runs at 128³ or larger on this Mac.

- The energy function accounts for every term in the equation of motion.
- Remove the legacy "Kreiss-Oliger" term (LESSONS 7). Any high-k filter that stays must be true damping and off by default.
- A switchable absorbing boundary layer.
- Diagnostics as time series: energy inside a sphere of radius r_c about a point, N, and core frequency at a point.
- Checkpoint and resume for long runs.

Checks:
1. With g4 = g6 = 0 and no absorber, a Gaussian evolved to T = 50 matches the exact solution in which each Fourier mode rotates at sqrt(k² + m²), target 1e-4 relative L2.
2. With Set B and no absorber, total energy drifts less than 1e-5 relative over T = 100, with no step like the legacy 13.4618 to 13.4637.
3. For the cube phase arrays in Appendix B of `legacy/paper/paper_draft_v9.md`, t = 0 E_bind is within 0.05 of the paper's Table 2 and within the larger of 2% or 0.1 of the closed form in LESSONS 2.
4. A free Gaussian in the L = 50 box with the absorber on matches an L = 100 run without absorber over r < 10 up to T = 75, target 5% relative L2 at every sampled time. If L = 50 cannot reach it, report the smallest box that does.

### WP1C. Lessons audit (runner)

Read-only on `legacy/`. Writes `docs/LESSONS_AUDIT.md` and `outputs/audit/`.

- Recompute every number in the LESSONS items marked [audit], from `legacy/outputs` and the legacy code.
- One table: item, number in LESSONS, recomputed number, agree or disagree. Do not edit LESSONS. A disagreement is a finding.

### WP1L. Prior-art brief (scholar)

Writes `docs/PRIOR_ART.md`.

- What published work says about oscillons in potentials of this form: existence, the frequency and amplitude range, lifetimes, the stability criterion, the minimum energy or size, and the role of the sextic term.
- What published work says about oscillon interactions: dependence on relative phase and distance, merger and repulsion, and any reported multi-oscillon bound states.
- A check of each prediction on record in this plan (Steps 1 to 4) against those sources: agrees, disagrees, or no source found.
- Known pitfalls in simulating oscillons: boundary conditions, initial profiles, resolution, lifetime measurement.

The orchestrator reads the brief before the parameter pick. If the brief contradicts a prediction or the parameter suggestion, the orchestrator says how it resolved that in the report.

### WP1D. Benchmark (runner)

- Time engine v2 per RK4 step at 64³, 128³, 192³ and 256³ on this Mac: one run using all performance cores, and several single-core runs side by side.
- Report steps per second, memory per run, and projected wall time for T = 1000 at each size. Write `outputs/benchmark/benchmark.json` and a table for the report.
- If `docs/AWS_REPORT.md` exists, add the projected cost and wall time of the same runs on the largest instance the quota allows.

### WP2. Parameter pick (orchestrator)

Apply the suggestion in section 4 to the WP1A and WP1D results. Record the chosen beta, omega, grid, box and time step, with the reasons, in the report.

### WP3. Step 2 gate (runner or cloud runs, orchestrator judges)

Runs, in priority order:
1. The chosen oscillon in 3D, absorber on, T = 1000.
2. Its linear twin (g4 = g6 = 0, same initial data).
3. The same oscillon with a small non-spherical distortion of the profile, which the radial solver cannot test.
4. A resolution check: run 1 repeated on a grid 1.5 times finer for a shorter time, comparing E_loc and omega.

Where to run: the orchestrator decides from the benchmark. If `docs/AWS_REPORT.md` shows enough core quota, rent within the working budget through the `cloud` worker. If not, run locally in the background with checkpoints and a log, with the Mac kept awake. If local wall time for all four would pass about 12 hours and no rented compute is available, do runs 1 and 2 and list the others as pending.

Any target that is missed here is handled as a result: analysis, up to three cheap diagnostics, and a `docs/FAILURES.md` entry from the scholar.

### WP4. Review (reviewer)

- After 1A and 1B merge: review each diff against its work package. Confirm every listed check exists as a test and tests what the package says.
- After WP3: recompute the retained-energy fraction and omega independently from the saved time series and compare with the orchestrator's verdict.
- Before the report is final: check that every miss has its own analysis at the same depth as the hits.

### WP5. Report (orchestrator)

`docs/SESSION_01_REPORT.md`, about three pages:
- Environment and benchmark table.
- Family minima per beta in both unit systems, and the radial-evolution results against the prediction.
- Each engine check with its measured number.
- Audit disagreements.
- What the prior-art brief found, and where it agrees or disagrees with the plan's predictions.
- Chosen parameters and why.
- Step 2 result against the targets, with the E_loc curve.
- Hits and misses at the same depth. For each miss: expected, measured, ranked causes, diagnostics run, what it rules in and out.
- Worker table: for each worker, the model and effort it actually ran at, whether its checks passed first time, and what the reviewer caught. End with a recommendation on any effort level that should change.
- Rented compute used, with the ledger total.
- Reviewer findings, deviations from guidelines, open issues, and a parking lot of ideas that were not pursued.
