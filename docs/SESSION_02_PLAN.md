# Session 2 plan

Date: 2026-10-02
Status: agreed with Aaron. This file supersedes section 1 and Step 4 of `SWIRL_PLAN_2026-10-01.md`. Everything else in that plan, and all of `CLAUDE.md`, still applies.
Inputs: `SESSION_01_REPORT.md`, `PRIOR_ART.md`, `FAILURES.md`, and the theory-review handoff of 2026-10-01.

## 1. The central claim, corrected

The earlier plan tested "a single swirl is transient". That wording came from the Summary of the theory file and is not Aaron's position. The claim is:

1. **A lone swirl is metastable.** Left alone it persists. A modest disturbance can dissipate it or let something absorb it.
2. **A bonded group is robust.** Once bonded, it withstands dissipation and absorption.
3. **Breaking is selective.** Only something specific can break a bonded group.

**A bonded group means distinct swirls holding an arrangement.** Swirls that merge into one bigger lump are not a bonded group.

Where the model stands against this after Session 1: a lone lump at beta = 4/3 persists for about 1,500 oscillations and then dies even when undisturbed. That falls short of point 1, and the lifetime map below measures by how much across beta.

## 2. Decisions on record

| Topic | Decision |
|---|---|
| Model | Stay with the single real field |
| Step 3 | Run at beta = 4/3, omega = 0.909, the verified 3D point |
| Step 4 | A break-threshold test: a lone lump and a bonded group are hit with the same disturbances. A lifetime comparison, lone against group, is kept as a second test. Both are designed after this session and are not run in it |
| Geometric binding | Remains the project's distinctive question. No published work on it was found. This session searches for a bonded arrangement |
| Twist | Stepped-phase patterns (the oscillation timing stepping around a ring or shell) are in scope. The multi-component topological fork stays dropped. A single spinning lump gets a literature check only |
| Orchestrator | Fable |
| Rented compute | Same rules. 47.20 dollars of the 50 remain |

## 3. Work packages

Order matters, because the session may be interrupted. After each package finishes, the orchestrator updates `docs/SESSION_02_STATUS.md` with what is done, what is running and where the outputs are, so a new session can resume from it.

Dependencies: WP0 first. Then S, A and B in parallel with the start of C. D needs the pair law from C. E needs C and D.

### WP0. Housekeeping (runner)

- Add an Errata section at the end of `LESSONS_2026-10-01.md`: item 1's nonlinear share is 0.178%, not 0.06% (an arithmetic error by Claude Desktop; the conclusion is unchanged), and the audit's findings on items 2 and 3 stand as recorded in `LESSONS_AUDIT.md`.
- Add one line at the top of `SWIRL_PLAN_2026-10-01.md` pointing to this file.
- Add to `.claude/agents/numerics.md`: "Quote the worst value over the whole record. List every number that misses its target."
- Re-check the AWS core quota and update `AWS_REPORT.md`.
- Confirm from the task list what model and effort each worker runs at, including what the numeric effort values "5" and "30" correspond to.

### WP-S. Prior-art brief for this session (scholar)

Adds to `docs/PRIOR_ART.md`.

- Bound arrangements of several solitons or oscillons, in any field: optical soliton clusters held by phase patterns, soliton molecules, Q-ball molecules, oscillon collisions and bound states, and the open lead from search 16.
- Angular momentum carried by real-scalar oscillons: spinning lumps, and stepped-phase patterns.
- Oscillon lifetime against potential shape: the sextic family above beta = 4/3, and other shapes with reported long lifetimes.
- A novelty search for geometric binding of oscillons in 3D.
- Two claims Claude Desktop made from memory, to confirm or refute: that optics has rotating soliton clusters held by stepped phases, and that some potential shapes give far longer oscillon lifetimes.

The orchestrator reads the brief before WP-D.

### WP-A. Lifetime map across beta (numerics builds, runner runs; radial solver)

**Question.** How does the lone-lump lifetime depend on beta, and is there a beta where a lone lump is effectively permanent?

- beta = 0.75, 0.85, 1.0, 1.15, 4/3, 1.6, 2.0, 3.0, 5.0. For each, scan omega along the stable branch finely enough to find the radiation-rate dips, since Session 1 showed lifetime is sensitive to omega.
- First build and test the path-integral lifetime estimate from the Session 1 parking lot (integrate dE/(E Gamma(E)) along the family). Validate it against the measured death at beta = 0.85 (T = 1146) and the beta = 4/3 crossing (T = 10,630) before using it to choose which profiles to run long.
- Run the best profile at each beta to death or T = 30,000, whichever comes first.
- Output: lifetime against beta, with the best omega at each.
- Any beta that beats 4/3 by a wide margin gets one full 3D confirmation run.

**Guess on record (Claude Desktop, no source).** Lifetime keeps rising for some way above 4/3, then falls as the stable branch narrows.

### WP-B. 3D lifetime extension (runner)

- Extend the Session 1 primary run to T = 5000 in 3D, in the background on the Mac.
- Target: it tracks the radial curve to 1e-3. A departure is a finding, because radial runs cannot see non-spherical instabilities.

### WP-C. Step 3: the two-body law (numerics builds, runner or cloud runs)

**Question.** What do two real swirls do to each other, as a function of distance and relative phase? And is there any pair that stays bonded?

Build first:
- Per-lump diagnostics, phase-averaged: centroid, energy and N inside a ball around each lump, and the relative phase between lumps.
- Angular momentum of the field about an axis.
- Pair initial data: two Session 1 profiles at separation d with phases theta1 and theta2.
- Control: a single lump started at each phase used behaves like the phase-0 lump. If it does not, that is a finding and the pair runs wait.

Derive, independently of the brief:
- The tail-overlap interaction. The brief gives U = -2 pi A^2 cos(dphi) exp(-kappa d)/d. Check it, state what is held fixed, and turn it into a predicted centroid acceleration using the lump's energy as its inertia.

Run:
- Relative phases 0, 45, 90, 135 and 180 degrees, at three separations. Choose separations where the predicted centroid shift over the run is measurable and the lumps start with little overlap.
- Write the predictions to a file before any pair run.

Measure against prediction:
- Sign of the force, its decay constant kappa, and its cosine dependence.
- Exchange of N between the lumps and drift of the relative phase.
- Outcome of each pair by the end of the run: merged, separated, or still at bounded distance.

**Predictions on record.** In phase attracts and merges, anti-phase repels (every primary source agrees). At 90 degrees the leading force vanishes and N flows from one lump to the other (Claude Desktop's expectation, no source).

### WP-D. Arrangement search: the maths (numerics)

**Question.** Using the pair law as measured in WP-C, which arrangements and phase patterns could hold?

- Treat each lump as a point with the measured pair interaction.
- Arrangements: rings of 3 to 8 lumps; tetrahedron, octahedron, cube and icosahedron, each with and without a central lump.
- Phase patterns: every 0 and 180 degree pattern distinct under rotations, reflections and the global sign flip; and stepped phases around a symmetry axis.
- For each: net force on every lump against size, any size where the forces balance, and whether that balance restores under small displacements of the lumps.
- Flag any balance where neighbouring lumps sit closer than about 2.5 half-widths. The point model is not valid there, and only a 3D run can decide.
- Note which patterns carry angular momentum.
- Output: a ranked candidate list with predicted spacing and stability.

**Rough numbers from Claude Desktop, to confirm or refute.** A five-lump ring with 72 degree steps balances near kappa R of about 1.07 and is restoring in the radial direction, but that spacing is inside the overlap region. An octahedral shell around an opposite-phase centre has a balance radius that is not restoring.

### WP-E. Arrangement search: 3D tests (runner or cloud runs, orchestrator judges)

- Test the best candidates from WP-D in full 3D. The orchestrator chooses how many from the budget and the benchmark, and includes at least one stepped-phase pattern if any made the list.
- Controls for each: the same lumps placed far apart, and the same geometry with a plain 0 and 180 degree pattern.
- Measure: lump-to-lump distances over time, N of each lump, energy retained by the whole group, and angular momentum.
- Outcome for each: merged, flew apart, or bonded. Bonded means the lumps stay distinct and the distances stay bounded to the end of the run. Write the numeric criteria in the prediction file before running.
- A bonded outcome gets a second run at finer resolution and a longer time before it is reported as a result.

### WP-F. Review and report (reviewer, scholar, orchestrator)

- Reviewer: each numerics diff, the pair-law verdict recomputed from saved data, and every bonded or near-bonded outcome recomputed from saved data.
- Scholar: failure ledger entries for every miss, and the novelty check for anything the report would call new.
- `docs/SESSION_02_REPORT.md`, in the Session 1 format: results against predictions, hits and misses at equal depth, the worker table, rented compute, deviations, open issues and parking lot.
- A closing section, "Proposals for Step 4": which disturbances to use, a definition of "broken", which beta to use for the break-threshold test and for the lifetime comparison, and what the lone-lump size control should be. These are proposals for Aaron and Claude Desktop, not decisions.

## 4. Stop conditions

- Do not start Step 4.
- The hard stops in `CLAUDE.md`.
- If WP-C's single-lump phase control fails, or the measured pair law contradicts the derivation in sign, stop after WP-C, finish WP-A and WP-B, and report. WP-D and WP-E depend on the pair law being understood.
