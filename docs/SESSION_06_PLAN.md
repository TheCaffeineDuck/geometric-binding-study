# Session 6 plan

Date: 2026-11-11
Status: agreed with Aaron.
The model for this session comes from `docs/SWIRL_RULES_2026-11-11.md`. That sheet replaces every earlier statement of the model, including the grid lattice proposed after Session 5. Everything else in `CLAUDE.md` still applies.
Inputs: `SWIRL_RULES_2026-11-11.md`, `SESSION_05_REPORT.md`, `PRIOR_ART.md`, `FAILURES.md`, `LESSONS_2026-10-01.md`.

## 1. What this session is for

Sessions 1 to 5 tested traps that are not the one in Aaron's rules. In his picture a swirl is held by a wall it makes itself: its pull stretches the surrounding medium to the elastic limit, the medium there goes rigid, and the rigid shell keeps the wave in. The same pull, spread over a growing surface, is the gravity well. The medium is continuous. It has no grid.

This session builds that medium for the first time and tests it with the one object that can be solved cheaply in true three dimensions: the pinch. A pinch is the same in every direction from its center, so the solver works along one line (distance from the center) while the medium is fully three-dimensional.

The S-bend, bonding, and clocks made of structures need a full three-dimensional solver. That is Session 7.

## 2. The medium

Work in material coordinates. A point of the medium that rests at distance R from the center sits at distance r(R, t) when deformed.

- Radial stretch: s_r = dr/dR.
- Sideways stretch: s_t = r/R, in each of the two directions across the radius.
- Volume ratio: J = s_r s_t^2. Density is the rest density divided by J. J below 1 is denser space.

The stored energy W is a function of the stretches. Any candidate must meet these requirements, each tied to a rule:

| # | Requirement | Rule |
|---|---|---|
| M1 | W depends only on true lengths (the stretches). It is the same in every direction. | 1, 4, 11 |
| M2 | At small strain the medium is linear and waves travel at one speed. | 5, 6 |
| M3 | W has an asymptote at a maximum squeeze and at a maximum pull. The two limits are equal in log stretch: ln(s) lies between -L and +L. | 2, 14 (pinch) |
| M4 | The stiffness for further deformation climbs toward infinity at either limit. | 2, 3 |
| M5 | The wave speed is fixed per unit of medium. Solve in material coordinates and convert to distances afterward. | 6 |

Starting candidate, to be checked and improved by `numerics` and `scholar`:

- **C1, log-strain form.** With e_r = ln(s_r) and e_t = ln(s_t), W = mu [ g(e_r) + 2 g(e_t) ] + (Lambda/2) G(e_r + 2 e_t), where g(e) = -L^2 ln(1 - e^2/L^2), and G has the same form with its own limit.
- **C2.** A second form meeting M1 to M4, chosen by `numerics`. A Gent-type stretch limit with a matching squeeze barrier is one option.

Two facts Aaron should know about, recorded here:

- **The limit L is a pure number.** In a medium described by true lengths, the maximum stretch is a ratio and cannot be turned into a unit. It is not inserted. It is scanned: three values, and every result is reported as a function of L.
- **One wave speed fixes the ratio of the two stiffness constants.** In a classical elastic solid, equal speeds for squeeze waves and sideways waves requires Lambda = -mu. Waves then travel properly, but the medium has an unusual response to uniform squeeze. `reviewer` checks that each candidate stays well posed over the range the runs reach. A conventional ratio (Lambda = mu) is run alongside as a comparison.

Units: wave speed = 1 and stiffness mu = 1. The medium has no built-in length. The size of the starting object is the unit of length, and results are reported as scaling laws.

A verdict has to hold across the candidates and the values of L tested, or the report says it depends on them.

## 3. What is measured

### Part A. The structure around a contracted core

A stand-in for the steady pull of trapped energy: a central region is held contracted by a fixed amount, and the medium around it is allowed to settle.

- A1. Squeeze and pull as a function of distance. Is squeeze at the core matched by pull around it (rule 4).
- A2. How far the medium is drawn inward, against distance. Does it fall as the inverse square (rule 18).
- A3. Density against distance. Where is space denser, and by how much (rule 17).
- A4. The gap. As the contraction is increased, does a shell reach the pull limit. Where, and how thick (rule 12, black hole).
- A5. Whether the response is proportional to the size of the contraction.

### Part B. The gap as a wall

- B1. Send a small wave at a shell held near the pull limit. Measure how much reflects and how much passes, against how close the shell is to the limit (rule 3).

### Part C. Crossing time

- C1. Time for a small wave to cross a span through the contracted region, against the same span far away, along a radial path (rule 20, black hole).
- C2. From the measured stretches, compute the crossing time for a path that passes beside the core. This is a derived number, since a sideways path needs the Session 7 solver.

### Part D. The pinch itself

Start a pinch: a central region set breathing in and out.

- D1. Energy inside a fixed material radius over time. Lifetime in cycles. Does it stay (rule 13).
- D2. Does a shell near the pull limit form around it, and at which moments of the cycle (rule 12, rule 14).
- D3. The time-averaged position of the surrounding medium. Is it left drawn inward. Is that draw proportional to the pinch's energy (rule 17).
- D4. Rate, energy, and energy per cycle against size, and against L.
- D5. Each run has a linear twin: the same start in the medium with no limits.

## 4. Predictions on record

Copy these into the predictions file before any run. Tolerances are targets.

| # | Prediction | Basis |
|---|---|---|
| P1 | Around a weakly contracted core the inward draw falls as the inverse square of distance. | Classical elasticity, center of contraction. `scholar` to source. |
| P2 | Outside the core the medium is pulled along the radius and squeezed across it, and to first order its density is unchanged. Denser space is found inside the core only. | Same. This would not match rule 17 outside the core. |
| P3 | With a strong enough contraction a shell just outside the core reaches the pull limit first. | Follows from P1: the pull is largest at the core's edge. |
| P4 | A shell near the limit reflects a wave almost completely, and more so the closer it is to the limit. | Large stiffness mismatch reflects waves. |
| P5 | A radial wave is slowed inside the core but not in the pulled region outside it, where the medium is stretched along its path. A path passing beside the core is slowed. | Follows from P2 and rule 6. Part of this would not match "approaching the gap takes a very long time." |
| P6 | The pinch outlasts its linear twin but is not permanent. Its wall vanishes twice per cycle at the neutral instant, and it leaks then. | Reasoning from rule 14. No source. |
| P7 | The medium has no built-in length, so energy grows as size cubed, rate falls as one over size, and energy per cycle grows as size to the fourth power. | Scaling. |
| P8 | A vibrating pinch leaves a steady draw on the medium proportional to its energy. No prediction on its direction. | Time-averaged stress of a confined wave. `scholar` to source. |

Aaron's expectations, recorded so the data can speak to them:

- Trapped energy contracts the medium and leaves it denser, and that is the gravity well (rule 17).
- The pull forms a rigid gap that holds the wave (rule 12).
- Approaching the gap, space is so dense that movement takes a very long time.
- The S-bend is a more stable trap than the other objects. The pinch is the object he started from and he wants to see how it behaves.

Where P2, P5 and P6 differ from these, the run decides. A difference is a result about what the medium needs, not a reason to adjust the test.

## 5. Work packages

Order: WP0, then WP-A and WP-B in parallel, then WP-C and WP-D in parallel, then WP-E, then WP-F.

### WP0. Housekeeping (`runner`, `reviewer`)

- `reviewer` re-reviews the corrected text of `SESSION_05_REPORT.md` independently. Session 5 applied 44 corrections that were not checked.
- Write the hold rule for this session before any run, fixing the two defects Session 5 reported in its classifier. It must be stated in material coordinates.
- Any analysis definition is fixed before its data is seen. A change after the fact is disclosed in the report.
- Confirm at first delegation that each worker's model and effort match `CLAUDE.md`. Session 5 found mismatches.
- Create `docs/SESSION_06_STATUS.md` and the predictions file.

### WP-A. Literature and novelty search (`scholar`)

A documented search. Nothing in this session is called new without it.

1. Elastic media with a limit of stretch. Start from Gent (1996) and from Rajagopal's strain-limiting models. What is known about waves, shocks and trapped vibrations in them.
2. The classical solution for a center of contraction and a contracted inclusion (Lamé, Eshelby). Source P1 and P2.
3. Elastic models of the ether, including the one-speed condition (MacCullagh, Kelvin). What went wrong with them.
4. Gravity as a change in the density or wave speed of a medium (start from Dicke 1957, the polarizable vacuum line, and Kleinert's world crystal). State exactly where rules 11, 12 and 17 depart from each.
5. The time-averaged stress of a confined wave. Source P8.
6. Whether any published work has a wave that walls itself in by driving its surroundings to an elastic limit.

Output: additions to `docs/PRIOR_ART.md`, each prediction annotated with what the literature expects and a source, and a one-page novelty statement: what is known, what is not found, and how hard the search was.

### WP-B. Radial solver for the medium (`numerics`, own worktree)

A solver for r(R, t) in material coordinates, for any candidate W.

Checks that define done, each one a test:

1. **Small waves.** At small strain a wave travels at speed 1.
2. **Statics.** A weak contraction reproduces the classical solution from WP-A item 2.
3. **Energy.** The energy function includes every term and is conserved in a closed box.
4. **Resolution.** Results hold on a grid 1.5 times finer.
5. **Outer boundary.** Static fields are checked at box sizes of 1, 2 and 4 times, since a slowly fading field feels the edge. Dynamic runs use an absorbing boundary validated against a box twice the size.
6. **Guards.** Every step checks that the stretches stay inside the limits and that the candidate is well posed there. If a guard trips, or a shock forms, the run stops with a diagnosis. That is a finding. No smoothing is added to hide it without being reported.

`reviewer` derives the equation of motion and the energy independently for each candidate before any production run, and checks M1 to M5.

### WP-C. Parts A, B and C (`runner`)

The static structure, the reflection test, and the crossing times, for each candidate and each value of L.

### WP-D. Part D (`runner`)

The pinch at five or more sizes of starting amplitude, for each candidate and each value of L, each with its linear twin.

### WP-E. Review (`reviewer`)

- Recompute the headline numbers from the saved data.
- Review the report text against the data before the final commit, and review the corrected text again after corrections. Every verdict quotes its measured number next to the rule it was judged by.

### WP-F. Report (orchestrator)

`docs/SESSION_06_REPORT.md`, with:

- A table of the rules tested (2, 3, 4, 12, 13, 17, 18, 20), each marked supported, contradicted, split or not tested, with the number that decides it.
- The predictions in section 4, each with its measured value.
- How every result depends on L and on the candidate.
- The scaling of energy per cycle with size, written as a relationship.
- Hits, split results and misses at the same depth. New entries in `docs/FAILURES.md`.
- The novelty statement from WP-A.
- What a contradicted rule would need from the medium to hold. Proposals only.
- Proposals for Session 7, and a parking lot.

## 6. Limits of this session

- **Round objects only.** No S-bend, no fixed-length wave, so rule 11 is not tested. The contracted core in Part A stands in for its steady pull.
- **A lone object.** No structures, no bonding, no clocks.
- **Candidate formulas are Claude Desktop's, not Aaron's.** His rules fix what the medium must do, not its formula.
- **Units only.** No physical values are attached.
- **Local compute.** No cloud.

## 7. Stop conditions

- The hard stops in `CLAUDE.md`.
- If no candidate can meet M1 to M5 together: stop, report which requirements conflict, and run nothing built on a broken medium.
- If the WP-B checks fail after two fix attempts: stop the runs and report.
- If a guard trips or a shock forms: log it as a failure entry with a diagnosis. Do not change the medium to avoid it.
- Do not start the full three-dimensional solver, the S-bend, structures, or the twist.
- Push only as `CLAUDE.md` allows.

## 8. Parking lot carried in

- Session 7 candidates: the full three-dimensional solver for this medium; the S-bend as a fixed-length wave and its end pull (rule 11); whether it walls itself in (rule 12); two objects and the drift of one toward another; bonding by geometry; a clock made of a structure, at rest in dense space and in motion (rules 20 and 23).
- The grid question from Session 5: a second numerical scheme and other smooth candidates.
- Session 4: the 2 times box rerun of the far-tail force (old equation).
- Open on the rule sheet: the twist; what connects two swirls created together.
- Later: energy per cycle as a route to Planck's constant; lensing as a route to the stiffness of space.
