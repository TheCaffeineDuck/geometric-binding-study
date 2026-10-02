# Session 3 plan

Date: 2026-10-02
Status: agreed with Aaron. Builds on `SESSION_02_PLAN.md` and `SESSION_02_REPORT.md`. Everything in `CLAUDE.md` still applies.

## 1. Decisions on record

| Topic | Decision |
|---|---|
| Session 2 stop condition | The orchestrator's reading (the phase control covers the phases used) is ratified by Aaron |
| Geometric binding | Negative for equal swirls in a single real field, at first order, with a derivation. Recorded in `docs/FINDINGS.md` this session. Still open for unequal lumps, orbital motion, second-order forces and a second field |
| Step 4 | Only the lone-lump half runs now: a break-threshold test of single swirls |
| Binding search | Continues in the point model with unequal lumps and orbital motion. These change only the starting conditions, not the equations |
| Second field | Held. Literature check only this session |
| Break-test starts | beta = 4/3 at two starts (inside the long-life dip, and outside it), and beta = 5 at one start |
| Rented compute | Same rules. 40.31 dollars of the 50 remain |

## 2. Work packages

Order matters. After each package finishes, the orchestrator updates `docs/SESSION_03_STATUS.md` so a new session can resume from it.

Dependencies: WP0 first. Then S, L and P in parallel. L3 needs L1 and L2. P3 needs P1 and P2. F needs everything.

### WP0. Housekeeping (runner)

- Create `docs/FINDINGS.md`: one short entry per established result from Sessions 1 and 2, each with its evidence file, its limits, and whether it is new, published, or not checked. Include the negative result for equal swirls with the structural reason, the mirror-plane no-go (N15) as derived but not tested, the lifetime map, and the long-lived lone swirl.
- Replace the bonded rule with the corrected rule from section 13 of the Session 2 report: the lump count equals K in every sample; nearest-neighbour distances within about 10% with no trend above a few times the far-apart control's drift; a run of at least five interaction times, with the in-phase control's contact time as the unit; per-lump N and phase steps held; a response that differs from the far-apart control. Put it in the code and in the predictions template.
- Fix the lump finder so debris on a nearly empty field is not counted as lumps, and rename or document `E_retained_end` as residence in the group sphere.

### WP-S. Prior-art brief (scholar)

Adds to `docs/PRIOR_ART.md`.

- Oscillon response to perturbations: decay thresholds, amplitude kicks, radiation hitting an oscillon, and oscillon collisions (absorption against bounce).
- Interaction of oscillons or solitons with unequal frequencies.
- Bound states held by orbital motion: rotating soliton molecules and orbiting oscillons.
- The held second field: are lumps of a complex field (Q-balls) absolutely stable when their charge is conserved, metastable, or either depending on parameters? Does that conflict with "a lone swirl is metastable"? Which multi-component models give bound stepped-phase clusters while keeping lone lumps metastable?

### WP-L. Lone-lump break test (numerics builds, runner or cloud runs)

**Question.** How large a disturbance does a lone swirl survive, and how does a disturbance change its lifetime?

**Starts.** beta = 4/3, omega = 0.9065 (in the dip); beta = 4/3, omega = 0.9091 (the Session 1 primary, outside the dip); beta = 5, omega = 0.9756. Use settled lumps as the starting state.

**Disturbances.** Each one is described by the fractional change it makes in the lump's N, with the energy delivered as a second number.
- (a) Amplitude kick: scale the field or its velocity inside the lump's ball by 1 + epsilon, for both signs of epsilon.
- (b) N removal: remove a fraction of N from the core.
- (c) Incoming spherical radiation packet, at two wavenumbers: a slow one, and one near the third-harmonic wavenumber.
- (d) Non-spherical distortion: a quadrupolar stretch and a perturbation with no symmetry, at several amplitudes. This also gives the long-lived dip start its first generic-perturbation test.
- (e) Collision with a second lump at a chosen phase and speed, at two speeds and phases 0, 90 and 180 degrees. This is the absorption test.

**Definitions.**
- **Broken outright:** within 2,000 time units, the energy inside 3 half-widths of the tracked centre falls below half its value just before the disturbance, while the undisturbed control stays above 0.9.
- **Degraded:** the loss rate and the omega shift after the disturbance, relative to the undisturbed control, with the radial solver giving the long horizon.

**Work.**
- L1: spherical disturbances (a, b, c) in the radial solver, scanning amplitude to find the threshold for broken outright and the degradation curve below it.
- L2: build and test the 3D initial data for (d) and (e). For (e), moving lumps need a boosted settled lump. Before any collision run, check that a single boosted lump moves at its set speed with energy gamma E0 and oscillation frequency omega/gamma.
- L3: 3D runs for (d) and (e), plus 3D spot checks of a few radial points from L1.

**Output.** Break thresholds and degradation curves per start and disturbance, in units of the fractional change in N.

**Prediction on record (Claude Desktop, no source).** The dip start at beta = 4/3 is the most fragile in lifetime terms, because a small change in N moves it out of the dip; beta = 5 is the most robust. Collisions: in phase absorbs, anti-phase bounces.

### WP-P. Binding search with unequal lumps and orbital motion (numerics)

**Question.** Can unequal lumps or orbital motion hold distinct swirls together, where equal static lumps cannot?

- P1: extend the point model if needed and validate each new piece: lumps of different N and omega; lumps with initial velocities, including tangential velocities for rotating groups. Check that a single moving lump behaves like the 3D boosted lump from L2.
- P2: scan.
  - Pairs and small rings of unequal lumps across a range of N ratios and phases.
  - Orbiting pairs and rotating rings across speed, spacing and phase pattern, including stepped phases, looking for orbits that stay bounded with the phase pattern holding.
  - Record whether the phase instability is slower or faster than the orbit, since that decides whether rotation can outrun it.
  - Flag anything with lumps closer than about 3.6 half-widths, where the model is not valid.
- P3: test the best candidates in 3D, if any, with the corrected bonded rule, a far-apart control and an in-phase control. A bonded outcome gets a second run at finer resolution and a longer time before it is reported.

**Guess on record (Claude Desktop, no source).** Unequal frequencies make the relative phase rotate, so the first-order force averages close to zero and unequal lumps neither bind nor merge quickly. Orbital motion is the more likely route.

### WP-F. Review and report (reviewer, scholar, orchestrator)

- Reviewer: each numerics diff, the break thresholds recomputed from saved data, and any bonded or near-bonded outcome recomputed from saved data.
- Scholar: failure ledger entries for every miss, and the novelty check for anything the report would call new.
- `docs/SESSION_03_REPORT.md` in the Session 2 format, ending with a section on what the results mean for claim 1, and proposals for what to do next.

## 3. Stop conditions

- No change to the equations of the model. The second field stays a literature question.
- If the boosted single-lump check in L2 fails, collision runs and 3D orbital runs wait; report what failed.
- The hard stops in `CLAUDE.md`.
