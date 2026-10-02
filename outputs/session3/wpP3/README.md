# WP-P3 batch (prepared, not run)

These are 3D tests of the two P2 point-model candidates the orchestrator picked, both labelled "bounded, beating": unequal static pairs at 7.2 hw. The point model cannot settle unequal lumps: with the missing N34 force added, or with a run three times longer, both drift together (`outputs/session3/wpP/p2_diagnostics.json`). The 3D runs are the test.

## Lumps and sources (beta = 4/3, Set-B units)

| key | N / N0 | omega (profile) | N | E | A | kappa | hw | source |
|---|---|---|---|---|---|---|---|---|
| N100 | 1.00 | 0.909149 | 2865.5 | 2801.7 | 56.70 | 0.4165 | 3.344 | reused: `outputs/session2/wpE/source/src_prod_N160_L88.npz` (settled omega 0.909532, sha 6193e7b1...) |
| N175 | 1.75 | 0.878988 | 5014.7 | 4718.4 | 122.37 | 0.4768 | 4.081 | new `src_p3_N175` |
| N250 | 2.50 | 0.863859 | 7163.9 | 6589.9 | 204.30 | 0.5037 | 4.636 | new `src_p3_N250` |

- **The two candidates.** p2_0230 is N100 + N175 (phases 0 and 0). p2_0242 is N100 + N250 (phases 0 and 90).
- **Settled lumps.** The settled 3D lumps sit slightly off the profile omega (N100 settled from 0.909149 to 0.909532). The realized N ratio is therefore the ratio of the settled lumps; `make_sources.py` records the settled omegas.
- **Two-source starts.** `init["snapshots"]` is a list of snapshot entries, one per source.
  - Each entry is placed exactly like an `init["snapshot"]` entry.
  - The change is additive, in `swirl/engine/initial.py` and `swirl/engine/lumps.py`, as a new block before the existing snapshot block. The existing block and `snapshot.py` are untouched.
  - Test: `tests/engine/test_snapshot_multi.py`.
  - `git merge-tree` against the WP-L3 branch shows no conflict.

## Deviation: the control is constructed from isolated single-lump runs (for cost)

A far-apart pair (d = 45) needs a box of N = 200, about 24 h per control. Instead, each lump is run alone (P3_iso_*, N = 120, same dx and T). `score.py` builds the rule's control series from the two isolated runs of a candidate's lumps:
- **count:** 2 in every window;
- **nearest-neighbour distance:** the candidate's start distance plus the two lumps' measured centroid drifts;
- **per-lump N:** from each isolated run;
- **neighbour step:** the start phase difference plus the time integral of the two runs' own omega difference.

This control has no interaction at all, where a far-apart pair has a small one. It also leaves out any effect of the second lump's radiation on the first.

## Jobs, packed 8 slots x 2 threads on one c8g.4xlarge (0.6358 USD/h)

Rates: Session 2 measured 4.19 steps/s per job at N = 160 with 8 jobs x 2 threads; other N are scaled as N^-3. A packed job's cost share is its hours / 8 x 0.6358 USD.

Phase 1 (4 slots used, 4 spare):

| job | what | N | T | steps | hours | USD share |
|---|---|---|---|---|---|---|
| src_p3_N175 | source (make_sources.py) | 160 | 300 + 1 period | 7,600 | 0.5 | 0.04 |
| src_p3_N250 | source | 160 | 300 + 1 period | 7,600 | 0.5 | 0.04 |
| P3_inphase | two N100 lumps in phase, d = 24.08; contact time = interaction unit | 160 | 2000 | 50,000 | 3.3 | 0.26 |
| P3_c0230_lin | linear twin of P3_c0230 (g4 = g6 = 0) | 160 | 1000 | 25,000 | 1.7 | 0.13 |

The in-phase control and the twin use only the N100 source, so they can start before the two new sources are ready.

Phase 2 (after P3_inphase; 5 slots used, 3 spare). T = min(7500, max(5 x 3D contact, 3000)), written to `T_choice.json`. Hours are shown at the cap and at the point model's unit (contact 921, so T = 4605):

| job | what | N | hours at T = 7500 | USD share | hours at T = 4605 | USD share |
|---|---|---|---|---|---|---|
| P3_c0230 | candidate p2_0230, d = 24.08 | 160 | 12.4 | 0.99 | 7.6 | 0.61 |
| P3_c0242 | candidate p2_0242, d = 24.08 | 160 | 12.4 | 0.99 | 7.6 | 0.61 |
| P3_iso_N100 | isolated N0 lump | 120 | 5.2 | 0.42 | 3.2 | 0.26 |
| P3_iso_N175 | isolated 1.75 N0 lump | 120 | 5.2 | 0.42 | 3.2 | 0.26 |
| P3_iso_N250 | isolated 2.5 N0 lump | 120 | 5.2 | 0.42 | 3.2 | 0.26 |

Totals:

| | at T = 7500 | at T = 4605 |
|---|---|---|
| job-hours | 46.6 | 30.9 |
| packed share | 3.70 USD | 2.46 USD |
| on its own instance (wall) | 15.8 h, 10.0 USD | 11.0 h, 7.0 USD |

## Order (from the repo root)

1. `NUMBA_NUM_THREADS=2 <python> outputs/session3/wpP3/make_sources.py N175`, and the same for `N250`. The profiles are already written.
2. `<python> outputs/session3/wpP3/make_configs.py A`, then launch phase 1 with `run_batch.sh P3_inphase P3_c0230_lin`.
3. When P3_inphase has completed: `<python> outputs/session3/wpP3/make_configs.py B` (this sets and records T), then launch phase 2 with `run_batch.sh P3_c0230 P3_c0242 P3_iso_N100 P3_iso_N175 P3_iso_N250`.
4. `<python> outputs/session3/wpP3/score.py`, which writes `score.json`.

All jobs are resumable from checkpoints, saved every 50 time units.

## Scoring (`score.py`)

- **Unit.** The 3D contact time of P3_inphase: the first time the tracked distance falls below 2 hw0, or failing that the first count window with fewer than 2 distinct lumps. On the Session 2 d = 20 in-phase pair this gives 347.82, Session 2's value.
- **Rule.** The corrected rule (`bonded.py`, unchanged) is applied to each candidate against its constructed control, with that unit.
- **Labels.** These come from `swirl.arrange.p2scan.label`, the same function the P2 scan used.
  - "Locked": the unwrapped relative-phase drift times the run length, and the excursion about that drift, are both within 0.25 degrees (or 3 times the control's excursion).
  - "Bounded, beating": clauses 1 to 4 hold and the step drifts at the measured omega difference of the two lumps, within 10%.
  - Otherwise the label is the rule's verdict.
  - Clause 5 uses the lab step, which equals D_eff for lumps at rest.
- **Flag.** A closest approach below 3.6 hw (the mean hw of the two lumps) is flagged and not counted.
- **Before reporting.** A "bonded" or "bounded, beating" outcome needs the plan's second run at finer resolution and longer T before it is reported.
