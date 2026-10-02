# WP-P3 batch (prepared, not run)

3D tests of the two P2 point-model candidates labelled "bounded, beating" that the orchestrator picked: unequal static pairs at 7.2 hw. The point-model diagnostics in `outputs/session3/wpP/p2_diagnostics.json` already show both drifting together, once the missing N34 force is added or the run is three times longer.

## Lumps and sources (beta = 4/3, Set-B units)

| key | N / N0 | omega (profile) | N | E | A | kappa | hw | source |
|---|---|---|---|---|---|---|---|---|
| N100 | 1.00 | 0.909149 | 2865.5 | 2801.7 | 56.70 | 0.4165 | 3.344 | reused: `outputs/session2/wpE/source/src_prod_N160_L88.npz` (settled omega 0.909532, sha 6193e7b1...) |
| N175 | 1.75 | 0.878988 | 5014.7 | 4718.4 | 122.37 | 0.4768 | 4.081 | new `src_p3_N175` |
| N250 | 2.50 | 0.863859 | 7163.9 | 6589.9 | 204.30 | 0.5037 | 4.636 | new `src_p3_N250` |

- p2_0230 is N100 + N175 (ratio 1.75, phases 0 and 0); p2_0242 is N100 + N250 (ratio 2.5, phases 0 and 90). The N100 source is shared.
- The settled 3D lumps will sit slightly off the profile omega: Session 2's N100 source moved from 0.909149 to 0.909532. The realized N ratio is therefore the ratio of the settled lumps; `make_sources.py` records each settled omega.
- Two-source starts: `init["snapshot"]` may now be a list of snapshot entries, one per source.
  - This is an additive change to `swirl/engine/initial.py` and `swirl/engine/lumps.py`; a single dict behaves exactly as before.
  - The test is `tests/engine/test_snapshot_multi.py`: a one-entry list is bit-identical to the dict form, and a two-entry list is the sum of the entries.
  - Merge note for the WP-L3 branch (worktree-agent-a511a354d573f02da): `snapshot.py` itself is unchanged.

## Jobs (grid dx 0.55, dt 0.04; boxes by the WP-E auto-box rule per lump; window (26, 30))

| run | role | d | N | L | T | steps | estimated hours (2 threads) |
|---|---|---|---|---|---|---|---|
| P3_c0230 | candidate p2_0230 | 24.08 (7.2 hw0) | 160 | 88 | 7500 | 187,500 | 12.4 |
| P3_c0242 | candidate p2_0242 | 24.08 | 160 | 88 | 7500 | 187,500 | 12.4 |
| P3_c0230_far | far-apart control | 45 | 200 | 110 | 7500 | 187,500 | 24.3 |
| P3_c0242_far | far-apart control | 45 | 200 | 110 | 7500 | 187,500 | 24.3 |
| P3_inphase | in-phase control, equal N100 lumps; contact time = interaction unit | 24.08 | 160 | 88 | 2000 | 50,000 | 3.3 |
| P3_c0230_lin | linear twin of P3_c0230 | 24.08 | 160 | 88 | 1000 | 25,000 | 1.7 |

- **T for the candidates.** 7500 is at least 5 interaction units for any in-phase contact up to 1500. The point model gives 921 at this spacing; Session 2's 3D d = 24 in-phase pair tracked the model to 0.03 by T = 500.
- **Rate basis.** Session 2 ran 33.5 steps/s in total at N = 160 with 8 jobs of 2 threads on a c8g.4xlarge, which is 4.19 steps/s per job. N = 200 is scaled as N^-3.
- **Cost, all six jobs at once on one c8g.4xlarge (12 of 16 cores):** the far controls set the wall time, about 24.3 instance-hours, about 15.4 USD.
  - Option: give the two far controls 4 threads (16 cores in use). If they scale to about 1.9 times faster (not measured on this instance), the wall time is about 12.8 h and the cost about 8.1 USD.
  - Sources first: 2 jobs of about 0.5 h at N = 160 on 2 threads, on the Mac or on the instance (adds about 0.3 USD).

## Order

1. `NUMBA_NUM_THREADS=2 <python> outputs/session3/wpP3/make_sources.py N175`, then `N250`. The profiles are already written.
2. `<python> outputs/session3/wpP3/make_configs.py`. This writes the configs with the source hashes and runs the engine guards. `--dry` gives the plan only (`plan.json`).
3. `nohup outputs/session3/wpP3/run_batch.sh > outputs/session3/wpP3/logs/batch.log 2>&1 &`. Jobs are resumable from checkpoints every 50 time units.
4. `<python> outputs/session3/wpP3/score.py`, which writes `score.json`.

## Scoring (`score.py`)

- **Interaction unit.** The contact time of P3_inphase: the first time the tracked centroid distance falls below 2 hw0, or failing that the first count window with fewer than 2 distinct lumps.
- **The rule.** The corrected rule (`bonded.py`, unchanged) is applied to each candidate against its far-apart control, with that unit.
- **Block P labels.** These come from `swirl.arrange.p2scan.label`, the same function the P2 scan used.
  - "Locked": the unwrapped relative-phase drift times the run length, and the excursion about that drift, are both within 0.25 degrees (or 3 times the control's excursion).
  - "Bounded, beating": clauses 1 to 4 hold and the step drifts at the measured omega difference of the two lumps, within 10%.
  - Otherwise the label is the rule's verdict.
  - Clause 5 uses the lab step, which equals D_eff for lumps at rest.
- **Flag.** A closest approach below 3.6 hw (the mean hw of the two lumps) is flagged and not counted.
- **Before reporting.** A "bonded" or "bounded, beating" outcome needs the plan's second run at finer resolution and longer T before it is reported.
