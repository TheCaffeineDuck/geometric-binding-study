# Session 2 status

Resume file. Updated by the orchestrator after each package. Plan: `docs/SESSION_02_PLAN.md`. Predictions: `docs/SESSION_02_PREDICTIONS.md`.

Last update: 2026-10-02 04:50 (+07). Pair batch in and analysed; A9 hit; E01, E02 done; rest of WP-E on AWS; WP-B nearly done. Remaining after that: reviewer passes, ledger entries for WP-C and WP-E misses, novelty check, final report, pytest on main.

**Stop-condition note (plan section 4).** The phase control missed at theta = 90 only. The plan's control is over the phases used, and the redesigned batch uses none that fails, so I read the stop as not triggered; the 3D gate (G1, G2 in the predictions file) must pass before pair runs start. WP-D and WP-E go ahead only if the gate passes and the measured force agrees with the derivation in sign. This reading is flagged to Aaron and goes in the report as a deviation.

WP-A sweep command (resumes if rerun), from the repo root:
`PY=.venv/bin/python; nohup caffeinate -i sh -c "$PY -m swirl.family.wpa families --workers 2 && $PY -m swirl.family.wpa maps --workers 2 && $PY -m swirl.family.wpa rank && $PY -m swirl.family.wpa long --workers 2" > logs/session2/wpa_sweep.log 2>&1 &`

## Packages

| Package | Worker | State | Outputs |
|---|---|---|---|
| WP0 housekeeping | runner | **done**. Effort values "5" and "30": the Claude Code docs list only named levels (low, medium, high, xhigh, max) and give no numeric scale, and no documented way for the parent session to confirm a subagent's effort after the fact. So the mapping stays unconfirmed. The WP0 runner reported model `claude-sonnet-5-5`, effort not visible | Errata in `LESSONS_2026-10-01.md`; pointer line in `SWIRL_PLAN_2026-10-01.md`; reporting rule in `.claude/agents/numerics.md`; `AWS_REPORT.md` re-check: quota now 64 vCPU, request closed, 120 USD credits, nothing left in us-east-2 |
| WP-S prior-art brief | scholar | **done**; orchestrator has read S0, S6, S7, S8 | `docs/PRIOR_ART.md` lines 468 to 813, "Session 2 brief" (S0 to S10, notes N12 to N18, searches 41 to 82) |
| WP-A lifetime map | numerics build, then runner | **sweep and supplement done and committed** (100 long runs; `lifetime_table.json`, `lifetime_table_v2.json`). Lifetime rises with beta, no fall to beta = 5. Longest-lived starts: T_death 868 (0.75), 3380 (0.85), 29,404 (1.0); alive at 30,000 from 1.15 up. To T = 300,000: beta 4/3 from omega 0.906505 keeps 0.9946 (rate 4e-9 rising to 1.7e-8); 1.6: 0.9884; 2.0: 0.9882; 3.0: 0.969 and 0.984; 5.0: 0.969 and 0.998 (rate zero within error). Doubled box at 4/3 to 300,000: 5.3e-6. Predictions: A1 miss (F-007); A2 hit to the cap; A2-CD not supported; A3 misses at 0.75, 1.15, and 1.0 in the supplement; A4 hit; A5 untested; A6 miss (longer than predicted); A7 hit; A8 miss (T_90 at matched size is not monotone; T_death is). Ledger entries F-010 to F-012 being written. **Pending:** 3D confirmation of the 4/3 start at omega 0.906505 (prediction A9; running on the Mac, `outputs/session2/wpA/confirm3d/`); a beta = 5 3D confirmation needs rented compute | `outputs/session2/wpA/`, `docs/WPA_BUILD_NOTES.md` |
| WP-B 3D primary to T = 5000 | runner | **running detached**, started 22:28, expected finish about 02:40. Collection and the isotropy check (B2) still to do | `outputs/session2/wpB/` |
| WP-C two-body law | numerics build, then cloud | **Batch done and committed**: 24 AWS results in `outputs/session2/wpC/results/`, analysis in `analysis/criteria.json`. Hits: C1 (sign, all 12), C3 (kappa 0.4172 against 0.416), C5a, C5c (phase at the end 215, 160, 107 against beyond 180, 160, 102), C5d, C6, C7 (contact 126.8 and 347.8 against 126 and 346; merged pair keeps 0.76), C8, C9, C10 (no bonded pair), C12, C13. Misses: C2 on the sliding-force measure for d16_0, d16_45, d20_0; C4 at d = 16; C5b at d = 24 (and C5-CD with it); C11. Stop condition on the force sign: not triggered. Running: reviewer recomputing the verdict from saved data; runner making -theta single-lump runs for the C5b diagnostic and rerunning the WP-D scan with the measured law (`outputs/session2/wpD_measured/`). Local Mac copies of the d = 16 runs are in `results_mac/` | `outputs/session2/wpC/`, `docs/WPC_DERIVATION.md`, `docs/WPC_BUILD_NOTES.md` |
| WP-D arrangement maths | numerics | **done with the derived law and merged** (f049374); review in progress; to be rerun with the measured law (`python -m swirl.arrange.scan --params <new>.json --out outputs/session2/wpD`, about 100 s). Result: 595 patterns (312 classes of 0/180, 283 stepped), 28 force balances, none stable once phases and N move. CD's two numbers confirmed (5-ring 72-degree steps: kappa R 1.0695, radially restoring, 4 negative modes of 15, spacing 0.90 hw; octahedron around anti-phase centre: 1.591, not restoring). 27 of 28 balances lie below 3.6 hw where the tail law fails. Point-model evolution: stepped rings merge 1.2 to 2 times later than in-phase controls | `outputs/session2/wpD/` (candidates.json, predictions_point_model.json, tables.md), `docs/WPD_NOTES.md` |
| WP-E arrangement 3D tests | numerics tooling, then runner and cloud | Tooling merged and reviewed; outcome rule corrected (c2c96ef). E01 and E02 done on the Mac and committed: both merged (contact 295.7 and 160.4; ratio 1.84 against 1.85 predicted); the merged objects disperse (group energy about 2% at the end). **Other 16 configs running on AWS** (cloud worker; results to `outputs/session2/wpE/results/`). Then `outputs/session2/wpE/postprocess.py`, reviewer recomputation of any bonded or near-bonded outcome, ledger entries | `outputs/session2/wpE/`, `docs/WPE_BUILD_NOTES.md` |
| WP-F review and report | reviewer, scholar, orchestrator | waits | `docs/SESSION_02_REPORT.md` |

## Running now

Checked 2026-10-02 04:50 (+07).
- **AWS:** the cloud worker is running the WP-E batch (16 configs, two c8g.4xlarge) and will clean up and write the ledger rows. See "Rented compute".
- **Mac:** WP-B 3D run (PID in `logs/session2/wpB.pid`; at T about 4700 of 5000; restart command in `outputs/session2/wpB/run_T5000.py` header and below); a runner doing the -theta single-lump runs (`outputs/session2/wpC/results/single_x12_tm*.json`), the WP-D rerun with the measured law, and the WP-B collection (compare_radial.py, shape at T = 5000).
  - WP-B restart if it dies: `NUMBA_NUM_THREADS=2 nohup caffeinate -i .venv/bin/python outputs/session2/wpB/run_T5000.py --ckpt checkpoints/session2/wpB --out outputs/session2/wpB/run1_primary_T5000.json --chunk-steps 6250 >> logs/session2/wpB.log 2>&1 &`
- **Reviewer:** recomputing the pair-law verdict from the saved results.
- **Done on the Mac since the last update:** A9 3D confirmation of the beta = 4/3, omega 0.9065 start: hit (max difference from radial 7.0e-5 to T = 3000; `outputs/session2/wpA/confirm3d/`).

## Rented compute

**AWS sign-in restored at about 03:28 (+07) on 2026-10-02** (Aaron ran `aws login --profile swirl`). Checked at 03:30: both WP-C instances are terminated (dead-man timers); the bucket swirl-wpc-156084119557-71737 holds all 24 WP-C result JSONs.

The `cloud` worker has been resumed with two parts:
1. **WP-C collection:** pull the 24 results and logs into `outputs/session2/wpC/results/` and `logs/`, verify, run `outputs/session2/wpC/postprocess.py` (writes `analysis/criteria.json`), write the ledger row.
2. **WP-E batch on two c8g.4xlarge** (cap 10 USD): 16 configs (E03 to E14, the twin, the two checks, and E01 as a cross-machine duplicate named `E01_ring5_m1_a16_aws`). E01 and E02 finish on the Mac; the Mac slot lists were truncated in place to those two runs (full lists kept as `slotA_full.txt`, `slotB_full.txt`). Results come back to `outputs/session2/wpE/results/`; then `outputs/session2/wpE/postprocess.py`; then full cleanup and the second ledger row.

Still on AWS until the worker cleans up: the WP-C bucket, role swirl-wpc-role, profile swirl-wpc-profile, security group sg-0fd9e12ecf65ba23b, plus whatever Part 2 creates. If the session is resumed cold: check `aws ec2 describe-instances` and `aws s3 ls` with profile `swirl` in us-east-2, terminate and delete, and bring `docs/CLOUD_LEDGER.md` up to date (it still says 2.80; WP-C cost about 3 USD).

## Notes for a resuming session

WP-A review (done): measured numbers are sound; path-integral predictions and the long-run selection are not. To do after the sweep returns: (1) a supplement of long runs per beta: the 4 to 6 lowest measured-rate starts, the longest measured death, the grid-edge starts, midpoints at the two-point dips (beta 1.15, 1.6, 2.0), a grid extension past alpha^2 = 0.14 where the best sits at the edge (beta 0.75, 0.85, 1.0), and matched N/N_min starts for a fair comparison across beta; (2) fill "Reviewer check" in F-007 (wait until the scholar has finished F-009, to avoid two writers in FAILURES.md); corrections are appended to `docs/WPA_BUILD_NOTES.md`.

From the WP-S brief, to carry into the predictions and into WP-D and WP-E:
- The tail law is published for the non-relativistic limit (Malomed 1998) and matches the Session 1 form. The N-transfer rate has the same prefactor with sin(dphi) (N14, derived).
- At 90 degrees expect repulsion with a large N split, not zero force (Burlak & Malomed 2018, 3D cubic-quintic NLS). Expect 45 degrees to merge.
- Patterns with a phase-exchanging mirror plane (0/180 pairs, alternating rings, chains) have no radiation-free bound state (N15). Use them as controls.
- Any static point-model balance with net attractive single-lump energy is unstable in some direction (N16). CD's 5-ring and octahedron numbers reproduce but sit at 0.90 and 1.6 half-widths.
- Literature candidates for WP-E: stepped rings with N = 5 or 6 (step 72 or 60 degrees) and N = 8 with two turns; controls: all in phase, 0/180 ring, step above 90 degrees, lumps far apart. Published clusters are complex fields, rotate, and last a few revolutions.
- WP-A: record omega_dip and omega_death against beta; a "longevity cliff" may sit near beta 1.6 to 2.1 (N17, low confidence).

- Predictions Block 1 (WP-A, WP-B, WP-C phase control) is written. Block 2 must be written before any pair run, Block 3 before any arrangement run.
- numerics workers commit on their worktree branches. The orchestrator merges to `main`.
