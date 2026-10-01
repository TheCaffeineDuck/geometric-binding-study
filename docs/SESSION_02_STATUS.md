# Session 2 status

Resume file. Updated by the orchestrator after each package. Plan: `docs/SESSION_02_PLAN.md`. Predictions: `docs/SESSION_02_PREDICTIONS.md`.

Last update: 2026-10-02 02:10 (+07). AWS SIGN-IN EXPIRED (see Rented compute). WP0, WP-S, WP-A sweeps, WP-D (derived law) done. Running on the Mac: WP-B (T about 3200 of 5000); local copies of the d = 16 pair runs for the force-sign check (`outputs/session2/wpC/results_mac/`, analysis in `analysis_mac/`); the A9 3D confirmation. WP-E tooling build running. Scholar writing F-010 to F-012.

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
| WP-C two-body law | numerics build, then runner or cloud | **build done and merged** (a41254e). Derivation confirms U = -2 pi A^2 cos(dphi) exp(-kappa d)/d; exact force has an extra 1/d^2 term. Radial phase control: passes at 0, +-22.5, +-45, +-67.5, 135, 180; MISSES at 90 (dfrac 2.46e-2 at T = 1000 against 0.01; a lighter lump, cause diagnosed). Pair design changed to symmetric phases so no 90-degree start is used. Predictions Block 2 committed (d17e208). Batch of 24 runs handed to cloud, gate runs first; review of the diff in progress | `outputs/session2/wpC/`, `docs/WPC_DERIVATION.md`, `docs/WPC_BUILD_NOTES.md` |
| WP-D arrangement maths | numerics | **done with the derived law and merged** (f049374); review in progress; to be rerun with the measured law (`python -m swirl.arrange.scan --params <new>.json --out outputs/session2/wpD`, about 100 s). Result: 595 patterns (312 classes of 0/180, 283 stepped), 28 force balances, none stable once phases and N move. CD's two numbers confirmed (5-ring 72-degree steps: kappa R 1.0695, radially restoring, 4 negative modes of 15, spacing 0.90 hw; octahedron around anti-phase centre: 1.591, not restoring). 27 of 28 balances lie below 3.6 hw where the tail law fails. Point-model evolution: stepped rings merge 1.2 to 2 times later than in-phase controls | `outputs/session2/wpD/` (candidates.json, predictions_point_model.json, tables.md), `docs/WPD_NOTES.md` |
| WP-E arrangement 3D tests | numerics tooling, then cloud | **tooling build running** (snapshot starts so that lumps at any phase are identical, K-lump diagnostics, angular-momentum fix, configs and post-processing). No arrangement run until Block 3 of the predictions is written and C1 (force sign) is confirmed by the pair batch | `outputs/session2/wpE/`, `docs/WPE_BUILD_NOTES.md` |
| WP-F review and report | reviewer, scholar, orchestrator | waits | `docs/SESSION_02_REPORT.md` |

## Running now

**WP-B 3D run.** PID in `logs/session2/wpB.pid` (python 59202 under `caffeinate -i`), 2 threads, about 6.7 steps/s, T = 950 to 5000.
- Log: `logs/session2/wpB.log` (one line per 250 time units). Checkpoint: `checkpoints/session2/wpB/checkpoint.npz`. Result, rewritten each chunk: `outputs/session2/wpB/run1_primary_T5000.json`.
- If it dies, rerun from the repo root; the driver continues from the checkpoint:
  `NUMBA_NUM_THREADS=2 nohup caffeinate -i .venv/bin/python outputs/session2/wpB/run_T5000.py --ckpt checkpoints/session2/wpB --out outputs/session2/wpB/run1_primary_T5000.json --chunk-steps 6250 >> logs/session2/wpB.log 2>&1 &`
- When done: `.venv/bin/python outputs/session2/wpB/compare_radial.py` writes `compare_radial.json` and `.png`. Then run `outputs/step2/diagnostics/shape.py` on the final checkpoint for B2.
- Resume fidelity was checked: T = 950 to 1000 regenerated bit-identical to Session 1. The Session 1 checkpoint stops at T = 950 because the engine does not checkpoint at the last step.
- Comparison on T <= 1000: max |frac 3D - radial| = 7.26e-5 at t = 255.

## Rented compute

**BLOCKED since 18:24Z (01:24 +07) on 2026-10-02: the AWS sign-in for profile `swirl` has expired.** Both the CLI and the AWS MCP tool fail. Aaron must run `aws login --profile swirl`. Nobody in the session can check or clean up AWS until then.

Still on AWS (us-east-2, account 156084119557, tagged project=swirl):
- Instance A i-0eb9855e4546429f2 and instance B i-069c4011d363790e3 (both c8g.4xlarge, 0.6358 USD/h; launched 16:23Z and 17:25Z). **Dead-man timers terminate them at 19:54:50Z and 19:55:20Z (02:55 +07) without any sign-in.** Root volumes delete on termination.
- S3 bucket swirl-wpc-156084119557-71737: each instance syncs results, logs and checkpoints every 5 minutes to `A/` and `B/`. It survives termination. Role swirl-wpc-role, profile swirl-wpc-profile, security group sg-0fd9e12ecf65ba23b also remain until deleted.
- State at the last contact (18:15Z): gate passed (G1 2.9e-5; G2 worst 5.35e-3 and 0.00275); 7 runs back on the Mac (4 gate runs, 3 twins); four d = 16 pairs finished on B but not copied; the other pairs at t = 450 of 500; `pair_d16_p180` at 150 of 300; `check_box` at 2500 of 7500 steps and `check_res` at 3750 of 7500. Everything was expected to finish by about 19:00Z, before the timers.
- Cost so far about 2.0 USD; projected 2.9 to 3.3 USD for the batch (cap 12). Ledger row not yet written; ledger file still says 2.80.

After sign-in (hand to the `cloud` worker): pull `A/results`, `A/logs`, `B/results`, `B/logs` from the bucket into `outputs/session2/wpC/results/` and `logs/`; check all 24 JSONs have `completed: true` (a run cut off by the timer is resumable from its checkpoint in the bucket); run `.venv/bin/python outputs/session2/wpC/postprocess.py`; confirm both instances are terminated; delete the bucket (after the results are verified locally), the role (detach AmazonSSMManagedInstanceCore, delete inline policy `s3bucket`), the instance profile and the security group; verify nothing remains; write the ledger row.

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
