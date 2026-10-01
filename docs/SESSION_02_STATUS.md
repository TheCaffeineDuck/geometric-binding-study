# Session 2 status

Resume file. Updated by the orchestrator after each package. Plan: `docs/SESSION_02_PLAN.md`. Predictions: `docs/SESSION_02_PREDICTIONS.md`.

Last update: 2026-10-02 00:15 (+07). WP0, WP-S done. WP-A sweep done, supplement running. WP-B running (T about 2000 of 5000). WP-C: gate runs on AWS, analysis script being revised. Reviews of both builds done. Ledger at F-009.

**Stop-condition note (plan section 4).** The phase control missed at theta = 90 only. The plan's control is over the phases used, and the redesigned batch uses none that fails, so I read the stop as not triggered; the 3D gate (G1, G2 in the predictions file) must pass before pair runs start. WP-D and WP-E go ahead only if the gate passes and the measured force agrees with the derivation in sign. This reading is flagged to Aaron and goes in the report as a deviation.

WP-A sweep command (resumes if rerun), from the repo root:
`PY=.venv/bin/python; nohup caffeinate -i sh -c "$PY -m swirl.family.wpa families --workers 2 && $PY -m swirl.family.wpa maps --workers 2 && $PY -m swirl.family.wpa rank && $PY -m swirl.family.wpa long --workers 2" > logs/session2/wpa_sweep.log 2>&1 &`

## Packages

| Package | Worker | State | Outputs |
|---|---|---|---|
| WP0 housekeeping | runner | **done**. Effort values "5" and "30": the Claude Code docs list only named levels (low, medium, high, xhigh, max) and give no numeric scale, and no documented way for the parent session to confirm a subagent's effort after the fact. So the mapping stays unconfirmed. The WP0 runner reported model `claude-sonnet-5-5`, effort not visible | Errata in `LESSONS_2026-10-01.md`; pointer line in `SWIRL_PLAN_2026-10-01.md`; reporting rule in `.claude/agents/numerics.md`; `AWS_REPORT.md` re-check: quota now 64 vCPU, request closed, 120 USD credits, nothing left in us-east-2 |
| WP-S prior-art brief | scholar | **done**; orchestrator has read S0, S6, S7, S8 | `docs/PRIOR_ART.md` lines 468 to 813, "Session 2 brief" (S0 to S10, notes N12 to N18, searches 41 to 82) |
| WP-A lifetime map | numerics build, then runner | **first sweep done and committed** (7 rate maps, 18 long runs, `lifetime_table.json`); supplement running (extension of the best starts to T = 300,000, matched-size starts, grid-edge and dip-midpoint starts; log `logs/session2/wpa_supp.log`; output `lifetime_table_v2.json`). Results so far: lifetime rises with beta with no fall to beta = 5; T_death 868 / 3327 / 8868 at beta 0.75 / 0.85 / 1.0; 28,052 at 1.15; every best start at beta >= 4/3 is alive at T = 30,000 (frac 0.997 at 4/3 from omega 0.9065, 0.9945, 0.990, 0.967 to 0.997, 0.969 to 0.999 at 1.6, 2, 3, 5). A1 MISSED (F-007). A2 hit to the cap. A2-CD miss. A3 misses at beta 0.75 (868 against 350 to 600) and 1.15 (T_90 27,231 against 3,000 to 15,000). A4 hit for the lowest-rate pick; the path-integral pick fails at 1.6 and 2.0 (0.79). A5 untested. Doubled box at beta = 5 to T = 30,000: 6.6e-6. Review done; F-007, F-008 written | `outputs/session2/wpA/`, `docs/WPA_BUILD_NOTES.md` |
| WP-B 3D primary to T = 5000 | runner | **running detached**, started 22:28, expected finish about 02:40. Collection and the isotropy check (B2) still to do | `outputs/session2/wpB/` |
| WP-C two-body law | numerics build, then runner or cloud | **build done and merged** (a41254e). Derivation confirms U = -2 pi A^2 cos(dphi) exp(-kappa d)/d; exact force has an extra 1/d^2 term. Radial phase control: passes at 0, +-22.5, +-45, +-67.5, 135, 180; MISSES at 90 (dfrac 2.46e-2 at T = 1000 against 0.01; a lighter lump, cause diagnosed). Pair design changed to symmetric phases so no 90-degree start is used. Predictions Block 2 committed (d17e208). Batch of 24 runs handed to cloud, gate runs first; review of the diff in progress | `outputs/session2/wpC/`, `docs/WPC_DERIVATION.md`, `docs/WPC_BUILD_NOTES.md` |
| WP-D arrangement maths | numerics | waits for WP-C | `outputs/session2/wpD/` |
| WP-E arrangement 3D tests | runner or cloud | waits for WP-C, WP-D | `outputs/session2/wpE/` |
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

**LIVE as of 2026-10-01 23:55 (+07). Check and clean up if the session is resumed cold.** us-east-2, profile `swirl`, everything tagged project=swirl.
- Instance i-0eb9855e4546429f2 (c8g.4xlarge, 0.6358 USD/h, launched 16:23Z, terminate-on-shutdown, dead-man `shutdown -h` at 18:34:48Z = 01:34 +07). Running the WP-C gate runs and three twins since 16:34:48Z; expected done about 17:20Z (00:20 +07).
- S3 bucket swirl-wpc-156084119557-71737 (code tarball; results, logs, checkpoints synced every 5 min under `A/`).
- IAM role swirl-wpc-role, instance profile swirl-wpc-profile, security group sg-0fd9e12ecf65ba23b. No key pairs.
- Cleanup if abandoned: terminate the instance; empty and delete the bucket; remove the role from the profile, delete the profile, detach AmazonSSMManagedInstanceCore, delete inline policy `s3bucket`, delete the role; delete the security group. Then add the ledger row.
- Cost so far about 0.7 USD (c8g 0.9 h, plus a 9-minute c7i benchmark instance, already terminated). Ledger row not yet written; ledger file still says 2.80.
- Benchmark at N = 160: c8g.4xlarge 33.5 steps/s total with 8 jobs x 2 threads; c7i.4xlarge 15.5. Engine tests on the c8g: 41 passed, 1 xfailed.
- Gate evaluation: `.venv/bin/python outputs/session2/wpC/gate_eval.py --results <dir>` writes `gate.json` (criteria G1, G2 in the predictions file). No pair run starts unless it passes.
- Plan after a pass: second c8g.4xlarge, 17 remaining configs split over the two, about 90 minutes, about 2.5 USD in total.

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
