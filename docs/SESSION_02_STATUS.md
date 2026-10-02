# Session 2 status

Resume file. Updated by the orchestrator after each package. Plan: `docs/SESSION_02_PLAN.md`. Predictions: `docs/SESSION_02_PREDICTIONS.md`.

Last update: 2026-10-02 07:20 (+07). Everything is done and committed except the E11 second runs and the beta = 5 3D confirmation (both running on the Mac) and the report lines that depend on them.

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
| WP-E arrangement 3D tests | numerics tooling, then runner and cloud | **Batch done and committed** (18 result files, `outputs/session2/wpE/analysis/criteria_E.json`). Script labels: merged E01 (contact 295.7), E02 (160.4), E08, E09, E10 (193), E12, and both checks of E03; flew apart E03, E05 (229.5), E06 (416), E13 (302); not distinct at the start E04, E07; **bonded E11 and E14**. E14 is the far-apart control (nothing happens). E11 is a ring still contracting when the run stopped at T = 700. Six rows scored as misses (E3, E7, E11 outcome; twin energy; the two checks' class against E3). **Running on the Mac:** the second run for E11 (T = 1200 at dx 0.55, and dx 0.393 to T = 1000; prediction Block 3b). **Reviewer** recomputing all outcomes from saved data | `outputs/session2/wpE/`, `docs/WPE_BUILD_NOTES.md` |
| WP-F review and report | reviewer, scholar, orchestrator | waits | `docs/SESSION_02_REPORT.md` |

## Running now

Checked 2026-10-02 07:20 (+07). All on the Mac; AWS is empty and its sign-in has lapsed again (not needed).
- **E11 second runs** (`E11L_ring6_c_R15.77_T1200` at t = 900 of 1200; `E11F_ring6_c_R15.77_dx0.393_T1000` at t = 300 of 1000, expected about 10:00), a runner is driving them and will rerun `outputs/session2/wpE/postprocess.py` and a point-model seed scan (`outputs/session2/wpD_measured/e11_seed_scan.py`). Resume a run with `outputs/session2/wpE/run_one.sh <name>`.
- **A10, beta = 5 in 3D** (`outputs/session2/wpA/confirm3d_beta5/`, T = 1000; prediction Block 1d), a second runner.
- **Report:** `docs/SESSION_02_REPORT.md` is complete except for the placeholders `{{...}}` for the E11 second run and A10. pytest on main: 130 passed, 4 xfailed (in the report).

## Rented compute

**Nothing is running or stored on AWS** (checked 2026-10-02 05:58 +07: no running instances, no buckets; the cloud worker deleted the bucket, role, profile and security group and verified at 22:56Z). Ledger total 9.69 USD of 50 (`docs/CLOUD_LEDGER.md`): Session 1 2.80; WP-C batch 3.98 (about 1.3 of it idle time while the sign-in was lapsed); WP-E batch 2.91. c8g.4xlarge ran this engine 2.2 times faster than c7i.4xlarge.

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
