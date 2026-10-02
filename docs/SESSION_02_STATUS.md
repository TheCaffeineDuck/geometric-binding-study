# Session 2 status

Resume file. Plan: `docs/SESSION_02_PLAN.md`. Predictions: `docs/SESSION_02_PREDICTIONS.md`. Report: `docs/SESSION_02_REPORT.md`.

Last update: 2026-10-02 10:10 (+07). **Session 2 is complete.** Every package WP0 to WP-F is done and committed; every planned run has finished; nothing is running on the Mac or on AWS; pytest on main: 130 passed, 4 xfailed. Step 4 has not been started. The table below is the record of each package; the sections after it are kept for history.

**For the next session:** read the report's Verdict, section 12 (deviations, including the stop-condition reading that Aaron has not yet ruled on) and section 13 (proposals for Step 4).

## Packages

| Package | Worker | State | Outputs |
|---|---|---|---|
| WP0 housekeeping | runner | **done**. Effort values "5" and "30": the Claude Code docs list only named levels (low, medium, high, xhigh, max) and give no numeric scale, and no documented way for the parent session to confirm a subagent's effort after the fact. So the mapping stays unconfirmed. The WP0 runner reported model `claude-sonnet-5-5`, effort not visible | Errata in `LESSONS_2026-10-01.md`; pointer line in `SWIRL_PLAN_2026-10-01.md`; reporting rule in `.claude/agents/numerics.md`; `AWS_REPORT.md` re-check: quota now 64 vCPU, request closed, 120 USD credits, nothing left in us-east-2 |
| WP-S prior-art brief | scholar | **done**; orchestrator has read S0, S6, S7, S8 | `docs/PRIOR_ART.md` lines 468 to 813, "Session 2 brief" (S0 to S10, notes N12 to N18, searches 41 to 82) |
| WP-A lifetime map | numerics build, then runner | **sweep and supplement done and committed** (100 long runs; `lifetime_table.json`, `lifetime_table_v2.json`). Lifetime rises with beta, no fall to beta = 5. Longest-lived starts: T_death 868 (0.75), 3380 (0.85), 29,404 (1.0); alive at 30,000 from 1.15 up. To T = 300,000: beta 4/3 from omega 0.906505 keeps 0.9946 (rate 4e-9 rising to 1.7e-8); 1.6: 0.9884; 2.0: 0.9882; 3.0: 0.969 and 0.984; 5.0: 0.969 and 0.998 (rate zero within error). Doubled box at 4/3 to 300,000: 5.3e-6. Predictions: A1 miss (F-007); A2 hit to the cap; A2-CD not supported; A3 misses at 0.75, 1.15, and 1.0 in the supplement; A4 hit; A5 untested; A6 miss (longer than predicted); A7 hit; A8 miss (T_90 at matched size is not monotone; T_death is). Ledger entries F-010 to F-012 being written. 3D confirmations: A9 (beta 4/3, omega 0.9065, T = 3000) hit at 7.0e-5; A10 (beta 5, T = 1000) hit at 2.4e-5 | `outputs/session2/wpA/`, `docs/WPA_BUILD_NOTES.md` |
| WP-B 3D primary to T = 5000 | runner | **done**. B1 hit (max difference from radial 7.26e-5 over T <= 5000; frac 0.9755 at 5000). B2 hit by symmetry. F-016: my "0.956" was wrong | `outputs/session2/wpB/` |
| WP-C two-body law | numerics build, then cloud | **Batch done and committed**: 24 AWS results in `outputs/session2/wpC/results/`, analysis in `analysis/criteria.json`. Hits: C1 (sign, all 12), C3 (kappa 0.4172 against 0.416), C5a, C5c (phase at the end 215, 160, 107 against beyond 180, 160, 102), C5d, C6, C7 (contact 126.8 and 347.8 against 126 and 346; merged pair keeps 0.76), C8, C9, C10 (no bonded pair), C12, C13. Misses: C2 on the sliding-force measure for d16_0, d16_45, d20_0; C4 at d = 16; C5b at d = 24 (and C5-CD with it); C11. Stop condition on the force sign: not triggered. Reviewer recomputed the verdict from saved data and agrees (law holds for d >= 3.6 hw; one unexplained excess in the 45-degree pair at d = 16). -theta single-lump runs explain C5b at d = 24. Ledger F-013 to F-015. Local Mac copies of the d = 16 runs are in `results_mac/` | `outputs/session2/wpC/`, `docs/WPC_DERIVATION.md`, `docs/WPC_BUILD_NOTES.md` |
| WP-D arrangement maths | numerics | **done with the derived law and merged** (f049374); reviewed; rerun with the measured law in `outputs/session2/wpD_measured/` (same balances, times 1 to 3% shorter) (`python -m swirl.arrange.scan --params <new>.json --out outputs/session2/wpD`, about 100 s). Result: 595 patterns (312 classes of 0/180, 283 stepped), 28 force balances, none stable once phases and N move. CD's two numbers confirmed (5-ring 72-degree steps: kappa R 1.0695, radially restoring, 4 negative modes of 15, spacing 0.90 hw; octahedron around anti-phase centre: 1.591, not restoring). 27 of 28 balances lie below 3.6 hw where the tail law fails. Point-model evolution: stepped rings merge 1.2 to 2 times later than in-phase controls | `outputs/session2/wpD/` (candidates.json, predictions_point_model.json, tables.md), `docs/WPD_NOTES.md` |
| WP-E arrangement 3D tests | numerics tooling, then runner and cloud | **done**. 14 arrangements, twin, two checks, plus the second runs of E11. Nothing bonded: stepped rings below 90 degrees merge (5-ring: contact 296 against 160 for the in-phase control), alternating and above-90 rings fly apart, 90-degree rings exchange N and break up, Claude Desktop's two balance points are not distinct lumps in 3D, and E11 (6-ring around an opposite-phase centre) merged at t = 699 (720 on a finer grid). The criterion I wrote labels the far-apart control bonded; no conclusion rests on that label. Ledger F-017 to F-019 | `outputs/session2/wpE/`, `docs/WPE_BUILD_NOTES.md` |
| WP-F review and report | reviewer, scholar, orchestrator | **done**. Reviews of all four builds, the pair verdict, the arrangement outcomes and the draft report; ledger F-007 to F-019; novelty check (Addendum F); report with Proposals for Step 4 | `docs/SESSION_02_REPORT.md`, `docs/FAILURES.md`, `docs/PRIOR_ART.md` |

## Running now

Nothing. Checked 2026-10-02 10:10 (+07): no engine, batch or test processes on the Mac.

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
