# Session 2 status

Resume file. Updated by the orchestrator after each package. Plan: `docs/SESSION_02_PLAN.md`. Predictions: `docs/SESSION_02_PREDICTIONS.md`.

Last update: 2026-10-01 23:25 (+07). WP0 and WP-S done; WP-A build merged, sweep running; WP-B running; WP-C build merged, pair batch going to AWS.

**Stop-condition note (plan section 4).** The phase control missed at theta = 90 only. The plan's control is over the phases used, and the redesigned batch uses none that fails, so I read the stop as not triggered; the 3D gate (G1, G2 in the predictions file) must pass before pair runs start. WP-D and WP-E go ahead only if the gate passes and the measured force agrees with the derivation in sign. This reading is flagged to Aaron and goes in the report as a deviation.

WP-A sweep command (resumes if rerun), from the repo root:
`PY=.venv/bin/python; nohup caffeinate -i sh -c "$PY -m swirl.family.wpa families --workers 2 && $PY -m swirl.family.wpa maps --workers 2 && $PY -m swirl.family.wpa rank && $PY -m swirl.family.wpa long --workers 2" > logs/session2/wpa_sweep.log 2>&1 &`

## Packages

| Package | Worker | State | Outputs |
|---|---|---|---|
| WP0 housekeeping | runner | **done**. Effort values "5" and "30": the Claude Code docs list only named levels (low, medium, high, xhigh, max) and give no numeric scale, and no documented way for the parent session to confirm a subagent's effort after the fact. So the mapping stays unconfirmed. The WP0 runner reported model `claude-sonnet-5-5`, effort not visible | Errata in `LESSONS_2026-10-01.md`; pointer line in `SWIRL_PLAN_2026-10-01.md`; reporting rule in `.claude/agents/numerics.md`; `AWS_REPORT.md` re-check: quota now 64 vCPU, request closed, 120 USD credits, nothing left in us-east-2 |
| WP-S prior-art brief | scholar | **done**; orchestrator has read S0, S6, S7, S8 | `docs/PRIOR_ART.md` lines 468 to 813, "Session 2 brief" (S0 to S10, notes N12 to N18, searches 41 to 82) |
| WP-A lifetime map | numerics build, then runner | **build done and merged** (061fc83); sweep handed to runner; review and ledger entry in progress. Prediction A1 MISSED: path integral gives T_death 11,540 against 1145.5 (beta 0.85) and T_90 33,006 against 10,630 (beta 4/3). Cause found by diagnostics: gaps in the map's energy coverage; with the run's own Gamma(E) the integral gives 10,673 (+0.4%). Consequence: rankings are provisional, so the long stage runs two picks per beta (path-integral best, and lowest measured rate). Also missed: box doubling at beta 0.75, 3.29e-4 against 1e-4, after the lump has died | `outputs/session2/wpA/`, `docs/WPA_BUILD_NOTES.md` |
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

None launched this session. Ledger total 2.80 USD.

## Notes for a resuming session

From the WP-S brief, to carry into the predictions and into WP-D and WP-E:
- The tail law is published for the non-relativistic limit (Malomed 1998) and matches the Session 1 form. The N-transfer rate has the same prefactor with sin(dphi) (N14, derived).
- At 90 degrees expect repulsion with a large N split, not zero force (Burlak & Malomed 2018, 3D cubic-quintic NLS). Expect 45 degrees to merge.
- Patterns with a phase-exchanging mirror plane (0/180 pairs, alternating rings, chains) have no radiation-free bound state (N15). Use them as controls.
- Any static point-model balance with net attractive single-lump energy is unstable in some direction (N16). CD's 5-ring and octahedron numbers reproduce but sit at 0.90 and 1.6 half-widths.
- Literature candidates for WP-E: stepped rings with N = 5 or 6 (step 72 or 60 degrees) and N = 8 with two turns; controls: all in phase, 0/180 ring, step above 90 degrees, lumps far apart. Published clusters are complex fields, rotate, and last a few revolutions.
- WP-A: record omega_dip and omega_death against beta; a "longevity cliff" may sit near beta 1.6 to 2.1 (N17, low confidence).

- Predictions Block 1 (WP-A, WP-B, WP-C phase control) is written. Block 2 must be written before any pair run, Block 3 before any arrangement run.
- numerics workers commit on their worktree branches. The orchestrator merges to `main`.
