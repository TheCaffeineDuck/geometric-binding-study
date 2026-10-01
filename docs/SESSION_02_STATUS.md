# Session 2 status

Resume file. Updated by the orchestrator after each package. Plan: `docs/SESSION_02_PLAN.md`. Predictions: `docs/SESSION_02_PREDICTIONS.md`.

Last update: 2026-10-01 21:45 (+07), WP0 done; S, A build, B, C build running.

## Packages

| Package | Worker | State | Outputs |
|---|---|---|---|
| WP0 housekeeping | runner | **done**. Effort values "5" and "30": the Claude Code docs list only named levels (low, medium, high, xhigh, max) and give no numeric scale, and no documented way for the parent session to confirm a subagent's effort after the fact. So the mapping stays unconfirmed. The WP0 runner reported model `claude-sonnet-5-5`, effort not visible | Errata in `LESSONS_2026-10-01.md`; pointer line in `SWIRL_PLAN_2026-10-01.md`; reporting rule in `.claude/agents/numerics.md`; `AWS_REPORT.md` re-check: quota now 64 vCPU, request closed, 120 USD credits, nothing left in us-east-2 |
| WP-S prior-art brief | scholar | started | `docs/PRIOR_ART.md`, "Session 2 brief" |
| WP-A lifetime map | numerics build, then runner | build started | `outputs/session2/wpA/` |
| WP-B 3D primary to T = 5000 | runner | **running detached**, started 22:28, expected finish about 02:40. Collection and the isotropy check (B2) still to do | `outputs/session2/wpB/` |
| WP-C two-body law | numerics build, then runner or cloud | build started | `outputs/session2/wpC/`, `docs/WPC_DERIVATION.md` |
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

- Predictions Block 1 (WP-A, WP-B, WP-C phase control) is written. Block 2 must be written before any pair run, Block 3 before any arrangement run.
- numerics workers commit on their worktree branches. The orchestrator merges to `main`.
