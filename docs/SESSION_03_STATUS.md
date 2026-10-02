# Session 3 status

Resume file. Plan: `docs/SESSION_03_PLAN.md`. Predictions: `docs/SESSION_03_PREDICTIONS.md`. Report (when written): `docs/SESSION_03_REPORT.md`.

Last update: 2026-10-02 (+07), after WP-S. WP0 and WP-S done; WP-L build and WP-P build running. AWS sign-in (`aws login --profile swirl`) renewed by Aaron at about 12:46 (+07); it lasts about 12 hours.

## Packages

| Package | Worker | State | Outputs |
|---|---|---|---|
| WP0 housekeeping | runner | **done** (2effa98; FINDINGS.md written by the orchestrator, because the harness stops subagents from creating report files). Corrected bonded rule: E14 gives "no interaction", E11 first run "not bonded". Lump-finder floor 0.05 of the first window's maximum: E3, E7, E4 now read "merged"; E1, E2, E14, lone lump unchanged. Changed `outputs/session2/wpE/postprocess.py` (E_residence_end) sits on the external drive, not in git. Runner model Sonnet 5.5, effort not visible; matches CLAUDE.md as far as visible | `docs/FINDINGS.md`, `swirl/engine/bonded.py`, `swirl/engine/lumps.py`, `tests/engine/test_bonded.py`, `tests/engine/test_lump_floor.py`, `docs/SESSION_03_PREDICTIONS.md` (definitions) |
| WP-S prior-art brief | scholar | **done**. No published break threshold for a real 3D oscillon; derived ceiling on the N removal that can break a lump: 1 - N_min/N = 0.45, 0.42, 0.49 for the three starts (N31). Kick sign matters at a dip (N32). Unequal frequencies: first-order force averages to zero (published), but a second-order phase-independent attraction binds or spirals NLS pairs; the point model lacks it (N34). No real-field orbiting pair published. Slow identical collisions: in phase merges, anti-phase bounces (published). Scholar model Sonnet 5.5, effort value 30 visible (mapping unconfirmed) | `docs/PRIOR_ART.md` lines 1127 to 1352, "Session 3 brief" (T0 to T11, N31 to N35, searches 120 to 151) |
| WP-L lone-lump break test | numerics build; runner or cloud runs | L1/L2 build running | |
| WP-P binding search, unequal lumps and orbits | numerics | P1 build running | |
| WP-F review and report | reviewer, scholar, orchestrator | waits for all | |

## Running now

Nothing.

## Rented compute

Nothing running on AWS. Ledger total 9.69 USD of 50 at the start of the session (the plan quotes 40.31 remaining).

## Repository layout found at the start

`outputs/` and `checkpoints/` in the main checkout are symlinks (made 2026-10-02 11:41) to `/Volumes/BokehSSD/BokehData/archive/home/Desktop/SWIRL/`. Every tracked file under `outputs/` is present through the link (checked: 0 of 620 missing), but git shows them all as deleted and cannot add files behind a symlink. Consequences and handling:
- Never run `git add -A`, `git add .` or `git commit -a` in the main checkout: that would commit the deletion of every tracked result. Stage explicit paths only.
- New result JSON and plots written under `outputs/` land on the SSD and cannot be committed while the link stays. Asked Aaron; until he answers, results stay on the SSD and their paths are listed in the report (deviation from the CLAUDE.md rule "commit result JSON and plots").
- numerics worktrees have their own real `outputs/` directory from git; results made there are committed on the worktree branch as usual.
