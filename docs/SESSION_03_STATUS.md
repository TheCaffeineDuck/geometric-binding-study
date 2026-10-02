# Session 3 status

Resume file. Plan: `docs/SESSION_03_PLAN.md`. Predictions: `docs/SESSION_03_PREDICTIONS.md`. Report (when written): `docs/SESSION_03_REPORT.md`.

Last update: 2026-10-02 12:50 (+07). Session started. AWS sign-in (`aws login --profile swirl`) renewed by Aaron at about 12:46 (+07); it lasts about 12 hours.

## Packages

| Package | Worker | State | Outputs |
|---|---|---|---|
| WP0 housekeeping | runner | starting | |
| WP-S prior-art brief | scholar | waits for WP0 | |
| WP-L lone-lump break test | numerics build; runner or cloud runs | waits for WP0 | |
| WP-P binding search, unequal lumps and orbits | numerics | waits for WP0 | |
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
