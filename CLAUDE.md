# SWIRL

Restart of the Swirl Theory study. The plan is `docs/SWIRL_PLAN_2026-10-01.md`. The record of past mistakes is `docs/LESSONS_2026-10-01.md`. Read both before planning work.

## Layout

- `legacy/`: the frozen 2026-04 repo. Read-only reference.
- `swirl/`: new code. `tests/`: pytest. `outputs/`: result JSON and plots.
- `docs/`: plan, lessons, reports, `PRIOR_ART.md`, `FAILURES.md`, `CLOUD_LEDGER.md`.

## Who does what

The main session is the orchestrator (Fable). It plans, delegates, judges results against the predictions on record, and writes the reports. It normally does not write solver code itself.

| Worker | Model, effort | Work |
|---|---|---|
| `numerics` | Opus, high, own worktree | Solvers, the engine, diagnostics |
| `runner` | Sonnet 5.5, medium | Setup, audits, sweeps, benchmarks, long local runs, plots, tables |
| `reviewer` | Sonnet 5.5, xhigh, read-only | Independent check of a finished diff or result |
| `scholar` | Sonnet 5.5, xhigh | Prior-art briefs, the failure ledger, novelty checks |
| `cloud` | Sonnet 5.5, high | Rented AWS compute: launch, run, collect, shut down |

A good delegation names the goal, the files to read first, the checks that define done, and what to return. Workers return numbers, file paths and open issues, not file contents. Run independent packages in parallel. On the first delegation to each worker, confirm from the task list that it runs the model and effort above, and note any mismatch in the report.

## How to treat the rules

These are guidelines with reasons, written so that nobody has to stop over a technicality.

- **Hard stops, ask Aaron first:** pushing to any remote; cloud spend beyond the working budget below; changing the AWS plan, spend limit or advanced features; adding a dependency beyond numpy, scipy, numba, matplotlib and pytest; writing outside this folder; editing `legacy/`.
- **Everything else is judgment.** If a guideline blocks sensible progress, take the reading that best serves the plan's question, note the deviation and the reason in the report, and continue.
- **Tolerances are targets.** Report the measured value. Within about 3 times the target is a pass with a note. Beyond that is a miss.
- **Scope guard.** Work should trace to a step in the plan. Other ideas go in the report's parking lot instead of becoming tasks.

## Failures count as much as successes

There are two kinds, and they are handled differently.

- **A failing code check** is an engineering problem. Two attempts to fix it, then log it as open with a diagnosis and move to work that does not depend on it. Stop only if everything left depends on it.
- **A missed target or a failed prediction** is a result. Do not fix it or rerun to rescue it. Investigate it:
  - what was expected and what was measured;
  - candidate causes, ranked;
  - up to three cheap diagnostics run now to separate them, with expensive ones listed as proposals;
  - what the miss rules in and what it rules out.
- `scholar` records every miss in `docs/FAILURES.md` and checks whether published work has already seen and explained it.
- Reports give hits and misses the same depth. `reviewer` checks that they did.

## Standards for every run

1. Regime check first: omega relative to m, and the nonlinear share of the energy.
2. Stability means localization: energy inside a radius over time, never binding energy alone.
3. The energy function includes every term in the equation of motion.
4. Absorbing boundaries, validated against a box twice the size.
5. Every nonlinear run has a linear twin (g4 = g6 = 0).
6. Analytic or t = 0 value before the PDE.
7. Symmetry classes use rotations, reflections and the field sign flip.
8. Structural quantities come from coordinates, not labels.
9. Each prediction on record sits next to what the literature expects, with a source.
10. No claim that something is new without a documented search.

## Compute

- One environment at `<repo root>/.venv`. From a worktree, call its Python by absolute path.
- Parallel workers = performance cores. Set `NUMBA_NUM_THREADS=1` before importing numba in multiprocessing runs.
- Write outputs atomically (`.tmp` then `os.replace`). Every result JSON carries its full parameters.
- Anything over 5 minutes checkpoints and can resume. Anything over 30 minutes runs detached with a log, under `caffeinate` so the Mac stays awake.
- Commit code, result JSON and plots. Never commit field arrays, checkpoints or the environment.

## Rented compute

- Available only once `docs/AWS_REPORT.md` exists. It names the AWS profile and shows the core quota.
- Working budget: 50 US dollars in total, pre-approved. The orchestrator decides when renting is worth it and delegates to `cloud`. No approval is needed inside the budget.
- Every dollar goes in `docs/CLOUD_LEDGER.md`. At 40 dollars spent, tell Aaron in the report. At 50, stop renting.
- Use the Mac for anything that finishes locally in under about an hour.
