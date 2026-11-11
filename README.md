# Swirl Theory

Research by Aaron Choi Ramos, an independent researcher in Bangkok.

Swirl Theory is a proposal for unification. Its core claim is that there is no matter separate from energy. Space is an elastic medium, and everything in it is structured movement of energy: either waves that travel, or waves trapped in place, called swirls. Trapped energy contracts the medium. Contracted medium is denser space, denser space is a gravity well, and mass is that attraction.

This repository is the working record of the attempt to turn that picture into mathematics and test it by computation. It holds the plans, the code, the results, the failures and the corrections, in the order they happened.

## Start here

| If you want | Read |
|---|---|
| What the theory says, in 23 rules | `docs/SWIRL_RULES_2026-11-11.md` |
| How the work got to where it is | `docs/HISTORY.md` |
| The current plan | `docs/SESSION_06_PLAN.md` |
| What has been established | `docs/FINDINGS.md` |
| What missed, and why | `docs/FAILURES.md` |
| What others have already published | `docs/PRIOR_ART.md` |
| Lessons from the first study | `docs/LESSONS_2026-10-01.md` |
| Source materials | `docs/research/README.md` |

## Where the work stands (2026-11-11)

- Five sessions of the restart are complete. Session 6 is planned and not yet run.
- Sessions 1 to 5 tested two ways of trapping a wave that turned out not to be the one in the theory. A swirl trapped below a lowest note of the medium is real and long-lived, but no two of them were ever found to bond. A swirl trapped above a highest note holds without loss on a grid, but not in a continuous medium.
- The theory's own trap is different: the wave pulls the surrounding medium to its elastic limit and so builds its own wall. No session has tested that yet. Session 6 is the first.
- Nothing in this repository is a confirmed result about nature. It is a record of a theory being tested, including where it failed.

## Layout

| Path | Contents |
|---|---|
| `docs/` | Plans, reports, predictions, the rule sheet, and the ledgers |
| `docs/research/` | Source materials and where to find them |
| `swirl/` | Solver and analysis code for the restart |
| `tests/` | The test suite |
| `outputs/` | Result files and plots from every session |
| `legacy/` | A frozen copy of the 2026-04 study, kept as reference |
| `SESSION_04_PLAN.md`, `SESSION_04_REPORT.md` | Session 4's plan and report sit at the root |
| `CLAUDE.md` | Working rules for the sessions that run the computations |

## How the work is done

Each session has a plan written before it runs, predictions recorded before the results are seen, an independent review, and a report that gives misses the same depth as hits. Planning and interpretation are done by Aaron with Claude. The computations are run by Claude Code on a Mac, following the rules in `CLAUDE.md`.

## History in git

- `main` carries the whole history: the 2026-04 study first, then the restart from October 2026.
- The branch `legacy-2026-04` and the tags `legacy-2026-04-final` and `frozen-v9-2026-04` mark the end of the first study.
- The tags `session-01` to `session-05` mark the commit that closed each session's report.

## The first study

The earlier work, "Geometric Selection Rules for Multi-Oscillon Cluster Stability" (April 2026), is in `legacy/`. A review in October 2026 found that its central results did not describe bound structures. That review is `docs/LESSONS_2026-10-01.md`. The study is kept because it is how the present work began.
