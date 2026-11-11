# History of the work

A dated timeline of what was done, what was decided and what was found. Each entry names the file it comes from. Misses and corrections are included. Later entries override earlier ones where they disagree.

## Before the restart

**2023-06-24. The theory is first written down.**
Aaron writes the Aether-Aatheron theory, later called Swirl Theory: space is a medium, and matter is the medium in a different configuration.
Source: the original theory document (see `docs/research/README.md`).

**2026-03-16 to 2026-04-07. The first numerical study.**
A study of clusters of localized waves placed on the corners of regular solids, and a paper draft, "Geometric Selection Rules for Multi-Oscillon Cluster Stability" (version 9). The last commit of this study is `8a62dd2`.
Source: `legacy/README.md`, `legacy/paper/paper_draft_v9.md`. Tags: `frozen-v9-2026-04`, `legacy-2026-04-final`.

## The restart

**2026-10-01. The first study is found not to hold.**
A review of the paper and its data found:
- The localized waves were not trapped. They were free wave packets.
- The "binding energy" was interference between overlapping packets, fixed at the start of each run.
- The variance rule presented as a many-body effect was a sum over second-neighbour pairs.
- The soliton claimed in the related NEST model does not exist.

The paper was shelved and the NEST term retired. A restart was planned: find a real self-trapped swirl first, then test bonding.
Source: `docs/LESSONS_2026-10-01.md`, `docs/SWIRL_PLAN_2026-10-01.md`.

**2026-10-01. Session 1: a real self-trapped swirl exists.**
At one setting of the potential (beta = 4/3) a swirl pulsing at 0.909 of the medium's lowest note survives in full three dimensions, keeping 0.994 of its energy where its linear twin keeps 0.004. Other settings tried did not survive.
Source: `docs/SESSION_01_REPORT.md`. Tag: `session-01`.

**2026-10-02. Session 2: no arrangement of swirls bonded.**
Fourteen arrangements were tested. None stayed bonded. The one still intact after its first run merged in its second. The session also found a narrow range of pulse rates where a lone swirl lives very long.
Source: `docs/SESSION_02_REPORT.md`. Tag: `session-02`.

**2026-10-02 to 2026-10-05. Session 3: how hard a lone swirl is to break.**
In the spherical solver a lone swirl breaks outright only when a disturbance removes about 40% of its energy. Smaller disturbances change its lifetime, by an amount that depends on where it sits in its family.
The rented cloud project was closed on 2026-10-05. The last cloud batch was not copied first and its results are lost. Spending reported by the provider was about 125 US dollars against a budget of 50. All work since has run on the Mac.
Source: `docs/SESSION_03_REPORT.md`, `docs/CLOUD_LEDGER.md`. Tag: `session-03`.

**2026-11-09 to 2026-11-10. Session 4: the force between two swirls, and the gap.**
- The force between two swirls falls off with distance and follows the known short-range law. In step attracts, out of step repels, and there is no distance at which the pair is held. A constant, confining force is excluded above about 3e-4.
- A lone swirl radiates only at odd multiples of its pulse rate, all above the medium's lowest note. Nothing below that note travels.
- Predictions: 3 hits, 3 split, 2 misses.

Correction: the far-distance check was rerun in a box 1.33 times larger, not 1.5 times as the message of commit `9bd140c` says, and not the 2 times the rules ask for. Box reflection is disfavoured, not ruled out.
Source: `SESSION_04_REPORT.md` (at the repository root). Tag: `session-04`.

**2026-11-10. The model changes.**
Aaron sets out that the medium has a direction of motion at every point, that its elastic limit is an asymptote, and that a swirl is trapped because it is too short and too fast for the medium to carry, the way a tight pluck on a string at its limit stays where it was plucked. This is the opposite of the trap Sessions 1 to 4 tested.
Source: `docs/SESSION_05_PLAN.md`, section 1.

**2026-11-10. Session 5: the pluck on a string, and a comparison.**
- On a string with a grid, the trap works and does not leak. The S-bend (two halves moving opposite ways around a still center) ran unchanged for 32,000 cycles at every rate tested. A tighter vibration is faster.
- The single bump is unstable on the string and turns into an S-bend.
- Without a grid, no smooth trapped vibration was found.
- In the comparison track, a limit on the rate of motion slowed dispersal but trapped nothing. The pinch holds for 13,105 cycles in the old equation.
- Eight predictions missed or split on the string track, and several on the comparison track. The final review required 44 text corrections; the corrected text was not re-reviewed in that session.

Source: `docs/SESSION_05_REPORT.md`. Tag: `session-05`.

**2026-11-11. The rules of the medium are written down.**
After several mismatches between Aaron's picture and the models being tested, the theory was set out as 23 rules and agreed line by line. The main points that had not been in any model:
- A swirl has a fixed length. When it bows, its ends are drawn in, and that pull is always inward.
- The pull stretches the surrounding medium to its elastic limit, where it goes rigid. That rigid shell is the wall that holds the wave. The medium is continuous and needs no grid.
- The same pull, spread over a growing surface, is the gravity well.
- Time is cycles. An event takes longer when its waves must cross more medium, whether because space is denser or because the structure is moving.

Source: `docs/SWIRL_RULES_2026-11-11.md`.

**2026-11-11. Session 6 is planned.**
The first test of that medium, using the pinch, which can be solved along one radial line while the medium is fully three-dimensional. Predictions on record include two places where standard elastic theory is expected to disagree with the rules.
Source: `docs/SESSION_06_PLAN.md`.

**2026-11-11. This repository becomes the public record.**
The restart, which had never been pushed, was joined to the history of the first study in commit `2e67086` and pushed. From this date every session pulls at its start and pushes at its end.
Source: `CLAUDE.md`, git history.

## Which model orchestrated each session

`CLAUDE.md` names Fable as the orchestrator. In practice:

| Session | Orchestrator | Source |
|---|---|---|
| 1 | Opus 5.5 | `docs/SESSION_01_REPORT.md` |
| 2 | Fable | `docs/SESSION_02_REPORT.md` |
| 3 | Opus 5.5 | `docs/SESSION_03_REPORT.md` |
| 4 | Opus 5.5 | Aaron, 2026-11-11. Not recorded in the report. |
| 5 | Opus 5.5 | Aaron, 2026-11-11. Not recorded in the report. |

The final text reviews of Sessions 4 and 5 required 20 and 44 corrections. Those sessions were also larger than the earlier ones, so this is not a clean comparison between models.
