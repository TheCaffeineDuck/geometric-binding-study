# Failure ledger

A missed target or a failed prediction is a result, not a bug. Every miss gets an entry here, with the same depth as a hit. The `scholar` worker writes the entries. The orchestrator supplies the expected value, the measured value, the ranked causes and the diagnostics. The scholar adds the literature check.

A failing code check is a different thing (an engineering problem, handled by the rules in `CLAUDE.md`) and does not go here unless it turns into a missed prediction.

Status: empty. No misses recorded yet. Session 1 has not produced results.

## Entry format

Entries are appended in order. Newest last. Each entry uses this template.

```
### F-NNN  Short title
- Date, session, work package, step:
- Prediction on record (with the page and section it comes from):
- Expected value or outcome, with its tolerance:
- Measured value or outcome, with parameters (beta, omega, N, grid, box, T) and the result file path:
- Size of the miss against the tolerance (target, within about 3 times, beyond):
- Ranked causes (orchestrator): 1 ... 2 ... 3 ...
- Diagnostics run now (up to three cheap ones), what each returned:
- Diagnostics proposed (expensive ones), not run:
- What the miss rules in, and what it rules out:
- Literature check (scholar): has published work already seen and explained this? Sources with links and tags
  (READ, ABSTRACT, SECONDHAND) and the searches run. Write "no source found" where that is the case.
- Cross-reference: PRIOR_ART.md section, SESSION report section, reviewer finding if any:
- Status: open | explained | rerun not needed | superseded
```

## Rules for entries

- No rerunning to rescue a result. The entry records what happened and why.
- Numbers carry their parameters. A number from a different potential, dimension or amplitude regime is context, not a prediction, and is labelled so.
- Literature claims follow the evidence rules in `PRIOR_ART.md`: a link and one tag per claim, and "no source found" when nothing turned up.
- If a miss was already anticipated in `PRIOR_ART.md` (for example, the unstable-branch radial run settling onto the stable branch instead of dispersing, or a Step 2 energy retention below 90% near a radiation peak), the entry says so and links the section.

## Entries

None yet.
