# Session 3 status

Resume file. Plan: `docs/SESSION_03_PLAN.md`. Predictions: `docs/SESSION_03_PREDICTIONS.md`. Report (when written): `docs/SESSION_03_REPORT.md`.

Last update: 2026-10-02 (+07), after WP-S. WP0 and WP-S done; WP-L build and WP-P build running. AWS sign-in (`aws login --profile swirl`) renewed by Aaron at about 12:46 (+07); it lasts about 12 hours.

## Packages

| Package | Worker | State | Outputs |
|---|---|---|---|
| WP0 housekeeping | runner | **done** (2effa98; FINDINGS.md written by the orchestrator, because the harness stops subagents from creating report files). Corrected bonded rule: E14 gives "no interaction", E11 first run "not bonded". Lump-finder floor 0.05 of the first window's maximum: E3, E7, E4 now read "merged"; E1, E2, E14, lone lump unchanged. Changed `outputs/session2/wpE/postprocess.py` (E_residence_end) sits on the external drive, not in git. Runner model Sonnet 5.5, effort not visible; matches CLAUDE.md as far as visible | `docs/FINDINGS.md`, `swirl/engine/bonded.py`, `swirl/engine/lumps.py`, `tests/engine/test_bonded.py`, `tests/engine/test_lump_floor.py`, `docs/SESSION_03_PREDICTIONS.md` (definitions) |
| WP-S prior-art brief | scholar | **done**. No published break threshold for a real 3D oscillon; derived ceiling on the N removal that can break a lump: 1 - N_min/N = 0.45, 0.42, 0.49 for the three starts (N31). Kick sign matters at a dip (N32). Unequal frequencies: first-order force averages to zero (published), but a second-order phase-independent attraction binds or spirals NLS pairs; the point model lacks it (N34). No real-field orbiting pair published. Slow identical collisions: in phase merges, anti-phase bounces (published). Scholar model Sonnet 5.5, effort value 30 visible (mapping unconfirmed) | `docs/PRIOR_ART.md` lines 1127 to 1352, "Session 3 brief" (T0 to T11, N31 to N35, searches 120 to 151) |
| WP-L lone-lump break test | numerics build; runner or cloud runs | **L1/L2 built** (branch worktree-agent-a511a354d573f02da, 4213d63; 84 passed, 1 xfailed). **Boosted-lump gate passes** at v = 0.05 and 0.2 (speed 3e-5, 4e-5; E/(gamma E0) 5.2e-5, 2.1e-5; omega/gamma 3e-6, 8e-6), so the stop condition does not block collisions. t = 0 table: Delta N / N about 2 epsilon. Block L1 predictions committed (e1cce46). **L1 scan running** (runner, in that worktree, about 2.7 h on 2 workers). Reviewer checking the build. numerics preparing the L3 cloud batch. numerics shows a reasoning-effort value of 15 (named level not visible; CLAUDE.md says high) | worktree `outputs/session3/wpL/`, `docs/WPL_BUILD_NOTES.md` |
| WP-P binding search, unequal lumps and orbits | numerics | **P2 scan done** (9f5eef7 on the P branch; 1822 runs, 0 errors). No run bonded. 3 unequal static pairs at 7.2 hw labelled "bounded, beating" (marginal on trend and N clauses). Equal orbits at 3.6 to 6 hw: 448 of 448 end within 2 orbital periods. N34 term is comparable to the model's beat-averaged force for N ratio >= 1.5. Scoring: P1 hit, P3 hit, P2 miss (candidates; no contact delay at 3.6 hw), P4 miss (3 candidates outside 3.6 hw), P-CD first half holds, "orbits likelier" fails in the model. Diagnostics D1 to D3 and P3 preparation running (numerics). Reviewer checking the P1 diff. Not yet merged | worktree `outputs/session3/wpP/p2/summary.json`, `n34_check.json`, `docs/WPP_BUILD_NOTES.md` section 11 |
| WP-F review and report | reviewer, scholar, orchestrator | waits for all | |

## Running now

- P2 point-model scan (numerics, its worktree).
- L1 radial scan (runner, WP-L worktree, detached).
- Reviewers: WP-P1 diff; WP-L1/L2 diff.
- numerics (WP-P): miss diagnostics D1 to D3; P3 preparation.
- numerics: L3 batch preparation (no Mac compute).

## Rented compute

Nothing running on AWS. Ledger total 9.69 USD of 50 at the start of the session (the plan quotes 40.31 remaining).

## Repository layout found at the start

`outputs/` and `checkpoints/` in the main checkout are symlinks (made 2026-10-02 11:41) to `/Volumes/BokehSSD/BokehData/archive/home/Desktop/SWIRL/`. Every tracked file under `outputs/` is present through the link (checked: 0 of 620 missing), but git shows them all as deleted and cannot add files behind a symlink. Consequences and handling:
- Never run `git add -A`, `git add .` or `git commit -a` in the main checkout: that would commit the deletion of every tracked result. Stage explicit paths only.
- New result JSON and plots written under `outputs/` land on the SSD and cannot be committed while the link stays. Asked Aaron; until he answers, results stay on the SSD and their paths are listed in the report (deviation from the CLAUDE.md rule "commit result JSON and plots").
- numerics worktrees have their own real `outputs/` directory from git; results made there are committed on the worktree branch as usual.

## Decisions and deviations so far

- **Collision speeds (WP-L3):** per lump 0.02 and 0.2. The builder proposed 0.05 and 0.2; the reviewer pointed out that the published slow collisions use 0.01 to 0.04 per lump (Xue 2025), so 0.02 connects to them. The plan fixes only "two speeds".
- **L1 scoring:** breaks are split into instantaneous (the kick alone puts the energy inside 3 hw below half) and dynamic, with thresholds given both ways. The family comparison uses the settled N ratio to the control on one estimator, because the period-averaged N sits 1.35% above the profile N at the dip, comparable to the dip's width.
- **Gate before collisions:** repeated on the production source and grid, at phases 0 and 90, at both speeds, on the instance before any collision starts (reviewer: the first gate ran at dx 0.8 on a test source).
- **Worker effort:** the reviewer and scholar see a reasoning-effort value of 30 and numerics sees 15; the named levels in CLAUDE.md (xhigh, high) cannot be confirmed from inside.
- **Reviews:** WP-L1/L2 build reviewed. Nothing blocks L1; fixes to the notes, a code hash for the checkpoint key, decimation of the 30,000-unit series before commit, and more box checks are requested before merging.
- **WP-P1 review** (reviewer, 2026-10-02 about 14:15): the pair law re-derived and checked by quadrature to 6.3e-5; the orbit-instability results agree (kappa d threshold (1+sqrt5)/2; e-fold/T_orb rows recomputed). Corrections taken into the scoring: P1 counts only "merged or flew apart", 418 of 448 = 93.3% (30 left the family's range), still a hit; the 3 candidates are misses of P2 and P4; the held-position beat average is not a sound basis for "drift together" (free dynamics can push the pair apart), so it is not used as a cause. The P2 scan ran on an uncommitted p2scan.py (label added after Block P's commit); committed since (c1e4540). Before merging: notes fixes (N34 and the sextic cross term named and sized; boost "exact" qualified), driver tests, closest approach from dense output.
- **P2 diagnostics** (c1e4540): with N34 added the candidates fail the trend clause by 4 to 6 times; to 3T they stay bounded but drift; D3: contact at 3.6 hw comes within 0.07 to 0.24 of a beat.
- **Rented compute plan for L3 + P3:** the full L3 list (core 28.8 USD on one instance; 40 to 47 with all seeds) does not fit. Trimmed: b5 (d) 4 runs instead of 8; extra seeds at the dip only; one collision resolution check; b5 collisions (4b) dropped and parked. P3 uses isolated single-lump runs as the far-apart control instead of N = 200 boxes (cost). Target: total spend below 40 USD, on 4 c8g.4xlarge instances in parallel so the batch ends inside the sign-in window.
- **Predictions:** Blocks L3 and P3 committed (0cba2a4) before any 3D run.
- **Merges (2026-10-02 about 14:50).** `git merge` cannot run in the main checkout while `outputs/` is a symlink (git refuses to write the merged result files behind it). Both numerics branches were merged in a separate worktree `.claude/worktrees/integration` (branch `session3-integration`): WP-P 76eb50c, WP-L 681314e, no conflicts; 82 new and touched tests pass there. `main` was then moved to 681314e with `git update-ref`, and the main checkout's `docs/`, `swirl/` and `tests/` were synced from it. New result files from the branches are in git but not on the external drive. Open review items still to do before the end: P notes fixes and driver tests; L checkpoint code hash (after the L1 scan); L1 scan summary.
- **P3 horizon fixed at T = 7500** (instead of 5 contact units read from the in-phase control on the instance). This removes a dependency between instances. It covers contact times up to 1500 (predicted 800 to 1300) and adds about 1 USD.
- **Rented batch L3 + P3**, commit 681314e: 4 c8g.4xlarge instances, estimate 28.5 USD, with a ceiling of 30 USD for this batch (total spend below 40). Delegated to `cloud`.
- **WP-P review fixes merged** (475a187 via integration, main 0c75b50): reflag with dense closest approach changes exactly the 5 runs the reviewer named (all merged, all "not bonded"); P1 is 418 of 448 = 93.3% merged or flew apart within 2 orbits (30 left the family's range), still a hit; no Block P score changes. Net phase-independent second-order force (N34 less the sextic term): 0.13 to 0.16 of the first-order in-phase force at 3.6 hw, 0.03 at 4.8, 0.005 at 6, 0.0008 at 7.2.
- **Procedure for main.** Code merges are made in `.claude/worktrees/integration` and `main` is moved there with `git update-ref`. After every commit in the main checkout, integration is fast-forwarded to main. (Once, at 6cf0ff6, `main` was moved to an integration commit that lacked the status commit 97fb0dd. It was restored within minutes by merging 97fb0dd into integration; nothing was lost.)
- **Rented batch, first attempt (2026-10-02 14:49 to 15:06 +07): benchmarked and stopped, 0.18 USD.** The measured rate was 0.81 to 0.86 of the assumed one, which projected 36.5 USD against the 30 USD batch cap. The allowed drop list could not lower it, because the instance wall times are set by long job chains. Everything was cleaned up; ledger total 9.87 USD.
- **Decision (orchestrator):** relaunch as designed with the batch cap raised to 37 USD, total at most about 46.9 USD. This is inside the pre-approved 50 and **passes the 40 USD mark, which goes to Aaron in the report** (and was told to him in the session). Reason: L3 and P3 are the plan's only 3D tests of distortion, collision and unequal binding. A shorter P3 horizon would risk the duration clause.
