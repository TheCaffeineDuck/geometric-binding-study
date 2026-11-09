# Session 4 plan

Date: 2026-11-09 (written 11:15 +07, before any Session 4 run).
Orchestrator: this session. Brief from Aaron, 2026-11-09: Test A (two-body potential V(r)) and Test B (radiation spectrum of the lone oscillon), all compute on the Mac.
Governing files: `CLAUDE.md`, `docs/SWIRL_PLAN_2026-10-01.md` (standards, section 5), `docs/LESSONS_2026-10-01.md`. This file is the only plan file Session 4 writes. Results go in `SESSION_04_REPORT.md` in the SWIRL root; the resume file is `docs/SESSION_04_STATUS.md`.

## 0. State at the start

- **Session 3 report** (`docs/SESSION_03_REPORT.md`) exists and is complete (sections Verdict to 13). Lost to the AWS closure of 2026-10-05: the four beta = 5 distortion runs, the three 3D spot checks of radial break thresholds, the dip-start gates, six dip collisions and their twin, every collision box and resolution check, `P3_iso_N100` and `P3_inphase`. None of these is an input to Session 4.
- **`COMPUTE_GUIDELINES.md` does not exist** in the folder. I follow the compute section of `CLAUDE.md` and the conventions in the brief. **`Swirl_Theory.txt` and the paper draft are not in this folder** (the only paper is under the read-only `legacy/`); nothing here touches them.
- **`SexticEvolver`** is the legacy engine class (`legacy/engine/evolver.py`). The engine that produced every Session 1 to 3 result is its successor, `swirl/engine/evolver.py::Evolver` (3D), with the radial solver `swirl/family/radial.py::Radial`. Test B derives the dispersion relation from all three: the one named in the brief and the two that produced the data.
- **What is already known for Test A** (Session 2, `docs/SESSION_02_REPORT.md` section 5): at beta = 4/3, omega = 0.9091, the centroid force between equal lumps at d = 16, 20, 24 follows F = 2 pi A^2 cos(dtheta) e^(-kappa d)(kappa/d + 1/d^2) to 3 to 7%, with fitted kappa 0.4172 against 0.4165. That was a force at three separations; Test A adds the potential itself, a dense range out to the resolution floor, unequal lumps, explicit fits of competing forms and a static/dynamical comparison.
- **What is already known for Test B:** core omega = 0.9091 (Session 1 primary) and 0.9065 (the long-lived dip start), so omega/m = 0.909 and 0.907. Radial loss rates are measured (Sessions 1 to 3). No far-field spectrum has been recorded; the only spectral look at radiation is F-027's one-point probe in 3D after a distortion (power near omega 1.07 and little at 3 omega).

## 1. Units, parameters, regime

Set-B-like units: m = 1, g4 = 0.30, g6 = 0.09 beta. Linear threshold expected at omega = m = 1 (to be derived from the code in B1, not assumed).

| Point | beta | omega | omega/m | N/N_min | hw | A | kappa = sqrt(1 - omega^2) | Status |
|---|---|---|---|---|---|---|---|---|
| primary | 4/3 | 0.9091 | 0.909 | 1.754 | 3.344 | 56.70 | 0.4165 | Session 1 Go; 3D to T = 5000 |
| dip | 4/3 | 0.9065 | 0.907 | 1.83 | ~3.3 | from family | ~0.422 | radial alive at 300,000; 3D to 3000 |
| b5 | 5 | 0.9756 | 0.976 | 1.97 | 6.62 | from family | 0.2196 | radial alive at 300,000; 3D to 1000 |

These three are the points Sessions 1 to 3 established as long-lived self-trapped lumps (beta 1.15 to 3 were radial only, and are not used). Every run reports omega/m and the nonlinear energy share at t = 0 and on average (standard 1).

## 2. Test A: two-body potential V(r)

### A0. Definitions (fixed before running)

r is the centre-to-centre distance. dtheta is the relative phase at t = 0. Reference law (Session 2 derivation, `docs/WPC_DERIVATION.md`):

    U_tail(r) = -2 pi A^2 cos(dtheta) e^(-kappa r) / r        (2 pi A^2 = 20,200 at the primary)

For unequal lumps the reference is the Session 3 point-model law (`swirl/arrange/unequal.py`), which carries both decay constants kappa_1 and kappa_2.

Three measured potentials, each with its own resolution floor sigma_V(r):

1. **V_raw(r), static.** The t = 0 energy excess of the superposed single-frequency profiles (Session 1 profile, phi_t = 0): E[Phi_1 + Phi_2] - E[Phi_1] - E[Phi_2], by quadrature with the full potential. This is what the brief calls the static interaction energy from superposition. **By LESSONS 2 and 4 and the Session 2 derivation it is not a binding energy:** it contains omega N_cross, the extra charge the overlap adds, which is about omega^2 r / kappa (about 2r) times |U|, and of the opposite sign for in-phase lumps.
2. **U_static(r), static at fixed N.** The overlap term of the period-averaged E - omega N of the same superposition (the quantity the derivation shows equals the fixed-N interaction energy to first order). Computed by quadrature at any r density.
3. **Dynamical, two independent routes from the 3D PDE.** Pairs are built from copies of one settled lump (snapshot starts, `swirl/engine/snapshot.py`, which removes the start-up shedding and the phase dependence of F-009).
   - **V_N(r):** fixed-N interaction energy from the evolved fields, V_N = [E_pair - 2 E_single] - omega_0 [N_pair - 2 N_single] (+ the second-order term in dN), with N as a whole-period average and E + absorbed bookkeeping, pair and isolated-single runs on the same grid and box, read in the window below.
   - **V_F(r):** from the measured force. Each run's relative acceleration in the window gives F(r) = (E/2) d''(r); V_F(r) = -integral from r to r_max of F, with the tail beyond r_max taken from the fit (stated with and without).

**The window.** Start t1 = 40 (about 6 periods): past the overlap adjustment and the first crossing of each lump's start-up waves across the partner (justified in the report from the isolated-single control). End t2 = the earlier of T_max = 400 and the time at which kappa |d(t) - r| reaches 0.10 (so the tail force changes by at most about 10% inside the window). A run whose t2 - t1 is under 5 periods (35 time units) is marked "moving": its V_N is reported but it is excluded from the long-range fits. Close in-phase pairs (r below about 14) will be "moving" by this rule; that is expected, not a defect.

**Resolution floors.** sigma_V for the static quantities: the change in the value between two quadrature resolutions, and the value for a single lump shifted against itself. For V_N: the scatter of [E_pair - 2E_single] - omega_0 [...] for the largest-r pair and for the isolated-single control against its own shifted copy. For V_F: the far-apart pair's acceleration noise (Session 2's control drifted 5e-4 in d over T = 500). A point is "resolved" if |V| > 3 sigma_V.

### A1. Runs

- **r grid, beta = 4/3 primary, equal lumps, dtheta = 0 and 180:** static at r = 6 to 45 in steps of 0.25. Dynamical at r = 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40 (20 values, from about 2 hw to past the expected floor at r of about 30 to 32). Grid dx = 0.55, dt = 0.04 (the Session 2 pair production grid); box sized per r so that both lumps sit at least 3 hw plus the validated sponge inner radius from the faces.
- **Unequal lumps, beta = 4/3:** two family members with N ratio about 1.3 to 1.6, both passing the snapshot settling guard (Session 3 showed N/N_min 3.1 and 4.4 do not), dtheta = 0 and 180 at t = 0, the same dynamical r grid from r = 10 up. Their phases beat at Delta omega, so the dynamical potential is V(r, t) = V_amp(r) cos(Delta omega t + dtheta) at first order: the amplitude V_amp(r) and the force amplitude are fitted with the measured relative phase, and the window must contain at least half a beat or Delta omega t2 < 0.3. The pair is chosen by numerics from the family table and recorded in an addendum to this file **before** the runs.
- **beta = 5 (b5), equal lumps:** static on the full r grid (cheap). Dynamical at five r values, only if the measured cost is under about 8 wall hours on the Mac at 4 workers; otherwise listed as pending. Its kappa (0.2196 against 0.4165) is a discriminator between a Yukawa tail set by the lump's own kappa and any range set by something else.
- **Controls and standards:** isolated singles on every box size used (standard 6 baseline and V_N reference); linear twins (g4 = g6 = 0) of the in-phase pairs at r = 12 and 24 (standard 5; Session 2 showed twins disperse and cannot test "no interaction", F-015); one box check at 1.5x the box for r = 24, and one resolution check at dx = 0.40 for r = 16 (standard 4); t = 0 energy of every pair against independent quadrature (standard 6).

### A2. Fits

On each potential and channel (static and dynamical; 0 and 180; equal and unequal), over the fit range r_lo = 12 (3.6 hw, where the tail law is within 10% of the full overlap) to r_hi = the last resolved r, weighted by sigma_V, and again over the full resolved range:

| Candidate | Form | Free parameters |
|---|---|---|
| E1 exponential | C e^(-k r) | C, k |
| E2 Yukawa (exponential with 1/r) | C e^(-k r)/r | C, k |
| G Gaussian | C e^(-(r/s)^2) | C, s |
| P power law | C r^(-n) | C, n |
| L constant force plus short-range term | V_inf + sigma r + C e^(-k r)/r | V_inf, sigma, C, k |

Goodness of fit for each: reduced chi-squared, AIC and BIC, and a residual runs test. For L also sigma with its standard error. Fitted k compared with sqrt(1 - omega^2).

### A3. Verdict criteria for the long-range behaviour (written before running)

Applied separately to U_static, V_N and V_F, for each channel. The headline verdict uses V_N and V_F (the dynamical potentials); V_raw is reported and fitted but cannot decide, for the reason in A0.

- **"Decays to zero"** if all of:
  1. |V| falls by at least a factor 100 between r_lo and r_hi, and falls below 3 sigma_V at some sampled r;
  2. in fit L, sigma is within 3 standard errors of zero, and its 3-standard-error bound corresponds to a constant force below 1% of the measured |F| at r = 16;
  3. at least one decaying form (E1, E2, G, P) has reduced chi-squared at most 3.
- **"Grows"** if all of:
  1. in fit L, sigma is non-zero at more than 5 standard errors, with the confining sign (V rising with r in the attractive channel);
  2. |V| at r_hi is above 3 sigma_V and is not decreasing over the last three resolved points;
  3. L beats every decaying form by Delta AIC > 10.
- **"Undetermined"** otherwise. This includes: fewer than two decades of resolved |V|; static and dynamical verdicts that disagree; a channel where no form reaches reduced chi-squared 3.
- **Functional form** (a second, separate statement): a form is "preferred" if it beats every other by Delta AIC > 10 over r_lo to r_hi. A sign-alternating residual at the level of 1e-4 in force at r above 30 (possible radiation-mediated interference, wavenumber k3 = 2.54) counts as neither growth nor a form; it is reported as a residual.
- **Sign of the force at each r:** from V_F (finite differences of the fitted static and of the measured accelerations); "attractive", "repulsive" or "unresolved" (|F| < 3 sigma_F). **A minimum** is a sign change of F between two resolved neighbouring r values, located by interpolation.

### A4. Predictions on record, with the literature expectation

| # | Prediction | Literature / source |
|---|---|---|
| A-P1 | Every dynamical channel "decays to zero"; E2 (Yukawa) is preferred; fitted k = 0.4165 +- 0.01 at the primary (and within 5% of 0.2196 at b5, if run) | Massive field with omega < m: tails A e^(-kappa r)/r and a Yukawa overlap force (Malomed 1998, NLS limit; Battye & Sutcliffe 2000, Q-balls; Session 2 measured 0.4172). Confinement would need a massless mediator; this model has none. Scholar to check |
| A-P2 | For r >= 12, V_N and V_F agree with U_tail within 10% (the Session 2 force excess was 2 to 6%); inside 12 they fall below U_tail in magnitude (the full overlap is 0.35 of the tail law at r = 8) | Session 2 derivation and runs |
| A-P3 | In-phase: attractive at every resolved r. Anti-phase: repulsive at every resolved r. No minimum in either channel for r >= 7 | Every primary source agrees on the signs (Session 1 brief); no source predicts a minimum for identical real oscillons |
| A-P4 | V_raw has the opposite sign to U_static in each channel (positive for in-phase) and |V_raw/U_static| is about 2r at large r; it also decays to zero | LESSONS 2 and 4; derivation section 2 |
| A-P5 | U_static (t = 0) and V_N (dynamical) agree within 10% for r >= 12; the PDE adds no new long-range term | Derivation; LESSONS 2 guard: if the PDE adds nothing, the result is not dynamics, and that is what A-P5 predicts |
| A-P6 | Unequal pair: interaction amplitude follows the Session 3 law within 15% for r >= 12, oscillating at the measured Delta omega; at long range the fitted k tends to min(kappa_1, kappa_2), the lighter lump's kappa | Session 3 P1 model; Axenides 2000 and Battye & Sutcliffe 2000 (beating of unequal frequencies) |
| A-P7 | The resolution floor for V_N is reached between r = 28 and 34 at the primary | U_tail = 6e-3 at r = 28, 1e-3 at 32, against an N floor of about 2e-3 x omega |

## 3. Test B: radiation spectrum of the lone oscillon

### B1. Dispersion relation from the code

Derived from the source of each solver, not assumed: `legacy/engine/evolver.py::SexticEvolver` (named in the brief; it has the legacy stiffness term sigma dx^4 k^4 phi, which changes the dispersion at high k), `swirl/engine/evolver.py::Evolver` (spectral Laplacian, RK4) and `swirl/family/radial.py::Radial` (4th-order finite differences in u = r phi, RK4). For each: the semi-discrete omega(k), the RK4 correction at the dt used, the threshold omega_th = omega(k -> 0), and the group velocity near threshold. Numerics derives; the reviewer re-derives independently from the code without reading numerics' derivation first.

### B2. Runs (radial solver; 3D only if the radial result is ambiguous)

- **B-run 1, primary:** beta = 4/3, omega = 0.9091, the Session 1 start (profile, phi_t = 0). T = 20,000 (the radial control dies near 22,932). Field recorded every 0.2 time units at r = 0 and at r = 10, 15, 20, 30, 45, 60, 80, 120 (outside the core at 3 hw = 10). Box R and sponge chosen so that every probe sits in the undamped interior; sponge validated by a box twice the size over the same probes.
- **B-run 2, dip:** beta = 4/3, omega = 0.9065, same recording. Here the 3 omega source is predicted to nearly vanish (N33), so it is the more sensitive test of what else the lump emits.
- **Controls:** a linear twin of B-run 1 (its spectrum must contain only free waves above threshold); a doubled box for B-run 1 to T = 5000 (probe series compared directly); a dr/2 run of B-run 1 to T = 2000 (standard 4 and resolution).
- **First check whether usable series exist**: no Session 1 to 3 run is known to have stored far-field probes, so the runs above are expected to be needed. If they do exist, they are used in addition.

### B3. Analysis

- Spectra in windows of 2,000 time units (Hann, 50% overlap) at four epochs, so the slow omega drift is visible and the resolution is 2 pi/2000 = 0.0031. Lines identified by frequency and labelled n omega_core where they match within 2 resolution bins; anything else is reported as an unidentified line.
- **Leakage floor:** a synthetic pure line just above threshold (omega = 1.005) and the measured core line put through the same pipeline, giving the sub-threshold power that windowing alone produces.
- **Evanescent floor:** the fundamental's predicted bound tail A e^(-kappa r)/r at each probe (7e-6 at r = 30, 1e-11 at r = 60 for the primary).
- **Flux:** per line, the outgoing energy flux through r_probe, 4 pi r^2 <-phi_t phi_r> from the recorded series (phi_r by a second nearby probe or from the solver state), and in the plane-wave form 2 pi |a_n|^2 omega_n k_n. Sum against the measured loss rate -dE_lump/dt (from E inside 3 hw and from the absorbed energy, over the same epochs).

### B4. Verdict criteria (written before running)

- **"Emission only above threshold"** if all of:
  1. at every probe with r >= 30, the power below omega_th - delta (delta = 3 resolution bins) is at most 10 times the larger of the leakage floor and the evanescent floor, or below 1e-4 of the total power at that probe;
  2. any sub-threshold line present falls with r faster than 1/r across the far probes (a fitted exponent steeper than -1.3, or consistent with e^(-kappa r)/r), i.e. it is bound, not radiated;
  3. the outgoing flux in lines above threshold accounts for the measured loss rate within a factor 3 (the CLAUDE.md tolerance rule applied to a target of equality).
- **"Sub-threshold emission present"** if: a line at omega_s < omega_th - delta at two or more probes with r >= 30, at least 10 times both floors, with amplitude falling as r^-1 (exponent -1 +- 0.3), present in the doubled-box run too; or the above-threshold flux falls short of the loss rate by more than 3x and a sub-threshold line carries the difference.
- **"Undetermined"** otherwise (for example a flux mismatch above 3x with no identifiable sub-threshold line, or a sub-threshold line that the doubled box does not reproduce, which would point to the sponge).

### B5. Predictions on record, with the literature expectation

| # | Prediction | Literature / source |
|---|---|---|
| B-P1 | omega_th = m = 1 in all three solvers to better than 1e-6 at the dt and dr used; the solvers differ only at high k (and SexticEvolver's stiffness term raises omega(k) there) | Klein-Gordon dispersion omega^2 = k^2 + m^2 from V''(0) = m^2 |
| B-P2 | Core: fundamental at 0.9091 (primary) and 0.9065 (dip), each within 0.001, drifting up slowly; odd harmonics only (the potential is even) | Session 1 to 3 measurements; symmetry |
| B-P3 | Far field: 3 omega (2.727) dominant at the primary; 5 omega present and at least 10x weaker in amplitude; no even harmonics above the leakage floor; no fundamental | Oscillon radiation at 3 omega leading (Fodor et al. 2009; Hertzberg 2010; Ibe et al. 2019). Scholar to check |
| B-P4 | At the dip the 3 omega line is suppressed by at least 10x in flux relative to the primary, and 5 omega (or a combination line) carries a share at least 10 times larger than at the primary | N33 (the dip's 3 omega source vanishes); Zhang et al. 2020 and Li et al. 2025 on radiation-rate dips |
| B-P5 | Verdict "emission only above threshold" at both points | A propagating wave below m is evanescent in a massive linear medium; the far field is linear |
| B-P6 | The above-threshold line flux matches the measured loss rate within 30% at the primary | Energy conservation, with E + absorbed already conserved to 1e-5 |
| B-P7 | The early transient (shedding of the phi_t = 0 start) shows at far probes as a slowly chirping signal approaching omega = 1 from above (the local frequency of a dispersing Klein-Gordon wave is m t / sqrt(t^2 - r^2)), never below it | Free Klein-Gordon dispersion |

## 4. Workers, worktrees, order

| Package | Worker | Where |
|---|---|---|
| A-build: static quadrature (V_raw, U_static), pair starts per r, V_N and V_F extraction, fits, cost estimate, unequal-pair choice | numerics | `.claude/worktrees/s4-testA`, branch `s4-testA` |
| A-runs: the run list, monitoring, plots, tables | runner | same worktree |
| B-build: dispersion derivation B1, probe recording in the radial solver, spectra, leakage test, flux | numerics | `.claude/worktrees/s4-testB`, branch `s4-testB` |
| B-runs: runs, controls, plots | runner | same worktree |
| Review: re-derive the dispersion relation independently (B); re-check Test A energy bookkeeping (E + absorbed, V_N construction, the omega N_cross term, the t = 0 quadrature) and recompute verdicts from stored files | reviewer | read-only on both worktrees |
| Literature: two-oscillon interaction (force law, range, any long-range or confining claim), oscillon radiation (harmonics, sub-threshold emission, dips); verdict per prediction: agrees, disagrees, untested; ledger entries for misses | scholar | `docs/PRIOR_ART.md`, `docs/FAILURES.md` (from F-030) |

Both tests run in parallel. Compute: at most 4 worker processes on the Mac at once across both tests (4 performance cores), `NUMBA_NUM_THREADS=1` before any numba import in pool workers, atomic writes, checkpoints for anything over 5 minutes, detached under `caffeinate` with a log for anything over 30 minutes. No commits in a worktree while its runs are active. No RNG is used (no seeded perturbations are needed in either test).

Merge: code, result JSON and plots are committed on the test branches after their runs finish and are merged to `main` through `.claude/worktrees/integration` (the main checkout's `outputs/` is a link to the external drive; Session 3 section 11).

## 5. Stop conditions

- The hard stops in `CLAUDE.md` (no pushes, no AWS, no new dependencies, nothing outside this folder, nothing in `legacy/`).
- If the isolated-single control drifts in position or in N by more than 10% of the smallest resolved pair signal, the Test A dynamical runs wait; the static part and Test B continue.
- If B1 finds a threshold that is not m, Test B stops after B1 and the finding is reported before any spectrum is interpreted.
- No change to the equations of the model.

## Addendum 1 (2026-11-09 12:05 +07): written after the Test A static results and before any production pair run

**What was known when this was written.** The static potentials (no PDE) for the primary, b5 and the unequal pair, the numerics smoke run at r = 20 in phase, and the Test B smoke run. No production pair run had started. The dynamical results are unknown. Section 2's criteria stay on record and are scored as written. The rules below are scored next to them, and the report gives both.

**1. Unequal pair (A1).** No pair with N ratio 1.3 to 1.6 has both members passing the settling guard: on the beta = 4/3 family, only omega 0.9025 to 0.9115 and 0.855 to 0.8585 pass (`outputs/session4/testA/unequal_choice.json`). The pair used is the two long-lived dip lumps, omega 0.9065045 (N 2988.6, A 60.07, kappa 0.4222, hw 3.39) and 0.8573703 (N 8512.4, A 264.61, kappa 0.5147, hw 4.93). This is Session 3's proposal 2. N ratio 2.84, settled Delta omega 0.0478, beat period 131.6, so the window 40 to 400 holds 2.7 beats. **Headline for the unequal channel: the force amplitude (V_F route).** The V_N amplitude is dominated by the beating overlap charge (10 to 35 times U_amp), so it is reported but does not decide.

**2. Why A3 needs a second reading.** On the static potentials, which are exact to a quadrature floor near 1e-11, A3 misfires in three ways:
- the floor is never reached, so decays-clause 1 fails;
- every form has reduced chi-squared 1e15 or more, because the residual is model departure (U_static/U_tail = 1.106 at r = 12), not noise;
- fit L's sigma takes up that departure at a formal 1e18 standard errors, so grows-clauses 1 and 3 pass on a potential that plainly decays. Only grows-clause 2 blocks it.

A-P1's "E2 preferred by Delta AIC > 10" also cannot happen, because L contains E2: on Yukawa data L trails E2 by at most Delta AIC = 4.

**A3' (scored alongside A3, for every channel, static and dynamical).**
- **Weights:** sigma_eff(r) = sqrt(sigma_V(r)^2 + (0.02 V(r))^2). The 2% term is the size of the tail law's departure at r = 16 in the static quadrature and the Session 2 force agreement. It was chosen from the static data, not from the dynamical data.
- **Fit range:** r_lo = 16 (4.8 hw) as well as 12. Both are reported; 16 decides.
- **Decays to zero:**
  1. |V| falls by at least a factor 100 over [r_lo, r_hi] and is still falling over the last three resolved points. Reaching the floor is reported but not required.
  2. The 3-standard-error upper bound on |sigma| from fit L is below 1% of |F(16)|.
  3. At least one decaying form has reduced chi-squared at most 3 under sigma_eff.
- **Grows:**
  1. sigma has the confining sign, exceeds 5 standard errors, and |sigma| is above 1% of |F(16)|.
  2. A3 grows-clause 2, unchanged.
  3. L beats every decaying form by Delta AIC > 10.
- **Undetermined:** otherwise.
- **Form preference:** among the non-nested decaying forms E1, E2, G and P only, the winner must lead by Delta AIC > 10. L is judged by its sigma clause alone.

**3. V_N bookkeeping.** Both are reported: V_N with "E + absorbed" and box N (the plan's version, the headline), and V_N_box, which numerics expects to be less exposed to coherent 3 omega interference. A difference larger than their joint floor is reported as a finding.

**4. Window.** The smoke run shows the in-phase pair at r = 20 is already "moving" (t2 = 73). So the dynamical long-range fits rest on r of about 22 and above in phase. This is expected under A0's rule and is not changed.

**5. A-P2 as written refers to the dynamical potentials and is scored on them.** The static quadrature already shows that U_static exceeds the tail law inside r = 12 in phase (1.106 at 12), while the mid-plane flux falls below it (0.907). My premise ("falls below") was taken from the flux and is wrong for the energy. That is recorded as a miss of the static reading now, before the dynamics.

**6. Not added:** a dense r grid at r >= 30 to resolve the possible 3 omega interference term of period 2.48 (PRIOR_ART N40). It goes in the parking lot.
