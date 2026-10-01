# WP-E build notes: snapshot starts, K-lump diagnostics, arrangement driver, design and cost

Session 2, WP-E tooling (numerics worker). Branch `worktree-agent-a015af75025bf5371`. No arrangement batch was run; the only arrangement runs are a coarse smoke run and a benchmark.

Code: `swirl/engine/snapshot.py` (new), `swirl/engine/lumps.py`, `swirl/engine/initial.py`, `swirl/engine/run.py`. Tests: `tests/engine/test_snapshot.py`, `tests/engine/test_lumps_k.py`. Scripts and outputs: `outputs/session2/wpE/`. Check records: `outputs/session2/wpE/checks/*.json`.

## 1. Status in one paragraph

- **Snapshot starts work and are exact where they can be.**
  - An uncropped, unshifted snapshot continues bit for bit.
  - Relative phases come out at the set value to 5e-5 rad.
  - With the wide window (26, 30), copies at different phases start equal in N to 0.4e-6 to 4.7e-6 (**first-window values on the dx = 0.8 test source**) and stay equal in E to 3.1e-6 over T = 100. The worst ΔN/N over a T = 100 record is 6.4e-5 (section 4c). With the single-frequency ansatz (F-009) the corresponding differences were 2.5e-2 in frac and 9.5% in N.
- **Misses against the targets as written.**
  - **The specified window (14, 18) does not make equal lumps.** Copies at different phases differ in N by up to 4.3e-4. The crop transient is first order in the cut tail amplitude, about Φ(r_in)/Φ0, while the crop loss is an energy and therefore second order (section 3).
  - **A sub-grid shifted copy matches the shifted source to 1.2e-6 Φ0 at t = 0** (max norm, production dx), not 1e-8. In energy norm the difference is 8e-9 of E.
  - **The period-averaged N over the whole record has a 4e-5 to 6e-5 jitter** between copies that are otherwise equal to 3e-6 in E.
- **K-lump diagnostics.**
  - Extended: a nearest-centre split that is a partition of unity for any K, identical to the WP-C split for K = 2.
  - New: group energy and angular momentum about the group centroid, and the box-face lever fix.
  - A per-step ball N record was added, because the 10-step sampling of a φ_t² that carries 4ω and 6ω gave a 7e-5 error in N.
- **Driver.** Generator, run scripts and post-processing work end to end (K = 3 smoke run). Every planned arrangement fits N ≤ 256 at dx = 0.55. The 2x box checks do not, for spacing 8 and above.

## 2. Snapshot starts: method

- **Source** (`make_source.py prod`). One lump of the β = 4/3, ω = 0.909149 profile at the centre of N = 160, L = 88 (dx = 0.55, dt = 0.04, default sponge from r = 32), evolved by the engine to t_s = 300, checkpointed.
  - Stage 2 follows φ(0, t) to the first upward zero crossing t_up after t_s.
  - ω is the whole-period mean over [t_up − 20, t_up].
  - The settling guard compares that with the same estimate ending about 100 earlier.
- **Phase convention.** t_0 = t_up + π/(2ω) is the source time of phase 0, so phase 0 is a maximum of φ(0, t). For odd harmonics in phase this is exact. A copy at phase θ is the source state at t_0 + θ/ω. Started at t = 0, it oscillates as Φ cos(ωt + θ), the convention of the ansatz and of the relative-phase estimator.
- **Stored.** One reference state only: the source at the step n_ref = floor(t_0/dt). It is reproduced bit for bit by rewinding from the t_s state: the sha256 was identical after a second stage-2 pass. No snapshot is stored per phase.
- **Any phase on demand.** From the reference: full RK4 steps of the source dt, then one final partial step that lands exactly on τ = t_0 − n_ref dt + (θ mod 2π)/ω. All distinct phases of a config are produced in one pass of at most one period. On the production grid that is 24.5 s at run start (benchmark), so the number of distinct phases costs nothing extra.
- **Placement.** Each copy is:
  1. multiplied by the window w(r): 1 for r ≤ r_in, cos² taper to 0 at r_out, applied to φ and φ_t;
  2. cropped to a cube of half-size ceil((r_out + 2)/dx);
  3. shifted by the sub-grid remainder with an exact spectral shift on the cube (Nyquist bins zeroed);
  4. added at the nearest grid point of the target (integer shift, periodic).

  The target must have the source dx; a larger box is zero padding. The resolution check resamples the source spectrally (`"resample": true`, recorded).
- **Guards** (`snapshot.check`, run at generation and again at run start):
  - same sha256 as the source file;
  - ω drift below 1e-4;
  - crop loss below 1e-4 of E and N (raise above 3x, the CLAUDE.md rule; the ratio is recorded as `crop_loss_over_target`);
  - lump tail at the sponge ≤ 1e-3 Φ0 (the `check_lumps` tolerance);
  - tail × window at the faces ≤ 1e-6 Φ0 (the `radial_checks` tolerance);
  - all copies from one source (one source per config).
- **Sources made.** "Shell" is the energy between r_out and the sponge.

| Source | N, L, dx | ω | ω drift (guard 1e-4) | per-period ω over the last 100 | sha256 (first 16) |
|---|---|---|---|---|---|
| prod | 160, 88, 0.55 | 0.9095318 | 9.7e-6 | 0.909462 to 0.909549 | 6193e7b109b7369c |
| test | 80, 64, 0.8 | 0.9095308 | 2.3e-5 | 0.909482 to 0.909540 | 25811196b0f73e7e |
| testbig | 112, 89.6, 0.8 | 0.9095397 | 2.3e-6 | 0.909465 to 0.909551 | 4fccfca55d835b5c |

Regime (standard 1): ω/m = 0.9095. The nonlinear share at the reference slice is 0.350 (the run's definition, |E4| + |E6| over the sum of absolute terms).

Crop loss on the production source (worst over one period, against the whole box). Most of it is radiation still in the box, which is what the window is meant to drop:

| window | 14,18 | 16,20 | 18,22 | 20,24 | 22,26 | 24,28 | 26,30 |
|---|---|---|---|---|---|---|---|
| E loss | 9.9e-5 | 7.5e-5 | 6.5e-5 | 5.8e-5 | 5.1e-5 | 4.4e-5 | 3.7e-5 |
| N loss | 1.07e-4 | 8.1e-5 | 7.0e-5 | 6.2e-5 | 5.5e-5 | 4.8e-5 | 4.0e-5 |
| shell energy | 5.2e-5 | 4.2e-5 | 3.4e-5 | 2.8e-5 | 2.1e-5 | 1.5e-5 | 8.0e-6 |

## 3. Check table

Measured values are in `outputs/session2/wpE/checks/`. "Worst" is over the whole record. Test grids use dx = 0.8 unless stated.

| # | Check (test) | Target | Measured (worst) | Verdict |
|---|---|---|---|---|
| 2a-i | Uncropped, unshifted snapshot continues bit for bit, T = 100 (`test_snapshot_continuation_bit_identical`) | identical | φ(0, t) series and final φ, φ_t identical | pass |
| 2a-ii | Cropped copy against uncropped, window (14,18), T = 100 (`test_snapshot_continuation_crop`): frac (r < 10.03) | about the crop loss, 5.7e-5 | 1.53e-4 (2.7x) | pass with note |
| 2a-iii | same: ball N, ball E, ω (whole-period, W = 20), ω per period, φ(0, t) (`test_snapshot_crop_matches_to_crop_loss`, strict xfail) | about the crop loss (5.7e-5) | ball N 3.53e-4 (6.2x), ball E 1.88e-4 (3.3x), ω (W = 20) 2.17e-4 (3.8x), ω per period 6.84e-4 (12x), φ(0, t) 2.45e-3 Φ0. After t = 60: ball N 1.07e-4, ω (W = 20) 5.9e-5, φ(0) 4.6e-4 | **miss** |
| 2a-iv | same, window (16,20) | crop loss 3.5e-5 | frac 5.4e-5 (1.6x), ball N 1.98e-4 (5.7x), ball E 7.1e-5 (2.0x), ω (W = 20) 9.2e-5 (2.6x), ω per period 2.86e-4 (8x), φ(0, t) 9.9e-4 Φ0 | frac, E, ω (W = 20) pass with note; N and ω per period miss |
| 2b-i | Off-lattice copy (δ = (0.31, −0.17, 0.23) dx), field shifted back against the unshifted copy, test grid, T = 100 (`test_snapshot_offlattice_shift`) | 1e-8 or state | t = 0: 1.14e-5 Φ0 (φ), 7.0e-5 (φ_t); worst 1.67e-4 (φ), 4.4e-4 (φ_t); energy of the difference 1.25e-7 E; ball E 4.1e-6, ball N 3.5e-5, ω 1.4e-5, centroid 2.5e-4. Sponge on and off agree to 1e-6 | **miss** (strict xfail `..._target`) |
| 2b-ii | same at the production dx 0.55, T = 40 (`check_shift_prod.py`) | 1e-8 | t = 0: 1.18e-6 Φ0; worst 3.85e-6 (φ), 1.26e-5 (φ_t). Energy of the difference field 8.0e-9 E. Ball E 8.7e-7, centroid 1.9e-7, ω 4.6e-6 | **miss** in max norm (120 to 380x); meets 1e-8 in energy norm |
| 2c | Two copies at 0 and 72°, d = 40, T = 100 (testbig, `test_snapshot_two_far_copies`): relative phase | 1e-3 rad | 2.8e-4 (14,18); 4.9e-5 (26,30) | pass |
| 2c | same: ΔN/N at equal lump age, first window | 1e-6 | −6.2e-5 (14,18); −3.7e-6 (26,30) | **miss** at (14,18) (62x); **miss** at (26,30) (3.7x, just over 3x) |
| 2c | same: ΔN/N over T = 100 / ΔE/E | 1e-6 | 1.67e-4 / 3.9e-5 (14,18); 4.4e-5 / 1.1e-5 (26,30) | **miss** |
| 2d | Singles at 0, 72, 144, 216, 288°, worst over phases and record (testbig, `test_snapshot_phase_control`) | identical up to the time shift | (14,18): ΔN/N 4.25e-4, ΔE/E 8.8e-5, Δω 2.1e-4, Δfrac 1.1e-4. (26,30): first window ΔN/N ≤ 4.7e-6; record ΔN/N 6.4e-5, ΔE/E 3.1e-6, Δω 2.7e-5, Δfrac 3.4e-6 | F-009 criterion (0.01, 0.005) passed by 100x or more; "identical" missed at about the levels shown |
| D | Diagnostic: ball samples every step against every 5 steps, (26,30), 72° against 0° | — | first-window ΔN/N 3.9e-7 (every step) against −6.9e-5 (every 5) | instrument fix adopted (`it_every_step`) |
| 3a | Split partition of unity, K = 2, 3, 5 at deep overlap (`test_split_partition_of_unity`) | 1e-14 | 4.4e-16, 6.7e-16, 8.9e-16; K = 2 against the WP-C tanh split: 2.2e-16 | pass |
| 3b | K = 3, 5 Gaussian lumps at spacing 22, analytic (`test_k_lumps_constructed`) | 1e-8 | E 4.7e-11; It, Ip 4.5e-12; P 2.3e-14; centroid 7.2e-13; distance matrix 7.4e-13; count = K (peaks within 6.5e-3); group E 3.6e-14; group centroid 1.1e-10; group L 1.1e-16 | pass |
| 3c | Overlapping balls against an independent full-grid evaluation, K = 3 and 5 at spacing 4, K = 5 at 11.76 (`test_k_lumps_split_machinery`) | 1e-12 | ≤ 4.6e-16 | pass |
| 3d | Lump count from coordinates, K = 3, 5 separated / one blob (`test_k_lumps_count_overlap`) | K / 1 | K / 1 | pass |
| 3e | Relative phases of K = 5 stepped synthetic signals, neighbour differences (`test_relative_phase_k5_synthetic`) | 1e-3 rad | 1.09e-4 | pass |
| 3f | Mirror-symmetric field with content on the face plane: L_z (`test_angular_momentum_mirror_face`) | 0 to rounding | −2.3e-14 against a scale of 4.3e4 (5e-19). The WP-C lever gives −417 (1e-2 of the scale) | pass |
| 3g | Three spinning l = 1 Gaussians: L about the pattern axis through the group centroid (`test_angular_momentum_group_spinning_k3`) | 1e-8 | 2.2e-16 (box), 6.3e-14 (sphere) | pass |
| 3h | Spinning Gaussian, WP-C test 3 (unchanged, rerun) | 1e-8 | pass | pass |
| R | Checkpoint resume with a snapshot start, K = 3, group block and per-step It, two chunkings (`test_resume_bit_identical_snapshot_k3`) | bit-identical | identical fields and JSON | pass |
| G | Guards: sponge, face, settling, crop, hash, window, dx (`test_snapshot_guards`) | raise | raise | pass |
| S | Smoke: K = 3 ring at dx = 0.8 through the generator, `run_one.sh`, post-processing | runs | completed, T = 60; ΔE + absorbed 5.8e-6; outputs written | pass |
| W | Whole suite, this branch (`.venv/bin/python -m pytest`, NUMBA_NUM_THREADS = 4) | all pass | `107 passed, 4 xfailed in 1645.21s (0:27:25)`. The xfails are strict records: F-001's check 4 at L = 50, WP-A's box doubling at β = 0.75, and the two WP-E misses 2a-iii and 2b | pass |

Code checks that needed a fix:

- **The test construction for 3b at spacing 11.76.** It measured E errors of 1e-4. Single lumps and the total were exact to 1e-14, so the cause was not the code under test. The bisector split has width ell = 1, so lump i keeps 1/(1 + e^{−d/ell}) of its own core, and that share goes to no lump, because the neighbour's ball does not reach there. This is a property of the WP-C split, which is kept for exact K = 2 compatibility, and it is recorded as the "split leak": −6.6e-5 to −9.7e-5 of E_i at d = 11.76; at d = 4 the split defines the lumps (−6.5% to +24%). The analytic test moved to spacing 22, and the overlap is tested against an independent evaluation (3c).
- **The group centroid used minimum image about lump 0.** That breaks when the group is wider than L/2 about one lump (found with K = 5 at radius 18.7, L = 56). Fixed (one attempt): wrap about the previous group centre.
- **Ball N sampled every 10 steps** (17 samples per period) carries a spline-integration error that does not cancel between copies at different phases. Fixed by `it_every_step` (diagnostic D).
- **The resume test's count window** was longer than the run. That was a test-configuration fix.

## 4. Investigation of the misses

**(a) The crop transient (2a-iii, 2c and 2d at (14,18)).**

- Expected: a cropped copy matches the source to about the crop loss.
- Measured: energy-like global quantities do (frac 2.7x). Amplitudes and frequencies do not (φ(0, t) 2.45e-3 Φ0, ω per period 6.8e-4).
- Ranked causes:
  1. The window removes the tail at r_in, a first-order perturbation of size about Φ(r_in)/Φ0. It excites the lump's shape mode. The crop loss is the energy of the removed tail and is second order.
  2. A placement bug (ruled out by 2a-i).
- Diagnostic: centre kick against window, uncropped reference, T = 100 on the test source (scratch run, values in section 4 only). For windows 10, 14, 16 and 18 to the taper start, max |Δφ(0)|/Φ0 = 1.4e-2, 2.4e-3, 9.9e-4, 6.0e-4, against Φ(r_in)/Φ0 of about 1.3e-2, 1.7e-3, 7e-4, 3e-4. Frac differences were 1.5e-3, 1.5e-4, 5.4e-5, 2.0e-5. The kick tracks Φ(r_in) and decays (to 3.5e-4 Φ0 after t = 60 at (14,18)).
- For lumps at different phases, the cut is applied at different points of the oscillation, so the kick differs: ΔN/N up to 4.3e-4 at (14,18), about 4e-6 at (26,30).
- Rules in: the window radius sets lump equality. Rules out: a code defect.
- **Consequence taken:** the generator default window is (26, 30). This is a deviation from the "about 14 / about 18" design, made because equal lumps are the purpose of the snapshot start. It costs no box size: the face guard already needs 30 from each lump. Any candidate can set "window".

**(b) The sub-grid shift (2b).**

- Expected 1e-8. Measured 1.1e-5 Φ0 (dx = 0.8) and 1.2e-6 (dx = 0.55) at t = 0.
- Ranked causes:
  1. The windowed lump is not band-limited (Nyquist content, zeroed by the shift), and the periodic cube interpolant differs from the full-grid one by that content.
  2. Aliasing of the nonlinear terms on a shifted grid.
  3. The sponge (ruled out: sponge on and off agree to 1e-6 over T = 100).
- The difference falls 10x from dx = 0.8 to 0.55, and its energy is 8e-9 of E at 0.55. It sits in high wavenumbers. Lump observables differ by 1e-7 (centroid) to 5e-6 (ω).
- Rules in: the shift is exact for what the diagnostics see, not to 1e-8 in max norm.

**(c) N jitter over the record (2c, 2d at (26,30)).**

- With per-step It, copies start equal to 0.4e-6 to 4.7e-6 in N and stay equal to 3e-6 in E, but the whole-period N differs by up to 6.4e-5 later in the record.
- Ranked causes:
  1. The window ω (from zero crossings of φ at the centre, dt = 0.08) has a 2e-5 to 3e-5 jitter that enters N = ⟨It⟩/ω; equal energies to 3e-6 imply equal ω to about 2e-7.
  2. A genuine N difference while E stays equal (unlikely for one family member).
- Cheap check not run; proposal: the same at dt = 0.04, and N from a fixed ω.
- What this means for WP-E: compare lumps by E or at a fixed ω, and treat a ΔN/N below about 5e-5 over the record as not resolved by this estimator.

**(e) The N difference between two far copies drifts, and not as the interaction predicts.** `snapshot_two_far_copies_w26_30.json` gives d(N1 − N0)/dt at equal lump age = +2.1e-4 per time unit, against −5.6e-5 predicted from the interaction at d = 40 (lump 1 leads by 72° and should lose N). At (14, 18) it is −1.3e-3.
- At (26, 30) the sign is wrong and the fitted change over T = 100 (2.1e-2, or 7e-6 of N) is below the record-level jitter of the N estimator (4.4e-5 of N, section 4c). I read it as a fit to that jitter, not a trend.
- At (14, 18) it is 23 times the prediction, with the predicted sign. Over T = 100 it amounts to 4.5e-5 of N, the size of the phase-dependent crop transient (section 4a).
- Either way the interaction at d = 40 (2e-6 of N over T = 100) is below what these runs resolve. That is what a far-apart control should show. It also means N exchange between lumps can only be read where it exceeds about 5e-5 of N over the record.

**(f) The source carries radiation above the evanescent profile.** Measured on the production reference state (scratch check):
- φ rms in 26 < r < 34 is 5.1e-5 Φ0, and φ_t rms is 1.4e-4 ωΦ0. The single-frequency profile there is 2.3e-6 Φ0 rms (r 22 to 26: 7.4e-5 against 2.0e-5).
- The face guard (Φ ≤ 1e-6 Φ0 beyond r_face) uses the evanescent profile, so it underestimates the field a lump carries at 30 by a factor of about 20. That field is the lump's steady 3ω radiation.
- The (26, 30) window removes it beyond 30 at t = 0. Every run, like the source run itself, regrows it within about 30 time units, and the sponge absorbs it.
- What the guard bounds is the initial data, not the radiation reaching the sponge and faces. That radiation is present in every run whatever the start, and is handled by the sponge, validated by the box checks.

**(d) Crop guard at (14,18).** On the production and testbig sources the crop loss is 1.07e-4 and 1.09e-4 (1.1x the 1e-4 target). That is a pass with a note. The guard raises only above 3x; the ratio is recorded per config.

## 5. K-lump diagnostics (lumps.py)

- **K > 2 was already handled** for per-lump balls, E, N, phase (against lump 0), lump count from coordinates (any number of maxima) and the per-lump phase windows.
- **Extended.**
  - **Split.** W_i = b_i / Σ_j exp((r_i² − r_j²)/(ell d_nn)), with j over i and the lumps whose balls can overlap. This is the nearest-centre (Voronoi) split with smooth width ell. It is a partition of unity for any K. For K = 2 it equals the WP-C tanh split (2.2e-16). The WP-C product of pair factors would leave 25% unassigned at a triple point.
  - **`pair_distances`** (minimum image), used by the post-processing for the distance matrix.
  - **`lever_arms`.** The grid plane at exactly L/2 from the origin has lever 0, the mean of its two images. This removes the WP-C L_z artefact (4e-5 to 9e-5) from the unpaired box-face plane.
  - **`group`** (optional): E inside a sphere about the energy-weighted centroid of the lumps; L about that centroid for the whole box and inside the sphere (vectors), and their component on the pattern axis.
  - **`it_every_step`** (optional): ∫ W φ_t² per lump every step (no FFT); `summarize` uses it for N.
- **Compatibility.** `LUMP_DEFAULTS` is unchanged, so WP-C configs and checkpoints keep identical parameters. Two changes reach WP-C if its code is swapped mid-run: (1) the lever fix changes L; (2) the K = 2 split agrees with the old one to rounding (1e-16), so a resumed WP-C run would not be bit-identical to its first part. Merge after the WP-C batch, or accept both.
- **`run.py`.** The snapshot guards run whenever the init has a snapshot entry. The init build record (`init_info`: source, hash, per-lump τ, sub-grid shift, cube size, ringing, Nyquist fraction) goes into the checkpoint and the result.

## 6. Arrangement configs, driver, post-processing

**Candidate list** (`make_configs.py`; example `candidates_example.json`, smoke `smoke/candidates_smoke.json`):

    {"defaults": {...any key below...},
     "candidates": [{"name", "family", "positions": [[x,y,z],...], "phases_deg": [...],
                     "start": "snapshot"|"ansatz", "box": "auto"|{"L","N"}, "T",
                     "pattern_axis", "linear_twin", "res_check", "box_check", "window", "lumps": {...}}]}

Defaults: dx 0.55, dt 0.04, T 500, snapshot from `src_prod_N160_L88.npz`, window (26, 30), auto box, 2 threads, lumps every 10 steps, count every 25 (3-period Hann windows), per-step It, group sphere R_ext + 3 hw + 2.

- **Linear twin.** `<name>_lin`, g4 = g6 = 0.
- **Per family.** One `_res` config (dx = 88/224 = 0.393, source resampled) and one `_box` config (L x 1.5) on the first candidate unless flagged.
- **Index.** `configs/index.json` with the guards. The generator fails on any guard.

**Run** (from the repo root; resumable; atomic result with full parameters):

    outputs/session2/wpE/run_one.sh <name> [configs_dir] [results_dir]

which executes (the engine command is the same on Linux; caffeinate wraps it only where it exists):

    NUMBA_NUM_THREADS=2 $PYTHON -m swirl.engine.run outputs/session2/wpE/configs/<name>.json \
        --out outputs/session2/wpE/results/<name>.json --checkpoint-dir checkpoints/session2/wpE/<name> --verbose \
        >> outputs/session2/wpE/logs/<name>.log 2>&1

- `PYTHON` defaults to `<repo>/.venv/bin/python`. From a worktree, set it to the main repo's.
- Batch: `nohup outputs/session2/wpE/run_batch.sh LIST > /dev/null 2>&1 &`, one per free 2-thread slot.
- **A rented instance needs the source file.** Ship `outputs/session2/wpE/source/src_prod_N160_L88.npz` (66 MB, not in git). The hash is checked at load.
- Remake the source: `NUMBA_NUM_THREADS=2 $PYTHON outputs/session2/wpE/make_source.py prod` (stage 1 about 45 min under load, resumable; stage 2 about 5 min).

**Post-processing** (revised after review, Block 3 implemented as written):

    $PYTHON outputs/session2/wpE/postprocess.py [--results DIR] [--criteria FILE] [--emap FILE] [--out DIR]

**Outcome** (`criteria.json`, Block 3). Samples are the count windows with t_mid ≥ 50. A sample is ok when the distinct-lump count equals K and every lump's nearest-neighbour distance is within 0.5 to 1.5 of its initial value; distances come from the coordinates of the count-window maxima, matched to the tracked lumps.
- bonded: ok in at least 95% of the samples and in the last sample, the run completed, and K distinct lumps in the first count window;
- merged: count below K at the end;
- flew apart: count K at the end and some ratio above 1.5;
- not distinct at the start: fewer than K distinct lumps in the first window; never bonded; the end state is reported;
- otherwise: "bounded at T, not bonded", with the slope of the mean nearest-neighbour distance over the last third.

A missing parameter, or K = 1, gives "not evaluated". min_lump_count "K" is the run's own K.

**Mechanical report per run:**
- contact time: the first count window with count < K, or a pair within 2 hw = 6.688. Tracked pairs resolve to half a period; count windows to 25.
- the first time a nearest-neighbour ratio passes 1.5;
- the hold fraction and the count series;
- min and max ratio over the run and at the end;
- group energy at the end (and at t = 100, 250, 300);
- L in the first period and its extremes; total momentum extremes;
- the per-lump N spread and the neighbour phase steps.

Tracked and per-lump series are set to null from the start of the first count window with count < K (`per_lump_valid_until`, with a note in the JSON).

**Criteria table** (`analysis/criteria_E.json`): rows E1 to E15 of Block 3 where they are numeric, each item with measured value, tolerance and hit / miss / not evaluable with the reason. Run names come from the editable `criteria_E_map.json`: the exact name, else a unique result with the row's prefix.

**Synthetic-series tests** (`tests/engine/test_wpe_postprocess.py`) cover every outcome class, the 95% rule (one failing sample of 18 is not bonded, of 26 is bonded), the last-sample rule, count above K, incomplete runs, missing parameters, K = 1, the guards after a merger, and the table.

**Generator.** It refuses a snapshot window other than (26, 30) unless `--allow-window` is given, and writes each config's window to `index.json`. The stale (14, 18) is removed from `candidates_example.json`. The smoke list needs `--allow-window`, because the dx = 0.8 test source only measured windows up to (16, 20).

**Smoke run** (`smoke/`; pipeline only, coarse dx = 0.8, T = 60; generated before the defaults count_every = 25 and it_every_step were set, so it counted every 10, about every 30 in effect, and sampled the ball every 5 steps; its result carries its full parameters). K = 3 ring, a = 10, 120° steps, snapshot window (14,18). It completed with ΔE + absorbed of 5.8e-6, count_distinct 3 throughout, and group energy retained 0.994. The pipeline-test criteria (not the bonded criteria) gave "bonded". Not a result, noted because it is the first stepped-phase ring run: L_axis ≈ −1460 (the stepped ring carries angular momentum from the start), and the nearest-neighbour distance grew from 10.4 to 15.5 by t = 60.

## 7. Design and cost

Box rule: L/2 = R_ext + max(d_sp + 12, d_face) + 0.5, with d_sp = 15.13 (Φ = 1e-3 Φ0, the sponge guard) and d_face = 30.06 (Φ = 1e-6 Φ0, the face guard), and N even and 5-smooth.

- **The face guard is the binding one**, so every box is at least 2(R_ext + 30.56) wide.
- dx = 0.55 (validated for pairs), dt = 0.04, T = 500 (12,500 steps).
- Memory: about 150 B per point for the run (WP-C), plus about 0.7 GB during the snapshot build.

**Benchmark** (`bench_ring5.py`, `bench_ring5_a8.json`). K = 5 ring, a = 8, 72° steps, N = 144, L = 79.2, 2 threads, diagnostics on (lumps every 10, count every 25, group, per-step It):

- 6.87 steps/s with diagnostics, 7.79 without (13% overhead);
- snapshot init build 24.5 s;
- peak RSS 1.28 GB, including the build;
- load average about 7.5 throughout, with WP-B (3D, 2 threads), the WP-A radial sweep (3 workers) and the WP-D worker's scan and tests on the Mac.

Table (`design_cost.py` → `design_cost.json`, all 10 spacings 3 to 30). Mac time is per run at 2 threads under the load above, scaled as N³ log N from the benchmark. c8g is per 2-thread job, scaled from 33.5 steps/s total for 8 N = 160 jobs.

| family (K) | a = 3: L, N | a = 8: L, N | a = 16: L, N | control a = 30: L, N | Mac h (a = 16) | c8g h (a = 16) | box x1.5 / x2 N at a = 16 |
|---|---|---|---|---|---|---|---|
| ring3 | 66.0, 120 | 70.4, 128 | 82.5, 150 | 99.0, 180 | 0.58 | 0.67 | 240 / 300 |
| ring4 | 66.0, 120 | 79.2, 144 | 88.0, 160 | 105.6, 192 | 0.71 | 0.83 | 240 / 320 |
| ring5 | 70.4, 128 | 79.2, 144 | 89.1, 162 | 118.8, 216 | 0.74 | 0.86 | 250 / 324 |
| ring6 | 70.4, 128 | 79.2, 144 | 99.0, 180 | 132.0, 240 | 1.03 | 1.21 | 270 / 360 |
| ring7 | 70.4, 128 | 82.5, 150 | 99.0, 180 | 132.0, 240 | 1.03 | 1.21 | 270 / 360 |
| ring8 | 70.4, 128 | 82.5, 150 | 105.6, 192 | 140.8, 256 | 1.27 | 1.48 | 288 / 384 |
| tetrahedron (4; +centre 5) | 66.0, 120 | 79.2, 144 | 82.5, 150 | 99.0, 180 | 0.58 | 0.67 | 240 / 300 |
| octahedron (6; 7) | 66.0, 120 | 79.2, 144 | 88.0, 160 | 105.6, 192 | 0.71 | 0.83 | 240 / 320 |
| cube (8; 9) | 70.4, 128 | 79.2, 144 | 89.1, 162 | 118.8, 216 | 0.74 | 0.86 | 250 / 324 |
| icosahedron (12; 13) | 70.4, 128 | 79.2, 144 | 99.0, 180 | 118.8, 216 | 1.03 | 1.21 | 270 / 360 |

Memory at a = 16: 0.51 to 1.06 GB per run (2.5 GB at N = 256), plus 0.7 GB during the build.

- **Does not fit N ≤ 256 at dx = 0.55** (from `design_cost.json`).
  - Main runs: none. The largest is the K = 8 ring control at a = 30, N = 256 exactly.
  - 2x box checks: every family at a ≥ 8 except ring3 at a = 8 (N = 256). Also at a = 6: ring5 to ring8, cube and icosahedron (with and without centre). Also ring8 at a = 4.
  - 1.5x box checks: ring8 from a = 12; ring6 and ring7 from a = 14; the icosahedron (with and without centre) from a = 16; every a = 24 control except ring3; every a = 30 control.
  - Resolution checks (dx = 0.393): ring8 from a = 16; the a = 24 controls of ring5 to ring8, the cube and the icosahedron; the a = 30 controls of every family except ring3 and the tetrahedron.
  - A 2x box check at a ≥ 8 needs N 288 to 384 at dx = 0.55 (3.6 to 8.5 GB, 3 to 9 h at 2 threads on the Mac) or a rented machine.
- **Sponge clearance.** Because the face guard binds, the lump-to-sponge distance is at least 30.06 − 12 = 18.1, where Φ ≈ 2.5e-4 Φ0, inside the 1e-3 guard. The generator records it per lump (`index.json`).
- **A batch, for scale.** For example 10 candidates at a ≤ 16, each with a twin, plus 4 family checks: about 24 runs of 0.5 to 1.3 h on the Mac, 12 to 30 slot-hours. Two free slots give about 6 to 15 h, or about 2 h on one c8g.4xlarge with 8 jobs.

## 8. Open issues, misses, deviations

**Misses (results, reported as measured):**

1. 2a-iii: the cropped copy matches the source in frac (2.7x the crop loss), but not in ball N, ω or φ(0, t) at (14,18) (section 4a). Strict xfail `test_snapshot_crop_matches_to_crop_loss`.
2. 2b: sub-grid shift error 1.2e-6 Φ0 at production dx, 1.1e-5 at dx = 0.8, worst 3.8e-6 and 1.7e-4 over the record, against 1e-8 (section 4b). Strict xfail `test_snapshot_offlattice_shift_target` (test grid). Energy norm 8e-9 at production dx.
3. 2c: ΔN/N at the start −6.2e-5 (14,18) and −3.7e-6 (26,30) against 1e-6. Over the record 1.7e-4 and 4.4e-5.
4. 2d: copies at different phases are not identical up to the time shift. Worst ΔN/N 4.25e-4 at (14,18). At (26,30): ΔE/E 3.1e-6, Δfrac 3.4e-6, Δω 2.7e-5, ΔN/N 6.4e-5.
5. Crop guard at (14,18): 1.07e-4 (production) and 1.09e-4 (testbig) against 1e-4, within 3x.

**Open:**

- The N estimator's jitter over the record (section 4c). Proposal: rerun 2d at dt = 0.04, and N at a fixed ω.
- The split leak e^{−d/ell} assigns about 1e-4 of each lump's energy to no lump at spacing 12. Inherited from WP-C, kept for compatibility.
- The source still shows a per-period ω modulation of 8.7e-5 over its last 100 time units (the drift guard over 20-unit windows is 9.7e-6). Copies inherit it identically; a later t_s would reduce it.
- `run_one.sh` needs `PYTHON` from a worktree, which has no `.venv`.
- Running the whole suite rewrites the WP-C check records in `outputs/session2/wpC/checks/` (WP-C tests). Only `pair_t0_quadrature.json` changed, in the last digit (2e-16 relative, from the new split). It was restored from git, not committed.

**Deviations:**

- Default window (26, 30) instead of about (14, 18) (section 4a).
- Crop guard raises at 3x its 1e-4 target, not at 1x.
- Box checks are 1.5x (as WP-C); 2x does not fit N ≤ 256 for most a ≥ 8.
- The resolution check resamples the source spectrally instead of making a fine-grid source.
- Tests 2c and 2d use a second, larger test source, because windows beyond r_out = 20 need a source with its sponge further out.
- Every benchmark here ran with WP-B, the WP-A sweep and the WP-D worker on the Mac.

**Parking lot:** a C∞ window taper (the cos² taper has a second-derivative jump, which feeds the Nyquist content behind 2b); a radiation-free source (a later t_s, or a source box with a far sponge) to cut the crop loss, which is mostly radiation.
