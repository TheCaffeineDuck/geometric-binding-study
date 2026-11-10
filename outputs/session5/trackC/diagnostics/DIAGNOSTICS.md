# WP-C follow-up diagnostics (no verdict in TABLES.md changes)

Scripts d1.py, d2_summary.py, run_d2.py, d3.py, d4.py; results D1.json to D4.json; D2 runs in runs/.

## D1. S-bend 2.3 push: reproduced, explained
- Build number reproduced exactly with the build's settings (smoke-job settings: N 800, dt 0.01, T 300, record_periods 4, centroid fit after the first 10% of the run): speed 0.0386 at lambda 0.5 and 0.0528 at lambda 1 (build notes: 0.039 to 0.053). lambda 0.2: 1.6e-4.
- Same state, T = 2000 or 5000 (production): speed 0.0026 (lambda 0.5), 0.0018 (lambda 1). Production A6 reported speed at most 1e-4 ("pinned").
- Cause: the object does move, 10 to 20 sites (lambda 1: centroid 0.2 to 18 in T 300, to 22.6 in T 2000), and then stops. The production speed and distance come from a centroid fit after the first 10% of a 5,000 time-unit run, i.e. after the transit. So the A6 "pinned" wording in my report was wrong: max centroid excursions are in summary.json A6 `max_abs_centroid`: S-bend 2.3: 19.8 (lambda 1), 5.6 (3), 3.1 (10); bump 2.3: 9.4 (0.3), 8.5 (1). S-bend 3.0 moves at most 0.24. S-bend 2.1 travels 99 to 292 sites.
- Ruled out (no effect on speed to 8 digits): old sponge 100/0.3 against production 300/0.05, seeded noise 1e-8 on or off, dt 0.01 against the production dt 0.0051. Fix commit 49b9dc9 is therefore not the cause (sponge and noise are the changes it made to this path).
- The push still never reaches 100 sites at 2.3 or 3.0; the C5 "miss as written" is unchanged, but "pinned" should read "moves 5 to 20 sites and stops".

## D2. Jump S-bend 2.0, other h (nonlinear runs only, no twins; dt 0.004 as in B9)
| run | cycles reached | frac 1,000 | frac 3,000 | frac at end | late rate | L_half late | largest jump | jump distance / cells |
|---|---|---|---|---|---|---|---|---|
| fd h=1/4 | 4,982 | 0.99999 | 0.99999 | 0.99999 | 1.8395 | 0.1731 | 1.195 | 0.25 / 1 |
| fd h=1/8 (B9) | 5,000 | 0.99999 | 0.99999 | 0.9997 | 1.8457 | 0.1507 | 1.216 | 0.25 / 1 |
| fd h=1/16 | 2,001 (capped; run took about 17 min) | 0.99999 | not reached | 0.99999 | 1.8471 | 0.1502 | 1.223 | 0.25 / 1 |
- Twin frac: not rerun (cost); B3/B9 twins are 0.28 and 0.11 at h=1/8 (tables in TABLES.md).
- spectral h=1/8: FAILED, not run. The spectral sampling of the jump-2.0 profile has |stretch| >= 1 at the start (invalid log in core.py:50, zero-size reduction in window_halflength). One retry after the failure was identical in cause, so logged as open. (B8 jump 1.5 spectral did start.)
- Result: frac, rate (to 0.4%), jump distance and cells are h-independent over 1/4 to 1/16 for fd; L_half 0.173 / 0.151 / 0.150. Scheme independence not tested.

## D3. What the bump becomes (re-run, not analysis only)
No snapshots were saved for A2, so A2 bump 2.2, 2.5, 3.0, 3.5 were rerun with the identical configuration for 10,000 cycles (not 32,000), with a field snapshot at the end (diagnostics/runs; snapshots in checkpoints, not committed).
| start rate | centroid (mod 1) | odd fraction about nearest bond centre | odd fraction about nearest site | late core rate | E in window | family S-bend E at that rate | ratio | family rate at same E |
|---|---|---|---|---|---|---|---|---|
| 2.2 | -0.141 (0.859) | 0.973 | 0.004 | 2.201 | 0.6328 | 0.6292 | 1.0057 | 2.203 |
| 2.5 | -0.502 (0.498) | 1.000 | 0.125 | 2.581 | 1.0773 | 1.0730 | 1.0041 | 2.586 |
| 3.0 | -0.485 (0.515) | 0.9997 | 0.168 | 3.370 | 1.6991 | 1.6986 | 1.0003 | 3.371 |
| 3.5 | -0.543 (0.457) | 0.984 | 0.157 | 4.076 | 2.2874 | 2.2504 | 1.0164 | 4.121 |
Quadratic odd/even split of the field about the centre. At 2.5, 3.0 and 3.5 the object is centred on a bond (distance 0.002 to 0.04) with S-bend (odd) symmetry; at 2.2 it is odd about a bond but its centroid sits 0.14 from the site (still drifting). Each (rate, energy) point lies on the A1 S-bend curve within 0.03% to 1.6% in energy. Note: the "late rate" is the A2 core rate (displacement at the start site); it matches the S-bend family rate to 0.2% to 1.2%. So the bump becomes the S-bend of the same energy, with the 1.4% to 2.6% (2.5, 3.0, 3.5) energy shed. Cycles at end 10,005 to 11,617.

## D4. A7 bump over 200 cycles: mean u and the centroid
- Centroid shift reaches 0.1 at t = 28.7 (cycle 11.4), 0.25 at cycle 17.6, 0.5 at cycle 24.4 (final -0.544); 5%-of-max onset at t = 17.0.
- Running mean of u at 20 sites (the only far probe in the series): 5%-of-max onset t = 68.4 (cycle 27.2), i.e. 40 to 50 time units after the centroid onset. Correlation of |running mean| with |centroid shift| over the run: 0.16 (cumulative mean), 0.21 (250-sample window mean). Maximum 9.2e-5.
- Running means at the centre site and at the peak-stretch bond are dominated by the oscillation of the object itself (correlation -0.32 over the run, -0.65 after cycle 10); they do not isolate a mean field.
- So the mean field does not track the centroid proportionally: it starts after the object has turned into an S-bend (cycle 27 against 7 to 17) and is small and noisy. The 200-cycle averages (2.5e-4 displacement, 4.8e-4 stretch at 60 sites) come from the transient radiation of that conversion, not from a static field. Time-resolved profiles were not stored, so this is from probe series only.
