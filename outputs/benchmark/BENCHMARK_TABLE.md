# Engine v2 benchmark (WP1D), 2026-10-01

Apple M4 (4 P + 6 E cores, 16 GB). Set B, absorber on, dt = dx/8, numba 0.68. Rates are RK4 steps/second (Evolver.step); L = 50N/64 for the timing runs (rate depends on N only).

## Throughput and memory

| N | 4 threads (steps/s) | 1 core alone | 4 x 1-core side by side: per run | aggregate | aggregate / 4-thread | 4thr / 1core | peak RSS per run |
|---|---|---|---|---|---|---|---|
| 64^3 | 265.4 | 115.40 | 87.43 | 349.7 | 1.32 | 2.30 | 0.19 GB |
| 128^3 | 33.9 | 14.81 | 8.93 | 35.7 | 1.05 | 2.29 | 0.38 GB |
| 192^3 | 9.4 | 4.06 | 2.48 | 9.9 | 1.06 | 2.30 | 0.92 GB |
| 256^3 | 3.5 | 1.75 | 0.74 | 2.9 | 0.84 | 2.00 | 1.97 GB |

With diagnostics (sample_every=1.0, point_every=1, through run()): 128^3 25.8 steps/s (0.76 of raw, RSS 0.44 GB); 192^3 7.21 steps/s (0.77 of raw, RSS 1.12 GB). For 64^3 and 256^3 the projections below apply the mean factor 0.766.

## Projected wall time for T = 1000 (threads=4 on the Mac, with diagnostics) and AWS c7i.4xlarge

| N | L | dx | dt | steps | Mac wall (h) | AWS wall/run (h) | AWS cost/run, 2 concurrent (USD) | AWS cost/run, alone (USD) |
|---|---|---|---|---|---|---|---|---|
| 64^3 | 50 | 0.781 | 0.0977 (dx/8) | 10,240 | 0.01 | 0.01 | 0.00 | 0.01 |
| 128^3 | 50 | 0.391 | 0.0488 (dx/8) | 20,480 | 0.22 | 0.22 | 0.08 | 0.16 |
| 192^3 | 50 | 0.260 | 0.0326 (dx/8) | 30,720 | 1.18 | 1.18 | 0.42 | 0.85 |
| 256^3 | 50 | 0.195 | 0.0244 (dx/8) | 40,960 | 4.24 | 4.24 | 1.51 | 3.03 |
| 64^3 | 50 | 0.781 | 0.0500 (0.05) | 20,000 | 0.03 | 0.03 | 0.01 | 0.02 |
| 128^3 | 50 | 0.391 | 0.0500 (0.05) | 20,000 | 0.22 | 0.22 | 0.08 | 0.15 |
| 192^3 | 50 | 0.260 | 0.0500 (0.05) | 20,000 | 0.77 | 0.77 | 0.28 | 0.55 |
| 256^3 | 50 | 0.195 | 0.0500 (0.05) | 20,000 | 2.07 | 2.07 | 0.74 | 1.48 |
| 64^3 | 70 | 1.094 | 0.1367 (dx/8) | 7,314 | 0.01 | 0.01 | 0.00 | 0.01 |
| 128^3 | 70 | 0.547 | 0.0684 (dx/8) | 14,629 | 0.16 | 0.16 | 0.06 | 0.11 |
| 192^3 | 70 | 0.365 | 0.0456 (dx/8) | 21,943 | 0.85 | 0.85 | 0.30 | 0.60 |
| 256^3 | 70 | 0.273 | 0.0342 (dx/8) | 29,257 | 3.03 | 3.03 | 1.08 | 2.16 |
| 64^3 | 70 | 1.094 | 0.0500 (0.05) | 20,000 | 0.03 | 0.03 | 0.01 | 0.02 |
| 128^3 | 70 | 0.547 | 0.0500 (0.05) | 20,000 | 0.22 | 0.22 | 0.08 | 0.15 |
| 192^3 | 70 | 0.365 | 0.0500 (0.05) | 20,000 | 0.77 | 0.77 | 0.28 | 0.55 |
| 256^3 | 70 | 0.273 | 0.0500 (0.05) | 20,000 | 2.07 | 2.07 | 0.74 | 1.48 |

## AWS assumptions

- Quota request f00511d0... is CASE_OPENED (not yet granted); applied quota is still 16 vCPU, so the largest instance is c7i.4xlarge (16 vCPU, 8 physical cores, 0.7140 USD/h; c8g.4xlarge 0.6358). Checked read-only through the AWS MCP service-quotas calls.
- Assumed: c7i per-core speed equals an M4 performance core (unverified; likely optimistic). Scaling: two 4-thread runs side by side on the 8 physical cores, each at the Mac 4-thread rate; no extra bandwidth penalty (optimistic). Cost per run = wall x price / 2 when both slots are used, wall x price when one run is alone. A 16xlarge (once 64 vCPU is granted) gives 8 concurrent runs at the same per-run cost (price per slot is the same), only throughput increases.
- The Mac numbers are measured; the AWS numbers are projections and need a short trial on the instance before relying on them.
