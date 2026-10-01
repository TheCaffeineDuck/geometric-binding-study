"""Engine v2 speed and memory at one grid size (run one size per process).

    python -m swirl.engine.bench N [--threads 4] [--out path.json]

Reports raw RK4 steps/second (Evolver.step only), steps/second through run()
with the absorber on and diagnostics sampled every 1.0 time unit at
dt = dx/8, and the process peak RSS.
"""

import argparse
import json
import resource
import sys
import time


from swirl.engine.evolver import Evolver, SPONGE_DEFAULT
from swirl.engine.initial import build_initial
from swirl.engine.run import atomic_write_json, run


def bench(N, threads=4, n_steps=None):
    L = 50.0 * N / 64  # dx fixed at 50/64
    dx = L / N
    dt = dx / 8
    n_steps = n_steps or (400 if N <= 64 else 100)
    init = {"gaussians": [{"center": [0, 0, 0], "amp": 2.0, "R": 2.5}]}
    ev = Evolver(N, L, 1.0, 0.30, 0.055, absorber=SPONGE_DEFAULT, threads=threads)
    ev.set_state(*build_initial(ev.coords, ev.L, init))
    ev.step(dt)  # JIT warm-up
    t0 = time.perf_counter()
    for _ in range(n_steps):
        ev.step(dt)
    raw = n_steps / (time.perf_counter() - t0)
    del ev
    cfg = {"N": N, "L": L, "dt": dt, "t_end": n_steps * dt, "init": init, "absorber": True,
           "sample_every": 1.0, "r_c": [5.0, 10.0, 15.0], "threads": threads}
    res = run(cfg)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 2 ** 20 if sys.platform == "darwin" else rss / 2 ** 10
    return {"N": N, "L": L, "dt": dt, "threads": threads, "n_steps": n_steps,
            "raw_steps_per_second": raw,
            "run_steps_per_second_with_diagnostics": res["steps_per_second_this_call"],
            "peak_rss_MB": rss_mb,
            "projected_hours_T1000_dt_dx8": 1000.0 / dt / res["steps_per_second_this_call"] / 3600}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("N", type=int)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = bench(a.N, a.threads)
    print(json.dumps(r), flush=True)
    if a.out:
        atomic_write_json(r, a.out)


if __name__ == "__main__":
    main()
