"""WP-E benchmark: one K = 5 ring config (72-degree steps, snapshot start from the production source)
at dx = 0.55 with the diagnostics on, and the same without the lumps block. 2 threads per run.

Usage (from the repo root): NUMBA_NUM_THREADS=2 <python> outputs/session2/wpE/bench_ring5.py [a] [steps]
Writes outputs/session2/wpE/bench_ring5_a<a>.json (steps/s, peak RSS, load average, other processes).
"""
import json
import os
import resource
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

from geometry import ring  # noqa: E402
from make_configs import DEFAULTS, auto_box, config  # noqa: E402
from swirl.engine.run import run  # noqa: E402


def main():
    a = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    prof = json.load(open(os.path.join(ROOT, "outputs/step2/profiles/profile_beta1.3333_w0.9091.json")))
    pos = ring(5, a)
    c = {**DEFAULTS, "name": "bench_ring5_a%g" % a, "family": "ring5", "positions": pos.tolist(),
         "phases_deg": [0, 72, 144, 216, 288]}
    N, L = auto_box(pos, c["dx"], prof, c["sponge_tol"])
    cfg, guards = config(c, prof, N, L, c["dx"])
    load0 = os.getloadavg()
    t0 = time.perf_counter()
    r1 = run({**cfg, "t_end": steps * cfg["dt"]})
    t_build = time.perf_counter() - t0 - r1["wall_seconds"]
    del r1["_evolver"]
    cfg2 = {k: v for k, v in cfg.items() if k != "lumps"}
    r2 = run({**cfg2, "t_end": steps * cfg["dt"]})
    load1 = os.getloadavg()
    ps = subprocess.run(["ps", "-Ao", "pcpu,command"], capture_output=True, text=True).stdout.splitlines()
    others = [ln.strip()[:160] for ln in ps[1:] if ln.strip() and float(ln.split()[0]) > 20.0]
    out = {"a": a, "K": 5, "N": N, "L": L, "dx": L / N, "dt": cfg["dt"], "steps": steps, "threads": cfg["threads"],
           "steps_per_s_with_diagnostics": r1["steps_per_second_this_call"],
           "steps_per_s_without_lumps": r2["steps_per_second_this_call"],
           "init_build_seconds_snapshot": t_build,
           "peak_rss_GB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,   # bytes on macOS
           "bytes_per_point_peak": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / N ** 3,
           "load_average_before": load0, "load_average_after": load1, "other_processes_over_20pct_cpu": others,
           "lumps_every": cfg["lumps"]["every"], "count_every": cfg["lumps"]["count_every"]}
    path = os.path.join(HERE, "bench_ring5_a%g.json" % a)
    with open(path + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(path + ".tmp", path)
    print(json.dumps({k: v for k, v in out.items() if k != "other_processes_over_20pct_cpu"}, indent=1))


if __name__ == "__main__":
    main()
