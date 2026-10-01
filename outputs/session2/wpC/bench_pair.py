"""Steps/s for the pair-run grid with the new diagnostics on and off (WP-C item 6).

Runs a config from configs/ for a fixed number of steps (no checkpoint, no output) and
records the wall time, with and without the `lumps` block, and the load on the machine.
Usage: NUMBA_NUM_THREADS=2 <venv python> outputs/session2/wpC/bench_pair.py <name> [steps]
Appends to outputs/session2/wpC/bench_pair.json.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from swirl.engine.run import run  # noqa: E402


def load():
    try:
        ps = subprocess.run(["ps", "-Ao", "pcpu,command"], capture_output=True, text=True).stdout.splitlines()
        busy = [l.strip()[:160] for l in ps[1:] if l.strip() and float(l.split()[0]) > 20.0]
        la = os.getloadavg()
    except Exception as e:  # noqa: BLE001
        busy, la = [repr(e)], None
    return {"loadavg": la, "processes_over_20pct_cpu": busy}


def main():
    name = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    cfg = json.load(open(os.path.join(HERE, "configs", name + ".json")))
    out = {"config": name, "N": cfg["N"], "L": cfg["L"], "threads": cfg["threads"], "steps": steps,
           "numba_threads_env": os.environ.get("NUMBA_NUM_THREADS"), "lump_every_steps": cfg["lumps"]["every"],
           "machine_before": load()}
    for tag, c in (("with_lumps", cfg), ("without_lumps", {k: v for k, v in cfg.items() if k != "lumps"})):
        t0 = time.perf_counter()
        r = run(c, max_steps=steps)
        w = time.perf_counter() - t0
        out[tag] = {"wall_s": w, "steps_per_s_run_call": r["steps_per_second_this_call"],
                    "steps_per_s_incl_setup_and_postproc": steps / w}
        print(tag, out[tag], flush=True)
    out["machine_after"] = load()
    out["overhead_lumps"] = out["without_lumps"]["steps_per_s_run_call"] / out["with_lumps"]["steps_per_s_run_call"] - 1
    path = os.path.join(HERE, "bench_pair.json")
    old = json.load(open(path)) if os.path.exists(path) else []
    old.append(out)
    with open(path + ".tmp", "w") as f:
        json.dump(old, f, indent=1)
    os.replace(path + ".tmp", path)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
