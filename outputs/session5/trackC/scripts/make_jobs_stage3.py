"""Stage 3: reruns at dt/4 of pushes whose guard trip the driver labelled 'numerical (moves with dt)'
or 'numerical (no trip at dt/2 ...)' (engineering fix, first attempt), plus the A7 bump over 100 cycles."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib                                   # noqa: E402

JOBS = os.path.join(lib.TC, "jobs")
out = []
for fn in sorted(os.listdir(lib.RUNS)):
    if not (fn.startswith(("A6_", "B7_")) and fn.endswith(".json")) or fn.endswith("_twin.json") or "dt4" in fn:
        continue
    r = lib.J(fn[:-5])
    g = r["summary"].get("guard")
    if g and g.get("rerun_half_dt", {}).get("label", "").startswith("numerical"):
        p = r["params"]
        cfg = {k: v for k, v in p.items() if k in ("kind", "medium", "law", "h", "N", "dt", "sample_dt", "T", "record_periods", "probes",
                                                    "probe_max", "profile_range", "start", "push")}
        cfg["kind"] = "evolve"
        cfg["dt"] = p["dt"] / 4
        cfg["name"] = fn[:-5] + "_dt4"
        cfg["prio"] = 1
        cfg["est_s"] = 60
        out.append(cfg)
        print("rerun", cfg["name"], "dt", cfg["dt"])
json.dump(out, open(os.path.join(JOBS, "stage3.json"), "w"), indent=1)
