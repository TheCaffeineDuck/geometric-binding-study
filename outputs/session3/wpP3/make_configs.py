"""WP-P3 configs: 3D runs of the P2 point-model candidates (unequal static pairs at 7.2 hw), their controls and one
linear twin. Two-source snapshot starts: init["snapshot"] is a list of entries, one per source (the additive
extension of swirl/engine/initial.py and lumps.py on this branch).

Usage (from the repo root):
  <python> outputs/session3/wpP3/make_configs.py --dry     # plan only: boxes, steps, cost (no sources needed)
  <python> outputs/session3/wpP3/make_configs.py           # configs (needs the three sources; see make_sources.py)
Writes outputs/session3/wpP3/plan.json (always) and configs/<name>.json + configs/index.json (not --dry).

Runs (positions on the x axis, centred; phases phi ~ Phi cos(omega t + theta)):
  P3_c0230       N0 (theta 0) + 1.75 N0 (theta 0),  d = 7.2 hw0 = 24.08   point-model run p2_0230
  P3_c0242       N0 (theta 0) + 2.5 N0 (theta 90),  d = 24.08             point-model run p2_0242
  P3_c0230_far   the same lumps at d = 45 (far-apart control; the P2 control spacing)
  P3_c0242_far   the same lumps at d = 45
  P3_inphase     N0 + N0 in phase at d = 24.08: its contact time is the interaction unit of the rule
  P3_c0230_lin   linear twin of P3_c0230 (g4 = g6 = 0)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from swirl.engine.evolver import SPONGE_DEFAULT  # noqa: E402
from swirl.engine.initial import TAIL_TOL, radial_spline  # noqa: E402

HW0 = 3.344038183621855
D_CAND, D_FAR = 7.2 * HW0, 45.0
DX, DT, THREADS, WINDOW, SPONGE_TOL = 0.55, 0.04, 2, [26.0, 30.0], 1e-3
SOURCES = {"N100": "outputs/session2/wpE/source/src_prod_N160_L88.npz",
           "N175": "outputs/session3/wpP3/source/src_p3_N175.npz",
           "N250": "outputs/session3/wpP3/source/src_p3_N250.npz"}
# Session 2 cloud rate: 33.5 steps/s total at N = 160 with 8 jobs x 2 threads on c8g.4xlarge
RATE_160_PER_JOB, JOBS_PER_INSTANCE, USD_PER_H = 33.5 / 8, 8, 0.6358
T_CAND = 7500.0          # >= 5 interaction units for an in-phase contact up to 1500 (point model: 921)
RUNS = {
    "P3_c0230": {"lumps": [("N100", 0.0), ("N175", 0.0)], "d": D_CAND, "T": T_CAND, "role": "candidate p2_0230"},
    "P3_c0242": {"lumps": [("N100", 0.0), ("N250", 90.0)], "d": D_CAND, "T": T_CAND, "role": "candidate p2_0242"},
    "P3_c0230_far": {"lumps": [("N100", 0.0), ("N175", 0.0)], "d": D_FAR, "T": T_CAND, "role": "far-apart control of P3_c0230"},
    "P3_c0242_far": {"lumps": [("N100", 0.0), ("N250", 90.0)], "d": D_FAR, "T": T_CAND, "role": "far-apart control of P3_c0242"},
    "P3_inphase": {"lumps": [("N100", 0.0), ("N100", 0.0)], "d": D_CAND, "T": 2000.0,
                   "role": "in-phase control (equal lumps): contact time = interaction unit"},
    "P3_c0230_lin": {"lumps": [("N100", 0.0), ("N175", 0.0)], "d": D_CAND, "T": 1000.0, "linear": True,
                     "role": "linear twin of P3_c0230"},
}


def profile(key):
    return json.load(open(os.path.join(HERE, "profiles", "profile_beta1.3333_%s.json" % key)))


def clearance(prof, tol):
    spl = radial_spline(prof["r"], prof["Phi"])
    P0 = abs(prof["Phi"][0])
    return float(brentq(lambda r: abs(spl(r)) / P0 - tol, 3.0, prof["r"][-1] - 1.0))


def smooth5(n):
    n = int(np.ceil(n))
    while True:
        if n % 2 == 0:
            m = n
            for p in (2, 3, 5):
                while m % p == 0:
                    m //= p
            if m == 1:
                return n
        n += 1


def box(run):
    """Smallest box meeting the sponge and face guards of every lump (the WP-E auto-box rule, per lump)."""
    half = 0.0
    for (key, _), x in zip(run["lumps"], (-run["d"] / 2, run["d"] / 2)):
        prof = profile(key)
        need = abs(x) + max(clearance(prof, SPONGE_TOL) + SPONGE_DEFAULT["width"], clearance(prof, TAIL_TOL)) + 0.5
        half = max(half, need)
    N = smooth5(2 * half / DX)
    return N, round(N * DX, 9)


def config(name, run, N, L):
    from swirl.engine.lumps import check_lumps
    from swirl.engine.snapshot import load_meta
    pos = [[-run["d"] / 2, 0.0, 0.0], [run["d"] / 2, 0.0, 0.0]]
    snaps, oms, hws = [], [], []
    for (key, th), x in zip(run["lumps"], pos):
        meta = load_meta(SOURCES[key])
        snaps.append({"source": SOURCES[key], "sha256": meta["sha256"], "centers": [x], "phases": [np.deg2rad(th)],
                      "window": WINDOW, "margin": 2.0})
        oms.append(meta["omega"])
        hws.append(profile(key)["half_width"])
    init = {"snapshot": snaps}
    lumps = {"centers": pos, "R_ball": 10.0, "taper": 2.0, "split_width": 1.0, "every": 10,
             "count_every": 25.0, "count_periods": 3, "omega_nominal": float(np.mean(oms)), "count_threshold": 0.01,
             "it_every_step": True, "count_min_sep": 1.0, "saddle_merge": 0.9, "avg_window": 20.0, "planes": [],
             "axes": [{"origin": [0.0, 0.0, 0.0], "dir": [0.0, 0.0, 1.0]}],
             "group": {"R": run["d"] / 2 + 3 * max(hws) + 2.0, "taper": 2.0, "axis": [0.0, 0.0, 1.0]}}
    p0 = profile("N100")
    lin = run.get("linear", False)
    cfg = {"N": int(N), "L": float(L), "m": p0["m"], "g4": 0.0 if lin else p0["g4"], "g6": 0.0 if lin else p0["g6"],
           "dt": DT, "t_end": float(run["T"]), "init": init, "absorber": True, "kfilter": None, "center": pos[0],
           "r_c": [10.0, 3 * max(hws), float(lumps["group"]["R"])], "sample_every": 1.0, "point_every": 1,
           "avg_window": 20.0, "checkpoint_every": 50.0, "threads": THREADS, "lumps": lumps,
           "_candidate": {"name": name, "role": run["role"], "positions": pos,
                          "phases_deg": [th for _, th in run["lumps"]], "lump_keys": [k for k, _ in run["lumps"]],
                          "source_omegas": oms, "K": 2, "nearest_spacing": run["d"], "window": WINDOW, "start": "snapshot"},
           "_note": "WP-P3 %s" % run["role"]}
    guards = check_lumps(init, cfg["L"], dict(SPONGE_DEFAULT), SPONGE_TOL)
    return cfg, guards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    plan, total_job_h, longest = {}, 0.0, 0.0
    for name, run in RUNS.items():
        N, L = box(run)
        steps = int(round(run["T"] / DT))
        rate = RATE_160_PER_JOB * (160.0 / N) ** 3
        h = steps / rate / 3600
        total_job_h += h
        longest = max(longest, h)
        plan[name] = {"role": run["role"], "lumps": run["lumps"], "d": run["d"], "d_over_hw0": run["d"] / HW0,
                      "N": N, "L": L, "dx": DX, "dt": DT, "T": run["T"], "steps": steps,
                      "est_steps_per_s_per_job": rate, "est_job_hours": h}
    inst_h = max(longest, total_job_h / JOBS_PER_INSTANCE)
    out = {"runs": plan, "jobs": len(plan), "total_job_hours": total_job_h, "longest_job_hours": longest,
           "instance": "c8g.4xlarge, 8 jobs x 2 threads", "instance_hours_one_instance": inst_h,
           "usd_at_0.6358_per_h": inst_h * USD_PER_H,
           "rate_basis": "Session 2: 33.5 steps/s total at N = 160, 8 jobs x 2 threads; scaled as N^-3",
           "sources_first": "src_p3_N175 and src_p3_N250 (make_sources.py): about 7,600 steps each at N = 160 "
                            "(about 0.5 h per source on 2 threads); N0 reuses Session 2's src_prod_N160_L88"}
    with open(os.path.join(HERE, "plan.json.tmp"), "w") as f:
        json.dump(out, f, indent=1)
    os.replace(os.path.join(HERE, "plan.json.tmp"), os.path.join(HERE, "plan.json"))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))
    for k, v in plan.items():
        print("%-14s N %d L %.1f T %.0f steps %d est %.2f h" % (k, v["N"], v["L"], v["T"], v["steps"], v["est_job_hours"]))
    if a.dry:
        return
    d = os.path.join(HERE, "configs")
    os.makedirs(d, exist_ok=True)
    index = {}
    for name, run in RUNS.items():
        cfg, guards = config(name, run, plan[name]["N"], plan[name]["L"])
        with open(os.path.join(d, name + ".json.tmp"), "w") as f:
            json.dump(cfg, f)
        os.replace(os.path.join(d, name + ".json.tmp"), os.path.join(d, name + ".json"))
        index[name] = {**plan[name], "guards": guards}
    with open(os.path.join(d, "index.json.tmp"), "w") as f:
        json.dump(index, f, indent=1)
    os.replace(os.path.join(d, "index.json.tmp"), os.path.join(d, "index.json"))
    print(len(index), "configs in", d)


if __name__ == "__main__":
    main()
