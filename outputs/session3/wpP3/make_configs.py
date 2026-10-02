"""WP-P3 configs: 3D runs of the P2 point-model candidates (unequal static pairs at 7.2 hw), the in-phase control,
isolated single-lump controls (one per source) and one linear twin. Two-source starts use init["snapshots"], a list
of snapshot entries (the additive extension of swirl/engine/initial.py and lumps.py on this branch).

Two stages (from the repo root):
  <python> outputs/session3/wpP3/make_configs.py --dry       # plan only (plan.json); no sources needed
  <python> outputs/session3/wpP3/make_configs.py A           # P3_inphase, P3_c0230_lin (run these first)
  <python> outputs/session3/wpP3/make_configs.py B           # after P3_inphase finished: reads its contact time,
                                                             # sets T = min(7500, max(5 x contact, 3000)) (7500 if no
                                                             # contact), writes T_choice.json and the configs of
                                                             # P3_c0230, P3_c0242, P3_iso_N100, P3_iso_N175, P3_iso_N250
Configs go to configs/<name>.json and configs/index.json.

Runs (positions on the x axis; phases phi ~ Phi cos(omega t + theta)):
  P3_inphase     N0 + N0 in phase at d = 7.2 hw0 = 24.08; its 3D contact time is the interaction unit; T = 2000
  P3_c0230_lin   linear twin of P3_c0230 (g4 = g6 = 0), T = 1000
  P3_c0230       N0 (theta 0) + 1.75 N0 (theta 0), d = 24.08       (point-model run p2_0230)
  P3_c0242       N0 (theta 0) + 2.5 N0 (theta 90), d = 24.08       (point-model run p2_0242)
  P3_iso_N100, P3_iso_N175, P3_iso_N250   one lump alone at the origin, smallest box passing the guards, same T
                 as the candidates. score.py builds the rule's control from these (a deviation from a far-apart pair,
                 for cost; README).
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
D_CAND = 7.2 * HW0
DX, DT, THREADS, WINDOW, SPONGE_TOL = 0.55, 0.04, 2, [26.0, 30.0], 1e-3
SOURCES = {"N100": "outputs/session2/wpE/source/src_prod_N160_L88.npz",
           "N175": "outputs/session3/wpP3/source/src_p3_N175.npz",
           "N250": "outputs/session3/wpP3/source/src_p3_N250.npz"}
RATE_160_PER_JOB, SLOTS, USD_PER_H = 33.5 / 8, 8, 0.6358    # Session 2: 33.5 steps/s total, 8 jobs x 2 threads
T_MIN, T_CAP, T_UNITS = 3000.0, 7500.0, 5
STAGE_A = {
    "P3_inphase": {"lumps": [("N100", 0.0, -1), ("N100", 0.0, 1)], "d": D_CAND, "T": 2000.0,
                   "role": "in-phase control (equal lumps): its contact time is the interaction unit"},
    "P3_c0230_lin": {"lumps": [("N100", 0.0, -1), ("N175", 0.0, 1)], "d": D_CAND, "T": 1000.0, "linear": True,
                     "role": "linear twin of P3_c0230"},
}
STAGE_B = {
    "P3_c0230": {"lumps": [("N100", 0.0, -1), ("N175", 0.0, 1)], "d": D_CAND, "role": "candidate p2_0230"},
    "P3_c0242": {"lumps": [("N100", 0.0, -1), ("N250", 90.0, 1)], "d": D_CAND, "role": "candidate p2_0242"},
    "P3_iso_N100": {"lumps": [("N100", 0.0, 0)], "d": 0.0, "role": "isolated N0 lump (control source)"},
    "P3_iso_N175": {"lumps": [("N175", 0.0, 0)], "d": 0.0, "role": "isolated 1.75 N0 lump (control source)"},
    "P3_iso_N250": {"lumps": [("N250", 0.0, 0)], "d": 0.0, "role": "isolated 2.5 N0 lump (control source)"},
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


def positions(run):
    return [[s * run["d"] / 2, 0.0, 0.0] for _, _, s in run["lumps"]]


def box(run):
    """Smallest box meeting the sponge and face guards of every lump (the WP-E auto-box rule, per lump) and holding
    the snapshot cube (2 ceil((r_out + margin)/dx) points)."""
    half = 0.0
    for (key, _, _), x in zip(run["lumps"], positions(run)):
        prof = profile(key)
        need = abs(x[0]) + max(clearance(prof, SPONGE_TOL) + SPONGE_DEFAULT["width"], clearance(prof, TAIL_TOL)) + 0.5
        half = max(half, need)
    N = max(smooth5(2 * half / DX), smooth5(2 * np.ceil((WINDOW[1] + 2.0) / DX)))
    return N, round(N * DX, 9)


def config(name, run, N, L, T):
    from swirl.engine.lumps import check_lumps
    from swirl.engine.snapshot import load_meta
    pos = positions(run)
    snaps, oms, hws = [], [], []
    for (key, th, _), x in zip(run["lumps"], pos):
        meta = load_meta(SOURCES[key])
        snaps.append({"source": SOURCES[key], "sha256": meta["sha256"], "centers": [x], "phases": [float(np.deg2rad(th))],
                      "window": WINDOW, "margin": 2.0})
        oms.append(meta["omega"])
        hws.append(profile(key)["half_width"])
    init = {"snapshots": snaps}
    K = len(pos)
    R_ext = max(abs(x[0]) for x in pos)
    lumps = {"centers": pos, "R_ball": 10.0, "taper": 2.0, "split_width": 1.0, "every": 10,
             "count_every": 25.0, "count_periods": 3, "omega_nominal": float(np.mean(oms)), "count_threshold": 0.01,
             "it_every_step": True, "count_min_sep": 1.0, "saddle_merge": 0.9, "avg_window": 20.0, "planes": [],
             "axes": [{"origin": [0.0, 0.0, 0.0], "dir": [0.0, 0.0, 1.0]}],
             "group": {"R": R_ext + 3 * max(hws) + 2.0, "taper": 2.0, "axis": [0.0, 0.0, 1.0]}}
    p0 = profile("N100")
    lin = run.get("linear", False)
    cfg = {"N": int(N), "L": float(L), "m": p0["m"], "g4": 0.0 if lin else p0["g4"], "g6": 0.0 if lin else p0["g6"],
           "dt": DT, "t_end": float(T), "init": init, "absorber": True, "kfilter": None, "center": pos[0],
           "r_c": [10.0, 3 * max(hws), float(lumps["group"]["R"])], "sample_every": 1.0, "point_every": 1,
           "avg_window": 20.0, "checkpoint_every": 50.0, "threads": THREADS, "lumps": lumps,
           "_candidate": {"name": name, "role": run["role"], "positions": pos,
                          "phases_deg": [th for _, th, _ in run["lumps"]], "lump_keys": [k for k, _, _ in run["lumps"]],
                          "source_omegas": oms, "K": K, "nearest_spacing": run["d"] if K > 1 else None,
                          "window": WINDOW, "start": "snapshot"},
           "_note": "WP-P3 %s" % run["role"]}
    guards = check_lumps(init, cfg["L"], dict(SPONGE_DEFAULT), SPONGE_TOL)
    return cfg, guards


def job(run, T):
    N, L = box(run)
    steps = int(round(T / DT))
    rate = RATE_160_PER_JOB * (160.0 / N) ** 3
    return {"role": run["role"], "lumps": run["lumps"], "d": run["d"], "N": N, "L": L, "T": T, "steps": steps,
            "est_steps_per_s_per_job": rate, "est_job_hours": steps / rate / 3600}


def plan(T_B):
    """Phase 1 (stage A) then phase 2 (stage B); hours at 2 threads per job when the instance is packed 8 x 2."""
    A = {n: job(r, r["T"]) for n, r in STAGE_A.items()}
    B = {n: job(r, T_B) for n, r in STAGE_B.items()}
    srcs = {"src_p3_N175": 7600 / RATE_160_PER_JOB / 3600, "src_p3_N250": 7600 / RATE_160_PER_JOB / 3600}
    jh = sum(j["est_job_hours"] for j in A.values()) + sum(j["est_job_hours"] for j in B.values()) + sum(srcs.values())
    wall_A = max([j["est_job_hours"] for j in A.values()] + list(srcs.values()))
    wall_B = max(j["est_job_hours"] for j in B.values())
    return {"phase1_sources_and_stage_A": {"jobs": {**A, **{k: {"est_job_hours": v, "N": 160, "steps": 7600}
                                                          for k, v in srcs.items()}},
                                           "slots_used": len(A) + len(srcs), "slots_spare": SLOTS - len(A) - len(srcs),
                                           "wall_hours": wall_A},
            "phase2_stage_B": {"jobs": B, "slots_used": len(B), "slots_spare": SLOTS - len(B), "wall_hours": wall_B,
                               "T": T_B},
            "total_job_hours": jh, "usd_share_of_packed_instance": jh / SLOTS * USD_PER_H,
            "standalone_wall_hours": wall_A + wall_B, "standalone_usd": (wall_A + wall_B) * USD_PER_H,
            "basis": "c8g.4xlarge 0.6358 USD/h, 8 jobs x 2 threads; 4.19 steps/s per job at N = 160 (Session 2), "
                     "scaled as N^-3; a packed job's cost share is its hours / 8 x 0.6358"}


def write(runs_T, stage):
    d = os.path.join(HERE, "configs")
    os.makedirs(d, exist_ok=True)
    ip = os.path.join(d, "index.json")
    index = json.load(open(ip)) if os.path.exists(ip) else {}
    for name, (run, T) in runs_T.items():
        N, L = box(run)
        cfg, guards = config(name, run, N, L, T)
        with open(os.path.join(d, name + ".json.tmp"), "w") as f:
            json.dump(cfg, f)
        os.replace(os.path.join(d, name + ".json.tmp"), os.path.join(d, name + ".json"))
        index[name] = {**job(run, T), "stage": stage, "guards": guards}
    with open(ip + ".tmp", "w") as f:
        json.dump(index, f, indent=1)
    os.replace(ip + ".tmp", ip)
    print("\n".join(sorted(runs_T)), "->", d)


def contact_time(path):
    sys.path.insert(0, HERE)
    from score import inphase_contact
    return inphase_contact(json.load(open(path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=["A", "B"])
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    out = {"T_cap": plan(T_CAP), "T_from_point_model_unit_921": plan(max(T_MIN, min(T_CAP, T_UNITS * 921.0)))}
    with open(os.path.join(HERE, "plan.json.tmp"), "w") as f:
        json.dump(out, f, indent=1)
    os.replace(os.path.join(HERE, "plan.json.tmp"), os.path.join(HERE, "plan.json"))
    for k, p in out.items():
        print(k, "phase 1 wall %.2f h (spare slots %d), phase 2 wall %.2f h at T %.0f (spare slots %d); job-hours %.1f; "
              "packed share %.2f USD; standalone %.1f h, %.2f USD" % (
                  p["phase1_sources_and_stage_A"]["wall_hours"], p["phase1_sources_and_stage_A"]["slots_spare"],
                  p["phase2_stage_B"]["wall_hours"], p["phase2_stage_B"]["T"], p["phase2_stage_B"]["slots_spare"],
                  p["total_job_hours"], p["usd_share_of_packed_instance"], p["standalone_wall_hours"],
                  p["standalone_usd"]))
    if a.dry or a.stage is None:
        return
    if a.stage == "A":
        write({n: (r, r["T"]) for n, r in STAGE_A.items()}, "A")
        return
    tc = contact_time(os.path.join(HERE, "results", "P3_inphase.json"))
    T = T_CAP if tc is None else float(min(T_CAP, max(T_UNITS * tc, T_MIN)))
    choice = {"inphase_contact_time": tc, "T": T, "rule": "T = min(7500, max(5 x contact, 3000)); 7500 if no contact",
              "units_covered": (T / tc) if tc else None}
    with open(os.path.join(HERE, "T_choice.json.tmp"), "w") as f:
        json.dump(choice, f, indent=1)
    os.replace(os.path.join(HERE, "T_choice.json.tmp"), os.path.join(HERE, "T_choice.json"))
    print(choice)
    write({n: (r, T) for n, r in STAGE_B.items()}, "B")


if __name__ == "__main__":
    main()
