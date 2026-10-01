"""WP-E arrangement configs: candidate list JSON -> one engine config per run.

Usage (from the repo root):
  <python> outputs/session2/wpE/make_configs.py CANDIDATES.json [--out DIR]
Default out: outputs/session2/wpE/configs. Prints the run names; writes DIR/index.json.

Candidate list format (JSON):
{
  "defaults": {                                   # optional; any candidate key may be set here
    "dx": 0.55, "dt": 0.04, "T": 500.0, "start": "snapshot",
    "source": "outputs/session2/wpE/source/src_prod_N160_L88.npz",
    "window": [26.0, 30.0], "box": "auto", "sponge_tol": 1e-3, "threads": 2,
    "linear_twin": false, "res_check": "first", "box_check": "first",
    "res_dx": 0.39285714285714285, "box_factor": 1.5, "lumps": {}      # lumps: overrides of the diagnostics block
  },
  "candidates": [
    {"name": "ring5_step72_a8",                   # unique run name
     "family": "ring5",                            # one resolution check and one box check per family
     "positions": [[x, y, z], ...],                # lump centres (the group should sit near the origin)
     "phases_deg": [0, 72, 144, 216, 288],         # one per lump; phi_k ~ Phi cos(omega t + theta_k)
     "start": "snapshot" | "ansatz",
     "box": "auto" | {"L": 70.4, "N": 128},       # auto: smallest box with the sponge guard met
     "T": 500.0, "pattern_axis": [0, 0, 1],
     "linear_twin": true,                          # also write <name>_lin (g4 = g6 = 0)
     "res_check": true, "box_check": true}         # force the checks on this candidate of its family
  ]
}
"first" (default) puts the family's resolution and box checks on its first candidate.

Box "auto": R_ext = max |x_i| (about the origin); clearances from the profile tail: d_sp with
Phi(d_sp) = sponge_tol Phi(0) (lump to sponge, check_lumps) and d_face with Phi(d_face) = 1e-6 Phi(0)
(lump to face, radial_checks); L/2 = R_ext + max(d_sp + sponge width 12, d_face) + 0.5; N = the
smallest even 5-smooth integer >= L/dx; L = N dx. Snapshot starts use one source at the same dx (the resolution
check resamples the source spectrally, "resample": true, recorded in the config).
Each config carries the candidate record ("_candidate") and full parameters; the engine guards
(check_lumps, snapshot.check) run here at generation and again at run start.

Run one config: outputs/session2/wpE/run_one.sh <name>
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
from swirl.engine.lumps import check_lumps, lumps_init, pair_distances  # noqa: E402
from swirl.engine.snapshot import load_meta  # noqa: E402

PROFILE = os.path.join(ROOT, "outputs/step2/profiles/profile_beta1.3333_w0.9091.json")
DEFAULTS = {"dx": 0.55, "dt": 0.04, "T": 500.0, "start": "snapshot",
            "source": "outputs/session2/wpE/source/src_prod_N160_L88.npz", "window": [26.0, 30.0],
            "box": "auto", "sponge_tol": 1e-3, "threads": 2, "linear_twin": False, "res_check": "first",
            "box_check": "first", "res_dx": 88.0 / 224, "box_factor": 1.5, "lumps": {},
            "pattern_axis": [0.0, 0.0, 1.0]}


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


def sponge_clearance(prof, tol):
    spl = radial_spline(prof["r"], prof["Phi"])
    P0 = abs(prof["Phi"][0])
    return float(brentq(lambda r: abs(spl(r)) / P0 - tol, 5.0, prof["r"][-1] - 1.0))


def auto_box(pos, dx, prof, tol):
    R_ext = float(np.max(np.linalg.norm(pos, axis=1)))
    face = sponge_clearance(prof, TAIL_TOL)              # radial_checks: Phi < 1e-6 Phi0 at the faces
    L = 2 * (R_ext + max(sponge_clearance(prof, tol) + SPONGE_DEFAULT["width"], face) + 0.5)
    N = smooth5(L / dx)
    return N, round(N * dx, 9)


def config(c, prof, N, L, dx, g4=None, g6=None, resample=False, note=""):
    pos = np.asarray(c["positions"], float)
    K = len(pos)
    th = np.deg2rad(np.asarray(c["phases_deg"], float))
    if len(th) != K:
        raise ValueError("%s: one phase per lump" % c["name"])
    if c["start"] == "snapshot":
        meta = load_meta(c["source"])
        snap = {"source": c["source"], "sha256": meta["sha256"], "centers": pos.tolist(), "phases": th.tolist(),
                "window": c["window"], "margin": 2.0}
        if resample:
            snap["resample"] = True
        init = {"snapshot": snap}
        om = meta["omega"]
    elif c["start"] == "ansatz":
        init = lumps_init(prof, pos.tolist(), th.tolist(), prof["omega"])
        om = prof["omega"]
    else:
        raise ValueError("start must be snapshot or ansatz")
    hw = prof["half_width"]
    R_ext = float(np.max(np.linalg.norm(pos - pos.mean(0), axis=1)))
    D = pair_distances(pos, 1e9) if K > 1 else np.zeros((1, 1))
    nn = float(np.min(D[np.triu_indices(K, 1)])) if K > 1 else None
    lumps = {"centers": pos.tolist(), "R_ball": 10.0, "taper": 2.0, "split_width": 1.0, "every": 10,
             "count_every": 25.0, "count_periods": 3, "omega_nominal": om, "count_threshold": 0.01, "it_every_step": True,
             "count_min_sep": 1.0, "saddle_merge": 0.9, "avg_window": 20.0, "planes": [],
             "axes": [{"origin": [0.0, 0.0, 0.0], "dir": list(map(float, c["pattern_axis"]))}],
             "group": {"R": R_ext + 3 * hw + 2.0, "taper": 2.0, "axis": list(map(float, c["pattern_axis"]))}}
    lumps.update(c.get("lumps", {}))
    cfg = {"N": int(N), "L": float(L), "m": prof["m"], "g4": prof["g4"] if g4 is None else g4,
           "g6": prof["g6"] if g6 is None else g6, "dt": c["dt"], "t_end": float(c["T"]), "init": init,
           "absorber": True, "kfilter": None, "center": pos[0].tolist(),
           "r_c": [10.0, 3 * hw, float(lumps["group"]["R"])], "sample_every": 1.0, "point_every": 1,
           "avg_window": 20.0, "checkpoint_every": 50.0, "threads": int(c["threads"]), "lumps": lumps,
           "_candidate": {"name": c["name"], "family": c.get("family"), "positions": pos.tolist(),
                          "phases_deg": list(map(float, c["phases_deg"])), "start": c["start"], "K": K,
                          "nearest_spacing": nn, "R_ext": R_ext, "pattern_axis": c["pattern_axis"],
                          "window": c["window"] if c["start"] == "snapshot" else None},
           "_note": note, "_profile_file": os.path.relpath(PROFILE, ROOT)}
    guards = check_lumps(init, cfg["L"], dict(SPONGE_DEFAULT), c["sponge_tol"])
    return cfg, guards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--out", default=os.path.join(HERE, "configs"))
    a = ap.parse_args()
    spec = json.load(open(a.candidates))
    prof = json.load(open(PROFILE))
    dflt = {**DEFAULTS, **spec.get("defaults", {})}
    os.makedirs(a.out, exist_ok=True)
    runs, fam_res, fam_box = {}, set(), set()
    for raw in spec["candidates"]:
        c = {**dflt, **raw}
        dx = c["dx"]
        pos = np.asarray(c["positions"], float)
        if c["box"] == "auto":
            N, L = auto_box(pos, dx, prof, c["sponge_tol"])
        else:
            N, L = int(c["box"]["N"]), float(c["box"]["L"])
            if abs(L / N - dx) > 1e-9 * dx:
                raise ValueError("%s: box L/N = %g differs from dx %g" % (c["name"], L / N, dx))
        runs[c["name"]] = config(c, prof, N, L, dx, note="arrangement %s" % c["name"])
        if c["linear_twin"]:
            runs[c["name"] + "_lin"] = config(c, prof, N, L, dx, g4=0.0, g6=0.0,
                                              note="linear twin (g4 = g6 = 0) of %s" % c["name"])
        fam = c.get("family") or c["name"]
        if c["res_check"] is True or (c["res_check"] == "first" and fam not in fam_res):
            fam_res.add(fam)
            dxf = c["res_dx"]
            Nf = smooth5(L / dxf)
            runs[c["name"] + "_res"] = config({**c, "dx": dxf}, prof, Nf, round(Nf * dxf, 9), dxf,
                                              resample=c["start"] == "snapshot",
                                              note="resolution check of %s at dx = %.4f" % (c["name"], dxf))
        if c["box_check"] is True or (c["box_check"] == "first" and fam not in fam_box):
            fam_box.add(fam)
            Nb = smooth5(c["box_factor"] * L / dx)
            runs[c["name"] + "_box"] = config(c, prof, Nb, round(Nb * dx, 9), dx,
                                              note="box check of %s, L x %.2f" % (c["name"], Nb * dx / L))
    index = {}
    for name, (cfg, guards) in runs.items():
        path = os.path.join(a.out, name + ".json")
        with open(path + ".tmp", "w") as f:
            json.dump(cfg, f)
        os.replace(path + ".tmp", path)
        index[name] = {"N": cfg["N"], "L": cfg["L"], "dx": cfg["L"] / cfg["N"], "T": cfg["t_end"],
                       "K": cfg["_candidate"]["K"], "start": cfg["_candidate"]["start"],
                       "linear": cfg["g4"] == 0 and cfg["g6"] == 0, "note": cfg["_note"], "guards": guards}
    with open(os.path.join(a.out, "index.json.tmp"), "w") as f:
        json.dump(index, f, indent=1)
    os.replace(os.path.join(a.out, "index.json.tmp"), os.path.join(a.out, "index.json"))
    print("\n".join(sorted(runs)))
    print(len(runs), "configs in", a.out)


if __name__ == "__main__":
    main()
