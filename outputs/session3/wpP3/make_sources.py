"""WP-P3 snapshot sources: one settled lump per N at beta = 4/3 (swirl.engine.snapshot.make_source).

Lumps (N in units of N0 = 2865.55, the Session 1/2 profile lump):
  N0       omega 0.909149  reuse Session 2's source outputs/session2/wpE/source/src_prod_N160_L88.npz
                           (settled omega 0.909532, sha256 6193e7b1...; same grid as below)
  1.75 N0  omega 0.878988  new: src_p3_N175 (candidate p2_0230)
  2.5 N0   omega 0.863859  new: src_p3_N250 (candidate p2_0242)
Profiles are solved here (swirl.family.profile, tol 1e-10) on r = 0..34 step 0.01 in the step-2 profile format and
written to profiles/. Sources: N = 160, L = 88, dt = 0.04, t_s = 300, 2 threads, windows measured up to (26, 30)
(as Session 2). Each source is about 7,500 + one period of steps at N = 160: about 30 min on 2 threads.

Usage (from the repo root; resumable, stage 1 checkpoints every 50 time units):
  NUMBA_NUM_THREADS=2 <python> outputs/session3/wpP3/make_sources.py profiles      # profiles only (seconds)
  NUMBA_NUM_THREADS=2 <python> outputs/session3/wpP3/make_sources.py N175           # one source
  NUMBA_NUM_THREADS=2 <python> outputs/session3/wpP3/make_sources.py N250
Writes outputs/session3/wpP3/source/<name>.npz (never committed), <name>.json (meta; committed).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

N0 = 2865.545211117668
LUMPS = {"N100": 1.0, "N175": 1.75, "N250": 2.5}
SRC_N100 = "outputs/session2/wpE/source/src_prod_N160_L88.npz"
GRID = {"N": 160, "L": 88.0, "dt": 0.04, "t_s": 300.0, "threads": 2}


def profile_path(key):
    return os.path.join(HERE, "profiles", "profile_beta1.3333_%s.json" % key)


def make_profiles():
    from swirl.arrange import unequal as un
    from swirl.family.profile import profile_curve, solve_profile
    fam = un.Family.from_table()
    os.makedirs(os.path.join(HERE, "profiles"), exist_ok=True)
    out = {}
    for key, mult in LUMPS.items():
        w = fam.omega_of_N(N0 * mult)
        p = solve_profile(w, m=un.M, g4=un.G4, g6=un.G6, tol=1e-10)
        r = np.round(np.arange(0, 34.0 + 1e-9, 0.01), 10)
        Phi = profile_curve(p, r)
        rec = {"name": key, "beta": un.BETA, "omega": w, "m": un.M, "g4": un.G4, "g6": un.G6,
               "units": "Set-B-like: m=1, g4=0.30, g6=beta*g4^2", "dr": 0.01, "r_max": 34.0,
               "n_points": int(r.size), "Phi0": float(Phi[0]), "half_width": p["half_width"], "N": p["N"],
               "E": p["E"], "A": p["A"], "kappa": p["kappa"], "N_over_N0": p["N"] / N0,
               "Phi_rmax_over_Phi0": float(Phi[-1] / Phi[0]), "r": r.tolist(), "Phi": Phi.tolist(),
               "solver": "swirl.family.profile.solve_profile (Set-B units, tol 1e-10), WP-P3"}
        path = profile_path(key)
        with open(path + ".tmp", "w") as f:
            json.dump(rec, f)
        os.replace(path + ".tmp", path)
        out[key] = {k: rec[k] for k in ("omega", "N", "E", "A", "kappa", "half_width", "N_over_N0")}
    print(json.dumps(out, indent=1))
    return out


def source_config(prof):
    return {"N": GRID["N"], "L": GRID["L"], "m": prof["m"], "g4": prof["g4"], "g6": prof["g6"], "dt": GRID["dt"],
            "init": {"radial": [{"r": prof["r"], "Phi": prof["Phi"], "center": [0.0, 0.0, 0.0],
                                 "phase": 0.0, "omega": prof["omega"]}], "gaussians": []},
            "absorber": True, "kfilter": None, "r_c": [10.0, 3 * prof["half_width"], 18.0],
            "sample_every": 1.0, "avg_window": 20.0, "checkpoint_every": 50.0, "threads": GRID["threads"]}


def build(key):
    from swirl.engine.snapshot import WINDOWS_MEASURED, make_source
    prof = json.load(open(profile_path(key)))
    name = "src_p3_%s" % key
    r_in = GRID["L"] / 2 - 12.0
    meta = make_source(name, source_config(prof), GRID["t_s"], os.path.join(HERE, "source"),
                       windows=[w for w in WINDOWS_MEASURED if w[1] <= r_in],
                       checkpoint_dir=os.path.join(ROOT, "checkpoints/session3/wpP3", name), verbose=True)
    print(json.dumps({k: meta[k] for k in ("omega", "omega_drift", "t0", "crop_loss_E_max", "crop_loss_N", "sha256")},
                     indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "profiles":
        make_profiles()
    else:
        build(sys.argv[1])
