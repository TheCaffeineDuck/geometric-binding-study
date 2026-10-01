"""WP3 run 5 configs (spec: addendum 'run 5' in docs/WP2_PARAMETER_PICK.md).
Run: .venv/bin/python outputs/step2/make_configs_run5.py
Gaussian convention (swirl/engine/initial.py): phi = amp cos(phase) exp(-r^2 / (2 R^2)), minimum image."""
import json, os, sys
sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL")
import numpy as np
from swirl.engine.initial import build_initial, radial_checks
D = "/Users/aaronramos/Desktop/SWIRL/outputs/step2"
P = json.load(open(D + "/profiles/profile_beta1.3333_w0.9091.json"))
C0 = [0.137, -0.071, 0.213]
G = [([2.5, 0.8, -1.1], 0.138), ([-1.2, 2.9, 0.6], 0.138), ([0.4, -1.7, 2.6], -0.138), ([-2.0, -1.5, -1.9], -0.138)]


def cfg(gauss):
    ent = {"r": P["r"], "Phi": P["Phi"], "center": C0, "phase": 0.0, "omega": 0.0}
    return {"N": 128, "L": 70.0, "m": 1.0, "g4": 0.30, "g6": P["g6"], "dt": 0.04, "t_end": 1000.0,
            "init": {"radial": [ent], "gaussians": [{"center": c, "amp": a, "R": 1.0, "phase": 0.0, "omega": 0.0} for c, a in G] if gauss else []},
            "absorber": True, "kfilter": None, "center": C0,
            "r_c": [5.0, 7.5, 3 * P["half_width"], 15.0, 20.0],
            "sample_every": 1.0, "point_every": 1, "avg_window": 20.0, "checkpoint_every": 50.0, "threads": 2,
            "_note": "WP3 run 5%s: primary profile beta=%.4f omega=%.12g, lump at %s; Gaussians exp(-r^2/(2R^2))" % (
                "" if gauss else "r (no Gaussians)", P["beta"], P["omega"], C0)}


for name, g in (("run5_primary_generic", True), ("run5r_primary_offcentre", False)):
    c = cfg(g)
    path = "%s/configs/%s.json" % (D, name)
    json.dump(c, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
    N, L = c["N"], c["L"]
    coords = -L / 2 + (L / N) * np.arange(N)
    phi, pi = build_initial(coords, L, c["init"])
    print(name, "g6", c["g6"], "r_c", c["r_c"], "guards", [radial_checks(p, L) for p in c["init"]["radial"]], "max|phi|", abs(phi).max())
