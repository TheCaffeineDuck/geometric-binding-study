"""WP3 task B: write one engine config per run into outputs/step2/configs/ (spec: docs/WP2_PARAMETER_PICK.md).
Run: .venv/bin/python outputs/step2/make_configs.py
"""
import json, os
D = "/Users/aaronramos/Desktop/SWIRL/outputs/step2"
PROF = {"primary": D + "/profiles/profile_beta1.3333_w0.9091.json",
        "secondary": D + "/profiles/profile_beta0.8500_w0.7793.json"}


def cfg(which, N=192, L=70.0, dt=0.04, t_end=1000.0, linear=False, quad_eps=0.0, threads=2):
    p = json.load(open(PROF[which]))
    ent = {"r": p["r"], "Phi": p["Phi"], "center": [0.0, 0.0, 0.0], "phase": 0.0, "omega": 0.0}
    if quad_eps:
        ent["quad_eps"] = quad_eps
    return {"N": N, "L": L, "m": 1.0, "g4": 0.0 if linear else 0.30, "g6": 0.0 if linear else p["g6"],
            "dt": dt, "t_end": t_end, "init": {"radial": [ent], "gaussians": []},
            "absorber": True, "kfilter": None, "center": [0.0, 0.0, 0.0],
            "r_c": [3 * p["half_width"], 5.0, 7.5, 15.0, 20.0],
            "sample_every": 1.0, "point_every": 1, "avg_window": 20.0,
            "checkpoint_every": 50.0, "threads": threads,
            "_note": "WP3 %s profile beta=%.4f omega=%.12g (profile file %s)" % (which, p["beta"], p["omega"], os.path.basename(PROF[which]))}


RUNS = {
    "run1_primary": cfg("primary"),
    "run2_primary_twin": cfg("primary", linear=True),
    "run3_primary_quad": cfg("primary", quad_eps=0.02, threads=4),
    "run4_primary_res": cfg("primary", N=288, dt=0.0267, t_end=300.0, threads=4),
    "run1b_secondary": cfg("secondary"),
    "run2b_secondary_twin": cfg("secondary", linear=True),
    "box_L70_N128": cfg("primary", N=128, L=70.0, dt=0.05, t_end=300.0, threads=4),
    "box_L140_N256": cfg("primary", N=256, L=140.0, dt=0.05, t_end=300.0, threads=4),
}
for k, c in RUNS.items():
    path = "%s/configs/%s.json" % (D, k)
    json.dump(c, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
    print(k, c["N"], c["L"], c["dt"], c["t_end"], c["g4"], c["g6"], c["threads"], [round(x, 3) for x in c["r_c"]])
