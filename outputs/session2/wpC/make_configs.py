"""WP-C pair batch: write every run config to outputs/session2/wpC/configs/.

Design (docs/WPC_BUILD_NOTES.md): box L = 88, N = 160 (dx = 0.55), dt = 0.04, default
spherical sponge (gamma 2, width 12, cubic; r_in = 32), lumps at x = -+d/2 on the x
axis, phases assigned symmetrically theta_{1,2} = -/+ dtheta/2 (so the two lumps are
time-reversed images with equal slice energy; a theta = 90 start, which fails the
phase control, is never used).

Run one config (resumable; atomic result JSON with full parameters):
  cd /Users/aaronramos/Desktop/SWIRL && NUMBA_NUM_THREADS=2 caffeinate -i .venv/bin/python -m swirl.engine.run \
     outputs/session2/wpC/configs/<name>.json --out outputs/session2/wpC/results/<name>.json \
     --checkpoint-dir checkpoints/session2/wpC/<name> --verbose >> outputs/session2/wpC/logs/<name>.log 2>&1
(outputs/session2/wpC/run_one.sh <name> does exactly this.)

Usage: <venv python> outputs/session2/wpC/make_configs.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from swirl.engine.lumps import lumps_init, check_lumps  # noqa: E402
from swirl.engine.evolver import SPONGE_DEFAULT  # noqa: E402

PROFILE = os.path.join(ROOT, "outputs/step2/profiles/profile_beta1.3333_w0.9091.json")
CONF = os.path.join(HERE, "configs")

N, L, DT = 160, 88.0, 0.04
SEPS = [16.0, 20.0, 24.0]
DPHIS = [0, 45, 90, 135, 180]
T_PAIR = 500.0
T_PAIR_D16 = 300.0   # d = 16: in phase merges by ~126 (point model), 90 deg reaches dtheta ~ 224 by 300
T_TWIN = 300.0
T_CHECK = 300.0


def base(prof, centers, phases, T, n=N, box=L, g4=None, g6=None, note=""):
    init = lumps_init(prof, centers, phases, prof["omega"])
    hw = prof["half_width"]
    cfg = {"N": n, "L": box, "m": prof["m"], "g4": prof["g4"] if g4 is None else g4,
           "g6": prof["g6"] if g6 is None else g6, "dt": DT, "t_end": T, "init": init,
           "absorber": True, "kfilter": None, "center": list(centers[0]),
           "r_c": [10.0, 3 * hw, 15.0], "sample_every": 1.0, "point_every": 1, "avg_window": 20.0,
           "checkpoint_every": 50.0, "threads": 2,
           "lumps": {"centers": [list(c) for c in centers], "R_ball": 10.0, "taper": 2.0, "split_width": 1.0,
                     "every": 10, "count_every": 25.0, "count_periods": 3, "omega_nominal": prof["omega"],
                     "count_threshold": 0.01, "count_min_sep": 1.0, "saddle_merge": 0.9, "avg_window": 20.0,
                     "planes": [{"axis": 0, "pos": 0.0}, {"axis": 0, "pos": -4.0}, {"axis": 0, "pos": 4.0}],
                     "axes": [{"origin": [0, 0, 0], "dir": [0, 0, 1]}, {"origin": [0, 0, 0], "dir": [1, 0, 0]}]},
           "_note": note, "_profile_file": os.path.relpath(PROFILE, ROOT)}
    check_lumps(init, box, dict(SPONGE_DEFAULT), 1e-3)
    return cfg


def main():
    prof = json.load(open(PROFILE))
    os.makedirs(CONF, exist_ok=True)
    runs = {}
    for d in SEPS:
        c = [[-d / 2, 0.0, 0.0], [d / 2, 0.0, 0.0]]
        for p in DPHIS:
            th = np.deg2rad(p)
            ph = [0.0, np.pi] if p == 180 else [-th / 2, th / 2]
            T = T_PAIR_D16 if d == 16.0 else T_PAIR
            runs["pair_d%02d_p%03d" % (d, p)] = base(
                prof, c, ph, T, note="pair d=%g dtheta=%d deg, theta=%s rad" % (d, p, ph))
        runs["twin_d%02d_p000" % d] = base(prof, c, [0.0, 0.0], T_TWIN, g4=0.0, g6=0.0,
                                           note="linear twin (g4 = g6 = 0) of pair d=%g dtheta=0" % d)
        if d == 16.0:   # added after review (Block 2a): one anti-phase twin, run on the Mac
            runs["twin_d16_p180"] = base(prof, c, [0.0, np.pi], T_TWIN, g4=0.0, g6=0.0,
                                         note="linear twin (g4 = g6 = 0) of pair d=16 dtheta=180")
    for th in (0.0, 22.5, 45.0, 67.5):
        runs["single_x12_t%04.1f" % th] = base(
            prof, [[12.0, 0.0, 0.0]], [np.deg2rad(th)], T_PAIR,
            note="single lump at x = +12 (largest pair offset), theta = %g deg: 3D phase control and, "
                 "for theta = 0, the sponge-asymmetry drift control" % th)
    runs["check_res_d20_p000_N224"] = base(prof, [[-10.0, 0, 0], [10.0, 0, 0]], [0.0, 0.0], T_CHECK, n=224,
                                           note="resolution check: pair d=20 dtheta=0 at dx = 0.393")
    runs["check_box_d24_p000_L132"] = base(prof, [[-12.0, 0, 0], [12.0, 0, 0]], [0.0, 0.0], T_CHECK, n=240,
                                           box=132.0, note="box check: pair d=24 dtheta=0, L = 132 (same dx)")
    for name, cfg in runs.items():
        path = os.path.join(CONF, name + ".json")
        with open(path + ".tmp", "w") as f:
            json.dump(cfg, f)
        os.replace(path + ".tmp", path)
    print("\n".join(sorted(runs)))
    print(len(runs), "configs")


if __name__ == "__main__":
    main()
