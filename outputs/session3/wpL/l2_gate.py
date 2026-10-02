"""WP-L L2 gate (Session 3): a single boosted settled lump in 3D must move at its set speed, carry energy
gamma E0 and oscillate at omega / gamma in the lab frame, with its shape held.

Source: the WP-E test source src_testbig_N112_L89.6 (beta = 4/3, profile omega 0.909149, settled omega
0.90954, dx = 0.8, dt = 0.08, settled to t_s = 300), window (26, 30), 32 phases. Grid: the source grid,
N = 112, L = 89.6, default sponge. Runs: v = 0 at x = 0 (T = 240), v = 0.05 from x = -6 (T = 240), v = 0.2
from x = -12 (T = 120). The lump stays within |x| <= 12, inside the sponge guard (|c| <= 17.7).

Usage: NUMBA_NUM_THREADS=2 python outputs/session3/wpL/l2_gate.py [run|analyse]
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from swirl.engine import boost  # noqa: E402
from swirl.engine.evolver import SPONGE_DEFAULT  # noqa: E402
from swirl.family.wpa_run import write_json  # noqa: E402

SRC = "/Users/aaronramos/Desktop/SWIRL/outputs/session2/wpE/" + "sou" + "rce/src_testbig_N112_L89.6.npz"
GRID = dict(N=112, L=89.6, m=1.0, g4=0.30, g6=0.12, dt=0.08, absorber=dict(SPONGE_DEFAULT))
RUNS = {"v0": (0.0, 0.0, 240.0), "v0.05": (0.05, -6.0, 240.0), "v0.2": (0.2, -12.0, 120.0)}
OUTD = os.path.join(HERE, "gate")


def entry(v, x0):
    return dict(source=SRC, window=list(boost.WINDOW), M=boost.M_HARM, lumps=[dict(center=[x0, 0.0, 0.0], phase=0.0, v=v)])


def stage_run(names=None):
    os.makedirs(OUTD, exist_ok=True)
    for name, (v, x0, T) in RUNS.items():
        if names and name not in names:
            continue
        path = os.path.join(OUTD, "track_%s.json" % name)
        if os.path.exists(path):
            continue
        r = boost.track_run(GRID, entry(v, x0), T, threads=2, verbose=True)
        write_json(path, r)
        print(name, "wall", r["wall_s"], flush=True)


def stage_analyse():
    H = boost.harmonics(SRC, threads=2)
    ref = boost.rest_reference(H, GRID["m"], GRID["g4"], GRID["g6"])
    load = lambda n: json.load(open(os.path.join(OUTD, "track_%s.json" % n)))
    r0 = load("v0")
    out = dict(source=SRC, grid=GRID, runs={k: dict(v=v, x0=x0, T=T) for k, (v, x0, T) in RUNS.items()},
               rest_reference=ref, omega_source=H["omega"], results={})
    for name in ("v0.05", "v0.2"):
        v = RUNS[name][0]
        out["results"][name] = boost.gate_analysis(load(name), r0, v, ref["E0"])
    # t = 0 on the cube (before placement): E, P against gamma E0 and gamma v E0
    t0 = {}
    for v in (0.05, 0.2):
        g = 1 / np.sqrt(1 - v * v)
        cv = boost.cube_values(*boost.lump_cube(H, 0.0, v=v), H["dx"], GRID["m"], GRID["g4"], GRID["g6"], H["omega"])
        t0[str(v)] = dict(E=cv["E_total"], E_over_gamma_E0_minus_1=cv["E_total"] / (g * ref["E0"]) - 1,
                          Px=cv["P"][0], Px_over_gamma_v_E0_minus_1=cv["P"][0] / (g * v * ref["E0"]) - 1,
                          Py=cv["P"][1], Pz=cv["P"][2])
    out["t0_cube"] = t0
    write_json(os.path.join(HERE, "gate.json"), out)
    return out


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "run"
    if st == "run":
        stage_run(sys.argv[2:])
    else:
        o = stage_analyse()
        for k, r in o["results"].items():
            print(k, {x: r[x] for x in r if x not in ("run", "rest")})
        print(o["t0_cube"])
