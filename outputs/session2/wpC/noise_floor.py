"""Centroid noise floor on the pair grid (WP-C item 6).

Runs the single off-centre lump config single_x12_t00.0 (x = +12, theta = 0, N = 160,
L = 88) to T = 60 and reports, for the period-averaged ball centroid X(t):
  drift: X(t) - X(first window); jitter: rms and max of the residual about a
  quadratic fit (what a pair fit can resolve); and the same for N and E.
Usage: NUMBA_NUM_THREADS=2 <venv python> outputs/session2/wpC/noise_floor.py [T]
Writes outputs/session2/wpC/noise_floor_single_x12.json (summary only).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from swirl.engine.run import run  # noqa: E402


def main():
    T = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    cfg = json.load(open(os.path.join(HERE, "configs", "single_x12_t00.0.json")))
    cfg["t_end"] = T
    res = run(cfg)
    av = res["lumps"]["averages"]["lumps"][0]
    ok = [i for i, x in enumerate(av["X"]) if x is not None]
    tc = np.array([av["t_center"][i] for i in ok])
    X = np.array([av["X"][i] for i in ok])
    out = {"config": "single_x12_t00.0", "T": T, "params": res["params"],
           "steps_per_second": res["steps_per_second_this_call"],
           "t_center": tc.tolist(), "X": X.tolist(),
           "N": [av["N"][i] for i in ok], "E": [av["E"][i] for i in ok], "omega": [av["omega"][i] for i in ok]}
    for a, name in enumerate("xyz"):
        y = X[:, a]
        c = np.polyfit(tc, y, 2)
        r = y - np.polyval(c, tc)
        out["centroid_" + name] = {"start": float(y[0]), "end": float(y[-1]), "drift_end_minus_start": float(y[-1] - y[0]),
                                   "max_abs_dev_from_start": float(np.max(np.abs(y - y[0]))),
                                   "quadratic_fit_residual_rms": float(np.sqrt(np.mean(r ** 2))),
                                   "quadratic_fit_residual_max": float(np.max(np.abs(r))),
                                   "fit_acceleration": float(2 * c[0])}
    path = os.path.join(HERE, "noise_floor_single_x12.json")
    with open(path + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(path + ".tmp", path)
    print(json.dumps({k: v for k, v in out.items() if k.startswith("centroid") or k == "steps_per_second"}, indent=1))


if __name__ == "__main__":
    main()
