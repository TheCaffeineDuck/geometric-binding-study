"""WP-E snapshot sources: one settled lump at the centre of a grid (swirl.engine.snapshot).

Usage (from the repo root; resumable, stage 1 checkpoints every 50 time units):
  NUMBA_NUM_THREADS=2 <python> outputs/session2/wpE/make_source.py prod     # N = 160, L = 88, dx = 0.55
  NUMBA_NUM_THREADS=4 <python> outputs/session2/wpE/make_source.py test     # N = 80, L = 64, dx = 0.8 (tests, smoke)
  NUMBA_NUM_THREADS=4 <python> outputs/session2/wpE/make_source.py testbig  # N = 112, L = 89.6, dx = 0.8 (sponge from
                                                                            # 32.8: windows up to r_out = 30)

Writes outputs/session2/wpE/source/<name>.npz (reference state, never committed),
<name>.json (meta: omega, t0, hash, guards; committed) and <name>_run.json (the stage-1 run).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from swirl.engine.snapshot import WINDOWS_MEASURED, make_source  # noqa: E402

PROFILE = os.path.join(ROOT, "outputs/step2/profiles/profile_beta1.3333_w0.9091.json")
OUT = os.path.join(HERE, "source")

SOURCES = {
    "prod": {"N": 160, "L": 88.0, "dt": 0.04, "t_s": 300.0, "threads": 2},
    "test": {"N": 80, "L": 64.0, "dt": 0.08, "t_s": 300.0, "threads": 4},
    "testbig": {"N": 112, "L": 89.6, "dt": 0.08, "t_s": 300.0, "threads": 4},
}


def source_config(N, L, dt, threads):
    prof = json.load(open(PROFILE))
    return {"N": N, "L": L, "m": prof["m"], "g4": prof["g4"], "g6": prof["g6"], "dt": dt,
            "init": {"radial": [{"r": prof["r"], "Phi": prof["Phi"], "center": [0.0, 0.0, 0.0],
                                 "phase": 0.0, "omega": prof["omega"]}], "gaussians": []},
            "absorber": True, "kfilter": None, "r_c": [10.0, 3 * prof["half_width"], 18.0],
            "sample_every": 1.0, "avg_window": 20.0, "checkpoint_every": 50.0, "threads": threads}


def build(key, windows=None, verbose=True):
    s = SOURCES[key]
    if windows is None:   # windows must lie inside the source's sponge-free interior
        r_in = s["L"] / 2 - 12.0
        windows = [w for w in WINDOWS_MEASURED if w[1] <= r_in]
    name = "src_%s_N%d_L%g" % (key, s["N"], s["L"])
    cfg = source_config(s["N"], s["L"], s["dt"], s["threads"])
    meta = make_source(name, cfg, s["t_s"], OUT, windows=windows,
                       checkpoint_dir=os.path.join(ROOT, "checkpoints/session2/wpE", name), verbose=verbose)
    return os.path.relpath(os.path.join(OUT, name + ".npz"), ROOT), meta


if __name__ == "__main__":
    path, meta = build(sys.argv[1])
    print(path)
    print(json.dumps({k: meta[k] for k in ("omega", "omega_drift", "omega_per_period_last_lag", "t_up", "t0",
                                           "t_ref", "crop", "nonlinear_share_ref", "sha256")}, indent=1))
