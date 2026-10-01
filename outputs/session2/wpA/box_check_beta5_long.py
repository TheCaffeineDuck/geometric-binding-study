"""Doubled-box long run at beta = 5, w0 = best_by_measured_rate pick, T = 30000.
Standard and doubled box re-run here with identical settings (same rcs, record dt)."""
import os, sys, json, time
os.environ["NUMBA_NUM_THREADS"] = "1"
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL")
from swirl.family.wpa_run import run, profile, grid_for, write_json
OUT = "/Users/aaronramos/Desktop/SWIRL/outputs/session2/wpA"
BETA, W0, T = 5.0, 0.9911904876957681, 30000.0

def job(scale):
    p = profile(BETA, W0)
    lj = json.load(open(os.path.join(OUT, "long_beta5.0000_w0.991190.json")))
    rcs = lj["params"]["rcs"]
    g = grid_for(BETA, scale=scale)
    t0 = time.time()
    r = run(BETA, W0, T, grid=g, rcs=rcs, prof=p, stop_at_death=False)
    return scale, dict(t=r["series"]["t"], frac=r["series"]["E_in"][:, 0] / r["E0"], E_lump=r["series"]["E_in"][:, 1],
                       grid=g, wall=time.time() - t0, T_death=r["T_death"], T_90=r["T_90"])

if __name__ == "__main__":
    with Pool(2) as pl:
        res = dict(pl.map(job, [1.0, 2.0]))
    a, d = res[1.0], res[2.0]
    n = min(a["t"].size, d["t"].size)
    diff = np.abs(a["frac"][:n] - d["frac"][:n])
    dl = np.abs(a["E_lump"][:n] - d["E_lump"][:n]) / a["E_lump"][:n]
    lj = json.load(open(os.path.join(OUT, "long_beta5.0000_w0.991190.json")))
    out = dict(beta=BETA, omega=W0, T=float(a["t"][n-1]), grids=[a["grid"], d["grid"]],
               max_abs_diff_frac=float(diff.max()), t_at_max=float(a["t"][int(diff.argmax())]), target=1e-4,
               max_rel_diff_E_lump=float(dl.max()),
               max_abs_diff_by_5000={"%d-%d" % (lo, lo+5000): float(diff[(a["t"][:n] >= lo) & (a["t"][:n] < lo+5000)].max()) for lo in range(0, 30000, 5000)},
               frac_end=[float(a["frac"][n-1]), float(d["frac"][n-1])],
               frac_end_stored_long=lj["frac_end"], T_death=[a["T_death"], d["T_death"]], T_90=[a["T_90"], d["T_90"]],
               walls=[a["wall"], d["wall"]])
    write_json(os.path.join(OUT, "box_check_beta5_long.json"), out)
    print(json.dumps(out, default=str))
