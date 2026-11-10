"""D4: does the growth of the time-averaged u / stretch track the bump's centroid shift (A7_bump_w2.5, 200 cycles)?"""
import os, sys, json
os.environ["NUMBA_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import numpy as np, lib

nm = "A7_bump_w2.5"
res = lib.J(nm)
ser = lib.S(nm)
c = lib.curves(res)
t = ser["t"]
probes = ser["probes"].astype(float)          # u at distances 0, 20, 50, 80 from the object's start centre
cd = ser["core_stretch"]                      # stretch at the bond of initial peak stretch
cum = lambda q: np.cumsum(q) / np.arange(1, q.size + 1)
sig = {"cum_mean_u_centre": cum(probes[0]), "cum_mean_u_20": cum(probes[1]), "cum_mean_stretch_peak_bond": cum(cd)}
# centroid shift on the series time grid (records are per 10 periods)
cen = np.interp(t, c["t"], np.abs(c["cen"] - c["cen"][0]))
out = {"source": lib.REL(lib.path(nm)), "t_end": float(t[-1]), "centroid_final_shift": float(c["cen"][-1] - c["cen"][0])}
def onset(x, frac=0.05):
    x = np.abs(x)
    i = np.where(x >= frac * x.max())[0]
    return float(t[i[0]]) if i.size else None
out["onset_centroid_5pct_of_max"] = onset(cen)
out["onset_centroid_t_shift_0.1"] = float(t[np.where(cen >= 0.1)[0][0]]) if (cen >= 0.1).any() else None
for k, v in sig.items():
    # baseline-free: remove the value at the end of the first 10 periods
    n0 = 10
    r = {"onset_5pct_of_max": onset(v), "max_abs": float(np.abs(v).max()),
         "corr_with_centroid_shift": float(np.corrcoef(np.abs(v), cen)[0, 1]),
         "corr_after_first_10_cycles": float(np.corrcoef(np.abs(v[int(10 * 2.5):]), cen[int(10 * 2.5):])[0, 1])}
    out[k] = r
# the running mean of |u| over a window vs the instantaneous: also growth of the rms in the far field (probe 20)
w = 250
rm = np.array([probes[1][max(0, i - w):i + 1].mean() for i in range(t.size)])
out["window_mean_u_20_onset"] = onset(rm)
out["corr_windowmean_u20_vs_centroid"] = float(np.corrcoef(np.abs(rm), cen)[0, 1])
# time grid: when does the centroid cross 0.25 and 0.5, in cycles
cyc = t * 2.5 / (2 * np.pi)
out["cycles_centroid_0.25"] = float(cyc[np.where(cen >= 0.25)[0][0]]) if (cen >= 0.25).any() else None
out["cycles_centroid_0.5"] = float(cyc[np.where(cen >= 0.5)[0][0]]) if (cen >= 0.5).any() else None
for k in ("cum_mean_u_centre", "cum_mean_u_20", "cum_mean_stretch_peak_bond"):
    out[k]["onset_cycles"] = None if out[k]["onset_5pct_of_max"] is None else out[k]["onset_5pct_of_max"] * 2.5 / (2 * np.pi)
json.dump(out, open(os.path.join(HERE, "D4.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
