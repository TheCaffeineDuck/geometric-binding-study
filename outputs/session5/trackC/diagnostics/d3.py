"""D3: what does the A2 bump become? No snapshots were saved, so the A2 bump runs at 2.2, 2.5, 3.0, 3.5 are
re-run with the identical configuration for 10,000 cycles (not 32,000), nonlinear only, with a field snapshot at the end."""
import os, sys, json
os.environ["NUMBA_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np
from swirl.string import driver as D
from swirl.string.core import String
D.OUT = os.path.join(HERE, "runs")
D.CKPT = os.path.join(HERE, "ck")
RUNS = os.path.join(ROOT, "outputs", "session5", "trackC", "runs")
fam = json.load(open(os.path.join(RUNS, "A1_family_grid_G1_sbend.json")))["result"]["points"]
famE = (np.array([p["omega"] for p in fam]), np.array([p["E"] for p in fam]))
out = []
for r in (2.2, 2.5, 3.0, 3.5):
    nm = "D3_bump_w%g" % r
    a2 = json.load(open(os.path.join(RUNS, "A2_bump_w%g.json" % r)))
    p = {k: v for k, v in a2["params"].items() if k in ("kind", "medium", "law", "N", "dt", "sample_dt", "record_periods", "probes", "probe_max", "start", "sponge_len", "sigma_max", "profile_range")}
    p["kind"] = "evolve"
    p["T"] = a2["params"]["T"] * 10000.0 / 32000.0
    p["twin"] = False
    p["snap_times"] = [p["T"] - 2 * p["sample_dt"]]
    p["name"] = nm
    res = D.evolve(p, name=nm, log=lambda *a: None)
    z = np.load(os.path.join(D.CKPT, "%s_snap_t%g.npz" % (nm, p["snap_times"][0])))
    S = String("grid", "G1", p["N"])
    u, v, t = z["u"], z["v"], float(z["t"])
    e = S.density(u, v)
    x = S.x
    m = np.abs(x) < 30
    c = float((e[m] @ x[m]) / e[m].sum())            # energy centroid
    cs = round(c)                                   # nearest site
    cb = np.floor(c) + 0.5                          # nearest bond centre
    cb = cb if abs(cb - c) <= abs(cb + 1 - c) else cb + 1
    def split(c0):
        idx = np.round(2 * c0 - x).astype(int) + S.N // 2 - S.N // 2   # mirror site index
        j = np.round(2 * c0).astype(int) - np.arange(S.N) + 0          # n -> 2*c0 - n (x = n - N//2 shift handled below)
        n = np.arange(S.N) - S.N // 2
        mirror = np.round(2 * c0 - n).astype(int) + S.N // 2
        ok = (mirror >= 0) & (mirror < S.N) & (np.abs(n - c0) <= 20)
        ii, jj = np.where(ok)[0], mirror[ok]
        uo, ue = 0.5 * (u[ii] - u[jj]), 0.5 * (u[ii] + u[jj])
        vo, ve = 0.5 * (v[ii] - v[jj]), 0.5 * (v[ii] + v[jj])
        Eo = 0.5 * np.sum(vo ** 2) + 0.5 * np.sum(np.diff(uo) ** 2)
        Ee = 0.5 * np.sum(ve ** 2) + 0.5 * np.sum(np.diff(ue) ** 2)
        return float(Eo / (Eo + Ee))
    rate = res["summary"]["rate_late"]
    Ew = res["records"][-1]["E_fixed"]
    Efam = float(np.interp(rate, *famE))
    row = {"rate_start": r, "name": nm, "t_snapshot": t, "cycles_at_end": res["summary"]["cycles_at_end"], "centroid": c,
           "centroid_mod_1": float(c % 1.0), "nearest_site": cs, "nearest_bond_centre": float(cb),
           "dist_to_site": abs(c - cs), "dist_to_bond_centre": abs(c - cb),
           "odd_fraction_about_nearest_bond_centre": split(cb), "odd_fraction_about_nearest_site": split(float(cs)),
           "rate_late_core": rate, "rate_late_note": "A2 core rate (displacement at the start site); in the S-bend state this is the rate of u at a site next to the centre",
           "E_window_end": Ew, "E0_fixed": res["summary"]["E_fixed0"], "E_start_total": res["summary"]["E0"],
           "family_S_bend_E_at_late_rate": Efam, "E_over_family_E": Ew / Efam,
           "family_S_bend_rate_at_same_E": float(np.interp(Ew, famE[1], famE[0])), "frac_end": res["summary"]["frac_end"],
           "A2_32000cycle_frac_end": a2["summary"]["frac_end"], "A2_32000cycle_rate_late": a2["summary"]["rate_late"],
           "A2_centroid_end": a2["summary"]["centroid_end"]}
    out.append(row)
    print(row, flush=True)
json.dump(out, open(os.path.join(HERE, "D3.json"), "w"), indent=1)
