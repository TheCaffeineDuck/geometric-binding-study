"""Job A: per-lump early dN/dt (window EARLY_N = [15, 60], same slope() as postprocess.py) of every single-lump run,
and of the pair lumps at d = 24 for comparison. Writes analysis/single_dNdt.json. Uses postprocess.py's own functions."""
import json, os, sys, glob
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); sys.path.insert(0, os.path.join(H, "..", "..", ".."))
import postprocess as pp
prof, fam = pp.load_constants()
out = {"window": list(pp.EARLY_N), "singles": {}, "pairs_d24_lumps": {}}
ref = pp.run_basics("pair_d24_p045", json.load(open(os.path.join(H, "results", "pair_d24_p045.json"))), prof)["_tr"][0]["tN"]
mref = (ref >= pp.EARLY_N[0]) & (ref <= pp.EARLY_N[1])
out["grid_note"] = "dNdt_on_pair_grid: single N interpolated onto the N-sample times of pair_d24_p045 lump 0 (as postprocess.nflow_singles does), slope over the window"
for f in sorted(glob.glob(os.path.join(H, "results", "*.json"))):
    name = os.path.basename(f)[:-5]
    if not (name.startswith("single_x12") or name.startswith("pair_d24")):
        continue
    res = json.load(open(f)); r = pp.run_basics(name, res, prof)
    for k, tr in enumerate(r["_tr"]):
        tN, N = tr["tN"], tr["N"]; m = (tN >= pp.EARLY_N[0]) & (tN <= pp.EARLY_N[1])
        v = dict(dNdt=float(pp.slope(tN[m], N[m])), dNdt_on_pair_grid=float(pp.slope(ref[mref], np.interp(ref[mref], tN, N))), n_samples_window=int(m.sum()), n_samples_pair_grid=int(mref.sum()), N_first=float(N[m][0]), N_last=float(N[m][-1]))
        (out["singles"] if name.startswith("single") else out["pairs_d24_lumps"])[name + ("" if name.startswith("single") else "_lump%d" % k)] = v
p = os.path.join(H, "analysis", "single_dNdt.json"); json.dump(out, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
for k, v in out["singles"].items(): print(k, v["dNdt"], v["dNdt_on_pair_grid"], v["n_samples_window"])
for k, v in out["pairs_d24_lumps"].items(): print(k, v["dNdt"], v["dNdt_on_pair_grid"], v["n_samples_window"])
