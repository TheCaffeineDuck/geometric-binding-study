"""F-A and F-E (family part): tables from outputs/radial5/family_*.json. Analysis only."""
import json, os
import numpy as np
from scipy.stats import spearmanr
ROOT = "/Users/aaronramos/Desktop/SWIRL/.claude/worktrees/s5-trackF"
OUT = ROOT + "/outputs/session5/trackF"
names = ["O_bump", "O_pinch", "R_bump", "R_pinch"]
fam = {n: json.load(open(f"{ROOT}/outputs/radial5/family_{n}.json")) for n in names}

def branches(d):
    """split rows at folds in omega (index where omega turns); returns list of index arrays"""
    w = np.array([r["omega"] for r in d["rows"]])
    # turning points of omega along the curve
    idx = [0]
    s = np.sign(np.diff(w))
    for i in range(1, len(s)):
        if s[i] != s[i-1] and s[i] != 0 and s[i-1] != 0:
            idx.append(i)
    idx.append(len(w) - 1)
    return [np.arange(a, b + 1) for a, b in zip(idx[:-1], idx[1:])]

res = {}
for n in names:
    d = fam[n]; R = d["rows"]
    w = np.array([r["omega"] for r in R]); E = np.array([r["E"] for r in R])
    hw = np.array([r["hw"] for r in R]); epc = np.array([r["E_per_cycle"] for r in R])
    nl = np.array([r["nonlinear_share_signed"] for r in R]); sp = np.array([r["shoot_param"] for r in R])
    Xm = np.array([r["X_max_cycle"] for r in R])
    out = dict(model=n, n_points=len(R), partial=bool(d.get("partial")),
               rate_range=[float(w.min()), float(w.max())], amplitude_param_range=[float(sp.min()), float(sp.max())],
               hw_range=[float(hw.min()), float(hw.max())], E_range=[float(E.min()), float(E.max())],
               Epc_range=[float(epc.min()), float(epc.max())], nl_range=[float(nl.min()), float(nl.max())],
               Xmax_cycle_max=float(Xm.max()),
               folds_in_omega=d["folds_in_omega"], folds_in_x=d["folds_in_x"], energy_minima=d["energy_minima"],
               end=d["end"]["why"])
    # E interior minimum per family solver; also own check against both ends
    imin = int(np.argmin(E))
    out["E_argmin_index"] = imin; out["E_min"] = float(E[imin]); out["E_min_omega"] = float(w[imin])
    out["E_min_interior"] = bool(0 < imin < len(R) - 1)
    out["E_min_below_both_ends_by_rel"] = float(1 - E[imin] / min(E[0], E[-1]))
    out["spearman"] = {}
    r, p = spearmanr(w, hw); out["spearman"]["all_points_rate_vs_hw"] = dict(rho=float(r), n=len(R))
    for k, b in enumerate(branches(d)):
        if len(b) >= 3:
            r, p = spearmanr(w[b], hw[b])
            out["spearman"][f"branch{k}_idx{b[0]}-{b[-1]}_w{w[b].min():.5f}-{w[b].max():.5f}"] = dict(rho=float(r), n=int(len(b)))
    # rule 2.2 verdict per branch with >=8 points
    verdicts = {}
    for k, v in out["spearman"].items():
        if v["n"] >= 8:
            verdicts[k] = ("tighter-slower (floor)" if v["rho"] >= 0.9 else
                           "tighter-faster (ceiling)" if v["rho"] <= -0.9 else "no monotone relation")
    out["spearman_verdict"] = verdicts
    out["dlnEpc_over_dlnE_family"] = None
    res[n] = out
    # table csv
    with open(f"{OUT}/family_table_{n}.csv", "w") as f:
        f.write("idx,omega,shoot_param,F_peak,hw,E,E_per_cycle,nonlinear_share,X_max\n")
        for i, r_ in enumerate(R):
            f.write("%d,%.8f,%.6g,%.5f,%.4f,%.5f,%.3f,%.5f,%.4f\n" % (i, r_["omega"], r_["shoot_param"], r_["F_peak"], r_["hw"], r_["E"], r_["E_per_cycle"], r_["nonlinear_share_signed"], r_["X_max_cycle"]))

# F-E matched rate and matched energy: bump vs pinch, per model, interpolating on the lower-rate-first monotone branches
def interp_branch(n, key, wq, which):
    d = fam[n]; R = d["rows"]; bs = branches(d)
    out = []
    for b in bs:
        w = np.array([R[i]["omega"] for i in b]); y = np.array([R[i][key] for i in b])
        o = np.argsort(w)
        out.append((w[o], y[o]))
    return out

fe = {}
for m in ["O", "R"]:
    qs = [0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999] if m == "O" else [0.990, 0.992, 0.994, 0.995, 0.996, 0.998, 0.999]
    rows = []
    for wq in qs:
        rec = dict(omega=wq)
        for obj in ["bump", "pinch"]:
            n = f"{m}_{obj}"
            vals = []
            for bi, (w, E) in enumerate(interp_branch(n, "E", wq, 0)):
                if w.min() <= wq <= w.max():
                    hw = np.interp(wq, *interp_branch(n, "hw", wq, 0)[bi]); vals.append((bi, float(np.interp(wq, w, E)), float(hw)))
            rec[obj] = vals
        rows.append(rec)
    fe[m] = dict(matched_rate=rows)
    # matched energy: at E targets, find rate and hw on each family (all branches)
    Es = [3000, 2000, 1000, 500, 400, 350] if m == "R" else [20000, 10000, 8000, 5000, 3000]
    rowsE = []
    for Eq in Es:
        rec = dict(E=Eq)
        for obj in ["bump", "pinch"]:
            n = f"{m}_{obj}"; R = fam[n]["rows"]
            E = np.array([r["E"] for r in R]); w = np.array([r["omega"] for r in R]); hw = np.array([r["hw"] for r in R])
            hits = []
            for i in range(len(R) - 1):
                if (E[i] - Eq) * (E[i+1] - Eq) < 0:
                    t = (Eq - E[i]) / (E[i+1] - E[i]); hits.append((i, float(w[i] + t*(w[i+1]-w[i])), float(hw[i] + t*(hw[i+1]-hw[i]))))
            rec[obj] = hits
        rowsE.append(rec)
    fe[m]["matched_energy"] = rowsE
res["F_E_family"] = fe
json.dump(res, open(f"{OUT}/fa_fe_family.json.tmp", "w"), indent=1)
os.replace(f"{OUT}/fa_fe_family.json.tmp", f"{OUT}/fa_fe_family.json")
for n in names:
    o = res[n]; print(n, o["n_points"], o["rate_range"], "hw", [round(x,2) for x in o["hw_range"]], "Emin", round(o["E_min"],1), "@", round(o["E_min_omega"],5), "interior", o["E_min_interior"], "below ends", round(o["E_min_below_both_ends_by_rel"],3))
    print("  ", o["spearman"]); print("  ", o["spearman_verdict"])
print(json.dumps(fe, indent=0)[:3500])
