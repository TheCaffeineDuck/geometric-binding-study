"""Compare wpD (default pair law) with wpD_measured (measured kappa, A). Writes compare_default.json."""
import json, os
H = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(H, "..", "wpD")
def L(d, f): return json.load(open(os.path.join(d, f)))
cd, cm = L(W, "candidates.json"), L(H, "candidates.json")
pd_, pm = L(W, "predictions_point_model.json"), L(H, "predictions_point_model.json")
def key(b): return (b["model"], b["arrangement"], b["pattern_index"], round(b["kappa_R"], 1) if b["model"] == "A" else b["pattern"])
out = {"balances": {}, "outcome_changes": [], "n_runs": 0}
for m in "AB":
    out["balances"]["n_model_" + m] = [sum(b["model"] == m for b in c["ranked_balances"]) for c in (cd, cm)]
rows = []
bm = {}
for b in cm["ranked_balances"]: bm.setdefault((b["model"], b["arrangement"], b["pattern_index"]), []).append(b)
for b in cd["ranked_balances"]:
    cands = bm.get((b["model"], b["arrangement"], b["pattern_index"]), [])
    n = min(cands, key=lambda x: abs(x["kappa_R"] - b["kappa_R"])) if cands else None
    f = lambda x: None if x is None else dict(kR=x["kappa_R"], R=x["R"], nn_hw=x["nn_spacing_hw"], neg=x["n_negative"], efold=x["efold_time"], unstable=x["n_unstable_full"])
    rows.append(dict(model=b["model"], arr=b["arrangement"], pat=b["pattern"], default=f(b), measured=f(n)))
out["balances"]["rows"] = rows
def idx(p):
    d = {}
    for e in p["runs"]:
        role = "; ".join("%s: %s" % (r["role"], r["for"]) for r in e["roles"])
        nb = {"A": 0, "B": 0}
        for r in e["runs"]:
            if "balance" in r["size"]:
                nb[r["model"]] += 1; sz = "balance#%d" % nb[r["model"]]
            else: sz = round(r["nn_spacing0"], 3)
            d[(e["arrangement"], tuple(e["theta_deg"]), r["model"], sz)] = (role, r)
    return d
a, b2 = idx(pd_), idx(pm)
for k in a:
    out["n_runs"] += 1
    if k not in b2:
        out.setdefault("absent_in_measured", []).append(str(k)); continue
    ra, rb = a[k][1], b2[k][1]
    if ra["outcome"] != rb["outcome"]:
        out["outcome_changes"].append(dict(arr=k[0], theta=k[1], model=k[2], size=k[3], role=a[k][0], default=(ra["outcome"], ra["outcome_time"], ra["size"]), measured=(rb["outcome"], rb["outcome_time"], rb["size"])))
out["missing_in_default"] = [str(k) for k in b2 if k not in a]
json.dump(out, open(os.path.join(H, "compare_default.json"), "w"), indent=1)
print(out["balances"]["n_model_A"], out["balances"]["n_model_B"], out["n_runs"], len(out["outcome_changes"]), out["missing_in_default"][:3])
for r in rows:
    if r["model"] == "A" or r["measured"] is None or abs(r["measured"]["kR"] / r["default"]["kR"] - 1) > 0.01: print(r["model"], r["arr"], r["pat"][:30], r["default"], r["measured"])
print("absent", out.get("absent_in_measured"))
for c in out["outcome_changes"]: print(c)
