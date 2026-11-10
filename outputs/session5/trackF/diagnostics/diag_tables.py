"""Follow-up diagnostics and additive table corrections (analysis only)."""
import json, glob, os
import numpy as np
ROOT = "/Users/aaronramos/Desktop/SWIRL/.claude/worktrees/s5-trackF"; OUT = ROOT + "/outputs/session5/trackF"; D = OUT + "/diagnostics"
def L(p): return json.load(open(p))
def path(tag):
    for d in (OUT + "/runs", D + "/runs"):
        if os.path.exists(f"{d}/{tag}.json"): return f"{d}/{tag}.json"
def twin_frac_at(tw, t):
    ts = np.array(tw["series"]["t"]); return float(np.array(tw["series"]["frac"])[np.argmin(abs(ts - min(t, ts[-1])))]), float(min(t, ts[-1]))
def metrics(tag):
    d = L(path(tag)); r = d["run"]; tw = d.get("twin"); p = r["params"]; st = p["start"]
    a = np.array(r["per_period"]["rows"], float); n = len(a); E0 = r["E0"]; f0 = r["frac_t0"]
    m = dict(tag=tag, model=p["model"], obj=p["obj"], hw=p["hw"], frac0=f0, T_end=r["T_end"], n_periods=n, frac_end_inst=r["frac_end"],
             T_90=r["T_90"], T_death=r["T_death"], nl_profile=st.get("nonlinear_share_signed"), nl_run=r["regime"].get("nonlinear_share_cyc_50_150"),
             rate_start=st.get("omega"), guard=r.get("guard") is not None, E=st.get("E"))
    if n >= 1000:
        k = 999; m["cyc_ref"] = 1000; m["t_ref"] = float(a[k, 0])
    else:
        k = n - 1; m["cyc_ref"] = n; m["t_ref"] = float(a[k, 0])
    m["frac_cyc_ref"] = float(a[k, 2] / E0); m["retained_ref"] = m["frac_cyc_ref"] / f0
    lo, hi = max(0, k - 10), min(n, k + 11); m["rate_at_ref"] = float(np.median(a[lo:hi, 1]))
    # cycle at which the 21-period median rate exceeds 1
    cross = None
    if n > 21:
        med = np.array([np.median(a[max(0, i - 10):i + 11, 1]) for i in range(n)])
        idx = np.where(med > 1.0)[0]; cross = int(idx[0]) + 1 if len(idx) else None
    m["cycle_rate_above_1"] = cross
    if tw:
        m["twin_T_end"] = tw["T_end"]; m["twin_T_death"] = tw["T_death"]
        m["twin_frac_ref"], m["twin_ref_time"] = twin_frac_at(tw, m["t_ref"])
        m["twin_frac_end"] = tw["frac_end"]
    if m["T_death"] is not None: m["Td_over_hw2"] = m["T_death"] / m["hw"] ** 2
    if m.get("twin_T_death") is not None: m["twin_Td_over_hw2"] = m["twin_T_death"] / m["hw"] ** 2
    if m["T_death"] is not None and m.get("twin_T_death"): m["ratio_run_twin"] = m["T_death"] / m["twin_T_death"]
    return m
if __name__ == "__main__":
    import sys
    rows = {}
    for f in sorted(glob.glob(D + "/runs/*.json")):
        t = os.path.basename(f)[:-5]
        try: rows[t] = metrics(t)
        except Exception as e: rows[t] = dict(tag=t, error=repr(e))
    json.dump(rows, open(D + "/diag_metrics.json", "w"), indent=1, default=float)
    for t, m in rows.items(): print(t, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items() if k in ("hw","frac0","T_end","frac_cyc_ref","retained_ref","rate_at_ref","cycle_rate_above_1","T_90","T_death","Td_over_hw2","twin_T_death","twin_Td_over_hw2","ratio_run_twin","twin_frac_ref","nl_run","error")})
