"""Lifetime table from run JSON in outputs/session5/trackF/runs (analysis only; reads run + twin series)."""
import json, glob, os, sys
import numpy as np
ROOT = "/Users/aaronramos/Desktop/SWIRL/.claude/worktrees/s5-trackF"
OUT = ROOT + "/outputs/session5/trackF"

def cyc_table(run):
    pp = run["per_period"]["rows"]
    if not pp: return None
    a = np.array(pp, float)
    return a  # t_mid, rate, E_in_3hw_mean, E_in_rlump_mean, Epc_rlump, nl_share

def summarize(path):
    d = json.load(open(path)); r = d["run"]; tw = d.get("twin")
    ec = d["end_class"]
    p = r["params"]; st = p["start"]
    E0 = r["E0"]
    a = cyc_table(r)
    out = dict(tag=d["tag"], model=p["model"], obj=p["obj"], start=d["command"]["start"], scale=d["command"]["scale"],
               T_run=r["T_end"], omega_profile=st.get("omega"), hw=p["hw"], E=st.get("E"), E0_grid=E0,
               frac_t0=r["frac_t0"], frac_t50=r["frac_t50"], frac_end=r["frac_end"], T_90=r["T_90"], T_90_cyc=r.get("T_90_cyc"),
               T_death=r["T_death"], cycles_T_death=r.get("cycles_T_death"), cycles_T_end=r["cycles_T_end"],
               end_class=ec, guard=r.get("guard"), regime=r["regime"])
    w_late = r["regime"]["rate_end_last100"] or r["regime"]["rate_settled_50_150"]
    out["late_rate"] = w_late
    out["rate_minus_floor"] = None if w_late is None else w_late - 1.0
    if a is not None and len(a) > 5:
        # cycle count integrates the measured per-period rate (one row per period)
        n = len(a); out["n_periods_logged"] = n
        out["Epc_first"] = float(a[0, 4]); out["Epc_last"] = float(a[-1, 4])
        out["E_in_rlump_first"] = float(a[0, 3]); out["E_in_rlump_last"] = float(a[-1, 3])
        # frac at 1000 cycles (cycle-averaged region energy / E0) if reached
        if n >= 1000:
            row = a[999]; out["t_1000_cycles"] = float(row[0]); 
            out["frac_cyc_1000"] = float(row[2] / E0); out["frac_cyc_1000_rlump"] = float(row[3] / E0)
        else:
            out["t_1000_cycles"] = None
            out["frac_cyc_end"] = float(a[-1, 2] / E0)
        out["frac_cyc_first"] = float(a[0, 2] / E0)
        out["frac_cyc_rel_first"] = None
    # instantaneous frac series at 1000 cycles or end
    t = np.array(r["series"]["t"]); fr = np.array(r["series"]["frac"])
    if w_late:
        T1000 = 1000 * 2 * np.pi / w_late
        out["T_1000_cycles_from_late_rate"] = T1000
        if t[-1] >= T1000:
            out["frac_inst_1000"] = float(fr[np.argmin(abs(t - T1000))])
    out["frac_inst_end"] = float(fr[-1])
    if tw:
        tt = np.array(tw["series"]["t"]); ft = np.array(tw["series"]["frac"])
        out["twin_T_end"] = tw["T_end"]; out["twin_frac_end"] = float(ft[-1]); out["twin_T_death"] = tw["T_death"]
        tref = t_ref = min(tt[-1], t[-1])
        out["twin_frac_at_run_end_or_1000cyc"] = float(ft[np.argmin(abs(tt - min(tt[-1], out.get("T_1000_cycles_from_late_rate", tt[-1]) if t[-1] >= out.get("T_1000_cycles_from_late_rate", 1e99) else t[-1])))])
        a2 = cyc_table(tw)
        out["twin_rate_late"] = tw["regime"]["rate_end_last100"]
        out["twin_nonlinear_share_cyc_50_150"] = tw["regime"]["nonlinear_share_cyc_50_150"]
    rg = r["regime"]
    out["nonlinear_share_profile"] = st.get("nonlinear_share_signed")
    out["nonlinear_share_run_first_period"] = rg.get("nonlinear_share_cyc_first_period")
    out["nonlinear_share_run_50_150"] = rg.get("nonlinear_share_cyc_50_150")
    out["X_max_all_stages"] = rg.get("X_max_all_stages")
    # F4: change in ln E and ln Epc over the run's cycle-averaged rlump energy
    if a is not None and len(a) > 10:
        E_in = a[:, 3]; epc = a[:, 4]
        # use whole table; E in rlump measure of the lump's energy
        out["dlnE_rlump_first_to_last"] = float(np.log(E_in[-1] / E_in[0]))
        out["dlnEpc_rlump_first_to_last"] = float(np.log(epc[-1] / epc[0]))
    return out

if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    rows = []
    for f in sorted(glob.glob(OUT + f"/runs/{prefix}*.json")):
        try: rows.append(summarize(f))
        except Exception as e: print("FAIL", f, repr(e))
    tmp = OUT + "/lifetime_all.json.tmp"; json.dump(rows, open(tmp, "w"), indent=1, default=float); os.replace(tmp, OUT + "/lifetime_all.json")
    for x in rows:
        print("%-24s w=%.4f cls=%-28s fr0=%.3f fr_end=%.4f Td=%s T90=%s w_late=%s twin_end=%s" % (x["tag"], x["omega_profile"] or 0, x["end_class"]["cls"], x["frac_t0"], x["frac_end"], x["T_death"], x["T_90"], x["late_rate"] and round(x["late_rate"],4), x.get("twin_frac_end")))
