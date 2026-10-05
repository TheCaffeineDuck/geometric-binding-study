"""F-024 diagnostic: three dip-family starts at N/N_min = 1.821, 1.824, 1.827 (profile N over the dip row's N_min, the
convention of the ledger entry), radial, T = 30,000, control-type runs (no disturbance). Early rate (log fit of E3,
t_d+500..2000) and late rate (mean of 500-wide windows after t_d+2000 while alive), same estimators as l1_summary.py.
Compared with the dip packet rows of scan_summary.json."""
import json
import os
import sys
import numpy as np
from scipy.optimize import brentq
from dcommon import HERE, ROOT, write_json, wpl
from swirl.family.profile import solve_profile
from swirl.family.wpa_run import G4

BETA = 4.0 / 3.0
DIP = json.load(open(os.path.join(ROOT, "outputs", "session3", "wpL", "scan_summary.json")))
NMIN = DIP["N_min"]["dip"]["N_profile"] / DIP["N_min"]["dip"]["N_over_Nmin_table"]
TARGETS = [1.821, 1.824, 1.827]


def N_of(omega):
    return solve_profile(omega, 1.0, 1.0, BETA, tol=1e-10)["N"] / G4


def fit_rate(res, lo, hi):
    t = np.asarray(res["series"]["t"]); E = np.asarray(res["series"]["E3"])
    s = (t >= lo) & (t <= hi) & (E > 0)
    return float(-np.polyfit(t[s], np.log(E[s]), 1)[0]) if s.sum() >= 20 else None


def rates(res):
    t_d = res["t_d"]
    tt = np.asarray(res["series"]["t"]) - t_d
    fr = np.asarray(res["series"]["frac"])
    late = []
    for g in res["gamma_windows"]:
        a, b = g["t0"] - t_d, g["t1"] - t_d
        if a < 2000:
            continue
        m = (tt >= a) & (tt <= b)
        if m.any() and fr[m].min() > 0.5:
            late.append(g["Gamma"])
    dead = np.where(fr < 0.5)[0]
    return dict(early_500_2000=fit_rate(res, t_d + 500, t_d + 2000), late_alive_mean=float(np.mean(late)) if late else None,
                n_late_windows=len(late), frac_end=res["frac_end"], death_time=float(tt[dead[0]]) if dead.size else None)


if __name__ == "__main__":
    out = dict(N_min=NMIN, targets=TARGETS, starts=[])
    for tg in TARGETS:
        om = brentq(lambda w: N_of(w) / NMIN - tg, 0.9050, 0.9075, xtol=1e-12)
        name = "fam%g" % tg
        wpl.STARTS[name] = (BETA, om)
        r = wpl.run_one(dict(start=name, kind="none", T=30000.0), ckpt=os.path.join(HERE, "..", "..", "..", "checkpoints", "session3", "d024_%s.npz" % name))
        row = dict(N_over_Nmin=tg, omega=om, N_profile=N_of(om), omega_s=r["params"]["omega_s"], wall_s=r["wall_s"], **rates(r))
        out["starts"].append(row)
        print(row, flush=True)
        write_json(os.path.join(HERE, "d024_partial.json"), out)
    rows = [x for x in DIP["rows"] if x["start"] == "dip" and x["T"] == 30000.0 and x["kind"] == "packet"]
    out["packet_rows"] = [dict(kind=x["kind"], packet=x["packet"]["k0"], efrac=x["packet"]["efrac"], dN_settled=x["dN_over_N_settled"],
                               early=x["Gamma_500_2000"], late=x["Gamma_late_alive_mean"], early_control=x["Gamma_500_2000_control"], late_control=x["Gamma_late_alive_mean_control"], frac_end=x["frac_end"]) for x in rows]
    out["control_rows"] = DIP["controls"].get("dip_s1_T30000")
    write_json(os.path.join(HERE, "d024_partial.json"), out)
