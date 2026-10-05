"""F-023 diagnostic: does the dip's outright-break threshold sit at a fixed t = 0 dE/E (energy reading) whatever the kick?
Phase 45 is not available in wpl (phase is 0 or 90 only; a new phase needs code), so the kick's dE/dN ratio is varied
through the kick's radial shape instead (spec key 'ball', configuration), for field (phase 0) and velocity (phase 90).
Bisection on eps (negative side), T = 2000, break = compare()['broken']['broken_outright'] against the dip control."""
import json
import sys
import time
from dcommon import HERE, load_ctrl, write_json, wpl
import os

TOL_DE = 0.004   # bracket width in t = 0 dE/E_lump at which a bisection stops
BALLS = [(2.0, 3.0), (1.0, 2.0), (3.0, 4.0)]


def run(kind, eps, ball, ctrl):
    sp = dict(start="dip", kind=kind, eps=eps, T=2000.0, ball=ball)
    t = time.time()
    r = wpl.run_one(sp)
    c = wpl.compare(r, ctrl)["broken"]
    return dict(eps=eps, broken=c["broken_outright"], t_first=c["first_below_half"], run_min_frac=c["run_min_frac"],
                dE_over_E=r["t0"]["dE_over_Elump"], dN_over_N=r["t0"]["dN_over_N"], phase=r["t0"]["phase"], wall_s=time.time() - t)


def bisect(kind, ball, ctrl):
    hist = []
    lo = hi = None
    for eps in (-0.1, -0.2, -0.3, -0.4, -0.5, -0.7):
        e = run(kind, eps, ball, ctrl)
        hist.append(e)
        if e["broken"]:
            hi = e
            break
        lo = e
    if hi is None or lo is None:
        return dict(kind=kind, ball=ball, status="no bracket", history=hist)
    while abs(hi["dE_over_E"] - lo["dE_over_E"]) > TOL_DE and len(hist) < 20:
        e = run(kind, round(0.5 * (lo["eps"] + hi["eps"]), 6), ball, ctrl)
        hist.append(e)
        if e["broken"]:
            hi = e
        else:
            lo = e
    return dict(kind=kind, ball=list(ball), status="ok", unbroken=lo, broken=hi, history=hist,
                dE_over_E_threshold=0.5 * (lo["dE_over_E"] + hi["dE_over_E"]), dN_over_N_threshold=0.5 * (lo["dN_over_N"] + hi["dN_over_N"]),
                bracket_width_dE=abs(hi["dE_over_E"] - lo["dE_over_E"]), broken_time=hi["t_first"])


if __name__ == "__main__":
    which = sys.argv[1]   # "field" or "velocity"
    ctrl = load_ctrl("dip")
    out = []
    for ball in BALLS:
        res = bisect(which, ball, ctrl)
        out.append(res)
        print(which, ball, res.get("dE_over_E_threshold"), res.get("dN_over_N_threshold"), len(res["history"]), flush=True)
        write_json(os.path.join(HERE, "d023_%s_partial.json" % which), out)
    write_json(os.path.join(HERE, "d023_%s_partial.json" % which), out)
