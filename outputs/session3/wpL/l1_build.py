"""WP-L L1 build checks and demos (Session 3). Not the production scan.

Usage (from the repo root or a worktree, PYTHONPATH = that root):
    NUMBA_NUM_THREADS=1 python outputs/session3/wpL/l1_build.py STAGE [--workers 2]

Stages:
    settle      settled states (scale 1, the packet scales, scale 2 for the box check) and their records
    t0table     t0_table.json: Delta N / N and Delta E / E at t_d for (a) field, (a) velocity, (b) removal
    demos       one T = 2000 run per disturbance type per start, plus the controls -> runs/
    twins       linear twins (g4 = g6 = 0) of the demos, T = 500 -> runs/
    linearity   small-epsilon field kicks at the dip start, T = 400 -> linearity.json
    box         box-doubling check of one disturbed run (dip, field eps = +0.1, T = 2000) -> box_check.json
    estimator   radial N against the 3D A9 run (Session 2) -> estimator_check.json
    report      demos_summary.json: every demo against its control
    grid        the proposed scan grids as spec lists: scan_specs_T2000.json, scan_specs_T30000.json
    scan        run a spec list: l1_build.py scan --specs scan_specs_T2000.json --workers 2 (resumable)
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from swirl.family import wpl  # noqa: E402
from swirl.family.wpa_run import write_json  # noqa: E402

MAIN_OUT = "/Users/aaronramos/Desktop/SWIRL/outputs"
A9 = os.path.join(MAIN_OUT, "session2", "wpA", "confirm3d", "results", "confirm3d_primary.json")
T_DEMO = 2000.0
T_TWIN = 500.0
SLOW = dict(k0=0.3, sigma=20.0)
SIG3 = 5.0
DEMO_AMP = dict(field=0.1, velocity=0.1, removal=0.2, efrac=0.05)


def k3(start):
    w = wpl.settle(start)["omega_s"]
    return float(np.sqrt(9 * w * w - 1))


def packets(start):
    return [("slow", dict(SLOW)), ("k3", dict(k0=k3(start), sigma=SIG3))]


def demo_specs():
    out = []
    for st in wpl.STARTS:
        out.append(dict(start=st, kind="none", T=T_DEMO))
        out.append(dict(start=st, kind="field", eps=DEMO_AMP["field"], T=T_DEMO))
        out.append(dict(start=st, kind="velocity", eps=DEMO_AMP["velocity"], T=T_DEMO))
        out.append(dict(start=st, kind="removal", s=DEMO_AMP["removal"], T=T_DEMO))
        for _, pk in packets(st):
            sc = wpl.packet_scale(st, pk)
            if sc != 1.0:
                out.append(dict(start=st, kind="none", T=T_DEMO, scale=sc))
            out.append(dict(start=st, kind="packet", packet=dict(pk, efrac=DEMO_AMP["efrac"]), T=T_DEMO, scale=sc))
    return out


def stage_settle():
    keys = set()
    for st in wpl.STARTS:
        keys.add((st, 1.0))
        keys.add((st, 2.0))
        for _, pk in packets(st):
            keys.add((st, wpl.packet_scale(st, pk)))
    rows = {}
    for st, sc in sorted(keys):
        s = wpl.settle(st, sc)
        rows["%s_s%g" % (st, sc)] = {k: v for k, v in s.items() if k not in ("u", "v", "u90", "v90", "grid", "params")}
    write_json(os.path.join(HERE, "settle_summary.json"), rows)


def stage_t0table():
    tab = wpl.t0_table()
    write_json(os.path.join(HERE, "t0_table.json"),
               dict(description="Delta N / N (instantaneous N, ball rN) and Delta E / E_lump (E inside rN) at t_d, "
                                "by quadrature on the settled state (no PDE after settling). field: phi scaled at "
                                "phase 0 (max of phi(0)); velocity: phi_t scaled at phase 90 (downward zero of "
                                "phi(0)); w = 1 to 2 hw, C-inf taper to 0 at 3 hw. removal: phi and phi_t scaled by "
                                "1 - s in r < 1 hw, taper to 0 at 1.5 hw, phase 0.", table=tab))


def stage_demos(workers):
    return wpl.scan(demo_specs(), workers=workers)


def stage_twins(workers):
    specs = [dict(s, T=T_TWIN, linear=True) for s in demo_specs()]
    return wpl.scan(specs, workers=workers)


def stage_linearity(workers):
    T = 400.0
    eps = [-0.01, -0.005, 0.005, 0.01]
    specs = [dict(start="dip", kind="none", T=T)] + [dict(start="dip", kind="field", eps=e, T=T) for e in eps]
    wpl.scan(specs, workers=workers)
    rd = os.path.join(wpl.OUT, "runs")
    load = lambda s: json.load(open(os.path.join(rd, wpl.spec_name(s) + ".json")))
    ctrl = load(specs[0])
    rows = {}
    for s in specs[1:]:
        r = load(s)
        c = wpl.compare(r, ctrl, windows=((50, 250), (250, 390)))
        rows[s["eps"]] = dict(t0_dN_over_N=r["t0"]["dN_over_N"], t0_dE_over_Elump=r["t0"]["dE_over_Elump"],
                              dN_over_N_50_250=c["windows"][0]["dN_over_N"], dN_over_N_250_390=c["windows"][1]["dN_over_N"],
                              domega_50_250=c["windows"][0]["domega"], first_period_dN_over_N=c["first_period"]["dN_over_N"])
    out = dict(rows=rows)
    for key in ("t0_dN_over_N", "dN_over_N_50_250", "dN_over_N_250_390", "domega_50_250"):
        odd1 = 0.5 * (rows[0.005][key] - rows[-0.005][key])
        odd2 = 0.5 * (rows[0.01][key] - rows[-0.01][key])
        even1 = 0.5 * (rows[0.005][key] + rows[-0.005][key])
        out[key] = dict(odd_ratio_2eps_over_eps=odd2 / odd1, slope_per_eps=odd1 / 0.005,
                        even_over_odd_at_0p01=(0.5 * (rows[0.01][key] + rows[-0.01][key])) / odd2,
                        even_over_odd_at_0p005=even1 / odd1)
    write_json(os.path.join(HERE, "linearity.json"), out)
    return out


def stage_box(workers):
    specs = []
    for sc in (1.0, 2.0):
        specs += [dict(start="dip", kind="none", T=T_DEMO, scale=sc),
                  dict(start="dip", kind="field", eps=0.1, T=T_DEMO, scale=sc)]
    wpl.scan(specs, workers=workers)
    rd = os.path.join(wpl.OUT, "runs")
    load = lambda s: json.load(open(os.path.join(rd, wpl.spec_name(s) + ".json")))
    c1, r1, c2, r2 = [load(s) for s in specs]
    f1, f2 = np.asarray(r1["series"]["frac"]), np.asarray(r2["series"]["frac"])
    n = min(f1.size, f2.size)
    t1, t2 = np.asarray(r1["series"]["t"]) - r1["t_d"], np.asarray(r2["series"]["t"]) - r2["t_d"]
    f2i = np.interp(t1, t2, f2)
    sel = t1 <= T_DEMO
    d = np.abs(f1[sel] - f2i[sel])
    cm1, cm2 = wpl.compare(r1, c1), wpl.compare(r2, c2)
    out = dict(spec="dip, field eps = +0.1, T = 2000, scale 1 (R_box 200) against scale 2 (R_box 400)",
               max_abs_frac_diff=float(d.max()), t_of_max=float(t1[sel][np.argmax(d)]),
               frac_end=[float(f1[sel][-1]), float(f2i[sel][-1])],
               dN_over_N_windows=[[w["dN_over_N"] for w in cm1["windows"]], [w["dN_over_N"] for w in cm2["windows"]]],
               domega_windows=[[w["domega"] for w in cm1["windows"]], [w["domega"] for w in cm2["windows"]]],
               note="each box settles its own start from t = 0; times compared relative to t_d")
    write_json(os.path.join(HERE, "box_check.json"), out)
    return out


def stage_estimator():
    rec = json.load(open(os.path.join(wpl.OUT, "settle", "settle_dip_s1.json")))
    pr = rec["periods"]
    tp, Nb = np.asarray(pr["t"]), np.asarray(pr["N_box"])
    out = dict(description="Radial whole-box N (per-period records, mean over the periods inside [t - 20, t]) "
                           "against the 3D A9 run's whole-box N (run.window_averages, W = 20, N = 128, L = 70), "
                           "same start (dip), both from the single-frequency profile at t = 0.", rows=[])
    if os.path.exists(A9):
        a9 = json.load(open(A9))
        t3 = np.asarray(a9["series"]["t"])
        N3 = np.asarray([np.nan if x is None else x for x in a9["series"]["N"]])
        for tt in (50.0, 100.0, 200.0, 300.0, 500.0, 700.0, 1000.0):
            sel = (tp >= tt - 20) & (tp <= tt)
            Nr = float(np.mean(Nb[sel]))
            N3v = float(N3[np.argmin(np.abs(t3 - tt))])
            out["rows"].append(dict(t=tt, N_radial=Nr, N_3d=N3v, rel_diff=Nr / N3v - 1))
        out["max_abs_rel_diff_t_ge_100"] = max(abs(r["rel_diff"]) for r in out["rows"] if r["t"] >= 100)
        out["max_abs_rel_diff_t_ge_300"] = max(abs(r["rel_diff"]) for r in out["rows"] if r["t"] >= 300)
        sel_ = lambda tt: int(np.argmin(np.abs(tp - tt)))
        out["radiation_share_of_box_N_radial"] = {str(tt): float(Nb[sel_(tt)] / np.asarray(pr["N_rN"])[sel_(tt)] - 1)
                                                  for tt in (100.0, 200.0, 300.0, 1000.0)}
        out["note"] = ("Before t of about 300 the whole-box N depends on the box: start-up radiation in transit is "
                       "counted until the sponge takes it (radial sponge from r = 100, 3D from r = 23). The radial box "
                       "holds 1.3e-3 of N outside rN at t = 100 and 1.2e-4 from t = 300 on (radiation_share_of_box_N_radial), "
                       "which accounts for the t = 100 difference. The check is t >= 300.")
    sel = (tp >= 300) & (tp <= 1000)
    Nn = np.asarray(pr["N_rN"])[sel]
    fit = np.polyval(np.polyfit(tp[sel], Nn, 1), tp[sel])
    out["jitter"] = dict(window="t = 300 to 1000, per-period N in the ball rN, dip start",
                         rel_std_about_linear_trend=float(np.std(Nn - fit) / np.mean(Nn)),
                         rel_max_about_linear_trend=float(np.max(np.abs(Nn - fit)) / np.mean(Nn)),
                         trend_rel_per_1000=float(np.polyfit(tp[sel], Nn, 1)[0] * 1000 / np.mean(Nn)))
    for st in wpl.STARTS:
        r = json.load(open(os.path.join(wpl.OUT, "settle", "settle_%s_s1.json" % st)))
        p = r["periods"]
        t_, n_ = np.asarray(p["t"]), np.asarray(p["N_rN"])
        s_ = (t_ >= 500) & (t_ <= 1000)
        f_ = np.polyval(np.polyfit(t_[s_], n_[s_], 1), t_[s_])
        out["jitter_" + st] = dict(rel_std=float(np.std(n_[s_] - f_) / np.mean(n_[s_])),
                                   rel_max=float(np.max(np.abs(n_[s_] - f_)) / np.mean(n_[s_])),
                                   N_settled_rN=r["meta"]["N_s"], N_profile=wpl.start_numbers(st)[3]["N_profile"],
                                   N_period_over_N_profile=r["meta"]["N_s"] / wpl.start_numbers(st)[3]["N_profile"])
    write_json(os.path.join(HERE, "estimator_check.json"), out)
    return out


def stage_report():
    rd = os.path.join(wpl.OUT, "runs")
    load = lambda s: json.load(open(os.path.join(rd, wpl.spec_name(s) + ".json")))
    rows = []
    for s in demo_specs():
        if s["kind"] == "none":
            continue
        r = load(s)
        c = load(dict(start=s["start"], kind="none", T=T_DEMO, scale=s.get("scale", 1.0)))
        win = ((50, 250), (250, 750), (750, 2000))
        extra = {}
        if s["kind"] == "packet":
            fp = wpl.free_packet(s["start"], s["packet"], s["scale"])
            tp_ = fp["t_pass"]
            win = ((tp_, tp_ + 200), (tp_ + 200, T_DEMO))
            extra = dict(free_packet={k: v for k, v in fp.items() if k != "packet"})
        cmp = wpl.compare(r, c, windows=win)
        lin = None
        sl = dict(s, T=T_TWIN, linear=True)
        pth = os.path.join(rd, wpl.spec_name(sl) + ".json")
        if os.path.exists(pth):
            tw = json.load(open(pth))
            tt = np.asarray(tw["series"]["t"]) - tw["t_d"]
            fr = np.asarray(tw["series"]["frac"])
            lin = {str(x): float(fr[np.argmin(np.abs(tt - x))]) for x in (100, 300, 500) if tt[-1] >= x - 1}
        rows.append(dict(name=wpl.spec_name(s), spec=s, t0=r["t0_check"], phase_measured_deg=r["kick"]["phase_measured_deg"],
                         E_plus_absorbed=r["E_plus_absorbed"], regime=r["regime"], compare=cmp,
                         packet_info=r["kick"]["info"], linear_twin_frac=lin, wall_s=r["wall_s"], **extra))
    ctrls = {}
    for s in demo_specs():
        if s["kind"] == "none":
            c = load(s)
            ctrls[wpl.spec_name(s)] = dict(E_plus_absorbed=c["E_plus_absorbed"], regime=c["regime"], frac_end=c["frac_end"],
                                           frac_min=c["frac_min"], wall_s=c["wall_s"], gamma=c["gamma_windows"])
    write_json(os.path.join(HERE, "demos_summary.json"), dict(rows=rows, controls=ctrls))


EPS_SCAN = [-0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, -0.02, -0.01, -0.005,
            0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
S_SCAN = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]
EFRAC_SCAN = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
EPS_LONG = [-0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.1]
S_LONG = [0.02, 0.05, 0.1, 0.2]
EFRAC_LONG = [0.05, 0.2, 0.5]


def grid_specs(T, long=False):
    """The proposed scan grid (not run here). long=True: the degradation points at T = 30,000."""
    eps, ss, ef = (EPS_LONG, S_LONG, EFRAC_LONG) if long else (EPS_SCAN, S_SCAN, EFRAC_SCAN)
    out = []
    for st in wpl.STARTS:
        out.append(dict(start=st, kind="none", T=T))
        out += [dict(start=st, kind=k, eps=e, T=T) for k in ("field", "velocity") for e in eps]
        out += [dict(start=st, kind="removal", s=s_, T=T) for s_ in ss]
        for _, pk in packets(st):
            sc = wpl.packet_scale(st, pk)
            if sc != 1.0:
                out.append(dict(start=st, kind="none", T=T, scale=sc))
            out += [dict(start=st, kind="packet", packet=dict(pk, efrac=e), T=T, scale=sc) for e in ef]
    return out


def stage_grid():
    """Write the proposed scan grids as spec lists (runner input for stage scan)."""
    a, b = grid_specs(2000.0), grid_specs(30000.0, long=True)
    write_json(os.path.join(HERE, "scan_specs_T2000.json"), a)
    write_json(os.path.join(HERE, "scan_specs_T30000.json"), b)
    return dict(n_T2000=len(a), n_T30000=len(b))


def stage_scan(workers, specs_file):
    """Run a spec list (JSON) in a process pool; resumable (results and checkpoints are kept)."""
    return wpl.scan(json.load(open(specs_file)), workers=workers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--specs", default=None)
    a = ap.parse_args()
    f = globals()["stage_" + a.stage]
    if a.stage == "scan":
        out = f(a.workers, a.specs)
    else:
        out = f(a.workers) if a.stage in ("demos", "twins", "linearity", "box") else f()
    if out is not None:
        print(json.dumps(out, default=str)[:3000])
