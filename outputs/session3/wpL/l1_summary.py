"""WP-L1 scan summary: one row per run (T = 2000 and 30,000, plus bisection, twins, doubled box).
Writes scan_summary.json, scan_summary.md and the plots. Reads runs/*.json only (no PDE)."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from swirl.family import wpl  # noqa: E402
from swirl.family.wpa_run import write_json  # noqa: E402

LT = "/Users/aaronramos/Desktop/SWIRL/outputs/session2/wpA/lifetime_table_v2.json"
WPA = "/Users/aaronramos/Desktop/SWIRL/outputs/session2/wpA/"
WINDOWS = ((50, 250), (250, 750), (750, 2000), (1000, 2000))
INSTANT = 7.0  # about one oscillation period (2 pi / omega ~ 7)
BETA = {"dip": 4.0 / 3, "primary": 4.0 / 3, "b5": 5.0}


def load_runs():
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, "runs", "*.json"))):
        r = json.load(open(f))
        r["_file"] = os.path.basename(f)
        out.append(r)
    return out


def key_ctrl(p):
    return (p["start"], p["scale"], p["T"])


def amp(p):
    k = p["kind"]
    if k in ("field", "velocity"):
        return p["eps"]
    if k == "removal":
        return p["s"]
    if k == "packet":
        return p["packet"]["efrac"]
    return 0.0


def packet_name(p):
    return None if p["kind"] != "packet" else ("slow" if p["packet"]["k0"] < 0.5 else "k3")


def fit_rate(res, t_lo, t_hi):
    """-d ln E3 / dt over the absolute-time window (linear fit of ln E3)."""
    t = np.asarray(res["series"]["t"])
    E = np.asarray(res["series"]["E3"])
    s = (t >= t_lo) & (t <= t_hi) & (E > 0)
    if s.sum() < 20:
        return None
    return float(-np.polyfit(t[s], np.log(E[s]), 1)[0])


def lifetime_rows():
    d = json.load(open(LT))["rows"]
    return [r for r in d if abs(r["beta"] - 4 / 3) < 1e-3 or abs(r["beta"] - 5) < 1e-3]


def build():
    runs = load_runs()
    ctrls = {key_ctrl(r["params"]): r for r in runs
             if r["params"]["kind"] == "none" and not r["params"]["linear"]}
    fam = lifetime_rows()
    # N_min at each beta from the start's own family row
    nmin = {}
    for st in ("dip", "primary", "b5"):
        b, om, _, nums = wpl.start_numbers(st)
        cand = [r for r in fam if abs(r["beta"] - b) < 1e-3]
        r0 = min(cand, key=lambda r: abs(r["omega_start"] - om))
        nmin[st] = dict(N_profile=nums["N_profile"], family_omega_start=r0["omega_start"],
                        omega_profile=om, omega_mismatch=r0["omega_start"] - om, N_over_Nmin_table=r0["N_over_Nmin"],
                        group=r0.get("group"))
        if st == "primary":  # no table row at omega 0.9091: N_profile / N_min, N_min from the dip row (same beta)
            n_min = nmin["dip"]["N_profile"] / nmin["dip"]["N_over_Nmin_table"]
            nmin[st].update(N_over_Nmin_table=nums["N_profile"] / n_min, N_min=n_min,
                            source="N_profile / N_min with N_min = %.3f from the dip row of lifetime_table_v2 (same beta); "
                                   "the earlier 1.73 was wrong (summary fix a)" % n_min)
    rows, twins = [], []
    tpass = {}

    def t_pass(p):
        k = (p["start"], round(p["packet"]["k0"], 6), p["packet"]["sigma"], p["scale"])
        if k not in tpass:
            fp = wpl.free_packet(p["start"], dict(p["packet"], efrac=0.0) if False else p["packet"], p["scale"])
            tpass[k] = fp["t_pass"]
        return tpass[k]

    for r in runs:
        p = r["params"]
        if p["linear"]:
            fr = np.asarray(r["series"]["frac"])
            t = np.asarray(r["series"]["t"]) - r["t_d"]
            step = float(np.median(np.diff(t)))
            short = (r["t_d"] - p["t_s"]) + step     # a phase-90 kick starts t_d - t_s after the run start

            def ie(tt):
                if tt <= t[-1]:
                    return float(np.interp(tt, t, fr))
                return float(fr[-1]) if tt - t[-1] <= short else None

            twins.append(dict(start=p["start"], kind=p["kind"], amp=amp(p), packet=packet_name(p), params=p,
                              T=p["T"], frac_at_100=ie(100), frac_at_300=ie(300), frac_at_500=ie(500),
                              frac_at_1000=ie(1000), t_last_rel=float(t[-1]), t0=r.get("t0"), file=r["_file"]))
            continue
        if p["kind"] == "none":
            continue
        c = ctrls.get(key_ctrl(p))
        if c is None:
            rows.append(dict(params=p, file=r["_file"], error="no control"))
            continue
        win = WINDOWS + ((p["T"] - 2000, p["T"]),) if p["T"] > 2500 else WINDOWS
        tp_ = None
        if p["kind"] == "packet":
            tp_ = t_pass(p)
            win = win + ((tp_, tp_ + 200), (tp_ + 200, 2000))
        step_ = float(np.median(np.diff(r["series"]["t"])))
        cm = wpl.compare(r, c, windows=win, tol=0.5 * step_ + (r["t_d"] - p["t_s"]))
        t_d = r["t_d"]
        t_end = t_d + p["T"]
        row = dict(start=p["start"], kind=p["kind"], amp=amp(p), packet=packet_name(p), T=p["T"], scale=p["scale"],
                   params=p, file=r["_file"], control_file=c["_file"],
                   t0_dN_over_N=r["t0"]["dN_over_N"] if r["t0"] else 0.0,
                   t0_dE_over_E=r["t0"]["dE_over_Elump"] if r["t0"] else 0.0,
                   first_period_dN_over_N=cm["first_period"]["dN_over_N"] if cm["first_period"] else None,
                   broken=cm["broken"], windows=cm["windows"],
                   dN_over_N_settled=next((w["dN_over_N"] for w in cm["windows"] if w["window"] == [750, 2000]), None),
                   t_pass=tp_,
                   dN_over_N_after_packet=(next((w["dN_over_N"] for w in cm["windows"] if w["window"] == [tp_ + 200, 2000]), None)
                                           if tp_ is not None else None),
                   window_after_packet=([tp_ + 200, 2000] if tp_ is not None else None),
                   window_settled=[750, 2000],
                   window_settled_used=next((w["window_used"] for w in cm["windows"] if w["window"] == [750, 2000]), None),
                   domega_after=next((w["domega"] for w in cm["windows"] if w["window"] == [750, 2000]), None),
                   frac_end=cm["frac_end"], frac_end_control=cm["frac_end_control"],
                   regime=r["regime"], wall_s=r["wall_s"])
        b = row["broken"]
        row["break_class"] = (None if not b["broken_outright"] else
                              "instantaneous" if b["first_below_half"] <= INSTANT else "dynamic")
        g_lo = 500.0 if tp_ is None else max(500.0, tp_ + 200.0)    # packets: after passage (summary fix d)
        row["Gamma_early_window"] = [g_lo, 2000.0]
        row["Gamma_500_2000"] = fit_rate(r, t_d + g_lo, t_d + 2000)
        row["Gamma_500_2000_control"] = fit_rate(c, t_d + g_lo, t_d + 2000)
        gw = cm["gamma"]
        gwr = []
        for i, g in enumerate(gw):
            d = dict(t0_rel=round(g["t0"] - t_d, 1), t1_rel=round(g["t1"] - t_d, 1), Gamma=g["Gamma"],
                     Gamma_control=g["Gamma_control"], sigma=g["sigma"])
            if i > 0 and gw[i - 1]["Gamma"] and gw[i - 1]["Gamma"] > 0:
                d["transient"] = bool(g["Gamma"] < 0.8 * gw[i - 1]["Gamma"])  # fell by more than 20% from the previous window
            gwr.append(d)
        row["gamma_windows"] = gwr
        row["Gamma_still_falling_last_pair"] = (gwr[-1].get("transient") if len(gwr) > 1 else None)
        tt = np.asarray(r["series"]["t"]) - t_d
        fr = np.asarray(r["series"]["frac"])
        ci = int(np.argmin(np.abs(np.asarray(c["series"]["t"]) - t_d)))
        frc = np.asarray(c["series"]["E3"]) / c["series"]["E3"][ci]
        hb = np.where(tt > 0)[0]
        dead = [i for i in hb if fr[i] < 0.5]
        deadc = [i for i in hb if frc[i] < 0.5]
        row["death_time_full"] = float(tt[dead[0]]) if dead else None
        row["death_time_control_full"] = float(tt[deadc[0]]) if deadc else None
        row["frac_at"] = {str(x): float(np.interp(x, tt, fr)) for x in (2000, 5000, 10000, 20000, p["T"]) if x <= tt[-1] + 1}
        if p["T"] > 2500:
            # late windows (t_d + 2000 on) read only while the lump is alive (frac > 0.5 over the window, run and control)
            def alive_late(key):
                out = []
                for g in gw:
                    a, b_ = g["t0"] - t_d, g["t1"] - t_d
                    if a < 2000:
                        continue
                    m = (tt >= a) & (tt <= b_)
                    f_ = fr if key == "Gamma" else frc
                    if m.any() and f_[m].min() > 0.5:
                        out.append(g[key])
                return out
            la, lc = alive_late("Gamma"), alive_late("Gamma_control")
            row["Gamma_late_alive_mean"] = float(np.mean(la)) if la else None
            row["n_late_alive_windows"] = len(la)
            row["Gamma_late_alive_mean_control"] = float(np.mean(lc)) if lc else None
            row["n_late_alive_windows_control"] = len(lc)
            alive_ws = [g for g in gw if (g["t0"] - t_d) >= 2000 and fr[(tt >= g["t0"] - t_d) & (tt <= g["t1"] - t_d)].min() > 0.5] if gw else []
            row["Gamma_last_alive_window"] = alive_ws[-1]["Gamma"] if alive_ws else None
            row["last_alive_window_rel"] = [round(alive_ws[-1]["t0"] - t_d, 1), round(alive_ws[-1]["t1"] - t_d, 1)] if alive_ws else None
            row["Gamma_late_fit_last2000"] = fit_rate(r, t_end - 2000, t_end) if not dead else None
            # family comparison: one estimator (period-averaged N of run over control, window 750..2000)
            st = p["start"]
            if row["dN_over_N_settled"] is not None:
                ratio = nmin[st]["N_over_Nmin_table"] * (1 + row["dN_over_N_settled"])
                cand = [f for f in fam if abs(f["beta"] - BETA[st]) < 1e-3 and f.get("Gamma_last_window") is not None]
                if cand:
                    f = min(cand, key=lambda f: abs(f["N_over_Nmin"] / ratio - 1))
                    fam_run = json.load(open(os.path.join(WPA, f["file"])))
                    # like windows (summary fix c): the run's last alive window [a, b] after t_d, and the family
                    # start over the same elapsed time [a, b] after its own start; one estimator (linear fit of
                    # ln E(r < 3 hw)) for both
                    wr = row["last_alive_window_rel"]
                    g_fam = g_run = None
                    if wr:
                        ft, ff = np.asarray(fam_run["series"]["t"]), np.asarray(fam_run["series"]["frac"])
                        m = (ft >= wr[0]) & (ft <= wr[1]) & (ff > 0)
                        g_fam = float(-np.polyfit(ft[m], np.log(ff[m]), 1)[0]) if m.sum() >= 20 else None
                        g_run = fit_rate(r, t_d + wr[0], t_d + wr[1])
                    row["family"] = dict(N_over_Nmin_eff=ratio, settled_ratio=1 + row["dN_over_N_settled"],
                                         window=[750, 2000], first_period_ratio=(1 + row["first_period_dN_over_N"]) if row["first_period_dN_over_N"] is not None else None,
                                         omega_start=f["omega_start"], N_over_Nmin=f["N_over_Nmin"],
                                         elapsed_window=wr, Gamma_family_same_elapsed=g_fam, Gamma_run_same_window=g_run,
                                         family_T_end=fam_run["T_end"], alive_at_end=f.get("alive_at_end"),
                                         match_rel=f["N_over_Nmin"] / ratio - 1,
                                         comparable=bool(abs(f["N_over_Nmin"] / ratio - 1) <= 0.01), group=f.get("group"))
        stp = 20 if p["T"] < 2500 else 100  # samples are 0.5 apart: every 10 (T = 2000) or 50 (T = 30,000) time units
        i_d = int(np.argmin(np.abs(np.asarray(c["series"]["t"]) - t_d)))
        row["series_dec"] = dict(step_samples=stp, t_rel=(np.asarray(r["series"]["t"])[::stp] - t_d).round(2).tolist(),
                                 frac=np.asarray(r["series"]["frac"])[::stp].round(5).tolist(),
                                 frac_control=(np.asarray(c["series"]["E3"]) / c["series"]["E3"][i_d])[::stp].round(5).tolist())
        rows.append(row)
    return rows, twins, nmin, ctrls


def brackets(rows, T=2000.0):
    """Per start, kind (and packet), side: the outright-break threshold bracket at T, twice:
    all breaks, and dynamic breaks only (breaks within one period of the kick are 'instantaneous')."""
    out = []
    sel = [r for r in rows if r.get("T") == T and r.get("scale") in (1.0, 2.25, 1.25) and "broken" in r]
    groups = {}
    for r in sel:
        groups.setdefault((r["start"], r["kind"], r["packet"]), []).append(r)
    for (st, k, pk), g in sorted(groups.items(), key=lambda x: str(x[0])):
        for side, rr in (("neg", [r for r in g if r["amp"] < 0 and k in ("field", "velocity")]),
                         ("pos", [r for r in g if r["amp"] > 0])):
            if not rr:
                continue
            rr = sorted(rr, key=lambda r: abs(r["amp"]))
            ent = dict(start=st, kind=k, packet=pk, side=side)
            for label, is_b in (("all_breaks", lambda r: r["broken"]["broken_outright"]),
                                ("dynamic_breaks", lambda r: r["break_class"] == "dynamic")):
                brk = [r for r in rr if is_b(r)]
                if not brk:
                    ent[label] = dict(status="no break up to", max_amp=rr[-1]["amp"],
                                      max_t0_dN_over_N=rr[-1]["t0_dN_over_N"])
                    continue
                fb = brk[0]
                below = [r for r in rr if abs(r["amp"]) < abs(fb["amp"]) and not is_b(r)]
                lu = below[-1] if below else None
                ent[label] = dict(status="bracket", unbroken_amp=lu["amp"] if lu else None,
                                  unbroken_t0_dN_over_N=lu["t0_dN_over_N"] if lu else None,
                                  broken_amp=fb["amp"], broken_t0_dN_over_N=fb["t0_dN_over_N"],
                                  broken_time=fb["broken"]["first_below_half"], broken_class=fb["break_class"],
                                  unbroken_amps_above=[r["amp"] for r in rr if abs(r["amp"]) > abs(fb["amp"]) and not is_b(r)])
            out.append(ent)
    return out


def box_compare(rows):
    """Scale-2 runs (T = 30,000) against the matching scale-1 run: differences in the summary numbers."""
    by = {}
    for r in rows:
        if r.get("T") == 30000.0 and "broken" in r and r["scale"] in (1.0, 2.0) and r["packet"] is None:
            by[(r["start"], r["kind"], r["amp"], r["scale"])] = r
    out = []
    for (st, k, a, sc), r2 in sorted(by.items(), key=lambda x: str(x[0])):
        if sc != 2.0:
            continue
        r1 = by.get((st, k, a, 1.0))
        if r1 is None:
            continue
        d = dict(start=st, kind=k, amp=a)
        for key in ("t0_dN_over_N", "dN_over_N_settled", "domega_after", "Gamma_500_2000", "Gamma_late_alive_mean",
                    "Gamma_last_alive_window", "death_time_full", "frac_end"):
            d[key] = [r1.get(key), r2.get(key)]
        d["frac_at"] = [r1["frac_at"], r2["frac_at"]]
        out.append(d)
    return out


if __name__ == "__main__":
    rows, twins, nmin, ctrls = build()
    out = dict(description="WP-L1 scan summary: one row per run; compare() against the same-start, same-scale, same-T control",
               N_min=nmin, rows=rows, linear_twins=twins, brackets=brackets(rows),
               controls={"%s_s%g_T%g" % k: dict(file=v["_file"], frac_end=v["frac_end"], min_frac=v["frac_min"],
                                                regime=v["regime"], omega_s=v["params"]["omega_s"])
                         for k, v in ctrls.items()})
    out["bisection"] = json.load(open(os.path.join(HERE, "bisection.json")))["results"] if os.path.exists(os.path.join(HERE, "bisection.json")) else None
    out["doubled_box"] = box_compare(rows)
    ser = {r["file"]: r.pop("series_dec") for r in rows if "series_dec" in r}
    write_json(os.path.join(HERE, "scan_series_decimated.json"), ser)
    write_json(os.path.join(HERE, "scan_summary.json"), out)
    print(len(rows), len(twins), len(out["brackets"]))
