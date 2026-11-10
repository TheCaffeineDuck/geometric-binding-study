"""Batch A (C-grid) analysis. Every function returns plain dicts with the source file of each number."""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

import lib
from lib import J, S, REL, path, exists, curves, lifetime_row, t_at_cycles, val_at_t, late, early

sys.path.insert(0, lib.ROOT)
from swirl.string import driver as D                    # noqa: E402
from swirl.string.core import String                    # noqa: E402

OBJS = ("sbend", "bump")
RATES = [2.05, 2.2, 2.5, 3.0, 3.5, 4.2]
CEIL = 2.0


def rng(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    return [float(x.min()), float(x.max())] if x.size else None


# ------------------------------------------------------------------ A1
def a1():
    out = {}
    for obj in OBJS:
        nm = "A1_family_grid_G1_%s" % obj
        f = J(nm)["result"]
        pts = f["points"]
        keys = ["omega", "peak_stretch", "peak_displacement", "E", "E_per_cycle", "L_half", "rms_width",
                "nonlinear_share", "tail_kappa_measured", "tail_kappa_O1", "period_return_error",
                "floquet_max_modulus", "floquet_max_modulus_excl_4_at_1",
                "floquet_nearest_kept_dist_from_1", "floquet_unstable_mode_parity"]
        rows = [{k: p.get(k) for k in keys} for p in pts]
        om = np.array([p["omega"] for p in pts])
        Lh = np.array([p["L_half"] for p in pts])
        rms = np.array([p["rms_width"] for p in pts])
        E = np.array([p["E"] for p in pts])
        Epc = np.array([p["E_per_cycle"] for p in pts])
        ex = np.array([p["floquet_max_modulus_excl_4_at_1"] for p in pts])
        kap = np.array([p["tail_kappa_measured"] if p["tail_kappa_measured"] is not None else np.nan
                        for p in pts])
        k1 = np.array([p["tail_kappa_O1"] for p in pts])
        rho_L = spearmanr(om, Lh)
        rho_r = spearmanr(om, rms)
        out[obj] = {
            "source": REL(path(nm)), "n_points": len(pts), "rows": rows,
            "window": {"rate": rng(om), "peak_stretch": rng([p["peak_stretch"] for p in pts]),
                       "peak_displacement": rng([p["peak_displacement"] for p in pts]),
                       "E": rng(E), "E_per_cycle": rng(Epc), "L_half": rng(Lh), "rms_width": rng(rms),
                       "nonlinear_share": rng([p["nonlinear_share"] for p in pts]),
                       "rate_ratio": float(om.max() / om.min()), "L_half_ratio": float(Lh.max() / Lh.min()),
                       "rate_over_ceiling_min": float(om.min() / CEIL),
                       "period_return_error_max": float(max(p["period_return_error"] for p in pts))},
            "spearman_rate_vs_L_half": {"rho": float(rho_L.statistic), "n": len(pts)},
            "spearman_rate_vs_rms_width": {"rho": float(rho_r.statistic), "n": len(pts)},
            "floquet_excl4_max_minus1": float(np.max(ex) - 1), "floquet_excl4_min_minus1": float(np.min(ex) - 1),
            "floquet_raw_max_minus1": float(max(p["floquet_max_modulus"] for p in pts) - 1),
            "floquet_nearest_kept_dist_min": float(min(p["floquet_nearest_kept_dist_from_1"] for p in pts)),
            "tail_kappa_rel_err_max": float(np.nanmax(np.abs(kap / k1 - 1))),
            "tail_kappa_rel_err_median": float(np.nanmedian(np.abs(kap / k1 - 1))),
            "end_reason": [p.get("end_reason") for p in pts if p.get("end_reason")],
        }
    return out


def floq_at(a1res, obj, rate):
    """Largest multiplier (excluding the four at +1) minus 1 at `rate`: log interpolation between the
    bracketing family points; for the stable object the larger of the two neighbours."""
    rows = a1res[obj]["rows"]
    om = np.array([r["omega"] for r in rows])
    ex = np.array([r["floquet_max_modulus_excl_4_at_1"] - 1 for r in rows])
    i = int(np.searchsorted(om, rate))
    i = min(max(i, 1), len(om) - 1)
    lo, hi = ex[i - 1], ex[i]
    if max(lo, hi) < 1e-6:
        return float(max(lo, hi)), "both neighbours below 1e-6"
    l1, l2 = np.log(max(lo, 1e-16)), np.log(max(hi, 1e-16))
    w = (rate - om[i - 1]) / (om[i] - om[i - 1])
    return float(np.exp(l1 + w * (l2 - l1))), "log interpolation between omega = %.4g and %.4g" % (om[i - 1], om[i])


# ------------------------------------------------------------------ A2 / A3 / C-grid common
def trapped_2_1(row, ceiling, res, ncyc=1000):
    """Block 2.1: frac >= 0.90 at 1,000 cycles, twin <= 0.50 at the same time, rate above the
    ceiling by more than the spectral resolution."""
    f = row.get("frac_fixed_%dcyc" % ncyc)
    tw = row.get("twin_frac_%dcyc" % ncyc)
    if f is None or tw is None:
        return None, {"reason": "run shorter than %d cycles" % ncyc}
    rate = row["rate_late"]
    ok = (f >= 0.90) and (tw <= 0.50) and (rate is not None and rate - ceiling > res)
    held_not_stiff = (f >= 0.90) and (tw > 0.50)
    return ("held, but not by stiffening" if held_not_stiff else bool(ok)), \
        {"frac_fixed": f, "twin_frac": tw, "rate_late": rate, "resolution": res}


def centroid_stats(c, cen0=None):
    cen = c["cen"] - (c["cen"][0] if cen0 is None else cen0)
    cyc = c["cyc"]
    absc = np.abs(cen)
    out = {"max_shift": float(absc.max()), "final_shift": float(cen[-1]), "start_centroid": float(c["cen"][0])}
    for thr in (0.25, 0.5, 1.0, 2.0):
        i = np.where(absc >= thr)[0]
        out["cycles_first_shift_%g" % thr] = float(cyc[i[0]]) if i.size else None
    n = len(cen)
    tail = slice(n // 2, n)
    if n > 8 and cyc[-1] > cyc[n // 2]:
        sl = np.polyfit(cyc[tail], cen[tail], 1)[0]
    else:
        sl = 0.0
    out["slope_second_half_sites_per_kcycle"] = float(sl * 1000)
    out["moved_in_second_half"] = float(abs(cen[-1] - cen[n // 2]))
    out["fc_min"] = float(c["fc"].min())
    return out


def a2_row(obj, rate, a1res, ncyc_list=(1000, 10000), prefix="A2"):
    nm = "%s_%s_w%g" % (prefix, obj, rate)
    row, c, tw = lifetime_row(nm, ncyc_list)
    res = J(nm)
    sm = res["summary"]
    row["obj"], row["rate_nominal"] = obj, rate
    row["cap_cycles"] = ncyc_list[-1] if prefix == "A3" else 32000
    row["resolution"] = sm["spectral_resolution_first_window"]
    row["trapped_2_1"], row["trapped_detail"] = trapped_2_1(row, CEIL, row["resolution"])
    row["centroid"] = centroid_stats(c)
    row["peak_u_ratio_end_over_start"] = float(np.nanmean(c["peak_u"][-3:]) / np.nanmean(c["peak_u"][:3]))
    row["tail_kappa_measured"] = sm["tail_kappa_at_late_rate"]
    row["tail_kappa_O1"] = sm["tail_kappa_O1_at_late_rate"]
    row["tail_fit_rate"] = sm["tail_fit_rate"]
    row["spectrum_resolution"] = sm["spectral_resolution_first_window"]
    if a1res is not None and prefix == "A2":
        fq, how = floq_at(a1res, obj, rate)
        row["floquet_excl4_minus1"], row["floquet_how"] = fq, how
    row["twin_frac_end"] = res["twin"]["frac_end"] if res.get("twin") else None
    return row, c, tw


def a2(a1res):
    rows, curves_ = [], {}
    for obj in OBJS:
        for r in RATES:
            nm = "A2_%s_w%g" % (obj, r)
            if not exists(nm):
                continue
            row, c, tw = a2_row(obj, r, a1res)
            rows.append(row)
            curves_[nm] = (c, tw)
    return rows, curves_


def a3(a2rows):
    out = []
    for obj, r in (("sbend", 2.5), ("sbend", 3.5), ("bump", 2.5)):
        base = next((x for x in a2rows if x["obj"] == obj and x["rate_nominal"] == r), None)
        trio = {}
        for sc in (0.99, 1.01):
            nm = "A3_%s_w%g_x%g" % (obj, r, sc)
            if not exists(nm):
                continue
            row, c, tw = lifetime_row(nm, (1000, 10000))
            row["resolution"] = J(nm)["summary"]["spectral_resolution_first_window"]
            row["trapped_2_1"], row["trapped_detail"] = trapped_2_1(row, CEIL, row["resolution"])
            row["centroid"] = centroid_stats(c)
            row["scale"] = sc
            trio[sc] = row
        if base is None or len(trio) < 2:
            out.append({"obj": obj, "rate": r, "incomplete": True})
            continue
        # the 1% rule (Block 6): trapped only if 0.99 and 1.01 times its amplitude are also trapped
        t3 = [base["trapped_2_1"], trio[0.99]["trapped_2_1"], trio[1.01]["trapped_2_1"]]
        robust = all(v is True for v in t3)
        if robust:
            cls = "trapped, robust to 1%"
        elif base["trapped_2_1"] is True:
            cls = "boundary state, not a trap (a scaled start is not trapped)"
        else:
            cls = "not trapped at the exact start"
        out.append({"obj": obj, "rate": r, "base": base["name"], "x0.99": trio[0.99], "x1.01": trio[1.01],
                    "trapped_triplet": t3, "class": cls,
                    "frac_end": [trio[0.99]["frac_end"], base["frac_fixed_10000cyc"], trio[1.01]["frac_end"]],
                    "frac_comoving_end": [trio[0.99]["frac_comoving_end"], base["frac_comoving_10000cyc"],
                                          trio[1.01]["frac_comoving_end"]],
                    "rate_late": [trio[0.99]["rate_late"], base["rate_late"], trio[1.01]["rate_late"]],
                    "T_death_comoving": [trio[0.99]["T_death_comoving"], base["T_death_comoving"],
                                         trio[1.01]["T_death_comoving"]],
                    "sources": [trio[0.99]["source"], base["source"], trio[1.01]["source"]]})
    return out


# ------------------------------------------------------------------ A4
def a4():
    rows = []
    for obj in OBJS:
        for env in (3, 6, 12, 24):
            for pk in (0.3, 0.6):
                nm = "A4_%s_env%d_pk%g" % (obj, env, pk)
                if not exists(nm):
                    continue
                row, c, tw = lifetime_row(nm, (1000,))
                res = J(nm)
                row.update({"obj": obj, "env": env, "pk": pk})
                row["cycles_1000_reached"] = row["frac_fixed_1000cyc"] is not None
                row["period_measured"] = res["start_info"].get("period_measured")
                row["twin_frac_end"] = res["twin"]["frac_end"] if res.get("twin") else None
                row["twin_frac_comoving_end"] = res["twin"]["frac_comoving_end"] if res.get("twin") else None
                row["centroid_shift_end"] = float(c["cen"][-1] - c["cen"][0])
                row["L_half_ratio_end_start"] = row["L_half_late"] / row["L_half_start"]
                # end state from the snapshot: how many separate energy lumps, and does the energy move
                row["snapshot"] = classify_snapshot(nm, res)
                rows.append(row)
    return rows


def classify_snapshot(nm, res):
    """Count energy lumps in the last snapshot (grid): local maxima of the 5-site-smoothed energy
    density above 10% of the largest, separated by at least 3 sites; also the fraction of energy in
    the window and whether its centroid has moved."""
    import glob
    ck = os.path.join(lib.ROOT, "checkpoints", "string")
    snaps = sorted(glob.glob(os.path.join(ck, nm + "_snap_t*.npz")),
                   key=lambda p: float(p.split("_snap_t")[1][:-4]))
    if not snaps:
        return None
    z = np.load(snaps[-1])
    p = res["params"]
    Sx = String(p["medium"], p["law"], p["N"], h=1.0, beta=p.get("beta", 0.0))
    e = Sx.density(z["u"], z["v"])
    k = np.ones(5) / 5
    es = np.convolve(e, k, mode="same")
    big = np.where((es[1:-1] > es[:-2]) & (es[1:-1] >= es[2:]) & (es[1:-1] > 0.1 * es.max()))[0] + 1
    peaks = []
    for i in big[np.argsort(-es[big])]:
        if all(abs(Sx.x[i] - Sx.x[j]) >= 3 for j in peaks):
            peaks.append(i)
    Etot = float(e.sum())
    # energy in the outer region of the interior (waves that have travelled), relative to total
    xs = np.abs(Sx.x - p["x0"])
    out = {"t_snapshot": float(z["t"]), "n_lumps": len(peaks),
           "lump_positions": [float(Sx.x[i]) for i in sorted(peaks)][:12],
           "lump_energy_fractions": [float(es[i] * 5 / Etot) for i in sorted(peaks)][:12],
           "E_total_snapshot": Etot, "frac_E_within_W": float(e[xs <= p["W"]].sum() / Etot),
           "frac_E_beyond_3W": float(e[xs > 3 * p["W"]].sum() / Etot),
           "max_energy_density_over_mean_in_W": float(e[xs <= p["W"]].max() / e[xs <= p["W"]].mean())}
    return out


# ------------------------------------------------------------------ A5
def a5():
    out = {}
    for tag in ("closed", "sponge"):
        nm = "A5_sbend_w2.5_%s" % tag
        if not exists(nm):
            return None
        res = J(nm)
        c = curves(res)
        sm = res["summary"]
        ff = c["ff"]
        d = {"source": REL(path(nm)), "cycles_at_end": sm["cycles_at_end"],
             "fixed_window_drift_max": float(np.max(np.abs(ff - ff[0]))),
             "fixed_window_drift_end": float(ff[-1] - ff[0]),
             "fixed_window_drift_first_record_ref": float(np.max(np.abs(ff - ff[0]))),
             "closed_ring_conservation_error": float(sm["max_abs_balance"]),
             "absorbed_over_E0": float(c["absorbed"][-1]), "frac_end": sm["frac_end"],
             "rate_late": sm["rate_late"]}
        out[tag] = d
        # spectrum at the far probes
        ser = S(nm)
        t = ser["t"]
        probes = ser["probes"].astype(float)
        pd_ = ser["probe_dist"]
        d["probe_dist"] = [float(x) for x in pd_]
        d["probe_span_t"] = float(t[-1])
        rate = sm["rate_late"]
        kap = 2 * np.arccosh(rate / 2)
        d["kappa_O1"] = float(kap)
        d["evanescent_level_exp_minus_kappa_L"] = {str(float(L)): float(np.exp(-kap * L)) for L in pd_[pd_ >= 60]}
        spec = {}
        for i, L in enumerate(pd_):
            if L < 60:
                continue
            q = probes[i]
            q = q - q.mean()
            w = np.hanning(q.size)
            Q = np.abs(np.fft.rfft(q * w)) / (w.sum() / 2)
            om = 2 * np.pi * np.fft.rfftfreq(q.size, d=float(t[1] - t[0]))
            dom = om[1] - om[0]
            lines = {}
            for k_, nlabel in ((1, "rate"), (3, "3rd harmonic"), (5, "5th harmonic"), (2, "2nd harmonic"),
                               (4, "4th harmonic")):
                w0 = k_ * rate
                if w0 > om[-1]:
                    continue
                m = np.abs(om - w0) <= 6 * dom
                amp = float(Q[m].max())
                # local floor: median of the spectrum within 0.5 of the line, excluding +-0.1
                side = (np.abs(om - w0) <= 0.5) & (np.abs(om - w0) > 0.1)
                fl = float(np.median(Q[side]))
                lines[nlabel] = {"omega": float(w0), "amp": amp, "local_floor_median": fl,
                                 "amp_over_floor": amp / fl if fl > 0 else None}
            prop = om < 2.0
            spec[str(float(L))] = {"lines": lines, "max_in_propagating_band_omega_lt_2": float(Q[prop].max()),
                                   "median_in_propagating_band": float(np.median(Q[prop])),
                                   "max_overall": float(Q.max()), "max_overall_omega": float(om[np.argmax(Q)]),
                                   "df_resolution": float(dom), "u_rms": float(q.std())}
        d["far_probe_spectrum"] = spec
    return out


# ------------------------------------------------------------------ A6
def a6():
    out = []
    cases = [("sbend", 2.1), ("sbend", 2.3), ("sbend", 3.0), ("bump", 2.3)]
    lams = [0.05, 0.1, 0.3, 1, 3, 10, 30, 100]
    for obj, om in cases:
        rows = []
        for lam in lams:
            nm = "A6_%s_w%g_l%g" % (obj, om, lam)
            if not exists(nm):
                continue
            res = J(nm)
            ps = D.push_summary(res, 100.0)
            c = curves(res)
            sm = res["summary"]
            ib = np.where(c["fc"] < 0.5)[0]
            dist_at_break = float(abs(c["cen"][ib[0]] - c["cen"][0])) if ib.size else None
            tw = res.get("twin") or {}
            rows.append({"lambda": lam, "name": nm, "source": REL(path(nm)), "speed": ps["speed"],
                         "distance": ps["distance"], "loss_per_distance": ps["energy_lost_per_unit_distance"],
                         "broke": ps["broke"], "t_break": ps["t_break"], "distance_at_break": dist_at_break,
                         "crossed_100": ps["crossed_target"],
                         "frac_comoving_at_100": ps["frac_comoving_at_target"],
                         "frac_comoving_end": ps["frac_comoving_end"],
                         "guard": sm["guard"] is not None,
                         "guard_label": (sm["guard"] or {}).get("rerun_half_dt", {}).get("label"),
                         "guard_t": (sm["guard"] or {}).get("t"), "t_end": sm["t_end"], "cap": sm["cap"],
                         "twin_frac_end": tw.get("frac_end"), "twin_frac_comoving_end": tw.get("frac_comoving_end"),
                         "E0": sm["E0"], "energy_last_good": (sm["guard"] or {}).get("energy_last_good"),
                         "max_abs_centroid": float(np.max(np.abs(c["cen"] - c["cen"][0])))})
        out.append({"obj": obj, "rate": om, "rows": rows})
    return out


def c5_verdict(case):
    """Rule 2.3 C5 for one (object, rate): the lowest push strength crosses 100 units with co-moving
    frac >= 0.90 (as written), and loss per unit distance does not decrease from one speed to the
    next. Break speed = the lowest measured speed at which co-moving frac < 0.5 within 100 units."""
    rows = [r for r in case["rows"] if not r["guard"]]
    allr = case["rows"]
    if not allr:
        return None
    low = allr[0]
    lowest_ok = bool(low["crossed_100"] and low["frac_comoving_at_100"] is not None
                     and low["frac_comoving_at_100"] >= 0.90)
    movers = sorted([r for r in rows if r["speed"] is not None and r["distance"] > 1 and
                     r["loss_per_distance"] is not None], key=lambda r: abs(r["speed"]))
    sp = [abs(r["speed"]) for r in movers]
    ls = [r["loss_per_distance"] for r in movers]
    nondec = all(ls[i + 1] >= ls[i] - 1e-15 for i in range(len(ls) - 1)) if len(ls) > 1 else None
    # alternative reading: the lowest strength that moves the object across 100 units
    cross = [r for r in rows if r["crossed_100"]]
    lowest_mover = min(cross, key=lambda r: r["lambda"]) if cross else None
    alt_ok = bool(lowest_mover and lowest_mover["frac_comoving_at_100"] is not None
                  and lowest_mover["frac_comoving_at_100"] >= 0.9)
    broke = [r for r in rows if r["broke"] and r["distance_at_break"] is not None and r["distance_at_break"] <= 100]
    brk = min(broke, key=lambda r: abs(r["speed"])) if broke else None
    return {"lowest_lambda": low["lambda"], "lowest_crosses_100_frac_ge_0.9": lowest_ok,
            "lowest_crossed": low["crossed_100"], "lowest_frac_at_100": low["frac_comoving_at_100"],
            "loss_nondecreasing_with_speed": nondec, "movers_speed": sp, "movers_loss": ls,
            "alt_reading_lowest_mover_lambda": lowest_mover["lambda"] if lowest_mover else None,
            "alt_reading_lowest_mover_frac_at_100": lowest_mover["frac_comoving_at_100"] if lowest_mover else None,
            "alt_reading_ok": alt_ok,
            "break_speed": abs(brk["speed"]) if brk else None,
            "break_lambda": brk["lambda"] if brk else None}


# ------------------------------------------------------------------ A7
def a7():
    out = []
    for nm, obj, ncyc in (("A7_sbend_w2.5", "sbend", 1000), ("A7_sbend_w3.5", "sbend", 1000),
                          ("A7_bump_w2.5", "bump", 200), ("A7_bump_w2.5_c10", "bump", 10)):
        if not exists(nm):
            continue
        res = J(nm)
        pr = res["profiles"]
        x = np.array(pr["x_rel"])
        mu, md, ru = np.array(pr["mean_u"]), np.array(pr["mean_d"]), np.array(pr["rms_u"])
        p = res["params"]
        # sample-count bias: the sums include both t = 0 and t = T of a periodic orbit
        S_ = D.build(p)
        u0, x0, om, info = D.build_start(S_, p)
        idx = np.round((x + x0)).astype(int) + S_.N // 2
        d0 = S_.stretch(u0)[idx]
        npf = int(np.ceil(p["T"] / p["sample_dt"])) + 1
        mu_c = (npf * mu - u0[idx]) / (npf - 1)
        md_c = (npf * md - d0) / (npf - 1)
        m60 = np.abs(x) <= 60
        # fold onto distance: average of +x and -x (the object is odd/even about its centre, so
        # report |value| maxima and the signed mean at each distance)
        dist = np.arange(0, 61)
        def fold(a):
            return [float(np.mean(a[np.abs(np.abs(x) - r) < 0.5])) for r in dist]
        def foldabs(a):
            return [float(np.max(np.abs(a[np.abs(np.abs(x) - r) < 0.5]))) for r in dist]
        out.append({"name": nm, "source": REL(path(nm)), "obj": obj, "cycles": ncyc, "npf": npf,
                    "max_abs_mean_u_raw": float(np.abs(mu[m60]).max()),
                    "max_abs_mean_d_raw": float(np.abs(md[m60]).max()),
                    "max_abs_mean_u_corrected": float(np.abs(mu_c[m60]).max()),
                    "max_abs_mean_d_corrected": float(np.abs(md_c[m60]).max()),
                    "max_rms_u": float(ru[m60].max()),
                    "rel_mean_u_corrected": float(np.abs(mu_c[m60]).max() / ru[m60].max()),
                    "rel_mean_d_corrected": float(np.abs(md_c[m60]).max() / ru[m60].max()),
                    "rel_mean_u_raw": float(np.abs(mu[m60]).max() / ru[m60].max()),
                    "distance": [int(v) for v in dist], "abs_mean_u_by_distance": foldabs(mu_c),
                    "abs_mean_d_by_distance": foldabs(md_c), "rms_u_by_distance": fold(ru),
                    "centroid_end": res["summary"]["centroid_end"], "frac_end": res["summary"]["frac_end"]})
    return out
