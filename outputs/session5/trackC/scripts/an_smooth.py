"""Batch B (C-smooth) analysis."""
import glob
import os
import sys
from math import erfc, exp, pi, sqrt

import numpy as np
from scipy.stats import spearmanr

import lib
from lib import J, S, REL, path, exists, curves, lifetime_row, t_at_cycles, val_at_t, late, early

sys.path.insert(0, lib.ROOT)
from swirl.string import diag as G                      # noqa: E402
from swirl.string import driver as D                    # noqa: E402

HS = [(0.25, "h4"), (0.125, "h8"), (0.0625, "h16")]
CEIL = 1.0


def bstarts():
    L = []
    for sg in (0.25, 0.5, 1, 2):
        L.append(("B1_bump_pk0.8_s%g" % sg, "B1 bump pk0.8", "bump", {"sigma": sg, "pk": 0.8}))
    for sg in (0.25, 0.5):
        L.append(("B1_bump_pk0.5_s%g" % sg, "B1 bump pk0.5", "bump", {"sigma": sg, "pk": 0.5}))
    for sg in (0.25, 0.5, 1, 2):
        L.append(("B2_sbend_pk0.8_s%g" % sg, "B2 sbend pk0.8", "sbend", {"sigma": sg, "pk": 0.8}))
    for obj, om in (("sbend", 1.2), ("sbend", 1.5), ("sbend", 2.0), ("bump", 1.2), ("bump", 1.5)):
        L.append(("B3_jump_%s_w%g" % (obj, om), "B3 jump %s" % obj, obj, {"omega": om}))
    L.append(("B4_orbit_sbend_s0.5_pk0.8", "B4 exact state sbend", "sbend", {}))
    return L


def probe_split(name, idx=2):
    try:
        ser = S(name)
    except Exception:
        return None
    if ser["probes"].shape[0] <= idx:
        return None
    r = lib.spectrum_split(ser["t"], ser["probes"][idx].astype(float), 1.0)
    if r:
        r["probe_distance"] = float(ser["probe_dist"][idx])
    return r


def b_row(tag, h, ht):
    nm = "%s_%s" % (tag, ht)
    if not exists(nm):
        return None
    row, c, tw = lifetime_row(nm, (300, 1000))
    res = J(nm)
    sm = res["summary"]
    row["h"] = h
    row["resolution"] = sm["spectral_resolution_first_window"]
    # frac reference: at 1,000 cycles if reached, else at the end of the run
    reached = row["frac_fixed_1000cyc"] is not None
    row["frac_ref_cycles_reached"] = reached
    row["frac_ref"] = row["frac_fixed_1000cyc"] if reached else row["frac_end"]
    row["frac_comoving_ref"] = row["frac_comoving_1000cyc"] if reached else row["frac_comoving_end"]
    row["twin_frac_ref"] = row["twin_frac_1000cyc"] if reached else (res["twin"]["frac_end"] if res.get("twin") else None)
    row["rate_late_over_1"] = row["rate_late"]
    ok = (row["frac_ref"] >= 0.90) and (row["twin_frac_ref"] is not None and row["twin_frac_ref"] <= 0.50) and \
        (row["rate_late"] is not None and row["rate_late"] - CEIL > row["resolution"])
    held_ns = (row["frac_ref"] >= 0.90) and (row["twin_frac_ref"] is not None and row["twin_frac_ref"] > 0.50)
    row["trapped_2_1"] = "held, but not by stiffening" if held_ns else bool(ok)
    # last-record jump measures and wavenumbers
    row["jump_max_end"], row["jump_x_end"] = float(c["jump_max"][-1]), float(c["jump_x"][-1])
    row["jump_cells_end"], row["tv_end"] = float(c["jump_cells"][-1]), float(c["tv"][-1])
    row["k1_end"], row["krms_end"] = float(c["k1"][-1]), float(c["krms"][-1])
    row["pi_over_h"] = pi / h
    row["probe_spectrum_split"] = probe_split(nm)
    row["tail_kappa_measured"] = sm["tail_kappa_at_late_rate"]
    row["tail_kappa_O1"] = sm["tail_kappa_O1_at_late_rate"]
    row["twin_frac_end"] = res["twin"]["frac_end"] if res.get("twin") else None
    row["start_info"] = {k: v for k, v in res["start_info"].items() if k in
                         ("x_s", "a_top", "a_s", "plateau_cells", "omega", "rate", "L_half", "peak_stretch",
                          "converged", "A")}
    return row, c, tw


def loglog_slope(h, y):
    h, y = np.asarray(h, float), np.asarray(y, float)
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 2:
        return None
    return float(np.polyfit(np.log(h[ok]), np.log(y[ok]), 1)[0])


def rel_spread(v):
    v = np.asarray(v, float)
    if not np.all(np.isfinite(v)) or np.any(v == 0):
        return None
    return float((v.max() - v.min()) / abs(v.mean()))


def classify_c6(rows):
    """Rule 2.3 C6 for one start over h = 1/4, 1/8, 1/16 (rows = list of b_row dicts in that order)."""
    hs = [r["h"] for r in rows]
    tr = [r["trapped_2_1"] for r in rows]
    det = {"trapped_2_1": tr, "frac_ref": [r["frac_ref"] for r in rows],
           "rate_late": [r["rate_late"] for r in rows], "L_half_late": [r["L_half_late"] for r in rows],
           "rms_late": [r["rms_late"] for r in rows], "jump_cells_end": [r["jump_cells_end"] for r in rows]}
    sL = loglog_slope(hs, det["L_half_late"])
    sR = loglog_slope(hs, det["rms_late"])
    det["slope_L_half_vs_h"], det["slope_rms_vs_h"] = sL, sR
    if any(r["guard"] for r in rows):
        det["guard_at"] = [r["h"] for r in rows if r["guard"]]
    if all(v == "held, but not by stiffening" for v in tr):
        return "held, but not by stiffening", det
    if not any(v is True for v in tr) and not any(v == "held, but not by stiffening" for v in tr):
        return "not trapped at any h (disperses like its twin or faster)", det
    if any(v == "held, but not by stiffening" for v in tr):
        return "held, but not by stiffening at some h (twin above 0.5): see numbers", det
    if not all(v is True for v in tr):
        # holds at some h and not at others
        finer_fail = any((v is not True) and any(w is True for w in tr[:i]) for i, v in enumerate(tr))
        if finer_fail:
            return "artifact of the numerical spacing (holds at a coarser h, not at a finer one)", det
        return "undetermined (holds only at the finer h; not matched by any rule)", det
    # trapped at all three
    slopes = [s for s in (sL, sR) if s is not None]
    if slopes and max(slopes) >= 0.5:
        return "artifact of the numerical spacing (width follows h)", det
    if all(v is not None and v <= 1 for v in det["jump_cells_end"]):
        return "a jump in stretch, not resolved by any spacing", det
    agree = [rel_spread(det[k]) for k in ("frac_ref", "rate_late", "L_half_late")]
    det["rel_spread_frac_rate_Lhalf"] = agree
    chg = []
    for k in ("frac_ref", "rate_late", "L_half_late"):
        v = det[k]
        chg.append((abs(v[2] - v[1]), abs(v[1] - v[0])))
    det["change_1/8->1/16_vs_1/4->1/8"] = chg
    if all(a is not None and a <= 0.05 for a in agree) and all(a < b for a, b in chg):
        return "grid-independent trap", det
    return "undetermined", det


def b_all():
    rows, cv = {}, {}
    for tag, cls, obj, par in bstarts():
        lst = []
        for h, ht in HS:
            r = b_row(tag, h, ht)
            if r is None:
                continue
            row, c, tw = r
            lst.append(row)
            cv["%s_%s" % (tag, ht)] = (c, tw)
        if lst:
            rows[tag] = lst
    return rows, cv


def c6_all(rows):
    out = {}
    for tag, cls, obj, par in bstarts():
        if tag in rows and len(rows[tag]) == 3:
            v, det = classify_c6(rows[tag])
            out[tag] = {"class": cls, "obj": obj, "params": par, "verdict": v, "detail": det,
                        "sources": [r["source"] for r in rows[tag]]}
    return out


def selection(rows, c6):
    """Rule 2.3 selection: from start widths 0.25, 0.5, 1, 2 the final widths agree within 20% ('the
    medium selects a size'); final/start within 0.67-1.5 for all four ('no selection'). Judged on the
    trapped starts only, fewer than three trapped starts is undetermined. Width = L_half; 'start' is
    the first record's value. h = 1/8 is used as the reference spacing; the other two are reported."""
    out = {}
    for obj, pre in (("bump", "B1_bump_pk0.8_s"), ("sbend", "B2_sbend_pk0.8_s")):
        for ih, (h, ht) in enumerate(HS):
            ent = []
            for sg in (0.25, 0.5, 1, 2):
                tag = pre + "%g" % sg
                if tag in rows and len(rows[tag]) > ih:
                    r = rows[tag][ih]
                    ent.append({"sigma": sg, "trapped": r["trapped_2_1"] is True,
                                "L_half_start": r["L_half_start"], "L_half_final": r["L_half_late"],
                                "ratio": r["L_half_late"] / r["L_half_start"],
                                "rms_start": r["rms_start"], "rms_final": r["rms_late"]})
            tr = [e for e in ent if e["trapped"]]
            d = {"entries": ent, "n_trapped": len(tr)}
            if len(tr) < 3:
                d["verdict"] = "undetermined (fewer than three trapped starts)"
            else:
                fin = [e["L_half_final"] for e in tr]
                spread = max(fin) / min(fin)
                d["final_width_max_over_min"] = spread
                rat = [e["ratio"] for e in tr]
                if spread <= 1.2:
                    d["verdict"] = "the medium selects a size"
                elif all(0.67 <= q <= 1.5 for q in rat):
                    d["verdict"] = "no selection: the state keeps the width it was given"
                else:
                    d["verdict"] = "neither (spread %.2f, ratios %s)" % (spread, ["%.2f" % q for q in rat])
            out["%s_%s" % (obj, ht)] = d
    return out


# ------------------------------------------------------------------ B5, B6
def o3_pred(sig):
    a = 5.76 * sig
    return erfc(a) + 2 * a / sqrt(pi) * exp(-a * a)


def b5():
    out = []
    for sg in (0.1, 0.25, 0.5):
        for h, ht in HS[1:]:
            nm = "B5_lin_bump_s%g_%s" % (sg, ht)
            if not exists(nm):
                continue
            res = J(nm)
            c = curves(res)
            f2000 = val_at_t(c, "ff", 2000.0)
            if f2000 is None:
                f2000 = float(c["ff"][-1])
            out.append({"sigma": sg, "h": h, "name": nm, "source": REL(path(nm)), "frac_t2000": f2000,
                        "t_last_record": float(c["t"][-1]), "predicted": o3_pred(sg),
                        "abs_diff": abs(f2000 - o3_pred(sg)), "E_balance": res["summary"]["max_abs_balance"]})
    return out


def zc_rate(t, q, t0, t1):
    m = (t >= t0) & (t <= t1)
    if m.sum() < 8:
        return None
    return G.rate_zero_crossings(t[m], q[m])


def b6():
    out = {"fixed_amplitude": [], "fixed_peak_stretch": {"bump": [], "sbend": []}, "O4": []}
    def one(nm, sg):
        if not exists(nm):
            return None
        res = J(nm)
        ser = S(nm)
        t = ser["t"]
        T = t[-1]
        core, cd = ser["core"], ser["core_stretch"]
        sm = res["summary"]
        return {"name": nm, "source": REL(path(nm)), "sigma": sg, "rate_late_core": sm["rate_late"],
                "rate_early_core": sm["rate_early"],
                "rate_late_peak_stretch": zc_rate(t, cd, 0.8 * T, T), "rate_early_peak_stretch":
                    zc_rate(t, cd, 0.0, 0.1 * T),
                "rate_early_peak_stretch_first_10_cycles": zc_rate(t, cd, 0.0, 10 * 2 * np.pi / 1.0),
                "guard": sm["guard"] is not None, "frac_end": sm["frac_end"],
                "twin_frac_end": res["twin"]["frac_end"] if res.get("twin") else None,
                "peak_d_initial": float(np.max(np.abs(cd[:5]))), "t_end": sm["t_end"]}
    for sg in (0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.4, 2.0):
        r = one("B6_A0.2_s%g_h8" % sg, sg)
        if r:
            out["fixed_amplitude"].append(r)
    for obj in ("bump", "sbend"):
        for sg in (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0):
            r = one("B6x_%s_pk0.8_s%g_h8" % (obj, sg), sg)
            if r:
                out["fixed_peak_stretch"][obj].append(r)
    for pk in (0.3, 0.5, 0.8, 0.95):
        r = one("B6_O4_pk%g_s0.25_h8" % pk, 0.25)
        if r:
            lo = lib.lone_oscillator_rate(pk)
            # early rate over the first 10 cycles of the lone oscillator
            tmax = 10 * 2 * np.pi / lo
            ser = S(r["name"])
            re = zc_rate(ser["t"], ser["core_stretch"], 0.0, tmax)
            r.update({"pk": pk, "lone_oscillator_rate": lo, "early_rate_first_10_lone_cycles": re,
                      "rel_diff": (re / lo - 1) if re else None})
            out["O4"].append(r)
    return out


def spearman(xs, ys):
    ok = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and np.isfinite(y)]
    if len(ok) < 3:
        return None
    r = spearmanr([a for a, b in ok], [b for a, b in ok])
    return {"rho": float(r.statistic), "n": len(ok)}


def tighter_faster_smooth(b6res):
    out = {}
    fa = b6res["fixed_amplitude"]
    out["fixed_amplitude_core_rate"] = spearman([r["sigma"] for r in fa], [r["rate_late_core"] for r in fa])
    out["fixed_amplitude_peak_stretch_rate"] = spearman([r["sigma"] for r in fa],
                                                        [r["rate_late_peak_stretch"] for r in fa])
    for obj in ("bump", "sbend"):
        fp = b6res["fixed_peak_stretch"][obj]
        out["fixed_peak_stretch_%s_core_rate" % obj] = spearman([r["sigma"] for r in fp],
                                                                [r["rate_late_core"] for r in fp])
        out["fixed_peak_stretch_%s_peak_stretch_rate" % obj] = spearman(
            [r["sigma"] for r in fp], [r["rate_late_peak_stretch"] for r in fp])
    return out


# ------------------------------------------------------------------ push, scheme
def b_push(tags=("gsbend", "jsbend"), hts=("h8", "h16")):
    out = []
    lams = [0.1, 0.3, 1, 3, 10, 30]
    for tag in tags:
        for ht in hts:
            rows = []
            for lam in lams:
                nm = "B7_%s_l%g_%s" % (tag, lam, ht)
                if not exists(nm):
                    continue
                res = J(nm)
                ps = D.push_summary(res, 100.0)
                c = curves(res)
                sm = res["summary"]
                # distance travelled by the centroid and by the largest jump
                jx = c["jump_x"]
                rows.append({"lambda": lam, "name": nm, "source": REL(path(nm)), "speed": ps["speed"],
                             "distance": ps["distance"], "loss_per_distance": ps["energy_lost_per_unit_distance"],
                             "broke": ps["broke"], "frac_comoving_end": ps["frac_comoving_end"],
                             "frac_fixed_end": sm["frac_end"], "guard": sm["guard"] is not None,
                             "guard_label": (sm["guard"] or {}).get("rerun_half_dt", {}).get("label"),
                             "twin_frac_end": res["twin"]["frac_end"] if res.get("twin") else None,
                             "jump_x_start": float(jx[0]), "jump_x_end": float(jx[-1]),
                             "jump_x_shift": float(jx[-1] - jx[0]), "centroid_shift": float(c["cen"][-1] - c["cen"][0]),
                             "absorbed_over_E0": float(c["absorbed"][-1]), "t_end": sm["t_end"],
                             "reached_sponge_possible": bool(ps["distance"] > 100)})
            out.append({"tag": tag, "h": ht, "rows": rows})
    return out


def b8(rows):
    out = []
    for tag, fdtag in (("jsbend1.5", "B3_jump_sbend_w1.5"), ("gbump0.5", "B1_bump_pk0.8_s0.5")):
        nm = "B8_spectral_%s_h8" % tag
        if not exists(nm):
            continue
        row, c, tw = lifetime_row(nm, (300,))
        fdrow = rows.get(fdtag, [None] * 3)[1] if fdtag in rows else None
        res = J(nm)
        fd = {}
        if fdrow:
            cfd = curves(J("%s_h8" % fdtag))
            t300 = fdrow["t_300cyc"]
            fd = {"source": fdrow["source"], "frac_fixed_300cyc": fdrow["frac_fixed_300cyc"],
                  "rate_late_at_300cyc": fdrow["rate_300cyc"], "L_half_300cyc": fdrow["lh_300cyc"],
                  "rms_300cyc": fdrow["rms_300cyc"], "jump_max_300cyc": fdrow["jump_max_300cyc"],
                  "jump_cells_300cyc": fdrow["jump_cells_300cyc"], "t_300cyc": t300,
                  "twin_frac_300cyc": fdrow["twin_frac_300cyc"]}
        out.append({"start": tag, "spectral": {"source": row["source"], "frac_fixed_300cyc": row["frac_fixed_300cyc"],
                                               "rate_300cyc": row["rate_300cyc"], "L_half_300cyc": row["lh_300cyc"],
                                               "rms_300cyc": row["rms_300cyc"], "jump_max_300cyc": row["jump_max_300cyc"],
                                               "jump_cells_300cyc": row["jump_cells_300cyc"],
                                               "t_300cyc": row["t_300cyc"], "twin_frac_300cyc": row["twin_frac_300cyc"],
                                               "frac_end": row["frac_end"], "guard": row["guard"]},
                    "fd": fd})
    return out


# ------------------------------------------------------------------ O5, O6, snapshots
def snap(name, t):
    p = os.path.join(lib.ROOT, "checkpoints", "string", "%s_snap_t%g.npz" % (name, t))
    return np.load(p) if os.path.exists(p) else None


def pointwise_agreement(tag, t, window=20.0):
    """Relative L2 difference of u at the common nodes (coarse-grid points) between h = 1/4, 1/8, 1/16
    snapshots at time t, within |x| <= window."""
    z = {ht: snap("%s_%s" % (tag, ht), t) for _, ht in HS}
    if any(v is None for v in z.values()):
        return None
    def at(ht):
        x, u = z[ht]["x"], z[ht]["u"]
        m = np.abs(x) <= window
        return dict(zip(np.round(x[m] * 16).astype(int), u[m]))
    d = {ht: at(ht) for ht in z}
    keys = sorted(set(d["h4"]) & set(d["h8"]) & set(d["h16"]))
    a = np.array([d["h4"][k] for k in keys])
    b = np.array([d["h8"][k] for k in keys])
    c = np.array([d["h16"][k] for k in keys])
    nrm = np.linalg.norm(c)
    return {"t": t, "n_common_nodes": len(keys), "rel_diff_h4_vs_h16": float(np.linalg.norm(a - c) / nrm),
            "rel_diff_h8_vs_h16": float(np.linalg.norm(b - c) / nrm),
            "rel_diff_h4_vs_h8": float(np.linalg.norm(a - b) / np.linalg.norm(b))}


def kgrowth(c, h, t0=5.0, t1=100.0):
    m = (c["t"] >= t0) & (c["t"] <= t1) & np.isfinite(c["k1"])
    if m.sum() < 4:
        return None
    p = np.polyfit(c["t"][m], c["k1"][m], 1)
    pred = np.polyval(p, c["t"][m])
    ss = np.sum((c["k1"][m] - pred) ** 2)
    st = np.sum((c["k1"][m] - c["k1"][m].mean()) ** 2)
    r2 = 1 - ss / st if st > 0 else None
    return {"slope": float(p[0]), "r2": float(r2) if r2 is not None else None,
            "k1_max": float(np.nanmax(c["k1"])), "k1_end": float(c["k1"][-1]),
            "k1_max_over_pi_over_h": float(np.nanmax(c["k1"]) / (pi / h)),
            "krms_max_over_pi_over_h": float(np.nanmax(c["krms"]) / (pi / h)),
            "t_of_k1_max": float(c["t"][int(np.nanargmax(c["k1"]))])}


def o5(rows, cv, sel):
    out = {"per_start": {}}
    tags = [t for t, cls, obj, par in bstarts() if t.startswith(("B1_bump_pk0.8", "B2_sbend_pk0.8"))]
    for tag in tags:
        d = {}
        for h, ht in HS:
            k = "%s_%s" % (tag, ht)
            if k in cv:
                d["kgrowth_" + ht] = kgrowth(cv[k][0], h)
        d["pointwise_t100"] = pointwise_agreement(tag, 100.0)
        d["pointwise_t1000"] = pointwise_agreement(tag, 1000.0)
        if tag in rows and len(rows[tag]) == 3:
            d["frac_ref_by_h"] = [r["frac_ref"] for r in rows[tag]]
            d["frac_ref_spread"] = rel_spread(d["frac_ref_by_h"])
        out["per_start"][tag] = d
    out["selection"] = sel
    return out


def o6(rows, cv):
    out = []
    for tag, cls, obj, par in bstarts():
        if not (tag.startswith("B3") or tag.startswith("B4")):
            continue
        for h, ht in HS:
            k = "%s_%s" % (tag, ht)
            if k not in cv:
                continue
            c, tw = cv[k]
            jx = c["jump_x"]
            d = {"tag": tag, "h": h, "source": REL(path(k)), "jump_x_start": float(jx[0]),
                 "max_shift_nonlinear": float(np.nanmax(np.abs(jx - jx[0]))),
                 "jump_cells_start": float(c["jump_cells"][0]), "jump_max_start": float(c["jump_max"][0]),
                 "jump_max_end": float(c["jump_max"][-1])}
            d["max_shift_nonlinear_in_cells"] = d["max_shift_nonlinear"] / h
            if tw is not None:
                jt = tw["jump_x"]
                d["max_shift_twin"] = float(np.nanmax(np.abs(jt - jt[0])))
                d["max_shift_twin_in_cells"] = d["max_shift_twin"] / h
                d["twin_jump_max_start"], d["twin_jump_max_end"] = float(tw["jump_max"][0]), float(tw["jump_max"][-1])
            out.append(d)
    return out


def b_surroundings(name, rmax=20.0):
    res = J(name)
    pr = res["profiles"]
    x = np.array(pr["x_rel"])
    m = np.abs(x) <= rmax
    mu, md, ru = np.array(pr["mean_u"])[m], np.array(pr["mean_d"])[m], np.array(pr["rms_u"])[m]
    dist = np.arange(0, int(rmax) + 1)
    xx = x[m]
    def fold(a):
        return [float(np.mean(a[np.abs(np.abs(xx) - r) < 0.5])) for r in dist]
    def foldabs(a):
        return [float(np.max(np.abs(a[np.abs(np.abs(xx) - r) < 0.5]))) for r in dist]
    return {"name": name, "source": REL(path(name)), "distance": [int(v) for v in dist],
            "abs_mean_u_by_distance": foldabs(mu), "abs_mean_d_by_distance": foldabs(md),
            "rms_u_by_distance": fold(ru), "max_abs_mean_u": float(np.abs(mu).max()),
            "max_abs_mean_d": float(np.abs(md).max()), "max_rms_u": float(ru.max())}
