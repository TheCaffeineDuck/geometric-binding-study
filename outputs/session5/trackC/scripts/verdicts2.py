"""Verdicts for the O rules and Aaron's expectations (continues verdicts.py)."""
import numpy as np

from verdicts import HIT, NOTE, MISS, UND, SPLIT, row, grade, fmt

ORDER = [MISS, UND, NOTE, HIT]


def worst(vs):
    return min(vs, key=lambda x: ORDER.index(x)) if vs else UND


def v_O1(A1, A2, brows):
    """Tails against the linear law, 5% (grid: 2 cosh(kappa/2) = rate; C-smooth: kappa = w/sqrt(w^2 - 1),
    component at the rate only)."""
    out = []
    errs, srcs = [], []
    for obj in ("sbend", "bump"):
        f = A1[obj]
        errs.append(("family %s (%d points)" % (obj, f["n_points"]), f["tail_kappa_rel_err_max"]))
        srcs.append(f["source"])
    for r in A2:
        k, k1 = r["tail_kappa_measured"], r["tail_kappa_O1"]
        if k is not None and k1:
            errs.append(("%s w=%g" % (r["obj"], r["rate_nominal"]), abs(k / k1 - 1)))
            srcs.append(r["source"])
    w = max(errs, key=lambda e: e[1])
    out.append(row("O1 (C-grid)", "tail of a trapped vibration falls by exp(-kappa) per site, 2 cosh(kappa/2) = rate",
                   "within 5% (family tail fits and the A2 runs' tail fit at the late rate)",
                   "worst %s: %s; best %s; %d fits" % (w[0], fmt(w[1]), fmt(min(e[1] for e in errs)), len(errs)),
                   "<= 5% (<= 15% pass with note)", grade(w[1], 0.05), "", srcs))
    errs, srcs = [], []
    for tag, rr in brows.items():
        if not (tag.startswith("B3") or tag.startswith("B4")):
            continue
        for r in rr:
            k, k1 = r["tail_kappa_measured"], r["tail_kappa_O1"]
            if k is not None and k1:
                errs.append(("%s h=%g" % (tag, r["h"]), abs(k / k1 - 1), k, k1))
                srcs.append(r["source"])
    if errs:
        w = max(errs, key=lambda e: e[1])
        b = min(errs, key=lambda e: e[1])
        out.append(row("O1 (C-smooth)", "tail at the rate falls as exp(-kappa x), kappa = w/sqrt(w^2 - 1), no alternation",
                       "within 5% on the component at the rate (jump-profile and exact starts; tail fit at the late rate)",
                       "%d fits; worst %s: measured %s against O1 %s (rel. error %s); best %s: %s" % (
                           len(errs), w[0], fmt(w[2]), fmt(w[3]), fmt(w[1]), b[0], fmt(b[1])),
                       "<= 5% (<= 15% pass with note)", grade(w[1], 0.05), "", srcs))
    else:
        out.append(row("O1 (C-smooth)", "tail law", "5%", "no tail fits available", "-", UND))
    return out


def v_O2(a7):
    if not a7:
        return row("O2", "time-averaged displacement and stretch of a lone object are zero", "-", "A7 incomplete", "-", UND)
    parts, vs, srcs = [], [], []
    for r in a7:
        parts.append("%s (%d cycles): max |mean u| %s, max |mean stretch| %s (end-sample corrected; raw u %s), max rms u %s" % (
            r["name"], r["cycles"], fmt(r["max_abs_mean_u_corrected"]), fmt(r["max_abs_mean_d_corrected"]),
            fmt(r["max_abs_mean_u_raw"]), fmt(r["max_rms_u"])))
        srcs.append(r["source"])
        if r["name"] in ("A7_sbend_w2.5", "A7_sbend_w3.5", "A7_bump_w2.5_c10"):
            vs.append(grade(max(r["max_abs_mean_u_corrected"], r["max_abs_mean_d_corrected"]), 1e-8))
    return row("O2", "time-averaged displacement and stretch of a lone bump or S-bend are zero at every distance",
               "zero to solver precision on family members (exact starts), out to 60 sites; solver precision read as 1e-8 absolute (the seeded noise), pass with note up to 3e-8",
               " | ".join(parts), "max |mean| <= 1e-8 (displacement amplitude near 1)", worst(vs),
               "the bump's 200-cycle run includes its departure from the site (it is unstable); the 10-cycle run is the family member", srcs)


def v_O3(b5):
    if not b5:
        return row("O3", "linear C-smooth pluck retained fraction", "-", "B5 incomplete", "-", UND)
    vs = [grade(r["abs_diff"], 0.05) for r in b5]
    meas = "; ".join("sigma %g, h=1/%d: %s against %s (|diff| %s)" % (r["sigma"], round(1 / r["h"]), fmt(r["frac_t2000"]),
                                                                    fmt(r["predicted"]), fmt(r["abs_diff"])) for r in b5)
    return row("O3", "linear C-smooth rest pluck: window-10 energy fraction at t = 2000 is 0.88, 0.25, 0.001 for sigma 0.1, 0.25, 0.5",
               "within 0.05 absolute (frac at t = 2000, h = 1/8 and 1/16)", meas, "|diff| <= 0.05 (<= 0.15 pass with note)",
               worst(vs), "", [r["source"] for r in b5])


def v_O4(b6):
    rows = b6["O4"]
    if not rows:
        return row("O4", "early rate at the point of peak stretch is the lone-oscillator rate", "-", "B6 O4 incomplete", "-", UND)
    vs = [grade(abs(r["rel_diff"]) if r["rel_diff"] is not None else None, 0.10) for r in rows]
    meas = "; ".join("pk %g: early rate %s against lone-oscillator %s (%+.1f%%)" % (
        r["pk"], fmt(r["early_rate_first_10_lone_cycles"]), fmt(r["lone_oscillator_rate"]),
        100 * r["rel_diff"] if r["rel_diff"] is not None else float("nan")) for r in rows)
    return row("O4", "C-smooth G1, sigma 0.25: rate at the point of peak stretch within 10% of the lone-oscillator rate",
               "within 10% (early rate = zero crossings of the stretch at the initial-peak bond over the first 10 lone-oscillator cycles; h = 1/8)",
               meas, "<= 10% (<= 30% pass with note)", worst(vs), "", [r["source"] for r in rows])


def v_O5(o5res):
    ps, sel = o5res["per_start"], o5res["selection"]
    r2s, ratios = [], []
    for tag, d in ps.items():
        for ht in ("h4", "h8", "h16"):
            kg = d.get("kgrowth_" + ht)
            if kg:
                r2s.append(kg["r2"] if kg["r2"] is not None else np.nan)
                ratios.append(kg["k1_max_over_pi_over_h"])
    p1 = np.nanmedian(r2s) if r2s else None
    p2 = np.nanmedian(ratios) if ratios else None
    d100 = [d["pointwise_t100"]["rel_diff_h8_vs_h16"] for d in ps.values() if d.get("pointwise_t100")]
    d1000 = [d["pointwise_t1000"]["rel_diff_h8_vs_h16"] for d in ps.values() if d.get("pointwise_t1000")]
    sp = [d["frac_ref_spread"] for d in ps.values() if d.get("frac_ref_spread") is not None]
    sv = {k: v["verdict"] for k, v in sel.items()}
    nosel = [k for k, s in sv.items() if s.startswith("no selection")]
    got = [
        ("P1 mean stretch wavenumber grows roughly linearly (median R^2 of a linear fit over t 5 to 100 >= 0.9)", p1, p1 is not None and p1 >= 0.9),
        ("P2 it reaches the cutoff (median of max k1 / (pi/h) >= 0.8)", p2, p2 is not None and p2 >= 0.8),
        ("P3 pointwise u disagreement between h = 1/8 and 1/16 is larger at t = 1000 than at t = 100 (medians, t=100 then t=1000)",
         (float(np.median(d100)) if d100 else None, float(np.median(d1000)) if d1000 else None),
         bool(d100 and d1000 and np.median(d1000) > np.median(d100))),
        ("P4 frac agrees across h (median spread of frac over h <= 5%)", float(np.median(sp)) if sp else None, bool(sp and np.median(sp) <= 0.05)),
        ("P5 rule 2.3 returns 'no selection' (%d of %d cases: %s)" % (len(nosel), len(sv), "; ".join("%s: %s" % (k, s.split(" (")[0]) for k, s in sv.items())),
         len(nosel), len(sv) > 0 and len(nosel) == len(sv)),
    ]
    nhit = sum(1 for g in got if g[2])
    v = HIT if nhit == len(got) else (MISS if nhit <= 1 else SPLIT)
    meas = "; ".join("%s: %s -> %s" % (k, (", ".join(fmt(x) for x in val) if isinstance(val, tuple) else fmt(val)), "holds" if ok else "fails") for k, val, ok in got)
    return row("O5", "C-smooth G1: no selected size; stretch winds up to the cutoff; pointwise profiles stop agreeing across h while frac(t) agrees",
               "Block 3 gives O5 no numerical test; evaluated as five parts with thresholds fixed here before reading the data: hit = all five hold, split = 2 to 4, miss = at most 1",
               meas, "five parts", v, "%d of %d parts hold" % (nhit, len(got)), [])


def v_O6(o6):
    if not o6:
        return row("O6", "a corner in u stays where it was made", "-", "B3/B4 incomplete", "-", UND)
    w = max(o6, key=lambda d: d["max_shift_nonlinear_in_cells"])
    twl = [d for d in o6 if "max_shift_twin_in_cells" in d]
    wt = max(twl, key=lambda d: d["max_shift_twin_in_cells"]) if twl else None
    wmax = max(w["max_shift_nonlinear_in_cells"], wt["max_shift_twin_in_cells"] if wt else 0)
    meas = "largest shift of the largest jump, in cells: nonlinear %s (%s, h=%g), linear twin %s (%s, h=%g)" % (
        fmt(w["max_shift_nonlinear_in_cells"]), w["tag"], w["h"],
        fmt(wt["max_shift_twin_in_cells"]) if wt else "n/a", wt["tag"] if wt else "", wt["h"] if wt else 0)
    return row("O6", "C-smooth: a jump in stretch stays where it was made, in the linear and the nonlinear medium",
               "distance from the centre of the largest jump (record peak-stretch sample; the S-bend has two jumps at +-x_s, so the sign is dropped) within one cell h of its start value over the whole run; up to 3 cells is a pass with note",
               meas, "<= 1 cell (<= 3 cells pass with note)", grade(wmax, 1.0),
               "the largest jump at a record is not necessarily the one made at t = 0: a shift can also mean that another jump became the largest",
               [d["source"] for d in o6])


def v_O7(o7):
    if not o7:
        return row("O7", "sigma 0.5, peak stretch 0.8 holds frac >= 0.5 at t = 2000", "-", "B1/B2 incomplete", "-", UND)
    vs = [grade(x["frac_nl"], 0.5, "ge") for x in o7]
    meas = "; ".join("%s h=1/%d: nonlinear %s, twin %s" % (x["tag"], round(1 / x["h"]), fmt(x["frac_nl"]), fmt(x["frac_tw"])) for x in o7)
    return row("O7", "C-smooth G1, sigma = 0.5, peak stretch 0.8: frac at t = 2000 is at least 0.5 where the twin holds 0.001",
               "frac(t = 2000) >= 0.5 (>= 0.167 pass with note); the prediction does not name the object: bump and S-bend both given",
               meas, "frac >= 0.5", worst(vs), "low-confidence prediction", [x["source"] for x in o7])


def v_O10(brows, c6):
    if not c6:
        return row("O10", "no smooth standing state; a held state has a one-cell jump", "-", "B incomplete", "-", UND)
    held = [t for t, v in c6.items() if v["verdict"] in ("a jump in stretch, not resolved by any spacing", "grid-independent trap")]
    claim = ("C-smooth G1 holds no smooth standing state above the ceiling; a held state has a jump in stretch 0.04 to 0.3 units from "
             "its centre, one cell wide at every h, with rate converging at first order in h")
    if not held:
        return row("O10", claim, "tested on each start trapped (2.1) at all three h", "no start is trapped at all three h (%d classified)" % len(c6),
                   "-", UND, "nothing to test the jump position or the rate convergence on", [])
    parts, oks, srcs = [], [], []
    for t in held:
        rr = brows[t]
        cells = [r["jump_cells_end"] for r in rr]
        pos = [abs(r["jump_x_end"]) for r in rr]
        rates = [r["rate_late"] for r in rr]
        d1, d2 = rates[1] - rates[0], rates[2] - rates[1]
        ratio = (d1 / d2) if d2 else None
        ok = all(0.04 <= p <= 0.3 for p in pos) and all(c <= 1 for c in cells) and ratio is not None and 1.4 <= abs(ratio) <= 3.0
        parts.append("%s: jump distance %s, cells %s, rate %s, ratio of rate differences %s" % (
            t, "/".join(fmt(p) for p in pos), "/".join(fmt(c) for c in cells), "/".join(fmt(x) for x in rates), fmt(ratio)))
        oks.append(ok)
        srcs += [r["source"] for r in rr]
    v = HIT if all(oks) else (SPLIT if any(oks) else MISS)
    return row("O10", claim, "per start trapped at all three h: jump distance in [0.04, 0.3] at every h; jump_cells <= 1 at every h; ratio of successive rate differences in [1.4, 3] (first order gives 2)",
               " | ".join(parts), "all three conditions", v, "", srcs)


def v_O11(brows):
    parts, vs, srcs = [], [], []
    for tag in ("B3_jump_sbend_w1.2", "B3_jump_sbend_w1.5", "B3_jump_bump_w1.2", "B3_jump_bump_w1.5"):
        rr = brows.get(tag)
        if not rr:
            continue
        fr = [r["frac_ref"] for r in rr]
        parts.append("%s: %s at h 1/4, 1/8, 1/16%s" % (tag, ", ".join(fmt(f) for f in fr),
                     "" if all(r["frac_ref_cycles_reached"] for r in rr) else " [1,000 cycles not reached: frac at end]"))
        vs.append(grade(min(fr), 0.99, "ge", factor=0.99 / 0.97))
        srcs += [r["source"] for r in rr]
    gp = []
    for tag in ("B1_bump_pk0.8_s1", "B1_bump_pk0.8_s2", "B2_sbend_pk0.8_s1", "B2_sbend_pk0.8_s2"):
        rr = brows.get(tag)
        if rr:
            gp.append("%s: nonlinear %s, twin %s" % (tag, "/".join(fmt(r["frac_ref"]) for r in rr), "/".join(fmt(r["twin_frac_ref"]) for r in rr)))
            srcs += [r["source"] for r in rr]
    if not vs:
        return row("O11", "jump-profile starts hold", "-", "B3 incomplete", "-", UND)
    return row("O11", "a jump-profile start at rate 1.2 or 1.5 keeps frac >= 0.99 to 1,000 cycles at all three h, where a Gaussian start of sigma >= 1 disperses like its twin",
               "frac >= 0.99 at 1,000 cycles at h = 1/4, 1/8, 1/16 (>= 0.97 pass with note); second clause reported: Gaussian sigma 1 and 2, nonlinear against twin",
               " | ".join(parts) + " || Gaussian: " + " | ".join(gp), "frac >= 0.99 at all h", worst(vs), "", srcs)
