"""Rule-by-rule verdicts (Blocks 1 to 6 of docs/SESSION_05_PREDICTIONS.md, applied as written).
Every verdict row: id, claim, rule, measured, threshold, verdict, note, sources.
Verdict words: hit, pass with note (within about 3 times the target), miss, undetermined, split, no
prediction (C6), not evaluated."""
import numpy as np

HIT, NOTE, MISS, UND, SPLIT = "hit", "pass with note", "miss", "undetermined", "split"


def row(id_, claim, rule, measured, threshold, verdict, note="", sources=()):
    return {"id": id_, "claim": claim, "rule": rule, "measured": measured, "threshold": threshold,
            "verdict": verdict, "note": note, "sources": list(dict.fromkeys(sources))}


def grade(value, target, kind="le", factor=3.0):
    """kind 'le': value <= target is a hit, <= factor*target a pass with note, else miss.
    kind 'ge': value >= target hit, >= target/factor pass with note, else miss."""
    if value is None or not np.isfinite(value):
        return UND
    if kind == "le":
        return HIT if value <= target else (NOTE if value <= factor * target else MISS)
    return HIT if value >= target else (NOTE if value >= target / factor else MISS)


def fmt(x, n=3):
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e5):
        return "%.*e" % (n - 1, x)
    return "%.*g" % (n + 1, x)


# ---------------------------------------------------------------------- C1 .. C6
def v_C1(A1, A2):
    meas, src, per = [], [], {}
    ok_obj = {}
    for obj in ("sbend", "bump"):
        f = A1[obj]
        trapped = [r for r in A2 if r["obj"] == obj and r["trapped_2_1"] is True]
        held = [r for r in A2 if r["obj"] == obj and r["trapped_2_1"] == "held, but not by stiffening"]
        n = len([r for r in A2 if r["obj"] == obj])
        exists_ok = f["window"]["period_return_error_max"] < 1e-8 and f["window"]["rate"][0] > 2.0
        ok_obj[obj] = exists_ok and len(trapped) >= 1
        meas.append("%s: family of %d points, rate %s to %s, worst period-return error %s; trapped (2.1) at %d of %d lifetime rates (%s)" % (
            obj, f["n_points"], fmt(f["window"]["rate"][0]), fmt(f["window"]["rate"][1]),
            fmt(f["window"]["period_return_error_max"]), len(trapped), n,
            ", ".join(fmt(r["rate_nominal"]) for r in trapped) or "none"))
        src += [f["source"]] + [r["source"] for r in A2 if r["obj"] == obj]
    v = HIT if all(ok_obj.values()) else (SPLIT if any(ok_obj.values()) else MISS)
    return row("C1", "both objects exist as trapped vibrations with rate above the ceiling 2 (C-grid, G1)",
               "2.3 C1: periodic solution to solver precision with rate > 2, and 'trapped' by 2.1 (frac >= 0.90 at 1,000 cycles, twin <= 0.50, rate above 2 by more than the spectral resolution)",
               "; ".join(meas), "period-return error < 1e-8; rate > 2; trapped at >= 1 lifetime rate per object", v,
               "" if v == HIT else "see per-rate table", src)


def v_C2():
    return row("C2", "C-grid, G2: S-bend stable and unchanged past 32,000 cycles; bump unstable and moves",
               "2.3 C2", "not run in WP-C (the production runs use G1 only, as the task states)", "-", "not evaluated",
               "The WP-B1 build regression (not a production run) is in docs/WPB1_BUILD_NOTES.md section 10: "
               "S-bend multiplier 1 + 8e-15 and frac 0.99999988 at 31,995 cycles (hit); bump multiplier 2.062, centroid 0 -> -0.5 and stays "
               "(unstable: hit; moves by >= 1 site: miss). Those numbers are the build's, not re-measured here.",
               ["docs/WPB1_BUILD_NOTES.md"])


def c3_cases(A1, A2):
    """Per rate: S-bend stable? bump unstable and moves?"""
    out = []
    for r in A2:
        d = {"obj": r["obj"], "rate": r["rate_nominal"], "source": r["source"],
             "floquet_excl4_minus1": r.get("floquet_excl4_minus1")}
        cen = r["centroid"]
        d["frac_fixed_end"] = r["frac_end"]
        d["frac_comoving_end"] = r["frac_comoving_end"]
        d["cycles_at_end"] = r["cycles_at_end"]
        if r["obj"] == "sbend":
            amp = abs(r["peak_u_ratio_end_over_start"] - 1)
            shift = abs(cen["final_shift"])
            stable = (d["floquet_excl4_minus1"] is not None and d["floquet_excl4_minus1"] <= 1e-6 and
                      amp <= 0.01 and shift <= 0.01 and r["frac_end"] >= 0.99)
            d.update({"peak_amp_change": amp, "centre_shift": shift, "stable": bool(stable)})
        else:
            fq = d["floquet_excl4_minus1"]
            sh = cen["max_shift"]
            unstable = fq is not None and fq > 1e-3
            moves = sh >= 1.0 and cen["fc_min"] >= 0.90
            shifts_half = 0.5 <= sh < 1.0 and cen["fc_min"] >= 0.90
            dies = r["frac_comoving_end"] < 0.50
            if unstable and moves:
                cls = "unstable and moves"
            elif unstable and shifts_half:
                cls = "unstable, shifts to the S-bend position, does not travel (split)"
            elif dies:
                cls = "dies"
            elif unstable and cen["fc_min"] < 0.90:
                cls = "other: unstable, sheds energy (co-moving frac falls to %.3f), max shift %.2f" % (cen["fc_min"], sh)
            elif unstable:
                cls = "other: unstable, centroid shift under 0.5 (%.2f), co-moving frac stays >= 0.90" % sh
            else:
                cls = "other: multiplier %.1e above 1 is below 1e-3 (not 'unstable' by the rule); max shift %.3f" % (fq if fq is not None else float("nan"), sh)
            d.update({"max_shift": sh, "final_shift": cen["final_shift"], "fc_min": cen["fc_min"],
                      "unstable": bool(unstable), "class": cls,
                      "cycles_first_shift_0.5": cen["cycles_first_shift_0.5"],
                      "cycles_first_shift_1": cen["cycles_first_shift_1"]})
        out.append(d)
    return out


def v_C3(A1, A2):
    cs = c3_cases(A1, A2)
    sb = [c for c in cs if c["obj"] == "sbend"]
    bp = [c for c in cs if c["obj"] == "bump"]
    if not sb or not bp:
        return row("C3", "same ordering as C2 with G1", "2.3 C3", "incomplete", "-", UND), cs
    sb_ok = sum(c["stable"] for c in sb)
    bp_mv = sum(c["class"] == "unstable and moves" for c in bp)
    bp_sh = sum(c["class"].startswith("unstable, shifts") for c in bp)
    meas = ("S-bend stable at %d of %d rates (rates %s); bump: unstable and moves at %d, shifts half a site and stays at %d, other at %d of %d" % (
        sb_ok, len(sb), ", ".join(fmt(c["rate"]) for c in sb), bp_mv, bp_sh, len(bp) - bp_mv - bp_sh, len(bp)))
    if sb_ok == len(sb) and bp_mv == len(bp):
        v = HIT
    elif sb_ok == len(sb) and bp_mv + bp_sh == len(bp):
        v = SPLIT
    elif sb_ok == 0 or bp_mv + bp_sh == 0:
        v = MISS
    else:
        v = SPLIT
    return row("C3", "same ordering with G1: S-bend stable, bump unstable and moves",
               "2.3 C2/C3: stable = largest Floquet multiplier within 1e-6 of the unit circle AND peak amplitude and centre within 1% (centre read as 0.01 unit) AND fixed-window frac >= 0.99 at 32,000 cycles; unstable-and-moves = multiplier > 1 + 1e-3 AND centroid leaves its site by >= 1 unit with co-moving frac >= 0.90; shift 0.5 to 1 that does not grow is a split (Block 5)",
               meas, "see rule", v, "", [c["source"] for c in cs]), cs


def v_C4(a5):
    if a5 is None:
        return row("C4", "no emission at the rate or overtones; fixed-window energy constant to solver precision",
                   "2.3 C4", "A5 not complete", "-", UND)
    cl, sp = a5["closed"], a5["sponge"]
    err = cl["closed_ring_conservation_error"]
    dr_s, dr_c = sp["fixed_window_drift_max"], cl["fixed_window_drift_max"]
    ratio_s, ratio_c = dr_s / err, dr_c / err
    lines_bad, ratios_all, ratios_even = [], [], []
    E0 = None
    for tag in ("closed", "sponge"):
        for L, d in a5[tag]["far_probe_spectrum"].items():
            for nm, ln in d["lines"].items():
                eq = (ln["omega"] * ln["amp"]) ** 2        # energy-density scale of the line (omega A)^2
                ratios_all.append((tag, L, nm, ln["amp_over_floor"], eq))
                if nm in ("2nd harmonic", "4th harmonic"):
                    ratios_even.append(ln["amp_over_floor"])
    E0 = a5["E0"]
    floor_abs = 1e-5 * E0                                   # Block 6: below 1e-5 of E0 is unresolved
    for tag, L, nm, rr, eq in ratios_all:
        if nm in ("rate", "3rd harmonic", "5th harmonic") and eq > floor_abs:
            lines_bad.append("%s L=%s %s: (omega A)^2 = %.3g > 1e-5 E0" % (tag, L, nm, eq))
    odd = [rr for tag, L, nm, rr, eq in ratios_all if nm in ("rate", "3rd harmonic", "5th harmonic")]
    maxeq = max(eq for _, _, nm, _, eq in ratios_all if nm in ("rate", "3rd harmonic", "5th harmonic"))
    drift_v = grade(max(ratio_s, ratio_c), 10.0, "le", 3.0)
    spec_v = HIT if not lines_bad else MISS
    v = drift_v if spec_v == HIT else MISS
    srcs = [cl["source"], sp["source"]]
    meas = ("fixed-window drift over %.0f cycles: closed ring %s, sponge %s; closed-ring conservation error %s; ratios %s (closed), %s (sponge). Far-probe lines with (omega A)^2 above 1e-5 E0: %s" % (
        sp["cycles_at_end"], fmt(dr_c), fmt(dr_s), fmt(err), fmt(ratio_c), fmt(ratio_s),
        "; ".join(lines_bad) or "none"))
    meas += ("; largest (omega A)^2 over the rate and odd overtones %.3g against the Block 6 floor 1e-5 E0 = %.3g; line amplitude over the local spectral floor: rate and odd overtones %.2f to %.2f, symmetry-forbidden even harmonics %.2f to %.2f (so the lines are not distinguishable from the floor); exp(-kappa L) = %.2g at 60 sites" % (
        maxeq, floor_abs, min(odd), max(odd), min(ratios_even), max(ratios_even), a5["sponge"]["evanescent_level_exp_minus_kappa_L"].get("60.0", float("nan"))))
    return row("C4", "a trapped vibration emits nothing at its rate or overtones; fixed-window energy constant",
               "2.3 C4: drift <= 10 x closed-string conservation error, and no line at the rate or overtones above the leakage floor at a far probe",
               meas, "ratio <= 10 (<= 30 pass with note); (omega A)^2 of every line <= 1e-5 E0 (Block 6 floor)", v,
               "drift ratio verdict %s; spectrum verdict %s" % (drift_v, spec_v), srcs)


def v_C5(a6):
    from an_grid import c5_verdict
    cases, parts, srcs = [], [], []
    allok, anymiss = True, False
    for case in a6:
        cv = c5_verdict(case)
        if cv is None:
            continue
        cases.append((case, cv))
        srcs += [r["source"] for r in case["rows"]]
        lit = cv["lowest_crosses_100_frac_ge_0.9"]
        mono = cv["loss_nondecreasing_with_speed"]
        parts.append("%s w=%g: lowest lambda %g %s (frac at 100: %s); loss non-decreasing with speed: %s; alt reading (lowest mover lambda %s, frac at 100 %s); break speed %s" % (
            case["obj"], case["rate"], cv["lowest_lambda"], "crosses 100" if cv["lowest_crossed"] else "does not cross 100",
            fmt(cv["lowest_frac_at_100"]), mono, fmt(cv["alt_reading_lowest_mover_lambda"]),
            fmt(cv["alt_reading_lowest_mover_frac_at_100"]), fmt(cv["break_speed"])))
        if not (lit and mono is not False):
            allok = False
        if mono is False or not lit:
            anymiss = True
    if not cases:
        return row("C5", "motion with small loss at low speed, loss grows with speed", "2.3 C5", "A6 not complete", "-", UND), []
    v = HIT if allok else MISS
    return row("C5", "a trapped vibration can be moved with small loss at low speed; loss grows with speed",
               "2.3 C5 (as written): at the lowest push strength the object crosses 100 units with co-moving frac >= 0.90, and loss per unit distance does not decrease from one speed to the next; break speed = lowest measured speed with co-moving frac < 0.5 within 100 units",
               " | ".join(parts), "lowest push crosses 100 with frac >= 0.90; loss non-decreasing", v,
               "literal reading applied; the 'alt reading' column gives the lowest push that does move the object. Added after review: S-bend 2.3 and bump 2.3 are not pinned: they move 5 to 20 sites and stop (max excursions 19.8 and 9.4); the speed column is a fit after the first 10% of a long run (diagnostics D1)", srcs), cases


def v_C6(c6):
    if not c6:
        return row("C6", "C-smooth: no prediction", "2.3 C6", "B not complete", "-", UND), {}
    from collections import Counter
    cnt = Counter(v["verdict"].split(" (")[0] for v in c6.values())
    meas = "; ".join("%s: %d" % (k, n) for k, n in cnt.items())
    return row("C6", "C-smooth: no prediction on whether a grid-independent trap exists",
               "2.3 C6 classification per start at h = 1/4, 1/8, 1/16", "%d starts classified: %s" % (len(c6), meas),
               "no prediction on record", "no prediction", "outcomes per start in table 5", []), cnt
