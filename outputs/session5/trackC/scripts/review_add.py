"""Additive tables and notes added after the review (every item marked 'added after review')."""
import json
import os

import numpy as np

import lib
import an_smooth as AS
from verdicts import fmt
from tables import md

f = fmt


def one_pct(brows):
    out = []
    for tag, base_tag, st in (("B4_orbit_sbend_1pct", "B4_orbit_sbend_s0.5_pk0.8", "exact C-smooth state S-bend"),
                              ("B4_jump_sbend_w1.5_1pct", "B3_jump_sbend_w1.5", "jump S-bend 1.5")):
        trio = {}
        for sc in (0.99, 1.01):
            nm = "%s_x%g_h8" % (tag, sc)
            if not lib.exists(nm):
                continue
            row, c, tw = lifetime_row_safe(nm)
            trio[sc] = row
        base = brows.get(base_tag, [None, None, None])[1]
        out.append({"start": st, "base": base, "x0.99": trio.get(0.99), "x1.01": trio.get(1.01)})
    return out


def lifetime_row_safe(nm):
    row, c, tw = lib.lifetime_row(nm, (300, 1000))
    res = lib.J(nm)
    row["resolution"] = res["summary"]["spectral_resolution_first_window"]
    reached = row["frac_fixed_1000cyc"] is not None
    row["frac_ref"] = row["frac_fixed_1000cyc"] if reached else row["frac_end"]
    row["twin_frac_ref"] = row["twin_frac_1000cyc"] if reached else (res["twin"]["frac_end"] if res.get("twin") else None)
    row["reached"] = reached
    ok = row["frac_ref"] >= 0.9 and row["twin_frac_ref"] is not None and row["twin_frac_ref"] <= 0.5 and row["rate_late"] - 1.0 > row["resolution"]
    row["trapped_2_1"] = bool(ok)
    return row, c, tw


def section(SUM, brows, verdicts, c6):
    L = []
    add = L.append
    add("\n## Added after review\n")
    # ---- 1: 1% runs
    add("### 1. C-smooth 1% runs (added after review): scaled 0.99 and 1.01 starts at h = 1/8, beside the exact start (Block 6)\n")
    rows, sumrows = [], []
    for e in one_pct(brows):
        b, lo, hi = e["base"], e["x0.99"], e["x1.01"]
        if not (b and lo and hi):
            continue
        t3 = [lo["trapped_2_1"], b["trapped_2_1"], hi["trapped_2_1"]]
        cls = "trapped, robust to 1%" if all(v is True for v in t3) else ("boundary state, not a trap (a scaled start is not trapped)" if b["trapped_2_1"] is True else "not trapped at the exact start")
        rows.append([e["start"], " / ".join(f(x["frac_ref"]) for x in (lo, b, hi)), " / ".join("%s" % f(x["cycles_at_end"]) for x in (lo, b, hi)),
                     " / ".join("yes" if x.get("reached", x.get("frac_ref_cycles_reached")) else "no" for x in (lo, b, hi)), " / ".join(f(x["rate_late"]) for x in (lo, b, hi)),
                     " / ".join(f(x["twin_frac_ref"]) for x in (lo, b, hi)), " / ".join(str(v) for v in t3), cls,
                     ", ".join(x["source"] for x in (lo, hi))])
        sumrows.append({"start": e["start"], "classification": cls, "trapped_triplet": t3,
                        "cycles_reached": [x["cycles_at_end"] for x in (lo, b, hi)], "frac_ref": [x["frac_ref"] for x in (lo, b, hi)],
                        "sources": [x["source"] for x in (lo, b, hi)]})
    add(md(["start", "frac at ref (0.99 / exact / 1.01)", "cycles reached", "1,000 cycles reached", "rate late", "twin frac", "trapped (2.1)", "Block 6 classification", "files"], rows))
    add("")
    SUM["added_after_review_1pct_smooth"] = sumrows
    # ---- 2: cycles reached
    add("### 2. Cycles reached by the jump and orbit rows (added after review)\n")
    add("The C-smooth runs of the jump and exact starts end at a fixed time, which is 920 to 975 cycles of their measured rate, not 1,000. Their frac, twin and 'trapped' values in tables 1 to 5 are therefore taken at the end of the run, at:\n")
    rows = []
    for tag, rr in brows.items():
        if tag.startswith(("B3", "B4")):
            for r in rr:
                rows.append([tag, "1/%d" % round(1 / r["h"]), f(r["cycles_at_end"]), "yes" if r["frac_ref_cycles_reached"] else "no (frac at end)", str(r["trapped_2_1"]) + ("" if r["frac_ref_cycles_reached"] else " (at %s cycles)" % f(r["cycles_at_end"]))])
    add(md(["start", "h", "cycles reached", "1,000 cycles reached", "trapped (2.1) and the cycle count it is judged at"], rows))
    add("")
    SUM["added_after_review_cycles_reached"] = [{"start": r[0], "h": r[1], "cycles": r[2]} for r in rows]
    # ---- 3: definition changes
    A5 = SUM.get("A5") or {}
    sp = (A5.get("sponge") or {})
    add("### 3. Definition changes made after the data were seen (added after review)\n")
    add("- **C4 spectrum criterion.** The first criterion coded was 'a line at the rate or an odd overtone above 3 times its local spectral floor'. Under it the 5th-harmonic line at the 60-site probe of the sponge run (3.24 times its floor) would have made the spectrum part of C4 a miss, and C4 a miss overall. I replaced it by the Block 6 floor (the line's (omega A)^2 against 1e-5 E0) because the even harmonics, which symmetry forbids, sit at 1.5 to 2.0 times their floors: the floor is quantisation noise (the far-probe series are stored as float32, relative precision 6e-8, and the probe amplitudes are 1e-19 to 1e-21) and the 3x test fired on it. Leak bound added: the sponge absorbed %s of E0 over the 10,000 cycles (closed ring: %s; E conservation error %s), so no more than that left the window in any form. Verdict under the first criterion: miss on the spectrum part; under the Block 6 floor: hit. Both are on record; the drift part is a hit under either (ratio %s)." % (
        f(sp.get("absorbed_over_E0")), f((A5.get("closed") or {}).get("absorbed_over_E0")), f((A5.get("closed") or {}).get("closed_ring_conservation_error")),
        f(sp.get("fixed_window_drift_max", 0) / (A5.get("closed") or {}).get("closed_ring_conservation_error", 1) if A5 else None)))
    o6 = SUM.get("O6") or []
    held = [d for d in o6 if d["tag"] in ("B3_jump_sbend_w1.5", "B3_jump_sbend_w2", "B4_orbit_sbend_s0.5_pk0.8")]
    add("- **O6 jump position.** Signed position of the largest jump gave a worst shift of 14 cells (the detector picks either of the two S-bend jumps at plus and minus 0.25); the distance from the centre gives 6 cells. The verdict is a miss under both definitions. All %d held states (jump S-bend 1.5 and 2.0 and the exact S-bend, at h = 1/4, 1/8, 1/16) shift %s cells (largest %s); the misses come from the starts that do not hold (jump S-bend 1.2 and the jump bumps)." % (
        len(held), "0" if held and max(d["max_shift_nonlinear_in_cells"] for d in held) == 0 else "at most %s" % f(max([d["max_shift_nonlinear_in_cells"] for d in held] or [0])),
        f(max([d["max_shift_nonlinear_in_cells"] for d in held] or [0]))))
    add("")
    SUM["added_after_review_definition_changes"] = {"C4_first_criterion": "5th harmonic 3.24x local floor (sponge, L = 60) would be a miss", "C4_sponge_absorbed_over_E0": sp.get("absorbed_over_E0"),
                                                      "O6_signed_worst_cells": 14, "O6_distance_worst_cells": 6}
    # ---- 4: C5 wording
    a6 = SUM.get("A6") or []
    add("### 4. Push wording (added after review)\n")
    add("Replaces 'pinned, no motion' for S-bend 2.3 and bump 2.3. The speed in tables A6 is a centroid fit after the first 10% of a long run (5,000 to 40,000 time units), which misses the early transit. The objects move and then stop. Maximum centroid excursions (sites):\n")
    rows = []
    for case in a6:
        rows.append(["%s %g" % (case["obj"], case["rate"]), "; ".join("lambda %g: %s" % (r["lambda"], f(r["max_abs_centroid"])) for r in case["rows"])])
    add(md(["case", "maximum centroid excursion"], rows))
    add("\nThe build's number is reproduced with the build's settings (diagnostics/DIAGNOSTICS.md, D1): S-bend 2.3 speed 0.0386 (lambda 0.5) and 0.0528 (lambda 1) over T = 300; 0.0026 and 0.0018 for T = 2000. The C5 row stays a miss: no push at 2.3 or 3.0 reaches 100 sites.\n")
    # ---- 5: per-row detail
    add("### 5. Detail hidden by worst-case aggregation (added after review)\n")
    errs = []
    for tag, rr in brows.items():
        if tag.startswith(("B3", "B4")):
            for r in rr:
                k, k1 = r["tail_kappa_measured"], r["tail_kappa_O1"]
                if k is not None and k1:
                    errs.append((abs(k / k1 - 1), tag, r["h"], k, k1))
    errs.sort()
    n17 = sum(1 for e in errs if e[0] <= 0.017)
    add("- **O1 C-smooth.** %d of %d fits are within 1.7%% of the linear tail law (worst of those %s). The others: %s." % (
        n17, len(errs), f(max([e[0] for e in errs if e[0] <= 0.017] or [0])),
        "; ".join("%s h=1/%d measured %s against %s (rel. error %s)" % (e[1], round(1 / e[2]), f(e[3]), f(e[4]), f(e[0])) for e in errs if e[0] > 0.017)))
    d5 = json.load(open(os.path.join(lib.TC, "diagnostics", "D5.json"))) if os.path.exists(os.path.join(lib.TC, "diagnostics", "D5.json")) else []
    Es = {r["h"]: r["E"] for r in d5 if "E" in r}
    ratio = None
    if all(h in Es for h in (0.25, 0.125, 0.0625)):
        ratio = (Es[0.25] - Es[0.125]) / (Es[0.125] - Es[0.0625])
    add("- **O10 clause by clause** (states trapped at all three h):")
    rows = []
    for tag in ("B3_jump_sbend_w1.5", "B3_jump_sbend_w2", "B4_orbit_sbend_s0.5_pk0.8"):
        rr = brows.get(tag)
        if not rr:
            continue
        pos = [abs(r["jump_x_end"]) for r in rr]
        cells = [r["jump_cells_end"] for r in rr]
        rates = [r["rate_late"] for r in rr]
        d1, d2 = rates[1] - rates[0], rates[2] - rates[1]
        rows.append([tag, "yes (%s)" % "/".join(f(p) for p in pos) if all(0.04 <= p <= 0.3 for p in pos) else "no", "yes (%s)" % "/".join(f(c) for c in cells) if all(c <= 1 for c in cells) else "no",
                     "%s / %s / %s; ratio %s" % (f(rates[0]), f(rates[1]), f(rates[2]), f(d1 / d2 if d2 else None)),
                     "not a measurement (the rate is the Newton input 1.4231)" if tag.startswith("B4") else "ratio outside [1.4, 3]" if not (d2 and 1.4 <= abs(d1 / d2) <= 3) else "within"])
    add("")
    add(md(["state", "jump 0.04 to 0.3 from centre", "one cell at every h", "late rate h = 1/4, 1/8, 1/16", "first-order convergence of the rate"], rows))
    add("\n  For the exact state the convergence test is run on the orbit energy instead: E = %s (h = 1/4), %s (1/8), %s (1/16); differences ratio %s (first order gives 2, second order 4)." % (
        f(Es.get(0.25)), f(Es.get(0.125)), f(Es.get(0.0625)), f(ratio)))
    sp_ = [d["frac_ref_spread"] for d in (SUM.get("O5") or {}).get("per_start", {}).values() if d.get("frac_ref_spread") is not None]
    absd = []
    for tag in (SUM.get("O5") or {}).get("per_start", {}):
        if tag in brows and len(brows[tag]) == 3:
            fr = [r["frac_ref"] for r in brows[tag]]
            absd.append(max(fr) - min(fr))
    add("\n- **O5 absolute frac spread across h** (max minus min of frac at the reference time, per Gaussian start, 8 starts): median %s, largest %s; the relative spread used in the verdict (median %s) is inflated by the starts with frac near zero." % (
        f(float(np.median(absd)) if absd else None), f(max(absd) if absd else None), f(float(np.median(sp_)) if sp_ else None)))
    SUM["added_after_review_per_row"] = {"O1_smooth_within_1.7pct": n17, "O1_smooth_total": len(errs), "O10_B4_energy_ratio": ratio, "O5_abs_frac_spread_median": float(np.median(absd)) if absd else None}
    add("")
    return "\n".join(L)
