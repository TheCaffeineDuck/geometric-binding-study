"""Aaron's three testable expectations (2.5), applied as written."""
import numpy as np

from verdicts import row, fmt, UND

SUP, CON, NOTD, SPL = "supported", "contradicted", "not distinguishable", "split"


def family_L_at(A1, obj, rate):
    rows = A1[obj]["rows"]
    om = np.array([r["omega"] for r in rows])
    L = np.array([r["L_half"] for r in rows])
    return float(np.interp(rate, om, L))


def loss_rate(r):
    cyc = max(r["cycles_at_end"], 1.0)
    return max(1.0 - r["frac_comoving_end"], 1e-12) / cyc


def expectations(A1, A2, A3, A4, A6, c5cases, brows, B7, c6, B6, TF):
    out, det = [], {}
    # ---------------------------------------------------------------- E1
    e1 = {}
    parts = []
    srcs = []
    if A1 and A2:
        for obj in ("sbend", "bump"):
            tr = [r for r in A2 if r["obj"] == obj and r["trapped_2_1"] is True]
            if not tr:
                e1[obj] = {"trapped_rates": []}
                parts.append("%s: no trapped lifetime rate" % obj)
                continue
            rates = [r["rate_nominal"] for r in tr]
            lo, hi = min(rates), max(rates)
            # the trapped range is extended to the family's own ends where the family exists
            Llo, Lhi = family_L_at(A1, obj, lo), family_L_at(A1, obj, hi)
            narrow = (hi / lo < 2.0) and (Llo / Lhi < 2.0)
            # optimal: an interior point with lifetime >= 2x both edges, or loss rate <= half both edges
            tr_sorted = sorted(tr, key=lambda r: r["rate_nominal"])
            edges = (tr_sorted[0], tr_sorted[-1])
            allcap = all(isinstance(r["T_death_comoving"], str) for r in tr_sorted)
            optimal = None
            if len(tr_sorted) >= 3:
                if allcap:
                    lr = [loss_rate(r) for r in tr_sorted]
                    optimal = any(lr[i] <= 0.5 * lr[0] and lr[i] <= 0.5 * lr[-1] for i in range(1, len(lr) - 1))
                    basis = "loss rates (all runs alive at the cap)"
                else:
                    td = [r["T_death_comoving"] if not isinstance(r["T_death_comoving"], str) else r["cap_t"] * 1e3
                          for r in tr_sorted]
                    optimal = any(td[i] >= 2 * td[0] and td[i] >= 2 * td[-1] for i in range(1, len(td) - 1))
                    basis = "T_death (co-moving)"
            e1[obj] = {"trapped_rates": rates, "rate_ratio": hi / lo, "L_half_ratio": Llo / Lhi, "narrow": bool(narrow),
                       "optimal_interior": optimal, "optimal_basis": basis if len(tr_sorted) >= 3 else "fewer than 3 points",
                       "loss_rates": {str(r["rate_nominal"]): loss_rate(r) for r in tr_sorted}}
            parts.append("%s: trapped %s to %s (x%.2f in rate), L_half %s to %s (x%.2f); narrow=%s; interior point with loss <= half of both edges: %s" % (
                obj, fmt(lo), fmt(hi), hi / lo, fmt(Lhi), fmt(Llo), Llo / Lhi, narrow, optimal))
            srcs += [r["source"] for r in tr]
    # too long -> travelling wave (A4)
    toolong = None
    if A4 and A1 and e1.get("sbend", {}).get("trapped_rates"):
        edge = max(family_L_at(A1, "sbend", min(e1["sbend"]["trapped_rates"])),
                   family_L_at(A1, "bump", min(e1.get("bump", {}).get("trapped_rates") or e1["sbend"]["trapped_rates"])))
        longer = [r for r in A4 if r["L_half_start"] > edge and r["cycles_ref"] >= 900 and not r["guard"]]
        below = [r for r in longer if r["frac_ref"] < 0.5]
        toolong = {"window_upper_edge_L_half": edge, "n_longer": len(longer), "n_below_half": len(below),
                   "names": [(r["name"], r["L_half_start"], r["frac_ref"], r["cycles_ref"]) for r in longer]}
        parts.append("starts longer than the window's upper edge (L_half > %s): %d of them, of which %d fall below frac 0.50 within 1,000 cycles" % (
            fmt(edge), len(longer), len(below)))
        srcs += [r["source"] for r in longer]
    # too fast -> travelling wave (A6)
    brk = [c["break_speed"] for case, c in c5cases if c["break_speed"] is not None]
    toofast = {"break_speeds": brk, "min_break_speed": min(brk) if brk else None}
    parts.append("break speed (lowest speed with co-moving frac < 0.5 within 100 units): %s (long-wave speed 1)" % (
        ", ".join(fmt(b) for b in brk) if brk else "none measured: no push broke the object within 100 units"))
    det["E1"] = {"per_object": e1, "too_long": toolong, "too_fast": toofast}
    # overall words for each clause
    clause = []
    for obj, d in e1.items():
        if "narrow" in d:
            clause.append("narrow: %s" % (SUP if d["narrow"] else CON))
    narrow_all = [d["narrow"] for d in e1.values() if "narrow" in d]
    narrow_w = SUP if narrow_all and all(narrow_all) else (CON if narrow_all and not any(narrow_all) else SPL)
    opt = [d["optimal_interior"] for d in e1.values() if d.get("optimal_interior") is not None]
    opt_w = SUP if opt and all(opt) else (CON if opt and not any(opt) else (SPL if opt else UND))
    tl_w = (SUP if toolong and toolong["n_longer"] and toolong["n_below_half"] == toolong["n_longer"] else
            (CON if toolong and toolong["n_longer"] and toolong["n_below_half"] == 0 else (SPL if toolong and toolong["n_longer"] else UND)))
    tf_w = SUP if brk and min(brk) < 1.0 else CON
    overall = SUP if all(w == SUP for w in (narrow_w, opt_w, tl_w, tf_w)) else (CON if all(w == CON for w in (narrow_w, opt_w, tl_w, tf_w)) else SPL)
    out.append(row("E1 (2.5)", "a swirl has an optimal length and rate and the range is narrow; too long or too fast and it becomes a travelling wave",
                   "narrow: trapped range < factor 2 in length and in rate; optimal: an interior point has lifetime >= 2x both edges (or, alive at the cap, loss rate <= half); too long: starts longer than the window's upper edge fall below frac 0.50 within 1,000 cycles; too fast: a break speed below 1 exists",
                   " | ".join(parts) + " || clauses: narrow %s, optimal %s, too long %s, too fast %s" % (narrow_w, opt_w, tl_w, tf_w),
                   "as stated", overall, "clause-by-clause words in the measured column; length range is the C-grid family's L_half at the trapped end rates", srcs))
    # ---------------------------------------------------------------- E2
    sb = sorted([r for r in A2 if r["obj"] == "sbend"], key=lambda r: r["E0"])
    bp = sorted([r for r in A2 if r["obj"] == "bump"], key=lambda r: r["E0"])
    if sb and bp:
        Eb = np.array([r["E0"] for r in bp])
        lb = np.log([loss_rate(r) for r in bp])
        fb = np.array([r["frac_comoving_end"] for r in bp])
        pts = []
        for r in sb:
            if Eb.min() <= r["E0"] <= Eb.max():
                lbi = float(np.exp(np.interp(np.log(r["E0"]), np.log(Eb), lb)))
                fbi = float(np.interp(np.log(r["E0"]), np.log(Eb), fb))
                pts.append({"E": r["E0"], "sbend_rate": r["rate_nominal"], "sbend_frac_end": r["frac_comoving_end"],
                            "sbend_loss": loss_rate(r), "bump_frac_end_interp": fbi, "bump_loss_interp": lbi,
                            "sbend_T_death": r["T_death_comoving"],
                            "better": r["frac_comoving_end"] > fbi, "worse": r["frac_comoving_end"] < fbi})
        n = len(pts)
        nb = sum(p["better"] for p in pts)
        nw = sum(p["worse"] for p in pts)
        allcap = all(isinstance(p["sbend_T_death"], str) for p in pts) and all(isinstance(r["T_death_comoving"], str) for r in bp)
        within2 = all(max(p["sbend_loss"], p["bump_loss_interp"]) / min(p["sbend_loss"], p["bump_loss_interp"]) <= 2 for p in pts)
        if n < 5:
            w = UND
        elif allcap and within2:
            w = NOTD
        elif nb >= 0.8 * n:
            w = SUP
        elif nw >= 0.8 * n:
            w = CON
        else:
            w = SPL
        det["E2"] = pts
        out.append(row("E2 (2.5)", "the S-bend traps energy better than the single bump",
                       "S-bend's T_death (co-moving) or retained frac at the cap exceeds the bump's at 4 or more of 5 points matched in energy; 'not distinguishable' if both exceed the cap with loss rates within a factor 2",
                       "%d S-bend runs inside the bump's energy range; S-bend retains more at %d, less at %d; all alive at cap: %s; loss rates within 2x at all: %s; per point (E: S-bend frac / bump frac interpolated in log E): %s" % (
                           n, nb, nw, allcap, within2, "; ".join("%s: %s / %s" % (fmt(p["E"]), fmt(p["sbend_frac_end"]), fmt(p["bump_frac_end_interp"])) for p in pts)),
                       ">= 4 of 5", w, "energy matching is by interpolation over the six bump lifetime runs (no dedicated runs at the S-bend energies); a bump that hops sites is scored by its co-moving frac",
                       [r["source"] for r in sb + bp]))
    # ---------------------------------------------------------------- E3
    if bp:
        per = []
        for r in bp:
            cen = r["centroid"]
            leaves = cen["max_shift"] >= 1.0
            t_leave = cen["cycles_first_shift_1"]
            leaves_1000 = bool(t_leave is not None and t_leave <= 1000)
            dies_1000 = (r["frac_comoving_1000cyc"] is not None and r["frac_comoving_1000cyc"] < 0.5)
            stays = (cen["max_shift"] < 1.0) and (r["frac_end"] >= 0.99) and r["cycles_at_end"] >= 31000
            per.append({"rate": r["rate_nominal"], "max_shift": cen["max_shift"], "cycles_first_shift_1": t_leave,
                        "cycles_first_shift_0.5": cen["cycles_first_shift_0.5"], "frac_fixed_end": r["frac_end"],
                        "frac_comoving_end": r["frac_comoving_end"], "supported": bool(leaves_1000 or dies_1000),
                        "contradicted": bool(stays)})
        ns = sum(p["supported"] for p in per)
        nc = sum(p["contradicted"] for p in per)
        w = SUP if ns == len(per) else (CON if nc == len(per) else SPL)
        det["E3"] = per
        out.append(row("E3 (2.5)", "the single bump is probably not stable enough to build a structure",
                       "supported if the unperturbed bump leaves its site (read as centroid shift >= 1 unit, as in rule C2) or dies within 1,000 cycles; contradicted if it stays within one unit with frac >= 0.99 for 32,000 cycles",
                       "; ".join("rate %g: max centroid shift %s, first reaches 0.5 at cycle %s and 1 at cycle %s, frac fixed/co-moving at the end %s/%s" % (
                           p["rate"], fmt(p["max_shift"]), fmt(p["cycles_first_shift_0.5"]), fmt(p["cycles_first_shift_1"]),
                           fmt(p["frac_fixed_end"]), fmt(p["frac_comoving_end"])) for p in per),
                       "leaves site (>= 1 unit) within 1,000 cycles", w,
                       "%d of %d rates supported, %d contradicted; a shift of 0.5 that stays and holds frac >= 0.99 reads as 'contradicted' under the letter of the rule although the bump has lost its site-centred form" % (ns, len(per), nc),
                       [r["source"] for r in bp]))
    out.append(row("2.5 structures and reach", "swirls hold together through their geometry; field reaches further", "Block 2 2.5",
                   "not tested here", "-", "not tested", "Session 6", []))
    out.append(row("2.5 pinch", "the pinch is the object Aaron started from", "Block 2 2.5", "Track F (WP-D), not Track C", "-", "not tested", "", []))
    return out, det
