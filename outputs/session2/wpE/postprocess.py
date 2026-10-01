"""WP-E post-processing: arrangement result JSON -> per-run record, outcome (Block 3), criteria table, plots.

Usage (from the repo root; works on partial results; analysis code only, never re-runs anything):
  <python> outputs/session2/wpE/postprocess.py [--results DIR] [--criteria FILE] [--emap FILE] [--out DIR] [--no-plots]
Defaults: results = outputs/session2/wpE/results, criteria = outputs/session2/wpE/criteria.json,
emap = outputs/session2/wpE/criteria_E_map.json, out = outputs/session2/wpE/analysis.
Writes <out>/<run>.json, <out>/summary.json, <out>/criteria_E.json and <out>/<run>.png.

Outcome (docs/SESSION_02_PREDICTIONS.md Block 3, implemented as written; parameters from criteria.json)
------------------------------------------------------------------------------------------------------
Samples: the lump-count windows with t_mid >= eval_from (50), to the end of the run.
  sample ok  <=>  the distinct-lump count EQUALS K (min_lump_count "K" = the run's own K) and every
                  lump's nearest-neighbour distance / its initial value is in [nn_distance_min,
                  nn_distance_max] (0.5, 1.5).
  bonded            ok in at least hold_fraction (0.95) of the samples AND in the last sample AND the run
                    completed (require_completed) AND distinct at the start.
  merged            count below K in the last count window.
  flew apart        count = K in the last count window and some ratio above nn_distance_max there.
  not distinct at the start   fewer than K distinct lumps in the first count window; never bonded; the
                    state at the end (merged / flew apart / bounded) is still reported in "end_state".
  otherwise         "bounded at T, not bonded", with the trend: slope of the mean nearest-neighbour
                    distance (count windows with count = K) over the last third of the run.
  A missing or null parameter gives "not evaluated" (no defaults). K = 1 gives "not evaluated (K = 1)".

Distances in the rule come from coordinates (distance_source "peaks"): the distinct maxima of each
count window (highest maximum per distinct group). Each lump's ratio needs a lump identity: the K peaks
are matched to the K tracked ball centroids at that time (minimum total distance; the initial positions
if the tracked series is not valid there). The initial value of each lump is its nearest-neighbour
distance in the configured start positions (exact for snapshot starts). With "tracked" the tracked
centroids are used directly.

Mechanical report per run (see report()): contact time (first count window with count < K, or first
time any pair is within 2 hw), the time the first nearest-neighbour ratio passes 1.5, hold fraction,
count series, min/max ratio over the run and at the end, group energy retained at the end, angular
momentum in the first period and its extremes, total momentum extremes, per-lump N spread and the
neighbour phase steps. The last two, and all tracked per-lump series, are only read while the count
equals K: from the start of the first count window with count < K they are set to null
("per_lump_valid_until").

Criteria table (criteria_E.json): rows E1 to E15 of Block 3 where they are numeric; run names from the
editable map criteria_E_map.json (exact name, else a unique result whose name starts with the prefix).
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from swirl.engine.lumps import pair_distances, summarize  # noqa: E402

PROFILE = os.path.join(ROOT, "outputs/step2/profiles/profile_beta1.3333_w0.9091.json")
HW = json.load(open(PROFILE))["half_width"]
CONTACT = 2 * HW                     # Block 3: "a pair comes within 2 hw" (6.688)
REQUIRED = ("min_lump_count", "count_key", "distance_source", "nn_distance_min", "nn_distance_max",
            "relative_to_initial", "eval_from", "hold_fraction", "require_completed")
PER_LUMP_NOTE = ("tracked and per-lump series (centroids, N, phases) are not meaningful once the distinct-lump "
                 "count is below K; they are null from the start of the first count window with count < K")


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def nn_dist(X, L):
    D = pair_distances(np.asarray(X, float), L)
    np.fill_diagonal(D, np.inf)
    return D.min(1), D[np.triu_indices(len(X), 1)]


def wrap_angle(x):
    return np.angle(np.exp(1j * np.asarray(x, float)))


def distinct_peaks(c, K):
    """Highest maximum of each distinct group of a count window, at most K, by value."""
    best = {}
    for lab, pk in zip(c["labels"], c["peaks"]):
        if lab not in best or pk["value"] > best[lab]["value"]:
            best[lab] = pk
    return np.array([q["pos"] for q in sorted(best.values(), key=lambda q: -q["value"])[:K]], float)


def match(peaks, ref, L):
    """Order the peaks like the reference positions (minimum total minimum-image distance)."""
    cost = np.linalg.norm(((peaks[None, :, :] - ref[:, None, :]) + L / 2) % L - L / 2, axis=-1)
    r, c = linear_sum_assignment(cost)
    out = np.empty_like(ref)
    out[r] = peaks[c]
    return out


# ---------------------------------------------------------------------------
# per-run record from a result JSON
# ---------------------------------------------------------------------------

def record(res):
    p = res["params"]
    cand = p.get("_candidate", {})
    L = p["L"]
    raw = res["lumps"]["raw"]
    K = len(raw["centers"])
    st = {k: raw[k] for k in ("centers", "series", "point")}
    s = raw["series"]
    ts = np.array(s["t"])
    T = float(res["t"])
    av = summarize(st, np.arange(0.0, T + 1e-9, 1.0), p["lumps"]["avg_window"])
    pos0 = np.asarray(cand.get("positions", raw["series"]["centers_used"][0] if s.get("centers_used") else raw["centers"]),
                      float)
    nn0 = nn_dist(pos0, L)[0] if K > 1 else None
    r = {"name": cand.get("name"), "K": K, "L": L, "completed": bool(res["completed"]), "t_end": T,
         "T_design": p["t_end"], "start": cand.get("start"), "linear": p["g4"] == 0 and p["g6"] == 0,
         "phases_deg": cand.get("phases_deg"), "initial_positions": pos0.tolist(),
         "initial_nn": None if nn0 is None else nn0.tolist(), "hw": HW,
         "omega_nominal": p["lumps"]["omega_nominal"]}
    lu0 = av["lumps"][0]
    om = [x for x in lu0["omega"] if x is not None]
    r["regime"] = {"omega_over_m_median": float(np.median(om)) / p["m"] if om else None,
                   "nonlinear_share_t0": res["series"]["nonlinear_share"][0]}
    # count windows
    cnt = res["lumps"]["counts"]
    win_len = p["lumps"]["count_periods"] * 2 * np.pi / p["lumps"]["omega_nominal"]
    r["counts"] = {"t_mid": [c["t_mid"] for c in cnt], "count": [c["count"] for c in cnt],
                   "count_distinct": [c["count_distinct"] for c in cnt], "window_length": win_len}
    bad = [c["t_mid"] for c in cnt if c["count_distinct"] < K]
    t_valid = (bad[0] - win_len / 2) if bad else None          # tracked / per-lump series valid before this
    r["per_lump_valid_until"] = t_valid
    r["per_lump_note"] = PER_LUMP_NOTE
    # tracked centroids on lump 0's window centres
    tc0 = np.array([np.nan if x is None else x for x in lu0["t_center"]])
    ok0 = ~np.isnan(tc0)
    tt = tc0[ok0]
    Xs = None
    if K > 1:
        cols = []
        for k in range(K):
            lk = av["lumps"][k]
            tk = np.array([x for x, X in zip(lk["t_center"], lk["X"]) if x is not None and X is not None])
            Xk = np.array([X for x, X in zip(lk["t_center"], lk["X"]) if x is not None and X is not None])
            if tk.size < 2:
                cols = None
                break
            cols.append(np.stack([np.interp(tt, tk, Xk[:, a]) for a in range(3)], 1))
        if cols is not None:
            Xs = np.stack(cols, 1)
    valid_t = (lambda t: t_valid is None or t < t_valid)
    trk = {"t": [], "nn_ratio_min": [], "nn_ratio_max": [], "nn_mean": [], "pair_min": []}
    if Xs is not None:
        for i, t in enumerate(tt):
            if not valid_t(t):
                continue
            nn, pd = nn_dist(Xs[i], L)
            trk["t"].append(float(t))
            trk["nn_ratio_min"].append(float(np.min(nn / nn0)))
            trk["nn_ratio_max"].append(float(np.max(nn / nn0)))
            trk["nn_mean"].append(float(nn.mean()))
            trk["pair_min"].append(float(pd.min()))
    r["tracked"] = trk
    # peaks: per-lump ratios in windows with count = K
    pk = {"t_mid": [], "ratios": [], "nn_mean": [], "pair_min": []}
    for c in cnt:
        pk["t_mid"].append(c["t_mid"])
        if K < 2 or c["count_distinct"] != K:
            pk["ratios"].append(None), pk["nn_mean"].append(None), pk["pair_min"].append(None)
            continue
        P = distinct_peaks(c, K)
        ref = pos0
        if Xs is not None and valid_t(c["t_mid"]) and tt.size:
            i = int(np.argmin(np.abs(tt - c["t_mid"])))
            ref = Xs[i]
        P = match(P, ref, L)
        nn, pd = nn_dist(P, L)
        pk["ratios"].append((nn / nn0).tolist())
        pk["nn_mean"].append(float(nn.mean()))
        pk["pair_min"].append(float(pd.min()))
    r["peaks"] = pk
    # per-lump N spread and neighbour phase steps (only while valid)
    Nk = np.array([[np.nan if x is None else x for x in av["lumps"][k]["N"]] for k in range(K)])
    spread = []
    for i, t in enumerate(tc0):
        if np.isnan(t) or not valid_t(t) or np.any(np.isnan(Nk[:, i])):
            continue
        spread.append((float(t), float((Nk[:, i].max() - Nk[:, i].min()) / Nk[:, i].mean())))
    r["N_spread"] = {"t": [a for a, _ in spread], "rel": [b for _, b in spread]}
    edges = []
    if K > 1:
        D0 = pair_distances(pos0, L)
        edges = [(i, j) for i in range(K) for j in range(i + 1, K) if D0[i, j] <= 1.01 * min(nn0)]
    rel = [np.zeros(len(tc0))] + [np.array([np.nan if x is None else x for x in q["dtheta"]])
                                  for q in av["relative_phase"]]
    th_set = np.deg2rad(np.asarray(cand.get("phases_deg", [0.0] * K), float))
    steps = {"t": [], "edges": edges, "set_deg": [float(np.rad2deg(wrap_angle(th_set[j] - th_set[i])))
                                                 for i, j in edges], "deg": []}
    for i, t in enumerate(tc0):
        if np.isnan(t) or not valid_t(t) or not edges:
            continue
        row = [float(np.rad2deg(wrap_angle(rel[b][i] - rel[a][i]))) for a, b in edges]
        if np.any(np.isnan(row)):
            continue
        steps["t"].append(float(t))
        steps["deg"].append(row)
    r["phase_steps"] = steps
    # group energy, momentum
    P_nom = 2 * np.pi / p["lumps"]["omega_nominal"]
    if s.get("E_group"):
        Eg = np.array(s["E_group"])
        La = np.array(s["L_axis_box"])
        first = ts <= ts[0] + P_nom
        rm = np.array([Eg[np.abs(ts - t) <= P_nom / 2].mean() for t in ts]) / Eg[0]
        r["group"] = {"t": ts.tolist(), "E_group_over_0_period_mean": rm.tolist(),
                      "E_retained_end": float(rm[-1]),
                      "E_retained_at": {str(tq): float(np.interp(tq, ts, rm)) for tq in (100.0, 250.0, 300.0)
                                        if tq <= ts[-1]},
                      "L_axis_first_period": float(La[first].mean()), "L_axis_min": float(La.min()),
                      "L_axis_max": float(La.max()), "L_axis_box": La.tolist(),
                      "L_axis_sphere": s["L_axis_sphere"]}
    Pt = np.array(s["P_total"])
    r["P_total"] = {"abs_max": float(np.max(np.linalg.norm(Pt, axis=1))), "min": Pt.min(0).tolist(),
                    "max": Pt.max(0).tolist()}
    r["max_abs_phi"] = {"t": res["series"]["t"], "v": res["series"]["max_abs_phi"]}
    return r


# ---------------------------------------------------------------------------
# mechanics and outcome on the record
# ---------------------------------------------------------------------------

def report(r):
    """Mechanical numbers for every run (no criteria needed)."""
    K = r["K"]
    c = r["counts"]
    out = {"count_series": {"t_mid": c["t_mid"], "count_distinct": c["count_distinct"]},
           "per_lump_valid_until": r["per_lump_valid_until"], "per_lump_note": r["per_lump_note"]}
    t_drop = next((t for t, n in zip(c["t_mid"], c["count_distinct"]) if n < K), None)
    tr, pk = r["tracked"], r["peaks"]
    t_pair_tr = next((t for t, d in zip(tr["t"], tr["pair_min"]) if d < CONTACT), None)
    t_pair_pk = next((t for t, d in zip(pk["t_mid"], pk["pair_min"]) if d is not None and d < CONTACT), None)
    cands = [x for x in (t_drop, t_pair_tr, t_pair_pk) if x is not None]
    out["contact"] = {"time": min(cands) if cands else None, "count_drop_t_mid": t_drop,
                      "pair_within_2hw_tracked": t_pair_tr, "pair_within_2hw_peaks": t_pair_pk,
                      "two_hw": CONTACT,
                      "resolution": "count windows every 25 (t_mid, window %.1f); tracked: whole-period window centres, steps of half a period (~3.5)" % c["window_length"]}
    rat = [(t, q) for t, q in zip(pk["t_mid"], pk["ratios"]) if q is not None]
    t15_pk = next((t for t, q in rat if max(q) > 1.5), None)
    t15_tr = next((t for t, q in zip(tr["t"], tr["nn_ratio_max"]) if q > 1.5), None)
    cands = [x for x in (t15_pk, t15_tr) if x is not None]
    out["nn_passes_1.5"] = {"time": min(cands) if cands else None, "peaks": t15_pk, "tracked": t15_tr}
    out["nn_ratio"] = {
        "peaks_min_over_run": min((min(q) for _, q in rat), default=None),
        "peaks_max_over_run": max((max(q) for _, q in rat), default=None),
        "peaks_at_end": (pk["ratios"][-1] if pk["ratios"] else None),
        "tracked_min_over_valid": min(tr["nn_ratio_min"], default=None),
        "tracked_max_over_valid": max(tr["nn_ratio_max"], default=None),
        "tracked_at_end_of_valid": ([tr["nn_ratio_min"][-1], tr["nn_ratio_max"][-1]] if tr["t"] else None)}
    if "group" in r:
        g = r["group"]
        out["group"] = {"E_retained_end": g["E_retained_end"], "E_retained_at": g["E_retained_at"],
                        "L_axis_first_period": g["L_axis_first_period"], "L_axis_min": g["L_axis_min"],
                        "L_axis_max": g["L_axis_max"]}
    out["P_total"] = r["P_total"]
    ns = r["N_spread"]
    out["N_spread_rel"] = {"max_over_valid": max(ns["rel"], default=None),
                           "max_until_t100": max((v for t, v in zip(ns["t"], ns["rel"]) if t <= 100), default=None),
                           "first": ns["rel"][0] if ns["rel"] else None, "n_windows": len(ns["rel"])}
    ps = r["phase_steps"]
    dev = [max(abs(float(wrap_angle(np.deg2rad(d - s)))) for d, s in zip(row, ps["set_deg"]))
           for row in ps["deg"]]
    out["phase_steps"] = {"edges": ps["edges"], "set_deg": ps["set_deg"],
                          "first_deg": ps["deg"][0] if ps["deg"] else None,
                          "max_dev_deg_until_t100": float(np.rad2deg(max((d for t, d in zip(ps["t"], dev) if t <= 100),
                                                                         default=np.nan))) if dev else None,
                          "max_dev_deg_over_valid": float(np.rad2deg(max(dev))) if dev else None,
                          "n_windows": len(dev)}
    return out


def _trend(r, t_from):
    pk = r["peaks"]
    pts = [(t, m) for t, m in zip(pk["t_mid"], pk["nn_mean"]) if m is not None and t >= t_from]
    tr = r["tracked"]
    ptr = [(t, m) for t, m in zip(tr["t"], tr["nn_mean"]) if t >= t_from]
    sl = lambda pts: float(np.polyfit([a for a, _ in pts], [b for _, b in pts], 1)[0]) if len(pts) >= 2 else None
    return {"from_t": t_from, "slope_nn_mean_peaks": sl(pts), "n_peaks": len(pts),
            "slope_nn_mean_tracked": sl(ptr), "n_tracked": len(ptr)}


def outcome(r, crit):
    """Block 3 outcome on the record r with the parameters crit (see module docstring)."""
    if crit is None:
        return {"outcome": "not evaluated", "reason": "no criteria file"}
    missing = [k for k in REQUIRED if crit.get(k) is None]
    if missing:
        return {"outcome": "not evaluated", "reason": "criteria not set: %s" % missing}
    K = r["K"]
    if K < 2:
        return {"outcome": "not evaluated", "reason": "K = 1: no neighbours, the rule does not apply"}
    if crit["min_lump_count"] != "K" and int(crit["min_lump_count"]) != K:
        return {"outcome": "not evaluated", "reason": "min_lump_count %s is not this run's K = %d (Block 3: "
                "the count must equal K)" % (crit["min_lump_count"], K)}
    if crit["count_key"] != "count_distinct" or not crit["relative_to_initial"]:
        return {"outcome": "not evaluated", "reason": "criteria differ from Block 3 (count_key %s, "
                "relative_to_initial %s)" % (crit["count_key"], crit["relative_to_initial"])}
    lo, hi = float(crit["nn_distance_min"]), float(crit["nn_distance_max"])
    c = r["counts"]
    if not c["t_mid"]:
        return {"outcome": "not evaluated", "reason": "no count windows"}
    if crit["distance_source"] == "peaks":
        ratios = r["peaks"]["ratios"]
    elif crit["distance_source"] == "tracked":
        tr = r["tracked"]
        ratios = []
        for t, n in zip(c["t_mid"], c["count_distinct"]):
            if n != K or not tr["t"] or t > tr["t"][-1]:
                ratios.append(None)
                continue
            i = int(np.argmin(np.abs(np.array(tr["t"]) - t)))
            ratios.append([tr["nn_ratio_min"][i], tr["nn_ratio_max"][i]])
    else:
        return {"outcome": "not evaluated", "reason": "unknown distance_source %s" % crit["distance_source"]}
    rows = []
    for t, n, q in zip(c["t_mid"], c["count_distinct"], ratios):
        if t < float(crit["eval_from"]):
            continue
        cnt_ok = n == K
        dist_ok = q is not None and min(q) >= lo and max(q) <= hi
        rows.append({"t": t, "count": n, "ratio_min": None if q is None else min(q),
                     "ratio_max": None if q is None else max(q), "count_ok": cnt_ok, "dist_ok": dist_ok,
                     "ok": cnt_ok and dist_ok})
    if not rows:
        return {"outcome": "not evaluated", "reason": "no count window with t_mid >= eval_from"}
    held = float(np.mean([x["ok"] for x in rows]))
    last = rows[-1]
    n_end = c["count_distinct"][-1]
    q_end = ratios[-1]
    distinct_start = c["count_distinct"][0] >= K
    if n_end < K:
        end = "merged"
    elif n_end == K and q_end is not None and max(q_end) > hi:
        end = "flew apart"
    else:
        end = None
    bonded = (distinct_start and held >= float(crit["hold_fraction"]) and last["ok"]
              and (r["completed"] or not crit["require_completed"]))
    out = {"held_fraction": held, "n_samples": len(rows), "n_ok": int(sum(x["ok"] for x in rows)),
           "last_sample": last, "completed": r["completed"], "distinct_at_start": distinct_start,
           "first_window_count": c["count_distinct"][0], "samples": rows, "criteria": crit}
    if not distinct_start:
        out["outcome"] = "not distinct at the start"
        out["end_state"] = end or "bounded at T, not bonded"
    elif bonded:
        out["outcome"] = "bonded"
    elif end is not None:
        out["outcome"] = end
    else:
        out["outcome"] = "bounded at T, not bonded"
        if not r["completed"]:
            out["note"] = "run not completed"
    if out["outcome"] == "bounded at T, not bonded" or out.get("end_state") == "bounded at T, not bonded":
        out["trend"] = _trend(r, r["t_end"] * 2 / 3)
    return out


# ---------------------------------------------------------------------------
# criteria table E1 to E15
# ---------------------------------------------------------------------------

def item(name, measured, tol, ok, reason=None):
    if ok is None:
        return {"item": name, "measured": measured, "tolerance": tol, "verdict": "not evaluable", "reason": reason}
    return {"item": name, "measured": measured, "tolerance": tol, "verdict": "hit" if ok else "miss"}


def _in(x, lo, hi):
    return None if x is None else bool(lo <= x <= hi)


def resolve(emap, names):
    out = {}
    for row, spec in emap["rows"].items():
        if spec["run"] in names:
            out[row] = spec["run"]
            continue
        hits = [n for n in names if n.startswith(spec["prefix"])]
        out[row] = hits[0] if len(hits) == 1 else None
    return out


def _first_drop(rec, t_max):
    c = rec["record"]["counts"]
    return next((t for t, n in zip(c["t_mid"], c["count_distinct"]) if n < rec["record"]["K"] and t <= t_max), None)


def criteria_E(runs, emap):
    """runs: name -> {"record", "report", "outcome"}. Returns the table rows."""
    m = resolve(emap, set(runs))
    g = lambda row: runs.get(m.get(row)) if m.get(row) else None
    rows = {}

    def missing(row):
        return {"run": m.get(row), "items": [item("run", None, None, None, "no result for this row yet")]}

    def oc(rec):
        return rec["outcome"]["outcome"]

    def contact(rec):
        return rec["report"]["contact"]["time"]

    def merged_by(rec, t):
        return _first_drop(rec, t)

    def contact_row(row, lo, hi, extra=()):
        rec = g(row)
        if rec is None:
            return missing(row)
        it = [item("outcome merged", oc(rec), "merged", oc(rec) == "merged" if oc(rec) != "not evaluated" else None,
                   rec["outcome"].get("reason")),
              item("contact time", contact(rec), [lo, hi], _in(contact(rec), lo, hi),
                   "no contact in the record" if contact(rec) is None else None)]
        return {"run": m[row], "items": it + list(extra)}

    # E1, E2
    r1, r2 = g("E1"), g("E2")
    if r1 is None:
        rows["E1"] = missing("E1")
    else:
        rp = r1["report"]
        fac = (contact(r1) / contact(r2)) if (r2 is not None and contact(r1) and contact(r2)) else None
        rows["E1"] = contact_row("E1", 190, 420, [
            item("contact factor E1 / E2", fac, [1.3, 2.5], _in(fac, 1.3, 2.5),
                 "E2 result or a contact time missing"),
            item("per-lump N spread until t = 100", rp["N_spread_rel"]["max_until_t100"], 0.02,
                 None if rp["N_spread_rel"]["max_until_t100"] is None else rp["N_spread_rel"]["max_until_t100"] <= 0.02,
                 "no valid window before t = 100"),
            item("phase steps within 10 deg of 72 until t = 100 (max deviation, deg)",
                 rp["phase_steps"]["max_dev_deg_until_t100"], 10.0,
                 None if rp["phase_steps"]["max_dev_deg_until_t100"] is None
                 else rp["phase_steps"]["max_dev_deg_until_t100"] <= 10.0, "no valid window before t = 100")])
    rows["E2"] = contact_row("E2", 105, 200)
    for row, K_t in (("E3", 300.0), ("E7", 300.0)):
        rec = g(row)
        if rec is None:
            rows[row] = missing(row)
            continue
        t = merged_by(rec, K_t)
        rows[row] = {"run": m[row], "items": [
            item("count below K by t = %g (first t_mid)" % K_t, t, "<= %g" % K_t, t is not None),
            item("outcome merged", oc(rec), "merged", oc(rec) == "merged" if oc(rec) != "not evaluated" else None,
                 rec["outcome"].get("reason")),
            item("not bonded", oc(rec), "not bonded", oc(rec) != "bonded" if oc(rec) != "not evaluated" else None,
                 rec["outcome"].get("reason"))]}
    for row in ("E4", "E8"):
        rec = g(row)
        if rec is None:
            rows[row] = missing(row)
            continue
        c = rec["record"]["counts"]
        K = rec["record"]["K"]
        n0 = c["count_distinct"][0] if c["count_distinct"] else None
        t50 = next((t for t, n in zip(c["t_mid"], c["count_distinct"]) if n < K and t <= 50.0), None)
        rows[row] = {"run": m[row], "items": [
            item("fewer than K distinct at the start (first window count) or by t = 50 (first t_mid)",
                 {"first_window_count": n0, "first_drop_t_mid_by_50": t50, "K": K}, "< K at start or by 50",
                 (n0 is not None and n0 < K) or t50 is not None),
            item("not bonded", oc(rec), "not bonded", oc(rec) != "bonded" if oc(rec) != "not evaluated" else None,
                 rec["outcome"].get("reason"))]}
    for row, lo, hi in (("E5", 160, 320), ("E6", 200, 450)):
        rec = g(row)
        if rec is None:
            rows[row] = missing(row)
            continue
        t15 = rec["report"]["nn_passes_1.5"]["time"]
        its = [item("outcome flew apart", oc(rec), "flew apart",
                    oc(rec) == "flew apart" if oc(rec) != "not evaluated" else None, rec["outcome"].get("reason")),
               item("nearest-neighbour distance passes 1.5 x start", t15, [lo, hi], _in(t15, lo, hi),
                    "never passes 1.5 in the record")]
        if row == "E5":
            drop = _first_drop(rec, np.inf)
            its.append(item("no merger (first count drop)", drop, "none", drop is None))
            its.append(item("per-lump N stay equal (max relative spread while count = K)",
                            rec["report"]["N_spread_rel"]["max_over_valid"], None, None,
                            "Block 3 gives no numeric tolerance for 'stay equal'; measured value reported"))
        rows[row] = {"run": m[row], "items": its}
    rec = g("E9")
    if rec is None:
        rows["E9"] = missing("E9")
    else:
        t_drop = _first_drop(rec, 250.0)
        t15 = rec["report"]["nn_passes_1.5"]["time"]
        rows["E9"] = {"run": m["E9"], "items": [
            item("merged or flew apart by T = 250 (first count drop, first pass of 1.5)",
                 {"count_drop": t_drop, "passes_1.5": t15}, "either <= 250",
                 t_drop is not None or (t15 is not None and t15 <= 250.0)),
            item("not bonded", oc(rec), "not bonded", oc(rec) != "bonded" if oc(rec) != "not evaluated" else None,
                 rec["outcome"].get("reason"))]}
    rows["E10"] = contact_row("E10", 170, 370)
    rows["E11"] = contact_row("E11", 320, 650)
    rows["E12"] = contact_row("E12", 80, 160, [
        item("the in-phase neighbours first", None, None, None,
             "pair identity at contact is not stored; not evaluable from the stored series")])
    rec = g("E13")
    if rec is None:
        rows["E13"] = missing("E13")
    else:
        rp = rec["report"]
        end = rp["nn_ratio"]["peaks_at_end"]
        T_ok = rec["record"]["t_end"] >= 500.0 - 1e-6
        drop = _first_drop(rec, np.inf)
        rows["E13"] = {"run": m["E13"], "items": [
            item("no merger (first count drop)", drop, "none", drop is None),
            item("every nearest-neighbour ratio above 1.2 at T = 500 (min over lumps, last count window)",
                 None if end is None else min(end), "> 1.2",
                 None if (end is None or not T_ok) else min(end) > 1.2,
                 "run shorter than 500" if not T_ok else "count not equal to K in the last window")]}
    rec = g("E14")
    if rec is None:
        rows["E14"] = missing("E14")
    else:
        rp = rec["report"]
        c = rec["record"]["counts"]
        K = rec["record"]["K"]
        lo_, hi_ = rp["nn_ratio"]["peaks_min_over_run"], rp["nn_ratio"]["peaks_max_over_run"]
        dev = None if lo_ is None else max(abs(lo_ - 1), abs(hi_ - 1))
        allK = all(n == K for n in c["count_distinct"])
        ns = rp["N_spread_rel"]["max_over_valid"]
        rows["E14"] = {"run": m["E14"], "items": [
            item("count = K in every window", c["count_distinct"], K, allK),
            item("distances within 1% (max |ratio - 1| over the run, peaks)", dev, 0.01,
                 None if dev is None else dev <= 0.01, "no window with count = K"),
            item("per-lump N equal (max relative spread)", ns, 1e-3, None if ns is None else ns <= 1e-3,
                 "no valid window")]}
    rt, rr, rb, r3 = g("E15_twin"), g("E15_res"), g("E15_box"), g("E3")
    its = []
    if rt is None:
        its.append(item("twin", None, None, None, "no twin result yet"))
    else:
        ge = rt["report"].get("group", {}).get("E_retained_at", {}).get("100.0")
        its.append(item("twin: group energy below 0.1 by T = 100 (period mean at t = 100)", ge, "< 0.1",
                        None if ge is None else ge < 0.1, "no group series"))
        mp = rt["record"]["max_abs_phi"]
        its.append(item("twin: no lumps left", {"count_distinct_end": rt["record"]["counts"]["count_distinct"][-1]
                                                if rt["record"]["counts"]["count_distinct"] else None,
                                                "max_abs_phi_t100_over_t0": float(np.interp(100.0, mp["t"], mp["v"]) / mp["v"][0])},
                        None, None, "the lump finder's threshold is relative to the maximum, so the count never "
                                    "reaches 0; measured values reported"))
    for name, rc in (("res", rr), ("box", rb)):
        if rc is None or r3 is None:
            its.append(item("%s check" % name, None, None, None, "%s check or E3 result missing" % name))
            continue
        its.append(item("%s check: same outcome class as E3" % name, [oc(rc), oc(r3)], "equal",
                        oc(rc) == oc(r3) if "not evaluated" not in (oc(rc), oc(r3)) else None,
                        "an outcome is not evaluated"))
        c1, c3 = contact(rc), contact(r3)
        rel = None if (c1 is None or c3 is None) else abs(c1 / c3 - 1)
        its.append(item("%s check: contact time within 15%% of E3" % name, {"check": c1, "E3": c3, "rel": rel}, 0.15,
                        None if rel is None else rel <= 0.15, "a contact time is missing"))
    rows["E15"] = {"runs": {k: m.get(k) for k in ("E15_twin", "E15_res", "E15_box")}, "items": its}
    for row in rows.values():
        v = [i["verdict"] for i in row["items"]]
        row["misses"] = sum(x == "miss" for x in v)
        row["hits"] = sum(x == "hit" for x in v)
        row["not_evaluable"] = sum(x == "not evaluable" for x in v)
    return {"mapping": m, "rows": rows,
            "all_misses": [(k, i["item"]) for k, row in rows.items() for i in row["items"] if i["verdict"] == "miss"]}


# ---------------------------------------------------------------------------
# plot and main
# ---------------------------------------------------------------------------

def plot(name, r, rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    a = ax[0, 0]
    a.step(r["counts"]["t_mid"], r["counts"]["count_distinct"], where="mid")
    a.axhline(r["K"], ls=":", c="k")
    a.set_title("distinct lumps (K = %d)" % r["K"])
    a = ax[0, 1]
    tr = r["tracked"]
    if tr["t"]:
        a.plot(tr["t"], tr["nn_ratio_min"], label="tracked nn ratio min")
        a.plot(tr["t"], tr["nn_ratio_max"], label="tracked nn ratio max")
    pk = [(t, q) for t, q in zip(r["peaks"]["t_mid"], r["peaks"]["ratios"]) if q]
    if pk:
        a.plot([t for t, _ in pk], [min(q) for _, q in pk], "ko", ms=3, label="peaks min")
        a.plot([t for t, _ in pk], [max(q) for _, q in pk], "ks", ms=3, label="peaks max")
    for y in (0.5, 1.5):
        a.axhline(y, ls=":", c="r")
    a.set_title("nearest-neighbour distance / initial")
    a.legend(fontsize=7)
    if "group" in r:
        ax[1, 0].plot(r["group"]["t"], r["group"]["E_group_over_0_period_mean"])
        ax[1, 0].set_title("group energy retained")
        ax[1, 1].plot(r["group"]["t"], r["group"]["L_axis_box"], label="L axis (box)")
        ax[1, 1].legend(fontsize=7)
        ax[1, 1].set_title("angular momentum")
    for a in ax.ravel():
        a.set_xlabel("t")
    fig.suptitle("%s: %s" % (name, rep.get("outcome", {}).get("outcome")))
    fig.tight_layout()
    fig.savefig(path + ".tmp.png", dpi=90)
    plt.close(fig)
    os.replace(path + ".tmp.png", path)


def _dump(obj, path):
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--criteria", default=os.path.join(HERE, "criteria.json"))
    ap.add_argument("--emap", default=os.path.join(HERE, "criteria_E_map.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "analysis"))
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    crit = json.load(open(a.criteria)) if os.path.exists(a.criteria) else None
    os.makedirs(a.out, exist_ok=True)
    runs, summary = {}, {"criteria_file": a.criteria if crit is not None else None, "runs": {}}
    for f in sorted(glob.glob(os.path.join(a.results, "*.json"))):
        res = json.load(open(f))
        if "lumps" not in res:
            continue
        name = os.path.basename(f)[:-5]
        r = record(res)
        rep = report(r)
        oc = outcome(r, crit)
        rep["outcome"] = oc
        runs[name] = {"record": r, "report": rep, "outcome": oc}
        _dump({"record": r, "report": rep}, os.path.join(a.out, name + ".json"))
        if not a.no_plots:
            plot(name, r, rep, os.path.join(a.out, name + ".png"))
        summary["runs"][name] = {"K": r["K"], "start": r["start"], "linear": r["linear"], "completed": r["completed"],
                                 "t_end": r["t_end"], "outcome": oc["outcome"], "end_state": oc.get("end_state"),
                                 "held_fraction": oc.get("held_fraction"), "contact": rep["contact"]["time"],
                                 "nn_passes_1.5": rep["nn_passes_1.5"]["time"],
                                 "E_retained_end": rep.get("group", {}).get("E_retained_end"),
                                 "per_lump_valid_until": r["per_lump_valid_until"]}
        print(name, oc["outcome"], oc.get("end_state") or "")
    _dump(summary, os.path.join(a.out, "summary.json"))
    if os.path.exists(a.emap):
        _dump(criteria_E(runs, json.load(open(a.emap))), os.path.join(a.out, "criteria_E.json"))


if __name__ == "__main__":
    main()
