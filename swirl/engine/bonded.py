"""The "bonded" verdict for a group of lumps (Session 3, WP0).

Two rules live here.

bonded_rule_session2   The rule written before the Session 2 arrangement runs (SESSION_02_PREDICTIONS.md,
                       Block 3). SUPERSEDED. It labels the far-apart control (E14) "bonded" because nothing
                       moves, and labelled E11's first run (T = 700) "bonded" because its collapse takes
                       about as long as the run (contact at 699). It has no trend test, no link between run
                       length and interaction time, and no comparison with a control. Kept so that the old
                       labels can be reproduced; do not use it for a claim.

bonded_rule            The corrected rule (SESSION_02_REPORT.md section 13). A run is "bonded" only if
                       every clause below holds AND the run responded to the interaction. Defaults are in
                       DEFAULTS and are arguments of the function.

  1. count      the distinct-lump count equals K in every count window from t_from (50) on.
  2. distance   every lump's nearest-neighbour distance stays within dist_tol (10%) of its start value.
  3. trend      no lump's nearest-neighbour distance has a linear trend (fit over t >= t_from) larger than
                trend_factor (3) times the far-apart control's own trend; the control's trend is floored at
                the noise floor of E14 (5e-4 in the distance ratio over T = 500).
  4. N held     no N exchange between lumps: each lump's relative N change from its first sample, minus
                the mean of all lumps' changes, stays within N_tol (1e-3 = 5 times the E14 floor of 2e-4,
                relative). The common drift is removed because an isolated lump loses N to radiation: in
                the far-apart control E14 every lump's N falls by 1.5e-3 over T = 500 (reported as
                "N_common_drift"), while the spread between its lumps is 1.5e-4 at most. The 2e-4 quoted
                for E14 in Session 2 is that spread, not the per-lump change. A group that radiates
                faster than the control is not caught by this clause (it is not a held-quantity test).
  5. phase held each neighbour phase step changes by no more than steps_tol_deg (0.25 degrees = 5 times the
                E14 floor of 0.05) from its first value.
  6. duration   the run lasts at least interaction_periods (5) interaction times; the unit is the in-phase
                control's contact time (an input, no default).
  7. response   the run's response (largest distance change, N change and phase-step change) exceeds
                response_factor (3) times the control's response, each floored at the E14 noise floors.
                A run that does not exceed it is indistinguishable from the far-apart control.

Verdicts, in this order of precedence:
  "not bonded"      any of clauses 1 to 5 fails (a held-quantity failure is a failure at any run length);
  "no interaction"  clauses 1 to 5 hold but clause 7 fails: the run did what the control did;
  "too short"       clauses 1 to 5 and 7 hold but clause 6 fails: held so far, not shown to be held;
  "bonded"          every clause holds.
The tolerances in 4 and 5 (5 times the floor) and the response factor (3) are the author's reading of
"scaled by a few times"; the window between them is 3 to 5 times the noise floor for N and phase steps.
Because the trend test is against a noise floor of 1e-6 per time unit, a bounded libration that does not
cover whole periods also fails it; that is intended (a bond must hold, not oscillate with a net drift).

series_from_result() builds the inputs from a stored WP-E result JSON.
"""

import numpy as np

from swirl.engine.lumps import pair_distances, summarize

DEFAULTS = {
    "t_from": 50.0,
    "dist_tol": 0.10,
    "trend_factor": 3.0,
    "N_tol": 1e-3,
    "steps_tol_deg": 0.25,
    "interaction_periods": 5.0,
    "response_factor": 3.0,
    "noise_nn": 5e-4,         # E14: distance ratio change over T = 500
    "noise_N": 2e-4,          # E14: per-lump N, relative
    "noise_steps_deg": 0.05,  # E14: neighbour phase steps
}


def _wrap_deg(x):
    return np.rad2deg(np.angle(np.exp(1j * np.deg2rad(np.asarray(x, float)))))


# ---------------------------------------------------------------------------
# Superseded rule
# ---------------------------------------------------------------------------

def bonded_rule_session2(t_mid, count_distinct, ratios, K, completed=True, eval_from=50.0,
                         nn_min=0.5, nn_max=1.5, hold_fraction=0.95, require_completed=True):
    """SUPERSEDED (Session 2, Block 3 as written; see the module docstring for why it is unsound).

    t_mid, count_distinct, ratios: per count window, the time, the distinct-lump count and the list of
    each lump's nearest-neighbour distance over its initial value (None where the count is not K).
    Returns {"bonded": bool, "held_fraction", "n_samples", "last_ok", "distinct_at_start"} where bonded
    means: ok (count = K and every ratio in [nn_min, nn_max]) in at least hold_fraction of the windows
    from eval_from on and in the last one, distinct at the start, run completed. The merged / flew-apart /
    bounded classes of the old outcome() are not reproduced here."""
    rows = []
    for t, n, q in zip(t_mid, count_distinct, ratios):
        if t < eval_from:
            continue
        ok = (n == K) and q is not None and min(q) >= nn_min and max(q) <= nn_max
        rows.append(bool(ok))
    if not rows:
        return {"bonded": False, "held_fraction": None, "n_samples": 0, "last_ok": None,
                "distinct_at_start": None}
    held = float(np.mean(rows))
    distinct_start = count_distinct[0] >= K
    return {"bonded": bool(distinct_start and held >= hold_fraction and rows[-1]
                           and (completed or not require_completed)),
            "held_fraction": held, "n_samples": len(rows), "last_ok": rows[-1],
            "distinct_at_start": bool(distinct_start)}


# ---------------------------------------------------------------------------
# Inputs from a stored result
# ---------------------------------------------------------------------------

def series_from_result(res):
    """Inputs of bonded_rule from a WP-E result JSON (the dict of outputs/session2/wpE/results/*.json).

    Returns {"K", "t_end", "completed", "t_count", "count" (distinct count per window), "t" (whole-period
    window centres of lump 0, kept while the count is K), "nn" (n x K nearest-neighbour distances of the
    tracked lump centroids), "N" (n x K per-lump N), "steps" (n x E neighbour phase steps, degrees), "edges"}.
    The tracked series are cut at the first count window with fewer than K distinct lumps (after a merger
    they are not per-lump quantities)."""
    p = res["params"]
    cand = p.get("_candidate", {})
    L = p["L"]
    raw = res["lumps"]["raw"]
    K = len(raw["centers"])
    T = float(res["t"])
    st = {k: raw[k] for k in ("centers", "series", "point")}
    av = summarize(st, np.arange(0.0, T + 1e-9, 1.0), p["lumps"]["avg_window"])
    cnt = res["lumps"]["counts"]
    win_len = p["lumps"]["count_periods"] * 2 * np.pi / p["lumps"]["omega_nominal"]
    bad = [c["t_mid"] for c in cnt if c["count_distinct"] < K]
    t_valid = (bad[0] - win_len / 2) if bad else np.inf
    out = {"K": K, "t_end": T, "completed": bool(res["completed"]),
           "t_count": [c["t_mid"] for c in cnt], "count": [c["count_distinct"] for c in cnt],
           "t": np.array([]), "nn": np.zeros((0, K)), "N": np.zeros((0, K)), "steps": np.zeros((0, 0)),
           "edges": []}
    lu0 = av["lumps"][0]
    tc0 = np.array([np.nan if x is None else x for x in lu0["t_center"]])
    ok0 = ~np.isnan(tc0) & (tc0 < t_valid)
    if K < 2 or ok0.sum() < 2:
        return out
    tt = tc0[ok0]
    X = np.zeros((tt.size, K, 3))
    Nk = np.zeros((tt.size, K))
    for k in range(K):
        lk = av["lumps"][k]
        sel = [i for i, (x, Xv, Nv) in enumerate(zip(lk["t_center"], lk["X"], lk["N"]))
               if x is not None and Xv is not None and Nv is not None]
        tk = np.array([lk["t_center"][i] for i in sel])
        for a in range(3):
            X[:, k, a] = np.interp(tt, tk, [lk["X"][i][a] for i in sel])
        Nk[:, k] = np.interp(tt, tk, [lk["N"][i] for i in sel])
    nn = np.zeros((tt.size, K))
    for i in range(tt.size):
        D = pair_distances(X[i], L)
        np.fill_diagonal(D, np.inf)
        nn[i] = D.min(1)
    pos0 = np.asarray(cand.get("positions", raw["series"]["centers_used"][0]), float)
    D0 = pair_distances(pos0, L)
    np.fill_diagonal(D0, np.inf)
    edges = [(i, j) for i in range(K) for j in range(i + 1, K) if D0[i, j] <= 1.01 * D0.min()]
    rel = [np.zeros(tc0.size)] + [np.array([np.nan if x is None else x for x in q["dtheta"]])
                                  for q in av["relative_phase"]]
    steps = np.array([[np.rad2deg(np.angle(np.exp(1j * (rel[b][i] - rel[a][i])))) for a, b in edges]
                      for i in np.where(ok0)[0]])
    good = ~np.any(np.isnan(steps), axis=1)
    out.update(t=tt[good], nn=nn[good], N=Nk[good], steps=steps[good], edges=edges)
    return out


# ---------------------------------------------------------------------------
# Corrected rule
# ---------------------------------------------------------------------------

def _metrics(run, t_from, noise):
    """Held-quantity metrics of a run: distance change, distance trend (per time unit), N change, phase-step
    change. Series are cut to t >= t_from for the trend; changes are from the first sample."""
    t, nn, N, st = run["t"], run["nn"], run["N"], run["steps"]
    if t.size < 3:
        return {"nn_dev": np.nan, "nn_slope": np.nan, "N_dev": np.nan, "steps_dev_deg": np.nan, "n_samples": int(t.size)}
    ratio = nn / nn[0]
    sel = t >= t_from
    slope = np.nan
    if sel.sum() >= 3:
        slope = float(max(abs(np.polyfit(t[sel], ratio[sel, k], 1)[0]) for k in range(ratio.shape[1])))
    Nrel = N / N[0] - 1.0
    Nexch = Nrel - Nrel.mean(axis=1, keepdims=True)   # exchange between lumps: the common drift removed
    dsteps = _wrap_deg(st - st[0]) if st.size else np.zeros((1, 1))
    return {"nn_dev": float(np.max(np.abs(ratio - 1.0))), "nn_slope": slope,
            "N_dev": float(np.max(np.abs(Nexch))), "N_common_drift": float(np.mean(Nrel[-1])),
            "steps_dev_deg": float(np.max(np.abs(dsteps))),
            "n_samples": int(t.size)}


def bonded_rule(run, control, contact_time, **kw):
    """Corrected bonded verdict (module docstring has the clauses). run, control: dicts from
    series_from_result (the control is the far-apart run, Session 2's E14); contact_time: the in-phase
    control's contact time, the interaction time. Keyword arguments override DEFAULTS.

    Returns {"verdict": "bonded" | "not bonded" | "no interaction" | "too short", "reason": str,
    "clauses": {name: {"value", "limit", "pass"}}, "parameters": {...}, "run_metrics", "control_metrics"}.
    A clause whose value could not be measured (too few samples) fails, with value None."""
    if run["K"] < 2:
        raise ValueError("K = 1: no neighbours, the rule does not apply")
    if control is None or contact_time is None or not contact_time > 0:
        raise ValueError("the far-apart control series and the in-phase contact time are inputs, not defaults")
    prm = dict(DEFAULTS)
    unknown = set(kw) - set(prm)
    if unknown:
        raise TypeError("unknown parameters: %s" % sorted(unknown))
    prm.update(kw)
    t_from = prm["t_from"]
    m = _metrics(run, t_from, prm)
    mc = _metrics(control, t_from, prm)
    clauses = {}

    def put(name, value, limit, ok):
        clauses[name] = {"value": None if (value is None or (isinstance(value, float) and np.isnan(value))) else value,
                         "limit": limit, "pass": bool(ok)}

    # 1 count
    cs = [n for t, n in zip(run["t_count"], run["count"]) if t >= t_from]
    put("count_equals_K", int(min(cs)) if cs else None, "== %d in all %d windows from t = %g" % (run["K"], len(cs), t_from),
        bool(cs) and all(n == run["K"] for n in cs))
    # 2 distance
    put("distance_within_tol", m["nn_dev"], prm["dist_tol"], m["nn_dev"] <= prm["dist_tol"])
    # 3 trend, per time unit; control floor = its own measured slope or the E14 floor spread over its length
    t_ctrl = float(control["t"][-1] - t_from) if control["t"].size and control["t"][-1] > t_from else float("nan")
    floor_slope = prm["noise_nn"] / t_ctrl if t_ctrl == t_ctrl else float("nan")
    known = [x for x in (mc["nn_slope"], floor_slope) if x == x]
    ctrl_slope = max(known) if known else float("nan")
    lim = prm["trend_factor"] * ctrl_slope
    put("no_distance_trend", m["nn_slope"], lim, m["nn_slope"] == m["nn_slope"] and lim == lim and m["nn_slope"] <= lim)
    # 4, 5 held
    put("N_held", m["N_dev"], prm["N_tol"], m["N_dev"] == m["N_dev"] and m["N_dev"] <= prm["N_tol"])
    put("phase_steps_held_deg", m["steps_dev_deg"], prm["steps_tol_deg"],
        m["steps_dev_deg"] == m["steps_dev_deg"] and m["steps_dev_deg"] <= prm["steps_tol_deg"])
    # 6 duration
    need = prm["interaction_periods"] * float(contact_time)
    put("duration", float(run["t_end"]), need, run["t_end"] >= need)
    # 7 response (each metric against the control's, floored at the noise floor)
    rf = prm["response_factor"]
    resp = {}
    for key, floor, name in (("nn_dev", prm["noise_nn"], "distance"), ("N_dev", prm["noise_N"], "N"),
                             ("steps_dev_deg", prm["noise_steps_deg"], "phase steps")):
        base = max(floor, mc[key]) if mc[key] == mc[key] else floor
        resp[name] = {"run": m[key], "control_limit": rf * base,
                      "exceeds": bool(m[key] == m[key] and m[key] > rf * base)}
    put("response_differs_from_control", resp, "any measure above %g times the control (floored)" % rf,
        any(v["exceeds"] for v in resp.values()))
    held = all(clauses[k]["pass"] for k in ("count_equals_K", "distance_within_tol", "no_distance_trend",
                                           "N_held", "phase_steps_held_deg"))
    if not held:
        failed = [k for k in ("count_equals_K", "distance_within_tol", "no_distance_trend", "N_held",
                              "phase_steps_held_deg") if not clauses[k]["pass"]]
        verdict, reason = "not bonded", "fails: " + ", ".join(failed)
    elif not clauses["response_differs_from_control"]["pass"]:
        verdict, reason = "no interaction", "holds everything, but does not differ from the far-apart control"
    elif not clauses["duration"]["pass"]:
        verdict, reason = "too short", "t_end %.4g is below %g interaction times (%.4g)" % (
            run["t_end"], prm["interaction_periods"], need)
    else:
        verdict, reason = "bonded", "every clause holds"
    if not clauses["duration"]["pass"] and verdict == "not bonded":
        reason += "; also too short"
    return {"verdict": verdict, "reason": reason, "clauses": clauses, "parameters": {**prm, "contact_time": float(contact_time)},
            "run_metrics": m, "control_metrics": mc}
