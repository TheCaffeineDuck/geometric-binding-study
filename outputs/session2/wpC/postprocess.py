"""WP-C post-processing, version 2: result JSON -> per-run analysis, criteria table, plots.

Usage:
  <venv python> outputs/session2/wpC/postprocess.py [--results DIR] [--out DIR] [--test-record-end-as-T]
Defaults: results = outputs/session2/wpC/results, out = outputs/session2/wpC/analysis.
Works on whatever results exist (complete or partial). Uses the stored series only (the
engine's whole-period averages at window centres t_center, the raw lump samples, the lump
counts); never re-runs anything. The predicted force never enters a fit: predictions are
only evaluated along the measured d(t), dtheta(t), omega(t) and compared.

Criteria: SESSION_02_PREDICTIONS.md Block 2 rows C1 to C13 with the Block 2a amendment.
Writes analysis/criteria.json (one entry per row: measured values, tolerance, hit or miss,
or "not evaluable" with the reason), analysis/summary.json (all per-run numbers) and one
PNG per run.

Definitions (fixed here; every evaluator choice not written in Block 2/2a is listed in
EVALUATOR_CHOICES and copied into criteria.json):
  d(t)       |X_1 - X_0| of the period-averaged ball centroids, lump 1 interpolated to lump 0's
             window centres.
  dtheta(t)  relative phase theta_1 - theta_0 (lump 1 at +d/2), Hann-weighted whole-period
             projection, unwrapped in time.
  start      first window centred at t >= 15 (Block 2a): d_start, N_start, E_start.
  shift      (d_start - d(T)) / 2, positive toward the partner.
  omega_m    mean of the two lumps' window omega at the same time; A(omega) from a cubic spline
             of the beta = 4/3 family table, kappa = sqrt(1 - omega^2).
  F_pred(t)  2 pi A^2 cos(dtheta) e^(-kappa d)(kappa/d + 1/d^2) at d(t), dtheta(t), omega_m(t).
  measure i  sliding half-space force: slope of the averaged momentum in x < 0 over windows of
             SLIDE time units centred at t_c >= 100, while two distinct lumps remain (no count
             window up to t_c + SLIDE/2 has count_distinct = 1, and d > 2 hw in the window);
             compared with the window mean of F_pred. Also reported: the slope of
             P(x<0) - P_total/2 (removes a common translation; P_total is not zero for the
             -/+ theta pairs, see notes).
  measure ii d(t) against the point model d'' = -2 F(d, dtheta_meas(t))/E from d0 at rest at
             t = 0 (E = 2801.7, A and kappa at the run's median omega over t_center 30..80);
             ratio = (d(T_v) - d_start) / (d_model(T_v) - d_model(t_start)) with T_v the end of
             the two-distinct-lump span. A second model with its own phase dynamics (pendulum
             with N exchange, dtheta(0) = measured start value) is reported.
  measure iii early slope [10, 60] with the single-lump contributions subtracted (d = 24 only):
             P_pair(x<0) - P_single(theta_2)(x<0) + [P_total - P(x<0)]_single(|theta_1|),
             the second from the mirror image of the +12 single; for theta_1 = -theta the +theta
             single stands in (exact at dtheta = 0 and 180).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PROFILE = os.path.join(ROOT, "outputs/step2/profiles/profile_beta1.3333_w0.9091.json")
FAMILY = os.path.join(ROOT, "outputs/family/family_beta1.3333.json")

SLIDE = 40.0
T_VALID = 100.0
T_START = 15.0
EARLY_F = (10.0, 60.0)
EARLY_N = (15.0, 60.0)
RATIO_MIN_PRED = 0.25          # windows with |F_pred| < 0.25 F0 have no defined ratio
PROMINENCE = 0.05              # interior extremum of d(t) for the C10 bonded test (between the C12
                               # box and resolution tolerances; start-up wiggles of ~1e-3 are not extrema)
L_REL_TOL = 1e-8               # C13: |L| / (E d0) counted as zero to noise
N_SYM_TOL = 1e-9               # C8: |N0 - N1| / N at rounding
TWIN_ATTRACT = 0.05            # C11: d(100) - d(0) below -0.05 counts as attraction
DESIGN_T = {16: 300.0, 20: 500.0, 24: 500.0}

EVALUATOR_CHOICES = [
    "start values from the first window centred at t >= 15 (Block 2a)",
    "measure (i): SLIDE = 40 time units; valid windows centred at t >= 100 with two distinct lumps "
    "(count_distinct = 2 in every count window up to the window end) and d > 2 hw; ratio = slope / "
    "window-mean F_pred, undefined (excluded, count reported) where |F_pred| < 0.25 F0",
    "C2 per run: hit needs measure (i) worst-window ratio and measure (ii) ratio both inside the tolerance; "
    "at d = 16 the early half-space ratio is a third item (tolerance 20%)",
    "C1 sign 'at first': sign of d(min(100, T_v)) - d_start (measure ii); measure (i) first valid window "
    "and the d = 16 early half-space sign reported",
    "C3: kappa and a free amplitude fitted to |F_w| / |cos dtheta_w| = C e^(-kappa d)(kappa/d + 1/d^2) "
    "over all valid windows of the dtheta = 0 and 180 runs (log least squares, kappa scanned 0.05..1.5, "
    "no kappa_0 used); needs valid windows at >= 2 separations",
    "C4: least-squares line of F_w/F0_w against cos(dtheta_w) over valid windows of all runs at one d; "
    "needs >= 2 runs",
    "C5b: the flow between the lumps is the antisymmetric rate (dN_0/dt - dN_1/dt)/2 over t_center in "
    "[15, 60]; per-lump rates reported; compared with the on-record rate at that d",
    "C5c at d = 16: the stated prediction 'beyond 180' is checked (no tolerance given)",
    "C5d at d = 24 and C6/C7/C8/C9 shifts use T_end = the design T (300 at d = 16, 500 otherwise); a record "
    "shorter than that is 'not evaluable'",
    "C6 d = 16 'not merged, separating': not merged and dd/dt over the last 100 > 0; d = 20 'turning': "
    "outcome label 'bounded'",
    "C7 contact time: first window centre with d <= 2 hw (the point model's contact); the first count window "
    "with one distinct lump is reported; retained energy = (E_0 + E_1) at the end over the start values, "
    "only where the last count window has one distinct lump",
    "C8 N symmetry: max over the record of |N_0 - N_1| / mean N <= 1e-9",
    "C10 interior extremum: scipy find_peaks on d(t) after the start with prominence >= 0.05 (first version "
    "1e-3 flagged start-up wiggles of 1.1e-3 on test data as a bond); the largest prominence of any extremum "
    "is reported for every run",
    "C11: ball energy (raw sample nearest t = 100) / its t = 0 value < 0.05 for each lump; attraction if "
    "d(100) - d(0) < -0.05 (raw centroids)",
    "C12: max |d_check(t) - d_partner(t)| over the common span of window centres (from the start)",
    "C13: max |L| over the raw record / (E_lump d0) <= 1e-8",
]

# On-record predicted numbers (SESSION_02_PREDICTIONS.md Block 2)
REC = {
    "C5b_rate": {16: 1.61, 20: 0.244, 24: 0.038},
    "C5c_end_deg": {20: 160.0, 24: 102.0},
    "C5d_shift": {16: -2.8, 20: -0.89, 24: -0.03},
    "C6_shift_d24": 0.59,
    "C7_contact": {16: 126.0, 20: 346.0},
    "C7_shift_d24": 0.90,
    "C7_retained": (0.6, 0.9),
    "C8_shift": {16: -5.7, 20: -3.2, 24: -0.71},
    "C9_shift": {16: -4.9, 20: -2.6, 24: -0.52},
    "C3_kappa": 0.416,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_constants():
    prof = json.load(open(PROFILE))
    fam = json.load(open(FAMILY))
    rows = fam["rows_rescaled"]
    w = np.array([r["omega"] for r in rows])
    Aw = np.array([r["A"] for r in rows]) / np.sqrt(prof["g4"])
    o = np.argsort(w)
    dNdw = np.gradient(np.array([r["N"] for r in rows])[o] / prof["g4"], w[o])
    return prof, (w[o], Aw[o], dNdw)


def A_of(w, fam):
    """A(omega) from a cubic spline of the family table (spacing 0.005; linear interpolation
    would bias A^2 by ~2% between nodes)."""
    from scipy.interpolate import CubicSpline
    return CubicSpline(fam[0], fam[1])(w)


def F_law(d, dth, A, kap):
    return 2 * np.pi * A ** 2 * np.cos(dth) * np.exp(-kap * d) * (kap / d + 1 / d ** 2)


def C_N(d, A, kap):
    return 2 * np.pi * A ** 2 * np.exp(-kap * d) / d


def clean(t, v):
    """Drop None, sort by t, merge duplicate t (consecutive outputs may share a window)."""
    ok = [i for i in range(len(t)) if t[i] is not None and v[i] is not None]
    if not ok:
        return np.array([]), np.array([])
    tt = np.array([t[i] for i in ok], float)
    vv = np.array([v[i] for i in ok], float)
    tt, idx = np.unique(tt, return_index=True)
    return tt, vv[idx]


def slope(t, y):
    return float(np.polyfit(t, y, 1)[0]) if t.size > 3 else None


def ftrue(x):
    return None if x is None else float(x)


# ---------------------------------------------------------------------------
# per run
# ---------------------------------------------------------------------------

def run_basics(name, res, prof):
    p = res["params"]
    lp = p["lumps"]
    K = len(lp["centers"])
    av = res["lumps"]["averages"]
    raw = res["lumps"]["raw"]["series"]
    out = {"name": name, "K": K, "completed": res["completed"], "t_reached": res["t"],
           "N": p["N"], "L": p["L"], "dt": p["dt"], "g4": p["g4"], "g6": p["g6"],
           "linear_twin": p["g4"] == 0 and p["g6"] == 0,
           "phases": [r.get("phase", 0.0) for r in p["init"]["radial"]], "centers": lp["centers"],
           "E0_total": res["summary"]["E0"], "max_rel_dev_E_plus_absorbed": res["summary"]["max_rel_dev_E_plus_absorbed"],
           "steps_per_second": res.get("steps_per_second_this_call")}
    tr = []
    for k in range(K):
        lu = av["lumps"][k]
        t, _ = clean(lu["t_center"], [None if x is None else 0.0 for x in lu["X"]])
        Xc = [clean(lu["t_center"], [None if x is None else x[a] for x in lu["X"]])[1] for a in range(3)]
        tN, N = clean(lu["t_center"], lu["N"])
        tE, E = clean(lu["t_center"], lu["E"])
        tw, w = clean(lu["t_center"], lu["omega"])
        tr.append({"t": t, "X": np.stack(Xc, 1) if t.size else np.zeros((0, 3)), "tN": tN, "N": N,
                   "tE": tE, "E": E, "tw": tw, "w": w})
    rt = np.array(raw["t"])
    out["_tr"] = tr
    out["_Ph"] = clean(av["t_center_lump0"], [None if x is None else x[0] for x in (av["P_half"] or [])])
    out["_Ptot"] = clean(av["t_center_lump0"], [None if x is None else x[0] for x in (av["P_total"] or [])])
    out["_raw_t"] = rt
    out["_raw_E"] = np.array(raw["E"]).reshape(rt.size, K)
    out["_raw_X"] = np.array(raw["X"]).reshape(rt.size, K, 3)
    Pt = np.array(raw["P_total"])
    La = np.array(raw["L_axes"]) if raw["L_axes"] and len(raw["L_axes"][0]) else np.zeros((rt.size, 0))
    out["momentum_extremes"] = {"P_x_min": float(Pt[:, 0].min()), "P_x_max": float(Pt[:, 0].max()),
                                "P_y_maxabs": float(np.abs(Pt[:, 1]).max()), "P_z_maxabs": float(np.abs(Pt[:, 2]).max()),
                                "P_x_t0": float(Pt[0, 0])}
    out["angular_momentum_maxabs"] = [float(np.abs(La[:, j]).max()) for j in range(La.shape[1])]
    out["angular_momentum_axes"] = lp.get("axes", [])
    out["counts"] = [{"t_mid": c["t_mid"], "count": c["count"], "count_distinct": c.get("count_distinct", c["count"]),
                      "saddle_ratios": [q["saddle_ratio"] for q in c["pairs"]]} for c in res["lumps"]["counts"]]
    # start and settled values per lump
    out["lumps"] = []
    for k in range(K):
        r = tr[k]
        i0 = int(np.argmax(r["tN"] >= T_START)) if np.any(r["tN"] >= T_START) else None
        j0 = int(np.argmax(r["tE"] >= T_START)) if np.any(r["tE"] >= T_START) else None
        late = r["tw"] >= (r["tw"][-1] - 100.0) if r["tw"].size else np.array([], bool)
        out["lumps"].append({
            "N_start": None if i0 is None else float(r["N"][i0]), "t_N_start": None if i0 is None else float(r["tN"][i0]),
            "E_start": None if j0 is None else float(r["E"][j0]),
            "N_first_window": float(r["N"][0]) if r["N"].size else None,
            "omega_first_window": float(r["w"][0]) if r["w"].size else None,
            "N_end": float(r["N"][-1]) if r["N"].size else None, "E_end": float(r["E"][-1]) if r["E"].size else None,
            "omega_settled_last100": float(np.median(r["w"][late])) if late.any() else None,
            "omega_median_30_80": (float(np.median(r["w"][(r["tw"] >= 30) & (r["tw"] <= 80)]))
                                   if np.any((r["tw"] >= 30) & (r["tw"] <= 80)) else None),
            "raw_E_t0": float(out["_raw_E"][0, k])})
    return out


def analyse_single(out):
    r = out["_tr"][0]
    X = r["X"]
    if X.shape[0] == 0:
        return out
    i0 = int(np.argmax(r["t"] >= T_START))
    out["single"] = {"x_start": float(X[i0, 0]), "x_end": float(X[-1, 0]),
                     "drift_vec_end_minus_start": (X[-1] - X[i0]).tolist(),
                     "drift_norm": float(np.linalg.norm(X[-1] - X[i0])),
                     "max_abs_drift": float(np.max(np.linalg.norm(X[i0:] - X[i0], axis=1)))}
    return out


def pendulum_model(d0, dth0, T, A, kap, dwdN, E):
    def rhs(t, y):
        d, v, ph, n = y
        d = max(d, 1e-3)
        return [v, -2 * F_law(d, ph, A, kap) / E, 2 * abs(dwdN) * n, C_N(d, A, kap) * np.sin(ph)]

    def hit(t, y):
        return y[0] - 2 * 3.344
    hit.terminal = True
    return solve_ivp(rhs, [0, T], [d0, 0.0, dth0, 0.0], rtol=1e-9, atol=1e-11, events=hit, dense_output=True,
                     max_step=1.0)


def analyse_pair(out, res, prof, fam):
    hw, E_l = prof["half_width"], prof["E"]
    tr = out["_tr"]
    av = res["lumps"]["averages"]
    t0 = tr[0]["t"]
    X0, X1 = tr[0]["X"], tr[1]["X"]
    if t0.size < 5 or tr[1]["t"].size < 5:
        out["pair_note"] = "too few windows"
        return out
    X1i = np.stack([np.interp(t0, tr[1]["t"], X1[:, a]) for a in range(3)], 1)
    d = np.linalg.norm(X1i - X0, axis=1)
    c = out["centers"]
    d0 = float(np.linalg.norm(np.array(c[1]) - np.array(c[0])))
    th = out["phases"]
    dnom = th[1] - th[0]
    out.update(d0=d0, dtheta_nominal=float(dnom), dtheta_nominal_deg=float(np.rad2deg(dnom)))
    # relative phase (unwrapped) on lump 0's window centres
    tph, ph = clean(av["t_center_lump0"], av["relative_phase"][0]["dtheta"])
    ph = np.unwrap(ph) if ph.size else ph
    if ph.size and dnom > 0 and ph[0] < 0 and abs(dnom - np.pi) < 1e-9:
        ph = ph + 2 * np.pi
    # omega_m(t)
    wm_t = tr[0]["tw"]
    wm = 0.5 * (tr[0]["w"] + np.interp(wm_t, tr[1]["tw"], tr[1]["w"]))
    s0 = int(np.argmax(t0 >= T_START)) if np.any(t0 >= T_START) else 0
    t_s, d_s = float(t0[s0]), float(d[s0])
    dth_s = float(np.interp(t_s, tph, ph)) if ph.size else float(dnom)
    out.update(t_start=t_s, d_start=d_s, dtheta_start_deg=float(np.rad2deg(dth_s)),
               sep_t=t0.tolist(), sep=d.tolist(), dtheta_t=tph.tolist(), dtheta_deg=np.rad2deg(ph).tolist())
    # merged / distinct bookkeeping
    first_merge = next((cc["t_mid"] for cc in out["counts"] if cc["count_distinct"] == 1), None)
    t_contact = next((float(t0[i]) for i in range(s0, t0.size) if d[i] <= 2 * hw), None)
    end_valid = min([x for x in (first_merge, t_contact, float(t0[-1])) if x is not None])
    out.update(t_first_count_distinct_1=first_merge, t_contact_d_le_2hw=t_contact, t_valid_end=end_valid)

    def preds(tt):
        dd = np.interp(tt, t0, d)
        pp = np.interp(tt, tph, ph) if ph.size else np.full_like(tt, dnom)
        ww = np.minimum(np.interp(tt, wm_t, wm), 0.999)   # guard: omega >= m has no bound tail
        Aw, kw = A_of(ww, fam), np.sqrt(1 - ww ** 2)
        return F_law(dd, pp, Aw, kw), F_law(dd, 0.0, Aw, kw), dd, pp, ww, Aw, kw

    # half-space momentum (plane 0: x < 0) and total momentum, averaged
    tp, Ph = clean(av["t_center_lump0"], [None if x is None else x[0] for x in (av["P_half"] or [])])
    tq, Ptot = clean(av["t_center_lump0"], [None if x is None else x[0] for x in (av["P_total"] or [])])
    Ptot_i = np.interp(tp, tq, Ptot) if tq.size else np.zeros_like(tp)
    out["P_half_t"], out["P_half"] = tp.tolist(), Ph.tolist()
    # early slope (reported for every run; criterion only at d = 16)
    m = (tp >= EARLY_F[0]) & (tp <= EARLY_F[1])
    if m.sum() > 3:
        Fe = slope(tp[m], Ph[m])
        Fp, F0, *_ = preds(tp[m])
        out["F_early_halfspace"] = Fe
        out["F_early_pred_mean"] = float(Fp.mean())
        out["F_early_ratio"] = Fe / float(Fp.mean()) if abs(Fp.mean()) > 0 else None
    # measure (i): sliding windows
    win = []
    for tc in tp:
        if tc - SLIDE / 2 < tp[0] or tc + SLIDE / 2 > tp[-1]:
            continue
        mm = (tp >= tc - SLIDE / 2) & (tp <= tc + SLIDE / 2)
        if mm.sum() < 6:
            continue
        Fs = slope(tp[mm], Ph[mm])
        Fs_rel = slope(tp[mm], Ph[mm] - 0.5 * Ptot_i[mm])
        Fp, F0, dd, pp, ww, *_ = preds(tp[mm])
        merged_by = first_merge is not None and first_merge <= tc + SLIDE / 2
        valid = tc >= T_VALID and not merged_by and dd.min() > 2 * hw and (t_contact is None or t_contact > tc + SLIDE / 2)
        fp, f0 = float(Fp.mean()), float(F0.mean())
        win.append({"t_c": float(tc), "F": Fs, "F_minus_half_Ptotal": Fs_rel, "F_pred": fp, "F0": f0,
                    "d": float(dd.mean()), "d_min": float(dd.min()), "dtheta_deg": float(np.rad2deg(pp.mean())), "omega": float(ww.mean()),
                    "valid": bool(valid),
                    "ratio": (Fs / fp) if (valid and abs(fp) >= RATIO_MIN_PRED * abs(f0)) else None,
                    "resid_over_F0": (Fs - fp) / f0 if valid else None})
    out["sliding"] = win
    vw = [w for w in win if w["valid"]]
    rat = [w["ratio"] for w in vw if w["ratio"] is not None]
    out["measure_i"] = {"n_valid_windows": len(vw), "n_ratio_defined": len(rat),
                        "n_excluded_small_pred": len(vw) - len(rat),
                        "ratio_median": float(np.median(rat)) if rat else None,
                        "ratio_worst": (float(max(rat, key=lambda x: abs(x - 1))) if rat else None),
                        "abs_resid_over_F0_worst": (float(max(abs(w["resid_over_F0"]) for w in vw)) if vw else None),
                        "ratio_worst_d_ge_12": (lambda q: float(max(q, key=lambda x: abs(x - 1))) if q else None)(
                            [w["ratio"] for w in vw if w["ratio"] is not None and w["d_min"] >= 12.0]),
                        "t_c_range": [vw[0]["t_c"], vw[-1]["t_c"]] if vw else None}
    # measure (ii): point model with the measured dtheta(t)
    om_med = np.median([x["omega_median_30_80"] for x in out["lumps"] if x["omega_median_30_80"] is not None]) \
        if any(x["omega_median_30_80"] is not None for x in out["lumps"]) else prof["omega"]
    om_med = min(float(om_med), 0.999)
    Am, km = float(A_of(om_med, fam)), float(np.sqrt(1 - om_med ** 2))
    T_end = float(t0[-1])

    def dth_meas(t):
        return float(np.interp(t, tph, ph)) if ph.size else float(dnom)

    def rhs(t, y):
        return [y[1], -2 * F_law(max(y[0], 1e-3), dth_meas(t), Am, km) / E_l]

    def hit(t, y):
        return y[0] - 2 * hw
    hit.terminal = True
    sol = solve_ivp(rhs, [0, T_end], [d0, 0.0], rtol=1e-9, atol=1e-11, events=hit, dense_output=True, max_step=1.0)
    t_m = t0[t0 <= sol.t[-1]]
    d_mod = sol.sol(t_m)[0]
    i_v = int(np.searchsorted(t0, end_valid, side="right")) - 1
    i_v = min(i_v, t_m.size - 1)
    dd_meas = d[s0:i_v + 1] - d_s
    dd_mod = d_mod[s0:i_v + 1] - d_mod[s0]
    ratio_ii = float(dd_meas[-1] / dd_mod[-1]) if (dd_mod.size and abs(dd_mod[-1]) > 1e-9) else None
    out["measure_ii"] = {"omega_used": float(om_med), "A_used": Am, "kappa_used": km, "E_used": E_l,
                         "T_v": float(t0[i_v]) if i_v >= s0 else None,
                         "delta_d_measured": float(dd_meas[-1]) if dd_meas.size else None,
                         "delta_d_model": float(dd_mod[-1]) if dd_mod.size else None,
                         "ratio": ratio_ii,
                         "max_abs_residual": float(np.max(np.abs(dd_meas - dd_mod))) if dd_meas.size else None,
                         "model_contact_t": float(sol.t[-1]) if sol.status == 1 else None,
                         "d_model_t": t_m.tolist(), "d_model": d_mod.tolist()}
    # second model: own phase dynamics (pendulum with N exchange), dtheta(0) = measured start value
    dwdN = 1.0 / float(np.interp(om_med, fam[0], fam[2]))
    sol2 = pendulum_model(d0, dth_s, T_end, Am, km, dwdN, E_l)
    t2 = t0[t0 <= sol2.t[-1]]
    y2 = sol2.sol(t2)
    out["model_own_phase"] = {"d_t": t2.tolist(), "d": y2[0].tolist(), "dtheta_deg": np.rad2deg(y2[2]).tolist(),
                              "contact_t": float(sol2.t[-1]) if sol2.status == 1 else None,
                              "dtheta_end_deg": float(np.rad2deg(y2[2][-1])) if t2.size else None,
                              "d_end": float(y2[0][-1]) if t2.size else None}
    # measure (ii) as a force for the kappa fit (indicative): quadratic over [t_s, t_q]
    far = np.nonzero(np.abs(d - d_s) > 0.5)[0]
    t_q = min(end_valid, float(t0[far[0]]) if far.size else np.inf, 0.5 * T_end + 0.5 * t_s if T_end > 2 * t_s else T_end)
    mq = (t0 >= t_s) & (t0 <= t_q)
    if mq.sum() > 8:
        cq = np.polyfit(t0[mq], d[mq], 2)
        out["quadratic_early"] = {"window": [t_s, float(t_q)], "d_ddot": float(2 * cq[0]),
                                  "F_from_d": float(-E_l * cq[0]), "d_mean": float(d[mq].mean()),
                                  "dtheta_mean_deg": float(np.rad2deg(np.mean([dth_meas(x) for x in t0[mq]])))}
    # N flow
    n0, n1 = tr[0], tr[1]
    me = (n0["tN"] >= EARLY_N[0]) & (n0["tN"] <= EARLY_N[1])
    if me.sum() > 3:
        N1i = np.interp(n0["tN"], n1["tN"], n1["N"])
        r0, r1 = slope(n0["tN"][me], n0["N"][me]), slope(n0["tN"][me], N1i[me])
        anti = 0.5 * (r0 - r1)
        sth = float(np.mean(np.sin([dth_meas(x) for x in n0["tN"][me]])))
        pred_meas = float(C_N(d_s, Am, km) * sth)
        leader = 1 if sth > 0 else 0
        flow_dir = "lump1->lump0" if anti > 0 else "lump0->lump1"
        out["N_flow_early"] = {"window": list(EARLY_N), "dN0_dt": r0, "dN1_dt": r1, "antisymmetric_rate": anti,
                               "pred_measured_inputs_C_N_sin": pred_meas, "mean_sin_dtheta": sth,
                               "leader": "lump%d" % leader, "flow": flow_dir,
                               "leader_to_lagger": bool((anti > 0) == (leader == 1)) if abs(anti) > 0 else None}
        sl = []
        for tc in n0["tN"]:
            mm = (n0["tN"] >= tc - SLIDE / 2) & (n0["tN"] <= tc + SLIDE / 2)
            if mm.sum() < 6 or tc - SLIDE / 2 < n0["tN"][0] or tc + SLIDE / 2 > n0["tN"][-1]:
                continue
            a = 0.5 * (slope(n0["tN"][mm], n0["N"][mm]) - slope(n0["tN"][mm], N1i[mm]))
            dd = float(np.interp(tc, t0, d))
            sl.append({"t_c": float(tc), "antisymmetric_rate": a,
                       "pred": float(C_N(dd, Am, km) * np.sin(dth_meas(tc)))})
        out["N_flow_sliding"] = sl
    out["N_sym_max_rel"] = (float(np.max(np.abs(n0["N"] - np.interp(n0["tN"], n1["tN"], n1["N"])) / n0["N"]))
                            if n0["N"].size else None)
    # outcome
    last = out["counts"][-1] if out["counts"] else None
    d_end = float(d[-1])
    late = t0 >= t0[-1] - 100
    v_late = slope(t0[late], d[late])
    merged = bool((last and last["count_distinct"] == 1) or d_end < 2 * hw)
    if merged:
        label = "merged"
    elif d_end > d0 + 1.0 and v_late is not None and v_late > 0:
        label = "separated"
    else:
        label = "bounded"
    ds = d[s0:]
    pk_max, _ = find_peaks(ds, prominence=PROMINENCE)
    pk_min, _ = find_peaks(-ds, prominence=PROMINENCE)
    within = bool(ds.min() >= 2 * hw and ds.max() <= d0 + 1.0)
    bonded = bool(last is not None and last["count_distinct"] == 2 and pk_max.size >= 1 and pk_min.size >= 1 and within)
    shift_end = (d_s - d_end) / 2
    retained = None
    if last is not None and last["count_distinct"] == 1 and all(x["E_start"] is not None for x in out["lumps"]):
        retained = (tr[0]["E"][-1] + np.interp(tr[0]["tE"][-1], tr[1]["tE"], tr[1]["E"])) / \
                   (out["lumps"][0]["E_start"] + out["lumps"][1]["E_start"])
    pk_any, pr_any = find_peaks(ds, prominence=0.0)
    mn_any, pr_any2 = find_peaks(-ds, prominence=0.0)
    out["outcome"] = {"label": label, "bonded_C10": bonded,
                      "largest_prominence_max": float(pr_any["prominences"].max()) if pk_any.size else 0.0,
                      "largest_prominence_min": float(pr_any2["prominences"].max()) if mn_any.size else 0.0, "d_end": d_end, "d_min": float(ds.min()),
                      "d_max": float(ds.max()), "dd_dt_last100": v_late, "shift_end": float(shift_end),
                      "n_interior_max": int(pk_max.size), "n_interior_min": int(pk_min.size),
                      "within_2hw_d0p1": within, "last_count": last,
                      "merged_retained_E_over_E1_E2": ftrue(retained), "T_end": T_end}
    out["dtheta_end_deg"] = float(np.rad2deg(ph[-1])) if ph.size else None
    return out


def analyse_twin(out):
    rt, rE, rX = out["_raw_t"], out["_raw_E"], out["_raw_X"]
    if rt[-1] < 100 - 1e-9:
        out["twin"] = {"note": "record ends before t = 100"}
        return out
    i = int(np.argmin(np.abs(rt - 100.0)))
    dd = np.linalg.norm(rX[:, 1] - rX[:, 0], axis=1) if rX.shape[1] > 1 else None
    out["twin"] = {"t": float(rt[i]), "E_ball_ratio_at_100": (rE[i] / rE[0]).tolist(),
                   "d0": float(dd[0]) if dd is not None else None,
                   "d_change_0_to_100": float(dd[i] - dd[0]) if dd is not None else None}
    return out


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------

def row(rid, prediction, tolerance, items, notes=None):
    """items: list of dicts with 'hit' True/False/None (None = not evaluable, with 'why')."""
    ev = [it for it in items if it.get("hit") is not None]
    ne = [it for it in items if it.get("hit") is None]
    misses = [it for it in ev if not it["hit"]]
    if not items or not ev:
        verdict = "not evaluable"
    elif misses:
        verdict = "miss"
    elif ne:
        verdict = "not evaluable (evaluated items hit; missing: %s)" % ", ".join(str(it.get("what")) for it in ne)
    else:
        verdict = "hit"
    return {"id": rid, "prediction": prediction, "tolerance": tolerance, "verdict": verdict,
            "items": items, "misses": misses, "notes": notes or []}


def within(x, target, rel):
    return None if x is None else bool(abs(x / target - 1) <= rel)


def find(pairs, d, deg):
    for r in pairs:
        if r.get("d0") is not None and abs(r["d0"] - d) < 1e-6 and abs(r["dtheta_nominal_deg"] - deg) < 1e-6:
            return r
    return None


def design_ok(r, d):
    return r["t_reached"] >= DESIGN_T[d] - 1e-6


def kappa_fit(points):
    """points: (d, |F| / |cos|). Fit ln y = ln C - kappa d + ln(kappa/d + 1/d^2), C free."""
    d = np.array([p[0] for p in points])
    y = np.log(np.array([p[1] for p in points]))

    def cost(k):
        r = y + k * d - np.log(k / d + 1 / d ** 2)
        return float(np.sum((r - r.mean()) ** 2))
    ks = np.linspace(0.05, 1.5, 2901)
    k0 = ks[int(np.argmin([cost(k) for k in ks]))]
    res = minimize_scalar(cost, bounds=(max(0.05, k0 - 0.01), k0 + 0.01), method="bounded")
    k = float(res.x)
    r = y + k * d - np.log(k / d + 1 / d ** 2)
    return k, float(np.exp(r.mean())), float(np.sqrt(np.mean((r - r.mean()) ** 2)))


def criteria(runs, prof, test_mode):
    pairs = [r for r in runs if r["name"].startswith("pair_") and r["K"] == 2 and not r["linear_twin"] and "d0" in r]
    hw = prof["half_width"]
    rows = []
    D = (16, 20, 24)

    def T_ok(r, d):
        return test_mode or design_ok(r, d)

    def nev(what, why):
        return {"what": what, "hit": None, "why": why}

    # C1
    items = []
    for d in D:
        for deg, want in ((0, "attract"), (45, "attract"), (135, "repel"), (180, "repel")):
            r = find(pairs, d, deg)
            if r is None or "measure_ii" not in r:
                items.append(nev("d%d_%d" % (d, deg), "run missing"))
                continue
            t_lim = min(100.0, r["t_valid_end"])
            dd = float(np.interp(t_lim, r["sep_t"], r["sep"]) - r["d_start"])
            got = "attract" if dd < 0 else "repel"
            vi = [w for w in r["sliding"] if w["valid"]]
            items.append({"what": "d%d_%d" % (d, deg), "measured_delta_d_to_t": [dd, t_lim], "sign": got,
                          "want": want, "hit": got == want,
                          "measure_i_first_valid_F": vi[0]["F"] if vi else None,
                          "early_halfspace_F_d16": r.get("F_early_halfspace") if d == 16 else None})
    rows.append(row("C1", "dtheta = 0, 45 attract at first; 135, 180 repel; all d", "sign", items))
    # C2
    items = []
    for d in D:
        for deg, tol in ((0, 0.15), (180, 0.15), (45, 0.25), (135, 0.25)):
            r = find(pairs, d, deg)
            if r is None:
                items.append(nev("d%d_%d" % (d, deg), "run missing"))
                continue
            mi, mii = r.get("measure_i", {}), r.get("measure_ii", {})
            wi = mi.get("ratio_worst")
            it = {"what": "d%d_%d" % (d, deg), "tol": tol, "measure_i_worst_ratio": wi,
                  "measure_i_median_ratio": mi.get("ratio_median"), "measure_i_n_valid": mi.get("n_valid_windows"),
                  "measure_i_n_excluded_small_pred": mi.get("n_excluded_small_pred"),
                  "measure_ii_ratio": mii.get("ratio"),
                  "reported_not_judged_measure_i_worst_ratio_windows_d_ge_12": mi.get("ratio_worst_d_ge_12")}
            hits = []
            if wi is None:
                it["why_i"] = "no valid t >= 100 window with a defined ratio"
            else:
                hits.append(abs(wi - 1) <= tol)
            if mii.get("ratio") is None:
                it["why_ii"] = "model displacement zero or no span"
            else:
                hits.append(abs(mii["ratio"] - 1) <= tol)
            if d == 16:
                fe = r.get("F_early_ratio")
                it["early_halfspace_ratio"] = fe
                if fe is None:
                    it["why_early"] = "no early window"
                else:
                    hits.append(abs(fe - 1) <= 0.20)
            need = 3 if d == 16 else 2
            it["hit"] = (all(hits) if len(hits) == need else (False if (hits and not all(hits)) else None))
            if it["hit"] is None:
                it["why"] = "not every measure evaluable"
            items.append(it)
    rows.append(row("C2", "measured force over predicted = 1 (measures i and ii; early half-space at d = 16)",
                    "15% at 0/180; 25% at 45/135; early half-space 20% (d = 16 only)", items))
    # C3
    pts, per_d = [], {}
    for r in pairs:
        if r["dtheta_nominal_deg"] in (0.0, 180.0):
            for w in r.get("sliding", []):
                if w["valid"] and w["F"] is not None and abs(np.cos(np.deg2rad(w["dtheta_deg"]))) > 0.5:
                    pts.append((w["d"], abs(w["F"]) / abs(np.cos(np.deg2rad(w["dtheta_deg"])))))
                    per_d[r["d0"]] = per_d.get(r["d0"], 0) + 1
    items = []
    if len(per_d) >= 2:
        k, C, rms = kappa_fit(pts)
        items.append({"what": "kappa_measure_i", "kappa": k, "amplitude_C": C, "log_rms": rms,
                      "windows_per_d": {str(a): b for a, b in per_d.items()}, "target": REC["C3_kappa"],
                      "hit": abs(k - REC["C3_kappa"]) <= 0.03})
    else:
        items.append(nev("kappa_measure_i", "valid t >= 100 windows at fewer than 2 separations: %s" % per_d))
    q = [(r["quadratic_early"]["d_mean"], abs(r["quadratic_early"]["F_from_d"]) /
          max(abs(np.cos(np.deg2rad(r["quadratic_early"]["dtheta_mean_deg"]))), 1e-12))
         for r in pairs if r["dtheta_nominal_deg"] in (0.0, 180.0) and "quadratic_early" in r]
    ind = None
    if len({round(x[0]) for x in q}) >= 2:
        k2, C2_, rms2 = kappa_fit(q)
        ind = {"kappa": k2, "amplitude_C": C2_, "log_rms": rms2, "n_runs": len(q),
               "note": "indicative only: quadratic fit of d(t) over the early span assumes a constant force"}
    rows.append(row("C3", "kappa = 0.416 from dtheta = 0 and 180 across d = 16, 20, 24 (exact 1/d^2 form)",
                    "0.03", items, notes=[{"kappa_from_measure_ii_indicative": ind}]))
    # C4
    items = []
    for d in D:
        xs, ys, nr = [], [], 0
        for r in pairs:
            if abs(r["d0"] - d) > 1e-6:
                continue
            used = False
            for w in r.get("sliding", []):
                if w["valid"] and w["F"] is not None and w["F0"] != 0:
                    xs.append(np.cos(np.deg2rad(w["dtheta_deg"])))
                    ys.append(w["F"] / w["F0"])
                    used = True
            nr += used
        if nr >= 2 and len(xs) > 3:
            sl_, ic = np.polyfit(xs, ys, 1)
            items.append({"what": "d%d" % d, "slope": float(sl_), "intercept": float(ic), "n_windows": len(xs),
                          "n_runs": nr, "hit": bool(abs(sl_ - 1) <= 0.15 and abs(ic) <= 0.1)})
        else:
            items.append(nev("d%d" % d, "valid windows from fewer than 2 runs"))
    rows.append(row("C4", "F/F0 against cos(measured dtheta): slope 1, intercept 0", "slope 0.15; intercept 0.1", items))
    # C5a
    items = []
    for d in D:
        r = find(pairs, d, 90)
        mi = r.get("measure_i", {}) if r else {}
        wv = mi.get("abs_resid_over_F0_worst")
        items.append({"what": "d%d" % d, "worst_abs_F_minus_F0cos_over_F0": wv, "n_valid": mi.get("n_valid_windows"),
                      "hit": None if wv is None else bool(wv <= 0.15),
                      **({"why": "no valid t >= 100 window" if r else "run missing"} if wv is None else {})})
    rows.append(row("C5a", "dtheta = 90: force on t >= 100 windows equals F0 cos(measured dtheta)", "0.15 F0(d)", items))
    c5a = rows[-1]
    # C5b
    items = []
    for d in D:
        r = find(pairs, d, 90)
        nf = r.get("N_flow_early") if r else None
        if nf is None:
            items.append(nev("d%d" % d, "run missing or no early N window"))
            continue
        rate = abs(nf["antisymmetric_rate"])
        items.append({"what": "d%d" % d, "antisymmetric_rate": nf["antisymmetric_rate"], "dN0_dt": nf["dN0_dt"],
                      "dN1_dt": nf["dN1_dt"], "leader": nf["leader"], "flow": nf["flow"],
                      "leader_to_lagger": nf["leader_to_lagger"], "on_record_rate": REC["C5b_rate"][d],
                      "rate_over_record": rate / REC["C5b_rate"][d],
                      "pred_with_measured_inputs": nf["pred_measured_inputs_C_N_sin"],
                      "reported_not_judged_singles_subtracted": r.get("N_flow_early_singles_subtracted"),
                      "hit": bool(nf["leader_to_lagger"] and abs(rate / REC["C5b_rate"][d] - 1) <= 0.30)})
    rows.append(row("C5b", "dtheta = 90: N flows leader -> lagger at 1.61, 0.244, 0.038", "direction; rate 30% (early window)", items))
    c5b = rows[-1]
    # C5c
    items = []
    for d in D:
        r = find(pairs, d, 90)
        if r is None or r.get("dtheta_end_deg") is None:
            items.append(nev("d%d" % d, "run missing"))
            continue
        if not T_ok(r, d):
            items.append(nev("d%d" % d, "record ends at %.0f < design T %.0f" % (r["t_reached"], DESIGN_T[d])))
            continue
        e = r["dtheta_end_deg"]
        own = r["model_own_phase"]["dtheta_end_deg"]
        if d == 16:
            items.append({"what": "d16", "dtheta_end_deg": e, "prediction": "> 180", "own_model_end": own, "hit": bool(e > 180)})
        else:
            items.append({"what": "d%d" % d, "dtheta_end_deg": e, "prediction": REC["C5c_end_deg"][d],
                          "own_model_end": own, "hit": bool(abs(e - REC["C5c_end_deg"][d]) <= 30)})
    rows.append(row("C5c", "dtheta = 90 runs toward 180: 160 (d 20), 102 (d 24), beyond 180 (d 16)", "30 deg at d = 20, 24", items))
    # C5d
    items = []
    for d in D:
        r = find(pairs, d, 90)
        if r is None or "outcome" not in r:
            items.append(nev("d%d" % d, "run missing"))
            continue
        if not T_ok(r, d):
            items.append(nev("d%d" % d, "record ends before design T"))
            continue
        s = r["outcome"]["shift_end"]
        it = {"what": "d%d" % d, "shift_end": s, "prediction": REC["C5d_shift"][d]}
        if d == 16:
            it["hit"] = bool(s < 0)
        elif d == 20:
            it["hit"] = bool(s < 0 and abs(s / REC["C5d_shift"][20] - 1) <= 0.5)
        else:
            it["hit"] = None
            it["why"] = "no tolerance given at d = 24 (reported only)"
        items.append(it)
    r5d = row("C5d", "dtheta = 90 ends repelling: shift -2.8, -0.89, -0.03", "sign at d 16, 20; 50% at d 20",
              [it for it in items if not it.get("why", "").startswith("no tolerance")],
              notes=[it for it in items if it.get("why", "").startswith("no tolerance")])
    rows.append(r5d)
    # C5-CD
    v = "hit" if (c5a["verdict"] == "hit" and c5b["verdict"] == "hit") else (
        "miss" if "miss" in (c5a["verdict"], c5b["verdict"]) else "not evaluable")
    rows.append({"id": "C5-CD", "prediction": "plan: leading force vanishes at 90 and N flows", "tolerance": "as C5a and C5b",
                 "verdict": v, "items": [{"C5a": c5a["verdict"], "C5b": c5b["verdict"]}], "misses": [], "notes": []})
    # C6
    items = []
    for d in D:
        r = find(pairs, d, 45)
        if r is None or "outcome" not in r:
            items.append(nev("d%d" % d, "run missing"))
            continue
        if not T_ok(r, d):
            items.append(nev("d%d" % d, "record ends before design T"))
            continue
        o = r["outcome"]
        if d == 16:
            items.append({"what": "d16", "label": o["label"], "dd_dt_last100": o["dd_dt_last100"],
                          "hit": bool(o["label"] != "merged" and (o["dd_dt_last100"] or 0) > 0)})
        elif d == 20:
            items.append({"what": "d20", "label": o["label"], "d_end": o["d_end"], "prediction": "turning, d about 17",
                          "hit": bool(o["label"] == "bounded")})
        else:
            items.append({"what": "d24", "shift_end": o["shift_end"], "prediction": REC["C6_shift_d24"],
                          "hit": within(o["shift_end"], REC["C6_shift_d24"], 0.30)})
    rows.append(row("C6", "dtheta = 45: d16 not merged and separating; d20 turning (bounded); d24 shift 0.59",
                    "outcome at d 16, 20; 30% at d 24", items))
    # C7
    items = []
    for d in D:
        r = find(pairs, d, 0)
        if r is None or "outcome" not in r:
            items.append(nev("d%d" % d, "run missing"))
            continue
        o = r["outcome"]
        if d in (16, 20):
            tc = r["t_contact_d_le_2hw"]
            if tc is None and not T_ok(r, d):
                items.append(nev("d%d" % d, "no contact before the record ends at %.0f" % r["t_reached"]))
                continue
            items.append({"what": "d%d_contact" % d, "contact_t": tc, "first_count_distinct_1": r["t_first_count_distinct_1"],
                          "label": o["label"], "prediction": REC["C7_contact"][d],
                          "hit": bool(tc is not None and o["label"] == "merged" and abs(tc / REC["C7_contact"][d] - 1) <= 0.25)})
            rt = o["merged_retained_E_over_E1_E2"]
            if rt is not None:
                lo, hi = REC["C7_retained"]
                items.append({"what": "d%d_retained" % d, "retained": rt, "range": [lo, hi], "hit": bool(lo <= rt <= hi)})
        else:
            if not T_ok(r, d):
                items.append(nev("d24_shift", "record ends before design T"))
                continue
            items.append({"what": "d24_shift", "shift_end": o["shift_end"], "prediction": REC["C7_shift_d24"],
                          "hit": within(o["shift_end"], REC["C7_shift_d24"], 0.25)})
    rows.append(row("C7", "dtheta = 0: merged at d 16 (contact ~126), d 20 (~346); d 24 shift 0.90; retained 0.6-0.9",
                    "contact 25%; shift 25%", items))
    # C8, C9
    for rid, deg, tol, key in (("C8", 180, 0.25, "C8_shift"), ("C9", 135, 0.30, "C9_shift")):
        items = []
        for d in D:
            r = find(pairs, d, deg)
            if r is None or "outcome" not in r:
                items.append(nev("d%d" % d, "run missing"))
                continue
            if not T_ok(r, d):
                items.append(nev("d%d" % d, "record ends before design T"))
                continue
            o = r["outcome"]
            it = {"what": "d%d" % d, "label": o["label"], "shift_end": o["shift_end"], "prediction": REC[key][d]}
            ok = o["label"] == "separated" and within(o["shift_end"], REC[key][d], tol)
            if rid == "C8":
                it["N_sym_max_rel"] = r["N_sym_max_rel"]
                ok = ok and r["N_sym_max_rel"] is not None and r["N_sym_max_rel"] <= N_SYM_TOL
            else:
                s, e = r["dtheta_start_deg"], r["dtheta_end_deg"]
                it["dtheta_start_deg"], it["dtheta_end_deg"] = s, e
                ok = ok and abs(e - 180) < abs(s - 180)
            it["hit"] = bool(ok)
            items.append(it)
        rows.append(row(rid, ("dtheta = 180: separated, shifts -5.7/-3.2/-0.71, N equal" if rid == "C8" else
                              "dtheta = 135: separated, phase toward 180, shifts -4.9/-2.6/-0.52"),
                        "shift %d%%%s" % (int(tol * 100), "; N difference at rounding" if rid == "C8" else ""), items))
    # C10
    items = []
    for r in pairs:
        if "outcome" in r:
            o = r["outcome"]
            items.append({"what": r["name"], "bonded": o["bonded_C10"], "label": o["label"],
                          "n_interior_min": o["n_interior_min"], "n_interior_max": o["n_interior_max"],
                          "within_2hw_d0p1": o["within_2hw_d0p1"], "hit": not o["bonded_C10"]})
    rows.append(row("C10", "no pair is bonded", "—", items or [nev("all", "no pair runs")]))
    # C11
    items = []
    for r in runs:
        if r["linear_twin"]:
            tw = r.get("twin", {})
            if "E_ball_ratio_at_100" not in tw:
                items.append(nev(r["name"], tw.get("note", "no data")))
                continue
            att = tw["d_change_0_to_100"] is not None and tw["d_change_0_to_100"] < -TWIN_ATTRACT
            items.append({"what": r["name"], "E_ball_ratio_at_100": tw["E_ball_ratio_at_100"],
                          "d_change_0_to_100": tw["d_change_0_to_100"],
                          "hit": bool(max(tw["E_ball_ratio_at_100"]) < 0.05 and not att)})
    rows.append(row("C11", "linear twins: ball energy < 0.05 of start by T = 100; no centroid attraction", "—",
                    items or [nev("twins", "no twin runs")]))
    # C12
    items = []
    byname = {r["name"]: r for r in runs}
    for chk, part, tol in (("check_res_d20_p000_N224", "pair_d20_p000", 0.1), ("check_box_d24_p000_L132", "pair_d24_p000", 0.02)):
        a, b = byname.get(chk), byname.get(part)
        if a is None or b is None or "sep" not in a or "sep" not in b:
            items.append(nev(chk, "check or partner missing"))
            continue
        ta, da = np.array(a["sep_t"]), np.array(a["sep"])
        tb, db = np.array(b["sep_t"]), np.array(b["sep"])
        lo, hi = max(ta[0], tb[0], T_START), min(ta[-1], tb[-1])
        m = (ta >= lo) & (ta <= hi)
        diff = np.abs(da[m] - np.interp(ta[m], tb, db))
        items.append({"what": chk, "partner": part, "span": [float(lo), float(hi)], "max_abs_d_diff": float(diff.max()),
                      "tol": tol, "hit": bool(diff.max() <= tol)})
    rows.append(row("C12", "resolution: d(t) within 0.1; box: within 0.02", "targets", items))
    # C13
    items = []
    for r in pairs:
        scale = prof["E"] * r["d0"]
        Lm = r["angular_momentum_maxabs"]
        items.append({"what": r["name"], "L_maxabs": Lm, "axes": r["angular_momentum_axes"],
                      "L_over_E_d0": [x / scale for x in Lm], "momentum_extremes": r["momentum_extremes"],
                      "hit": bool(Lm and max(Lm) / scale <= L_REL_TOL)})
    rows.append(row("C13", "angular momentum about z and x stays zero to noise", "evaluator: |L|/(E d0) <= 1e-8",
                    items or [nev("all", "no pair runs")]))
    return rows


# ---------------------------------------------------------------------------
# plots and main
# ---------------------------------------------------------------------------

def measure_iii(runs):
    """d = 24 early slope with the single-lump contributions subtracted (Block 2a)."""
    byname = {r["name"]: r for r in runs}
    for r in runs:
        if r["K"] != 2 or r["linear_twin"] or abs(r.get("d0", 0) - 24.0) > 1e-6 or "_Ph" not in r:
            continue
        th1, th2 = r["phases"]
        key = lambda th: "single_x12_t%04.1f" % abs(np.rad2deg(th) if abs(th - np.pi) > 1e-9 else 0.0)
        s2, s1 = byname.get(key(th2)), byname.get(key(th1))
        if s2 is None or s1 is None:
            r["measure_iii"] = {"note": "single-lump runs %s / %s missing" % (key(th2), key(th1))}
            continue
        tp, Pp = r["_Ph"]
        m = (tp >= EARLY_F[0]) & (tp <= EARLY_F[1])
        P2 = np.interp(tp, *s2["_Ph"])                       # lump 2 = the +12 single itself
        P1 = np.interp(tp, *s1["_Ptot"]) - np.interp(tp, *s1["_Ph"])   # mirror of the +12 single: -P(x>0)
        sub = Pp - P2 + P1
        Fs = slope(tp[m], sub[m]) if m.sum() > 3 else None
        r["measure_iii"] = {"F_early_singles_subtracted": Fs, "F_early_raw": r.get("F_early_halfspace"),
                            "F_early_pred_mean": r.get("F_early_pred_mean"),
                            "ratio": (Fs / r["F_early_pred_mean"]) if (Fs is not None and r.get("F_early_pred_mean")) else None,
                            "singles": [s1["name"], s2["name"]],
                            "exact_stand_in": bool(abs(abs(th1) - abs(th2)) < 1e-12 and (th1 == th2 or abs(th1) < 1e-12)),
                            "note": "lump 1 (theta_1 = %.1f deg) uses the +theta single as stand-in unless dtheta = 0 or 180; "
                                    "coherent radiation of the pair is not removed" % np.rad2deg(th1)}


def nflow_singles(runs):
    """Early antisymmetric N rate with each lump's own start-up N drift removed, where a single-lump run
    with that lump's phase exists (single_x12_t<theta> for theta >= 0, single_x12_tm<theta> for theta < 0;
    theta = 0 and 180 use single_x12_t00.0). Reported, not judged."""
    byname = {r["name"]: r for r in runs}

    def single_for(th):
        deg = np.rad2deg(th)
        if abs(deg) < 1e-9 or abs(abs(deg) - 180) < 1e-9:
            return byname.get("single_x12_t00.0")
        return byname.get(("single_x12_t%04.1f" if deg > 0 else "single_x12_tm%04.1f") % abs(deg))
    for r in runs:
        if not r["name"].startswith("pair_") or "N_flow_early" not in r:
            continue
        ss = [single_for(th) for th in r["phases"]]
        if any(x is None for x in ss):
            r["N_flow_early_singles_subtracted"] = {"note": "single-lump run missing for phase(s) %s deg" % [
                float(np.rad2deg(th)) for th, x in zip(r["phases"], ss) if x is None]}
            continue
        tr = r["_tr"]
        tN = tr[0]["tN"]
        me = (tN >= EARLY_N[0]) & (tN <= EARLY_N[1])
        c = []
        for k in (0, 1):
            Nk = np.interp(tN, tr[k]["tN"], tr[k]["N"])
            st = ss[k]["_tr"][0]
            c.append(Nk - np.interp(tN, st["tN"], st["N"]))
        a = 0.5 * (slope(tN[me], c[0][me]) - slope(tN[me], c[1][me]))
        r["N_flow_early_singles_subtracted"] = {"antisymmetric_rate": a, "singles": [x["name"] for x in ss],
                                                "pred": r["N_flow_early"]["pred_measured_inputs_C_N_sin"]}


def plot_run(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for k, tr in enumerate(r["_tr"]):
        ax[0, 0].plot(tr["t"], tr["X"][:, 0], label="lump %d x" % k)
        ax[1, 0].plot(tr["tN"], tr["N"], label="lump %d N" % k)
    ax[0, 0].set_ylabel("centroid x")
    ax[1, 0].set_ylabel("N in ball")
    if "sep" in r:
        ax[0, 1].plot(r["sep_t"], r["sep"], "k", label="d measured")
        ax[0, 1].plot(r["measure_ii"]["d_model_t"], r["measure_ii"]["d_model"], "--", label="model, measured dtheta")
        ax[0, 1].plot(r["model_own_phase"]["d_t"], r["model_own_phase"]["d"], ":", label="model, own phase")
        ax[0, 1].set_ylabel("d(t)")
        ax[0, 1].set_title("%s: %s" % (r["name"], r["outcome"]["label"]), fontsize=9)
        vw = [w for w in r["sliding"]]
        if vw:
            ax[1, 1].plot([w["t_c"] for w in vw], [w["F"] for w in vw], label="F sliding (P x<0)")
            ax[1, 1].plot([w["t_c"] for w in vw], [w["F_pred"] for w in vw], "--", label="F_pred(d, dtheta, omega)")
            ax[1, 1].axvline(T_VALID, color="gray", lw=0.6)
        ax[1, 1].set_ylabel("force")
    for a in ax.ravel():
        a.set_xlabel("t")
        a.legend(frameon=False, fontsize=7)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "analysis"))
    ap.add_argument("--test-record-end-as-T", action="store_true",
                    help="pipeline testing only: treat the end of a short record as the design T")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prof, fam = load_constants()
    runs = []
    for f in sorted(glob.glob(os.path.join(a.results, "*.json"))):
        name = os.path.basename(f)[:-5]
        res = json.load(open(f))
        if "lumps" not in res:
            continue
        r = run_basics(name, res, prof)
        if r["K"] == 1:
            analyse_single(r)
        elif r["linear_twin"]:
            analyse_twin(r)          # no force analysis: a dispersing twin has omega > m (kappa undefined)
        else:
            analyse_pair(r, res, prof, fam)
        runs.append(r)
        plot_run(r, os.path.join(a.out, name + ".png"))
    measure_iii(runs)
    nflow_singles(runs)
    rows = criteria(runs, prof, a.test_record_end_as_T)
    rows[1]["notes"].append({"measure_iii_d24": {r["name"]: r.get("measure_iii") for r in runs
                                                 if r["K"] == 2 and abs(r.get("d0", 0) - 24) < 1e-6 and not r["linear_twin"]}})
    crit = {"source": "SESSION_02_PREDICTIONS.md Block 2 rows C1-C13 with Block 2a", "results_dir": a.results,
            "test_mode_record_end_as_T": a.test_record_end_as_T, "evaluator_choices": EVALUATOR_CHOICES,
            "runs_present": [r["name"] for r in runs],
            "summary": {x["id"]: x["verdict"] for x in rows},
            "all_misses": [{"row": x["id"], **m} for x in rows for m in x["misses"]],
            "rows": rows}
    gate = os.path.join(a.results, "gate.json")
    if os.path.exists(gate):
        g = json.load(open(gate))
        crit["gate_from_gate_eval"] = {"G1": g.get("G1"), "G2_pass": g.get("G2", {}).get("pass"), "gate_pass": g.get("gate_pass")}
    for r in runs:
        for k in [k for k in r if k.startswith("_")]:
            del r[k]
    for path, obj in ((os.path.join(a.out, "criteria.json"), crit),
                      (os.path.join(a.out, "summary.json"), {"description": __doc__, "runs": runs})):
        with open(path + ".tmp", "w") as fh:
            json.dump(obj, fh, indent=1, default=float)
        os.replace(path + ".tmp", path)
    print(json.dumps(crit["summary"], indent=1))
    print("misses:", len(crit["all_misses"]))


if __name__ == "__main__":
    main()
