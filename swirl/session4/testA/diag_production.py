"""Diagnostics of the two Test A production misses, from the stored results only (no PDE runs).

Miss 1 (A-P2, primary far tail): F/F_law leaves 1 +- 0.1 for r_eff >= 30 with a residual F - F_static
proportional to cos(dtheta).
  D1  model comparison of the residual R = F - F_static (static force at r_eff, analysis.json) over r_eff 22..40
      and 26..40, both channels, s_c = cos(dtheta): (M0) zero; (M1) a constant per box, s_c c_box; (M2) the 3 omega
      interference form s_c B sin(k3 r + delta)/r, k3 = sqrt(9 omega^2 - 1); (M2p) the same with r^-p, p free;
      (M3) zero with an inflated sigma (s sigma); (M4) the parameter-free 3 omega prediction of D3. Gaussian
      -2 ln L (sigma terms kept, so M3 is comparable), AIC = -2 ln L + 2 p. A scan of chi^2 over k (M2 form) shows
      where k3 sits among the aliases of the r grid. The box-dependence check reads the stored 1.5x box run
      (controls.box, r = 24).
  D2  estimator and window: F from the ball momenta, F_P = d/dt (P_1x - P_2x)/2 (Hann-smoothed over 3 periods,
      linear fit, the analysis' n_eff rule), against the centroid F of analysis.json; then F_P on the first and
      second halves of [t1, t2] and on [120, t2] (r >= 26). A t = 0 transient fades between the halves; a steady
      radiation or box term does not.
  D3  the 3 omega-mediated force, first order in the partner's 3 omega wave psi_2 (collective coordinate
      dP_1/dt = int psi_2 d_x U'_NL(phi_1)): with s3 = (g4/24) f^3 - (g6/384) f^5 (the 3 omega part of -U'_NL) and
      Q = int s3 e^{-i k3 x} d^3x (so r a3 = |Q|/(4 pi) in the Born approximation),
          F_3(r, dtheta) = (k3 Q^2 / (8 pi r)) sin(k3 r - 3 dtheta)   (F > 0 attracts).
      Q from the stored profile (Born) and from Test B's measured r a3 = 0.0167.

Miss 2 (A-P6, unequal pair): F_amp / F_amp_static = 0.62 at r 18 falling to 0.42 to 0.44 beyond r 26.
  E1  normalisation: the static amplitude is split into the per-lump first-order forces from the stored moments,
      U_1 = -(1/2) int S_1 f_2 (force on lump 1 in lump 2's field, decays as e^{-kappa_2 r}) and
      U_2 = -(1/2) int S_2 f_1 (decays as e^{-kappa_1 r}); static.U_amp = w_1 U_1 + w_2 U_2 with
      w_i = omega_i/(omega_1 + omega_2). The relative acceleration measures instead
      F_dyn = (F_1/E_1 + F_2/E_2)/(1/E_1 + 1/E_2), weights a_i = (1/E_i)/(1/E_1 + 1/E_2).
  E2  per-lump forces, two estimators: ball momenta P_ix (fit p0 + p1 t + c G1, G1' = cos dphi) and centroids
      x_i (c0 + c1 t + c2 t^2 + c3 G, G'' = cos dphi), each against its static F_i; and F_amp rebuilt from the
      momentum route. Separates "the measured coordinate carries a beat-frequency dressing" from the law.
  E3  phase: d(t) refitted with G_cos, G_sin (G'' = sin dphi) and G_cos2 (G'' = cos 2 dphi); amplitude, lag,
      cos 2 dphi coefficient.
  Also: k side by side (log-linear e^{-kr}/r fits, same weights) and the drift of ln(ratio) with r beyond 26.

Run: python -m swirl.session4.testA.diag_production [results_dir]  (writes outputs/session4/testA/diag_production.json)
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

from swirl.session4.testA import analysis as AN
from swirl.session4.testA.driver import TYPES
from swirl.session4.testA.static import G4

ROOT = AN.ROOT
DATA_DEFAULT = "/Users/aaronramos/Desktop/SWIRL/.claude/worktrees/s4-testA/outputs/session4/testA"
OUT = os.path.join(ROOT, "outputs/session4/testA/diag_production.json")
RA3_TESTB = 0.0167
G6 = 0.12


# ------------------------------------------------------------------------------------------- generic fits
def lsq_neff(t, y, t1, t2, period, cols):
    """Weighted-equal least squares of y on [t1, t2] with basis columns cols(tc) (list of arrays on the full t);
    s.e. with the analysis' n_eff rule (min(window/period, AR(1) count), >= 3)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    s = (t >= t1) & (t <= t2)
    if s.sum() < len(cols) + 3:
        return None, None
    A = np.stack([c[s] for c in cols], axis=1)
    c, *_ = np.linalg.lstsq(A, y[s], rcond=None)
    res = y[s] - A @ c
    n = int(s.sum())
    rho = float(np.corrcoef(res[1:], res[:-1])[0, 1]) if n > 3 and np.std(res) > 0 else 0.0
    n_ar = n * (1 - rho) / (1 + rho) if rho < 0.999 else 1.0
    n_eff = max(min((t2 - t1) / period, n_ar), 3.0)
    s2 = float(np.sum(res ** 2) / max(n - A.shape[1], 1)) * n / n_eff
    se = np.sqrt(np.abs(np.diag(s2 * np.linalg.inv(A.T @ A))))
    return c, se


def cumtrap(y, dt):
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * dt)])


def radial_ft(f, r, k):
    """int f(|x|) e^{-i k x} d^3x = int_0^inf f(r) 4 pi r^2 sin(k r)/(k r) dr (trapezoid on the given grid)."""
    return float(np.trapezoid(f * 4 * np.pi * r * np.sin(k * r) / k, r))


def gauss_nll(res, sig):
    """-2 ln L for independent Gaussian residuals (constant 2 pi kept)."""
    return float(np.sum((res / sig) ** 2) + np.sum(np.log(2 * np.pi * sig ** 2)))


def wls(A, y, sig):
    Aw, yw = A / sig[:, None], y / sig
    c, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    cov = np.linalg.inv(Aw.T @ Aw)
    return c, np.sqrt(np.diag(cov)), float(np.sum((yw - Aw @ c) ** 2))


def loglin_k(r, F, sF):
    """k of |F| ~ C e^{-k r}/r by weighted least squares on ln(|F| r) (sigma_ln = sF/|F|)."""
    r, F, sF = (np.asarray(x, float) for x in (r, F, sF))
    c, se, chi2 = wls(np.stack([np.ones_like(r), -r], axis=1), np.log(np.abs(F) * r), sF / np.abs(F))
    return dict(k=float(c[1]), k_se=float(se[1]), chi2=chi2, n=int(r.size))


# ------------------------------------------------------------------------------------------- Miss 1
def k3_of(omega):
    return float(np.sqrt(9 * omega ** 2 - 1))


def born_Q(omega=TYPES["primary"]["omega"]):
    """Q = int s3 e^{-i k3 x} d^3x of the primary profile (stored JSON spline + exact tail), s3 the 3 omega part
    of -U'_NL(f cos)."""
    from swirl.session4.testA.static import Lump
    l = Lump(TYPES["primary"]["beta"], omega, source="json")
    r = np.linspace(1e-6, 40.0, 80001)
    f, _ = l.ff(r)
    s3 = G4 / 24.0 * f ** 3 - G6 / 384.0 * f ** 5
    return radial_ft(s3, r, k3_of(omega)), l.info()


def F3_pred(r, dtheta_deg, omega, Q):
    k3 = k3_of(omega)
    return k3 * Q * Q / (8 * np.pi * r) * np.sin(k3 * np.asarray(r) - 3 * np.deg2rad(dtheta_deg))


def miss1_rows(a, results):
    """Per-row record (both channels): r_eff, F (centroid), F_static, residuals, box, omega, D2 momentum route."""
    out = []
    ch = a["primary"]["channels"]
    for lab in ("p000", "p180"):
        c = ch[lab]
        stat = dict(zip(c["F"]["names"], c["F_static_at_r_eff"]))
        for q in c["rows"]:
            if q["force_excluded"] or q["r"] < 22 - 1e-9:
                continue
            res = AN.load(q["name"], results)
            om = q["pair"]["omega"]
            P = 2 * np.pi / om
            s = res["lumps"]["raw"]["series"]
            t = np.array(s["t"])
            Px = np.array(s["P"])[:, :, 0]
            ts, ys = AN.hann_smooth(t, 0.5 * (Px[:, 0] - Px[:, 1]), 3 * P)
            ts2, ysum = AN.hann_smooth(t, 0.5 * (Px[:, 0] + Px[:, 1]), 3 * P)
            t1, t2 = q["window"]["t1"], q["window"]["t2"]

            def slope(a_, b_, y=ys):
                tc = ts - 0.5 * (a_ + b_)
                cc, se = lsq_neff(ts, y, a_, b_, P, [np.ones_like(tc), tc])
                return (None, None) if cc is None else (float(cc[1]), float(se[1]))
            # reproduce the stored centroid F (sanity)
            w = AN.window_for(res, q["r"], a["primary"]["numbers"]["kappa"], q["single"]["omega"])
            cA, seA, _ = AN.accel_fit(w["t_d"], w["d"], t1, t2, P)
            F_c = -0.5 * q["E_inertia"] * 2 * float(cA[2])
            FP, sFP = slope(t1, t2)
            tm = 0.5 * (t1 + t2)
            rec = dict(name=q["name"], dtheta=q["dtheta"], r=q["r"], r_eff=q["r_eff"], box=res["job"]["box"],
                       L=res["params"]["L"], N=res["params"]["N"], omega_pair=om, t1=t1, t2=t2,
                       F=q["F"], sigma_F=q["sigma_F"], F_recomputed=F_c, F_static=stat[q["name"]], F_P=FP,
                       F_P_se=sFP, P_sum_slope=slope(t1, t2, ysum)[0], omega_single=q["single"]["omega"],
                       kappa_pair=float(np.sqrt(1 - om * om)))
            n = a["primary"]["numbers"]
            rec["F_law"] = float(np.cos(np.deg2rad(q["dtheta"])) * 2 * np.pi * n["A"] ** 2 * np.exp(-n["kappa"] * q["r_eff"])
                                 * (n["kappa"] / q["r_eff"] + 1 / q["r_eff"] ** 2))
            rec["F_P_sigma"] = float(np.hypot(sFP, q["F_floor_control"]))
            if q["r"] >= 26 - 1e-9:
                rec["F_P_half1"], rec["F_P_half1_se"] = slope(t1, tm)
                rec["F_P_half2"], rec["F_P_half2_se"] = slope(tm, t2)
                rec["F_P_late120"], rec["F_P_late120_se"] = slope(120.0, t2)
            out.append(rec)
    return out


def miss1_models(rows, rmin, Q_list):
    sel = [q for q in rows if q["r"] >= rmin - 1e-9]
    r = np.array([q["r_eff"] for q in sel])
    sc = np.array([np.cos(np.deg2rad(q["dtheta"])) for q in sel])
    R = np.array([q["F"] - q["F_static"] for q in sel])
    sig = np.array([q["sigma_F"] for q in sel])
    om = np.array([q["omega_pair"] for q in sel])
    k3 = np.sqrt(9 * om ** 2 - 1)
    n = r.size
    M = {}
    M["M0_zero"] = dict(p=0, nll=gauss_nll(R, sig), chi2=float(np.sum((R / sig) ** 2)))
    boxes = sorted(set(q["box"] for q in sel))
    A1 = np.stack([sc * np.array([q["box"] == b for q in sel], float) for b in boxes], axis=1)
    c, se, chi2 = wls(A1, R, sig)
    M["M1_const_per_box"] = dict(p=len(boxes), nll=chi2 + float(np.sum(np.log(2 * np.pi * sig ** 2))), chi2=chi2,
                                 boxes=boxes, c=c.tolist(), se=se.tolist())
    A2 = np.stack([sc * np.sin(k3 * r) / r, sc * np.cos(k3 * r) / r], axis=1)
    c, se, chi2 = wls(A2, R, sig)
    B, dl = float(np.hypot(*c)), float(np.arctan2(c[1], c[0]))
    M["M2_3omega_r1"] = dict(p=2, nll=chi2 + float(np.sum(np.log(2 * np.pi * sig ** 2))), chi2=chi2, B=B, delta=dl,
                             B_se=float(np.hypot(c[0] * se[0], c[1] * se[1]) / B),
                             delta_se=float(np.hypot(c[0] * se[1], c[1] * se[0]) / B ** 2))
    best = None
    for p in np.linspace(0.0, 4.0, 81):
        Ap = np.stack([sc * np.sin(k3 * r) / r ** p, sc * np.cos(k3 * r) / r ** p], axis=1)
        cp, _, chp = wls(Ap, R, sig)
        if best is None or chp < best[1]:
            best = (p, chp, cp)
    M["M2p_3omega_pfree"] = dict(p=3, nll=best[1] + float(np.sum(np.log(2 * np.pi * sig ** 2))), chi2=best[1],
                                 p_best=float(best[0]), B=float(np.hypot(*best[2])),
                                 delta=float(np.arctan2(best[2][1], best[2][0])))
    # baseline scale (the run's lumps sit at omega_pair, not the profile omega of F_static): alone and with M2
    Fs = np.array([q["F_static"] for q in sel])
    c, se, chi2 = wls(Fs[:, None], R, sig)
    M["M5_baseline_scale"] = dict(p=1, nll=chi2 + float(np.sum(np.log(2 * np.pi * sig ** 2))), chi2=chi2,
                                  scale=float(c[0]), scale_se=float(se[0]))
    c, se, chi2 = wls(np.concatenate([Fs[:, None], A2], axis=1), R, sig)
    M["M6_baseline_scale_plus_3omega"] = dict(
        p=3, nll=chi2 + float(np.sum(np.log(2 * np.pi * sig ** 2))), chi2=chi2, scale=float(c[0]),
        scale_se=float(se[0]), B=float(np.hypot(c[1], c[2])), delta=float(np.arctan2(c[2], c[1])),
        B_se=float(np.hypot(c[1] * se[1], c[2] * se[2]) / np.hypot(c[1], c[2])),
        delta_se=float(np.hypot(c[1] * se[2], c[2] * se[1]) / (c[1] ** 2 + c[2] ** 2)),
        residual_after=(R - np.concatenate([Fs[:, None], A2], axis=1) @ c).tolist())
    s_inf = float(np.sqrt(np.mean((R / sig) ** 2)))
    M["M3_zero_inflated"] = dict(p=1, nll=gauss_nll(R, s_inf * sig), chi2=float(n), inflation=s_inf)
    for lab, Q in Q_list:
        pred = np.array([F3_pred(q["r_eff"], q["dtheta"], q["omega_pair"], Q) for q in sel])
        M["M4_pred_" + lab] = dict(p=0, nll=gauss_nll(R - pred, sig), chi2=float(np.sum(((R - pred) / sig) ** 2)),
                                   Q=Q, B0=float(k3_of(np.mean(om)) * Q * Q / (8 * np.pi)),
                                   pred=pred.tolist())
    for m in M.values():
        m["AIC"] = m["nll"] + 2 * m["p"]
        m["chi2_red"] = m["chi2"] / max(n - m["p"], 1)
    # k scan (M2 form, k free on a grid): where k3 sits among the aliases
    ks = np.linspace(0.05, 3.2, 3151)
    kscan = dict(k3=float(np.mean(k3)))
    for lab, extra in (("M2_form", []), ("M6_form", [Fs])):
        chis = []
        for k in ks:
            Ak = np.stack(extra + [sc * np.sin(k * r) / r, sc * np.cos(k * r) / r], axis=1)
            chis.append(wls(Ak, R, sig)[2])
        chis = np.array(chis)
        loc = [i for i in range(1, ks.size - 1) if chis[i] < chis[i - 1] and chis[i] <= chis[i + 1]]
        loc = sorted(loc, key=lambda i: chis[i])[:5]
        kscan[lab] = dict(best_k=float(ks[np.argmin(chis)]), best_chi2=float(chis.min()),
                          chi2_at_k3=float(M["M2_3omega_r1" if lab == "M2_form" else "M6_baseline_scale_plus_3omega"]["chi2"]),
                          local_minima=[dict(k=float(ks[i]), chi2=float(chis[i])) for i in loc])
    return dict(r_min=rmin, n=n, names=[q["name"] for q in sel], r_eff=r.tolist(), residual=R.tolist(),
                sigma=sig.tolist(), models=M, k_scan=kscan,
                AIC_order=sorted(M, key=lambda k: M[k]["AIC"]))


def miss1(a, results):
    rows = miss1_rows(a, results)
    Qb, info = born_Q()
    QB = 4 * np.pi * RA3_TESTB
    Q_list = [("TestB_ra3", QB), ("Born_profile", abs(Qb))]
    d1 = {"r_eff_22_40": miss1_models(rows, 22.0, Q_list), "r_eff_26_40": miss1_models(rows, 26.0, Q_list)}
    box = a["controls"]["box"]
    prod = [q for q in rows if q["name"] == "pA_r24_p000"][0]
    d1["box_check_r24"] = dict(F_box240=box["F"], sigma_box240=box["sigma_F"], F_box160=prod["F"],
                               sigma_box160=prod["sigma_F"], diff=box["F"] - prod["F"],
                               residual_box160=prod["F"] - prod["F_static"])
    # D2 summaries
    d2 = []
    for q in rows:
        e = dict(name=q["name"], r_eff=q["r_eff"], F=q["F"], sigma_F=q["sigma_F"], F_P=q["F_P"], F_P_se=q["F_P_se"],
                 F_P_minus_F=q["F_P"] - q["F"], res_centroid=q["F"] - q["F_static"],
                 res_momentum=q["F_P"] - q["F_static"])
        for k in ("half1", "half2", "late120"):
            if ("F_P_" + k) in q and q["F_P_" + k] is not None:
                e["res_" + k] = q["F_P_" + k] - q["F_static"]
                e["se_" + k] = q["F_P_" + k + "_se"]
        d2.append(e)
    # D3 magnitudes
    om = TYPES["primary"]["omega"]
    k3 = k3_of(om)
    d3 = dict(k3=k3, wavelength=2 * np.pi / k3, Q_born=Qb, ra3_born=abs(Qb) / (4 * np.pi), Q_testB=QB,
              ra3_testB=RA3_TESTB, profile=info,
              B0_testB=k3 * QB ** 2 / (8 * np.pi), B0_born=k3 * Qb ** 2 / (8 * np.pi),
              amplitude_at={str(rr): dict(testB=k3 * QB ** 2 / (8 * np.pi * rr), born=k3 * Qb ** 2 / (8 * np.pi * rr))
                            for rr in (24, 30, 34, 40)},
              formula="F_3 = (k3 Q^2/(8 pi r)) sin(k3 r - 3 dtheta), F > 0 attracts; Q = int s3 e^{-i k3 x}, "
                      "s3 = (g4/24) f^3 - (g6/384) f^5; Born r a3 = |Q|/(4 pi)")
    return dict(rows=rows, D1=d1, D2=d2, D3=d3)


# ------------------------------------------------------------------------------------------- Miss 2
def static_split(st):
    """Per-lump first-order potentials U_1, U_2 (cos dtheta = 1) on the static r grid and their forces (the
    add_derived spline route with k = min kappa), plus the stored-U_amp reproduction check."""
    rows = st["rows"]
    r = np.array([q["r"] for q in rows])
    l1, l2 = st["lump1"], st["lump2"]
    mo = lambda key: np.array([q["moments"][key] for q in rows])
    X1 = G4 / 8.0 * mo("M31") - l1["g6"] / 192.0 * mo("M51")
    X2 = G4 / 8.0 * mo("M13") - l2["g6"] / 192.0 * mo("M15")
    U1, U2 = -0.5 * X1, -0.5 * X2
    w1 = l1["omega"] / (l1["omega"] + l2["omega"])
    k = min(l1["kappa"], l2["kappa"])
    from scipy.interpolate import CubicSpline

    def force(U):
        g = CubicSpline(r, U * r * np.exp(k * r))
        return (g(r, 1) - g(r) * (k + 1.0 / r)) * np.exp(-k * r) / r
    U_amp_stored = np.array([q["p000"]["U_amp"] for q in rows])
    return dict(r=r, U1=U1, U2=U2, F1=force(U1), F2=force(U2), w1=w1, w2=1 - w1,
                U_amp_check_max_rel=float(np.max(np.abs(w1 * U1 + (1 - w1) * U2 - U_amp_stored) / np.abs(U_amp_stored))))


def miss2_row(q, res, sp, om1):
    """E2 and E3 for one unequal pair row."""
    sc = 1.0 if q["dtheta"] == 0 else -1.0       # static amplitudes carry cos(dtheta); measured F_amp does not
    t1, t2 = q["window"]["t1"], q["window"]["t2"]
    P = 2 * np.pi / om1
    E1, E2 = q["inertia"]["E1"], q["inertia"]["E2"]
    inv_m = 1 / E1 + 1 / E2
    tph, dphi = AN.relative_phase_series(res, om1)
    # centroid series and smoothed separation, as window_for / pair_unequal
    t, X = AN.separation(res)
    ts, Xs = AN.hann_smooth(t, X.reshape(t.size, -1), 3 * P)
    Xs = Xs.reshape(ts.size, -1, 3)
    d = np.linalg.norm(Xs[:, 1, :] - Xs[:, 0, :], axis=1)
    dt = ts[1] - ts[0]
    ph = np.interp(ts, tph, dphi)
    G = {}
    for key, g in (("cos", np.cos(ph)), ("sin", np.sin(ph)), ("cos2", np.cos(2 * ph))):
        G1 = cumtrap(g, dt)
        G[key] = (G1, cumtrap(G1, dt))
    # sanity: reproduce the stored F_amp
    c, se, _ = AN.accel_fit(ts, d, t1, t2, P, extra=[G["cos"][1]])
    F_amp_re = -float(c[3]) / inv_m
    out = dict(name=q["name"], r=q["r"], r_eff=q["r_eff"], dtheta=q["dtheta"], F_amp=q["F_amp"],
               sigma_F_amp=q["sigma_F_amp"], F_amp_recomputed=F_amp_re, E1=E1, E2=E2)
    # E3: phase columns on d
    c3, se3, _ = AN.accel_fit(ts, d, t1, t2, P, extra=[G["cos"][1], G["sin"][1]])
    cc, cs = -c3[3] / inv_m, -c3[4] / inv_m
    c4, se4, _ = AN.accel_fit(ts, d, t1, t2, P, extra=[G["cos"][1], G["cos2"][1]])
    out["E3"] = dict(F_cos=float(cc), F_cos_se=float(se3[3] / inv_m), F_sin=float(cs), F_sin_se=float(se3[4] / inv_m),
                     amplitude=float(np.hypot(cc, cs)), lag=float(np.arctan2(cs, cc)),
                     F_cos_with_cos2=float(-c4[3] / inv_m), F_cos2=float(-c4[4] / inv_m),
                     F_cos2_se=float(se4[4] / inv_m))
    # E2: per-lump forces, centroid route (x_i) and ball-momentum route (P_ix), attraction positive
    per = {}
    for i, sgn in ((0, 1.0), (1, -1.0)):
        ci, sei, _ = AN.accel_fit(ts, Xs[:, i, 0], t1, t2, P, extra=[G["cos"][1]])
        Ei = E1 if i == 0 else E2
        per["centroid_F%d" % (i + 1)] = float(sgn * Ei * ci[3])
        per["centroid_F%d_se" % (i + 1)] = float(Ei * sei[3])
    s = res["lumps"]["raw"]["series"]
    tP = np.array(s["t"])
    Px = np.array(s["P"])[:, :, 0]
    tsP, PxS = AN.hann_smooth(tP, Px, 3 * P)
    phP = np.interp(tsP, tph, dphi)
    G1P = cumtrap(np.cos(phP), tsP[1] - tsP[0])
    G1Ps = cumtrap(np.sin(phP), tsP[1] - tsP[0])
    tc = tsP - 0.5 * (t1 + t2)
    for i, sgn in ((0, 1.0), (1, -1.0)):
        cp, sep = lsq_neff(tsP, PxS[:, i], t1, t2, P, [np.ones_like(tc), tc, G1P])
        per["momentum_F%d" % (i + 1)] = float(sgn * cp[2])
        per["momentum_F%d_se" % (i + 1)] = float(sep[2])
        cp2, _ = lsq_neff(tsP, PxS[:, i], t1, t2, P, [np.ones_like(tc), tc, G1P, G1Ps])
        per["momentum_F%d_cos_with_sin" % (i + 1)] = float(sgn * cp2[2])
        per["momentum_F%d_sin" % (i + 1)] = float(sgn * cp2[3])
    per["momentum_F_amp"] = (per["momentum_F1"] / E1 + per["momentum_F2"] / E2) / inv_m
    per["centroid_F_amp"] = (per["centroid_F1"] / E1 + per["centroid_F2"] / E2) / inv_m
    # static at r_eff
    F1s = sc * float(np.interp(q["r_eff"], sp["r"], sp["F1"]))
    F2s = sc * float(np.interp(q["r_eff"], sp["r"], sp["F2"]))
    a1, a2 = (1 / E1) / inv_m, (1 / E2) / inv_m
    out["E1"] = dict(F1_static=F1s, F2_static=F2s, F_amp_static_w=sp["w1"] * F1s + sp["w2"] * F2s,
                     F_dyn_static=a1 * F1s + a2 * F2s, a1=a1, a2=a2, w1=sp["w1"], w2=sp["w2"],
                     ratio_meas_over_w=q["F_amp"] / (sp["w1"] * F1s + sp["w2"] * F2s),
                     ratio_meas_over_dyn=q["F_amp"] / (a1 * F1s + a2 * F2s),
                     F1_share_of_F_dyn=a1 * F1s / (a1 * F1s + a2 * F2s))
    out["E2"] = per
    out["E2"]["ratio_F1_momentum_over_static"] = per["momentum_F1"] / F1s
    out["E2"]["ratio_F2_momentum_over_static"] = per["momentum_F2"] / F2s
    out["E2"]["ratio_F1_centroid_over_static"] = per["centroid_F1"] / F1s
    out["E2"]["ratio_F2_centroid_over_static"] = per["centroid_F2"] / F2s
    out["E2"]["ratio_F_amp_momentum_over_dyn"] = per["momentum_F_amp"] / (a1 * F1s + a2 * F2s)
    return out


def miss2(a, results, data):
    st = json.load(open(os.path.join(data, "static/static_unequal_0.906505_0.857370.json")))
    sp = static_split(st)
    om1 = TYPES["dip"]["omega"]
    out = dict(static_split=dict(U_amp_reproduction_max_rel=sp["U_amp_check_max_rel"], w1=sp["w1"],
                                 lump1=st["lump1"], lump2=st["lump2"],
                                 F1_over_F2_at={str(rr): float(np.interp(rr, sp["r"], sp["F1"]) / np.interp(rr, sp["r"], sp["F2"]))
                                                for rr in (18, 20, 24, 30, 36, 40)}),
               channels={})
    for lab in ("p000", "p180"):
        ch = a["unequal"]["channels"][lab]
        stored = dict(zip(ch["F_amp"]["names"], ch["F_amp_static_at_r_eff"]))
        rows = []
        for q in ch["rows"]:
            if q["force_excluded"] or not q["use_in_fits"]:
                continue
            res = AN.load(q["name"], results)
            rr = miss2_row(q, res, sp, om1)
            rr["F_amp_static_stored"] = stored[q["name"]]
            rows.append(rr)
        r = np.array([q["r_eff"] for q in rows])
        F = np.array([q["F_amp"] for q in rows])
        sF = np.array([q["sigma_F_amp"] for q in rows])
        Fw = np.array([q["E1"]["F_amp_static_w"] for q in rows]) * (1 if lab == "p000" else -1)
        Fd = np.array([q["E1"]["F_dyn_static"] for q in rows]) * (1 if lab == "p000" else -1)
        FP = np.array([q["E2"]["momentum_F_amp"] for q in rows])
        summ = {}
        for rmin in (18.0, 26.0):
            m = (r >= rmin - 0.5) & (np.abs(F) > 3 * sF)
            summ["r_ge_%g" % rmin] = dict(
                n=int(m.sum()),
                k_measured=loglin_k(r[m], F[m], sF[m]), k_static_w=loglin_k(r[m], Fw[m], sF[m] * np.abs(Fw[m] / F[m])),
                k_static_dyn=loglin_k(r[m], Fd[m], sF[m] * np.abs(Fd[m] / F[m])))
        m = (r >= 25.5) & (np.abs(F) > 3 * sF)
        for key, ref in (("w", Fw), ("dyn", Fd)):
            rat = F[m] / ref[m]
            c, se, chi2 = wls(np.stack([np.ones(m.sum()), r[m] - 30.0], axis=1), np.log(rat), sF[m] / np.abs(F[m]))
            summ["ln_ratio_drift_r_ge_26_" + key] = dict(ratio_at_30=float(np.exp(c[0])), slope=float(c[1]),
                                                         slope_se=float(se[1]), chi2=chi2, n=int(m.sum()))
        out["channels"][lab] = dict(rows=rows, summary=summ)
    return out


# ------------------------------------------------------------------------------------------- main
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(data=DATA_DEFAULT, out=OUT):
    results = os.path.join(data, "results")
    a = json.load(open(os.path.join(data, "analysis.json")))
    try:
        commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    rec = dict(what="Test A production misses: diagnostics from stored results (no PDE runs)", definitions=__doc__,
               inputs=dict(data_dir=data, analysis_json_sha256=sha(os.path.join(data, "analysis.json")),
                           code_commit=commit, RA3_TESTB=RA3_TESTB, g4=G4, g6=G6, omega_primary=TYPES["primary"]["omega"],
                           omega_dip_demod=TYPES["dip"]["omega"]),
               miss1=miss1(a, results), miss2=miss2(a, results, data))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w") as f:
        json.dump(rec, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    os.replace(out + ".tmp", out)
    return rec


if __name__ == "__main__":
    main(*(sys.argv[1:2] or [DATA_DEFAULT]))
