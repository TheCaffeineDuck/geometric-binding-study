"""Test A analysis: window rule, V_N, V_F, floors, the five fits and the A3 verdict (SESSION_04_PLAN A0, A2, A3).

Window (A0): t1 = 40; t2 = the earlier of 400, the run end, and the first time kappa |d(t) - r| reaches 0.10,
with d(t) the lump separation (per-lump energy centroids, smoothed by a Hann window of 3 periods); a run
with t2 - t1 < 35 is "moving" (reported, excluded from the long-range fits).

Window averages: every time average uses one Hann weight on [t1, t2] on the common step grid of the pair
and its singles (so a slow drift of N enters pair and single at the same effective time; whole-period
zero-crossing averages are recorded too). omega of a run = whole-period mean from the zero crossings of
phi(lump 1 centre, t) inside [t1, t2]. N = <int phi_t^2>_w / omega.

V_N (equal lumps, A0):
    V_N = [E_pair - 2 E_s] - omega_s [N_pair - 2 N_s] - (d omega/d N) (N_pair - 2 N_s)^2 / 4
with E = E + absorbed (as written in A0). V_N_box is the same with the box energy E (no absorbed): radiation
in the box carries int phi_t^2 = E (equipartition), so E_box - omega N is insensitive to radiation, while
E + absorbed counts radiated energy whose N has left; both are reported. The single values are interpolated
linearly in the lump-1 position between the two singles of the box.
V_F: d(t) on [t1, t2] fitted by c0 + c1 t + c2 t^2 (Hann-smoothed data; smoothing a quadratic changes only
c0); d'' = 2 c2; F = -(E_s/2) d'' (F > 0 attracts); unequal lumps F = -d'' / (1/E_1 + 1/E_2).
V_F(r) = -int_r^{r_max} F dr' (trapezoid on the measured r), with and without the tail beyond r_max taken
from the E2 fit.
Unequal lumps (A1): the force amplitude from d(t) = c0 + c1 t + c2 G(t), G'' = cos(dphi(t)) with dphi the
measured relative phase; V_N(t) (half-space N per lump) fitted by a + b cos(dphi(t)).

Floors: sigma_VN = max(|single shifted against itself| in the box, |V_N| of the largest-r pair, half-window
difference); sigma_F = max(fit standard error, |F| of the largest-r pair).

Fits (A2): E1 C e^{-kr}; E2 C e^{-kr}/r; G C e^{-(r/s)^2}; P C r^{-n}; L V_inf + sigma r + C e^{-kr}/r; weighted
least squares (absolute sigma), reduced chi^2, AIC = chi^2 + 2p, BIC = chi^2 + p ln n, Wald-Wolfowitz runs test
on the residual signs; L: sigma +- s.e.
"""
import json
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = "outputs/session4/testA"
T1, T_MAX, KAPPA_SHIFT, MOVING_MIN = 40.0, 400.0, 0.10, 35.0
R_LO = 12.0
DECAYING = ("E1", "E2", "G", "P")


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


# ------------------------------------------------------------------------------------------- time series
def hann_avg(t, y, t1, t2):
    t, y = np.asarray(t, float), np.asarray(y, float)
    s = (t >= t1) & (t <= t2)
    if s.sum() < 3:
        return None
    w = np.sin(np.pi * (t[s] - t1) / (t2 - t1)) ** 2
    return float(np.sum(w * y[s]) / np.sum(w))


def hann_smooth(t, y, W):
    """Centred Hann moving average of length W on a uniform grid (valid part, times returned)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    dt = t[1] - t[0]
    n = max(3, int(round(W / dt)) | 1)
    k = np.sin(np.pi * (np.arange(n) + 0.5) / n) ** 2
    k /= k.sum()
    if y.ndim == 1:
        ys = np.convolve(y, k, mode="valid")
    else:
        ys = np.stack([np.convolve(y[:, i], k, mode="valid") for i in range(y.shape[1])], axis=1)
    h = (n - 1) // 2
    return t[h:t.size - h], ys


def zero_cross(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float)
    i = np.nonzero(np.signbit(y[1:]) != np.signbit(y[:-1]))[0]
    return t[i] - y[i] * (t[i + 1] - t[i]) / (y[i + 1] - y[i])


def omega_window(t, y, t1, t2):
    z = zero_cross(t, y)
    z = z[(z >= t1) & (z <= t2)]
    n = z.size - 1
    if n % 2:
        z, n = z[:-1], n - 1
    if n < 2:
        return None
    return float(np.pi * n / (z[-1] - z[0]))


def separation(res):
    ls = res["lumps"]["raw"]["series"]
    t = np.array(ls["t"])
    X = np.array(ls["X"])
    return t, X


def window_for(res, r, kappa, omega):
    """(t1, t2, moving, d-series) per the A0 rule."""
    t, X = separation(res)
    P = 2 * np.pi / omega
    ts, Xs = hann_smooth(t, X.reshape(t.size, -1), 3 * P)
    Xs = Xs.reshape(ts.size, -1, 3)
    d = np.linalg.norm(Xs[:, 1, :] - Xs[:, 0, :], axis=1)
    t_end = min(T_MAX, float(res["t"]))
    hit = np.nonzero((kappa * np.abs(d - r) >= KAPPA_SHIFT) & (ts > 0))[0]
    t2 = min(t_end - 1.5 * P, float(ts[hit[0]]) if hit.size else np.inf)
    t1 = T1
    moving = (t2 - t1) < MOVING_MIN
    if t2 - t1 < 2 * P:          # short window: the last 2 periods before t2 (flagged "moving")
        t1 = max(0.0, t2 - 2 * P)
    return dict(t1=float(t1), t2=float(t2), moving=bool(moving), t_d=ts, d=d, t_hit=float(ts[hit[0]]) if hit.size else None)


def run_values(res, t1, t2):
    """E + absorbed, E_box, N (Hann) and omega in [t1, t2] for one run (lump 1 = the config centre)."""
    s, p = res["series"], res["point"]
    om = omega_window(p["t"], p["phi"], t1, t2)
    I = hann_avg(p["t"], p["I"], t1, t2)
    out = dict(E=hann_avg(s["t"], s["E_plus_absorbed"], t1, t2), E_box=hann_avg(s["t"], s["E_total"], t1, t2),
               I=I, omega=om, N=(I / om if (I is not None and om) else None),
               absorbed=hann_avg(s["t"], s["absorbed"], t1, t2))
    lp = res["lumps"]["raw"]["point"]
    if lp.get("It_half"):
        out["I_half"] = hann_avg(lp["t"], np.array(lp["It_half"])[:, 0], t1, t2)
    return out


def interp_single(sA, sB, x):
    """Linear interpolation in the lump position x between singles (x_A, vals_A), (x_B, vals_B)."""
    (xa, a), (xb, b) = sA, sB
    if abs(xb - xa) < 1e-12:
        return dict(a)
    u = (x - xa) / (xb - xa)
    return {k: (a[k] + u * (b[k] - a[k]) if (a.get(k) is not None and b.get(k) is not None) else None) for k in a}


def V_N_equal(pv, sv, domega_dN):
    dN = pv["N"] - 2 * sv["N"]
    om = sv["omega"]
    second = -domega_dN * dN * dN / 4.0
    return dict(V_N=(pv["E"] - 2 * sv["E"]) - om * dN + second, V_N_box=(pv["E_box"] - 2 * sv["E_box"]) - om * dN + second,
                dE=pv["E"] - 2 * sv["E"], dE_box=pv["E_box"] - 2 * sv["E_box"], dN=dN, omega0=om, second_order=second)


def accel_fit(t, d, t1, t2, period, basis=None):
    """Fit d(t) on [t1, t2] by c0 + c1 t + c2 b(t) (b = t^2 default). Returns c2, its s.e. (n_eff = window/period)."""
    s = (t >= t1) & (t <= t2)
    tt, dd = t[s], d[s]
    if tt.size < 5:
        return None, None, None
    tc = tt - 0.5 * (t1 + t2)
    b = tc ** 2 if basis is None else basis[s]
    A = np.stack([np.ones_like(tc), tc, b], axis=1)
    c, *_ = np.linalg.lstsq(A, dd, rcond=None)
    res = dd - A @ c
    n_eff = max((t2 - t1) / period, 4.0)
    s2 = float(np.sum(res ** 2) / max(tt.size - 3, 1)) * tt.size / n_eff
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(c[2]), float(np.sqrt(cov[2, 2])), float(np.sqrt(np.mean(res ** 2)))


def relative_phase_series(res, omega_ref):
    """Unwrapped relative phase theta_2 - theta_1 from the per-step phi at the lump centres (demodulated
    with exp(-i omega_ref t) and Hann-smoothed over 3 periods of omega_ref)."""
    lp = res["lumps"]["raw"]["point"]
    t = np.array(lp["t"])
    ph = np.array(lp["phi"])
    z = ph * np.exp(-1j * omega_ref * t)[:, None]
    ts, zs = hann_smooth(t, np.stack([z[:, 0].real, z[:, 0].imag, z[:, 1].real, z[:, 1].imag], axis=1),
                         3 * 2 * np.pi / omega_ref)
    c1, c2 = zs[:, 0] + 1j * zs[:, 1], zs[:, 2] + 1j * zs[:, 3]
    return ts, np.unwrap(np.angle(c2 * np.conj(c1)))


# ------------------------------------------------------------------------------------------- fits
def _ln_guess(x, y, w):
    A = np.stack([np.ones_like(x), x], axis=1) * w[:, None]
    c, *_ = np.linalg.lstsq(A, y * w, rcond=None)
    return c


def _model(form, r, q, s):
    if form == "E1":
        return s * np.exp(q[0] - q[1] * r)
    if form == "E2":
        return s * np.exp(q[0] - q[1] * r) / r
    if form == "G":
        return s * np.exp(q[0] - (r / q[1]) ** 2)
    if form == "P":
        return s * np.exp(q[0] - q[1] * np.log(r))
    if form == "L":
        return q[0] + q[1] * r + s * np.exp(q[2] - q[3] * r) / r
    raise ValueError(form)


def runs_test(res):
    sg = np.sign(res)
    sg = sg[sg != 0]
    n1, n2 = int(np.sum(sg > 0)), int(np.sum(sg < 0))
    n = n1 + n2
    if n1 == 0 or n2 == 0 or n < 3:
        return dict(runs=1, expected=None, z=None, p=None, n_pos=n1, n_neg=n2)
    R = 1 + int(np.sum(sg[1:] != sg[:-1]))
    mu = 2.0 * n1 * n2 / n + 1
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1))
    z = (R - mu) / np.sqrt(var) if var > 0 else 0.0
    return dict(runs=R, expected=mu, z=float(z), p=float(2 * norm.sf(abs(z))), n_pos=n1, n_neg=n2)


def fit_form(form, r, V, sig):
    """Weighted least squares of one candidate. Returns params (physical), s.e., chi2, reduced, AIC, BIC, runs."""
    r, V, sig = (np.asarray(x, float) for x in (r, V, sig))
    s = float(np.sign(V[np.argmax(np.abs(V))])) or 1.0
    lv = np.log(np.maximum(np.abs(V), 1e-300))
    w = np.abs(V) / sig
    w = w / w.max()
    starts = []
    if form == "E1":
        c = _ln_guess(r, lv, w)
        starts = [[c[0], -c[1]]]
    elif form == "E2":
        c = _ln_guess(r, lv + np.log(r), w)
        starts = [[c[0], -c[1]]]
    elif form == "G":
        c = _ln_guess(r * r, lv, w)
        s0 = np.sqrt(-1.0 / c[1]) if c[1] < 0 else float(np.mean(r))
        starts = [[c[0], s0], [lv[0] + (r[0] / (2 * r.max())) ** 2, 2 * r.max()]]
    elif form == "P":
        c = _ln_guess(np.log(r), lv, w)
        starts = [[c[0], -c[1]]]
    elif form == "L":
        c = _ln_guess(r, lv + np.log(r), w)
        for kf in (1.0, 0.8, 1.2, 0.5):
            starts.append([0.0, 0.0, c[0], -c[1] * kf])
    np_ = len(starts[0])
    best = None
    for q0 in starts:
        try:
            if form == "L":
                sc = np.array([max(np.abs(V).min(), 1e-30), max(np.abs(V).min(), 1e-30) / max(r.max(), 1.0), 1.0, 0.1])
                f = lambda q: (_model(form, r, q, s) - V) / sig
                sol = least_squares(f, q0, x_scale=sc, method="lm", max_nfev=20000, xtol=1e-15, ftol=1e-15)
            else:
                f = lambda q: (_model(form, r, q, s) - V) / sig
                sol = least_squares(f, q0, method="lm", max_nfev=20000, xtol=1e-15, ftol=1e-15)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        return dict(form=form, ok=False)
    q = best.x
    res = best.fun
    chi2 = float(np.sum(res ** 2))
    n = r.size
    dof = n - np_
    J = best.jac
    try:
        cov_q = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov_q = np.full((np_, np_), np.nan)
    se_q = np.sqrt(np.abs(np.diag(cov_q)))
    if form == "L":
        C = s * np.exp(q[2])
        params = dict(V_inf=q[0], sigma=q[1], C=C, k=q[3])
        se = dict(V_inf=se_q[0], sigma=se_q[1], C=abs(C) * se_q[2], k=se_q[3])
    else:
        C = s * np.exp(q[0])
        key = {"E1": "k", "E2": "k", "G": "s", "P": "n"}[form]
        params = {"C": C, key: q[1]}
        se = {"C": abs(C) * se_q[0], key: se_q[1]}
    return dict(form=form, ok=bool(best.success), params={k: float(v) for k, v in params.items()},
                se={k: float(v) for k, v in se.items()}, chi2=chi2, dof=dof,
                chi2_red=(chi2 / dof if dof > 0 else None), AIC=chi2 + 2 * np_, BIC=chi2 + np_ * np.log(n), n=n,
                runs=runs_test(res), residual_over_sigma=[float(x) for x in res])


def fit_all(r, V, sig):
    return {f: fit_form(f, r, V, sig) for f in DECAYING + ("L",)}


# ------------------------------------------------------------------------------------------- verdict (A3)
def resolved_range(r, V, sig, r_lo=R_LO):
    """Indices of the fit range r_lo .. r_hi, r_hi the last r of the contiguous resolved run from r_lo, and the
    mask of all resolved points."""
    r, V, sig = (np.asarray(x, float) for x in (r, V, sig))
    res = np.abs(V) > 3 * sig
    idx = [i for i in np.argsort(r) if r[i] >= r_lo - 1e-9]
    take = []
    for i in idx:
        if not res[i]:
            break
        take.append(i)
    return np.array(take, int), res


def verdict(r, V, sig, F16, r_lo=R_LO):
    """A3 verdict for one potential and channel. r, V, sig: the sampled points (moving runs already
    excluded); F16: the measured |F| at r = 16 for this potential and channel."""
    r, V, sig = (np.asarray(x, float) for x in (r, V, sig))
    rng, res = resolved_range(r, V, sig, r_lo)
    out = dict(r_lo=r_lo, n_fit=int(rng.size), r_hi=(float(r[rng[-1]]) if rng.size else None))
    if rng.size < 5:
        out.update(verdict="undetermined", reason="fewer than 5 resolved points from r_lo", fits=None)
        return out
    rr, VV, ss = r[rng], V[rng], sig[rng]
    fits = fit_all(rr, VV, ss)
    resolved_all = np.nonzero(res)[0]
    fits_full = fit_all(r[resolved_all], V[resolved_all], sig[resolved_all]) if resolved_all.size >= 5 else None
    fall = float(abs(VV[0]) / abs(VV[-1]))
    below = bool(np.any(np.abs(V) < 3 * sig))
    decades = float(np.log10(np.max(np.abs(V[res])) / np.min(np.abs(V[res])))) if res.any() else 0.0
    L = fits["L"]
    sg, se = (L["params"]["sigma"], L["se"]["sigma"]) if L.get("ok") else (np.nan, np.nan)
    red = {f: fits[f].get("chi2_red") for f in DECAYING}
    good = [f for f in DECAYING if red[f] is not None and red[f] <= 3.0]
    aic = {f: fits[f]["AIC"] for f in fits if fits[f].get("ok")}
    d1 = fall >= 100.0 and below
    d2 = bool(np.isfinite(sg) and abs(sg) <= 3 * se and (abs(sg) + 3 * se) < 0.01 * abs(F16))
    d3 = bool(good)
    last3 = np.abs(VV[-3:])
    g1 = bool(np.isfinite(sg) and se > 0 and abs(sg) > 5 * se and sg > 0)
    g2 = bool(abs(VV[-1]) > 3 * ss[-1] and last3[0] <= last3[1] <= last3[2])
    g3 = bool("L" in aic and all(aic["L"] + 10.0 < aic[f] for f in DECAYING if f in aic))
    if d1 and d2 and d3:
        v = "decays to zero"
    elif g1 and g2 and g3:
        v = "grows"
    else:
        v = "undetermined"
    order = sorted(aic, key=aic.get)
    pref = order[0] if len(order) > 1 and aic[order[0]] + 10.0 < aic[order[1]] else None
    k_fit = {f: fits[f]["params"].get("k") for f in ("E1", "E2", "L") if fits[f].get("ok")}
    out.update(verdict=v, clauses=dict(
        decays=dict(fall_factor=fall, fall_ge_100=fall >= 100.0, below_3sigma_somewhere=below, L_sigma=sg, L_sigma_se=se,
                    L_bound=abs(sg) + 3 * se if np.isfinite(sg) else None, F16=abs(F16),
                    L_sigma_within_3se_and_bound_below_1pct_F16=d2, decaying_forms_chi2red_le_3=good),
        grows=dict(L_sigma_over_se=(sg / se if se else None), sigma_confining_gt_5se=g1,
                   V_hi_resolved_and_not_decreasing=g2, L_beats_decaying_by_dAIC_10=g3),
        decades_resolved=decades), preferred_form=pref, AIC_order=order, chi2_red=red,
        k_fit=k_fit, fits=fits, fits_full_range=fits_full)
    if decades < 2.0 and v != "undetermined":
        out["verdict"], out["reason"] = "undetermined", "fewer than two decades of resolved |V|"
    return out


A3P_MODEL_FRAC = 0.02
A3P_R_LOS = (12.0, 16.0)
A3P_DECIDING_R_LO = 16.0


def verdict_prime(r, V, sig, F16, r_lo=A3P_DECIDING_R_LO):
    """A3' (SESSION_04_PLAN Addendum 1 section 2), scored next to A3 (unchanged).

    sigma_eff = sqrt(sigma_V^2 + (0.02 V)^2) weights the fits; "resolved" is still |V| > 3 sigma_V.
    Decays: (1) |V| falls >= 100x over [r_lo, r_hi] and is still falling over the last three resolved points
    (reaching the floor is reported, not required); (2) the 3-s.e. upper bound on |sigma| (fit L) < 1% |F(16)|;
    (3) some decaying form has reduced chi^2 <= 3 under sigma_eff.
    Grows: (1) sigma confining (> 0), > 5 s.e., and |sigma| > 1% |F(16)|; (2) A3 grows-clause 2; (3) L beats every
    decaying form by Delta AIC > 10. Undetermined otherwise. Form preference among E1, E2, G, P only (> 10)."""
    r, V, sig = (np.asarray(x, float) for x in (r, V, sig))
    rng, res = resolved_range(r, V, sig, r_lo)
    out = dict(rule="A3'", r_lo=r_lo, n_fit=int(rng.size), r_hi=(float(r[rng[-1]]) if rng.size else None),
               model_frac=A3P_MODEL_FRAC)
    if rng.size < 5:
        out.update(verdict="undetermined", reason="fewer than 5 resolved points from r_lo", fits=None)
        return out
    rr, VV, ss = r[rng], V[rng], sig[rng]
    se_eff = np.sqrt(ss ** 2 + (A3P_MODEL_FRAC * VV) ** 2)
    fits = fit_all(rr, VV, se_eff)
    fall = float(abs(VV[0]) / abs(VV[-1]))
    last3 = np.abs(VV[-3:])
    still_falling = bool(last3[0] > last3[1] > last3[2])
    below = bool(np.any(np.abs(V) < 3 * sig))
    L = fits["L"]
    sg, se = (L["params"]["sigma"], L["se"]["sigma"]) if L.get("ok") else (np.nan, np.nan)
    red = {f: fits[f].get("chi2_red") for f in DECAYING}
    good = [f for f in DECAYING if red[f] is not None and red[f] <= 3.0]
    aic = {f: fits[f]["AIC"] for f in fits if fits[f].get("ok")}
    bound = abs(sg) + 3 * se if np.isfinite(sg) else np.inf
    d1 = fall >= 100.0 and still_falling
    d2 = bool(bound < 0.01 * abs(F16))
    d3 = bool(good)
    g1 = bool(np.isfinite(sg) and se > 0 and sg > 0 and abs(sg) > 5 * se and abs(sg) > 0.01 * abs(F16))
    g2 = bool(abs(VV[-1]) > 3 * ss[-1] and last3[0] <= last3[1] <= last3[2])
    g3 = bool("L" in aic and all(aic["L"] + 10.0 < aic[f] for f in DECAYING if f in aic))
    v = "decays to zero" if (d1 and d2 and d3) else ("grows" if (g1 and g2 and g3) else "undetermined")
    dec = sorted([f for f in DECAYING if f in aic], key=aic.get)
    pref = dec[0] if len(dec) > 1 and aic[dec[0]] + 10.0 < aic[dec[1]] else None
    for f in fits.values():
        f.pop("residual_over_sigma", None)
    out.update(verdict=v, clauses=dict(
        decays=dict(fall_factor=fall, fall_ge_100=fall >= 100.0, still_falling_last3=still_falling,
                    below_3sigma_somewhere_reported=below, L_sigma=sg, L_sigma_se=se, L_bound_3se=bound,
                    one_pct_F16=0.01 * abs(F16), L_bound_below_1pct_F16=d2, decaying_forms_chi2red_le_3=good),
        grows=dict(L_sigma_over_se=(sg / se if se else None), sigma_confining_5se_and_above_1pct_F16=g1,
                   V_hi_resolved_and_not_decreasing=g2, L_beats_decaying_by_dAIC_10=g3)),
        preferred_form=pref, decaying_AIC_order=dec, chi2_red=red,
        k_fit={f: fits[f]["params"].get("k") for f in ("E1", "E2", "L") if fits[f].get("ok")}, fits=fits)
    return out


def verdicts_both(r, V, sig, F16):
    """A3 (as written, r_lo = 12) and A3' (r_lo = 12 and 16; 16 decides) for one potential and channel."""
    a3 = verdict(r, V, sig, F16, r_lo=R_LO)
    for f in (a3.get("fits") or {}).values():
        f.pop("residual_over_sigma", None)
    for f in (a3.get("fits_full_range") or {}).values():
        f.pop("residual_over_sigma", None)
    ap = {"r_lo_%g" % rl: verdict_prime(r, V, sig, F16, r_lo=rl) for rl in A3P_R_LOS}
    return {"A3": a3, "A3prime": dict(ap, verdict=ap["r_lo_%g" % A3P_DECIDING_R_LO]["verdict"],
                                      deciding_r_lo=A3P_DECIDING_R_LO)}


def force_signs(r, F, sF):
    """Per-r sign label and minima (sign change of F between resolved neighbours, linear interpolation)."""
    r, F, sF = (np.asarray(x, float) for x in (r, F, sF))
    o = np.argsort(r)
    r, F, sF = r[o], F[o], sF[o]
    lab = ["attractive" if f > 3 * s else ("repulsive" if f < -3 * s else "unresolved") for f, s in zip(F, sF)]
    mins = []
    for i in range(r.size - 1):
        if lab[i] != "unresolved" and lab[i + 1] != "unresolved" and lab[i] != lab[i + 1]:
            rz = r[i] - F[i] * (r[i + 1] - r[i]) / (F[i + 1] - F[i])
            mins.append(dict(between=[float(r[i]), float(r[i + 1])], r_zero=float(rz),
                             kind="minimum" if lab[i] == "repulsive" else "maximum"))
    return dict(r=r.tolist(), F=F.tolist(), sigma_F=sF.tolist(), label=lab, sign_changes=mins)


def headline(verdicts, keys=("U_static", "V_N", "V_F")):
    """Channel headline under A3 and under A3': the potentials in keys must agree (static and dynamical
    disagreeing -> undetermined); V_raw does not enter. Unequal channel: keys = ("V_F",) (Addendum 1 s. 1)."""
    out = {}
    for rule in ("A3", "A3prime"):
        vs = [verdicts.get(k, {}).get(rule, {}).get("verdict") for k in keys]
        out[rule] = vs[0] if (vs[0] is not None and all(v == vs[0] for v in vs)) else "undetermined"
    return out


def V_F_from_force(r, F, sF, tail=None):
    """V_F(r) = -int_r^{r_max} F dr' (+ tail(r_max)); trapezoid; sigma by propagation."""
    o = np.argsort(r)
    r, F, sF = (np.asarray(x, float)[o] for x in (r, F, sF))
    V = np.zeros_like(r)
    s2 = np.zeros_like(r)
    for i in range(r.size - 2, -1, -1):
        h = r[i + 1] - r[i]
        V[i] = V[i + 1] - 0.5 * h * (F[i] + F[i + 1])
        s2[i] = s2[i + 1] + (0.5 * h * sF[i]) ** 2 + (0.5 * h * sF[i + 1]) ** 2
    if tail is not None:
        V = V + tail
    return r, V, np.sqrt(s2 + (0.5 * (r[-1] - r[-2]) * sF[-1]) ** 2)


# ------------------------------------------------------------------------------------------- per-pair extraction
def load(name, results=OUT + "/results"):
    with open(_abs(os.path.join(results, name + ".json"))) as f:
        return json.load(f)


def single_record(res):
    j = res["job"]
    x = res["params"]["lumps"]["centers"][0][0]
    return dict(name=j["name"], x=x, box=j["box"], type=j["types"][0], res=res)


def pair_equal(res, singles, kappa, omega, domega_dN, E_lump):
    """V_N, V_N_box, F for one equal pair against the box's singles (list of single_record)."""
    j = res["job"]
    r = j["r"]
    w = window_for(res, r, kappa, omega)
    t1, t2 = w["t1"], w["t2"]
    pv = run_values(res, t1, t2)
    sv = [(s["x"], run_values(s["res"], t1, t2)) for s in singles]
    sv.sort(key=lambda q: q[0])
    x1 = res["params"]["lumps"]["centers"][0][0]
    si = interp_single(sv[0], sv[-1], x1) if len(sv) > 1 else sv[0][1]
    vn = V_N_equal(pv, si, domega_dN)
    tm = 0.5 * (t1 + t2)
    halves = []
    for a, b in ((t1, tm), (tm, t2)):
        pa = run_values(res, a, b)
        sa = [(s["x"], run_values(s["res"], a, b)) for s in singles]
        sa.sort(key=lambda q: q[0])
        sia = interp_single(sa[0], sa[-1], x1) if len(sa) > 1 else sa[0][1]
        halves.append(V_N_equal(pa, sia, domega_dN))
    shift = None
    if len(sv) > 1:
        (xa, a), (xb, b) = sv[0], sv[-1]
        shift = dict(x=[xa, xb], dE=b["E"] - a["E"], dE_box=b["E_box"] - a["E_box"], dN=b["N"] - a["N"],
                     V_shift=2 * ((b["E"] - a["E"]) - a["omega"] * (b["N"] - a["N"])),
                     V_shift_box=2 * ((b["E_box"] - a["E_box"]) - a["omega"] * (b["N"] - a["N"])))
    P = 2 * np.pi / omega
    td, d = w["t_d"], w["d"]
    c2, se2, rms = accel_fit(td, d, t1, t2, P)
    a_rel = 2 * c2 if c2 is not None else None
    F = -0.5 * E_lump * a_rel if a_rel is not None else None
    sF = 0.5 * E_lump * 2 * se2 if se2 is not None else None
    return dict(name=j["name"], r=r, dtheta=j["dtheta"], window=dict(t1=t1, t2=t2, moving=w["moving"], t_hit=w["t_hit"]),
                pair=pv, single=si, **vn, V_N_halves=[h["V_N"] for h in halves],
                V_N_box_halves=[h["V_N_box"] for h in halves], shift=shift, d_t1=float(np.interp(t1, td, d)),
                accel=a_rel, accel_se=(2 * se2 if se2 is not None else None), F=F, F_se=sF, d_fit_rms=rms,
                stopped_early=res.get("stopped_early"), t_end=res["t"])


def pair_unequal(res, singles1, singles2, lump1, lump2):
    """Unequal pair: V_N(t) amplitude against cos(dphi(t)) and the force amplitude (plan A1)."""
    j = res["job"]
    r = j["r"]
    om1, om2 = lump1["omega"], lump2["omega"]
    kmin = min(lump1["kappa"], lump2["kappa"])
    w = window_for(res, r, kmin, om1)
    t1, t2 = w["t1"], w["t2"]
    tph, dphi = relative_phase_series(res, om1)
    P = 2 * np.pi / om1
    lp = res["lumps"]["raw"]["point"]
    tp = np.array(lp["t"])
    Ih = np.array(lp["It_half"])[:, 0]
    It = np.array(res["point"]["I"])
    s = res["series"]
    Ep = np.interp(tp, s["t"], s["E_plus_absorbed"])

    def single_series(sl, x):
        """The box single nearest to the lump position x: per-step I, half-space I, E + absorbed."""
        q = min(sl, key=lambda s_: abs(s_["x"] - x))
        pp = q["res"]["point"]
        return (np.array(pp["I"]), np.array(q["res"]["lumps"]["raw"]["point"]["It_half"])[:, 0],
                np.interp(tp, q["res"]["series"]["t"], q["res"]["series"]["E_plus_absorbed"]))
    cen = res["params"]["lumps"]["centers"]
    I1, H1, E1 = single_series(singles1, cen[0][0])
    I2, H2, E2 = single_series(singles2, cen[1][0])
    n = min(tp.size, I1.size, I2.size, It.size)
    dN1 = (Ih[:n] - H1[:n] - H2[:n]) / om1
    dN2 = ((It[:n] - Ih[:n]) - (I1[:n] - H1[:n]) - (I2[:n] - H2[:n])) / om2
    dE = Ep[:n] - E1[:n] - E2[:n]
    VN = dE - om1 * dN1 - om2 * dN2
    ts, VNs = hann_smooth(tp[:n], VN, 3 * P)
    cph = np.cos(np.interp(ts, tph, dphi))
    sph = np.sin(np.interp(ts, tph, dphi))
    sel = (ts >= t1) & (ts <= t2)
    A = np.stack([np.ones(sel.sum()), cph[sel], sph[sel]], axis=1)
    cf, *_ = np.linalg.lstsq(A, VNs[sel], rcond=None)
    # force amplitude: d(t) = c0 + c1 t + c2 G(t), G'' = cos(dphi(t))
    td, d = w["t_d"], w["d"]
    g = np.cos(np.interp(td, tph, dphi))
    dt = td[1] - td[0]
    G1 = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * dt)])
    G = np.concatenate([[0.0], np.cumsum(0.5 * (G1[1:] + G1[:-1]) * dt)])
    c2, se2, rms = accel_fit(td, d, t1, t2, P, basis=G)
    E1l, E2l = lump1["E"], lump2["E"]
    F_amp = -c2 / (1 / E1l + 1 / E2l) if c2 is not None else None
    return dict(name=j["name"], r=r, dtheta=j["dtheta"], window=dict(t1=t1, t2=t2, moving=w["moving"]),
                delta_omega_measured=float(np.polyfit(tph[(tph >= t1) & (tph <= t2)], dphi[(tph >= t1) & (tph <= t2)], 1)[0]),
                V_N_amp_cos=float(cf[1]), V_N_amp_sin=float(cf[2]), V_N_mean=float(cf[0]),
                F_amp=F_amp, F_amp_se=(se2 / (1 / E1l + 1 / E2l) if se2 is not None else None), d_fit_rms=rms,
                beats_in_window=float((t2 - t1) * abs(om1 - om2) / (2 * np.pi)))


# ------------------------------------------------------------------------------------------- production
DOMEGA_DN = {"primary": -2.228e-5}


def domega_dN(t):
    """d omega / d N along the family at the type's profile omega (central difference of solved profiles)."""
    if t in DOMEGA_DN:
        return DOMEGA_DN[t]
    from swirl.session4.testA import driver as D
    from swirl.session4.testA.static import Lump
    ty = D.TYPES[t]
    h = 2e-4
    Np, Nm = Lump(ty["beta"], ty["omega"] + h).N, Lump(ty["beta"], ty["omega"] - h).N
    DOMEGA_DN[t] = 2 * h / (Np - Nm)
    return DOMEGA_DN[t]


def _static_channel(st, lab, key):
    r = np.array([q["r"] for q in st["rows"]])
    V = np.array([q[lab][key] for q in st["rows"]])
    s = np.array([q[lab]["sigma_" + key] for q in st["rows"]])
    F = np.array([q[lab]["F_" + key] for q in st["rows"]]) if ("F_" + key) in st["rows"][0][lab] else None
    return r, V, s, F


def analyze_equal(prefix, t, results, J, static_file, chans=(0, 180)):
    from swirl.session4.testA import driver as D
    n = D.numbers(t)
    have = {j["name"] for j in J if os.path.exists(_abs(os.path.join(results, j["name"] + ".json")))}
    singles = {}
    for j in J:
        if j["name"].startswith(prefix + "_single_") and j["name"] in have and not j.get("check"):
            singles.setdefault(j["box"], []).append(single_record(load(j["name"], results)))
    st = json.load(open(_abs(static_file)))
    om_s = None
    out = dict(type=t, numbers=n, domega_dN=domega_dN(t), channels={})
    for dth in chans:
        lab = "p%03d" % dth
        rows = []
        for j in J:
            if j["kind"] == "pair" and j["name"].startswith(prefix + "_r") and j["dtheta"] == dth and not j.get("check") \
                    and j["name"] in have and j["box"] in singles:
                res = load(j["name"], results)
                om_s = om_s or run_values(singles[j["box"]][0]["res"], T1, T_MAX)["omega"]
                rows.append(pair_equal(res, singles[j["box"]], n["kappa"], om_s, domega_dN(t), n["E"]))
        rows.sort(key=lambda q: q["r"])
        if not rows:
            continue
        r = np.array([q["r"] for q in rows])
        far = rows[-1]
        shift = {q["name"]: (q["shift"] or {}) for q in rows}
        ch = dict(rows=rows)
        for key in ("V_N", "V_N_box"):
            V = np.array([q[key] for q in rows])
            sh = np.array([abs(shift[q["name"]].get("V_shift" if key == "V_N" else "V_shift_box", 0.0)) for q in rows])
            half = np.array([abs(q[key + "_halves"][0] - q[key + "_halves"][1]) / 2 for q in rows])
            sig = np.maximum.reduce([sh, half, np.full(r.size, abs(far[key]))])
            ch[key] = dict(r=r.tolist(), V=V.tolist(), sigma=sig.tolist(), sigma_parts=dict(
                shift=sh.tolist(), half_window=half.tolist(), largest_r_value=abs(far[key])))
        F = np.array([q["F"] for q in rows])
        sF = np.maximum(np.array([q["F_se"] for q in rows]), abs(far["F"]))
        moving = np.array([q["window"]["moving"] for q in rows])
        ch["F"] = dict(r=r.tolist(), F=F.tolist(), sigma=sF.tolist(), moving=moving.tolist())
        ch["force_signs_dynamic"] = force_signs(r, F, sF)
        keep = ~moving & (r >= R_LO - 1e-9)
        rv, VF, sVF = V_F_from_force(r[keep], F[keep], sF[keep]) if keep.sum() >= 3 else (r, None, None)
        F16 = float(np.interp(16.0, r, F)) if r.min() <= 16 <= r.max() else np.nan
        verd = {}
        for key in ("V_N", "V_N_box"):
            m = ~moving
            verd[key] = verdicts_both(r[m], np.array(ch[key]["V"])[m], np.array(ch[key]["sigma"])[m], F16)
        if VF is not None:
            ch["V_F"] = dict(r=rv.tolist(), V=VF.tolist(), sigma=sVF.tolist(), tail="none")
            e2 = fit_form("E2", rv, VF, np.maximum(sVF, 1e-30)) if rv.size >= 3 else None
            if e2 and e2.get("ok"):
                tail = e2["params"]["C"] * np.exp(-e2["params"]["k"] * rv[-1]) / rv[-1]
                ch["V_F_with_tail"] = dict(V=(VF + tail).tolist(), tail_at_rmax=float(tail))
            verd["V_F"] = verdicts_both(rv[:-1], VF[:-1], sVF[:-1], F16)   # last point is 0 by construction
        rs, Us, sUs, FUs = _static_channel(st, lab, "U_static")
        verd["U_static"] = verdicts_both(rs, Us, sUs, float(np.interp(16.0, rs, FUs)))
        rs, Vr, sVr, FVr = _static_channel(st, lab, "V_raw")
        verd["V_raw"] = verdicts_both(rs, Vr, sVr, float(np.interp(16.0, rs, FVr)))
        ch["force_signs_static"] = force_signs(rs, FUs, 3e-30 + 0 * FUs)
        ch["verdicts"] = verd
        ch["headline"] = headline(verd)
        ch["U_tail_at_r"] = [float(-2 * np.pi * n["A"] ** 2 * np.cos(np.deg2rad(dth)) * np.exp(-n["kappa"] * x) / x) for x in r]
        ch["U_static_at_r"] = [float(np.interp(x, rs, Us)) for x in r]
        out["channels"][lab] = ch
    return out


def analyze_unequal(prefix, t1, t2, results, J, static_file):
    from swirl.session4.testA import driver as D
    n1, n2 = D.numbers(t1), D.numbers(t2)
    have = {j["name"] for j in J if os.path.exists(_abs(os.path.join(results, j["name"] + ".json")))}
    sing = {t1: {}, t2: {}}
    for j in J:
        if j["name"].startswith(prefix + "_single_") and j["name"] in have:
            sing[j["types"][0]].setdefault(j["box"], []).append(single_record(load(j["name"], results)))
    st = json.load(open(_abs(static_file))) if os.path.exists(_abs(static_file)) else None
    out = dict(types=[t1, t2], channels={})
    for dth in (0, 180):
        rows = []
        for j in J:
            if j["kind"] == "pair" and j["name"].startswith(prefix + "_r") and j["dtheta"] == dth and j["name"] in have \
                    and j["box"] in sing[t1] and j["box"] in sing[t2]:
                rows.append(pair_unequal(load(j["name"], results), sing[t1][j["box"]], sing[t2][j["box"]], n1, n2))
        rows.sort(key=lambda q: q["r"])
        if not rows:
            continue
        r = np.array([q["r"] for q in rows])
        F = np.array([q["F_amp"] for q in rows])
        sF = np.maximum(np.array([q["F_amp_se"] for q in rows]), abs(F[-1]))
        ch = dict(rows=rows, F_amp=dict(r=r.tolist(), F=F.tolist(), sigma=sF.tolist()),
                  force_signs_dynamic=force_signs(r, F, sF))
        verd = {}
        F16 = float(np.interp(16.0, r, F))
        if r.size >= 3:
            rv, VF, sVF = V_F_from_force(r, F, sF)
            ch["V_F_amp"] = dict(r=rv.tolist(), V=VF.tolist(), sigma=sVF.tolist())
            verd["V_F"] = verdicts_both(rv[:-1], VF[:-1], sVF[:-1], F16)
            VNa = np.array([q["V_N_amp_cos"] for q in rows])
            verd["V_N"] = verdicts_both(r, VNa, np.maximum(abs(VNa[-1]), 1e-30) + 0 * VNa, F16)
        if st:
            rs, Us, sUs, FUs = _static_channel(st, "p%03d" % dth, "U_static")
            verd["U_static"] = verdicts_both(rs, Us, sUs, float(np.interp(16.0, rs, FUs)))
            ch["U_amp_static_at_r"] = [float(np.interp(x, rs, Us)) for x in r]
            ch["U_law_at_r"] = [float(np.interp(x, rs, [q["p%03d" % dth]["U_law"] for q in st["rows"]])) for x in r]
            ch["omega_N_cross_amp_static_at_r"] = [float(np.interp(x, rs, [n1["omega"] * n2["omega"] * q["moments"]["I_bil"]
                                                                      for q in st["rows"]])) for x in r]
        ch["verdicts"] = verd
        ch["headline"] = headline(verd, keys=("V_F",))
        ch["note"] = ("V_N amplitude against cos(dphi) contains the beating overlap charge (about omega_1 omega_2 "
                      "int f_1 f_2, see omega_N_cross_amp_static_at_r), which is not the interaction; V_F (force "
                      "amplitude) is the clean route")
        out["channels"]["p%03d" % dth] = ch
    return out


def controls(results, J):
    """Box check (1.5x, r = 24), resolution check (dx = 0.40, r = 16): V_N and F against the production values;
    linear twins: energy inside 3 hw over time."""
    from swirl.session4.testA import driver as D
    n = D.numbers("primary")
    have = {j["name"] for j in J if os.path.exists(_abs(os.path.join(results, j["name"] + ".json")))}
    out = {}
    om = None
    for chk in ("box", "resolution"):
        pj = [j for j in J if j.get("check") == chk and j["kind"] == "pair"]
        sj = [j for j in J if j.get("check") == chk and j["kind"] == "single"]
        if not pj or pj[0]["name"] not in have or sj[0]["name"] not in have:
            out[chk] = "missing"
            continue
        res = load(pj[0]["name"], results)
        s = single_record(load(sj[0]["name"], results))
        om = run_values(s["res"], T1, T_MAX)["omega"]
        out[chk] = pair_equal(res, [s], n["kappa"], om, domega_dN("primary"), n["E"])
    for j in J:
        if j["kind"] == "twin" and j["name"] in have:
            res = load(j["name"], results)
            s = res["series"]
            E = np.array(s["E_loc"])[:, 0]
            out[j["name"]] = dict(t=s["t"][::10], E_in_3hw_over_E0=(E / s["E_total"][0])[::10].tolist(),
                                  final=float(E[-1] / s["E_total"][0]))
    return out


def analyze_all(results=OUT + "/results", jobs_file=OUT + "/jobs.json", out=OUT + "/analysis.json"):
    from swirl.session4.testA.static import dump
    J = json.load(open(_abs(jobs_file)))
    rec = dict(what="Test A analysis (plan A0 to A3)", definitions=__doc__,
               primary=analyze_equal("pA", "primary", results, J, OUT + "/static/static_primary.json"))
    un = json.load(open(_abs(OUT + "/unequal_choice.json")))["recommended"]
    rec["unequal"] = analyze_unequal("uA", "dip", "big", results, J,
                                     OUT + "/static/static_unequal_%.6f_%.6f.json" % (0.9065045072253043, 0.857370255463254))
    rec["unequal_choice"] = un["label"]
    if any(j["group"] == "b5" for j in J):
        rec["b5"] = analyze_equal("bA", "b5", results, J, OUT + "/static/static_b5.json", chans=(0,))
    rec["controls"] = controls(results, J)
    dump(rec, out)
    return rec


def static_verdicts(names=("primary", "b5", "unequal_0.906505_0.857370"), out=OUT + "/static/static_verdicts.json"):
    """A3 verdicts and A2 fits for the static potentials (U_static, V_raw; dtheta 0 and 180)."""
    from swirl.session4.testA.static import dump
    rec = {}
    for nm in list(names) + [n + "+ext" for n in names]:
        base = nm.replace("+ext", "")
        p = _abs(OUT + "/static/static_%s.json" % base)
        pe = _abs(OUT + "/static/static_%s_ext.json" % base)
        if not os.path.exists(p) or (nm.endswith("+ext") and not os.path.exists(pe)):
            continue
        st = json.load(open(p))
        if nm.endswith("+ext"):           # supplementary: the plan grid plus the extension beyond r = 45
            ste = json.load(open(pe))
            st = dict(st, rows=st["rows"] + ste["rows"])
            from swirl.session4.testA.static import add_derived
            add_derived(st)
        rec[nm] = {}
        for lab in ("p000", "p180"):
            for key in ("U_static", "V_raw"):
                r, V, s, F = _static_channel(st, lab, key)
                v = verdicts_both(r, V, s, float(np.interp(16.0, r, F)))
                rec[nm]["%s_%s" % (key, lab)] = v
    dump(rec, out)
    return rec


# ------------------------------------------------------------------------------------------- smoke
def smoke():
    from swirl.session4.testA import driver as D
    res = load("smoke_pA_r20_p000")
    sres = load("smoke_pA_single_primary_N144_x-10.0")
    n = D.numbers("primary")
    om_src = 0.909531801493928
    domega_dN = -2.228e-5
    single = single_record(sres)
    pe = pair_equal(res, [single], n["kappa"], om_src, domega_dN, n["E"])
    t0 = json.load(open(_abs(OUT + "/t0/smoke_pA_r20_p000.json")))["t0"]
    from swirl.session4.testA.static import U_tail, F_asym
    st = json.load(open(_abs(OUT + "/static/static_primary.json")))
    row = [q for q in st["rows"] if abs(q["r"] - 20.0) < 1e-9][0]["p000"]
    out = dict(pair=pe, t0=t0, U_tail=float(U_tail(20.0, n["A"], n["kappa"])), U_static=row["U_static"],
               V_raw=row["V_raw"], F_asym=float(F_asym(20.0, n["A"], n["kappa"])), F_static=row["F_U_static"],
               wall_seconds=dict(pair=res["wall_seconds"], single=sres["wall_seconds"]),
               steps=dict(pair=res["steps_done"], single=sres["steps_done"]), N=res["params"]["N"],
               E_plus_absorbed_max_rel_dev=dict(pair=res["summary"]["max_rel_dev_E_plus_absorbed"],
                                                single=sres["summary"]["max_rel_dev_E_plus_absorbed"]),
               regime=dict(omega_over_m=pe["single"]["omega"], nonlinear_share_t0=res["summary"]["nonlinear_share_0"],
                           nonlinear_share_mean=float(np.mean(res["series"]["nonlinear_share"]))))
    from swirl.session4.testA.static import dump
    dump(out, OUT + "/smoke_analysis.json")
    return out


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "all":
        analyze_all()
    elif sys.argv[1] == "static":
        static_verdicts()
    elif sys.argv[1] == "smoke":
        o = smoke()
        print(json.dumps({k: v for k, v in o.items() if k != "pair"}, indent=1, default=float))
        print(json.dumps({k: v for k, v in o["pair"].items() if k not in ("pair", "single")}, indent=1, default=float))
