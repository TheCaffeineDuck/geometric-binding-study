"""Test A analysis: window rule, V_F (headline), V_N, floors, the five fits, the A3 and A3' verdicts
(SESSION_04_PLAN A0, A2, A3; Addendum 1 section 2; Addendum 2 items 1 and 2).

Window (A0): t1 = 40; t2 = the earlier of 400, the run end less 1.5 periods, and the first time kappa |d(t) - r|
reaches 0.10, with d(t) the lump separation (per-lump energy centroids, Hann-smoothed over 3 periods); a run with
t2 - t1 < 35 is "moving" (reported, excluded from the long-range fits).

Averages (A0, Addendum 2): whole-period averages bounded by each run's own zero crossings of phi(lump-1 centre, t)
(cubic-refined, swirl.engine.run.zero_crossings) inside [t1, t2], an even number of half periods:
    omega = pi n / (z_b - z_a);  <I> = (C(z_b) - C(z_a)) / (z_b - z_a)  (C = per-step trapezoid integral of
    I = int phi_t^2, read by cubic Hermite interpolation with C' = I, as engine.run.window_averages);
    E, E_box: exact integral of the per-unit-time samples (piecewise linear) over [z_a, z_b];  N = <I> / omega.

V_N (equal lumps; reported, does not decide; Addendum 2 item 1):
    V_N = [E_pair - 2 E_s] - omega_s [N_pair - 2 N_s] - (d omega/d N) (N_pair - 2 N_s)^2 / 4
with E = E + absorbed (A0) and V_N_box with the box energy; their difference (= absorbed_pair - 2 absorbed_s)
is its own series. Singles are interpolated linearly in the lump-1 position between the two singles of the box.
sigma_VN = sqrt(sigma_omega^2 + V_shift^2): sigma_omega = the scatter of V_N over sliding whole-period
sub-windows of [t1, t2] (length L_s = max(2 periods, (t2 - t1)/3), half-period steps), scaled by sqrt(L_s/(t2 - t1));
V_shift = 2[(E_b - E_a) - omega (N_b - N_a)] between the box's two singles (the single against its own shifted copy).
sigma_VN is a LOWER BOUND: it omits the drift of omega_pair - omega_single between windows (1.15e-4 to 1.5e-4 in
the reviewer's anti-phase run) and the 3 omega-cloud systematic (+0.07 to +0.09 at r = 20, diag_surplus.json).

V_F (headline, Addendum 2 item 1): d(t) on [t1, t2] fitted by c0 + c1 t + c2 t^2 on the Hann-smoothed separation;
d'' = 2 c2; F = -(E_s/2) d'' with E_s the settled single's whole-period box energy (F > 0 attracts); F is labelled
with the window-mean separation r_eff. s.e. with n_eff = min(window/period, AR(1) effective count of the residuals).
Floor (isolated-single control): sigma_F^2 = se^2 + (E_s/2)^2 2 a_ctrl^2, a_ctrl = the largest |centroid
acceleration| (+ its s.e.) of the box's singles fitted on the same window.
V_F(r) = V_tail(r_max) - int_r^{r_max} F dr', the integral exact for log-linear |F| between neighbours of one sign
(trapezoid otherwise); V_tail(r_max) from a fit of F = sigma + Yukawa force [-C e^{-kr}(k/r + 1/r^2)] to the
resolved points (sigma kept only if > 5 s.e.), V_tail = sigma r_max + C e^{-k r_max}/r_max. The no-tail V_F is
also reported.
No force: when [t1, t2] holds fewer than 6 smoothed d samples (the smoothed series starts at 1.5 P, and near overlap
kappa |d - r| >= 0.10 is reached there, t2 = t_hit = 1.5 P), the row carries F = None, force_excluded and the reason;
such rows are left out of the force route (F, V_F, force signs, F16) and listed per channel under force_excluded.
Unequal lumps (A1, Addendum 2): d(t) = c0 + c1 t + c2 G(t) + c3 t^2, G'' = cos(dphi(t)) with dphi the measured
relative phase: F_amp = -c2 / (1/E_1 + 1/E_2) and the mean force F_mean = -2 c3 / (1/E_1 + 1/E_2); rows that are
"moving" or whose window holds less than half a beat (and Delta omega t2 >= 0.3) are excluded from the fits.
V_N(t) amplitude (half-space N per lump) against cos(dphi): reported, dominated by the beating overlap charge.

Fits (A2): E1 C e^{-kr}; E2 C e^{-kr}/r; G C e^{-(r/s)^2}; P C r^{-n}; L V_inf + sigma r + C e^{-kr}/r; weighted
least squares, reduced chi^2, AIC = chi^2 + 2p, BIC = chi^2 + p ln n, runs test. Fit L: if the covariance is
singular (e.g. C -> 0), the full-fit s.e. are NaN and reported as such, and sigma's s.e. is taken from the refit
with k fixed (linear in V_inf, sigma, C), flagged "k_fixed".
Verdicts: A3 (section 2) and A3' (Addendum 1), both with Addendum 2's corrections: grows-clause 2 holds on V and,
for V_F, |F(r_hi)| resolved; grows-clause 3 also needs fit L to fit (reduced chi^2 <= 3; reviewer's fit-quality
point on Addendum 2's corrections); a clause that
cannot be evaluated (NaN s.e.) is None and the verdict is "undetermined" with the reason.
Standard 1 per run: omega/m and the nonlinear share (t = 0 and window mean).
"""
import json
import os

import numpy as np
from scipy.interpolate import CubicHermiteSpline
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
    """Zero crossings (cubic refinement, as the engine)."""
    from swirl.engine.run import zero_crossings
    return zero_crossings(np.asarray(t, float), np.asarray(y, float))


def _crossings(res):
    if "_z" not in res:
        res["_z"] = zero_cross(res["point"]["t"], res["point"]["phi"])
    return res["_z"]


def _hermite(res):
    if "_H" not in res:
        p = res["point"]
        res["_H"] = CubicHermiteSpline(np.array(p["t"]), np.array(p["C"]), np.array(p["I"]))
    return res["_H"]


def wp_bounds(z, t1, t2):
    """(z_a, z_b, half periods) between zero crossings inside [t1, t2], even number of half periods (>= 2)."""
    z = z[(z >= t1) & (z <= t2)]
    n = z.size - 1
    if n % 2:
        z, n = z[:-1], n - 1
    if n < 2:
        return None
    return float(z[0]), float(z[-1]), int(n)


def omega_window(t, y, t1, t2):
    b = wp_bounds(zero_cross(t, y), t1, t2)
    return None if b is None else float(np.pi * b[2] / (b[1] - b[0]))


def lin_avg(t, y, a, b):
    """Exact mean over [a, b] of the piecewise-linear interpolant of samples (t, y)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    inner = (t > a) & (t < b)
    tt = np.concatenate([[a], t[inner], [b]])
    yy = np.concatenate([[np.interp(a, t, y)], y[inner], [np.interp(b, t, y)]])
    return float(np.trapezoid(yy, tt) / (b - a))


def separation(res):
    ls = res["lumps"]["raw"]["series"]
    return np.array(ls["t"]), np.array(ls["X"])


def window_for(res, r, kappa, omega):
    """(t1, t2, moving, smoothed d-series) per the A0 rule."""
    t, X = separation(res)
    P = 2 * np.pi / omega
    ts, Xs = hann_smooth(t, X.reshape(t.size, -1), 3 * P)
    Xs = Xs.reshape(ts.size, -1, 3)
    d = np.linalg.norm(Xs[:, 1, :] - Xs[:, 0, :], axis=1) if Xs.shape[1] > 1 else None
    t_end = min(T_MAX, float(res["t"]))
    hit = np.nonzero((kappa * np.abs(d - r) >= KAPPA_SHIFT) & (ts > 0))[0] if d is not None else np.array([], int)
    t2 = min(t_end - 1.5 * P, float(ts[hit[0]]) if hit.size else np.inf)
    t1 = T1
    moving = (t2 - t1) < MOVING_MIN
    if t2 - t1 < 2 * P:          # short window: the last 2 periods before t2 (flagged "moving")
        t1 = max(0.0, t2 - 2 * P)
    return dict(t1=float(t1), t2=float(t2), moving=bool(moving), t_d=ts, d=d, X=Xs,
                t_hit=float(ts[hit[0]]) if hit.size else None)


def run_values(res, t1, t2):
    """Whole-period E + absorbed, E_box, absorbed, <I>, omega, N over the run's own crossings inside [t1, t2]."""
    b = wp_bounds(_crossings(res), t1, t2)
    if b is None:
        return None
    za, zb, n = b
    H = _hermite(res)
    I = float((H(zb) - H(za)) / (zb - za))
    om = float(np.pi * n / (zb - za))
    s = res["series"]
    out = dict(za=za, zb=zb, half_periods=n, omega=om, I=I, N=I / om,
               E=lin_avg(s["t"], s["E_plus_absorbed"], za, zb), E_box=lin_avg(s["t"], s["E_total"], za, zb),
               absorbed=lin_avg(s["t"], s["absorbed"], za, zb))
    lp = res["lumps"]["raw"]["point"]
    if lp.get("It_half"):
        tp = np.array(lp["t"])
        out["I_half"] = lin_avg(tp, np.array(lp["It_half"])[:, 0], za, zb)
    return out


def standard1(res, t1, t2, omega=None):
    s = res["series"]
    t = np.array(s["t"])
    sel = (t >= t1) & (t <= t2)
    return dict(omega_over_m=(omega / res["params"]["m"]) if omega else None,
                nonlinear_share_t0=res["summary"]["nonlinear_share_0"],
                nonlinear_share_window_mean=float(np.mean(np.array(s["nonlinear_share"])[sel])) if sel.any() else None)


def interp_single(sA, sB, x):
    """Linear interpolation in the lump position x between singles (x_A, vals_A), (x_B, vals_B)."""
    (xa, a), (xb, b) = sA, sB
    if abs(xb - xa) < 1e-12:
        return dict(a)
    u = (x - xa) / (xb - xa)
    return {k: (a[k] + u * (b[k] - a[k]) if isinstance(a.get(k), float) and isinstance(b.get(k), float) else a.get(k))
            for k in a}


def V_N_equal(pv, sv, domega_dN):
    dN = pv["N"] - 2 * sv["N"]
    om = sv["omega"]
    second = -domega_dN * dN * dN / 4.0
    VN = (pv["E"] - 2 * sv["E"]) - om * dN + second
    VB = (pv["E_box"] - 2 * sv["E_box"]) - om * dN + second
    return dict(V_N=VN, V_N_box=VB, V_N_minus_box=VN - VB, dE=pv["E"] - 2 * sv["E"], dE_box=pv["E_box"] - 2 * sv["E_box"],
                dN=dN, omega0=om, second_order=second)


def accel_fit(t, d, t1, t2, period, extra=None):
    """Least squares of d(t) on [t1, t2] by c0 + c1 tc + c2 tc^2 (+ extra basis columns); returns coefficient
    vector, s.e. vector and residual rms. s.e. uses n_eff = min(window/period, n (1 - rho)/(1 + rho)), rho the
    lag-1 autocorrelation of the residuals."""
    t, d = np.asarray(t, float), np.asarray(d, float)
    s = (t >= t1) & (t <= t2)
    tt, dd = t[s], d[s]
    if tt.size < 6:
        return None, None, None
    tc = tt - 0.5 * (t1 + t2)
    cols = [np.ones_like(tc), tc, tc ** 2] + ([e[s] for e in extra] if extra is not None else [])
    A = np.stack(cols, axis=1)
    c, *_ = np.linalg.lstsq(A, dd, rcond=None)
    res = dd - A @ c
    n = tt.size
    rho = float(np.corrcoef(res[1:], res[:-1])[0, 1]) if n > 3 and np.std(res) > 0 else 0.0
    n_ar = n * (1 - rho) / (1 + rho) if rho < 0.999 else 1.0
    n_eff = max(min((t2 - t1) / period, n_ar), 3.0)
    s2 = float(np.sum(res ** 2) / max(n - A.shape[1], 1)) * n / n_eff
    try:
        cov = s2 * np.linalg.inv(A.T @ A)
        se = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        se = np.full(A.shape[1], np.nan)
    return c, se, float(np.sqrt(np.mean(res ** 2)))


def centroid_accel(res, t1, t2, omega, lump=0):
    """Acceleration (2 c2) and its s.e. of one lump's smoothed centroid x(t) on [t1, t2] (control floor)."""
    t, X = separation(res)
    P = 2 * np.pi / omega
    ts, Xs = hann_smooth(t, X[:, lump, 0], 3 * P)
    c, se, _ = accel_fit(ts, Xs, t1, t2, P)
    return (None, None) if c is None else (2 * float(c[2]), 2 * float(se[2]))


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
        return s * np.exp(np.minimum(q[0] - (r / q[1]) ** 2, 700.0))
    if form == "P":
        return s * np.exp(q[0] - q[1] * np.log(r))
    if form == "L":
        return q[0] + q[1] * r + q[2] * np.exp(-q[3] * r) / r
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


COND_MAX = 1e13


def _cov(J):
    """(J^T J)^-1, or None when it is singular: the condition number of the column-normalised J^T J above
    COND_MAX (column scaling removes the parameters' units, so only true degeneracy trips it)."""
    J = np.asarray(J, float)
    nrm = np.linalg.norm(J, axis=0)
    if not np.all(np.isfinite(J)) or np.any(nrm == 0):
        return None, np.inf
    Js = J / nrm
    JTJ = Js.T @ Js
    try:
        cnd = float(np.linalg.cond(JTJ))
        if not np.isfinite(cnd) or cnd > COND_MAX:
            return None, cnd
        return np.linalg.inv(JTJ) / np.outer(nrm, nrm), cnd
    except np.linalg.LinAlgError:
        return None, np.inf


def _L_kfixed(r, V, sig, k):
    """Fit L with k fixed: linear weighted least squares in (V_inf, sigma, C)."""
    A = np.stack([np.ones_like(r), r, np.exp(-k * r) / r], axis=1) / sig[:, None]
    b = V / sig
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    try:
        cov = np.linalg.inv(A.T @ A)
        se = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        se = np.full(3, np.nan)
    chi2 = float(np.sum((b - A @ c) ** 2))
    return dict(V_inf=float(c[0]), sigma=float(c[1]), C=float(c[2]), k=float(k)), \
        dict(V_inf=float(se[0]), sigma=float(se[1]), C=float(se[2]), k=0.0), chi2


def fit_form(form, r, V, sig, k_ref=None):
    """Weighted least squares of one candidate. Returns params (physical), s.e., chi2, reduced, AIC, BIC, runs.
    Fit L with a singular covariance: full s.e. NaN (reported) and sigma's s.e. from the k-fixed refit."""
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
        ks = [k for k in (k_ref, -c[1]) if k is not None and np.isfinite(k) and k > 1e-3] or [0.4]
        for k0 in ks:
            for kf in (1.0, 0.8, 1.25, 0.5):
                pk, _, _ = _L_kfixed(r, V, sig, k0 * kf)
                starts.append([pk["V_inf"], pk["sigma"], pk["C"], k0 * kf])
    np_ = len(starts[0])
    best = None
    err = np.seterr(over="ignore", invalid="ignore", divide="ignore")   # rejected trial steps may overflow
    for q0 in starts:
        try:
            f = lambda q: (_model(form, r, q, s) - V) / sig
            if form == "L":
                vs = max(np.abs(V).max(), 1e-300)
                sc = np.array([vs, vs / max(r.max(), 1.0), max(abs(q0[2]), vs * r.min()), 0.05])
                sol = least_squares(f, q0, x_scale=sc, method="lm", max_nfev=20000, xtol=1e-15, ftol=1e-15)
            else:
                # decaying forms: decay constant k >= 0 (E1, E2), n >= 0 (P), s > 0 (G)
                lo = [-np.inf, 0.0] if form in ("E1", "E2", "P") else [-np.inf, 1e-9]
                q0 = [q0[0], max(q0[1], 1e-6)]
                sol = least_squares(f, q0, method="trf", bounds=(lo, [np.inf, np.inf]), max_nfev=20000,
                                    xtol=1e-15, ftol=1e-15, gtol=1e-15)
        except Exception:
            continue
        if not np.all(np.isfinite(sol.fun)):
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    np.seterr(**err)
    if best is None:
        return dict(form=form, ok=False)
    q = best.x
    res = best.fun
    chi2 = float(np.sum(res ** 2))
    n = r.size
    dof = n - np_
    cov_q, cnd = _cov(best.jac)
    se_q = np.sqrt(np.abs(np.diag(cov_q))) if cov_q is not None else np.full(np_, np.nan)
    extra = {}
    if form == "L":
        params = dict(V_inf=q[0], sigma=q[1], C=q[2], k=q[3])
        se = dict(V_inf=se_q[0], sigma=se_q[1], C=se_q[2], k=se_q[3])
        if cov_q is None:
            kk = k_ref if (k_ref is not None and np.isfinite(k_ref) and k_ref > 0) else (q[3] if q[3] > 0 else 0.4)
            pk, sk, chik = _L_kfixed(r, V, sig, kk)
            extra = dict(cov_singular=True, se_full=dict(se), k_fixed=kk, params_kfixed=pk, se_kfixed=sk,
                         chi2_kfixed=chik)
            se = dict(se, sigma=sk["sigma"])          # sigma's s.e. from the k-fixed refit (flagged)
    else:
        C = s * np.exp(q[0])
        key = {"E1": "k", "E2": "k", "G": "s", "P": "n"}[form]
        params = {"C": C, key: q[1]}
        se = {"C": abs(C) * se_q[0], key: se_q[1]}
    return dict(form=form, ok=bool(best.success), params={k: float(v) for k, v in params.items()},
                se={k: float(v) for k, v in se.items()}, chi2=chi2, dof=dof, cond=cnd,
                chi2_red=(chi2 / dof if dof > 0 else None), AIC=chi2 + 2 * np_, BIC=chi2 + np_ * np.log(n), n=n,
                runs=runs_test(res), residual_over_sigma=[float(x) for x in res], **extra)


def fit_all(r, V, sig):
    out = {f: fit_form(f, r, V, sig) for f in DECAYING}
    k_ref = out["E2"]["params"]["k"] if out["E2"].get("ok") else None
    out["L"] = fit_form("L", r, V, sig, k_ref=k_ref)
    return out


# ------------------------------------------------------------------------------------------- verdicts
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


def _not_decreasing_last3(x, s):
    """|x| not decreasing over the last three points, read for measured data as "not significantly decreasing":
    |x_3| >= |x_1| - 2 sqrt(s_1^2 + s_3^2) (with s ~ 1e-11 on the static data this is the strict test)."""
    a = np.abs(np.asarray(x, float)[-3:])
    e = np.asarray(s, float)[-3:]
    return bool(a.size == 3 and a[2] >= a[0] - 2 * np.hypot(e[0], e[2]))


def _grows2(VV, ss, F, sF):
    """Grows-clause 2 (Addendum 2 item 2): on V (resolved and not decreasing over the last three points), and,
    when the force is given, |F(r_hi)| > 3 sigma_F (no monotone test on F: for sigma + Yukawa |F| falls toward
    sigma, so a monotone test would block exactly the confining case)."""
    gV = bool(abs(VV[-1]) > 3 * ss[-1] and _not_decreasing_last3(VV, ss))
    if F is None:
        return gV, dict(on_V=gV)
    gF = bool(abs(F[-1]) > 3 * sF[-1])          # on F: resolved at r_hi only (|F| of sigma + Yukawa falls toward sigma)
    return gV and gF, dict(on_V=gV, on_F=gF)


def _L_sigma(L):
    if not L.get("ok"):
        return np.nan, np.nan, "fit L failed"
    sg, se = L["params"]["sigma"], L["se"]["sigma"]
    note = "k fixed (full covariance singular)" if L.get("cov_singular") else None
    if not np.isfinite(se):
        note = "sigma s.e. is NaN"
    return sg, se, note


def verdict(r, V, sig, F16, r_lo=R_LO, F=None, sF=None):
    """A3 verdict (section 2) with Addendum 2's corrections. r, V, sig: the sampled points (moving runs excluded);
    F16: the measured |F| at r = 16; F, sF (optional, same points): the force, for grows-clause 2."""
    r, V, sig = (np.asarray(x, float) for x in (r, V, sig))
    rng, res = resolved_range(r, V, sig, r_lo)
    out = dict(rule="A3", r_lo=r_lo, n_fit=int(rng.size), r_hi=(float(r[rng[-1]]) if rng.size else None))
    if rng.size < 5:
        out.update(verdict="undetermined", reason="fewer than 5 resolved points from r_lo", fits=None)
        return out
    rr, VV, ss = r[rng], V[rng], sig[rng]
    FF = None if F is None else np.asarray(F, float)[rng]
    sFF = None if sF is None else np.asarray(sF, float)[rng]
    fits = fit_all(rr, VV, ss)
    resolved_all = np.nonzero(res)[0]
    fits_full = fit_all(r[resolved_all], V[resolved_all], sig[resolved_all]) if resolved_all.size >= 5 else None
    fall = float(abs(VV[0]) / abs(VV[-1]))
    below = bool(np.any(np.abs(V) < 3 * sig))
    decades = float(np.log10(np.max(np.abs(V[res])) / np.min(np.abs(V[res])))) if res.any() else 0.0
    sg, se, note = _L_sigma(fits["L"])
    red = {f: fits[f].get("chi2_red") for f in DECAYING}
    good = [f for f in DECAYING if red[f] is not None and red[f] <= 3.0]
    aic = {f: fits[f]["AIC"] for f in fits if fits[f].get("ok")}
    L_red = fits["L"].get("chi2_red")
    se_ok = bool(np.isfinite(se) and se > 0)
    d1 = fall >= 100.0 and below
    d2 = bool(abs(sg) <= 3 * se and (abs(sg) + 3 * se) < 0.01 * abs(F16)) if se_ok else None
    d3 = bool(good)
    g1 = bool(abs(sg) > 5 * se and sg > 0) if se_ok else None
    g2, g2_parts = _grows2(VV, ss, FF, sFF)
    g3 = bool("L" in aic and all(aic["L"] + 10.0 < aic[f] for f in DECAYING if f in aic)
              and L_red is not None and L_red <= 3.0)
    reason = None
    if d1 and d2 and d3:
        v = "decays to zero"
    elif g1 and g2 and g3:
        v = "grows"
    else:
        v = "undetermined"
        if se_ok is False:
            reason = "fit L sigma s.e. not available (%s)" % note
    order = sorted(aic, key=aic.get)
    pref = order[0] if len(order) > 1 and aic[order[0]] + 10.0 < aic[order[1]] else None
    k_fit = {f: fits[f]["params"].get("k") for f in ("E1", "E2", "L") if fits[f].get("ok")}
    out.update(verdict=v, reason=reason, clauses=dict(
        decays=dict(fall_factor=fall, fall_ge_100=fall >= 100.0, below_3sigma_somewhere=below, L_sigma=sg, L_sigma_se=se,
                    L_sigma_note=note, L_bound=abs(sg) + 3 * se if se_ok else None, F16=abs(F16),
                    L_sigma_within_3se_and_bound_below_1pct_F16=d2, decaying_forms_chi2red_le_3=good),
        grows=dict(L_sigma_over_se=(sg / se if se_ok else None), sigma_confining_gt_5se=g1,
                   V_hi_resolved_and_not_decreasing=g2, clause2_parts=g2_parts,
                   L_beats_decaying_by_dAIC_10_and_L_chi2red_le_3=g3, L_chi2_red=L_red),
        decades_resolved=decades), preferred_form=pref, AIC_order=order, chi2_red=red,
        k_fit=k_fit, fits=fits, fits_full_range=fits_full)
    if decades < 2.0 and v != "undetermined":
        out["verdict"], out["reason"] = "undetermined", "fewer than two decades of resolved |V|"
    return out


A3P_MODEL_FRAC = 0.02
A3P_R_LOS = (12.0, 16.0)
A3P_DECIDING_R_LO = 16.0


def verdict_prime(r, V, sig, F16, r_lo=A3P_DECIDING_R_LO, F=None, sF=None):
    """A3' (SESSION_04_PLAN Addendum 1 section 2) with Addendum 2's corrections, scored next to A3.

    sigma_eff = sqrt(sigma_V^2 + (0.02 V)^2) weights the fits; "resolved" is still |V| > 3 sigma_V.
    Decays: (1) |V| falls >= 100x over [r_lo, r_hi] and is still falling over the last three resolved points
    (reaching the floor is reported, not required); (2) the 3-s.e. upper bound on |sigma| (fit L) < 1% |F(16)|;
    (3) some decaying form has reduced chi^2 <= 3 under sigma_eff.
    Grows: (1) sigma confining (> 0), > 5 s.e., and |sigma| > 1% |F(16)|; (2) grows-clause 2 (V, and |F|);
    (3) L beats every decaying form by Delta AIC > 10 and fits (reduced chi^2 <= 3, a fit-quality guard added with
    Addendum 2's corrections). Undetermined otherwise.
    Form preference among E1, E2, G, P only (> 10)."""
    r, V, sig = (np.asarray(x, float) for x in (r, V, sig))
    rng, res = resolved_range(r, V, sig, r_lo)
    out = dict(rule="A3'", r_lo=r_lo, n_fit=int(rng.size), r_hi=(float(r[rng[-1]]) if rng.size else None),
               model_frac=A3P_MODEL_FRAC)
    if rng.size < 5:
        out.update(verdict="undetermined", reason="fewer than 5 resolved points from r_lo", fits=None)
        return out
    rr, VV, ss = r[rng], V[rng], sig[rng]
    FF = None if F is None else np.asarray(F, float)[rng]
    sFF = None if sF is None else np.asarray(sF, float)[rng]
    se_eff = np.sqrt(ss ** 2 + (A3P_MODEL_FRAC * VV) ** 2)
    fits = fit_all(rr, VV, se_eff)
    fall = float(abs(VV[0]) / abs(VV[-1]))
    last3 = np.abs(VV[-3:])
    still_falling = bool(last3[0] > last3[1] > last3[2])
    below = bool(np.any(np.abs(V) < 3 * sig))
    sg, se, note = _L_sigma(fits["L"])
    se_ok = bool(np.isfinite(se) and se > 0)
    red = {f: fits[f].get("chi2_red") for f in DECAYING}
    good = [f for f in DECAYING if red[f] is not None and red[f] <= 3.0]
    aic = {f: fits[f]["AIC"] for f in fits if fits[f].get("ok")}
    L_red = fits["L"].get("chi2_red")
    bound = abs(sg) + 3 * se if se_ok else None
    d1 = fall >= 100.0 and still_falling
    d2 = bool(bound < 0.01 * abs(F16)) if se_ok else None
    d3 = bool(good)
    g1 = bool(sg > 0 and abs(sg) > 5 * se and abs(sg) > 0.01 * abs(F16)) if se_ok else None
    g2, g2_parts = _grows2(VV, ss, FF, sFF)
    g3 = bool("L" in aic and all(aic["L"] + 10.0 < aic[f] for f in DECAYING if f in aic)
              and L_red is not None and L_red <= 3.0)
    v = "decays to zero" if (d1 and d2 and d3) else ("grows" if (g1 and g2 and g3) else "undetermined")
    reason = None if (v != "undetermined" or se_ok) else "fit L sigma s.e. not available (%s)" % note
    dec = sorted([f for f in DECAYING if f in aic], key=aic.get)
    pref = dec[0] if len(dec) > 1 and aic[dec[0]] + 10.0 < aic[dec[1]] else None
    for f in fits.values():
        f.pop("residual_over_sigma", None)
    out.update(verdict=v, reason=reason, clauses=dict(
        decays=dict(fall_factor=fall, fall_ge_100=fall >= 100.0, still_falling_last3=still_falling,
                    below_3sigma_somewhere_reported=below, L_sigma=sg, L_sigma_se=se, L_sigma_note=note,
                    L_bound_3se=bound, one_pct_F16=0.01 * abs(F16), L_bound_below_1pct_F16=d2,
                    decaying_forms_chi2red_le_3=good),
        grows=dict(L_sigma_over_se=(sg / se if se_ok else None), sigma_confining_5se_and_above_1pct_F16=g1,
                   V_hi_resolved_and_not_decreasing=g2, clause2_parts=g2_parts,
                   L_beats_decaying_by_dAIC_10_and_L_chi2red_le_3=g3, L_chi2_red=L_red)),
        preferred_form=pref, decaying_AIC_order=dec, chi2_red=red,
        k_fit={f: fits[f]["params"].get("k") for f in ("E1", "E2", "L") if fits[f].get("ok")}, fits=fits)
    return out


def verdicts_both(r, V, sig, F16, F=None, sF=None):
    """A3 (r_lo = 12) and A3' (r_lo = 12 and 16; 16 decides) for one potential and channel."""
    a3 = verdict(r, V, sig, F16, r_lo=R_LO, F=F, sF=sF)
    for f in (a3.get("fits") or {}).values():
        f.pop("residual_over_sigma", None)
    for f in (a3.get("fits_full_range") or {}).values():
        f.pop("residual_over_sigma", None)
    ap = {"r_lo_%g" % rl: verdict_prime(r, V, sig, F16, r_lo=rl, F=F, sF=sF) for rl in A3P_R_LOS}
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


def headline(verdicts, key="V_F", static_key="U_static"):
    """Addendum 2 item 1: the headline per channel is V_F's verdict (A3 and A3'); U_static's is reported beside it
    and a disagreement is flagged as a finding."""
    out = {}
    for rule in ("A3", "A3prime"):
        v = verdicts.get(key, {}).get(rule, {}).get("verdict", "undetermined")
        s = verdicts.get(static_key, {}).get(rule, {}).get("verdict")
        out[rule] = dict(verdict=v, U_static=s, disagree_with_static=(s is not None and s != v))
    return out


# ------------------------------------------------------------------------------------------- V_F
def integrate_force(r, F):
    """I_i = int_{r_i}^{r_max} F dr: exact for log-linear |F| between neighbours of one sign (trapezoid otherwise)."""
    r, F = np.asarray(r, float), np.asarray(F, float)
    I = np.zeros_like(r)
    for i in range(r.size - 2, -1, -1):
        h = r[i + 1] - r[i]
        a, b = F[i], F[i + 1]
        if a * b > 0 and abs(np.log(a / b)) > 1e-10:
            seg = (a - b) * h / np.log(a / b)
        else:
            seg = 0.5 * h * (a + b)
        I[i] = I[i + 1] + seg
    return I


def fit_force_tail(r, F, sF):
    """Fit F = sigma + Yukawa force C_F e^{-kr}(k/r + 1/r^2) (free-sign C_F, k bounded to [0.01, 3], multi-start
    from k-fixed linear solutions) to the given points; sigma kept only if > 5 s.e. Returns the tail
    V(r_max) = sigma r_max + C e^{-k r_max}/r_max (C = -C_F, so dV/dr = F) and the fit."""
    r, F, sF = (np.asarray(x, float) for x in (r, F, sF))
    shape = lambda k: np.exp(-k * r) * (k / r + 1 / r ** 2)
    best = {}
    for with_sigma in (False, True):
        if with_sigma and r.size < 4:
            continue
        sols = []
        for k0 in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6):
            cols = [shape(k0)] + ([np.ones_like(r)] if with_sigma else [])
            Am = np.stack(cols, axis=1) / sF[:, None]
            c0, *_ = np.linalg.lstsq(Am, F / sF, rcond=None)
            q0 = [c0[0] * np.exp(-k0 * r.min()), k0] + ([c0[1]] if with_sigma else [])

            def f(q, ws=with_sigma):          # amplitude at r_min for conditioning: C_F = q0 e^{k r_min}
                m = q[0] * np.exp(-q[1] * (r - r.min())) * (q[1] / r + 1 / r ** 2)
                return (m + (q[2] if ws else 0.0) - F) / sF
            lo = [-np.inf, 0.01] + ([-np.inf] if with_sigma else [])
            hi = [np.inf, 3.0] + ([np.inf] if with_sigma else [])
            try:
                sol = least_squares(f, q0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=5000)
            except Exception:
                continue
            if np.all(np.isfinite(sol.fun)):
                sols.append(sol)
        if not sols:
            continue
        sol = min(sols, key=lambda z: z.cost)
        cov, _ = _cov(sol.jac)
        se = np.sqrt(np.abs(np.diag(cov))) if cov is not None else np.full(len(sol.x), np.nan)
        best[with_sigma] = (sol, se)
    if not best:
        return 0.0, dict(ok=False)
    use_sigma = True in best and np.isfinite(best[True][1][2]) and abs(best[True][0].x[2]) > 5 * best[True][1][2]
    sol, se = best[True] if use_sigma else best[False]
    q = sol.x
    k = float(q[1])
    C_F = float(q[0] * np.exp(k * r.min()))
    C_U = -C_F
    sigma = float(q[2]) if use_sigma else 0.0
    rm = r.max()
    tail = sigma * rm + q[0] * -1.0 * np.exp(-k * (rm - r.min())) / rm
    return float(tail), dict(ok=True, C=C_U, k=k, sigma=sigma, sigma_used=bool(use_sigma),
                             sigma_fit=(float(best[True][0].x[2]) if True in best else None),
                             sigma_fit_se=(float(best[True][1][2]) if True in best else None),
                             chi2=float(np.sum(sol.fun ** 2)), n=int(r.size))


def V_F_from_force(r, F, sF, tail=True):
    """V_F(r) = V_tail(r_max) - int_r^{r_max} F dr' (exact for piecewise exponential F); sigma by propagation of
    the per-point force errors (each point owns half of each neighbouring interval). tail: fit the tail
    (fit_force_tail), a number, or None for V_F(r_max) = 0. Returns r, V, sigma, tail record."""
    o = np.argsort(r)
    r, F, sF = (np.asarray(x, float)[o] for x in (r, F, sF))
    I = integrate_force(r, F)
    w = np.zeros_like(r)
    h = np.diff(r)
    w[:-1] += 0.5 * h
    w[1:] += 0.5 * h
    s2 = np.array([np.sum((w[i:] * sF[i:]) ** 2) for i in range(r.size)])
    rec = None
    if tail is True:
        V0, rec = fit_force_tail(r, F, sF)
    elif tail is None:
        V0 = 0.0
    else:
        V0 = float(tail)
    return r, V0 - I, np.sqrt(s2), rec


# ------------------------------------------------------------------------------------------- per-pair extraction
def load(name, results=OUT + "/results"):
    with open(_abs(os.path.join(results, name + ".json"))) as f:
        return json.load(f)


def single_record(res):
    j = res["job"]
    x = res["params"]["lumps"]["centers"][0][0]
    return dict(name=j["name"], x=x, box=j["box"], type=j["types"][0], res=res)


def _singles_at(singles, t1, t2, x1):
    sv = sorted([(s["x"], run_values(s["res"], t1, t2)) for s in singles], key=lambda q: q[0])
    sv = [q for q in sv if q[1] is not None]
    if not sv:
        return None, sv
    return (interp_single(sv[0], sv[-1], x1) if len(sv) > 1 else sv[0][1]), sv


def sigma_omega_VN(res, singles, t1, t2, x1, domega_dN, P):
    """Scatter of V_N (both versions) over sliding whole-period sub-windows, scaled to the full window."""
    L = t2 - t1
    Ls = max(2 * P, L / 3.0)
    starts = np.arange(t1, t2 - Ls + 1e-9, 0.5 * P)
    vals = []
    for a in starts:
        pv = run_values(res, a, a + Ls)
        si, _ = _singles_at(singles, a, a + Ls, x1)
        if pv is None or si is None:
            continue
        v = V_N_equal(pv, si, domega_dN)
        vals.append((v["V_N"], v["V_N_box"], pv["omega"] - si["omega"]))
    if len(vals) < 3:
        return dict(n_sub=len(vals), sigma_V_N=None, sigma_V_N_box=None)
    a = np.array(vals)
    f = np.sqrt(Ls / L)
    return dict(n_sub=len(vals), sub_length=Ls, lower_bound=True,
                note="omits the omega_pair - omega_single drift between windows and the 3 omega-cloud systematic "
                     "(+0.07 to +0.09 at r = 20)",
                sigma_V_N=float(np.std(a[:, 0]) * f),
                sigma_V_N_box=float(np.std(a[:, 1]) * f), eps_scatter=float(np.std(a[:, 2])),
                eps_range=[float(a[:, 2].min()), float(a[:, 2].max())])


def pair_equal(res, singles, kappa, omega, domega_dN):
    """V_N, V_N_box (whole-period), F (V_F route) with control floors, for one equal pair against its box singles."""
    j = res["job"]
    r = j["r"]
    w = window_for(res, r, kappa, omega)
    t1, t2 = w["t1"], w["t2"]
    P = 2 * np.pi / omega
    x1 = res["params"]["lumps"]["centers"][0][0]
    pv = run_values(res, t1, t2)
    si, sv = _singles_at(singles, t1, t2, x1)
    vn = V_N_equal(pv, si, domega_dN)
    shift = None
    if len(sv) > 1:
        (xa, a), (xb, b) = sv[0], sv[-1]
        shift = dict(x=[xa, xb], dE=b["E"] - a["E"], dE_box=b["E_box"] - a["E_box"], dN=b["N"] - a["N"],
                     V_shift=2 * ((b["E"] - a["E"]) - a["omega"] * (b["N"] - a["N"])),
                     V_shift_box=2 * ((b["E_box"] - a["E_box"]) - a["omega"] * (b["N"] - a["N"])))
    so = sigma_omega_VN(res, singles, t1, t2, x1, domega_dN, P)
    sig = {}
    for key, sk in (("V_N", "V_shift"), ("V_N_box", "V_shift_box")):
        a_ = so.get("sigma_" + key)
        b_ = shift[sk] if shift else 0.0
        sig[key] = float(np.sqrt((a_ if a_ is not None else abs(vn[key])) ** 2 + b_ ** 2))
    # force
    td, d = w["t_d"], w["d"]
    c, se, rms = accel_fit(td, d, t1, t2, P)
    E_s = si["E_box"]
    sel = (td >= t1) & (td <= t2)
    r_eff = float(np.mean(d[sel])) if sel.any() else None
    if c is None:
        # no force: the 3-period Hann-smoothed d(t) starts at t_d[0] = 1.5 P, and kappa |d - r| >= 0.10 already
        # there (t2 = t_hit = t_d[0]), so [t1, t2] holds fewer than 6 smoothed samples
        t_raw, X_raw = separation(res)
        d_raw = np.linalg.norm(X_raw[:, 1, :] - X_raw[:, 0, :], axis=1)
        frc = dict(accel=None, accel_se=None, F=None, F_se=None, F_floor_control=None, sigma_F=None, d_fit_rms=None,
                   force_excluded=True,
                   force_reason="fewer than 6 smoothed d samples in [t1, t2] (%d): the 3-period smoothed separation "
                                "starts at t = %.2f and reaches kappa|d - r| >= %.2f at t_hit = %s"
                                % (int(sel.sum()), float(td[0]), KAPPA_SHIFT, w["t_hit"]),
                   d_raw={"%g" % tt: float(np.interp(tt, t_raw, d_raw)) for tt in (0.0, 5.0, 10.0, 20.0, 40.0)
                          if tt <= t_raw[-1]})
    else:
        a_rel, a_se = 2 * float(c[2]), 2 * float(se[2])
        F, F_se = -0.5 * E_s * a_rel, 0.5 * E_s * a_se
        ctrl = [centroid_accel(s["res"], t1, t2, omega) for s in singles]
        a_ctrl = max((abs(a) + s_ for a, s_ in ctrl if a is not None), default=0.0)
        F_floor = 0.5 * E_s * np.sqrt(2.0) * a_ctrl
        frc = dict(accel=a_rel, accel_se=a_se, F=F, F_se=F_se, F_floor_control=F_floor,
                   sigma_F=float(np.sqrt(F_se ** 2 + F_floor ** 2)), d_fit_rms=rms, force_excluded=False,
                   force_reason=None)
    return dict(name=j["name"], r=r, r_eff=r_eff, dtheta=j["dtheta"],
                window=dict(t1=t1, t2=t2, moving=w["moving"], t_hit=w["t_hit"]), pair=pv, single=si, **vn,
                sigma_V_N=sig["V_N"], sigma_V_N_box=sig["V_N_box"], sigma_omega=so, shift=shift,
                E_inertia=E_s, **frc,
                standard1=dict(pair=standard1(res, t1, t2, pv["omega"]),
                               single_omega_over_m=si["omega"] / res["params"]["m"]),
                stopped_early=res.get("stopped_early"), t_end=res["t"])


def half_beat_rule(t1, t2, delta_omega):
    """A1: the window must hold at least half a beat, or Delta omega t2 < 0.3."""
    return bool((t2 - t1) * abs(delta_omega) / (2 * np.pi) >= 0.5 or abs(delta_omega) * t2 < 0.3)


def unequal_usable(moving, t1, t2, delta_omega):
    """Rows entering the unequal fits: not "moving" and passing the half-beat rule (Addendum 2 item 2)."""
    return bool((not moving) and half_beat_rule(t1, t2, delta_omega))


def pair_unequal(res, singles1, singles2, lump1, lump2):
    """Unequal pair: force amplitude and mean force (V_F route, headline) and the V_N(t) amplitude (reported)."""
    j = res["job"]
    r = j["r"]
    om1, om2 = lump1["omega"], lump2["omega"]
    kmin = min(lump1["kappa"], lump2["kappa"])
    w = window_for(res, r, kmin, om1)
    t1, t2 = w["t1"], w["t2"]
    tph, dphi = relative_phase_series(res, om1)
    P = 2 * np.pi / om1
    cen = res["params"]["lumps"]["centers"]
    near = lambda sl, x: min(sl, key=lambda s_: abs(s_["x"] - x))
    s1, s2 = near(singles1, cen[0][0]), near(singles2, cen[1][0])
    v1, v2 = run_values(s1["res"], t1, t2), run_values(s2["res"], t1, t2)
    E1l, E2l = v1["E_box"], v2["E_box"]
    inv_m = 1 / E1l + 1 / E2l
    sel_ph = (tph >= t1) & (tph <= t2)
    dom = float(np.polyfit(tph[sel_ph], dphi[sel_ph], 1)[0]) if sel_ph.sum() > 2 else abs(om1 - om2)
    beats = float((t2 - t1) * abs(dom) / (2 * np.pi))
    half_beat_ok = half_beat_rule(t1, t2, dom)
    # V_N(t) against cos(dphi): half-space N per lump (reported only)
    lp = res["lumps"]["raw"]["point"]
    tp = np.array(lp["t"])
    Ih = np.array(lp["It_half"])[:, 0]
    It = np.array(res["point"]["I"])
    Ep = np.interp(tp, res["series"]["t"], res["series"]["E_plus_absorbed"])

    def ser(q):
        pp = q["res"]["point"]
        return (np.array(pp["I"]), np.array(q["res"]["lumps"]["raw"]["point"]["It_half"])[:, 0],
                np.interp(tp, q["res"]["series"]["t"], q["res"]["series"]["E_plus_absorbed"]))
    I1, H1, E1 = ser(s1)
    I2, H2, E2 = ser(s2)
    n = min(tp.size, I1.size, I2.size, It.size)
    dN1 = (Ih[:n] - H1[:n] - H2[:n]) / om1
    dN2 = ((It[:n] - Ih[:n]) - (I1[:n] - H1[:n]) - (I2[:n] - H2[:n])) / om2
    VN = Ep[:n] - E1[:n] - E2[:n] - om1 * dN1 - om2 * dN2
    ts, VNs = hann_smooth(tp[:n], VN, 3 * P)
    cph, sph = np.cos(np.interp(ts, tph, dphi)), np.sin(np.interp(ts, tph, dphi))
    sel = (ts >= t1) & (ts <= t2)
    if sel.sum() >= 6:
        cf, *_ = np.linalg.lstsq(np.stack([np.ones(sel.sum()), cph[sel], sph[sel]], axis=1), VNs[sel], rcond=None)
        cf = [float(x) for x in cf]
    else:                       # no window for the smoothed series (see the force below)
        cf = [None, None, None]
    # force: d(t) = c0 + c1 t + c3 t^2 + c2 G(t), G'' = cos(dphi(t))
    td, d = w["t_d"], w["d"]
    g = np.cos(np.interp(td, tph, dphi))
    dt = td[1] - td[0]
    G1 = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * dt)])
    G = np.concatenate([[0.0], np.cumsum(0.5 * (G1[1:] + G1[:-1]) * dt)])
    c, se, rms = accel_fit(td, d, t1, t2, P, extra=[G])
    seld = (td >= t1) & (td <= t2)
    if c is None:
        # as pair_equal: the smoothed d(t) starts at t_d[0] = 1.5 P and has already moved past the threshold
        t_raw, X_raw = separation(res)
        d_raw = np.linalg.norm(X_raw[:, 1, :] - X_raw[:, 0, :], axis=1)
        frc = dict(F_amp=None, F_amp_se=None, F_mean=None, F_mean_se=None, F_floor_control=None, sigma_F_amp=None,
                   d_fit_rms=None, force_excluded=True,
                   force_reason="fewer than 6 smoothed d samples in [t1, t2] (%d): the 3-period smoothed separation "
                                "starts at t = %.2f and reaches kappa|d - r| >= %.2f at t_hit = %s"
                                % (int(seld.sum()), float(td[0]), KAPPA_SHIFT, w["t_hit"]),
                   d_raw={"%g" % tt: float(np.interp(tt, t_raw, d_raw)) for tt in (0.0, 5.0, 10.0, 20.0, 40.0)
                          if tt <= t_raw[-1]})
    else:
        F_amp, F_amp_se = -float(c[3]) / inv_m, float(se[3]) / inv_m
        F_mean, F_mean_se = -2 * float(c[2]) / inv_m, 2 * float(se[2]) / inv_m
        a1 = centroid_accel(s1["res"], t1, t2, om1)
        a2 = centroid_accel(s2["res"], t1, t2, om1)
        a_ctrl = np.sqrt(sum((abs(a) + s_) ** 2 for a, s_ in (a1, a2) if a is not None))
        F_floor = a_ctrl / inv_m
        frc = dict(F_amp=F_amp, F_amp_se=F_amp_se, F_mean=F_mean, F_mean_se=F_mean_se,
                   F_floor_control=float(F_floor), sigma_F_amp=float(np.sqrt(F_amp_se ** 2 + F_floor ** 2)),
                   d_fit_rms=rms, force_excluded=False, force_reason=None)
    return dict(name=j["name"], r=r, r_eff=(float(np.mean(d[seld])) if seld.any() else None), dtheta=j["dtheta"],
                window=dict(t1=t1, t2=t2, moving=w["moving"], t_hit=w["t_hit"]), delta_omega_measured=dom,
                beats_in_window=beats, half_beat_ok=half_beat_ok,
                use_in_fits=bool(unequal_usable(w["moving"], t1, t2, dom) and not frc["force_excluded"]),
                inertia=dict(E1=E1l, E2=E2l), **frc, V_N_amp_cos=cf[1], V_N_amp_sin=cf[2], V_N_mean=cf[0],
                standard1=dict(pair=standard1(res, t1, t2, v1["omega"]), omega_over_m_singles=[v1["omega"], v2["omega"]]))


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


def _vf_block(r, F, sF, F16):
    """V_F with fitted tail and without, and the verdicts (grows-clause 2 also on |F|)."""
    rv, V, sV, tail = V_F_from_force(r, F, sF, tail=True)
    _, V0, _, _ = V_F_from_force(r, F, sF, tail=None)
    o = np.argsort(r)
    Fo, sFo = np.asarray(F)[o], np.asarray(sF)[o]
    return dict(r=rv.tolist(), V=V.tolist(), sigma=sV.tolist(), V_no_tail=V0.tolist(), tail=tail), \
        verdicts_both(rv, V, sV, F16, F=Fo, sF=sFo)


def analyze_equal(prefix, t, results, J, static_file, chans=(0, 180)):
    from swirl.session4.testA import driver as D
    n = D.numbers(t)
    have = {j["name"] for j in J if os.path.exists(_abs(os.path.join(results, j["name"] + ".json")))}
    singles = {}
    for j in J:
        if j["name"].startswith(prefix + "_single_") and j["name"] in have and not j.get("check"):
            singles.setdefault(j["box"], []).append(single_record(load(j["name"], results)))
    st = json.load(open(_abs(static_file)))
    out = dict(type=t, numbers=n, domega_dN=domega_dN(t), channels={})
    for dth in chans:
        lab = "p%03d" % dth
        rows = []
        for j in J:
            if j["kind"] == "pair" and j["name"].startswith(prefix + "_r") and j["dtheta"] == dth and not j.get("check") \
                    and j["name"] in have and j["box"] in singles:
                om_box = run_values(singles[j["box"]][0]["res"], T1, T_MAX)["omega"]
                rows.append(pair_equal(load(j["name"], results), singles[j["box"]], n["kappa"], om_box, domega_dN(t)))
        rows.sort(key=lambda q: q["r"])
        if not rows:
            continue
        r_all = np.array([q["r"] for q in rows])
        moving_all = np.array([q["window"]["moving"] for q in rows])
        ch = dict(rows=rows)
        for key in ("V_N", "V_N_box"):
            ch[key] = dict(r=r_all.tolist(), V=[q[key] for q in rows], sigma=[q["sigma_" + key] for q in rows])
        ch["V_N_minus_box"] = dict(r=r_all.tolist(), V=[q["V_N_minus_box"] for q in rows],
                                   joint_floor=[float(np.hypot(q["sigma_V_N"], q["sigma_V_N_box"])) for q in rows])
        # force route: rows without a force (no window, force_excluded) are left out and recorded
        frows = [q for q in rows if not q["force_excluded"]]
        ch["force_excluded"] = [dict(name=q["name"], r=q["r"], reason=q["force_reason"]) for q in rows
                                if q["force_excluded"]]
        r = np.array([q["r"] for q in frows])
        reff = np.array([q["r_eff"] for q in frows])
        moving = np.array([q["window"]["moving"] for q in frows], bool)
        F = np.array([q["F"] for q in frows], float)
        sF = np.array([q["sigma_F"] for q in frows], float)
        ch["F"] = dict(r_eff=reff.tolist(), r=r.tolist(), F=F.tolist(), sigma=sF.tolist(), moving=moving.tolist(),
                       names=[q["name"] for q in frows])
        ch["force_signs_dynamic"] = force_signs(reff, F, sF)
        F16 = float(np.interp(16.0, reff, F)) if reff.size and reff.min() <= 16 <= reff.max() else np.nan
        verd = {}
        for key in ("V_N", "V_N_box"):
            m = ~moving_all
            verd[key] = verdicts_both(r_all[m], np.array(ch[key]["V"])[m], np.array(ch[key]["sigma"])[m], F16)
        keep = ~moving & (r >= R_LO - 1e-9)
        if keep.sum() >= 3:
            ch["V_F"], verd["V_F"] = _vf_block(reff[keep], F[keep], sF[keep], F16)
        rs, Us, sUs, FUs = _static_channel(st, lab, "U_static")
        verd["U_static"] = verdicts_both(rs, Us, sUs, float(np.interp(16.0, rs, FUs)))
        rs, Vr, sVr, FVr = _static_channel(st, lab, "V_raw")
        verd["V_raw"] = verdicts_both(rs, Vr, sVr, float(np.interp(16.0, rs, FVr)))
        ch["force_signs_static"] = force_signs(rs, FUs, 3e-30 + 0 * FUs)
        ch["verdicts"] = verd
        ch["headline"] = headline(verd)
        ch["U_tail_at_r_eff"] = [float(-2 * np.pi * n["A"] ** 2 * np.cos(np.deg2rad(dth)) * np.exp(-n["kappa"] * x) / x)
                                 for x in reff]
        ch["U_static_at_r_eff"] = [float(np.interp(x, rs, Us)) for x in reff]
        ch["F_static_at_r_eff"] = [float(np.interp(x, rs, FUs)) for x in reff]
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
        # force route: rows without a force (no window, force_excluded) are left out and recorded
        excluded = [dict(name=q["name"], r=q["r"], reason=q["force_reason"]) for q in rows if q["force_excluded"]]
        rows_all, rows = rows, [q for q in rows if not q["force_excluded"]]
        use = np.array([q["use_in_fits"] for q in rows], bool) & (np.array([q["r"] for q in rows]) >= R_LO - 1e-9)
        reff = np.array([q["r_eff"] for q in rows], float)
        F = np.array([q["F_amp"] for q in rows], float)
        sF = np.array([q["sigma_F_amp"] for q in rows], float)
        ch = dict(rows=rows_all, force_excluded=excluded, used_in_fits=[q["name"] for q, u in zip(rows, use) if u],
                  F_amp=dict(r_eff=reff.tolist(), F=F.tolist(), sigma=sF.tolist(), used=use.tolist(),
                             names=[q["name"] for q in rows]),
                  F_mean=dict(r_eff=reff.tolist(), F=[q["F_mean"] for q in rows], se=[q["F_mean_se"] for q in rows]),
                  force_signs_dynamic=force_signs(reff[use], F[use], sF[use]) if use.any() else None)
        verd = {}
        F16 = float(np.interp(16.0, reff[use], F[use])) if use.sum() >= 2 else np.nan
        if use.sum() >= 3:
            ch["V_F_amp"], verd["V_F"] = _vf_block(reff[use], F[use], sF[use], F16)
        if st:
            rs, Us, sUs, FUs = _static_channel(st, "p%03d" % dth, "U_static")
            verd["U_static"] = verdicts_both(rs, Us, sUs, float(np.interp(16.0, rs, FUs)))
            ch["U_amp_static_at_r_eff"] = [float(np.interp(x, rs, Us)) for x in reff]
            ch["F_amp_static_at_r_eff"] = [float(np.interp(x, rs, FUs)) for x in reff]
            ch["U_law_at_r_eff"] = [float(np.interp(x, rs, [q["p%03d" % dth]["U_law"] for q in st["rows"]])) for x in reff]
            ch["omega_N_cross_amp_static_at_r_eff"] = [
                float(np.interp(x, rs, [n1["omega"] * n2["omega"] * q["moments"]["I_bil"] for q in st["rows"]])) for x in reff]
        ch["verdicts"] = verd
        ch["headline"] = headline(verd)
        ch["note"] = ("headline = V_F (force amplitude). V_N amplitude against cos(dphi) contains the beating overlap "
                      "charge (about omega_1 omega_2 int f_1 f_2) and is reported only")
        out["channels"]["p%03d" % dth] = ch
    return out


def controls(results, J):
    """Box check (1.5x, r = 24), resolution check (dx = 0.40, r = 16): V_N and F against the production values;
    linear twins: energy inside 3 hw over time."""
    from swirl.session4.testA import driver as D
    n = D.numbers("primary")
    have = {j["name"] for j in J if os.path.exists(_abs(os.path.join(results, j["name"] + ".json")))}
    out = {}
    for chk in ("box", "resolution"):
        pj = [j for j in J if j.get("check") == chk and j["kind"] == "pair"]
        sj = [j for j in J if j.get("check") == chk and j["kind"] == "single"]
        if not pj or pj[0]["name"] not in have or sj[0]["name"] not in have:
            out[chk] = "missing"
            continue
        s = single_record(load(sj[0]["name"], results))
        om = run_values(s["res"], T1, T_MAX)["omega"]
        out[chk] = pair_equal(load(pj[0]["name"], results), [s], n["kappa"], om, domega_dN("primary"))
    for j in J:
        if j["kind"] == "twin" and j["name"] in have:
            s = load(j["name"], results)["series"]
            E = np.array(s["E_loc"])[:, 0]
            out[j["name"]] = dict(t=s["t"][::10], E_in_3hw_over_E0=(E / s["E_total"][0])[::10].tolist(),
                                  final=float(E[-1] / s["E_total"][0]))
    return out


def standard1_all(results, J):
    """Standard 1 for every finished run: omega/m (whole period over [40, end]) and the nonlinear share."""
    out = {}
    for j in J:
        p = _abs(os.path.join(results, j["name"] + ".json"))
        if j["kind"] == "source" or not os.path.exists(p):
            continue
        res = load(j["name"], results)
        te = min(T_MAX, float(res["t"]))
        v = run_values(res, min(T1, te / 2), te)
        out[j["name"]] = standard1(res, min(T1, te / 2), te, v["omega"] if v else None)
    return out


def analyze_all(results=OUT + "/results", jobs_file=OUT + "/jobs.json", out=OUT + "/analysis.json"):
    from swirl.session4.testA.static import dump
    J = json.load(open(_abs(jobs_file)))
    rec = dict(what="Test A analysis (plan A0 to A3, Addenda 1 and 2); headline = V_F per channel",
               definitions=__doc__,
               primary=analyze_equal("pA", "primary", results, J, OUT + "/static/static_primary.json"))
    rec["unequal"] = analyze_unequal("uA", "dip", "big", results, J,
                                     OUT + "/static/static_unequal_%.6f_%.6f.json" % (0.9065045072253043, 0.857370255463254))
    if any(j["group"] == "b5" for j in J):
        rec["b5"] = analyze_equal("bA", "b5", results, J, OUT + "/static/static_b5.json", chans=(0,))
    rec["controls"] = controls(results, J)
    rec["standard1"] = standard1_all(results, J)
    dump(rec, out)
    return rec


def static_verdicts(names=("primary", "b5", "unequal_0.906505_0.857370"), out=OUT + "/static/static_verdicts.json"):
    """A3 and A3' verdicts and A2 fits for the static potentials (U_static, V_raw; dtheta 0 and 180), plus the
    supplementary records with the extension beyond r = 45."""
    from swirl.session4.testA.static import add_derived, dump
    rec = {}
    for nm in list(names) + [n + "+ext" for n in names]:
        base = nm.replace("+ext", "")
        p = _abs(OUT + "/static/static_%s.json" % base)
        pe = _abs(OUT + "/static/static_%s_ext.json" % base)
        if not os.path.exists(p) or (nm.endswith("+ext") and not os.path.exists(pe)):
            continue
        st = json.load(open(p))
        if nm.endswith("+ext"):
            st = dict(st, rows=st["rows"] + json.load(open(pe))["rows"])
            add_derived(st)
        rec[nm] = {}
        for lab in ("p000", "p180"):
            for key in ("U_static", "V_raw"):
                r, V, s, F = _static_channel(st, lab, key)
                rec[nm]["%s_%s" % (key, lab)] = verdicts_both(r, V, s, float(np.interp(16.0, r, F)))
    dump(rec, out)
    return rec


# ------------------------------------------------------------------------------------------- smoke
def smoke():
    from swirl.session4.testA import driver as D
    from swirl.session4.testA.static import F_asym, U_tail, dump
    res = load("smoke_pA_r20_p000")
    sres = load("smoke_pA_single_primary_N144_x-10.0")
    n = D.numbers("primary")
    om = run_values(sres, T1, T_MAX)["omega"]
    pe = pair_equal(res, [single_record(sres)], n["kappa"], om, domega_dN("primary"))
    t0 = json.load(open(_abs(OUT + "/t0/smoke_pA_r20_p000.json")))["t0"]
    st = json.load(open(_abs(OUT + "/static/static_primary.json")))
    row = [q for q in st["rows"] if abs(q["r"] - 20.0) < 1e-9][0]["p000"]
    out = dict(pair=pe, t0=t0, U_tail=float(U_tail(20.0, n["A"], n["kappa"])), U_static=row["U_static"],
               V_raw=row["V_raw"], F_asym=float(F_asym(20.0, n["A"], n["kappa"])), F_static=row["F_U_static"],
               wall_seconds=dict(pair=res["wall_seconds"], single=sres["wall_seconds"]),
               steps=dict(pair=res["steps_done"], single=sres["steps_done"]), N=res["params"]["N"],
               E_plus_absorbed_max_rel_dev=dict(pair=res["summary"]["max_rel_dev_E_plus_absorbed"],
                                                single=sres["summary"]["max_rel_dev_E_plus_absorbed"]))
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
        print(json.dumps({k: v for k, v in o["pair"].items() if k not in ("pair", "single")}, indent=1, default=float))
