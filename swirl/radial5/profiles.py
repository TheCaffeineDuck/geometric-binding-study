"""Single-frequency start profiles from the cycle-averaged full Lagrangian.

Ansatz f(r, t) = F(r) cos(theta), theta = omega t. Plan normalisation, units 1:

    L = P(X) - U(f),   X = f_t^2 - S,   S = f_r^2 + s2 f^2 / r^2
    O: P = X / 2,               U = f^2/2 - g4 f^4/4 + g6 f^6/6
    R: P = 1 - sqrt(1 - X),     U = f^2/2           (full square root)
    s2 = 0 single bump (f_r(0) = 0), s2 = 2 pinch (f(0) = 0)

Over one cycle X = omega^2 F^2 sin^2 - T cos^2 with T = F'^2 + s2 F^2/r^2. The
cycle averages <.> are taken by the midpoint rule in theta on [0, pi/2] (all
integrands depend on cos^2 only; the rule is spectrally accurate for smooth
periodic integrands). No series in X is used anywhere. With

    A = <2 cos^2 P'(X)>,  B = <2 sin^2 P'(X)>,  Ubar(F) = <U'(F cos) cos>

the Euler-Lagrange equation of the averaged Lagrangian l = r^2 (<P> - <U>) is

    (r^2 A F')' = r^2 (Ubar - omega^2 F B) + s2 A F.

It is solved by bisection shooting on F(0) (bump) or F'(0) (pinch) for the
nodeless profile, then the exact linear tail is attached where the bracketing
trajectories part. Observables (cycle-averaged, on-shell):

    N = dLbar/domega = int 4 pi r^2 omega F^2 B dr     (adiabatic invariant)
    E = omega N - Lbar,  energy per cycle = 2 pi E / omega
    Derrick (scaling) identity: 3 Lbar + int 4 pi r^2 T A dr = 0
    Along the family dE/domega = omega dN/domega (exact).
"""

import numpy as np
from scipy.integrate import simpson, solve_ivp
from scipy.optimize import brentq

FOUR_PI = 4.0 * np.pi
K_THETA = 96


def _theta_nodes(K=K_THETA):
    th = (np.arange(K) + 0.5) * (0.5 * np.pi / K)
    return np.cos(th) ** 2, np.sin(th) ** 2


class Model:
    """model: 'O' (with g4, g6) or 'R'; obj: 'bump' or 'pinch'."""

    def __init__(self, model, obj, g4=0.0, g6=0.0, K=K_THETA):
        assert model in ("O", "R") and obj in ("bump", "pinch")
        self.model, self.obj, self.g4, self.g6 = model, obj, float(g4), float(g6)
        self.s2 = 0.0 if obj == "bump" else 2.0
        self.c, self.s = _theta_nodes(K)
        self.c2 = self.c * self.c

    def F_limit(self, omega):
        """Largest admissible profile value. R: omega F < 1 (X < 1 over the cycle).
        O with g6 > 0: the outer root of Ubar(F) - omega^2 F / 2 (beyond it the shooting
        trajectory runs away; for the bump it is the thin-wall end)."""
        if self.model == "R":
            return 1.0 / omega
        if self.g6 > 0:
            x = np.roots([0.3125 * self.g6, -0.375 * self.g4, 0.5 * (1.0 - omega ** 2)])
            x = x[np.isreal(x) & (x.real > 0)].real
            if x.size:
                return float(np.sqrt(np.max(x)))
        return np.inf

    # P'(X), P''(X) on the theta nodes
    def _p12(self, X):
        if self.model == "O":
            return 0.5 * np.ones_like(X), np.zeros_like(X)
        om = 1.0 - X
        y = 1.0 / np.sqrt(om)
        return 0.5 * y, 0.25 * y ** 3

    def Pavg(self, X):
        """<P(X)> over theta, X on the nodes (last axis)."""
        if self.model == "O":
            return np.mean(0.5 * X, axis=-1)
        return np.mean(1.0 - np.sqrt(1.0 - X), axis=-1)

    def Ubar(self, F):
        if self.model == "O":
            return 0.5 * F - 0.375 * self.g4 * F ** 3 + 0.3125 * self.g6 * F ** 5
        return 0.5 * F

    def Uavg(self, F):
        if self.model == "O":
            return 0.25 * F ** 2 - 0.09375 * self.g4 * F ** 4 + (5.0 / 96.0) * self.g6 * F ** 6
        return 0.25 * F ** 2

    def rhs(self, omega):
        c, s, c2, s2 = self.c, self.s, self.c2, self.s2
        w2 = omega * omega

        def f(r, y):
            F, Q = y
            T = Q * Q + s2 * F * F / (r * r)
            X = w2 * F * F * s - T * c
            if self.model == "R" and np.max(X) >= 1.0:
                return np.array([Q, np.nan])
            P1, P2 = self._p12(X)
            A = np.mean(2.0 * c * P1)
            B = np.mean(2.0 * s * P1)
            AQ = -4.0 * Q * np.mean(c2 * P2)
            AF = np.mean(2.0 * c * P2 * (2.0 * w2 * F * s - 2.0 * s2 * F * c / (r * r)))
            Ar = np.mean(2.0 * c * P2 * (2.0 * s2 * F * F * c / r ** 3))
            rhs_eq = r * r * (self.Ubar(F) - w2 * F * B) + s2 * A * F
            num = rhs_eq - 2.0 * r * A * Q - r * r * Q * (AF * Q + Ar)
            den = r * r * (A + Q * AQ)
            return np.array([Q, num / den])
        return f

    def averages(self, r, F, Q, omega):
        """Per-radius cycle averages for observables (arrays over r)."""
        T = Q * Q + self.s2 * F * F / np.where(r > 0, r * r, 1.0)
        if self.s2 and r[0] == 0.0:
            T[0] = 3.0 * Q[0] ** 2                   # F ~ a r: F'^2 + 2 F^2/r^2 -> 3 a^2
        X = omega ** 2 * (F * F)[:, None] * self.s[None, :] - T[:, None] * self.c[None, :]
        P1, _ = self._p12(X)
        A = np.mean(2.0 * self.c * P1, axis=-1)
        B = np.mean(2.0 * self.s * P1, axis=-1)
        return dict(T=T, X=X, A=A, B=B, Pavg=self.Pavg(X), Uavg=self.Uavg(F))


def _shoot_once(mdl, omega, x, r_end, rtol=1e-12):
    """One trajectory. x = F(0) (bump) or F'(0) (pinch). Returns (kind, sol):
    'over'  F crosses zero (too large),
    'limit' |F| reaches mdl.F_limit (R: omega F = 1, where the ansatz leaves X < 1),
    'under' F turns back up, or (pinch) never peaks (too small)."""
    r0 = 1e-4
    rhs = mdl.rhs(omega)
    if mdl.obj == "bump":
        F0 = x
        # F = F0 + F2 r^2: 6 A F2 = Ubar - omega^2 F0 B at T = 0
        X0 = omega ** 2 * F0 * F0 * mdl.s
        P1, _ = mdl._p12(X0)
        A0, B0 = np.mean(2 * mdl.c * P1), np.mean(2 * mdl.s * P1)
        F2 = (mdl.Ubar(F0) - omega ** 2 * F0 * B0) / (6.0 * A0)
        y0 = [F0 + F2 * r0 * r0, 2.0 * F2 * r0]
    else:
        y0 = [x * r0, x]

    def ev_zero(r, y):
        return y[0]
    ev_zero.terminal, ev_zero.direction = True, -1

    def ev_turn(r, y):
        return y[1]
    ev_turn.terminal, ev_turn.direction = True, 1

    F_lim = mdl.F_limit(omega)

    def ev_limit(r, y):
        return 1.0 - abs(y[0]) / F_lim - 1e-12
    ev_limit.terminal, ev_limit.direction = True, -1

    lim = [ev_limit] if np.isfinite(F_lim) else []
    kw = dict(method="DOP853", rtol=rtol, atol=1e-14 * max(x, 1e-300), dense_output=True)

    def classify(s, i_turn):
        if s.t_events[0].size:
            return "over"
        if lim and s.t_events[-1].size:
            return "limit"
        return "under"

    if mdl.obj == "pinch":
        # F' > 0 at the start: the 'turn' event counts only after the peak
        def ev_peak(r, y):
            return y[1]
        ev_peak.terminal, ev_peak.direction = True, -1
        s1 = solve_ivp(rhs, (r0, r_end), y0, events=[ev_zero, ev_peak] + lim, **kw)
        if not s1.t_events[1].size:
            return classify(s1, None), s1               # limit, or never peaked: too small
        s2 = solve_ivp(rhs, (s1.t_events[1][0], r_end), s1.y_events[1][0],
                       events=[ev_zero, ev_turn] + lim, **kw)
        return classify(s2, 1), (s1, s2)
    s = solve_ivp(rhs, (r0, r_end), y0, events=[ev_zero, ev_turn] + lim, **kw)
    return classify(s, 1), s


class _Traj:
    """Piecewise dense output of a shooting trajectory (one or two solve_ivp parts)."""

    def __init__(self, sol):
        self.parts = list(sol) if isinstance(sol, tuple) else [sol]
        self.r_end = self.parts[-1].t[-1]
        self.r_start = self.parts[0].t[0]

    def __call__(self, r):
        r = np.atleast_1d(np.asarray(r, dtype=float))
        out = np.empty((2, r.size))
        lo = self.r_start
        for k, p in enumerate(self.parts):
            hi = p.t[-1]
            sel = (r >= lo - 1e-300) & ((r <= hi) | (k == len(self.parts) - 1))
            if sel.any():
                out[:, sel] = p.sol(np.clip(r[sel], p.t[0], p.t[-1]))
            lo = hi
        return out


def x_guess(mdl, omega):
    """Leading-order (NLS) value of the shooting parameter: F(0) = 4.3374 eps / sqrt(lam)
    (bump), F'(0) = 3.7782 eps^2 / sqrt(lam) (pinch); lam = 3 g4 / 4 (O) or 3/8 (R).
    4.3374 and 3.7782 are the l = 0 and l = 1 NLS ground states (nls_ground)."""
    lam = 0.75 * mdl.g4 if mdl.model == "O" else 0.375
    eps = np.sqrt(1.0 - omega * omega)
    lam = max(lam, 1e-12)
    return (4.3374 * eps if mdl.obj == "bump" else 3.7782 * eps * eps) / np.sqrt(lam)


def solve(mdl, omega, iters=200, rtol=1e-12):
    """Nodeless single-frequency profile at rate omega, by bisection between the last
    undershoot and the first overshoot of an upward scan of the shooting parameter
    (so the bracket is the smallest-amplitude one). Raises RuntimeError('no profile ...')
    when the scan reaches the limit (R: omega F = 1) or the runaway value (O, g6 > 0)
    before any overshoot."""
    kappa = np.sqrt(1.0 - omega * omega)
    r_end = 60.0 / kappa + 60.0
    F_lim = mdl.F_limit(omega)
    x = 0.25 * x_guess(mdl, omega)
    if mdl.obj == "bump" and np.isfinite(F_lim):
        x = min(x, 0.25 * F_lim)
    kind, s = _shoot_once(mdl, omega, x, r_end, rtol)
    while kind != "under":
        x *= 0.5
        if x < 1e-10:
            raise RuntimeError("no undershoot found at omega=%.6f" % omega)
        kind, s = _shoot_once(mdl, omega, x, r_end, rtol)
    lo, best_u = x, (x, s)
    last = False
    for _ in range(400):
        x *= 1.1
        if mdl.obj == "bump" and np.isfinite(F_lim) and x >= F_lim * (1 - 1e-10):
            x, last = F_lim * (1 - 1e-10), True
        kind, s = _shoot_once(mdl, omega, x, r_end, rtol)
        if kind != "under":
            break
        lo, best_u = x, (x, s)
        if last:
            raise RuntimeError("no profile at omega=%.6f: F(0) up to the limit %.6g undershoots"
                               % (omega, F_lim))
    else:
        raise RuntimeError("no overshoot found at omega=%.6f" % omega)
    if kind == "limit":
        raise RuntimeError("no profile at omega=%.6f: the first non-undershoot reaches the limit "
                           "|F| = %.6g (x = %.6g)" % (omega, F_lim, x))
    hi, best_o = x, (x, s)
    n_limit = 0
    for _ in range(iters):
        x = 0.5 * (lo + hi)
        if x <= lo or x >= hi:
            break
        kind, s = _shoot_once(mdl, omega, x, r_end, rtol)
        if kind == "under":
            lo, best_u = x, (x, s)
        else:
            n_limit += kind == "limit"
            hi, best_o = x, (x, s)
    if n_limit:
        raise RuntimeError("bracket at omega=%.6f touches the limit" % omega)
    return _assemble(mdl, omega, kappa, best_u, best_o)


def _side(kind):
    """'under' is one side of the family curve; 'over' and 'limit' the other."""
    return kind == "under"


def solve_point(mdl, omega=None, x=None, guess=None, delta=0.01, iters=200, rtol=1e-12):
    """Profile with omega fixed (search the shooting parameter x near guess) or with x fixed
    (F(0) for the bump, F'(0) for the pinch; search eps = sqrt(1 - omega^2) near guess).
    The search steps outward from the guess (log steps delta 2^n, both directions) to the
    nearest change undershoot <-> overshoot, then bisects to round-off. This finds the
    branch next to the guess, so a continuation can pass folds in omega or in x.
    Raises RuntimeError when no bracket is found or when the converged overshoot side is
    the limit (R: omega F = 1, the ansatz wall; O: the runaway value F_out)."""
    fix_omega = omega is not None
    if fix_omega:
        lo_b, hi_b = 1e-12, (mdl.F_limit(omega) * (1 - 1e-12) if mdl.obj == "bump" else np.inf)
        kappa = np.sqrt(1.0 - omega * omega)
        r_end = 60.0 / kappa + 60.0

        def shoot(v):
            return _shoot_once(mdl, omega, v, r_end, rtol)
    else:
        lo_b, hi_b = 1e-6, 1.0 - 1e-12
        if mdl.model == "R" and mdl.obj == "bump":
            # omega F(0) < 1 needs eps > sqrt(1 - 1/x^2)
            if x >= 1.0:
                lo_b = max(lo_b, np.sqrt(1.0 - 1.0 / (x * x)) * (1 + 1e-12))

        def shoot(e):
            w = np.sqrt(1.0 - e * e)
            return _shoot_once(mdl, w, x, 60.0 / e + 60.0, rtol)
    v0 = float(np.clip(guess, lo_b * (1 + 1e-9), hi_b * (1 - 1e-9)))
    k0, s0 = shoot(v0)
    bracket = None
    for n in range(40):
        step = delta * 2.0 ** n
        for sgn in (1.0, -1.0):
            v = v0 * np.exp(sgn * step)
            if not (lo_b < v < hi_b):
                v = lo_b * (1 + 1e-9) if sgn < 0 else hi_b * (1 - 1e-9)
            k, s = shoot(v)
            if _side(k) != _side(k0):
                bracket = ((v0, k0, s0), (v, k, s))
                break
        if bracket is not None:
            break
        if v0 * np.exp(-step) <= lo_b and v0 * np.exp(step) >= hi_b:
            break
    if bracket is None:
        raise RuntimeError("no bracket near %s = %.6g at %s = %.6g" % (
            "x" if fix_omega else "eps", v0, "omega" if fix_omega else "x",
            omega if fix_omega else x))
    a, b = bracket
    under, other = (a, b) if _side(a[1]) else (b, a)
    for _ in range(iters):
        v = 0.5 * (under[0] + other[0])
        if v in (under[0], other[0]):
            break
        k, s = shoot(v)
        if _side(k):
            under = (v, k, s)
        else:
            other = (v, k, s)
    if other[1] == "limit":
        what = "ansatz wall omega F = 1" if mdl.model == "R" else "runaway value F_out"
        raise RuntimeError("the curve here is the %s, not a profile (%s = %.8g)" % (
            what, "x" if fix_omega else "eps", 0.5 * (under[0] + other[0])))
    if fix_omega:
        res = _assemble(mdl, omega, kappa, (under[0], under[2]), (other[0], other[2]))
    else:
        e = 0.5 * (under[0] + other[0])
        res = _assemble(mdl, float(np.sqrt(1 - e * e)), e, (x, under[2]), (x, other[2]))
        res["eps_bracket_rel"] = float(abs(other[0] - under[0]) / e)
    res["fixed"] = "omega" if fix_omega else "amplitude"
    return res


def continue_family(mdl, eps0=0.01, ds0=0.04, ds_max=0.12, ds_min=2e-4, eps_max=0.999,
                    max_points=300, verbose=False, wall_cap=None, on_point=None):
    """Continuation along the family in c = (ln eps, ln x), x = F(0) (bump) or F'(0)
    (pinch), through folds in omega and in x. Predictor: the secant; corrector: solve_point
    with the parameter whose tangent component is larger held fixed. The step halves on
    failure; the family ends when it falls below ds_min, and the reason is returned."""
    w0 = np.sqrt(1 - eps0 ** 2)
    pts = [solve(mdl, w0)]
    pts.append(solve_point(mdl, omega=np.sqrt(1 - (eps0 * 1.08) ** 2), guess=pts[0]["shoot_param"] * 1.08))
    coord = lambda p: np.array([np.log(p["kappa"]), np.log(p["shoot_param"])])
    C = [coord(p) for p in pts]
    ds = ds0
    reason = None
    import time as _time
    t_start = _time.time()
    while len(pts) < max_points:
        if wall_cap is not None and _time.time() - t_start > wall_cap:
            reason = "wall cap %.0f s reached" % wall_cap
            break
        t = C[-1] - C[-2]
        t /= np.linalg.norm(t)
        pred = C[-1] + ds * t
        try:
            if abs(t[0]) >= abs(t[1]):
                e = float(np.exp(pred[0]))
                if e >= eps_max:
                    reason = "eps reached %.3f (omega %.4f): scan limit" % (eps_max, np.sqrt(1 - eps_max ** 2))
                    break
                p = solve_point(mdl, omega=float(np.sqrt(1 - e * e)), guess=float(np.exp(pred[1])),
                                delta=max(0.2 * ds, 1e-4))
            else:
                p = solve_point(mdl, x=float(np.exp(pred[1])), guess=float(np.exp(pred[0])),
                                delta=max(0.2 * ds, 1e-4))
            c = coord(p)
            d = c - C[-1]
            if np.dot(d, t) <= 0 or np.linalg.norm(d) > 3 * ds:
                raise RuntimeError("corrector left the branch (step %.3g, forward %.3g)"
                                   % (np.linalg.norm(d), np.dot(d, t)))
        except RuntimeError as exc:
            reason = str(exc)
            ds *= 0.5
            if ds < ds_min:
                break
            continue
        pts.append(p)
        C.append(c)
        if on_point is not None:
            on_point(pts, ds)
        if verbose:
            print("w=%.6f x=%.6g E=%.6g fixed=%s ds=%.3g" % (p["omega"], p["shoot_param"], p["E"],
                                                               p.get("fixed", "omega"), ds), flush=True)
        ds = min(ds * 1.3, ds_max)
    else:
        reason = "max_points reached"
    return pts, reason


def _tail_shape(obj, kappa, r):
    if obj == "bump":
        return np.exp(-kappa * r) / r
    x = kappa * r
    return np.exp(-x) * (1.0 + x) / (x * x)


def _tail_slope(obj, kappa, r):
    if obj == "bump":
        return -(kappa + 1.0 / r)
    x = kappa * r
    # d/dr ln[e^-x (1+x)/x^2] = kappa (-1 + 1/(1+x) - 2/x)
    return kappa * (-1.0 + 1.0 / (1.0 + x) - 2.0 / x)


def _assemble(mdl, omega, kappa, best_u, best_o):
    tu, to = _Traj(best_u[1]), _Traj(best_o[1])
    r_hi = min(tu.r_end, to.r_end)
    rr = np.linspace(tu.r_start, r_hi, 20001)
    Fu, Fo = tu(rr)[0], to(rr)[0]
    Fpk = max(np.max(Fu), np.max(Fo))
    i_pk = int(np.argmax(Fu))
    # match where the bracketing trajectories still agree to 1e-7 of the local value
    bad = np.where((np.abs(Fu - Fo) > 1e-7 * np.abs(Fu + Fo) / 2) & (np.arange(rr.size) > i_pk))[0]
    r_m = rr[bad[0] - 1] if bad.size else r_hi
    r_m = max(rr[i_pk] + 1e-3, r_m)
    Fm, Qm = 0.5 * (tu(r_m)[:, 0] + to(r_m)[:, 0])
    slope_lin = _tail_slope(mdl.obj, kappa, r_m)
    if Fm / Fpk > 1e-3 or abs(Qm / Fm - slope_lin) > 1e-3 * abs(slope_lin):
        raise RuntimeError("not resolved at omega=%.6f: the bracketing trajectories part at "
                           "F/F_peak = %.2e, where the slope differs from the linear tail by %.1e "
                           "(shooting precision limit, e.g. thin wall)"
                           % (omega, Fm / Fpk, abs(Qm / Fm - slope_lin) / abs(slope_lin)))
    slope_err = abs(Qm / Fm - _tail_slope(mdl.obj, kappa, r_m)) / abs(_tail_slope(mdl.obj, kappa, r_m))
    traj = tu
    x0 = 0.5 * (best_u[0] + best_o[0])
    res = dict(model=mdl.model, obj=mdl.obj, g4=mdl.g4, g6=mdl.g6, omega=float(omega),
               kappa=float(kappa), shoot_param=float(x0),
               shoot_bracket_rel=float((best_o[0] - best_u[0]) / x0),
               r_match=float(r_m), F_match_over_peak=float(Fm / Fpk),
               tail_slope_rel_err=float(slope_err))
    res["_curve"] = (traj, r_m, Fm, kappa, mdl.obj)
    _observables(mdl, res)
    return res


def curve(res, r):
    """F(r), F'(r) of a solved profile (tail beyond the matching radius)."""
    traj, r_m, Fm, kappa, obj = res["_curve"]
    r = np.atleast_1d(np.asarray(r, dtype=float))
    F = np.empty_like(r)
    Q = np.empty_like(r)
    inside = r <= r_m
    if obj == "bump":
        ri = np.maximum(r[inside], traj.r_start)
        y = traj(ri)
        # inside r0 = 1e-4 use the even expansion F(0) + F2 r^2
        F[inside], Q[inside] = y[0], y[1]
        small = r[inside] < traj.r_start
        if small.any():
            F2 = y[1][small] / (2 * traj.r_start)
            F[np.where(inside)[0][small]] = y[0][small] + F2 * (r[inside][small] ** 2 - traj.r_start ** 2)
            Q[np.where(inside)[0][small]] = 2 * F2 * r[inside][small]
    else:
        ri = np.maximum(r[inside], traj.r_start)
        y = traj(ri)
        F[inside], Q[inside] = y[0], y[1]
        small = r[inside] < traj.r_start
        if small.any():
            a = y[1][small]
            F[np.where(inside)[0][small]] = a * r[inside][small]
            Q[np.where(inside)[0][small]] = a
    ro = r[~inside]
    g = _tail_shape(obj, kappa, ro) / _tail_shape(obj, kappa, r_m)
    F[~inside] = Fm * g
    Q[~inside] = Fm * g * _tail_slope(obj, kappa, ro)
    return F, Q


def _observables(mdl, res):
    omega, kappa = res["omega"], res["kappa"]
    traj, r_m = res["_curve"][0], res["_curve"][1]
    # coarse look for the peak and half width
    rr = np.linspace(0.0, r_m, 40001)
    F, _ = curve(res, rr)
    Fpk = float(np.max(F))
    i_pk = int(np.argmax(F))
    j = i_pk + int(np.argmax(F[i_pk:] < 0.5 * Fpk))
    hw = float(brentq(lambda x: curve(res, x)[0][0] - 0.5 * Fpk, rr[j - 1], rr[j], xtol=1e-13))
    # quadrature grid: dense to where the tail is 1e-15 of the peak
    r_max = r_m + max(0.0, np.log(abs(res["F_match_over_peak"]) * 1e15)) / kappa + 5.0
    n = int(np.ceil(r_max / min(0.004, hw / 500.0))) // 2 * 2 + 1
    r = np.linspace(0.0, r_max, n)
    F, Q = curve(res, r)
    av = mdl.averages(r, F, Q, omega)
    w = FOUR_PI * r * r
    I = lambda y: float(simpson(w * y, x=r))
    N = I(omega * F * F * av["B"])
    Lbar = I(av["Pavg"] - av["Uavg"])
    E = omega * N - Lbar
    E_quad = I(0.25 * (omega ** 2 * F * F + av["T"] + F * F))
    TA = I(av["T"] * av["A"])
    Xmax = float(np.max(av["X"]))
    # energy of the evolution start f = F, f_t = 0 (t = 0 value before the PDE)
    if mdl.model == "O":
        Ustart = 0.5 * F * F - 0.25 * mdl.g4 * F ** 4 + mdl.g6 * F ** 6 / 6.0
        e0 = 0.5 * av["T"] + Ustart
    else:
        e0 = np.sqrt(1.0 + av["T"]) - 1.0 + 0.5 * F * F
    E_start = I(e0)
    out = dict(F_peak=Fpk, r_peak=float(r[np.argmax(F)]), hw=hw, N=N, E=E, I2=I(F * F),
               E_per_cycle=2.0 * np.pi * E / omega, L_bar=Lbar, E_quadratic=E_quad,
               nonlinear_share_signed=(E - E_quad) / E,
               derrick_rel=(3.0 * Lbar + TA) / TA,
               X_max_cycle=Xmax, X_min_cycle=float(np.min(av["X"])),
               omega_F_peak=omega * Fpk, E_start_t0=E_start, r_quad_max=float(r_max),
               n_quad=int(n))
    if mdl.model == "O":
        out["E4_share"] = I(-0.09375 * mdl.g4 * F ** 4) / E
        out["E6_share"] = I((5.0 / 96.0) * mdl.g6 * F ** 6) / E
    res.update(out)


def public(res):
    return {k: v for k, v in res.items() if not k.startswith("_")}


def nls_ground(obj):
    """Small-amplitude reference: R'' + 2R'/x - s2 R/x^2 = R - R^3 (lambda = 1, eps = 1).
    Equals model O at omega = 0 with g4 = 4/3, g6 = 0."""
    return solve(Model("O", obj, g4=4.0 / 3.0, g6=0.0), 1e-12)


def small_amplitude(obj, lam, omega, ground=None):
    """Leading-order NLS prediction for a family with cubic coefficient lam:
    F = (eps / sqrt(lam)) R(eps r), eps^2 = 1 - omega^2.
    O: lam = 3 g4 / 4;  R: lam = 3/8 (from <X^2>/8 with <sin^4> = 3/8)."""
    g = ground if ground is not None else nls_ground(obj)
    eps = np.sqrt(1.0 - omega * omega)
    # N = omega int F^2 B d3x -> (1/2) int F^2 d3x at leading order (B -> 1/2, omega -> 1);
    # E = N at leading order (E - omega N = -Lbar = O(eps N))
    N = 0.5 * g["I2"] / (lam * eps)
    return dict(F_peak=eps / np.sqrt(lam) * g["F_peak"], hw=g["hw"] / eps,
                r_peak=g["r_peak"] / eps, N=N, E=N)
