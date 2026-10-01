"""Single-frequency oscillon profiles.

phi = Phi(r) cos(omega t) in

    phi_tt - lap(phi) + m^2 phi - (g4/6) phi^3 + (g6/120) phi^5 = 0

gives, after time-averaging the Lagrangian,

    Phi'' + (2/r) Phi' = G'(Phi),   G'(P) = a P - b P^3 + c P^5
    a = m^2 - omega^2,  b = g4/8,  c = g6/192,  G(P) = a P^2/2 - b P^4/4 + c P^6/6

    E = int [ (1/4)(omega^2 + m^2) Phi^2 + (1/4) Phi'^2 - (g4/64) Phi^4 + (g6/2304) Phi^6 ] d3x
    N = (omega/2) int Phi^2 d3x

Rescaled form: m = 1, g4 = 1, g6 = beta (lengths in 1/m, field in m/sqrt(g4),
E in m/g4, N in 1/g4).

The nodeless solution is found by bisection shooting (used only as a starting
guess) and then solved as a boundary value problem with scipy's collocation
solver. The radial integrals int r^2 {P^2, P'^2, P^4, P^6} dr use 10-point
Gauss-Legendre on every mesh interval of the collocation solution (exact for
its piecewise-cubic interpolant).
The outer boundary condition is the exact linear tail, P'/P = -(kappa + 1/R).
"""

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp

FOUR_PI = 4.0 * np.pi


def coeffs(omega, m, g4, g6):
    return m * m - omega * omega, g4 / 8.0, g6 / 192.0


def omega_min_rescaled(beta):
    """Thin-wall end of the family (rescaled, m = 1): min over P of G/P^2 is zero.

    G/P^2 = eps^2/2 - P^2/32 + beta P^4/1152 has minimum eps^2/2 - 9/(32 beta),
    so solutions need 1 - omega^2 < 9/(16 beta). Returns 0 if no lower end.
    """
    w2 = 1.0 - 9.0 / (16.0 * beta)
    return float(np.sqrt(w2)) if w2 > 0 else 0.0


def shoot(omega, m, g4, g6, iters=200):
    """Bisection shooting for the nodeless profile. Returns (P0, r, P, dP).

    Overshoot (P crosses zero) means P0 too large; undershoot (P' > 0 while
    P > 0) means P0 too small. The returned trajectory is the last undershoot,
    truncated at its turning point.
    """
    a, b, c = coeffs(omega, m, g4, g6)
    # P0 lies between the first zero of G and the outer zero of G'.
    x = np.roots([c / 6.0, -b / 4.0, a / 2.0])        # G/P^2 = 0 in x = P^2
    xz = np.sort(x[np.isreal(x) & (x.real > 0)].real) if c > 0 else np.array([2 * a / b])
    xm = np.roots([c, -b, a]) if c > 0 else np.array([a / b])  # G'/P = 0
    xm = np.sort(xm[np.isreal(xm) & (xm.real > 0)].real)
    lo = np.sqrt(xz[0]) if len(xz) else 0.0
    hi = np.sqrt(xm[-1]) if (c > 0 and len(xm)) else 10.0 * np.sqrt(a / b) + 10.0

    def rhs(r, y):
        P, Q = y
        return [Q, a * P - b * P**3 + c * P**5 - 2.0 * Q / r]

    def ev_zero(r, y):
        return y[0]
    ev_zero.terminal = True
    ev_zero.direction = -1

    def ev_turn(r, y):
        return y[1]
    ev_turn.terminal = True
    ev_turn.direction = 1

    best = None
    r_end = 60.0 / np.sqrt(a) + 200.0
    for _ in range(iters):
        P0 = 0.5 * (lo + hi)
        if P0 == lo or P0 == hi:
            break
        r0 = 1e-6
        g = a * P0 - b * P0**3 + c * P0**5
        y0 = [P0 + g * r0**2 / 6.0, g * r0 / 3.0]
        s = solve_ivp(rhs, (r0, r_end), y0, method="DOP853", rtol=1e-12, atol=1e-14,
                      events=(ev_zero, ev_turn), dense_output=True)
        if s.t_events[0].size:
            hi = P0
        else:
            lo = P0
            best = (P0, s)
    P0, s = best
    return P0, s.t, s.y[0], s.y[1]


def _rhs(a, b, c):
    def f(r, y):
        P = y[0]
        return np.vstack([y[1], a * P - b * P**3 + c * P**5])
    return f


def _bc(kappa, R):
    def bc(ya, yb):
        return np.array([ya[1], yb[1] + (kappa + 1.0 / R) * yb[0]])
    return bc


_GX, _GW = np.polynomial.legendre.leggauss(10)


def _integrals(sol):
    """int_0^R r^2 {P^2, P'^2, P^4, P^6} dr on the collocation interpolant."""
    x = sol.x
    h = np.diff(x)
    mid = 0.5 * (x[1:] + x[:-1])
    r = (mid[:, None] + 0.5 * h[:, None] * _GX[None, :]).ravel()
    wts = (0.5 * h[:, None] * _GW[None, :]).ravel()
    P, Q = sol.sol(r)
    r2w = r * r * wts
    P2 = P * P
    return (np.sum(r2w * P2), np.sum(r2w * Q * Q), np.sum(r2w * P2 * P2),
            np.sum(r2w * P2**3))


def _mesh(R, core, n=2000):
    """Mesh dense in the core, geometric beyond."""
    x1 = np.linspace(0.0, min(core, R), n // 2, endpoint=False)
    x2 = np.geomspace(max(min(core, R), 1e-3), R, n - n // 2)
    return np.concatenate([x1, x2])


def guess_from_curve(r, P, kappa, mesh):
    """Interpolate a profile curve onto mesh, extended with the linear tail."""
    i_last = len(r) - 1
    Pm = np.interp(mesh, r, P)
    out = mesh > r[i_last]
    A = P[i_last] * r[i_last] * np.exp(kappa * r[i_last])
    Pm[out] = A * np.exp(-kappa * mesh[out]) / mesh[out]
    Qm = np.gradient(Pm, mesh)
    Qm[0] = 0.0
    return np.vstack([Pm, Qm])


def solve_profile(omega, m=1.0, g4=1.0, g6=0.6111111111111112, guess=None,
                  R=None, tol=1e-10, max_nodes=400000):
    """Solve for the nodeless profile. guess: (r, P) curve, or None to shoot.

    Returns a dict of the solution and its observables (in the units of the
    parameters passed).
    """
    a, b, c = coeffs(omega, m, g4, g6)
    kappa = np.sqrt(a)
    if guess is None:
        _, rg, Pg, _ = shoot(omega, m, g4, g6)
        # keep the part before the trajectory departs from the true solution
        k = np.argmin(np.abs(Pg)) if Pg[-1] > 0 else len(Pg) - 1
        rg, Pg = rg[: k + 1], Pg[: k + 1]
    else:
        rg, Pg = guess
    hw_guess = float(np.interp(0.5 * Pg[0], Pg[::-1], rg[::-1]))
    core = 1.5 * hw_guess + 5.0
    if R is None:
        R = core + 18.0 / kappa
    mesh = _mesh(R, core=min(core, 0.5 * R))
    y0 = guess_from_curve(rg, Pg, kappa, mesh)
    S = np.array([[0.0, 0.0], [0.0, -2.0]])
    sol = solve_bvp(_rhs(a, b, c), _bc(kappa, R), mesh, y0, S=S, tol=tol,
                    max_nodes=max_nodes, bc_tol=1e-12)
    if not sol.success:
        raise RuntimeError("solve_bvp failed at omega=%.6f: %s" % (omega, sol.message))
    return observables(sol, omega, m, g4, g6, R)


def observables(sol, omega, m, g4, g6, R):
    a, b, c = coeffs(omega, m, g4, g6)
    kappa = np.sqrt(a)
    r = sol.x
    P, Q = sol.y[0], sol.y[1]
    if np.any(P < -1e-8 * P[0]):
        raise RuntimeError("profile has a node at omega=%.6f" % omega)
    I2, Id, I4, I6 = _integrals(sol)
    # analytic linear tail beyond R: P = P_R (R/r) exp(-kappa (r-R))
    PR = P[-1]
    I2 += PR**2 * R**2 / (2 * kappa)
    Id += PR**2 * R**2 * (kappa + 1.0 / R) ** 2 / (2 * kappa)
    I2v, Idv, I4v, I6v = FOUR_PI * np.array([I2, Id, I4, I6])
    E = 0.25 * (omega**2 + m**2) * I2v + 0.25 * Idv - g4 / 64.0 * I4v + g6 / 2304.0 * I6v
    N = 0.5 * omega * I2v
    T = 0.5 * Idv
    Gint = 0.5 * a * I2v - 0.25 * b * I4v + c / 6.0 * I6v
    derrick = (T + 3.0 * Gint) / T
    # half-width: radius where P = P(0)/2 (P is monotone for the nodeless profile)
    rr = np.linspace(0.0, r[-1], 200001)
    Pr = sol.sol(rr)[0]
    j = np.argmax(Pr < 0.5 * P[0])
    x0, x1 = rr[j - 1], rr[j]
    y0, y1 = Pr[j - 1] - 0.5 * P[0], Pr[j] - 0.5 * P[0]
    hw = x0 - y0 * (x1 - x0) / (y1 - y0)
    # tail fit window: 1e-9 < P/P0 < 1e-3 and r < 0.8 R
    w = (Pr < 1e-3 * P[0]) & (Pr > 1e-9 * P[0]) & (rr < 0.8 * R)
    if w.sum() < 10:
        w = (Pr < 1e-3 * P[0]) & (rr < 0.8 * R)
    rw, Pw = rr[w], Pr[w]
    slope, icpt = np.polyfit(rw, np.log(rw * Pw), 1)
    Aloc = rw * Pw * np.exp(kappa * rw)
    A = float(np.mean(Aloc))
    return dict(
        omega=float(omega), m=float(m), g4=float(g4), g6=float(g6),
        beta=float(g6 * m * m / g4**2),
        P0=float(P[0]), half_width=float(hw), N=float(N), E=float(E),
        E_over_mN=float(E / (m * N)), kappa=float(kappa),
        A=A, A_rel_spread=float(np.std(Aloc) / A),
        kappa_fit=float(-slope), kappa_fit_rel_err=float(abs(-slope - kappa) / kappa),
        tail_window=[float(rw[0]), float(rw[-1])],
        derrick_rel=float(derrick), R_box=float(R), n_nodes=int(r.size),
        T=float(T), G_int=float(Gint),
        E4_share=float(-g4 / 64.0 * I4v / E), E6_share=float(g6 / 2304.0 * I6v / E),
        _sol=sol,
    )


def profile_curve(res, r):
    """Evaluate a solved profile at radii r (beyond the box: the linear tail)."""
    sol, R, kappa = res["_sol"], res["R_box"], res["kappa"]
    r = np.asarray(r, dtype=float)
    out = np.empty_like(r)
    inside = r <= R
    out[inside] = sol.sol(r[inside])[0]
    PR = sol.y[0, -1]
    ro = r[~inside]
    out[~inside] = PR * R / ro * np.exp(-kappa * (ro - R))
    return out


def next_guess(res):
    sol = res["_sol"]
    return sol.x, sol.y[0]


def public(res):
    return {k: v for k, v in res.items() if not k.startswith("_")}
