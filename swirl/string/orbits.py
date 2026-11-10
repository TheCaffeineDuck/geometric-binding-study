"""Exact time-periodic states (breathers) of the ring, by Newton on the quarter-period map, and
their Floquet multipliers.

A state with v(0) = 0 and u(T/4) = 0 everywhere is periodic with u(t + T/2) = -u(t) (F odd, the
scheme time-reversible): u(-t) = u(t) by reversibility, u(T/4 + s) = -u(T/4 - s) by the odd force.
So the unknowns are u(0) (N numbers) and the equations u(T/4) = 0 (N numbers). The uniform shift
is not a null direction (it survives to T/4), so the ring needs no pinning.

The periodic state is a periodic orbit of the discrete Yoshida map with dt = T/(4 nq); its
distance from the exact flow's orbit is O(dt^4), reported via a nq-doubling check where asked.
The Floquet matrix is the exact linearisation of that symplectic map, so multipliers of a stable
state sit on the unit circle to round-off.
"""

import numpy as np
from numba import njit

from swirl.string.core import LAWS, YOSHIDA, String, _cyc_solve


@njit(cache=True)
def _acc_jac(u, law, beta, invh, smooth, mlo, mcp, mdinv, mz, msc, Fb, dFb, a):
    N = u.shape[0]
    for j in range(N):
        jp = j + 1 if j + 1 < N else 0
        d = (u[jp] - u[j]) * invh
        if law == 1:
            if not (abs(d) < 1.0):
                return j, d
            q = 1.0 - d * d
            Fb[j] = d / q
            dFb[j] = (1.0 + d * d) / (q * q)
        elif law == 2:
            Fb[j] = d + beta * d * d * d
            dFb[j] = 1.0 + 3.0 * beta * d * d
        else:
            Fb[j] = d
            dFb[j] = 1.0
    a[0] = (Fb[0] - Fb[N - 1]) * invh
    for j in range(1, N):
        a[j] = (Fb[j] - Fb[j - 1]) * invh
    if smooth:
        _cyc_solve(mlo, mcp, mdinv, mz, msc, a)
    return -1, 0.0


@njit(cache=True)
def _japply(Tu, dFb, invh, smooth, mlo, mcp, mdinv, mz, msc, Ta, g):
    m, N = Tu.shape
    ih2 = invh * invh
    for c in range(m):
        for j in range(N):
            jp = j + 1 if j + 1 < N else 0
            g[j] = dFb[j] * (Tu[c, jp] - Tu[c, j])
        Ta[c, 0] = (g[0] - g[N - 1]) * ih2
        for j in range(1, N):
            Ta[c, j] = (g[j] - g[j - 1]) * ih2
        if smooth:
            _cyc_solve(mlo, mcp, mdinv, mz, msc, Ta[c])


@njit(cache=True)
def _tangent_run(u, v, Tu, Tv, nsteps, dt, W, law, beta, invh, smooth, mlo, mcp, mdinv, mz, msc):
    N = u.shape[0]
    m = Tu.shape[0]
    a = np.empty(N)
    Fb = np.empty(N)
    dFb = np.empty(N)
    g = np.empty(N)
    Ta = np.empty((m, N))
    j, d = _acc_jac(u, law, beta, invh, smooth, mlo, mcp, mdinv, mz, msc, Fb, dFb, a)
    if j >= 0:
        return 0, j, d
    _japply(Tu, dFb, invh, smooth, mlo, mcp, mdinv, mz, msc, Ta, g)
    for k in range(nsteps):
        for s in range(3):
            c = W[s] * dt
            for i in range(N):
                v[i] += 0.5 * c * a[i]
                u[i] += c * v[i]
            for r in range(m):
                for i in range(N):
                    Tv[r, i] += 0.5 * c * Ta[r, i]
                    Tu[r, i] += c * Tv[r, i]
            j, d = _acc_jac(u, law, beta, invh, smooth, mlo, mcp, mdinv, mz, msc, Fb, dFb, a)
            if j >= 0:
                return k, j, d
            _japply(Tu, dFb, invh, smooth, mlo, mcp, mdinv, mz, msc, Ta, g)
            for i in range(N):
                v[i] += 0.5 * c * a[i]
            for r in range(m):
                for i in range(N):
                    Tv[r, i] += 0.5 * c * Ta[r, i]
    return nsteps, -1, 0.0


class Guard(Exception):
    pass


def _flow(S, u0, v0, Tu, Tv, nsteps, dt):
    u, v = u0.copy(), v0.copy()
    M = S._M
    done, j, d = _tangent_run(u, v, Tu, Tv, int(nsteps), float(dt), YOSHIDA, S.lawc, S.beta,
                              1.0 / S.h, S.smooth, M[0], M[1], M[2], M[3], M[4])
    if j >= 0:
        raise Guard("bond %d stretch %.6g at step %d" % (j, d, done))
    return u, v


def quarter_map(S, u0, omega, nq):
    """Residual u(T/4) and its Jacobian d u(T/4) / d u0 (v0 = 0)."""
    N = S.N
    Tu = np.eye(N)
    Tv = np.zeros((N, N))
    dt = 2 * np.pi / omega / (4 * nq)
    u, _ = _flow(S, u0, np.zeros(N), Tu, Tv, nq, dt)
    return u, Tu.T.copy()


def monodromy(S, u0, omega, nq):
    """Full-period monodromy of the discrete map (2N x 2N, ordering [du; dv]) and the end state."""
    N = S.N
    Tu = np.zeros((2 * N, N))
    Tv = np.zeros((2 * N, N))
    Tu[:N] = np.eye(N)
    Tv[N:] = np.eye(N)
    dt = 2 * np.pi / omega / (4 * nq)
    u, v = _flow(S, u0, np.zeros(N), Tu, Tv, 4 * nq, dt)
    Mon = np.block([[Tu[:N].T, Tu[N:].T], [Tv[:N].T, Tv[N:].T]])
    return Mon, u, v


def choose_nq(S, u0, omega, wdt=0.08, extra=0.02, nq_min=32):
    dmax = float(np.max(np.abs(S.stretch(u0))))
    if S.law == "G1":
        dmax = min(dmax + extra, 0.5 * (1 + dmax))
    om = S.omega_max(dmax)
    T = 2 * np.pi / omega
    return max(nq_min, int(np.ceil(T / 4 / (wdt / om))))


def newton(S, u_guess, omega, nq, tol=1e-11, maxit=40):
    """Newton with step halving. Returns (u0, info)."""
    u = np.array(u_guess, dtype=float)
    hist = []
    try:
        R, J = quarter_map(S, u, omega, nq)
    except Guard as e:
        return u, {"converged": False, "reason": "guard at guess: %s" % e, "residuals": hist}
    for it in range(maxit):
        r = float(np.max(np.abs(R)))
        hist.append(r)
        if r < tol * max(1.0, float(np.max(np.abs(u)))):
            return u, {"converged": True, "iterations": it, "residual": r, "residuals": hist}
        try:
            du = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            return u, {"converged": False, "reason": "singular Jacobian", "residuals": hist}
        lam = 1.0
        while lam > 1e-4:
            try:
                R2, J2 = quarter_map(S, u + lam * du, omega, nq)
                if np.max(np.abs(R2)) < r or lam < 2e-4:
                    break
            except Guard:
                pass
            lam *= 0.5
        if lam <= 1e-4:
            return u, {"converged": False, "reason": "line search failed", "residuals": hist}
        u = u + lam * du
        R, J = R2, J2
    return u, {"converged": False, "reason": "maxit", "residuals": hist}


# ---------------------------------------------------------------- guesses
def F1(F, a, n=256):
    """First cosine coefficient of F(a cos theta) (rotating-wave force)."""
    th = 2 * np.pi * np.arange(n) / n
    return 2.0 * np.mean(F(a * np.cos(th)) * np.cos(th))


def rwa_amplitude(obj, law, beta, omega):
    """Grid rotating-wave amplitude A of the pattern (bump: -1/2,1,-1/2; sbend: -1,1) at rate omega.
    bump: omega^2 = 2 F1(1.5 A)/A ; sbend: omega^2 = (F1(A) + F1(2A))/A."""
    from scipy.optimize import brentq
    F = String("grid", law, 4, beta=beta).F
    if obj == "bump":
        g = lambda A: 2 * F1(F, 1.5 * A) / A - omega ** 2
        hi = (1 / 1.5) * (1 - 1e-9) if law == "G1" else 1e3
    else:
        g = lambda A: (F1(F, A) + F1(F, 2 * A)) / A - omega ** 2
        hi = 0.5 * (1 - 1e-9) if law == "G1" else 1e3
    return brentq(g, 1e-6, hi, xtol=1e-14)


def grid_pattern(N, obj, A, kappa=None):
    """Rotating-wave pattern on the ring, centre index c = N//2. bump: u_c = A, u_{c+-1} = -A/2.
    sbend: u_{c-1} = -A, u_c = A (centre at x = -1/2). Optional alternating exponential tail."""
    c = N // 2
    u = np.zeros(N)
    if obj == "bump":
        u[c] = A
        for n in range(1, N // 2):
            amp = 0.5 * A * (np.exp(-kappa * (n - 1)) if kappa else (1.0 if n == 1 else 0.0))
            u[(c + n) % N] = (-1) ** n * amp
            u[(c - n) % N] = (-1) ** n * amp
    else:
        u[c] = A
        u[c - 1] = -A
        if kappa:
            for n in range(1, N // 2 - 1):
                amp = A * np.exp(-kappa * n)
                u[(c + n) % N] = (-1) ** n * amp
                u[(c - 1 - n) % N] = -((-1) ** n) * amp
    return u


def kappa_grid(omega):
    """O1: tail decay per site, 2 cosh(kappa/2) = omega (omega > 2)."""
    return 2 * np.arccosh(omega / 2)


def kappa_smooth(omega):
    """O1: C-smooth tail decay rate kappa = omega / sqrt(omega^2 - 1) (omega > 1)."""
    return omega / np.sqrt(omega ** 2 - 1)


# ---------------------------------------------------------------- diagnostics of a periodic state
def l_half(x, e, h):
    """Half-length of the shortest interval holding half the energy (each point's energy spread
    uniformly over its cell of width h); and the rms energy width about the energy centroid."""
    e = np.clip(e, 0.0, None)
    E = e.sum()
    edges = np.concatenate([[x[0] - h / 2], x + h / 2])
    C = np.concatenate([[0.0], np.cumsum(e)])
    ok = C + 0.5 * E <= C[-1]             # left end at a cell edge; right end interpolated
    best = np.min(np.interp(C[ok] + 0.5 * E, C, edges) - edges[ok])
    ok = C - 0.5 * E >= 0                 # right end at a cell edge; left end interpolated
    best = min(best, np.min(edges[ok] - np.interp(C[ok] - 0.5 * E, C, edges)))
    xc = np.sum(e * x) / E
    rms = np.sqrt(np.sum(e * (x - xc) ** 2) / E)
    return 0.5 * best, rms, xc


def reflect_index(N, obj, smooth=False):
    """Index map of the reflection about the object's centre (grid bump: site c; grid sbend:
    bond c-1/2; C-smooth: both centred on the point c)."""
    c = N // 2
    n = np.arange(N)
    return (2 * c - n) % N if (obj == "bump" or smooth) else (2 * c - 1 - n) % N


def describe(S, u0, omega, nq, obj, floquet=True, nsamp=64):
    """Diagnostics of a periodic state (all measured from the solved orbit)."""
    out = {"omega": float(omega), "nq": int(nq)}
    T = 2 * np.pi / omega
    dt = T / (4 * nq)
    S.set_state(u0)
    E = S.energy()
    steps = 4 * nq
    every = max(1, steps // nsamp)
    while steps % every:
        every -= 1                                    # uniform samples over one period
    ebar = np.zeros(S.N)
    c1 = np.zeros(S.N)
    nl = 0.0
    dmax = 0.0
    ns = 0
    for k in range(0, steps, every):
        e = S.density()
        ebar += e
        c1 += S.u * np.cos(omega * k * dt)
        nl += (S.energy() - S.quadratic_energy()) / E
        dmax = max(dmax, float(np.max(np.abs(S.stretch()))))
        ns += 1
        S.advance(dt, min(every, steps - k))
    ebar /= ns
    ret = float(np.max(np.abs(S.u - u0)))
    out["profile_u0"] = u0.copy()
    lh, rms, xc = l_half(S.x, ebar, S.h)
    out.update({"E": float(E), "E_per_cycle": float(2 * np.pi * E / omega),
                "peak_stretch": dmax, "peak_displacement": float(np.max(np.abs(u0))),
                "L_half": float(lh), "rms_width": float(rms), "centroid": float(xc),
                "nonlinear_share": float(nl / ns), "period_return_error": ret})
    # tail decay of the component at the rate (Block 4 item 5), not of the envelope
    c1 *= 2.0 / ns
    dist = np.abs(S.x - xc)
    amp = np.abs(c1)
    sel = (dist > 2.5 * S.h if S.smooth else dist > 2.5) & (amp > 1e-11 * amp.max()) & \
          (amp < 1e-2 * amp.max()) & (dist < 0.4 * S.L)
    if sel.sum() >= 4:
        p = np.polyfit(dist[sel], np.log(amp[sel]), 1)
        out["tail_kappa_measured"] = float(-p[0])
    else:
        out["tail_kappa_measured"] = None
    amp0 = np.abs(u0)
    sel = sel & (amp0 > 1e-11 * amp0.max()) & (amp0 < 1e-2 * amp0.max())
    out["tail_kappa_envelope"] = float(-np.polyfit(dist[sel], np.log(amp0[sel]), 1)[0]) \
        if sel.sum() >= 4 else None
    out["tail_kappa_O1"] = float(kappa_grid(omega) if S.medium == "grid" else kappa_smooth(omega))
    if floquet:
        Mon, _, _ = monodromy(S, u0, omega, nq)
        lam = np.linalg.eigvals(Mon)
        i = int(np.argmax(np.abs(lam)))
        out["floquet_max_modulus"] = float(np.abs(lam[i]))
        out["floquet_max"] = [float(lam[i].real), float(lam[i].imag)]
        w, vecs = np.linalg.eig(Mon)
        vu = vecs[:S.N, int(np.argmax(np.abs(w)))]
        P = reflect_index(S.N, obj, S.smooth)
        sym = np.linalg.norm(vu + vu[P])
        anti = np.linalg.norm(vu - vu[P])
        out["floquet_unstable_mode_parity"] = "symmetric" if sym > anti else "antisymmetric"
        out["floquet_parity_ratio"] = float(min(sym, anti) / max(sym, anti))
        out["floquet_n_outside_1e-6"] = int(np.sum(np.abs(lam) > 1 + 1e-6))
        # time shift and uniform shift (ring) are exact symmetries: two Jordan pairs at +1, split
        # by round-off to ~sqrt(eps |M|). Largest modulus among the other multipliers:
        order = np.argsort(np.abs(lam - 1.0))
        rest = lam[order[4:]]
        out["floquet_max_modulus_excl_4_at_1"] = float(np.max(np.abs(rest)))
        dist = np.abs(lam[order] - 1.0)
        out["floquet_excluded_dists_from_1"] = [float(v) for v in dist[:4]]
        out["floquet_4_nearest_1_max_dev"] = float(dist[3])
        # a genuine multiplier close to 1 (e.g. within 3e-6) would show here, beside the 4 above
        out["floquet_nearest_kept_dist_from_1"] = float(dist[4])
        out["floquet_nearest_kept"] = [float(lam[order[4]].real), float(lam[order[4]].imag)]
    return out


def lone_rate(a, law="G1", beta=0.0):
    """O4: rate of the lone oscillator w_tt = -F(w) at amplitude a (energy V(a))."""
    from scipy.integrate import quad
    V = String("grid", law, 4, beta=beta).V
    f = lambda w: 1.0 / np.sqrt(2.0 * (V(a) - V(w)))
    I, _ = quad(f, 0, a, limit=400)
    return 2 * np.pi / (4 * I)


def smooth_attempt(h, law, obj, sigma, peak_stretch, omega=None, L=16.0, beta=0.0, floquet=True,
                   maxit=30):
    """Tool b: Newton for an exact periodic state of C-smooth ('fd' scheme at spacing h) from a
    Gaussian pluck guess (starts.gaussian) at rate omega (default: O4 lone-oscillator rate at the
    peak stretch). Returns what happens: convergence, residual history, and for a converged state
    its diagnostics and the conditioning of the quarter-map Jacobian (a near-singular Jacobian at
    the solution means the state is not isolated)."""
    from swirl.string import starts as ST
    N = int(round(L / h))
    N += N % 2
    S = String("fd", law, N, h=h, beta=beta)
    if omega is None:
        omega = lone_rate(peak_stretch, law, beta)
    g, _ = ST.gaussian(S, obj, sigma, peak_stretch=peak_stretch)
    nq = choose_nq(S, g, omega)
    u, info = newton(S, g, omega, nq, maxit=maxit)
    out = {"h": h, "law": law, "obj": obj, "sigma": sigma, "peak_stretch_guess": peak_stretch,
           "omega": omega, "N": N, "L": S.L, "nq": nq, "newton": info}
    if info["converged"]:
        _, J = quarter_map(S, u, omega, nq)
        sv = np.linalg.svd(J, compute_uv=False)
        out["jacobian_sv_min_over_max"] = float(sv[-1] / sv[0])
        out.update(describe(S, u, omega, nq, obj, floquet=floquet))
        out["profile_x"] = S.x
    return out


def orbit_at(law, obj, beta, omega, N, omega_from=2.3):
    """C-grid exact state at one rate on a ring of N: Newton from the rotating-wave guess, and if
    that fails, continuation in rate from omega_from. Returns (u0, nq, info)."""
    S = String("grid", law, N, beta=beta)
    A = rwa_amplitude(obj, law, beta, omega)
    g = grid_pattern(N, obj, A, kappa_grid(omega))
    nq = choose_nq(S, g, omega)
    u, info = newton(S, g, omega, nq)
    if info["converged"]:
        return u, nq, info
    lo, hi = (omega, omega_from) if omega < omega_from else (omega_from, omega)
    r = family("grid", law, obj, beta=beta, N=N, omega_start=omega_from, omega_min=lo,
               omega_stop=hi, d_stop=1.0, floquet=False)
    pts = [p for p in r["points"] if abs(p["omega"] - omega) < 1e-12]
    if not pts:
        raise RuntimeError("no exact state at omega=%g (direct Newton: %s)" % (omega, info))
    p = pts[0]
    return np.asarray(p["profile_u0"]), p["nq"], {"converged": True, "via": "continuation",
                                                  "residual": p["newton_residual"]}


# ---------------------------------------------------------------- families
def family(medium, law, obj, beta=0.0, h=1.0, N=None, omega_start=2.3, omega_min=2.01,
           d_stop=0.99, omega_stop=None, step0=0.02, floquet=True, max_points=200, log=None,
           u_start=None):
    """Continue a family in rate from omega_start down to omega_min and up until the peak stretch
    reaches d_stop (G1), omega_stop is passed, or Newton fails at the smallest step.
    C-grid only unless u_start is given (C-smooth, tool b)."""
    if N is None:
        N = 2 * int(np.ceil(14.0 / kappa_grid(omega_min))) + 41 if medium == "grid" else 256
        N += N % 2
    S = String(medium, law, N, h=h, beta=beta)
    if u_start is None:
        A = rwa_amplitude(obj, law, beta, omega_start)
        u_start = grid_pattern(N, obj, A, kappa_grid(omega_start))
    pts = []
    nq = choose_nq(S, u_start, omega_start)
    u0, info = newton(S, u_start, omega_start, nq)
    first = {"omega": omega_start, "newton": info}
    if not info["converged"]:
        return {"N": N, "points": [], "start_failed": first}
    info0, nq0 = info, nq
    for direction in (-1, +1):
        u, om, step, info, nq = u0.copy(), omega_start, step0, info0, nq0
        prev = None
        while len(pts) < max_points:
            d = describe(S, u, om, nq, obj, floquet=floquet)
            d["newton_residual"] = info.get("residual")
            d["newton_iterations"] = info.get("iterations")
            pts.append(d)
            if log:
                log("%s %s %s omega=%.5f peak_d=%.4f E=%.5g Lh=%.3f |lam|max-1=%.2e" % (
                    medium, law, obj, om, d["peak_stretch"], d["E"], d["L_half"],
                    d.get("floquet_max_modulus", 1.0) - 1))
            if direction < 0 and om <= omega_min + 1e-12:
                break
            if direction > 0 and (d["peak_stretch"] >= d_stop or (omega_stop and om >= omega_stop)):
                break
            ok = False
            while step > 1e-4:
                om_new = om + direction * step
                if direction < 0:
                    om_new = max(om_new, omega_min)
                guess = u if prev is None else u + (u - prev[0]) * (om_new - om) / (om - prev[1])
                nq2 = choose_nq(S, guess, om_new)
                try:
                    un, inf = newton(S, guess, om_new, nq2)
                except Guard:
                    inf = {"converged": False}
                if inf["converged"]:
                    prev = (u, om)
                    u, om, nq = un, om_new, nq2
                    info = inf
                    if inf["iterations"] <= 4:
                        step = min(step * 1.5, 0.2)
                    ok = True
                    break
                step *= 0.5
            if not ok:
                pts[-1]["end_reason"] = "Newton failed at step < 1e-4"
                break
        if direction < 0:
            pts.reverse()
            pts.pop()            # omega_start is recomputed going up
    pts.sort(key=lambda p: p["omega"])
    return {"N": N, "points": pts, "start": first}
