"""Equation O, radial, single bump (s2 = 0) and pinch (s2 = 2), plan normalisation:

    f_tt = f_rr + (2/r) f_r - s2 f / r^2 - f + g4 f^3 - g6 f^5

in u = r f on the nodes r_i = i h, i = 0..M (the kernel of swirl/family/radial.py):

    u_tt = u_rr - s2 u / r^2 - r U'(u/r) - sigma(r) u_t,   U'(f) = f (1 - g4 f^2 + g6 f^4)

u_rr is the 4th-order central stencil. At r = 0, u is odd for the bump (u_{-k} = -u_k,
f even) and even for the pinch (u_{-k} = +u_k, u = r f ~ a r^2); u_0 = 0 in both. At
r = R a reflecting wall (u_{M+k} = -u_{M-k}, u_M = 0). Both reflections keep the
discrete D2 symmetric, so with sigma = 0 the discrete energy

    E_h = 4 pi h sum_{i=1}^{M-1} [ v^2/2 - u (D2 u)/2 + (s2/2) u^2/r^2 + r^2 U(u/r) ]

is conserved exactly by the semi-discrete system; RK4 adds an O(dt^4) error.

Map to swirl/family/radial.py (m = 1): g4_plan = g4_file / 6, g6_plan = g6_file / 120.
The Session 1 point (g4_file = 0.30, g6_file = 0.12, beta = 4/3) is g4 = 0.05, g6 = 0.001
(passed as 0.30/6 and 0.12/120 so that the arithmetic is bit-identical to radial.py).
"""

import numpy as np
from numba import njit

FOUR_PI = 4.0 * np.pi
SESSION1 = dict(g4=0.30 / 6.0, g6=0.12 / 120.0)


@njit(cache=True)
def _accel(u, v, r, c2, s2, par, g4, g6, sig, out):
    M = u.shape[0] - 1
    out[0] = 0.0
    out[M] = 0.0
    for i in range(1, M):
        um1 = u[i - 1]
        um2 = u[i - 2] if i >= 2 else par * u[1]
        up1 = u[i + 1]
        up2 = u[i + 2] if i + 2 <= M else -u[M - 1]
        lap = (-up2 + 16.0 * up1 - 30.0 * u[i] + 16.0 * um1 - um2) * c2
        p = u[i] / r[i]
        p2 = p * p
        vp = p * (1.0 - p2 * (g4 - g6 * p2))
        a = lap - r[i] * vp - sig[i] * v[i]
        if s2 != 0.0:
            a -= s2 * u[i] / (r[i] * r[i])
        out[i] = a


@njit(cache=True)
def _advance(u, v, r, c2, s2, par, g4, g6, sig, dt, nsteps, h, i_probe, core_rec, probe_rec):
    n = u.shape[0]
    a1 = np.empty(n); a2 = np.empty(n); a3 = np.empty(n); a4 = np.empty(n)
    ut = np.empty(n); vt = np.empty(n)
    for k in range(nsteps):
        _accel(u, v, r, c2, s2, par, g4, g6, sig, a1)
        for i in range(n):
            ut[i] = u[i] + 0.5 * dt * v[i]
            vt[i] = v[i] + 0.5 * dt * a1[i]
        _accel(ut, vt, r, c2, s2, par, g4, g6, sig, a2)
        k2u = vt.copy()
        for i in range(n):
            ut[i] = u[i] + 0.5 * dt * k2u[i]
            vt[i] = v[i] + 0.5 * dt * a2[i]
        _accel(ut, vt, r, c2, s2, par, g4, g6, sig, a3)
        k3u = vt.copy()
        for i in range(n):
            ut[i] = u[i] + dt * k3u[i]
            vt[i] = v[i] + dt * a3[i]
        _accel(ut, vt, r, c2, s2, par, g4, g6, sig, a4)
        for i in range(n):
            u[i] += dt / 6.0 * (v[i] + 2.0 * k2u[i] + 2.0 * k3u[i] + vt[i])
            v[i] += dt / 6.0 * (a1[i] + 2.0 * a2[i] + 2.0 * a3[i] + a4[i])
        u[0] = 0.0
        u[n - 1] = 0.0
        v[0] = 0.0
        v[n - 1] = 0.0
        if s2 == 0.0:
            # f(0) = lim u/r, 4th order: f even => (4 f_1 - f_2)/3
            core_rec[k] = (4.0 * u[1] / r[1] - u[2] / r[2]) / 3.0
        else:
            # pinch: u = a r^2 + b r^4 => a = f_r(0) = (16 u_1 - u_2) / (12 h^2)
            core_rec[k] = (16.0 * u[1] - u[2]) / (12.0 * h * h)
        probe_rec[k] = u[i_probe] / r[i_probe]


class SolverO:
    """obj: 'bump' or 'pinch'. Fields stored as u = r f, v = u_t."""

    model = "O"

    def __init__(self, obj, R, h, g4, g6, sponge_start=None, sigma_max=0.0, r_probe=None):
        assert obj in ("bump", "pinch")
        self.obj = obj
        self.s2 = 0.0 if obj == "bump" else 2.0
        self.par = -1.0 if obj == "bump" else 1.0
        self.M = int(round(R / h))
        self.h = R / self.M
        self.R = R
        self.r = np.arange(self.M + 1) * self.h
        self.r_safe = self.r.copy()
        self.r_safe[0] = 1.0
        self.g4, self.g6 = float(g4), float(g6)
        self.sig = np.zeros(self.M + 1)
        if sigma_max > 0:
            x = np.clip((self.r - sponge_start) / (R - sponge_start), 0.0, None)
            self.sig = sigma_max * x * x
        rp = 0.5 * R if r_probe is None else r_probe
        self.i_probe = int(np.clip(round(rp / self.h), 1, self.M - 1))
        self.r_probe = float(self.r[self.i_probe])
        self.u = np.zeros(self.M + 1)
        self.v = np.zeros(self.M + 1)
        self.t = 0.0

    # ---- state ----
    def set_f(self, f, ft=None):
        """f, ft: callables of r or arrays on the grid (f_t = 0 if None)."""
        F = f(self.r) if callable(f) else np.asarray(f, dtype=float)
        self.u = self.r * F
        self.u[0] = self.u[-1] = 0.0
        self.v = np.zeros_like(self.u)
        if ft is not None:
            G = ft(self.r) if callable(ft) else np.asarray(ft, dtype=float)
            self.v = self.r * G
            self.v[0] = self.v[-1] = 0.0
        self.t = 0.0

    def get_state(self):
        return dict(u=self.u.copy(), v=self.v.copy(), t=self.t)

    def put_state(self, st):
        self.u, self.v, self.t = st["u"].copy(), st["v"].copy(), float(st["t"])

    def step(self, dt, n):
        core = np.empty(n)
        probe = np.empty(n)
        _advance(self.u, self.v, self.r_safe, 1.0 / (12.0 * self.h ** 2), self.s2, self.par,
                 self.g4, self.g6, self.sig, dt, n, self.h, self.i_probe, core, probe)
        self.t += n * dt
        return core, probe

    def core(self):
        u, h = self.u, self.h
        if self.s2 == 0.0:
            return (4.0 * u[1] / self.r[1] - u[2] / self.r[2]) / 3.0
        return (16.0 * u[1] - u[2]) / (12.0 * h * h)

    # ---- energy ----
    def U(self, f):
        f2 = f * f
        return 0.5 * f2 - 0.25 * self.g4 * f2 * f2 + self.g6 * f2 ** 3 / 6.0

    def energy_discrete(self, u=None, v=None):
        """The exactly conserved semi-discrete energy E_h (numpy; complex-safe)."""
        u = self.u if u is None else u
        v = self.v if v is None else v
        uu, vv, r = u[1:-1], v[1:-1], self.r[1:-1]
        lap = self._D2np(u)[1:-1]
        return FOUR_PI * self.h * np.sum(0.5 * vv * vv - 0.5 * uu * lap
                                         + 0.5 * self.s2 * uu * uu / (r * r) + r * r * self.U(uu / r))

    def _D2np(self, u):
        """D2 in numpy with the same ghosts (works on complex arrays)."""
        M = self.M
        ext = np.concatenate([self.par * u[2:0:-1], u, -u[M - 1:M - 3:-1]])
        c = 1.0 / (12.0 * self.h ** 2)
        out = (-ext[4:] + 16.0 * ext[3:-1] - 30.0 * ext[2:-2] + 16.0 * ext[1:-3] - ext[:-4]) * c
        out[0] = out[-1] = 0.0
        return out

    def rhs(self, u=None, v=None, sponge=False):
        """(u_t, v_t) from the kernel (for the energy-rate test)."""
        u = self.u if u is None else u
        v = self.v if v is None else v
        a = np.empty_like(u)
        sig = self.sig if sponge else np.zeros_like(u)
        _accel(u, v, self.r_safe, 1.0 / (12.0 * self.h ** 2), self.s2, self.par, self.g4,
               self.g6, sig, a)
        vt = v.copy()
        vt[0] = vt[-1] = 0.0
        return vt, a

    def fields(self):
        """f, f_t, f_r on the nodes (4th order); values at r = 0 by regularity."""
        r, h, u, v = self.r_safe, self.h, self.u, self.v
        f, ft = u / r, v / r
        M = self.M
        ext = np.concatenate([self.par * u[2:0:-1], u, -u[M - 1:M - 3:-1]])
        ur = (ext[:-4] - 8.0 * ext[1:-3] + 8.0 * ext[3:-1] - ext[4:]) / (12.0 * h)
        fr = (ur - f) / r
        if self.s2 == 0.0:
            f[0] = (4 * f[1] - f[2]) / 3.0
            ft[0] = (4 * ft[1] - ft[2]) / 3.0
            fr[0] = 0.0
        else:
            f[0] = ft[0] = 0.0
            fr[0] = (16.0 * u[1] - u[2]) / (12.0 * h * h)
        return f, ft, fr

    def density(self):
        """Physical energy density e(r) = f_t^2/2 + S/2 + U(f) on the nodes, and the
        quadratic (linear-twin) part of it. e(0) is irrelevant (weight r^2 = 0)."""
        f, ft, fr = self.fields()
        r = self.r_safe
        S = fr * fr + self.s2 * f * f / (r * r)
        S[0] = fr[0] ** 2 * (1.0 + self.s2)
        quad = 0.5 * ft * ft + 0.5 * S + 0.5 * f * f
        e = quad - 0.25 * self.g4 * f ** 4 + self.g6 * f ** 6 / 6.0
        return e, quad

    def cumulative(self, e):
        """4 pi int_0^r e r^2 dr on the nodes (trapezoid; r^2 e is even at r = 0)."""
        g = FOUR_PI * e * self.r ** 2
        return np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * self.h)])

    def energy_inside_q(self, rcs):
        """E_in at each rc, E_total, and the quadratic energy (f, f_t) inside rcs[-1]."""
        e, quad = self.density()
        cum = self.cumulative(e)
        cq = self.cumulative(quad)
        return np.interp(rcs, self.r, cum), cum[-1], float(np.interp(rcs[-1], self.r, cq))

    def energy_inside(self, rcs):
        e, _ = self.density()
        cum = self.cumulative(e)
        return np.interp(rcs, self.r, cum), cum[-1]

    def energy_parts(self):
        e, quad = self.density()
        E = self.cumulative(e)[-1]
        Eq = self.cumulative(quad)[-1]
        return dict(E=float(E), E_quadratic=float(Eq), nonlinear_share_signed=float((E - Eq) / E))

    def high_k_fraction(self):
        """Share of the spectral power of f and f_t above half the grid wavenumber."""
        from scipy.fft import dst
        out = []
        for g in (self.u, self.v):
            if self.s2:
                g = g / self.r_safe                  # pinch: f = u/r is odd at 0 and at R
            c = dst(g[1:-1], type=1)                 # bump: u is odd at 0 and at R
            p = c * c
            out.append(float(np.sum(p[p.size // 2:]) / max(np.sum(p), 1e-300)))
        return max(out)
