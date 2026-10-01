"""Spherically symmetric solver for the full field equation

    phi_tt - phi_rr - (2/r) phi_r + m^2 phi - (g4/6) phi^3 + (g6/120) phi^5 = 0

in u = r phi on the uniform grid r_i = i dr, i = 0..M:

    u_tt = u_rr - r V'(u/r) - sigma(r) u_t

u_rr is the 4th-order central stencil with odd reflection about r = 0
(u_{-k} = -u_k, which makes phi even and regular) and about r = R
(u_{M+k} = -u_{M-k}, a reflecting wall, u_M = 0). Both reflections keep the
discrete operator D2 symmetric, so with sigma = 0 the discrete energy

    E_h = 4 pi dr sum_{i=1}^{M-1} [ v_i^2/2 - u_i (D2 u)_i / 2 + r_i^2 V(u_i/r_i) ]

is conserved exactly by the semi-discrete system; RK4 adds an O(dt^4) error.
sigma(r) = sigma_max ((r - r_s)/(R - r_s))^2 for r > r_s is the sponge layer
(it only removes energy). sigma_max = 0 switches it off.
"""

import numpy as np
from numba import njit

FOUR_PI = 4.0 * np.pi


@njit(cache=True)
def _accel(u, v, r, c2, m2, g4, g6, sig, out):
    M = u.shape[0] - 1
    out[0] = 0.0
    out[M] = 0.0
    for i in range(1, M):
        um1 = u[i - 1]
        um2 = u[i - 2] if i >= 2 else -u[1]
        up1 = u[i + 1]
        up2 = u[i + 2] if i + 2 <= M else -u[M - 1]
        lap = (-up2 + 16.0 * up1 - 30.0 * u[i] + 16.0 * um1 - um2) * c2
        p = u[i] / r[i]
        p2 = p * p
        vp = p * (m2 - p2 * (g4 / 6.0 - g6 / 120.0 * p2))
        out[i] = lap - r[i] * vp - sig[i] * v[i]


@njit(cache=True)
def _advance(u, v, r, c2, m2, g4, g6, sig, dt, nsteps, phi0_rec):
    """nsteps of RK4. phi0_rec[k] gets phi(0) after step k+1."""
    n = u.shape[0]
    a1 = np.empty(n); a2 = np.empty(n); a3 = np.empty(n); a4 = np.empty(n)
    ut = np.empty(n); vt = np.empty(n)
    for k in range(nsteps):
        _accel(u, v, r, c2, m2, g4, g6, sig, a1)
        for i in range(n):
            ut[i] = u[i] + 0.5 * dt * v[i]
            vt[i] = v[i] + 0.5 * dt * a1[i]
        _accel(ut, vt, r, c2, m2, g4, g6, sig, a2)
        k2u = vt.copy()
        for i in range(n):
            ut[i] = u[i] + 0.5 * dt * k2u[i]
            vt[i] = v[i] + 0.5 * dt * a2[i]
        _accel(ut, vt, r, c2, m2, g4, g6, sig, a3)
        k3u = vt.copy()
        for i in range(n):
            ut[i] = u[i] + dt * k3u[i]
            vt[i] = v[i] + dt * a3[i]
        _accel(ut, vt, r, c2, m2, g4, g6, sig, a4)
        for i in range(n):
            u[i] += dt / 6.0 * (v[i] + 2.0 * k2u[i] + 2.0 * k3u[i] + vt[i])
            v[i] += dt / 6.0 * (a1[i] + 2.0 * a2[i] + 2.0 * a3[i] + a4[i])
        u[0] = 0.0
        u[n - 1] = 0.0
        v[0] = 0.0
        v[n - 1] = 0.0
        # phi(0) = lim u/r, 4th order: phi even => (4 phi_1 - phi_2)/3
        phi0_rec[k] = (4.0 * u[1] / r[1] - u[2] / r[2]) / 3.0


class Radial:
    def __init__(self, R, dr, m, g4, g6, sponge_start=None, sigma_max=0.0):
        self.M = int(round(R / dr))
        self.dr = R / self.M
        self.R = R
        self.r = np.arange(self.M + 1) * self.dr
        self.r_safe = self.r.copy()
        self.r_safe[0] = 1.0                       # never used for division at i = 0
        self.m, self.g4, self.g6 = m, g4, g6
        self.sig = np.zeros(self.M + 1)
        self.sponge_start = sponge_start
        self.sigma_max = sigma_max
        if sigma_max > 0:
            x = np.clip((self.r - sponge_start) / (R - sponge_start), 0.0, None)
            self.sig = sigma_max * x * x
        self.u = np.zeros(self.M + 1)
        self.v = np.zeros(self.M + 1)
        self.t = 0.0

    def set_phi(self, phi, phit=None):
        """phi, phit: callables of r (or arrays on the grid)."""
        p = phi(self.r) if callable(phi) else np.asarray(phi)
        self.u = self.r * p
        self.u[0] = self.u[-1] = 0.0
        if phit is not None:
            q = phit(self.r) if callable(phit) else np.asarray(phit)
            self.v = self.r * q
            self.v[0] = self.v[-1] = 0.0
        else:
            self.v = np.zeros_like(self.u)
        self.t = 0.0

    def step(self, dt, n):
        rec = np.empty(n)
        _advance(self.u, self.v, self.r_safe, 1.0 / (12.0 * self.dr**2), self.m**2,
                 self.g4, self.g6, self.sig, dt, n, rec)
        self.t += n * dt
        return rec

    def V(self, p):
        p2 = p * p
        return 0.5 * self.m**2 * p2 - self.g4 / 24.0 * p2 * p2 + self.g6 / 720.0 * p2**3

    def D2(self, u):
        a = np.empty_like(u)
        _accel(u, np.zeros_like(u), self.r_safe, 1.0 / (12.0 * self.dr**2), 0.0, 0.0,
               0.0, np.zeros_like(u), a)
        return a

    def energy_discrete(self):
        """The exactly conserved (semi-discrete) energy E_h."""
        u, v, r = self.u[1:-1], self.v[1:-1], self.r[1:-1]
        lap = self.D2(self.u)[1:-1]
        return FOUR_PI * self.dr * np.sum(0.5 * v * v - 0.5 * u * lap + r * r * self.V(u / r))

    def fields(self):
        r = self.r_safe
        phi = self.u / r
        pt = self.v / r
        phi[0] = (4 * phi[1] - phi[2]) / 3.0
        pt[0] = (4 * pt[1] - pt[2]) / 3.0
        return phi, pt

    def density(self):
        """Physical energy density e(r) and the nonlinear part of it."""
        phi, pt = self.fields()
        pr = np.gradient(phi, self.dr, edge_order=2)
        pr[0] = 0.0
        p2 = phi * phi
        nl = -self.g4 / 24.0 * p2 * p2 + self.g6 / 720.0 * p2**3
        e = 0.5 * pt * pt + 0.5 * pr * pr + 0.5 * self.m**2 * p2 + nl
        return e, nl

    def energy_inside(self, rcs):
        """E(r < rc) = 4 pi int_0^rc e r^2 dr for each rc (trapezoid, interpolated)."""
        e, _ = self.density()
        f = FOUR_PI * e * self.r**2
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * self.dr)])
        return np.interp(rcs, self.r, cum), cum[-1]

    def nonlinear_energy(self):
        _, nl = self.density()
        f = FOUR_PI * nl * self.r**2
        return float(np.sum(0.5 * (f[1:] + f[:-1]) * self.dr))


def core_frequency(t, phi0):
    """Angular frequency from upward zero crossings of phi(0, t).

    Returns crossing times (midpoints of consecutive upward crossings) and
    omega = 2 pi / (period) for each full period. Linear interpolation of the
    crossing time between samples.
    """
    s = np.sign(phi0)
    idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    tc = t[idx] - phi0[idx] * (t[idx + 1] - t[idx]) / (phi0[idx + 1] - phi0[idx])
    if tc.size < 2:
        return np.array([]), np.array([])
    return 0.5 * (tc[1:] + tc[:-1]), 2 * np.pi / np.diff(tc)
