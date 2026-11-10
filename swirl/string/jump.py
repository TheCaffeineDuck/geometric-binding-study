"""Single-frequency (rotating-wave) standing states of C-smooth with G1: the jump profiles.

Derivation (docs/WPB1_BUILD_NOTES.md, section 6). With w = u_x the medium is
(1 - d_xx) w_tt = d_xx F(w). Keep one harmonic, w = a(x) cos(omega t), F(a cos th) ~ f(a) cos th,
    f(a) = (2/a) ((1 - a^2)^(-1/2) - 1)          (G1, exact first cosine coefficient)
Then (f(a) - omega^2 a)'' = -omega^2 a. With y = g(a) = f(a) - omega^2 a:
    y'' = -omega^2 a,    first integral  y'^2/2 + Phi(a) = C,    Phi(a) = omega^2 (a g(a) - G(a)),
    G(a) = int_0^a g = 2 ln(2/(1 + sqrt(1 - a^2))) - omega^2 a^2/2   (Phi is even in a).
g falls from 0 to its minimum at the fold A* (f'(A*) = omega^2) and rises again; it crosses zero at
A_z (f(A_z) = omega^2 A_z). A localized profile cannot be continuous: it would have to pass
y = g(A*), the lowest value g takes, with y' != 0 (Phi(A*) < 0). Profiles on the lower branch alone
cannot be localized (y y'' > 0). So a localized single-harmonic state has jumps in a, across which
y and y' are continuous (weak form) and a switches branch at fixed y. Integrating the equation:
int a dx = 0 exactly (zero net stretch).

Constructed here (parameter a_top in (A_z, 1)):
  'sbend' (a even, u odd): a = a_top at x = 0 on the upper branch, y' = 0; a falls to a_s where it
          jumps to the opposite-phase lower branch a_out < 0, g(a_out) = g(a_s); tail ~ exp(-kappa x).
  'bump'  (a odd, u even): a jumps from -A_z to +A_z at x = 0 (y = 0), rises on the upper branch to
          a_top, falls to a_s and jumps to the same tail.
Matching at the outer jump: Phi(a_top) - Phi(a_s) + Phi(a_out(a_s)) = 0 (C = Phi(a_top) inside,
C = 0 in the tail). For a given rate it is a one-parameter family (a_top, equivalently x_s).
"""

import numpy as np
from scipy.optimize import brentq


def f1(a):
    a = np.asarray(a, dtype=float)
    small = np.abs(a) < 1e-4
    with np.errstate(divide="ignore", invalid="ignore"):
        big = (2.0 / a) * (1.0 / np.sqrt(1.0 - a * a) - 1.0)
    return np.where(small, a + 0.75 * a ** 3, big)


def df1(a):
    a = np.asarray(a, dtype=float)
    q = 1.0 / np.sqrt(1.0 - a * a)
    small = np.abs(a) < 1e-4
    with np.errstate(divide="ignore", invalid="ignore"):
        big = -2.0 * (q - 1.0) / (a * a) + 2.0 * q ** 3
    return np.where(small, 1.0 + 2.25 * a * a, big)


class Profile:
    def __init__(self, omega):
        self.w2 = omega * omega
        self.omega = omega
        self.kappa = omega / np.sqrt(omega * omega - 1.0)
        self.A_star = brentq(lambda a: df1(a) - self.w2, 1e-12, 1 - 1e-15)
        self.A_z = brentq(lambda a: f1(a) / a - self.w2, self.A_star, 1 - 1e-15)
        self.y_min = float(self.g(self.A_star))

    def g(self, a):
        return f1(a) - self.w2 * np.asarray(a)

    def dg(self, a):
        return df1(a) - self.w2

    def Phi(self, a):
        """Even in a. Series below |a| = 1e-2 (the closed form cancels to round-off there):
        f = a + 3a^3/4 + 5a^5/8 + ..., so Phi = w2 ((1-w2) a^2/2 + 9 a^4/16 + 25 a^6/48 + ...)."""
        a = np.asarray(a, dtype=float)
        a2 = a * a
        G = 2.0 * np.log(2.0 / (1.0 + np.sqrt(1.0 - a2))) - 0.5 * self.w2 * a2
        closed = self.w2 * (a * self.g(a) - G)
        series = self.w2 * ((1.0 - self.w2) * a2 / 2 + 9.0 * a2 ** 2 / 16 + 25.0 * a2 ** 3 / 48)
        return np.where(np.abs(a) < 1e-2, series, closed)

    def a_out(self, y):
        """Opposite-phase lower branch: a in (-A*, 0) with g(a) = y (0 < y <= -y_min)."""
        return brentq(lambda a: self.g(a) - y, -self.A_star, 0.0, xtol=1e-15)

    def match(self, a_top):
        """Outer jump amplitude a_s on the upper branch, or None if no matching exists."""
        a_b = brentq(lambda a: self.g(a) + self.y_min, self.A_z, 1 - 1e-15) \
            if self.g(1 - 1e-12) > -self.y_min else 1.0
        hi = min(a_top, a_b)
        h = lambda s: self.Phi(a_top) - self.Phi(s) + self.Phi(self.a_out(float(self.g(s))))
        lo = self.A_z * (1 + 1e-12)
        if hi <= lo or h(lo) * h(hi * (1 - 1e-13)) > 0:
            return None
        return brentq(h, lo, hi * (1 - 1e-13), xtol=1e-15)

    def X(self, a_top, a, n=20000):
        """x-distance from a to a_top along the upper branch with C = Phi(a_top):
        int_a^{a_top} g'(s) / sqrt(2 (Phi(a_top) - Phi(s))) ds, via s = a_top - tau^2."""
        taus = np.linspace(0.0, np.sqrt(a_top - a), n)
        s = a_top - taus ** 2
        num = 2.0 * taus * self.dg(s)
        den = np.sqrt(np.maximum(2.0 * (self.Phi(a_top) - self.Phi(s)), 1e-300))
        integ = np.where(taus > 0, num / den, 2.0 * self.dg(a_top) /
                         np.sqrt(2.0 * self.w2 * a_top * self.dg(a_top)))
        xs = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(taus))])
        return s, xs          # a values (decreasing from a_top) and their distance from the top

    def tail(self, a_os, tau_max=40.0, n=20000):
        """Outer branch from a_os (< 0) toward 0: a = a_os exp(-tau); returns a, distance."""
        tau = np.linspace(0.0, tau_max, n)
        a = a_os * np.exp(-tau)
        integ = np.abs(self.dg(a)) * np.abs(a) / np.sqrt(np.maximum(-2.0 * self.Phi(a), 1e-300))
        xs = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(tau))])
        return a, xs


def stretch_profile(obj, omega, a_top, x):
    """Stretch amplitude a(x) of the jump profile at points x (centre x = 0), and its numbers."""
    P = Profile(omega)
    if not (P.A_z < a_top < 1):
        raise ValueError("a_top must lie in (A_z, 1) = (%.6f, 1)" % P.A_z)
    a_s = P.match(a_top)
    info = {"omega": omega, "A_star": P.A_star, "A_z": P.A_z, "a_top": a_top, "kappa": P.kappa,
            "exists": a_s is not None}
    if a_s is None:
        return None, info
    a_os = P.a_out(float(P.g(a_s)))
    s_in, x_in = P.X(a_top, a_s)                 # a from a_top down to a_s
    if obj == "bump":
        s_up, x_up = P.X(a_top, P.A_z)
        xm = x_up[-1]                            # distance from A_z (x = 0) to the top
        core_x = np.concatenate([xm - x_up[::-1], xm + x_in[1:]])
        core_a = np.concatenate([s_up[::-1], s_in[1:]])
    else:
        xm = 0.0
        core_x, core_a = x_in, s_in
    xs = core_x[-1]
    ta, tx = P.tail(a_os)
    ax = np.abs(x)
    a = np.where(ax <= xs, np.interp(ax, core_x, core_a), 0.0)
    out = ax > xs
    tt = ax[out] - xs
    at = np.interp(tt, tx, ta)
    far = tt > tx[-1]
    at[far] = ta[-1] * np.exp(-P.kappa * (tt[far] - tx[-1]))
    a[out] = at
    if obj == "bump":
        a = np.sign(x) * a
    info.update({"a_s": a_s, "a_out": a_os, "x_s": xs, "x_top": xm,
                 "jump_outer": a_s - a_os, "jump_centre": 2 * P.A_z if obj == "bump" else 0.0})
    return a, info


def x_s_of(obj, omega, a_top):
    """Distance of the outer jump from the centre, or None where no matching exists."""
    a, info = stretch_profile(obj, omega, a_top, np.zeros(1))
    return info["x_s"] if info["exists"] else None


def a_top_for(obj, omega, x_s, n=200):
    """a_top giving the outer jump at distance x_s; if x_s is beyond the family's reach, the
    a_top of the widest existing profile (reported by the caller)."""
    P = Profile(omega)
    tops = P.A_z + (1 - P.A_z) * np.linspace(0.01, 0.999, n) ** 2
    xs = [x_s_of(obj, omega, t) for t in tops]
    ok = [i for i, v in enumerate(xs) if v is not None and np.isfinite(v) and v < 100]
    tops, xs = tops[ok], np.array([xs[i] for i in ok])
    last = int(np.argmax(xs))
    if x_s >= xs[last]:
        return float(tops[last])
    j = int(np.argmax(xs >= x_s))
    if j == 0:
        return float(tops[0])
    return brentq(lambda t: x_s_of(obj, omega, t) - x_s, tops[j - 1], tops[j], xtol=1e-14)


def start(S, obj, omega, a_top=None, x_s=0.25):
    """Displacement u0 on the string S (at rest) whose stretch is the jump profile, sampled where
    the scheme keeps its stretch (fd: bond midpoints; spectral: points). Give a_top, or x_s (the
    distance of the outer jump from the centre, default 0.25, inside the 0.04 to 0.3 range of
    Block 4 O10 and one cell at h = 1/4). The net stretch left by sampling the jumps is removed
    uniformly (exact profiles have none) and reported; info["plateau_cells"] = 2 x_s / h says how
    many cells the plateau covers (below 1: the scheme cannot represent it)."""
    from_xs = a_top is None
    if from_xs:
        a_top = a_top_for(obj, omega, x_s)
    xb = S.x if S.medium == "spectral" else S.x + 0.5 * S.h
    a, info = stretch_profile(obj, omega, a_top, xb)
    if a is None:
        raise ValueError("no jump profile at omega=%g, a_top=%g" % (omega, a_top))
    info["x_s_requested"] = x_s if from_xs else None
    if from_xs and abs(info["x_s"] - x_s) > 1e-6:
        info["x_s_note"] = "requested x_s not reached; widest profile of the family used"
    info["plateau_cells"] = float(2 * info["x_s"] / S.h)
    area = S.h * a.sum()
    info["net_stretch_removed"] = float(area)
    a = a - area / S.L
    if S.medium == "spectral":
        u = np.concatenate([[0.0], np.cumsum(0.5 * S.h * (a[:-1] + a[1:]))])
    else:
        u = np.concatenate([[0.0], np.cumsum(S.h * a[:-1])])
    return u, 0.0, info
