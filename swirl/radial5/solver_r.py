"""Equation R, radial, single bump (s2 = 0) and pinch (s2 = 2), in Hamiltonian form.

    L_R = 1 - sqrt(1 - X) - f^2/2,   X = f_t^2 - S,   S = f_r^2 + s2 f^2 / r^2
    p   = f_t / sqrt(1 - X)                     (momentum density conjugate to f)
    e   = sqrt((1 + S)(1 + p^2)) - 1 + f^2/2    (energy density, exact)
    1 - X = (1 + S) / (1 + p^2) > 0            for every finite (f, p)

    f_t = p sqrt((1 + S)/(1 + p^2)) = p / q,    q = sqrt((1 + p^2)/(1 + S)) = (1 - X)^(-1/2)
    p_t = r^-2 (r^2 q f_r)_r - s2 q f / r^2 - f - sigma(r) p

Discretisation (semi-discrete Hamiltonian, exact energy conservation without sponge):
cell centres r_i = (i + 1/2) h, i = 0..M-1, R = M h, weights w_i = r_i^2 h (midpoint rule,
spectrally accurate for the even integrand r^2 e). D is the 4th-order central first
derivative with ghosts from the parity of f: even at r = 0 for the bump (f_{-1-k} = f_k),
odd for the pinch (f_{-1-k} = -f_k), odd about r = R (a reflecting wall, f(R) = 0).

    H_h = 4 pi sum_i w_i e_i(f, p),   S_i = (D f)_i^2 + s2 f_i^2 / r_i^2
    f_t,i = dH_h/dP_i,  p_t,i = -(1/(4 pi w_i)) dH_h/df_i   (P_i = 4 pi w_i p_i)
          = -(1/w_i) (D^T (w q D f))_i - s2 q_i f_i / r_i^2 - f_i - sigma_i p_i

kg=True replaces e by the Klein-Gordon density p^2/2 + S/2 + f^2/2 (q = 1): the linear
twin of R on the same grid and the same discrete operators.

Guard (every RK4 stage, every node): f and p finite, 1 - X = (1+S)/(1+p^2) > 0 (X < 1,
the hyperbolicity condition of the radial equation and of the bump in the full
three-component theory), for the pinch also 1 - X > (f/r)^2 (equivalently
f_t^2 < 1 + f_r^2 + (f/r)^2: hyperbolicity of the three-component field for angular
wave vectors, which the radial equation cannot see; scholar DF-5, re-derived in
docs/WPB2_BUILD_NOTES.md), and q = (1 - X)^(-1/2) below q_max (momentum growing without
bound). The extremes min(1 - X), max|p| and the pinch margin, with their radii, are
returned for every chunk (self.last_ext) and logged by the driver in every run. Every record (rec_dt): the share of spectral
power of f and p above half the grid wavenumber below hk_max (loss of resolution, a
steepening front). A trip raises GuardTrip with a diagnosis; nothing is clipped.
"""

import numpy as np
from numba import njit

FOUR_PI = 4.0 * np.pi
OFFS = np.array([-2, -1, 1, 2])


class GuardTrip(RuntimeError):
    def __init__(self, diag):
        self.diag = diag
        super().__init__("guard tripped: %s" % {k: v for k, v in diag.items()
                                               if not isinstance(v, (list, np.ndarray))})


@njit(cache=True)
def _rhs(f, p, r, w, s2, kg, sig, idx, sgn, C, qmax, Df, g, ft, pt, fail, ext):
    """Right-hand side and guard. ext (running extremes, updated in place):
    [min(1 - X), node, max|p|, node, min(1 - X - s2/2 f^2/r^2), node]. Status: 0 ok;
    1 non-finite or 1 - X <= 0; 2 q > qmax; 3 pinch angular hyperbolicity lost
    (1 - X <= (f/r)^2, i.e. f_t^2 >= 1 + f_r^2 + (f/r)^2)."""
    n = f.shape[0]
    for i in range(n):
        d = 0.0
        for k in range(4):
            d += C[k] * sgn[i, k] * f[idx[i, k]]
        Df[i] = d
    status = 0
    for i in range(n):
        fr2 = f[i] * f[i] / (r[i] * r[i])
        S = Df[i] * Df[i] + s2 * fr2
        om = (1.0 + S) / (1.0 + p[i] * p[i])              # = 1 - X
        mg = om - 0.5 * s2 * fr2                           # pinch angular margin
        if om < ext[0]:
            ext[0] = om; ext[1] = i
        ap = abs(p[i])
        if ap > ext[2]:
            ext[2] = ap; ext[3] = i
        if mg < ext[4]:
            ext[4] = mg; ext[5] = i
        if kg:
            q = 1.0
            ft[i] = p[i]
        else:
            if not (om > 0.0) or not np.isfinite(om) or not np.isfinite(f[i]):
                if status == 0:
                    status = 1
                    fail[0] = i
                q = 1.0
            else:
                q = 1.0 / np.sqrt(om)
                if q > qmax and status == 0:
                    status = 2
                    fail[0] = i
                if s2 > 0.0 and not (mg > 0.0) and status == 0:
                    status = 3
                    fail[0] = i
            ft[i] = p[i] / q
        g[i] = w[i] * q * Df[i]
        pt[i] = -q * s2 * f[i] / (r[i] * r[i]) - f[i] - sig[i] * p[i]
    for i in range(n):
        for k in range(4):
            j = idx[i, k]
            pt[j] -= C[k] * sgn[i, k] * g[i] / w[j]
    return status


@njit(cache=True)
def _advance(f, p, r, w, s2, kg, sig, idx, sgn, C, qmax, dt, nsteps, par0, h, i_probe,
             core_rec, probe_rec, fail, ext, om_rec):
    """RK4. Returns -1 on success, else the step index k at which a stage failed
    (f, p then hold the last accepted state; fail = [node, code, stage])."""
    n = f.shape[0]
    Df = np.empty(n); g = np.empty(n)
    k1f = np.empty(n); k1p = np.empty(n); k2f = np.empty(n); k2p = np.empty(n)
    k3f = np.empty(n); k3p = np.empty(n); k4f = np.empty(n); k4p = np.empty(n)
    tf = np.empty(n); tp = np.empty(n)
    nd = np.zeros(1, dtype=np.int64)
    es = np.empty(6)
    for k in range(nsteps):
        es[0] = np.inf; es[1] = -1.0; es[2] = 0.0; es[3] = -1.0; es[4] = np.inf; es[5] = -1.0
        st = _rhs(f, p, r, w, s2, kg, sig, idx, sgn, C, qmax, Df, g, k1f, k1p, nd, es)
        if st:
            fail[0] = nd[0]; fail[1] = st; fail[2] = 1
            _fold(ext, es)
            return k
        for i in range(n):
            tf[i] = f[i] + 0.5 * dt * k1f[i]
            tp[i] = p[i] + 0.5 * dt * k1p[i]
        st = _rhs(tf, tp, r, w, s2, kg, sig, idx, sgn, C, qmax, Df, g, k2f, k2p, nd, es)
        if st:
            fail[0] = nd[0]; fail[1] = st; fail[2] = 2
            _fold(ext, es)
            return k
        for i in range(n):
            tf[i] = f[i] + 0.5 * dt * k2f[i]
            tp[i] = p[i] + 0.5 * dt * k2p[i]
        st = _rhs(tf, tp, r, w, s2, kg, sig, idx, sgn, C, qmax, Df, g, k3f, k3p, nd, es)
        if st:
            fail[0] = nd[0]; fail[1] = st; fail[2] = 3
            _fold(ext, es)
            return k
        for i in range(n):
            tf[i] = f[i] + dt * k3f[i]
            tp[i] = p[i] + dt * k3p[i]
        st = _rhs(tf, tp, r, w, s2, kg, sig, idx, sgn, C, qmax, Df, g, k4f, k4p, nd, es)
        if st:
            fail[0] = nd[0]; fail[1] = st; fail[2] = 4
            _fold(ext, es)
            return k
        for i in range(n):
            f[i] += dt / 6.0 * (k1f[i] + 2.0 * k2f[i] + 2.0 * k3f[i] + k4f[i])
            p[i] += dt / 6.0 * (k1p[i] + 2.0 * k2p[i] + 2.0 * k3p[i] + k4p[i])
        if par0 > 0:
            core_rec[k] = (9.0 * f[0] - f[1]) / 8.0            # f(0), f even: c + d r^2
        else:
            core_rec[k] = (27.0 * f[0] - f[1]) / (12.0 * h)    # f_r(0), f odd: a r + b r^3
        probe_rec[k] = f[i_probe]
        om_rec[k] = es[0]                                       # min over the step's stages
        _fold(ext, es)
    return -1


@njit(cache=True)
def _fold(ext, es):
    if es[0] < ext[0]:
        ext[0] = es[0]; ext[1] = es[1]
    if es[2] > ext[2]:
        ext[2] = es[2]; ext[3] = es[3]
    if es[4] < ext[4]:
        ext[4] = es[4]; ext[5] = es[5]


class SolverR:
    """obj: 'bump' or 'pinch'. kg=True: Klein-Gordon twin on the same discretisation."""

    model = "R"

    def __init__(self, obj, R, h, kg=False, sponge_start=None, sigma_max=0.0, r_probe=None,
                 qmax=1e4, hk_max=1e-6):
        assert obj in ("bump", "pinch")
        self.obj, self.kg = obj, bool(kg)
        self.s2 = 0.0 if obj == "bump" else 2.0
        self.par = 1.0 if obj == "bump" else -1.0          # parity of f at r = 0
        self.M = int(round(R / h))
        self.h = R / self.M
        self.R = R
        self.r = (np.arange(self.M) + 0.5) * self.h
        self.edges = np.arange(self.M + 1) * self.h
        self.w = self.r ** 2 * self.h
        self.sig = np.zeros(self.M)
        self.r_b = 0.9 * R
        if sigma_max > 0:
            x = np.clip((self.r - sponge_start) / (R - sponge_start), 0.0, None)
            self.sig = sigma_max * x * x
            self.r_b = sponge_start
        self.qmax, self.hk_max = float(qmax), float(hk_max)
        n = self.M
        self.idx = np.empty((n, 4), dtype=np.int64)
        self.sgn = np.empty((n, 4))
        for i in range(n):
            for k, o in enumerate(OFFS):
                j, s = i + o, 1.0
                if j < 0:
                    j, s = -1 - j, self.par
                elif j > n - 1:
                    j, s = 2 * n - 1 - j, -1.0
                self.idx[i, k], self.sgn[i, k] = j, s
        self.C = np.array([1.0, -8.0, 8.0, -1.0]) / (12.0 * self.h)
        rp = 0.5 * R if r_probe is None else r_probe
        self.i_probe = int(np.clip(np.searchsorted(self.r, rp), 0, n - 1))
        self.r_probe = float(self.r[self.i_probe])
        self.f = np.zeros(n)
        self.p = np.zeros(n)
        self.t = 0.0

    # ---- state ----
    def set_f(self, f, ft=None):
        """Start from f(r) and f_t(r) (default 0). p = f_t / sqrt(1 - X) needs X < 1 at
        every node: a start outside the domain trips the guard at t = 0."""
        self.f = f(self.r) if callable(f) else np.asarray(f, dtype=float).copy()
        self.t = 0.0
        if ft is None:
            self.p = np.zeros_like(self.f)
            return
        g = ft(self.r) if callable(ft) else np.asarray(ft, dtype=float)
        if self.kg:
            self.p = g.copy()
            return
        S = self.S(self.f)
        om = 1.0 + S - g * g                               # 1 - X
        mg = om - 0.5 * self.s2 * self.f ** 2 / self.r ** 2  # pinch: f_t^2 < 1 + f_r^2 + (f/r)^2
        if self.s2 and not np.all(mg > 0):
            i = int(np.argmin(mg))
            raise GuardTrip(dict(kind="pinch start outside the angular hyperbolicity domain",
                                 t=0.0, r=float(self.r[i]), X=float(1.0 - om[i]),
                                 pinch_margin=float(mg[i]), energy_so_far=None,
                                 f=float(self.f[i]), f_t=float(g[i]), S=float(S[i])))
        if not np.all(om > 0):
            i = int(np.argmin(om))
            raise GuardTrip(dict(kind="start outside the domain X < 1", t=0.0,
                                 r=float(self.r[i]), X=float(1.0 - om[i]),
                                 energy_so_far=None, f=float(self.f[i]), f_t=float(g[i]),
                                 S=float(S[i])))
        self.p = g / np.sqrt(om)

    def get_state(self):
        return dict(f=self.f.copy(), p=self.p.copy(), t=self.t)

    def put_state(self, st):
        self.f, self.p, self.t = st["f"].copy(), st["p"].copy(), float(st["t"])

    def step(self, dt, n, ckpt_dir=None):
        core = np.empty(n)
        probe = np.empty(n)
        fail = np.zeros(3, dtype=np.int64)
        ext = np.array([np.inf, -1.0, 0.0, -1.0, np.inf, -1.0])
        om = np.full(n, np.nan)
        k = _advance(self.f, self.p, self.r, self.w, self.s2, self.kg, self.sig, self.idx,
                     self.sgn, self.C, self.qmax, dt, n, self.par, self.h, self.i_probe,
                     core, probe, fail, ext, om)
        self.last_om = om                    # per-step min(1 - X) over the stages
        # extremes over every stage of this chunk: min(1 - X), max|p|, pinch margin, and where
        self.last_ext = dict(min_one_minus_X=float(ext[0]), r_min_one_minus_X=float(self.r[int(ext[1])]),
                             max_abs_p=float(ext[2]), r_max_abs_p=float(self.r[int(ext[3])]),
                             min_pinch_margin=float(ext[4]) if self.s2 else None,
                             r_min_pinch_margin=float(self.r[int(ext[5])]) if self.s2 else None)
        if k >= 0:
            self.t += k * dt
            i, code, stage = int(fail[0]), int(fail[1]), int(fail[2])
            raise GuardTrip(self._diagnosis(
                {1: "1 - X <= 0 or non-finite (ill-posed)",
                 2: "q = (1 - X)^-1/2 above q_max: momentum growing without bound",
                 3: "pinch: angular hyperbolicity lost, f_t^2 >= 1 + f_r^2 + (f/r)^2 "
                    "(1 - X <= (f/r)^2)"}[code], i, stage, ckpt_dir))
        self.t += n * dt
        hk = self.high_k_fraction()
        if hk > self.hk_max and not self.kg:
            raise GuardTrip(self._diagnosis("loss of resolution: high-k power share %.3g > %.3g"
                                            % (hk, self.hk_max), None, None, ckpt_dir, hk=hk))
        return core, probe

    def _diagnosis(self, kind, i, stage, ckpt_dir, hk=None):
        S = self.S(self.f)
        om = (1.0 + S) / (1.0 + self.p ** 2)
        if i is None:
            i = int(np.argmin(om))
        mg = om - 0.5 * self.s2 * self.f ** 2 / self.r ** 2
        d = dict(kind=kind, t=float(self.t), r=float(self.r[i]), node=int(i), stage=stage,
                 X=float(1.0 - om[i]), q=float(1.0 / np.sqrt(om[i])) if om[i] > 0 else None,
                 pinch_margin=float(mg[i]) if self.s2 else None,
                 min_one_minus_X=float(np.min(om)), max_abs_p=float(np.max(np.abs(self.p))),
                 max_abs_fr=float(np.max(np.abs(self.Df(self.f)))),
                 energy_so_far=float(self.energy_discrete()),
                 high_k=self.high_k_fraction() if hk is None else hk)
        if ckpt_dir is not None:
            import os
            os.makedirs(ckpt_dir, exist_ok=True)
            path = os.path.join(ckpt_dir, "guard_trip_state_t%.3f.npz" % self.t)
            np.savez(path, f=self.f, p=self.p, r=self.r, t=self.t)
            d["last_state"] = path
        else:
            d["last_state"] = dict(f_max=float(np.max(np.abs(self.f))), r_of_f_max=float(
                self.r[np.argmax(np.abs(self.f))]))
        return d

    def core(self):
        f, h = self.f, self.h
        return (9.0 * f[0] - f[1]) / 8.0 if self.par > 0 else (27.0 * f[0] - f[1]) / (12.0 * h)

    # ---- derivatives and energy (numpy; complex-safe) ----
    def Df(self, f):
        M = self.M
        ext = np.concatenate([self.par * f[1::-1], f, -f[M - 1:M - 3:-1]])
        return (ext[:-4] - 8.0 * ext[1:-3] + 8.0 * ext[3:-1] - ext[4:]) / (12.0 * self.h)

    def S(self, f):
        d = self.Df(f)
        return d * d + self.s2 * f * f / (self.r * self.r)

    def density(self, f=None, p=None):
        """Energy density e_i and its quadratic part at the cell centres. The quadratic part
        is the Klein-Gordon energy in the Lagrangian variables, f_t^2/2 + S/2 + f^2/2 with
        f_t = p/q (the same definition as the profile tables), not in p."""
        f = self.f if f is None else f
        p = self.p if p is None else p
        S = self.S(f)
        if self.kg:
            quad = 0.5 * p * p + 0.5 * S + 0.5 * f * f
            return quad, quad
        ft2 = p * p * (1.0 + S) / (1.0 + p * p)            # f_t^2 from Hamilton's equation
        quad = 0.5 * ft2 + 0.5 * S + 0.5 * f * f            # Klein-Gordon energy in (f, f_t)
        return np.sqrt((1.0 + S) * (1.0 + p * p)) - 1.0 + 0.5 * f * f, quad

    def energy_discrete(self, f=None, p=None):
        e, _ = self.density(f, p)
        return FOUR_PI * np.sum(self.w * e)

    def rhs(self, f=None, p=None, sponge=False):
        f = self.f if f is None else f
        p = self.p if p is None else p
        n = f.size
        Df, g, ft, pt = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
        sig = self.sig if sponge else np.zeros(n)
        _rhs(f, p, self.r, self.w, self.s2, self.kg, sig, self.idx, self.sgn, self.C, np.inf,
             Df, g, ft, pt, np.zeros(1, dtype=np.int64), np.array([np.inf, 0, 0, 0, np.inf, 0.0]))
        return ft, pt

    def X(self):
        """X = f_t^2 - S at the nodes, from the state (always < 1)."""
        S = self.S(self.f)
        return 1.0 - (1.0 + S) / (1.0 + self.p ** 2)

    def energy_inside_q(self, rcs):
        """E_in at each rc, E_total, and the quadratic energy (f, f_t) inside rcs[-1]."""
        e, quad = self.density()
        cum = np.concatenate([[0.0], np.cumsum(FOUR_PI * self.w * e)])
        cq = np.concatenate([[0.0], np.cumsum(FOUR_PI * self.w * quad)])
        return np.interp(rcs, self.edges, cum), cum[-1], float(np.interp(rcs[-1], self.edges, cq))

    def energy_inside(self, rcs):
        e, _ = self.density()
        cum = np.concatenate([[0.0], np.cumsum(FOUR_PI * self.w * e)])
        return np.interp(rcs, self.edges, cum), cum[-1]

    def energy_parts(self):
        e, quad = self.density()
        E = FOUR_PI * np.sum(self.w * e)
        Eq = FOUR_PI * np.sum(self.w * quad)
        return dict(E=float(E), E_quadratic=float(Eq), nonlinear_share_signed=float((E - Eq) / E),
                    X_max=float(np.max(self.X())), q_max=float(np.max(np.sqrt(
                        (1 + self.p ** 2) / (1 + self.S(self.f))))))

    def high_k_fraction(self):
        """Share of the spectral power of (f, p) above half the grid wavenumber, over the
        interior r < r_b (smooth cos^2 taper from 0.8 r_b to r_b; r_b = the sponge start,
        or 0.9 R in a closed box) so that the wall does not count. Transform: DCT-IV for
        the bump (even at r = 0), DST-II for the pinch (odd at r = 0)."""
        from scipy.fft import dct, dst
        rb = self.r_b
        ra = 0.8 * rb
        x = np.clip((self.r - ra) / (rb - ra), 0.0, 1.0)
        win = np.cos(0.5 * np.pi * x) ** 2
        hi = tot = 0.0
        for g in (self.f, self.p):
            c = dct(g * win, type=4) if self.par > 0 else dst(g * win, type=2)
            pw = c * c
            hi += np.sum(pw[pw.size // 2:])
            tot += np.sum(pw)
        return float(hi / tot) if tot > 0 else 0.0
