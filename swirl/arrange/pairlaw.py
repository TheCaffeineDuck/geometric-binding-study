"""Pair laws for the point-lump model.

Every pair energy is written as a short cosine series in the phase difference D = theta_i - theta_j:

    u(d, D) = sum_{k=0..3} b_k(d) cos(k D)

Model A (asymptotic tail law, WPC_DERIVATION.md section 1):
    b_1(d) = -C exp(-kappa d)/d,  C = 2 pi A^2, all other b_k = 0.
Model B (Model A plus the full-superposition correction from derive_pair_law.F_sup):
    the period-averaged superposition force is exactly a cosine series of order 3 in D for a sextic V,
    F_sup(d, D) = sum_k a_k(d) cos(k D) (F > 0 attracts, F = +du/dd). The correction
    da_k = a_k,sup - a_k,asym (both for the profile) is splined on [D_MIN, D_MAX] and integrated
    from D_MAX inward, b_k,corr(d) = -int_d^D_MAX da_k. Model B = Model A(params) + correction(profile).
    It is still incomplete at short range: each core polarizes in the other's tail (O(e^{-2 kappa d})),
    and below about d = 8 the mid-plane lies inside the cores.
"""
import importlib.util
import json
import os

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DERIVE = os.path.join(ROOT, "outputs", "session2", "wpC", "derive_pair_law.py")
KMAX = 3
D_MIN, D_MAX, D_STEP = 3.0, 40.0, 0.25
FIT_DEG = [0, 45, 90, 135, 180]


def load_params(path):
    """Parameter file: A, kappa, E (inertia), N (per lump), hw, omega, domega_dN. Everything else is ignored."""
    with open(path) as f:
        p = json.load(f)
    for k in ("A", "kappa", "E", "N", "hw", "omega", "domega_dN"):
        if k not in p:
            raise KeyError("params file lacks " + k)
    return p


def _load_derive():
    spec = importlib.util.spec_from_file_location("derive_pair_law", DERIVE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_fsup_table(d_grid=None):
    """Harmonic decomposition of F_sup(d, D) from derive_pair_law (the profile used for the 3D runs).
    Returns a dict with d, a_k (k = 0..3), the asymptotic F of the same profile and the fit residual."""
    mod = _load_derive()
    if d_grid is None:
        d_grid = np.arange(D_MIN, D_MAX + 1e-9, D_STEP)
    ph = np.deg2rad(FIT_DEG)
    M = np.cos(np.outer(ph, np.arange(KMAX + 1)))
    a = np.zeros((KMAX + 1, len(d_grid)))
    resid = np.zeros(len(d_grid))
    for n, d in enumerate(d_grid):
        F = np.array([mod.F_sup(d, 0.0, p) for p in ph])
        coef, *_ = np.linalg.lstsq(M, F, rcond=None)
        a[:, n] = coef
        resid[n] = np.max(np.abs(M @ coef - F)) / max(np.max(np.abs(F)), 1e-300)
    # true-profile overlap O(d) = int Phi(|x|) Phi(|x - d z|) d^3x (midpoint rule in rho, z; h = 0.05), for the
    # angular momentum of stepped patterns; the asymptotic value is 2 pi A^2 exp(-kappa d)/kappa
    h = 0.05
    rho = np.arange(0, 45, h) + h / 2
    O = []
    for d in d_grid:
        z = np.arange(-45, d + 45, h) + h / 2
        r1 = np.sqrt(rho[:, None] ** 2 + z[None, :] ** 2)
        r2 = np.sqrt(rho[:, None] ** 2 + (z[None, :] - d) ** 2)
        O.append(float(np.sum(2 * np.pi * rho[:, None] * mod.Phi(r1) * mod.Phi(r2)) * h * h))
    return {"d": d_grid.tolist(), "a": a.tolist(), "overlap_true": O,
            "overlap_asym": [float(2 * np.pi * mod.A ** 2 * np.exp(-mod.kap * d) / mod.kap) for d in d_grid],
            "F_asym_profile": [float(mod.F_asym(d)) for d in d_grid],
            "fit_max_rel_residual": resid.tolist(), "fit_phases_deg": FIT_DEG,
            "profile": {"A": mod.A, "kappa": mod.kap, "omega": mod.om, "E": mod.E_l, "N": mod.N_l, "hw": mod.hw,
                        "file": os.path.relpath(mod.PROFILE, ROOT)}}


class PairLaw:
    """u(d, D) = sum_k b_k(d) cos(k D). model 'A' or 'B'. b(d) returns (b, b', b'') each shape (4, len(d))."""

    def __init__(self, A, kappa, model="A", fsup=None):
        self.A, self.kappa, self.model = float(A), float(kappa), model
        self.C = 2 * np.pi * self.A ** 2
        self.d_min = 0.0
        if model == "B":
            if fsup is None:
                raise ValueError("model B needs the F_sup table")
            dg = np.asarray(fsup["d"])
            a = np.asarray(fsup["a"])
            pa = fsup["profile"]
            Cp, kp = 2 * np.pi * pa["A"] ** 2, pa["kappa"]
            a_asym = np.zeros_like(a)
            a_asym[1] = Cp * np.exp(-kp * dg) * (kp / dg + 1 / dg ** 2)
            da = a - a_asym
            self._spl = [CubicSpline(dg, da[k]) for k in range(KMAX + 1)]
            self._anti = [s.antiderivative() for s in self._spl]
            self._dspl = [s.derivative() for s in self._spl]
            self.d_min, self.d_max = float(dg[0]), float(dg[-1])
        elif model != "A":
            raise ValueError(model)

    def b(self, d):
        d = np.asarray(d, float)
        k, C = self.kappa, self.C
        e = np.exp(-k * d)
        b0 = np.zeros((KMAX + 1,) + d.shape)
        b1 = np.zeros_like(b0)
        b2 = np.zeros_like(b0)
        b0[1] = -C * e / d
        b1[1] = C * e * (k / d + 1 / d ** 2)
        b2[1] = -C * e * (k ** 2 / d + 2 * k / d ** 2 + 2 / d ** 3)
        if self.model == "B":
            inside = d <= self.d_max
            dc = np.clip(d, self.d_min, self.d_max)
            bad = d < self.d_min - 1e-12
            for q in range(KMAX + 1):
                corr = -(self._anti[q](self.d_max) - self._anti[q](dc))
                b0[q] += np.where(inside, corr, 0.0)
                b1[q] += np.where(inside, self._spl[q](dc), 0.0)
                b2[q] += np.where(inside, self._dspl[q](dc), 0.0)
                b0[q][bad] = np.nan
                b1[q][bad] = np.nan
                b2[q][bad] = np.nan
        return b0, b1, b2

    def pair(self, d, D):
        """u, u_d, u_dd, u_D, u_DD, u_dD for arrays d, D of equal shape."""
        b0, b1, b2 = self.b(d)
        kk = np.arange(KMAX + 1).reshape((-1,) + (1,) * np.ndim(d))
        c, s = np.cos(kk * D), np.sin(kk * D)
        return ((b0 * c).sum(0), (b1 * c).sum(0), (b2 * c).sum(0),
                -(kk * b0 * s).sum(0), -(kk ** 2 * b0 * c).sum(0), -(kk * b1 * s).sum(0))


def overlap_derivative(fsup):
    """d O_true/dd from the profile table (cubic spline), for the true-profile angular momentum."""
    return CubicSpline(np.asarray(fsup["d"]), np.asarray(fsup["overlap_true"])).derivative()


def make_laws(p, fsup):
    laws = {"A": PairLaw(p["A"], p["kappa"], "A"), "B": PairLaw(p["A"], p["kappa"], "B", fsup)}
    for law in laws.values():
        law.dOdd = overlap_derivative(fsup)
    return laws
