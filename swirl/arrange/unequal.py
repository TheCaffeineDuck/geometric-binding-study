"""WP-P (Session 3, P1): point-lump model for lumps of different N and omega, and for moving lumps.

Derivation and validity: docs/WPP_BUILD_NOTES.md. Summary.

State per lump i: position X_i, canonical momentum P_i, rotating-frame phase phi_i = theta_i - omega_ref t,
and the lump frequency W_i = omega_i (N_i = N(W_i) along the family; (phi_i, N_i) are the canonical pair).
Hamiltonian, first order in the tail overlap:

    H = sum_i [ sqrt(P_i^2 + E_i^2) - omega_ref N_i ] + sum_{i<j} U_ij

    U_ij = -(2 pi s A_i A_j / d) [ w_i q_ij e^{-kappa_j d} cos(D_ij - delta_j) + w_j q_ji e^{-kappa_i d} cos(D_ij - delta_i) ]

with d = |X_j - X_i|, D_ij = phi_i - phi_j, w_i = W_i/(W_i + W_j), q_ij = Q_i(kappa_j)/A_i,
Q_i(k) = (1/4 pi) int S_i(r) sinh(k r)/(k r) d^3x (S_i the lump's own nonlinear source; Q_i(kappa_i) = A_i),
and the Lorentz boost phase of lump j at the other lump, delta_j = W_j P_j.(X_j - X_i)/E_j (= W_j gamma_j V_j.r).
(Default form: the co-moving part of the two boost phases exactly, the counter-moving part to first order;
see pair_energy.)
Every lump property (E, N, kappa = sqrt(m^2 - W^2), A, hw) follows W along the beta = 4/3 family (Chebyshev
fits in omega of solved profiles; E is the exact integral of omega dN, so dE/dN = omega).

Equations: X' = dH/dP, P' = -dH/dX, phi' = dH/dN = (dH/dW)/N'(W) (= W/gamma + dU/dN - omega_ref),
N' = -dH/dphi, i.e. W' = -(dH/dphi)/N'(W). The gradient is taken by complex-step differentiation of the coded H
(exact to rounding), so H, sum N, total canonical momentum and angular momentum are conserved to integrator
tolerance; the physics is entirely in H.

Equal-lump reduction: Family.frozen(params) with relativistic=False, boost=False is the Session 2 model
(swirl/arrange/model.py, Model A) exactly.
"""
import json
import os

import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .pairlaw import ROOT

BETA, M, G4 = 4.0 / 3.0, 1.0, 0.30
G6 = 0.09 * BETA
OMEGA_GRID = np.round(np.arange(0.850, 0.94501, 0.0025), 6)
TABLE_PATH = os.path.join(ROOT, "outputs", "session3", "wpP", "family_coupling_beta1.3333.json")
H_CS = 1e-30          # complex step
VALID_HW = 3.6        # tail law within 10% beyond this spacing (WPC_DERIVATION section 6)


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    os.replace(path + ".tmp", path)


# ------------------------------------------------------------------ family table from solved profiles
def build_family_table(omegas=OMEGA_GRID, path=TABLE_PATH):
    """Solve the profile at each omega (beta = 4/3, Set-B-like units) and tabulate N, E, A, kappa, hw and the
    coupling charges Q[i][j] = Q_i(kappa_j) = int_0^inf S_i(r) r sinh(kappa_j r)/kappa_j dr."""
    from swirl.family.profile import next_guess, solve_profile
    rows, S_list, x_list = [], [], []
    guess = None
    for w in sorted(omegas, reverse=True):
        r = solve_profile(float(w), m=M, g4=G4, g6=G6, guess=guess, tol=1e-10)
        guess = next_guess(r)
        sol, R = r["_sol"], r["R_box"]
        x = np.linspace(0.0, R, 200001)
        P = sol.sol(x)[0]
        S_list.append((G4 / 8) * P ** 3 - (G6 / 192) * P ** 5)
        x_list.append(x)
        rows.append({k: r[k] for k in ("omega", "N", "E", "A", "kappa", "half_width", "A_rel_spread", "P0", "R_box")})
    kap = np.array([q["kappa"] for q in rows])
    Q = np.zeros((len(rows), len(rows)))
    for i in range(len(rows)):
        x, S = x_list[i], S_list[i]
        for j in range(len(rows)):
            Q[i, j] = np.trapezoid(S * x * np.sinh(kap[j] * x) / kap[j], x)
    out = {"what": "beta = 4/3 family, stable branch: profile observables and coupling charges Q_i(kappa_j)",
           "units": "Set-B-like: m = 1, g4 = 0.30, g6 = 0.12", "beta": BETA, "m": M, "g4": G4, "g6": G6,
           "solver": "swirl.family.profile.solve_profile, tol 1e-10; Q by trapezoid on 200001 points to R_box",
           "rows": rows, "Q": Q, "Q_self_over_A_minus_1": [Q[i, i] / rows[i]["A"] - 1 for i in range(len(rows))]}
    dump(out, path)
    return out


def load_table(path=TABLE_PATH):
    if not os.path.exists(path):
        return build_family_table(path=path)
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ the family as functions of omega
class Family:
    """Lump properties as analytic (complex-safe) functions of the lump frequency W = omega. The family is smooth
    in omega; N(omega) has a fold at N_min, so omega, not N, is the fitting variable. Family.from_table() is the
    full model; Family.frozen(params) is the Session 2 law (fixed A, kappa, hw; linear omega(N); equal weights;
    q = 1)."""

    DEG_1D, DEG_2D = 18, 12

    def __init__(self):
        self.coupling_scale = 1.0
        self.frozen_law = False

    @classmethod
    def from_table(cls, table=None, coupling_scale=1.0):
        t = load_table() if table is None else table
        f = cls()
        f.table = t
        f.coupling_scale = float(coupling_scale)
        rows = t["rows"]
        om = np.array([r["omega"] for r in rows])
        f.W_lo, f.W_hi = float(om.min()), float(om.max())
        x = f._x(om)
        N = np.array([r["N"] for r in rows])
        A = np.array([r["A"] for r in rows])
        hw = np.array([r["half_width"] for r in rows])
        E = np.array([r["E"] for r in rows])
        f.c_N = C.chebfit(x, N, cls.DEG_1D)
        f.c_lnA = C.chebfit(x, np.log(A), cls.DEG_1D)
        f.c_hw = C.chebfit(x, hw, cls.DEG_1D)
        f.c_dN = C.chebder(f.c_N) * 2.0 / (f.W_hi - f.W_lo)            # dN/domega, in x
        # E(omega) = E(omega_ref) + int omega (dN/domega) domega, so dE/dN = omega exactly
        mid, half = (f.W_hi + f.W_lo) / 2.0, (f.W_hi - f.W_lo) / 2.0
        dNdx = C.chebder(f.c_N)
        f.c_E = C.chebint(C.chebadd(mid * dNdx, half * C.chebmulx(dNdx)))
        k_ref = int(np.argmin(np.abs(om - 0.91)))
        f.c_E[0] += E[k_ref] - C.chebval(x[k_ref], f.c_E)
        Q = np.array(t["Q"])
        qv = Q / A[:, None]
        Xi, Xj = np.meshgrid(x, x, indexing="ij")
        V = C.chebvander2d(Xi.ravel(), Xj.ravel(), [cls.DEG_2D, cls.DEG_2D])
        coef, *_ = np.linalg.lstsq(V, qv.ravel(), rcond=None)
        f.c_q = coef.reshape(cls.DEG_2D + 1, cls.DEG_2D + 1)
        f.fit_residuals = {
            "N_max_rel": float(np.max(np.abs(C.chebval(x, f.c_N) / N - 1))),
            "A_max_rel": float(np.max(np.abs(np.exp(C.chebval(x, f.c_lnA)) / A - 1))),
            "hw_max_abs": float(np.max(np.abs(C.chebval(x, f.c_hw) - hw))),
            "E_max_rel": float(np.max(np.abs(C.chebval(x, f.c_E) / E - 1))),
            "q_max_abs": float(np.max(np.abs(C.chebval2d(Xi, Xj, f.c_q) - qv))),
        }
        f.E_ref = float(E[k_ref])
        return f

    @classmethod
    def frozen(cls, p):
        f = cls()
        f.frozen_law = True
        f.A0, f.kap0, f.hw0, f.E_ref = p["A"], p["kappa"], p["hw"], p["E"]
        f.om0, f.w0, f.N0 = p["omega"], p["domega_dN"], p["N"]
        f.W_lo, f.W_hi = -np.inf, np.inf
        f.fit_residuals = {}
        return f

    def _x(self, W):
        return (2.0 * W - (self.W_hi + self.W_lo)) / (self.W_hi - self.W_lo)

    def lump(self, W):
        """(kappa, A, E, N, dN/domega) for array W = omega (complex allowed)."""
        if self.frozen_law:
            nu = (W - self.om0) / self.w0
            z = 0 * W
            return (self.kap0 + z, self.A0 + z, self.E_ref + self.om0 * nu + 0.5 * self.w0 * nu ** 2,
                    self.N0 + nu, 1.0 / self.w0 + z)
        x = self._x(W)
        return (np.sqrt(M * M - W * W), np.exp(C.chebval(x, self.c_lnA)), C.chebval(x, self.c_E),
                C.chebval(x, self.c_N), C.chebval(x, self.c_dN))

    def N(self, W):
        return self.lump(np.asarray(W, float))[3]

    def E(self, W):
        return self.lump(np.asarray(W, float))[2]

    def A(self, W):
        return self.lump(np.asarray(W, float))[1]

    def hw(self, W):
        if self.frozen_law:
            return self.hw0 + 0 * np.asarray(W, float)
        return C.chebval(self._x(np.asarray(W, float)), self.c_hw)

    def q(self, Wi, Wj):
        if self.frozen_law:
            return 1.0 + 0 * Wi * Wj
        return C.chebval2d(self._x(Wi), self._x(Wj), self.c_q)

    def domega_dN(self, W):
        return 1.0 / float(self.lump(np.asarray(W, float))[4])

    def omega_of_N(self, N):
        if self.frozen_law:
            return self.om0 + self.w0 * (N - self.N0)
        return brentq(lambda w: float(self.N(w)) - N, self.W_lo, self.W_hi, xtol=1e-15)


# ------------------------------------------------------------------ Hamiltonian
def split(Z, n):
    return Z[..., :3 * n], Z[..., 3 * n:6 * n], Z[..., 6 * n:7 * n], Z[..., 7 * n:]


def pair_energy(X, P, ph, W, fam, boost=True, parts=False):
    """Sum of U_ij. X, P: (..., n, 3); ph, W: (..., n). Complex-safe.

    boost: the Lorentz boost phase of each lump's tail at the other lump, delta_j = W_j P_j.r/E_j (seen at core i)
    and delta_i = W_i P_i.r/E_i (seen at core j). With mean shift m = (delta_i + delta_j)/2 and e = (delta_j -
    delta_i)/2 (co-moving and counter-moving parts along the line of centres):
      True / "mean"   core i: cos(D - m) + e sin(D - m); core j: cos(D - m) - e sin(D - m). Exact for a uniformly
                      moving (boosted) equal pair, exact at first order in the velocities; the counter-moving part
                      enters only at first order (its O(e^2) effect is left out: see WPP_BUILD_NOTES section 3).
      "resummed"      core i: cos(D - delta_j); core j: cos(D - delta_i). Superposed boosted profiles at face value;
                      for head-on motion it overstates the e^2 effect of the superposition quadrature by about 1/0.69.
      False           no boost phase (the Session 2 law)."""
    n = X.shape[-2]
    i, j = np.triu_indices(n, 1)
    kap, A, E, N, dN = fam.lump(W)
    r = X[..., j, :] - X[..., i, :]
    d = np.sqrt(np.sum(r * r, axis=-1))
    D = ph[..., i] - ph[..., j]
    if fam.frozen_law:
        wi = wj = 0.5
    else:
        wi = W[..., i] / (W[..., i] + W[..., j])
        wj = 1.0 - wi
    qij, qji = fam.q(W[..., i], W[..., j]), fam.q(W[..., j], W[..., i])
    pref = -2.0 * np.pi * fam.coupling_scale * A[..., i] * A[..., j] / d
    ai = wi * qij * np.exp(-kap[..., j] * d)       # lump i's core in lump j's tail
    aj = wj * qji * np.exp(-kap[..., i] * d)
    if boost:
        dj = W[..., j] * np.sum(P[..., j, :] * r, axis=-1) / E[..., j]
        di = W[..., i] * np.sum(P[..., i, :] * r, axis=-1) / E[..., i]
        if boost == "resummed":
            U = pref * (ai * np.cos(D - dj) + aj * np.cos(D - di))
        else:
            mshift, e = 0.5 * (di + dj), 0.5 * (dj - di)
            c, s = np.cos(D - mshift), np.sin(D - mshift)
            U = pref * (ai * (c + e * s) + aj * (c - e * s))
    else:
        U = pref * (ai + aj) * np.cos(D)
    if parts:
        return U, d
    return np.sum(U, axis=-1)


def hamiltonian(Z, n, fam, omega_ref, relativistic=True, boost=True):
    Xf, Pf, ph, W = split(Z, n)
    X = Xf.reshape(Xf.shape[:-1] + (n, 3))
    P = Pf.reshape(Pf.shape[:-1] + (n, 3))
    kap, A, E, N, dN = fam.lump(W)
    p2 = np.sum(P * P, axis=-1)
    kin = np.sqrt(p2 + E * E) if relativistic else E + p2 / (2.0 * fam.E_ref)
    return np.sum(kin - omega_ref * N, axis=-1) + pair_energy(X, P, ph, W, fam, boost)


def grad_H(z, n, fam, omega_ref, relativistic=True, boost=True):
    Zc = z[None, :] + 1j * H_CS * np.eye(z.size)
    return hamiltonian(Zc, n, fam, omega_ref, relativistic, boost).imag / H_CS


def rhs_from_grad(g, z, n, fam, frozen=False):
    """X' = H_P, P' = -H_X; with (phi, N) canonical and N = N(W): phi' = H_W/N'(W), W' = -H_phi/N'(W)."""
    out = np.empty_like(g)
    out[:3 * n] = g[3 * n:6 * n]
    out[3 * n:6 * n] = -g[:3 * n]
    if frozen:
        out[6 * n:] = 0.0
    else:
        dN = fam.lump(z[7 * n:])[4]
        out[6 * n:7 * n] = g[7 * n:] / dN
        out[7 * n:] = -g[6 * n:7 * n] / dN
    return out


class Model:
    """One configuration of the extended model. State z = (X [3n], P [3n], phi [n], W [n])."""

    def __init__(self, fam, omega_ref, relativistic=True, boost=True, frozen=False):
        self.fam, self.omega_ref = fam, float(omega_ref)
        self.rel, self.boost, self.frozen = relativistic, boost, frozen

    def H(self, z, n):
        return float(hamiltonian(np.asarray(z, float), n, self.fam, self.omega_ref, self.rel, self.boost))

    def grad(self, z, n):
        return grad_H(np.asarray(z, float), n, self.fam, self.omega_ref, self.rel, self.boost)

    def rhs(self, t, z, n):
        return rhs_from_grad(self.grad(z, n), z, n, self.fam, self.frozen)

    @staticmethod
    def pack(X, P, ph, W):
        return np.concatenate([np.asarray(X, float).ravel(), np.asarray(P, float).ravel(),
                               np.asarray(ph, float), np.asarray(W, float)])

    def momenta_for_velocities(self, X, Vdot, ph, W, tol=1e-14, itmax=200):
        """Canonical momenta P such that X' = dH/dP equals Vdot (lab velocity of each lump centre). At rest
        (Vdot = 0) P is not zero when the overlap carries momentum (sine of the phase difference)."""
        n = len(X)
        E = self.fam.E(W)
        Vt = np.asarray(Vdot, float).copy()
        V = Vt.copy()
        for _ in range(itmax):
            v2 = np.sum(V * V, 1, keepdims=True)
            P = (E[:, None] * V / np.sqrt(1 - v2)) if self.rel else self.fam.E_ref * V
            xd = self.grad(self.pack(X, P, ph, W), n)[3 * n:6 * n].reshape(n, 3)
            res = xd - Vt
            if np.max(np.abs(res)) < tol:
                return P
            V = V - res
        raise RuntimeError("momenta_for_velocities did not converge (%.3g)" % np.max(np.abs(res)))

    def integrate(self, z0, n, T, t_eval=None, contact=True, far=None, rtol=1e-10, atol=1e-10):
        """DOP853. Terminal events: contact (a pair closer than hw_i + hw_j) and optionally far (every pair
        farther than `far`)."""
        fam = self.fam
        i, j = np.triu_indices(n, 1)
        events = []
        if contact:
            def hit(t, z, *_):
                X = z[:3 * n].reshape(n, 3)
                hw = fam.hw(z[7 * n:])
                return np.min(np.linalg.norm(X[i] - X[j], axis=1) - (hw[i] + hw[j]))
            hit.terminal, hit.direction = True, -1
            events.append(hit)
        if far is not None:
            def gone(t, z, *_):
                X = z[:3 * n].reshape(n, 3)
                return np.min(np.linalg.norm(X[i] - X[j], axis=1)) - far
            gone.terminal, gone.direction = True, 1
            events.append(gone)
        return solve_ivp(self.rhs, [0.0, T], np.asarray(z0, float), method="DOP853", rtol=rtol, atol=atol,
                         t_eval=t_eval, events=events or None, args=(n,))

    def conserved(self, z, n):
        X = z[:3 * n].reshape(n, 3)
        P = z[3 * n:6 * n].reshape(n, 3)
        return {"H": self.H(z, n), "N": float(np.sum(self.fam.N(z[7 * n:]))), "P": P.sum(0),
                "L": np.cross(X, P).sum(0)}


# ------------------------------------------------------------------ geometry helpers
def ring_positions(K, d):
    """K lumps on a circle in the xy plane, neighbour distance d (K = 2: a pair on the x axis)."""
    R = d / (2.0 * np.sin(np.pi / K))
    a = 2 * np.pi * np.arange(K) / K
    return np.c_[R * np.cos(a), R * np.sin(a), np.zeros(K)], R


def tangents(X):
    t = np.c_[-X[:, 1], X[:, 0], np.zeros(len(X))]
    return t / np.linalg.norm(t, axis=1, keepdims=True)


# ------------------------------------------------------------------ rigidly rotating equal rings
def rotating_ring(mdl, K, d, theta, W, sense=+1, n_scan=400):
    """Rigid rotation of an equal ring (K >= 2, lumps at frequency W) with lab centre phases theta, about +z
    (sense +1) or -z. Solves for the tangential canonical momentum p that balances the radial force; returns
    None when no balance exists for 0 < p < 0.3 E. Returns z, Omega_r (signed, about +z), lab speed v, T_orb,
    phidot (the common phase rate; set omega_ref to it to make phi stationary) and the residual."""
    X, R = ring_positions(K, d)
    tg = sense * tangents(X)
    rh = X / np.linalg.norm(X, axis=1, keepdims=True)
    Wv = np.full(K, float(W))
    E0 = float(mdl.fam.E(Wv)[0])

    def state(p):
        return mdl.pack(X, p * tg, theta, Wv)

    def f(p):
        g = mdl.grad(state(p), K)
        xd = g[3 * K:6 * K].reshape(K, 3)
        Pd = -g[:3 * K].reshape(K, 3)
        Om_r = np.sum(xd[0] * tg[0]) / R
        return np.sum(Pd[0] * rh[0]) + Om_r * p      # P' must be Omega_r (axis x P): radial part -Omega_r p
    ps = np.geomspace(1e-4 * E0, 0.3 * E0, n_scan)
    fs = np.array([f(p) for p in ps])
    k = np.where(np.sign(fs[:-1]) != np.sign(fs[1:]))[0]
    if len(k) == 0:
        return None
    p = brentq(f, ps[k[0]], ps[k[0] + 1], xtol=1e-15, rtol=1e-15)
    z = state(p)
    g = mdl.grad(z, K)
    xd = g[3 * K:6 * K].reshape(K, 3)
    Om_r = sense * np.sum(xd[0] * tg[0]) / R
    phidot = rhs_from_grad(g, z, K, mdl.fam)[6 * K:7 * K]
    return {"z": z, "R": R, "p": p, "Omega_r": float(Om_r), "v": float(np.linalg.norm(xd[0])),
            "T_orb": float(2 * np.pi / abs(Om_r)), "phidot": float(np.mean(phidot)),
            "phidot_spread": float(np.ptp(phidot)),
            "radial_residual_rel": float(abs(f(p)) / abs(p * Om_r)),
            "n_roots_found": int(len(k))}


def rotating_frame_jacobian(mdl, z, n, Om_r, rel_step=1e-6):
    """Jacobian of the flow of H - Om_r L_z at z (central differences of the exact complex-step gradient)."""
    def rhs_rot(zz):
        g = mdl.grad(zz, n).copy()
        X = zz[:3 * n].reshape(n, 3)
        P = zz[3 * n:6 * n].reshape(n, 3)
        g[:3 * n] -= Om_r * np.c_[P[:, 1], -P[:, 0], np.zeros(n)].ravel()
        g[3 * n:6 * n] -= Om_r * np.c_[-X[:, 1], X[:, 0], np.zeros(n)].ravel()
        return rhs_from_grad(g, zz, n, mdl.fam)
    scale = np.r_[np.full(3 * n, 10.0), np.full(3 * n, 100.0), np.ones(n), np.full(n, 0.01)]
    J = np.zeros((z.size, z.size))
    for k in range(z.size):
        h = rel_step * max(abs(z[k]), scale[k]) if k < 7 * n else rel_step * scale[k]
        e = np.zeros(z.size)
        e[k] = h
        J[:, k] = (rhs_rot(z + e) - rhs_rot(z - e)) / (2 * h)
    return J, rhs_rot(z)


def classify_modes(J, n, kappa, tol=1e-7):
    """Unstable eigenvalues of J, fastest first, with the position share kappa^2|dx|^2/(kappa^2|dx|^2 + |dphi|^2)
    of each eigenvector (Session 2 convention)."""
    lam, vec = np.linalg.eig(J)
    out = []
    for k in np.argsort(-lam.real):
        if lam[k].real <= tol:
            break
        v = vec[:, k]
        dx = np.sum(np.abs(v[:3 * n]) ** 2)
        dph = np.sum(np.abs(v[6 * n:7 * n]) ** 2)
        share = kappa ** 2 * dx / (kappa ** 2 * dx + dph) if (dx + dph) > 0 else np.nan
        out.append({"re": float(lam[k].real), "im": float(lam[k].imag), "position_share": float(share)})
    return out
