"""Point-lump Hamiltonian model.

State per lump: position x_i, momentum p_i, rotating-frame phase phi_i = theta_i - omega0 t, charge offset
nu_i = N_i - N0. With E(N) = E0 + omega0 (N - N0) + (w/2)(N - N0)^2, w = d omega/dN < 0,

    H = sum_i p_i^2/(2 E0) + (w/2) sum_i nu_i^2 + U(x, phi),   U = sum_{i<j} u(d_ij, phi_i - phi_j)

(the constant n E0 and omega0 sum N_i, which is conserved, are dropped). Hamilton's equations:
    x' = p/E0,  p' = -dU/dx,  phi' = w nu  (theta' = omega(N_i)),  nu' = -dU/dphi.
Conserved: H, sum nu, total momentum, orbital angular momentum sum x x p.
"""
import numpy as np
from scipy.integrate import solve_ivp


def _pairs(n):
    return np.triu_indices(n, 1)


def pair_terms(X, theta, law):
    i, j = _pairs(len(X))
    r = X[i] - X[j]
    d = np.linalg.norm(r, axis=1)
    rh = r / d[:, None]
    D = theta[i] - theta[j]
    u, ud, udd, uD, uDD, udD = law.pair(d, D)
    return i, j, d, rh, u, ud, udd, uD, uDD, udD


def energy(X, theta, law):
    return float(np.sum(pair_terms(X, theta, law)[4]))


def forces(X, theta, law):
    i, j, d, rh, u, ud, *_ = pair_terms(X, theta, law)
    F = np.zeros_like(X)
    np.add.at(F, i, -ud[:, None] * rh)
    np.add.at(F, j, ud[:, None] * rh)
    return F


def ndot(X, theta, law):
    """N_i' = -dU/dtheta_i (N flows from the phase leader to the lagger)."""
    i, j, d, rh, u, ud, udd, uD, *_ = pair_terms(X, theta, law)
    g = np.zeros(len(X))
    np.add.at(g, i, uD)
    np.add.at(g, j, -uD)
    return -g


def single_energies(X, theta, law):
    i, j, d, rh, u, *_ = pair_terms(X, theta, law)
    Ui = np.zeros(len(X))
    np.add.at(Ui, i, u)
    np.add.at(Ui, j, u)
    return Ui


def hessian(X, theta, law):
    """Hessian of U in (x_1..x_n [3n], theta_1..theta_n [n])."""
    n = len(X)
    i, j, d, rh, u, ud, udd, uD, uDD, udD = pair_terms(X, theta, law)
    H = np.zeros((4 * n, 4 * n))
    I3 = np.eye(3)
    for q in range(len(i)):
        a, b = i[q], j[q]
        rr = np.outer(rh[q], rh[q])
        B = udd[q] * rr + ud[q] / d[q] * (I3 - rr)
        sa, sb = slice(3 * a, 3 * a + 3), slice(3 * b, 3 * b + 3)
        H[sa, sa] += B
        H[sb, sb] += B
        H[sa, sb] -= B
        H[sb, sa] -= B
        ta, tb = 3 * n + a, 3 * n + b
        H[ta, ta] += uDD[q]
        H[tb, tb] += uDD[q]
        H[ta, tb] -= uDD[q]
        H[tb, ta] -= uDD[q]
        v = udD[q] * rh[q]
        for s_x, s_t, sg in ((sa, ta, 1), (sa, tb, -1), (sb, ta, -1), (sb, tb, 1)):
            H[s_x, s_t] += sg * v
            H[s_t, s_x] += sg * v
    return H


def rigid_modes(X):
    """Orthonormal basis of the 6 rigid displacements (3 translations, 3 rotations about the centroid)."""
    n = len(X)
    c = X - X.mean(0)
    V = []
    for k in range(3):
        t = np.zeros((n, 3))
        t[:, k] = 1
        V.append(t.ravel())
        e = np.zeros(3)
        e[k] = 1
        V.append(np.cross(e, c).ravel())
    U, s, _ = np.linalg.svd(np.array(V).T, full_matrices=False)
    return U[:, s > 1e-10 * s.max()]


def jacobian(X, theta, law, E0, w):
    """Linearization of (x, p, phi, nu) about (X, 0, theta, 0)."""
    n = len(X)
    H = hessian(X, theta, law)
    Hxx, Hxt, Htx, Htt = H[:3 * n, :3 * n], H[:3 * n, 3 * n:], H[3 * n:, :3 * n], H[3 * n:, 3 * n:]
    Z3, Zn = np.zeros((3 * n, 3 * n)), np.zeros((n, n))
    Zxn, Znx = np.zeros((3 * n, n)), np.zeros((n, 3 * n))
    return np.block([[Z3, np.eye(3 * n) / E0, Zxn, Zxn],
                     [-Hxx, Z3, -Hxt, Zxn],
                     [Znx, Znx, Zn, w * np.eye(n)],
                     [-Htx, Znx, -Htt, Zn]])


def angular_momentum_overlap(X, theta, A, kappa, omega, dOdd=None):
    """Period-averaged field angular momentum of a superposition of lumps (first order in the overlap):
        J = omega sum_{i<j} sin(theta_i - theta_j) (x_i x x_j) (-O'(d_ij)/d_ij),
    O(d) = int Phi_i Phi_j d^3x. With asymptotic tails O' = -C exp(-kappa d), C = 2 pi A^2, so
    J = omega sum_{i<j} I_{i->j} (x_i x x_j) with the pair N current I_{i->j} = C g sin(theta_i - theta_j).
    dOdd: optional callable for the true-profile O'(d). Derivation in docs/WPD_NOTES.md."""
    i, j = _pairs(len(X))
    d = np.linalg.norm(X[i] - X[j], axis=1)
    w = 2 * np.pi * A ** 2 * np.exp(-kappa * d) / d if dOdd is None else -dOdd(d) / d
    I = w * np.sin(theta[i] - theta[j])
    return omega * np.sum(I[:, None] * np.cross(X[i], X[j]), axis=0)


# ------------------------------------------------------------------ dynamics
def unpack(y, n):
    return y[:3 * n].reshape(n, 3), y[3 * n:6 * n].reshape(n, 3), y[6 * n:7 * n], y[7 * n:]


def rhs_factory(n, law, E0, w, frozen=False):
    def rhs(t, y):
        X, P, ph, nu = unpack(y, n)
        F = forces(X, ph, law)
        out = np.empty_like(y)
        out[:3 * n] = (P / E0).ravel()
        out[3 * n:6 * n] = F.ravel()
        if frozen:
            out[6 * n:] = 0.0
        else:
            out[6 * n:7 * n] = w * nu
            out[7 * n:] = ndot(X, ph, law)
        return out
    return rhs


def hamiltonian(y, n, law, E0, w):
    X, P, ph, nu = unpack(y, n)
    return float(np.sum(P ** 2) / (2 * E0) + 0.5 * w * np.sum(nu ** 2) + energy(X, ph, law))


def integrate(X0, theta0, law, E0, w, T, nu0=None, P0=None, frozen=False, d_stop=None, rtol=1e-11,
              atol=1e-11, t_eval=None):
    n = len(X0)
    y0 = np.concatenate([X0.ravel(), (np.zeros((n, 3)) if P0 is None else P0).ravel(), theta0,
                         np.zeros(n) if nu0 is None else nu0])
    events = None
    if d_stop is not None:
        i, j = _pairs(n)

        def hit(t, y):
            X = y[:3 * n].reshape(n, 3)
            return np.min(np.linalg.norm(X[i] - X[j], axis=1)) - d_stop
        hit.terminal = True
        hit.direction = -1
        events = hit
    return solve_ivp(rhs_factory(n, law, E0, w, frozen), [0, T], y0, method="DOP853", rtol=rtol, atol=atol,
                     events=events, t_eval=t_eval, dense_output=False)
