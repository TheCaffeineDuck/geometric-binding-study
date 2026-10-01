"""Angular momentum of stepped patterns: the overlap formula against a direct 3D grid integral of the
period-averaged field angular momentum -<phi_t d_phi phi> of superposed true profiles; 0/180 patterns carry none."""
import numpy as np

from swirl.arrange import geometry as geo
from swirl.arrange import model as mdl
from swirl.arrange.pairlaw import _load_derive


def grid_Jz(mod, X, th, om, h=0.3, pad=22.0):
    L = np.max(np.abs(X)) + pad
    xs = np.arange(-L, L, h) + h / 2
    zs = np.arange(-pad, pad, h) + h / 2
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    J = 0.0
    for z in zs:
        Psi = np.zeros_like(xx, dtype=complex)
        dPsi = np.zeros_like(Psi)
        for k in range(len(X)):
            dx, dy = xx - X[k, 0], yy - X[k, 1]
            r = np.sqrt(dx * dx + dy * dy + z * z)
            Psi += mod.Phi(r) * np.exp(1j * th[k])
            dPsi += mod.dPhi(r) * (xx * dy - yy * dx) / r * np.exp(1j * th[k])
        # phi = Re[Psi e^{i w t}]: <-phi_t d_phi phi> = -(1/2) Re[i w Psi conj(d_phi Psi)]
        J += np.sum(-0.5 * np.real(1j * om * Psi * np.conj(dPsi))) * h ** 3
    return J


def test_overlap_angular_momentum(params, laws, record):
    mod = _load_derive()
    om = params["omega"]
    out = {}
    worst_true = 0.0
    for K, s, m in ((3, 16.0, 1), (4, 14.0, 1), (5, 12.0, 2)):
        X = s / (2 * np.sin(np.pi / K)) * geo.ring(K)
        th = 2 * np.pi * m * np.arange(K) / K
        Jg = grid_Jz(mod, X, th, om)
        Ja = mdl.angular_momentum_overlap(X, th, params["A"], params["kappa"], om)[2]
        Jt = mdl.angular_momentum_overlap(X, th, params["A"], params["kappa"], om, dOdd=laws["A"].dOdd)[2]
        out["ring%d_m%d_s%g" % (K, m, s)] = {"grid": Jg, "asymptotic_formula": Ja, "true_profile_formula": Jt,
                                            "asym_over_grid": Ja / Jg, "true_over_grid": Jt / Jg}
        worst_true = max(worst_true, abs(Jt / Jg - 1))
        assert np.sign(Ja) == np.sign(Jg)
    # 0/180 pattern: zero
    X = 16 / np.sqrt(2) * geo.ring(4)
    J0 = mdl.angular_momentum_overlap(X, np.array([0, np.pi, 0, np.pi]), params["A"], params["kappa"], om)
    out["worst_true_rel"] = worst_true
    out["binary_J_abs"] = float(np.max(np.abs(J0)))
    record("angular_momentum", out)
    assert worst_true < 1e-3 and np.max(np.abs(J0)) < 1e-12
