"""Forces, N flow and Hessian against numerical derivatives of the total energy; the N16 identity; symmetry."""
import numpy as np

from swirl.arrange import geometry as geo
from swirl.arrange import model as mdl


def random_config(rng, n, spacing):
    while True:
        X = rng.uniform(-spacing, spacing, (n, 3))
        if min(np.linalg.norm(X[i] - X[j]) for i in range(n) for j in range(i)) > 0.6 * spacing:
            return X, rng.uniform(0, 2 * np.pi, n)


def num_grad(f, z, h):
    g = np.zeros_like(z)
    for k in range(len(z)):
        e = np.zeros_like(z)
        e[k] = h
        g[k] = (f(z + e) - f(z - e)) / (2 * h) - (f(z + 2 * e) - f(z - 2 * e) - 2 * f(z + e) + 2 * f(z - e)) / (12 * h)
    return g


def test_forces_and_ndot_are_minus_gradient(laws, record):
    rng = np.random.default_rng(1)
    worst, hworst = {}, {}
    for mname, law in laws.items():
        w = hw = 0.0
        for trial in range(6):
            n = [3, 5, 7, 9, 13, 6][trial]
            X, th = random_config(rng, n, 9.0)
            z = np.r_[X.ravel(), th]
            f = lambda z: mdl.energy(z[:3 * n].reshape(n, 3), z[3 * n:], law)
            g = num_grad(f, z, 1e-3)
            an = np.r_[-mdl.forces(X, th, law).ravel(), -mdl.ndot(X, th, law)]
            w = max(w, np.max(np.abs(an - g)) / np.max(np.abs(an)))
            # Hessian against the numerical gradient of the analytic gradient
            gr = lambda z: np.r_[-mdl.forces(z[:3 * n].reshape(n, 3), z[3 * n:], law).ravel(),
                                 -mdl.ndot(z[:3 * n].reshape(n, 3), z[3 * n:], law)]
            H = mdl.hessian(X, th, law)
            Hn = np.array([(gr(z + 1e-5 * e) - gr(z - 1e-5 * e)) / 2e-5 for e in np.eye(len(z))]).T
            hw = max(hw, np.max(np.abs(H - Hn)) / np.max(np.abs(H)))
        worst[mname], hworst[mname] = w, hw
    record("forces_minus_gradient", dict({"worst_rel_" + k: v for k, v in worst.items()},
                                         **{"hessian_worst_rel_" + k: v for k, v in hworst.items()}))
    assert max(worst.values()) < 1e-8 and max(hworst.values()) < 1e-6


def test_N16_identity(laws, params, record):
    """Model A: trace of each lump's own 3x3 block = kappa^2 U_i at every configuration, and U_i < 0 implies a
    negative eigenvalue of that block."""
    law, k = laws["A"], params["kappa"]
    rng = np.random.default_rng(2)
    worst, logic = 0.0, 0
    for trial in range(40):
        n = 3 + trial % 11
        X, th = random_config(rng, n, 6.0 + trial % 5 * 3)
        H = mdl.hessian(X, th, law)
        Ui = mdl.single_energies(X, th, law)
        for a in range(n):
            B = H[3 * a:3 * a + 3, 3 * a:3 * a + 3]
            worst = max(worst, abs(np.trace(B) - k ** 2 * Ui[a]) / np.max(np.abs(k ** 2 * Ui)))
            if Ui[a] < 0 and np.linalg.eigvalsh(B).min() >= 0:
                logic += 1
    record("N16_identity", {"worst_rel_trace_minus_k2Ui": worst, "Ui_negative_but_block_stable": logic})
    assert worst < 1e-10 and logic == 0


def test_regular_symmetric_patterns_have_radial_forces(laws, record):
    """Every scanned pattern whose symmetry (found from coordinates) makes the forces radial is checked
    numerically; the expected cases (stepped rings without centre, polyhedra in phase or around an anti-phase
    centre) must be among them."""
    from swirl.arrange.scan import enumerate_patterns, force_scale
    arr = geo.arrangements()
    worst = worst_net = 0.0
    expected = []
    for K in range(3, 9):
        for m in range(K):
            expected.append(("ring%d" % K, np.mod(2 * np.pi * m * np.arange(K) / K, 2 * np.pi)))
    for name in ("tetra", "octa", "cube", "icosa"):
        n = len(arr[name])
        expected += [(name, np.zeros(n)), (name + "+c", np.r_[np.zeros(n), np.pi]), (name + "+c", np.zeros(n + 1))]
    for name, th in expected:
        X1 = arr[name]
        assert geo.radial_by_symmetry(X1, geo.pattern_symmetry(geo.point_group(X1), th))[0], name
    cases = []
    n_flagged = 0
    for name, X1 in arr.items():
        G = geo.point_group(X1)
        pats, _, _ = enumerate_patterns(X1, G)
        for q in pats:
            n_flagged += 1
            if geo.radial_by_symmetry(X1, geo.pattern_symmetry(G, q["theta"]))[0]:
                cases.append((name, q["theta"]))
    for name, th in cases:
        X1 = arr[name]
        for mname, law in laws.items():
            for s in (8.0, 12.0, 16.0):
                X = s / np.min([np.linalg.norm(X1[i] - X1[j]) for i in range(len(X1)) for j in range(i)]) * X1
                F = mdl.forces(X, th, law)
                fs = force_scale(X, th, law)
                r = np.linalg.norm(X, axis=1)
                for i in range(len(X)):
                    if r[i] < 1e-12:
                        nonrad = np.linalg.norm(F[i])
                    else:
                        u = X[i] / r[i]
                        nonrad = np.linalg.norm(F[i] - (F[i] @ u) * u)
                    worst = max(worst, nonrad / fs)
                    worst_net = max(worst_net, nonrad / np.max(np.linalg.norm(F, axis=1)))
    record("radial_forces", {"worst_nonradial_over_pair_force_scale": worst,
                             "worst_nonradial_over_max_net_force": worst_net,
                             "n_radial_by_symmetry": len(cases), "n_patterns": n_flagged})
    assert worst < 1e-12
