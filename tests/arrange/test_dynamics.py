"""Point integrator: energy conservation with phases and N frozen, and the conserved quantities of the full model;
the full linearization against the decoupled sectors."""
import numpy as np

from swirl.arrange import geometry as geo
from swirl.arrange import model as mdl
from swirl.arrange.scan import analyze_balance, find_balances


def conserved(y, n, law, E, w):
    X, P, ph, nu = mdl.unpack(y, n)
    return (mdl.hamiltonian(y, n, law, E, w), nu.sum(), P.sum(0), np.cross(X, P).sum(0))


def test_energy_conservation_frozen_phases(laws, params, record):
    """Phases and N frozen, T = 1000, no pair below 2 hw during the run: the alternating 4-ring at spacing 12
    (flies apart) and the octahedron around an anti-phase centre at spacing 12 with random momenta."""
    E, w = params["E"], params["domega_dN"]
    out = {}
    cases = {"ring4_alternating": (12 / np.sqrt(2) * geo.ring(4), np.array([0, np.pi, 0, np.pi])),
             "octa_c_antiphase": (12 / np.sqrt(2) * geo.with_centre(geo.octahedron()), np.r_[np.zeros(6), np.pi])}
    for mname, law in laws.items():
        for cname, (X, th) in cases.items():
            n = len(X)
            rng = np.random.default_rng(3)
            P0 = 1e-1 * rng.standard_normal((n, 3))
            sol = mdl.integrate(X, th, law, E, w, 1000.0, P0=P0, frozen=True, rtol=1e-12, atol=1e-12,
                                d_stop=2 * params["hw"], t_eval=np.linspace(0, 1000, 201))
            Hs = np.array([mdl.hamiltonian(sol.y[:, k], n, law, E, w) for k in range(sol.y.shape[1])])
            scale = np.sum(np.abs(mdl.pair_terms(X, th, law)[4])) + np.sum(P0 ** 2) / (2 * E)
            drift = np.max(np.abs(Hs - Hs[0])) / scale
            moved = np.max(np.abs(sol.y[:3 * n, -1] - sol.y[:3 * n, 0]))
            out[mname + "_" + cname] = {"max_rel_drift": drift, "t_end": sol.t[-1], "max_displacement": moved}
            assert sol.t[-1] == 1000.0 and drift < 1e-8 and moved > 1.0
    record("energy_conservation_frozen", out)


def test_full_model_conserved_quantities(laws, params, record):
    """5-ring, step 72 degrees, spacing 20, perturbed, positions + phases + N, until T = 1000 or a pair reaches
    2 hw (it merges near t = 750 with N swings of 0.6 N). Conserved: H (point-model Hamiltonian, rotating frame),
    total N, total momentum, orbital angular momentum."""
    out = {}
    for mname, law in laws.items():
        E, w = params["E"], params["domega_dN"]
        n = 5
        X = 20 / (2 * np.sin(np.pi / 5)) * geo.ring(n)
        rng = np.random.default_rng(4)
        X = X + 1e-2 * rng.standard_normal(X.shape)
        th = 2 * np.pi * np.arange(n) / n + 1e-2 * rng.standard_normal(n)
        sol = mdl.integrate(X, th, law, E, w, 1000.0, P0=1e-3 * rng.standard_normal((n, 3)), nu0=5 * rng.standard_normal(n),
                            rtol=1e-12, atol=1e-12, d_stop=2 * params["hw"], t_eval=np.linspace(0, 1000, 1001))
        c0 = conserved(sol.y[:, 0], n, law, E, w)

        def terms(y):
            Xk, Pk, phk, nuk = mdl.unpack(y, n)
            return np.sum(np.abs(mdl.pair_terms(Xk, phk, law)[4])) + np.sum(Pk ** 2) / (2 * E) + abs(w) / 2 * np.sum(nuk ** 2)
        # scale: the largest sum of |energy terms| met along the run (pair energies grow ~100x toward merger)
        scale = max(terms(sol.y[:, k]) for k in range(sol.y.shape[1]))
        scale0 = terms(sol.y[:, 0])
        dH = dN = dP = dL = 0.0
        for k in range(sol.y.shape[1]):
            c = conserved(sol.y[:, k], n, law, E, w)
            dH = max(dH, abs(c[0] - c0[0]) / scale)
            dN = max(dN, abs(c[1] - c0[1]) / params["N"])
            dP = max(dP, np.max(np.abs(c[2] - c0[2])))
            dL = max(dL, np.max(np.abs(c[3] - c0[3])))
        nu = sol.y[7 * n:]
        out[mname] = {"H_rel": dH, "H_rel_to_initial_terms": dH * scale / scale0, "sumN_rel": dN, "P_abs": dP, "L_abs": dL, "t_end": sol.t[-1],
                      "N_swing_max_over_N": float(np.max(np.abs(nu)) / params["N"]),
                      "phase_change_rad": float(np.max(np.abs(sol.y[6 * n:7 * n, -1] - th)))}
        assert sol.t[-1] > 600 and out[mname]["N_swing_max_over_N"] > 0.1
        assert dH < 1e-8 and dN < 1e-12 and dP < 1e-9 and dL < 1e-7
    record("conserved_full_model", out)


def test_linearization_matches_sectors(laws, params, record):
    """For a 0/180 balance the position and phase sectors decouple: the full spectrum must be the union of
    sqrt(-hessian_eig/E) and sqrt(|w| * phase_hessian_eig)."""
    law = laws["A"]
    X1 = geo.with_centre(geo.octahedron())
    th = np.r_[np.zeros(6), np.pi]
    bal, _ = find_balances(X1, th, law, params["kappa"])
    Rb = bal[0]["R"]
    a = analyze_balance(X1, th, Rb, law, params)
    n = 7
    H = mdl.hessian(Rb * X1, th, law)
    pos = np.sqrt(np.clip(-np.linalg.eigvalsh(H[:3 * n, :3 * n]), 0, None) / params["E"])
    ph = np.sqrt(np.clip(abs(params["domega_dN"]) * np.linalg.eigvalsh(H[3 * n:, 3 * n:]), 0, None))
    sect = np.sort(np.r_[pos, ph][np.r_[pos, ph] > 1e-6 * max(pos.max(), ph.max())])[::-1]
    full = np.asarray(a["growth_rates_unstable"])
    err = np.max(np.abs(full - sect) / sect)
    record("linearization_sectors", {"n_full": len(full), "n_sectors": len(sect), "max_rel_diff": err})
    assert len(full) == len(sect) and err < 1e-6
