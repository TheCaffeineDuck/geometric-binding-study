"""WP-P P1: checks of the extended point model (unequal and moving lumps), swirl/arrange/unequal.py.

Each test records its measured values in outputs/session3/wpP/checks/<name>.json."""
import json
import os

import numpy as np
import pytest
from scipy.optimize import brentq

from swirl.arrange import model as old
from swirl.arrange import unequal as un
from swirl.arrange.pairlaw import ROOT, PairLaw

CHECKS = os.path.join(ROOT, "outputs", "session3", "wpP", "checks")
WPC_CRIT = os.path.join(ROOT, "outputs", "session2", "wpC", "analysis", "criteria.json")
N0 = 2865.545211117668          # the Session 1/2 profile lump (omega 0.909149)


def record(name, data):
    un.dump(data, os.path.join(CHECKS, name + ".json"))


@pytest.fixture(scope="module")
def fam():
    return un.Family.from_table()


@pytest.fixture(scope="module")
def W0(fam):
    return fam.omega_of_N(N0)


def pair_X(d):
    return np.array([[-d / 2, 0, 0], [d / 2, 0, 0]], float)


# ---------------------------------------------------------------- family and coupling table
def test_family_table(fam, params):
    """Q_i(kappa_i) = A_i; the table against the Session 1 family file; fits at the nodes and between them
    (fresh profiles at the midpoints); dE/dN = omega."""
    from swirl.family.profile import solve_profile
    t = fam.table
    q_self = float(np.max(np.abs(t["Q_self_over_A_minus_1"])))
    with open(os.path.join(ROOT, "outputs", "family", "family_beta1.3333.json")) as f:
        rows1 = json.load(f)["rows_setB_units"]
    worst_tab = 0.0
    n_match = 0
    for r in t["rows"]:
        for s in rows1:
            if abs(s["omega"] - r["omega"]) < 1e-9:
                n_match += 1
                worst_tab = max(worst_tab, *(abs(r[k] / s[k] - 1) for k in ("N", "E", "A")))
    worst_mid = {"N": 0.0, "E": 0.0, "A": 0.0, "hw_abs": 0.0, "q_abs": 0.0}
    mids = [0.85125, 0.87375, 0.90125, 0.92625, 0.94375]
    sols = {}
    for w in mids:
        r = solve_profile(w, m=un.M, g4=un.G4, g6=un.G6, tol=1e-10)
        sols[w] = r
        worst_mid["N"] = max(worst_mid["N"], abs(float(fam.N(w)) / r["N"] - 1))
        worst_mid["E"] = max(worst_mid["E"], abs(float(fam.E(w)) / r["E"] - 1))
        worst_mid["A"] = max(worst_mid["A"], abs(float(fam.A(w)) / r["A"] - 1))
        worst_mid["hw_abs"] = max(worst_mid["hw_abs"], abs(float(fam.hw(w)) - r["half_width"]))
    for wi in mids[::2]:
        ri = sols[wi]
        x = np.linspace(0, ri["R_box"], 200001)
        P = ri["_sol"].sol(x)[0]
        S = un.G4 / 8 * P ** 3 - un.G6 / 192 * P ** 5
        for wj in mids:
            kj = np.sqrt(1 - wj ** 2)
            qd = np.trapezoid(S * x * np.sinh(kj * x) / kj, x) / ri["A"]
            worst_mid["q_abs"] = max(worst_mid["q_abs"], abs(float(fam.q(np.array(wi), np.array(wj))) - qd))
    # dE/dN = omega (complex step through the fits)
    ws = np.linspace(fam.W_lo + 1e-3, fam.W_hi - 1e-3, 50)
    lump = fam.lump(ws + 1j * 1e-30)
    dEdN = lump[2].imag / lump[3].imag
    legendre = float(np.max(np.abs(dEdN / ws - 1)))
    W0 = fam.omega_of_N(N0)
    ref = {"omega": abs(W0 / params["omega"] - 1), "A": abs(float(fam.A(W0)) / params["A"] - 1),
           "E": abs(float(fam.E(W0)) / params["E"] - 1)}
    # domega/dN: the fit's derivative against a central difference of fresh profiles (the Session 2 value came
    # from a spline of the 0.005-spaced family table and is reported, not used as the reference)
    hh = 2e-5
    Np = solve_profile(W0 + hh, m=un.M, g4=un.G4, g6=un.G6, tol=1e-11)["N"]
    Nm = solve_profile(W0 - hh, m=un.M, g4=un.G4, g6=un.G6, tol=1e-11)["N"]
    w_fd = 2 * hh / (Np - Nm)
    ref["domega_dN_vs_fresh_fd"] = abs(fam.domega_dN(W0) / w_fd - 1)
    ref_info = {"domega_dN_fit": fam.domega_dN(W0), "domega_dN_fd": w_fd, "domega_dN_session2": params["domega_dN"],
                "session2_vs_fd_rel": params["domega_dN"] / w_fd - 1}
    out = {"Q_self_over_A_max": q_self, "table_vs_family_file_max_rel": worst_tab, "rows_matched": n_match,
           "fit_residuals_nodes": fam.fit_residuals, "fit_vs_fresh_profiles_at_midpoints": worst_mid,
           "dEdN_over_omega_minus_1_max": legendre, "N0_lump_vs_session2_params_rel": ref, "domega_dN": ref_info,
           "omega_range": [fam.W_lo, fam.W_hi], "N_range": [float(fam.N(fam.W_hi)), float(fam.N(fam.W_lo))]}
    record("family_table", out)
    assert q_self < 1e-6 and worst_tab < 1e-6 and n_match >= 10
    assert worst_mid["N"] < 1e-7 and worst_mid["E"] < 1e-7 and worst_mid["A"] < 1e-6 and worst_mid["q_abs"] < 1e-6
    assert legendre < 1e-12 and max(ref.values()) < 1e-6


# ---------------------------------------------------------------- reductions to the Session 2 law
def test_frozen_family_reproduces_session2_model(params):
    """Family.frozen + nonrelativistic + no boost phase is the Session 2 model (model.py, Model A): same
    trajectory for three moving, exchanging lumps."""
    fz = un.Family.frozen(params)
    law = PairLaw(params["A"], params["kappa"], "A")
    X = np.array([[-8, 0, 0], [8, 0.5, 0], [0, 12, 1.0]])
    th = np.array([0.0, 1.0, 2.5])
    nu0 = np.array([10.0, -30.0, 5.0])
    P0 = 3 * np.array([[0.5, 1, 0], [0, -1, 0.2], [-0.5, 0, 0]])
    T = [100.0, 300.0]
    a = old.integrate(X, th, law, params["E"], params["domega_dN"], 300.0, nu0=nu0, P0=P0, rtol=1e-12,
                      atol=1e-12, t_eval=T)
    m = un.Model(fz, params["omega"], relativistic=False, boost=False)
    b = m.integrate(m.pack(X, P0, th, params["omega"] + params["domega_dN"] * nu0), 3, 300.0, t_eval=T,
                    contact=False, rtol=1e-12, atol=1e-12)
    yb = b.y.copy()
    yb[21:] = (yb[21:] - params["omega"]) / params["domega_dN"]
    diff = np.max(np.abs(a.y - yb), axis=0) / np.max(np.abs(a.y), axis=0)
    H_old = old.hamiltonian(a.y[:, 0], 3, law, params["E"], params["domega_dN"])
    H_new = m.H(b.y[:, 0], 3) - 3 * (params["E"] - params["omega"] * params["N"])
    record("frozen_reduction", {"max_rel_state_diff": diff.tolist(), "t": T, "H_old": H_old, "H_new_minus_const": H_new,
                                "H_rel_diff": abs(H_new / H_old - 1)})
    assert np.max(diff) < 1e-9 and abs(H_new / H_old - 1) < 1e-10


def test_equal_lumps_full_family_is_the_wpc_law(fam, W0, wpc):
    """Full model, two lumps at W0 at rest: U, the force and the N flow at 90 degrees equal the WP-C table
    (pair_law_predictions.json) to the accuracy of the fits; weights 1/2 and q = 1 exactly in the equal limit."""
    m = un.Model(fam, W0)
    m_nb = un.Model(fam, W0, boost=False)
    W = np.full(2, W0)
    worst = 0.0
    rows = {}
    for row in wpc["table"]:
        d = row["d"]
        z = m.pack(pair_X(d), np.zeros((2, 3)), [0.0, 0.0], W)
        U = m.H(z, 2) - 2 * (float(fam.E(W0)) - W0 * N0)
        F = -m.grad(z, 2)[0]          # force on lump 1 (at -d/2) along +x: toward lump 2
        dNdW = float(fam.lump(np.array([W0]))[4][0])
        # the law itself (no boost phase: lumps at rest with P = 0)
        z90 = m_nb.pack(pair_X(d), np.zeros((2, 3)), [0.0, np.pi / 2], W)
        dN2 = m_nb.rhs(0, z90, 2)[15] * dNdW                       # dW_2/dt -> dN_2/dt
        F90_nb = -m_nb.grad(z90, 2)[0]
        # full model: lumps at rest carry the overlap momentum, which shifts the effective phase by delta
        z90b = m.pack(pair_X(d), m.momenta_for_velocities(pair_X(d), np.zeros((2, 3)), [0.0, np.pi / 2], W),
                      [0.0, np.pi / 2], W)
        dN2b = m.rhs(0, z90b, 2)[15] * dNdW
        F90b = -m.rhs(0, z90b, 2)[3]
        rel = [abs(U / row["U_asym_dphi0"] - 1), abs(F / row["F_asym_dphi0"] - 1),
               abs(dN2 / row["dN2dt_asym_dphi90"] - 1)]
        rows[str(d)] = {"U": U, "F": F, "dN2dt_90": dN2, "rel": rel, "F_90_no_boost": F90_nb,
                        "boost_on_dN2dt_90_rel_change": dN2b / dN2 - 1, "boost_on_force_90_over_F0": F90b / F}
        worst = max(worst, max(rel))
    q00 = float(fam.q(np.array(W0), np.array(W0)))
    record("equal_limit_vs_wpc", {"worst_rel": worst, "q_W0_W0_minus_1": q00 - 1, "rows": rows})
    assert worst < 1e-6


# ---------------------------------------------------------------- the unequal law against direct quadrature
def test_unequal_law_against_quadrature(fam):
    """The cross term of the averaged Lagrangian for two unequal resting lumps, by axisymmetric quadrature of
    the true profiles, int [W1 W2 F1 F2 - grad F1.grad F2 - m^2 F1 F2 + S1 F2 + S2 F1], against the closed form
    4 pi [W1 Q1(k2) A2 e^{-k2 d} + W2 Q2(k1) A1 e^{-k1 d}] / ((W1 + W2) d) that the model uses (-2 U / cos D)."""
    from swirl.family.profile import profile_curve, solve_profile
    out = {}
    worst = 0.0
    for N_ratio in (1.0, 1.5, 2.0):
        w1, w2 = 0.909149280617113, fam.omega_of_N(N0 * N_ratio)
        p1 = solve_profile(w1, m=un.M, g4=un.G4, g6=un.G6, tol=1e-10)
        p2 = solve_profile(w2, m=un.M, g4=un.G4, g6=un.G6, tol=1e-10)

        def prof(p, r):
            F = profile_curve(p, r)
            h = 1e-4
            dF = (profile_curve(p, r + h) - profile_curve(p, np.maximum(r - h, 0))) / (r + h - np.maximum(r - h, 0))
            return F, dF
        for d in (16.0, 20.0, 24.0):
            h = 0.05
            rho = np.arange(0, 36, h) + h / 2
            z = np.arange(-36, d + 36, h) + h / 2
            R, Z = np.meshgrid(rho, z, indexing="ij")
            r1 = np.sqrt(R ** 2 + Z ** 2)
            r2 = np.sqrt(R ** 2 + (Z - d) ** 2)
            F1, d1 = prof(p1, r1)
            F2, d2 = prof(p2, r2)
            cosang = (R ** 2 + Z * (Z - d)) / (r1 * r2)
            S1 = un.G4 / 8 * F1 ** 3 - un.G6 / 192 * F1 ** 5
            S2 = un.G4 / 8 * F2 ** 3 - un.G6 / 192 * F2 ** 5
            integrand = w1 * w2 * F1 * F2 - d1 * d2 * cosang - F1 * F2 + S1 * F2 + S2 * F1
            I = float(np.sum(2 * np.pi * R * integrand) * h * h)
            k1, k2 = np.sqrt(1 - w1 ** 2), np.sqrt(1 - w2 ** 2)
            q12 = float(fam.q(np.array(w1), np.array(w2))) * p1["A"]
            q21 = float(fam.q(np.array(w2), np.array(w1))) * p2["A"]
            closed = 4 * np.pi * (w1 * q12 * p2["A"] * np.exp(-k2 * d) + w2 * q21 * p1["A"] * np.exp(-k1 * d)) / ((w1 + w2) * d)
            # the model's own U at rest, in phase: U = -closed/2
            U = float(un.pair_energy(pair_X(d)[None], np.zeros((1, 2, 3)), np.zeros((1, 2)), np.array([[w1, w2]]), fam)[0])
            rel = abs(I / closed - 1)
            out["N2/N1=%.1f d=%g" % (N_ratio, d)] = {"quadrature": I, "closed_form": closed, "rel": rel,
                                                     "model_U": U, "model_U_over_minus_half_closed": U / (-closed / 2),
                                                     "single_exponential_equal_weights_rel":
                                                         abs((2 * np.pi * p1["A"] * p2["A"] * np.exp(-0.5 * (k1 + k2) * d) / d) / (closed / 2) - 1)}
            if d >= 20:
                worst = max(worst, rel)
    record("unequal_law_quadrature", {"rows": out, "worst_rel_d_ge_20": worst,
                                      "note": "first order in the overlap; the residual at d = 16 includes O(overlap^2)"})
    assert worst < 5e-3
    assert all(abs(v["model_U_over_minus_half_closed"] - 1) < 1e-6 for v in out.values())


# ---------------------------------------------------------------- conservation, reversal
def ring_unequal_moving(fam, n=4, d=16.0):
    X, R = un.ring_positions(n, d)
    Ws = np.array([fam.omega_of_N(N0 * s) for s in (1.0, 1.6, 1.15, 2.0)[:n]])
    th = np.array([0.0, 1.3, 2.1, 4.0])[:n]
    V = 0.04 * un.tangents(X) + 0.01 * np.array([[1, 0, 0.2], [0, -1, 0], [0.3, 0.2, 0], [-0.5, 0.1, -0.1]])[:n]
    m = un.Model(fam, float(np.mean(Ws)))
    P = m.momenta_for_velocities(X, V, th, Ws)
    return m, m.pack(X, P, th, Ws)


def test_conservation_full_model(fam):
    m, z0 = ring_unequal_moving(fam)
    n = 4
    sol = m.integrate(z0, n, 1500.0, t_eval=np.linspace(0, 1500, 301), rtol=1e-12, atol=1e-12)
    c0 = m.conserved(z0, n)
    X0, P0, ph0, W0s = un.split(z0, n)
    U0 = un.pair_energy(X0.reshape(n, 3), P0.reshape(n, 3), ph0, W0s, fam, parts=True)[0]
    kin0 = float(np.sum(np.sqrt(np.sum(P0.reshape(n, 3) ** 2, 1) + fam.E(W0s) ** 2) - fam.E(W0s)))
    scale = float(np.sum(np.abs(U0))) + kin0
    dH = dN = dP = dL = 0.0
    Ns = []
    for k in range(sol.y.shape[1]):
        c = m.conserved(sol.y[:, k], n)
        dH = max(dH, abs(c["H"] - c0["H"]) / scale)
        dN = max(dN, abs(c["N"] - c0["N"]) / c0["N"])
        dP = max(dP, float(np.max(np.abs(c["P"] - c0["P"]))))
        dL = max(dL, float(np.max(np.abs(c["L"] - c0["L"]))))
        Ns.append(fam.N(sol.y[7 * n:, k]))
    Ns = np.array(Ns)
    swing = float(np.max(np.abs(Ns - Ns[0]) / Ns[0]))
    out = {"t_end": float(sol.t[-1]), "status": int(sol.status), "H_rel_to_initial_terms": dH, "sumN_rel": dN,
           "P_abs": dP, "L_abs": dL, "P_scale": float(np.max(np.abs(P0))), "L_scale": float(np.max(np.abs(c0["L"]))),
           "energy_scale": scale, "max_N_change_rel": swing, "rtol": 1e-12}
    record("conservation_full", out)
    assert sol.t[-1] > 300 and swing > 1e-3
    assert dH < 1e-8 and dN < 1e-12 and dP < 1e-9 and dL < 1e-7


def test_time_reversal(fam):
    """Forward to T, flip P and phi, forward to T again: back to the start (flipped)."""
    m, z0 = ring_unequal_moving(fam)
    n = 4
    T = 300.0
    a = m.integrate(z0, n, T, contact=False, rtol=1e-12, atol=1e-12)
    z1 = a.y[:, -1].copy()
    z1[3 * n:7 * n] *= -1
    b = m.integrate(z1, n, T, contact=False, rtol=1e-12, atol=1e-12)
    z2 = b.y[:, -1].copy()
    z2[3 * n:7 * n] *= -1
    X0 = z0[:3 * n]
    err = {"X_abs": float(np.max(np.abs(z2[:3 * n] - X0))), "P_rel": float(np.max(np.abs(z2[3 * n:6 * n] - z0[3 * n:6 * n])) / np.max(np.abs(z0[3 * n:6 * n]))),
           "phi_abs": float(np.max(np.abs(z2[6 * n:7 * n] - z0[6 * n:7 * n]))), "W_abs": float(np.max(np.abs(z2[7 * n:] - z0[7 * n:]))),
           "displacement_over_run": float(np.max(np.abs(z1[:3 * n] - X0))), "T": T}
    record("time_reversal", err)
    assert err["X_abs"] < 1e-6 and err["P_rel"] < 1e-6 and err["phi_abs"] < 1e-8 and err["W_abs"] < 1e-11


# ---------------------------------------------------------------- moving lumps
def test_single_moving_lump(fam, W0):
    """A lone lump with lab speed v: X' = v, H = gamma E0 - omega_ref N0, phase rate omega/gamma. These are the
    numbers to compare with the boosted 3D lump of WP-L (L2)."""
    out = {}
    worst = 0.0
    for v in (0.02, 0.05, 0.1, 0.3):
        m = un.Model(fam, 0.0)
        P = m.momenta_for_velocities(np.zeros((1, 3)), [[v, 0, 0]], [0.0], [W0])
        z = m.pack(np.zeros((1, 3)), P, [0.0], [W0])
        sol = m.integrate(z, 1, 100.0, t_eval=[100.0], contact=False, rtol=1e-13, atol=1e-13)
        g = 1 / np.sqrt(1 - v * v)
        E0 = float(fam.E(W0))
        meas = {"speed": sol.y[0, -1] / 100.0, "H": m.H(z, 1), "phase_rate": sol.y[6, -1] / 100.0}
        pred = {"speed": v, "H": g * E0, "phase_rate": W0 / g}
        rel = {k: abs(meas[k] / pred[k] - 1) for k in meas}
        worst = max(worst, max(rel.values()))
        out["v=%g" % v] = {"measured": meas, "predicted": pred, "rel": rel, "gamma_minus_1": g - 1,
                           "phase_rate_shift_omega_minus_omega_over_gamma": W0 - W0 / g}
    record("single_moving_lump", {"rows": out, "worst_rel": worst, "omega": W0, "E0": float(fam.E(W0))})
    assert worst < 1e-12


def test_overlap_momentum_at_rest(fam, W0):
    """Two resting lumps at phase difference dtheta carry canonical momentum P = -omega C e^{-kappa d} sin(dtheta)
    x-hat (WP-C, FND-11: P_x = -omega I(d) sin(dtheta), asymptotic I = 2 pi A^2 e^{-kappa d}); the boost-phase term
    is what puts it in the model."""
    m = un.Model(fam, W0)
    A0, k0 = float(fam.A(W0)), np.sqrt(1 - W0 ** 2)
    out, worst = {}, 0.0
    for d in (16.0, 20.0):
        for deg in (45.0, 90.0, 135.0):
            th = [-np.deg2rad(deg) / 2, np.deg2rad(deg) / 2]        # dtheta = theta_2 - theta_1
            P = m.momenta_for_velocities(pair_X(d), np.zeros((2, 3)), th, [W0, W0])
            pred = -W0 * 2 * np.pi * A0 ** 2 * np.exp(-k0 * d) * np.sin(np.deg2rad(deg))
            # first order in the overlap: P_i = -gamma E dU/dP_i evaluated at P = 0
            z0 = m.pack(pair_X(d), np.zeros((2, 3)), th, [W0, W0])
            P1 = -float(fam.E(W0)) * m.grad(z0, 2)[6:12].reshape(2, 3)
            rel = abs(P1.sum(0)[0] / pred - 1)
            worst = max(worst, rel)
            out["d=%g dtheta=%g" % (d, deg)] = {"P_x_first_order": float(P1.sum(0)[0]), "predicted": pred, "rel": rel,
                                                "P_x_self_consistent": float(P.sum(0)[0]),
                                                "self_consistent_over_first_order_minus_1": float(P.sum(0)[0] / P1.sum(0)[0] - 1),
                                                "P_y_P_z": P.sum(0)[1:].tolist()}
    record("overlap_momentum", {"rows": out, "worst_rel_first_order": worst,
                                "note": "the self-consistent value differs at second order in the overlap (the boost "
                                        "phase of the overlap momentum feeds back into U)"})
    assert worst < 1e-7


def circular_orbit_analytic(F, E1, E2, d):
    """Relativistic two-body circular orbit under a central attraction F (phases frozen): equal and opposite
    momenta p, common angular velocity w_o, r_i = v_i / w_o, r_1 + r_2 = d, p w_o = F."""
    def g(wo):
        p = F / wo
        return (p / np.hypot(p, E1) + p / np.hypot(p, E2)) / wo - d
    wo = brentq(g, 1e-8, 1.0, xtol=1e-16, rtol=1e-15)
    p = F / wo
    return wo, p, p / np.hypot(p, E1), p / np.hypot(p, E2)


def test_circular_orbit_frozen_phase(fam, W0):
    """Phases and N frozen, in phase: an equal and an unequal (N ratio 2) pair on the analytic circular orbit.
    Measured: period (time for the separation to turn by 2 pi) and the largest change of d over one period.
    The orbit is unstable (screened potential, kappa d > 1.618), so d drifts from round-off at a rate
    Omega_orb sqrt(-d F'/F - 3); the drift is reported, not hidden."""
    out, worst_T = {}, 0.0
    for ratio in (1.0, 2.0):
        for d in (16.0, 20.0):
            W1, W2 = W0, fam.omega_of_N(N0 * ratio)
            E1, E2 = float(fam.E(W1)), float(fam.E(W2))
            m = un.Model(fam, W1, frozen=True)
            hfun = lambda dd: un.pair_energy(pair_X(dd)[None], np.zeros((1, 2, 3)), np.zeros((1, 2)),
                                             np.array([[W1, W2]]), fam)[0].real
            hc = lambda dd: un.pair_energy((pair_X(1.0) * dd)[None], np.zeros((1, 2, 3)), np.zeros((1, 2)),
                                           np.array([[W1, W2]]), fam)[0]
            F = float(hc(d + 1e-30j).imag / 1e-30)                  # attraction: +dU/dd (complex step)
            wo, p, v1, v2 = circular_orbit_analytic(F, E1, E2, d)
            r1, r2 = v1 / wo, v2 / wo
            X = np.array([[-r1, 0, 0], [r2, 0, 0]])
            V = np.array([[0, -v1, 0], [0, v2, 0]])
            P = m.momenta_for_velocities(X, V, [0.0, 0.0], [W1, W2])
            z = m.pack(X, P, [0.0, 0.0], [W1, W2])
            T_an = 2 * np.pi / wo

            def turned(t, y, *_):
                r = y[3:6] - y[0:3]
                return np.arctan2(r[1], r[0]) if t > 0.5 * T_an else 1.0
            turned.direction = 1
            sol = solve_turn(m, z, T_an, turned)
            t_turn = float(sol.t_events[0][0])
            dd = np.linalg.norm(sol.y[3:6] - sol.y[0:3], axis=0)
            rel_T = abs(t_turn / T_an - 1)
            worst_T = max(worst_T, rel_T)
            # analytic instability rate of the circular orbit (nonrelativistic): Omega sqrt(-d F'/F - 3)
            Fp = (hfun(d + 1e-3) - 2 * hfun(d) + hfun(d - 1e-3)) / 1e-6
            lam = wo * np.sqrt(-d * Fp / F - 3) if -d * Fp / F > 3 else 0.0
            out["N2/N1=%g d=%g" % (ratio, d)] = {"v1": v1, "v2": v2, "T_orb_analytic": T_an, "T_orb_measured": t_turn,
                                                 "rel_T": rel_T, "max_rel_d_change_one_period": float(np.max(np.abs(dd / d - 1))),
                                                 "instability_rate": lam, "efolds_per_period": lam * T_an,
                                                 "nonrel_v_rel_check": abs((v1 + v2) / np.sqrt(F * d * (E1 + E2) / (E1 * E2)) - 1)}
    record("circular_orbit", {"rows": out, "worst_rel_T": worst_T})
    assert worst_T < 1e-6


def solve_turn(m, z, T_an, ev):
    from scipy.integrate import solve_ivp
    return solve_ivp(m.rhs, [0, 1.2 * T_an], z, method="DOP853", rtol=1e-12, atol=1e-12, events=ev,
                     args=(2,), dense_output=False, max_step=T_an / 200)


def test_linearization_against_analytic_pair(params):
    """Frozen law, nonrelativistic, no boost phase: the rotating-frame linearization of the in-phase circular
    orbit has exactly the static phase pendulum rate sqrt(2 C g |w|) and the orbit's radial instability
    Omega sqrt(-d F'/F - 3)."""
    fz = un.Family.frozen(params)
    out, worst = {}, 0.0
    W0 = params["omega"]
    C0, k0, E0, w = 2 * np.pi * params["A"] ** 2, params["kappa"], params["E"], params["domega_dN"]
    for d in (12.0, 16.0, 20.0):
        m = un.Model(fz, W0, relativistic=False, boost=False)
        r = un.rotating_ring(m, 2, d, np.zeros(2), W0)
        J, res = un.rotating_frame_jacobian(m, r["z"], 2, r["Omega_r"])
        modes = un.classify_modes(J, 2, k0)
        F = C0 * np.exp(-k0 * d) * (k0 / d + 1 / d ** 2)
        Om = 2 * np.sqrt(F * d / (2 * E0)) / d
        lam_r = Om * np.sqrt(k0 * d + (k0 + 2 / d) / (k0 + 1 / d) - 3)
        lam_p = np.sqrt(2 * C0 * np.exp(-k0 * d) / d * abs(w))
        got_r = max(x["re"] for x in modes if x["position_share"] > 0.5)
        got_p = max(x["re"] for x in modes if x["position_share"] < 0.5)
        rel = [abs(got_r / lam_r - 1), abs(got_p / lam_p - 1), abs(r["Omega_r"] / Om - 1)]
        worst = max(worst, max(rel))
        out["d=%g" % d] = {"radial": [got_r, lam_r], "phase": [got_p, lam_p], "Omega": [r["Omega_r"], Om],
                           "rel": rel, "fixed_point_residual": float(np.max(np.abs(res)))}
    record("linearization_analytic_pair", {"rows": out, "worst_rel": worst})
    assert worst < 1e-6


# ---------------------------------------------------------------- regression against the Session 2 3D pairs
def wpc_measured():
    with open(WPC_CRIT) as f:
        rows = {r["id"]: r for r in json.load(f)["rows"]}
    meas = {}
    for it in rows["C8"]["items"]:
        meas[(int(it["what"][1:]), 180)] = {"shift": it["shift_end"]}
    for it in rows["C9"]["items"]:
        meas[(int(it["what"][1:]), 135)] = {"shift": it["shift_end"], "dtheta_end": it["dtheta_end_deg"]}
    for it in rows["C5d"]["items"]:
        meas[(int(it["what"][1:]), 90)] = {"shift": it["shift_end"]}
    for it in rows["C5c"]["items"]:
        meas.setdefault((int(it["what"][1:]), 90), {})["dtheta_end"] = it["dtheta_end_deg"]
    for it in rows["C7"]["items"]:
        if it["what"] == "d16_contact":
            meas[(16, 0)] = {"contact": it["contact_t"]}
        elif it["what"] == "d20_contact":
            meas[(20, 0)] = {"contact": it["contact_t"]}
        elif it["what"] == "d24_shift":
            meas[(24, 0)] = {"shift": it["shift_end"]}
    for it in rows["C6"]["items"]:
        if it["what"] == "d24":
            meas[(24, 45)] = {"shift": it["shift_end"]}
    return meas



def wpc_3d_series(d, deg):
    """Tracked whole-period lump centroids of the 3D pair run: t, d(t), and the distinct-count windows."""
    path = os.path.join(ROOT, "outputs", "session2", "wpC", "results", "pair_d%d_p%03d.json" % (d, deg))
    with open(path) as f:
        res = json.load(f)
    lu = res["lumps"]["averages"]["lumps"]
    tr = []
    for k in range(2):
        t = np.array([x for x, X in zip(lu[k]["t_center"], lu[k]["X"]) if x is not None and X is not None])
        X = np.array([X for x, X in zip(lu[k]["t_center"], lu[k]["X"]) if x is not None and X is not None])
        _, idx = np.unique(t, return_index=True)
        tr.append((t[idx], X[idx]))
    tt = tr[0][0]
    X1 = np.stack([np.interp(tt, tr[1][0], tr[1][1][:, a]) for a in range(3)], 1)
    dd = np.linalg.norm(X1 - tr[0][1], axis=1)
    bad = [c["t_mid"] for c in res["lumps"]["counts"] if c["count_distinct"] < 2]
    return tt, dd, (min(bad) if bad else np.inf), float(res["t"])


def pair_run(kind, fam_or_params, d, deg, T, t_eval):
    """Static pair, theta = -/+ dtheta/2, to T. kind: 'old' (Session 2 model, Model A), 'new' (full extended
    model), 'new_noboost' (extended model without the boost phase). Returns d at t_eval (nan after contact), the
    contact time, the shift per lump and dtheta at the end."""
    if kind == "old":
        p = fam_or_params
        law = PairLaw(p["A"], p["kappa"], "A")
        sol = old.integrate(pair_X(d), np.deg2rad([-deg / 2, deg / 2]), law, p["E"], p["domega_dN"], T,
                            d_stop=2 * p["hw"], rtol=1e-12, atol=1e-12, t_eval=t_eval)
        dist = sol.y[3] - sol.y[0]
        res = {"contact": float(sol.t_events[0][0]) if sol.status == 1 else None}
        y = sol.y[:, -1]
        res["dtheta_end"] = float(np.rad2deg(y[13] - y[12]))
    else:
        fam = fam_or_params
        W0 = fam.omega_of_N(N0)
        m = un.Model(fam, W0, boost={"new": True, "new_resummed": "resummed", "new_noboost": False}[kind])
        th = np.deg2rad([-deg / 2, deg / 2])
        P = m.momenta_for_velocities(pair_X(d), np.zeros((2, 3)), th, [W0, W0])
        sol = m.integrate(m.pack(pair_X(d), P, th, [W0, W0]), 2, T, t_eval=t_eval, rtol=1e-11, atol=1e-11)
        dist = np.linalg.norm(sol.y[3:6] - sol.y[0:3], axis=0)
        res = {"contact": float(sol.t_events[0][0]) if sol.status == 1 else None}
        y = sol.y[:, -1]
        res["dtheta_end"] = float(np.rad2deg(y[13] - y[12]))
        res["N_asym_end"] = float(abs(fam.N(y[14]) - fam.N(y[15])) / (fam.N(y[14]) + fam.N(y[15])))
    dd = np.full(len(t_eval), np.nan)
    dd[:dist.size] = dist
    res["d"] = dd
    res["shift_end"] = float((d - dist[-1]) / 2) if res["contact"] is None else None
    return res


def test_regression_wpc_pairs(fam, params):
    """The extended model on the 15 Session 2 3D pairs (equal lumps, static start, theta = -/+ dtheta/2): the
    distance d(t) every 25 time units against the tracked 3D centroids (while the 3D count is 2 and before
    contact), next to the Session 2 point model and the extended model without the boost phase (a diagnostic).
    Target (regression): the extended model's worst |d - d_3D| no larger than 0.5 length units or 10% of the 3D
    displacement |d_3D - d0|, whichever is larger; contact times within 10%; the same outcome as 3D in all 15."""
    meas = wpc_measured()
    out = {}
    worst = {"old": 0.0, "new": 0.0, "new_resummed": 0.0, "new_noboost": 0.0}
    worst_contact = {"old": 0.0, "new": 0.0, "new_resummed": 0.0, "new_noboost": 0.0}
    passes = []
    for d in (16, 20, 24):
        for deg in (0, 45, 90, 135, 180):
            t3, d3, t_bad, T = wpc_3d_series(d, deg)
            t_eval = np.arange(25.0, T + 1e-9, 25.0)
            row = {"T": T, "measured_3D": meas.get((d, deg))}
            d3i = np.interp(t_eval, t3, d3)
            ok3 = (t_eval < t_bad - 20) & (t_eval <= t3[-1]) & (d3i > 2 * params["hw"])
            for kind, src in (("old", params), ("new", fam), ("new_resummed", fam), ("new_noboost", fam)):
                r = pair_run(kind, src, d, deg, T, t_eval)
                ok = ok3 & ~np.isnan(r["d"])
                err = np.abs(r["d"] - d3i)[ok]
                disp = np.abs(d3i - d)[ok]
                e_max = float(np.max(err)) if err.size else None
                lim = float(max(0.5, 0.1 * np.max(disp))) if err.size else None
                row[kind] = {"max_abs_d_err": e_max, "limit": lim, "contact": r["contact"], "shift_end": r["shift_end"],
                             "dtheta_end": r["dtheta_end"], "N_asym_end": r.get("N_asym_end")}
                if e_max is not None:
                    worst[kind] = max(worst[kind], e_max / lim)
                c3 = (meas.get((d, deg)) or {}).get("contact")
                if c3 and r["contact"]:
                    row[kind]["contact_rel_err"] = r["contact"] / c3 - 1
                    worst_contact[kind] = max(worst_contact[kind], abs(r["contact"] / c3 - 1))
            row["3D_d_at_samples"] = {"t": t_eval[ok3].tolist(), "d": d3i[ok3].tolist()}
            row["class_3D"] = "merged" if np.isfinite(t_bad) else ("separated" if d3[-1] > d else "approached")
            row["class_new"] = "merged" if row["new"]["contact"] else (
                "separated" if row["new"]["shift_end"] < 0 else "approached")
            passes.append(row["class_3D"] == row["class_new"])
            out["d%d_%d" % (d, deg)] = row
    summary = {"worst_d_err_over_limit": worst, "worst_contact_rel_err": worst_contact,
               "classes_agree_with_3D": int(sum(passes)), "n_runs": len(passes)}
    record("regression_wpc_pairs", {"rows": out, "summary": summary})
    assert all(passes)
    # target 1.0 (d error) and 0.10 (contact); within 3 times the target is a pass with a note (CLAUDE.md)
    assert worst["new"] <= 3.0 and worst_contact["new"] <= 0.30


# ---------------------------------------------------------------- the boost phase against the superposition
def boosted_cross_lagrangian(p, d, v1, v2, Delta, h=0.05, half=34.0):
    """Averaged-Lagrangian cross term of two superposed, exactly boosted profiles moving along the line of
    centres (lump 1 at z = 0 with velocity v1, lump 2 at z = d with v2; Delta = theta_1 - theta_2 at lab time 0),
    first order in the overlap (axisymmetric quadrature)."""
    from swirl.family.profile import profile_curve
    w = p["omega"]
    rho = np.arange(0, half, h) + h / 2
    z = np.arange(-half, d + half, h) + h / 2
    R, Z = np.meshgrid(rho, z, indexing="ij")
    fl = []
    for Zc, v, th in ((0.0, v1, 0.0), (d, v2, -Delta)):
        g = 1 / np.sqrt(1 - v * v)
        s = np.sqrt(R ** 2 + (g * (Z - Zc)) ** 2)
        F = profile_curve(p, s)
        dF = (profile_curve(p, s + 1e-4) - profile_curve(p, np.maximum(s - 1e-4, 0))) / (s + 1e-4 - np.maximum(s - 1e-4, 0))
        dFz = dF * g * g * (Z - Zc) / np.maximum(s, 1e-12)
        dFr = dF * R / np.maximum(s, 1e-12)
        e = np.exp(1j * (th - w * g * v * (Z - Zc)))
        S = un.G4 / 8 * F ** 3 - un.G6 / 192 * F ** 5
        fl.append(((-v * dFz + 1j * w * g * F) * e, (dFz - 1j * w * g * v * F) * e, dFr * e, F * e, S, F, e))
    a, b = fl
    lt = 2 * np.real(a[0] * np.conj(b[0]))
    lg = 2 * np.real(a[1] * np.conj(b[1]) + a[2] * np.conj(b[2]))
    lm = 2 * np.real(a[3] * np.conj(b[3]))
    nl = 0.5 * np.real((a[4] * b[5] + b[4] * a[5]) * a[6] * np.conj(b[6]))
    return float(np.sum(2 * np.pi * R * (0.25 * (lt - lg - lm) + nl)) * h * h)


def test_boost_phase_against_superposition(fam, W0):
    """Superposed boosted profiles, by quadrature. (1) A uniformly moving pair is the boosted static pair:
    L = L0(gamma d)/gamma (Lorentz check of the quadrature), and the model's U does not change (co-moving shift
    exact). (2) Head-on motion (+v, -v): the superposition's cross term against the model's two forms (default:
    counter-moving shift to first order only, no e^2 term; 'resummed': cos(e) at each core)."""
    from swirl.family.profile import solve_profile
    p = solve_profile(W0, m=un.M, g4=un.G4, g6=un.G6, tol=1e-10)
    d = 16.0
    E0 = p["E"]
    L0 = boosted_cross_lagrangian(p, d, 0.0, 0.0, 0.0)
    Xs = np.array([[0, 0, 0], [0, 0, d]], float)
    out = {"L0": L0, "model_minus_U_static": -float(un.pair_energy(Xs[None], np.zeros((1, 2, 3)), np.zeros((1, 2)),
                                                                   np.array([[W0, W0]]), fam)[0]), "rows": {}}
    worst_lorentz = 0.0
    for v in (0.02, 0.04, 0.06):
        g = 1 / np.sqrt(1 - v * v)
        X = np.array([[0, 0, 0], [0, 0, d]], float)
        Lt = boosted_cross_lagrangian(p, d, v, v, W0 * g * v * d)
        Lt_pred = boosted_cross_lagrangian(p, g * d, 0.0, 0.0, 0.0) / g
        worst_lorentz = max(worst_lorentz, abs(Lt / Lt_pred - 1))
        Pt = np.array([[0, 0, g * E0 * v], [0, 0, g * E0 * v]])
        Ut = float(un.pair_energy(X[None], Pt[None], np.array([[0.0, -W0 * g * v * d]]), np.array([[W0, W0]]), fam)[0])
        Lh = boosted_cross_lagrangian(p, d, v, -v, 0.0)
        Ph = np.array([[0, 0, g * E0 * v], [0, 0, -g * E0 * v]])
        rows = {}
        for form in (True, "resummed"):
            Uh = float(un.pair_energy(X[None], Ph[None], np.zeros((1, 2)), np.array([[W0, W0]]), fam, boost=form)[0])
            rows[str(form)] = -Uh / L0
        out["rows"]["v=%g" % v] = {"translation_L_over_Lorentz_prediction_minus_1": Lt / Lt_pred - 1,
                                   "translation_model_minus_U_over_static": -Ut / out["model_minus_U_static"],
                                   "headon_superposition_L_over_L0": Lh / L0,
                                   "headon_model_default_over_static": rows["True"] * L0 / out["model_minus_U_static"],
                                   "headon_model_resummed_over_static": rows["resummed"] * L0 / out["model_minus_U_static"],
                                   "superposition_e2_effect_over_resummed": (1 - Lh / L0) / (1 - rows["resummed"] * L0 / out["model_minus_U_static"])}
    out["worst_lorentz"] = worst_lorentz
    record("boost_phase_superposition", out)
    assert worst_lorentz < 1e-4
    assert all(abs(r["translation_model_minus_U_over_static"] - 1) < 1e-9 for r in out["rows"].values())


def test_overlap_angular_momentum_of_stepped_rings(fam, W0, params):
    """Resting stepped rings: the canonical angular momentum (first order) equals the Session 2 overlap formula
    J = omega sum I_ij (x_i x x_j) (model.angular_momentum_overlap, asymptotic tails). The extended model thus
    carries the overlap angular momentum dynamically (the coupling WPD_NOTES section 10 said was missing)."""
    out, worst = {}, 0.0
    for K, mm in ((4, 1), (5, 1), (6, 1), (8, 2)):
        X, R = un.ring_positions(K, 16.0)
        th = 2 * np.pi * mm * np.arange(K) / K
        m = un.Model(fam, W0)
        z0 = m.pack(X, np.zeros((K, 3)), th, np.full(K, W0))
        P1 = -float(fam.E(W0)) * m.grad(z0, K)[3 * K:6 * K].reshape(K, 3)
        L1 = float(np.cross(X, P1).sum(0)[2])
        J = float(old.angular_momentum_overlap(X, th, params["A"], params["kappa"], params["omega"])[2])
        out["ring%d m=%d" % (K, mm)] = {"L_first_order": L1, "J_session2_formula": J, "rel": abs(L1 / J - 1)}
        worst = max(worst, abs(L1 / J - 1))
    record("overlap_angular_momentum", {"rows": out, "worst_rel": worst,
                                        "note": "asymptotic-tail J; the true-profile J is 19 to 27% smaller at d = 12 to 16 (WPD_NOTES 1.5)"})
    assert worst < 1e-6
