"""WP-C item 1: numbers behind docs/WPC_DERIVATION.md.

All from the profile table actually used for initial data
(outputs/step2/profiles/profile_beta1.3333_w0.9091.json) and the beta = 4/3 family
(outputs/family/family_beta1.3333.json, for dN/domega). No PDE is run here.

Computes
 1. tail check: Phi(r) / (A e^{-kappa r}/r) against r, and the size of the
    nonlinear source relative to the linear tail operator, (g4/8) Phi^2 / kappa^2;
 2. the leading-order tail force F(d) = 2 pi A^2 cos(dphi) e^{-kappa d} (kappa/d + 1/d^2)
    and U(d) = -2 pi A^2 cos(dphi) e^{-kappa d}/d;
 3. the superposition plane force F_sup(d, dphi): the period-averaged momentum flux
    through the mid-plane of the superposed single-frequency profiles, with the
    true Phi(r) and the full nonlinear V, self terms subtracted (a check of the
    asymptotic formula and an estimate of the next order, incomplete at O(e^{-2 kappa d}));
 4. the N current through the mid-plane, dN_2/dt = -2 pi A^2 sin(theta_2 - theta_1) e^{-kappa d}/d,
    and the phase pendulum d^2(dphi)/dt^2 = K sin(dphi), K = 2 C_N |domega/dN|;
 5. the predicted table: F, a = F/E, centroid shift per lump over T = 100, 300, 500
    (constant force, a T^2/2) and from the integrated point model (fixed dphi, and
    with the N-exchange / phase dynamics).
Writes outputs/session2/wpC/pair_law_predictions.json (atomic).
"""
import json
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PROFILE = os.path.join(ROOT, "outputs/step2/profiles/profile_beta1.3333_w0.9091.json")
FAMILY = os.path.join(ROOT, "outputs/family/family_beta1.3333.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pair_law_predictions.json")

D_LIST = [8, 10, 12, 14, 16, 18, 20, 24]
T_LIST = [100, 300, 500]
DPHI_DEG = [0, 45, 90, 135, 180]

prof = json.load(open(PROFILE))
m, g4, g6 = prof["m"], prof["g4"], prof["g6"]
om, A, kap, E_l, N_l, hw = prof["omega"], prof["A"], prof["kappa"], prof["E"], prof["N"], prof["half_width"]
r_tab, P_tab = np.array(prof["r"]), np.array(prof["Phi"])
spl = CubicSpline(r_tab, P_tab, bc_type=((1, 0.0), "not-a-knot"))
dspl = spl.derivative()
R_MAX = r_tab[-1]


def Phi(r):
    r = np.asarray(r, float)
    tail = A * np.exp(-kap * r) / np.maximum(r, 1e-300)
    return np.where(r <= R_MAX, spl(np.minimum(r, R_MAX)), tail)


def dPhi(r):
    r = np.asarray(r, float)
    tail = -A * np.exp(-kap * r) * (kap * r + 1) / np.maximum(r, 1e-300) ** 2
    return np.where(r <= R_MAX, dspl(np.minimum(r, R_MAX)), tail)


def V(p):
    p2 = p * p
    return 0.5 * m * m * p2 - g4 / 24 * p2 * p2 + g6 / 720 * p2 ** 3


def F_asym(d, dphi=0.0):
    return 2 * np.pi * A ** 2 * np.cos(dphi) * np.exp(-kap * d) * (kap / d + 1 / d ** 2)


def U_asym(d, dphi=0.0):
    return -2 * np.pi * A ** 2 * np.cos(dphi) * np.exp(-kap * d) / d


def C_N(d):
    """|dN/dt| coefficient: dN_2/dt = -C_N sin(theta_2 - theta_1)."""
    return 2 * np.pi * A ** 2 * np.exp(-kap * d) / d


def F_sup(d, th1, th2, n_tau=96, n_rho=4000):
    """Period-averaged momentum flux into the half-space x < 0 (lump 1 at -d/2) through x = 0,
    pair minus the two single lumps. Integrand on the plane, rho in [0, rho_max], 2 pi rho drho."""
    a = d / 2
    rho_max = np.sqrt(max(60.0 ** 2 - a * a, 1.0))
    rho = np.linspace(0, rho_max, n_rho)
    r = np.sqrt(a * a + rho ** 2)
    f, fp = Phi(r), dPhi(r)
    fx1, fx2 = fp * a / r, -fp * a / r          # d/dx of Phi(|x - x_i|) at x = 0
    fpe = fp * rho / r                           # perpendicular gradient, same for both
    tau = 2 * np.pi * (np.arange(n_tau) + 0.5) / n_tau

    def flux(c1, s1, c2, s2):
        px = fx1 * c1 + fx2 * c2
        pp = fpe * (c1 + c2)
        p = f * (c1 + c2)
        pt = -om * f * (s1 + s2)
        return -0.5 * px ** 2 + 0.5 * pp ** 2 - 0.5 * pt ** 2 + V(p)

    def flux_single(c, s, fx):
        return -0.5 * (fx * c) ** 2 + 0.5 * (fpe * c) ** 2 - 0.5 * (om * f * s) ** 2 + V(f * c)

    acc = np.zeros_like(rho)
    for t in tau:
        c1, s1, c2, s2 = np.cos(t + th1), np.sin(t + th1), np.cos(t + th2), np.sin(t + th2)
        acc += flux(c1, s1, c2, s2) - flux_single(c1, s1, fx1) - flux_single(c2, s2, fx2)
    acc /= n_tau
    w = 2 * np.pi * rho
    return float(np.trapezoid(acc * w, rho))


def F_sup_linear_part(d, dphi, n_rho=4000):
    """Same plane integral with V -> m^2 phi^2/2 (true Phi): (cos dphi / 2) int [Phi'^2 + kappa^2 Phi^2] dA."""
    a = d / 2
    rho_max = np.sqrt(max(60.0 ** 2 - a * a, 1.0))
    rho = np.linspace(0, rho_max, n_rho)
    r = np.sqrt(a * a + rho ** 2)
    f, fp = Phi(r), dPhi(r)
    return float(0.5 * np.cos(dphi) * np.trapezoid((fp ** 2 + kap ** 2 * f ** 2) * 2 * np.pi * rho, rho))


def N_current_sup(d, th1, th2, n_rho=4000):
    """dN_2/dt from the averaged-Lagrangian current j = -Im(Psi* grad Psi)/2 with the true Phi."""
    a = d / 2
    rho_max = np.sqrt(max(60.0 ** 2 - a * a, 1.0))
    rho = np.linspace(0, rho_max, n_rho)
    r = np.sqrt(a * a + rho ** 2)
    f, fp = Phi(r), dPhi(r)
    integrand = np.sin(th2 - th1) * (f * (-fp * a / r) - f * (fp * a / r))  # sin(dth)(f1 d_x f2 - f2 d_x f1)
    jx = -0.5 * np.trapezoid(integrand * 2 * np.pi * rho, rho)          # flux in +x (into lump 2's half)
    return float(jx)


def domega_dN():
    fam = json.load(open(FAMILY))
    rows = fam["rows_rescaled"]
    g4f = 0.30
    w = np.array([r["omega"] for r in rows])
    N = np.array([r["N"] for r in rows]) / g4f
    o = np.argsort(w)
    cs = CubicSpline(w[o], N[o])
    dNdw = float(cs.derivative()(om))
    return dNdw, float(cs(om))


def point_model(d0, dphi0, T, dwdN, with_exchange):
    """y = (d, d_dot, dphi, n): n = N_1 - N0 (lump 1 at -d/2, dphi = theta_2 - theta_1).
    d'' = -2 F(d, dphi)/E;  n' = C_N sin(dphi);  dphi' = 2 |dw/dN| n (stable branch dw/dN < 0)."""
    def rhs(t, y):
        d, v, ph, n = y
        F = F_asym(max(d, 1e-3), ph)
        dn = C_N(max(d, 1e-3)) * np.sin(ph) if with_exchange else 0.0
        dph = 2 * abs(dwdN) * n if with_exchange else 0.0
        return [v, -2 * F / E_l, dph, dn]

    def hit(t, y):
        return y[0] - 2 * hw
    hit.terminal = True
    sol = solve_ivp(rhs, [0, T], [d0, 0.0, dphi0, 0.0], rtol=1e-10, atol=1e-12, events=hit,
                    dense_output=True, max_step=1.0)
    return sol


def main():
    out = {"description": "WP-C pair law: derived numbers (no PDE). See docs/WPC_DERIVATION.md",
           "params": {"profile_file": os.path.relpath(PROFILE, ROOT), "family_file": os.path.relpath(FAMILY, ROOT),
                      "m": m, "g4": g4, "g6": g6, "beta": prof["beta"], "omega": om, "A": A, "kappa": kap,
                      "E_lump": E_l, "N_lump": N_l, "half_width": hw, "Phi0": prof["Phi0"],
                      "d_list": D_LIST, "T_list": T_LIST, "dphi_deg": DPHI_DEG,
                      "brief_values_note": "the brief's A = 55.7, E = 2767, N = 2828 are the family row at omega = 0.910 "
                                           "(A_resc 30.494 / sqrt(0.3) = 55.67); the profile used here is omega = 0.909149"}}
    # 1. tail check
    rr = np.array([3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25, 30])
    ratio = Phi(rr) / (A * np.exp(-kap * rr) / rr)
    nl = g4 / 8 * Phi(rr) ** 2 / kap ** 2
    out["tail_check"] = {"r": rr.tolist(), "Phi": Phi(rr).tolist(), "Phi_over_tail": ratio.tolist(),
                         "nonlinear_source_over_kappa2": nl.tolist(),
                         "r_where_tail_within_1pct": float(rr[np.argmax(np.abs(ratio - 1) < 0.01)]),
                         "r_where_tail_within_10pct": float(rr[np.argmax(np.abs(ratio - 1) < 0.10)])}
    # finer: first r beyond which |ratio - 1| stays below 1% and 10%
    rf = np.linspace(1, 30, 2901)
    rf_ratio = Phi(rf) / (A * np.exp(-kap * rf) / rf)
    for tol in (0.01, 0.03, 0.1):
        bad = np.nonzero(np.abs(rf_ratio - 1) >= tol)[0]
        out["tail_check"]["r_beyond_which_within_%g" % tol] = float(rf[bad[-1] + 1]) if bad.size else 1.0
    # 3/4. dN/domega and the phase-pendulum constants
    dNdw, N_spline = domega_dN()
    dwdN = 1.0 / dNdw
    out["dN_domega"] = dNdw
    out["domega_dN"] = dwdN
    out["N_family_spline_at_omega"] = N_spline
    # 5. table
    rows = []
    for d in D_LIST:
        F0 = F_asym(d)
        Fs = {str(p): F_sup(d, 0.0, np.deg2rad(p)) for p in DPHI_DEG}
        Fl = {str(p): F_sup_linear_part(d, np.deg2rad(p)) for p in DPHI_DEG}
        jN = N_current_sup(d, 0.0, np.pi / 2)
        K = 2 * C_N(d) * abs(dwdN)
        row = {"d": d, "d_over_hw": d / hw, "kappa_d": kap * d,
               "Phi_at_midplane": float(Phi(d / 2)), "Phi_mid_over_Phi0": float(Phi(d / 2) / prof["Phi0"]),
               "nonlinear_source_over_kappa2_at_midplane": float(g4 / 8 * Phi(d / 2) ** 2 / kap ** 2),
               "U_asym_dphi0": U_asym(d), "F_asym_dphi0": F0, "a_asym_dphi0": F0 / E_l,
               "F_sup_by_dphi_deg": Fs, "F_sup_linear_part_by_dphi_deg": Fl,
               "F_sup_over_F_asym_dphi0": Fs["0"] / F0,
               "F_sup_linear_part_over_F_asym_dphi0": Fl["0"] / F0,
               "C_N_asym": C_N(d), "dN2dt_sup_dphi90": jN, "dN2dt_asym_dphi90": -C_N(d),
               "K_phase_pendulum": K, "phase_time_1_over_sqrtK": 1 / np.sqrt(K),
               "small_osc_period_about_pi": 2 * np.pi / np.sqrt(K),
               "N_swing_max_from_90deg": float(np.sqrt(C_N(d) / abs(dwdN))),
               "shift_const_force": {}, "shift_point_model_fixed_dphi": {}, "shift_point_model_exchange": {}}
        for p in DPHI_DEG:
            ph = np.deg2rad(p)
            a = F_asym(d, ph) / E_l
            row["shift_const_force"][str(p)] = {str(T): 0.5 * a * T ** 2 for T in T_LIST}
            for key, ex in (("shift_point_model_fixed_dphi", False), ("shift_point_model_exchange", True)):
                sol = point_model(d, ph, max(T_LIST), dwdN, ex)
                rec = {}
                for T in T_LIST:
                    if T <= sol.t[-1]:
                        y = sol.sol(T)
                        rec[str(T)] = {"shift_per_lump": float((d - y[0]) / 2), "d": float(y[0]),
                                       "dphi_deg": float(np.rad2deg(y[2])), "N1_minus_N0": float(y[3])}
                    else:
                        rec[str(T)] = {"reached_d_2hw_at": float(sol.t[-1])}
                row[key][str(p)] = rec
        rows.append(row)
    out["table"] = rows
    # merger time of the in-phase pair to d = 2 hw, point model, fixed phase
    out["merge_time_point_model_dphi0"] = {}
    for d in D_LIST:
        sol = point_model(d, 0.0, 1e5, dwdN, False)
        out["merge_time_point_model_dphi0"][str(d)] = float(sol.t[-1]) if sol.status == 1 else None
    with open(OUT + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(OUT + ".tmp", OUT)
    # console summary
    print("dN/domega = %.1f, domega/dN = %.3e" % (dNdw, dwdN))
    print("tail:", {k: v for k, v in out["tail_check"].items() if k.startswith("r_")})
    print("%4s %7s %9s %9s %9s %9s %8s %8s %8s %9s %9s" % ("d", "Phi_mid", "F_asym", "Fsup/Fas", "Fsup90", "Fsup180", "a", "s100", "s300", "s500", "merge"))
    for r in rows:
        s = r["shift_const_force"]["0"]
        print("%4d %7.4f %9.4g %9.4f %9.3g %9.4g %8.2e %8.3g %8.3g %9.3g %9s" % (
            r["d"], r["Phi_at_midplane"], r["F_asym_dphi0"], r["F_sup_over_F_asym_dphi0"],
            r["F_sup_by_dphi_deg"]["90"], r["F_sup_by_dphi_deg"]["180"], r["a_asym_dphi0"],
            s["100"], s["300"], s["500"], out["merge_time_point_model_dphi0"][str(r["d"])]))
    for r in rows:
        print("d=%d C_N %.4g (sup %.4g)  K %.3g  1/sqrtK %.1f  N swing %.1f  exch90@500 %s" % (
            r["d"], r["C_N_asym"], r["dN2dt_sup_dphi90"], r["K_phase_pendulum"], r["phase_time_1_over_sqrtK"],
            r["N_swing_max_from_90deg"], r["shift_point_model_exchange"]["90"]["500"]))


if __name__ == "__main__":
    main()
