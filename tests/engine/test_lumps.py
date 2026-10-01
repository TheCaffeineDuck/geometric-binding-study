"""WP-C diagnostics against analytic or independent values (standard 6).

Every test records its measured numbers to outputs/session2/wpC/checks/<name>.json.
Profile: the beta = 4/3, omega = 0.909149 primary (outputs/step2/profiles/).
"""

import json
import os

import numpy as np
import pytest
from scipy.special import roots_legendre

from swirl.engine.evolver import Evolver
from swirl.engine.initial import build_initial, radial_spline
from swirl.engine.lumps import (LumpDiagnostics, ball_weight, check_lumps, densities, find_lumps,
                                lump_config, lumps_init, summarize)
from swirl.engine.run import full_config, run

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROF = json.load(open(os.path.join(ROOT, "outputs/step2/profiles/profile_beta1.3333_w0.9091.json")))
M, G4, G6, OM = PROF["m"], PROF["g4"], PROF["g6"], PROF["omega"]
A, KAP = PROF["A"], PROF["kappa"]
SPL = radial_spline(PROF["r"], PROF["Phi"])
DSPL = SPL.derivative()
RMAX = PROF["r"][-1]
CHECK_DIR = os.path.join(ROOT, "outputs", "session2", "wpC", "checks")


@pytest.fixture
def rec():
    def _rec(name, data):
        os.makedirs(CHECK_DIR, exist_ok=True)
        path = os.path.join(CHECK_DIR, name + ".json")
        with open(path + ".tmp", "w") as f:
            json.dump(data, f, indent=1, default=float)
        os.replace(path + ".tmp", path)
    return _rec


def Phi(r):
    r = np.asarray(r, float)
    return np.where(r <= RMAX, SPL(np.minimum(r, RMAX)), 0.0)


def dPhi(r):
    r = np.asarray(r, float)
    return np.where(r <= RMAX, DSPL(np.minimum(r, RMAX)), 0.0)


def V(p):
    p2 = p * p
    return 0.5 * M * M * p2 - G4 / 24 * p2 * p2 + G6 / 720 * p2 ** 3


def gl_nodes(a, b, panels, n=16):
    x, w = roots_legendre(n)
    edges = np.linspace(a, b, panels + 1)
    X = (0.5 * (edges[1:] - edges[:-1])[:, None] * x[None, :] + 0.5 * (edges[1:] + edges[:-1])[:, None]).ravel()
    Wt = (0.5 * (edges[1:] - edges[:-1])[:, None] * w[None, :]).ravel()
    return X, Wt


def axisym_quad(f, xlim=46.0, rholim=46.0):
    """int f(x, rho) d^3x for a field axisymmetric about the x axis (Gauss-Legendre panels)."""
    x, wx = gl_nodes(-xlim, xlim, int(2 * xlim))
    rho, wr = gl_nodes(0.0, rholim, int(rholim))
    X, R = np.meshgrid(x, rho, indexing="ij")
    return float(np.einsum("i,j,ij->", wx, wr * 2 * np.pi * rho, f(X, R)))


def F_asym(d, dphi=0.0):
    return 2 * np.pi * A ** 2 * np.cos(dphi) * np.exp(-KAP * d) * (KAP / d + 1 / d ** 2)


def _evolver_with(init, N, L, absorber=None):
    ev = Evolver(N, L, M, G4, G6, absorber=absorber, threads=4)
    phi, pi = build_initial(ev.coords, ev.L, init)
    ev.set_state(phi, pi)
    return ev


# ---------------------------------------------------------------------------
# 1. centroid of an off-lattice lump (exact answer: the lump centre)
# ---------------------------------------------------------------------------

def test_centroid_off_lattice(rec):
    c = np.array([0.37, -0.21, 0.29])
    ev = _evolver_with(lumps_init(PROF, [c], [0.0], OM), 96, 70.0)
    start = c + np.array([0.4, -0.3, 0.2])          # ball starts 0.54 off the lump
    ld = LumpDiagnostics(ev, lump_config({"centers": [start.tolist()]}))
    for _ in range(4):                               # the ball follows the centroid
        ld.sample(0.0)
    err = [float(np.linalg.norm(np.array(x[0]) - c)) for x in ld.series["X"]]
    rec("centroid_off_lattice", {"N": 96, "L": 70.0, "dx": ev.dx, "center": c.tolist(),
                                 "ball_start_offset": 0.5385, "R_ball": 10.0, "taper": 2.0,
                                 "abs_error_per_iteration": err, "target": 1e-4})
    assert err[-1] < 1e-4
    assert err[1] < err[0]


# ---------------------------------------------------------------------------
# 2. slowly moving lump: P against quadrature, dX/dt = P/E, P = E v of the ansatz
# ---------------------------------------------------------------------------

def _boosted(v, b):
    """Lorentz-boosted single-frequency lump along x, centred at (0, b, 0), t = 0."""
    g = 1 / np.sqrt(1 - v * v)

    def fields(x, rho2):
        rp = np.sqrt(g * g * x * x + rho2)
        ph = OM * g * v * x
        P, dP = Phi(rp), dPhi(rp)
        rs = np.maximum(rp, 1e-12)
        phi = P * np.cos(ph)
        phit = -v * g * g * x * dP / rs * np.cos(ph) + OM * g * P * np.sin(ph)
        phix = dP * g * g * x / rs * np.cos(ph) - P * OM * g * v * np.sin(ph)
        drho_over_rho = dP / rs * np.cos(ph)          # d phi / d rho = drho_over_rho * rho
        return phi, phit, phix, drho_over_rho
    return fields


def test_moving_lump_momentum(rec):
    v, b, N, L = 0.02, 3.0, 96, 70.0
    fields = _boosted(v, b)
    # independent quadrature (axisymmetric about the line y = b, z = 0)
    def e_q(X, R):
        phi, pt, px, dr = fields(X, R * R)
        return 0.5 * pt ** 2 + 0.5 * px ** 2 + 0.5 * (dr * R) ** 2 + V(phi)

    def p_q(X, R):
        phi, pt, px, dr = fields(X, R * R)
        return -pt * px
    E_q, P_q = axisym_quad(e_q), axisym_quad(p_q)
    # engine
    cfg = {"N": N, "L": L, "m": M, "g4": G4, "g6": G6, "dt": L / N / 10, "t_end": 30.0,
           "init": {"gaussians": []}, "r_c": [10.0], "center": [0.0, b, 0.0], "sample_every": 1.0,
           "lumps": {"centers": [[0.0, b, 0.0]], "every": 4, "R_ball": 12.0,
                     "axes": [{"origin": [0, 0, 0], "dir": [0, 0, 1]}, {"origin": [0, b, 0], "dir": [0, 0, 1]}],
                     "count_every": 1e9}}
    c = full_config(cfg)
    ev = Evolver(N, L, M, G4, G6, threads=4)
    x = ev.coords
    X3, Y3, Z3 = np.meshgrid(x, x - b, x, indexing="ij")
    phi, pt, _, _ = fields(X3, Y3 ** 2 + Z3 ** 2)
    ev.set_state(phi, pt)
    ld = LumpDiagnostics(ev, c["lumps"])
    E0 = ev.energy()
    # evolve by hand (same loop as run(): read points every step, sample every 4)
    dt, n_steps = c["dt"], int(round(c["t_end"] / c["dt"]))
    for n in range(n_steps + 1):
        ld.read_points(n * dt)
        if n % 4 == 0:
            ld.sample(n * dt)
        if n < n_steps:
            ev.step(dt)
    st = ld.state()
    s = st["series"]
    Pt = np.array(s["P_total"])
    P0 = Pt[0, 0]
    Lz = np.array(s["L_axes"])
    av = summarize(st, np.arange(0, c["t_end"] + 1e-9, 1.0), 14.0)
    lu = av["lumps"][0]
    ok = [i for i, xx in enumerate(lu["X"]) if xx is not None]
    tc = np.array([lu["t_center"][i] for i in ok])
    Xc = np.array([lu["X"][i][0] for i in ok])
    u = float(np.polyfit(tc, Xc, 1)[0])
    E_end = ev.energy()
    G = 4 * np.pi * np.trapezoid(dPhi(np.array(PROF["r"])) ** 2 * np.array(PROF["r"]) ** 2, PROF["r"])
    Q = 4 * np.pi * np.trapezoid(Phi(np.array(PROF["r"])) ** 2 * np.array(PROF["r"]) ** 2, PROF["r"])
    P_small_v = v * (G / 3 + OM ** 2 * Q / 2)                 # O(v) of the ansatz momentum
    out = {"v": v, "b": b, "N": N, "L": L, "dt": dt, "T": c["t_end"],
           "E0_engine": E0, "E0_quadrature": E_q, "rel_err_E0": E0 / E_q - 1,
           "P0_engine": P0, "P0_quadrature": P_q, "rel_err_P0": P0 / P_q - 1,
           "P_small_v_formula_v(G/3+omega^2 Q/2)": P_small_v, "rel_P_quad_vs_small_v_formula": P_q / P_small_v - 1,
           "max_rel_drift_P_total": float(np.max(np.abs(Pt[:, 0] - P0)) / abs(P0)),
           "max_abs_P_transverse": float(np.max(np.abs(Pt[:, 1:]))),
           "rel_drift_E": E_end / E0 - 1,
           "centroid_velocity_fit": u, "P_over_E": P0 / E0, "rel_err_u_vs_P_over_E": u / (P0 / E0) - 1,
           "P_over_E_v_minus_1": P0 / (E0 * v) - 1,
           "Lz_origin_t0": float(Lz[0, 0]), "minus_b_P0": -b * P0, "rel_err_Lz_lever": float(Lz[0, 0] / (-b * P0) - 1),
           "Lz_own_axis_t0": float(Lz[0, 1]),
           "targets": {"rel_err_E0": 1e-6, "rel_err_P0": 1e-5, "P_drift": 1e-4, "u_vs_P_over_E": 2e-3,
                       "Lz_lever": 1e-5},
           "note": "P = E v holds for exactly periodic solutions; the boosted single-frequency ansatz is not one, "
                   "so P/(E v) - 1 is reported, not targeted. The identity tested is dX_E/dt = P/E."}
    rec("moving_lump_momentum", out)
    assert abs(out["rel_err_E0"]) < 1e-6
    assert abs(out["rel_err_P0"]) < 1e-5
    assert out["max_rel_drift_P_total"] < 1e-4
    assert abs(out["rel_err_u_vs_P_over_E"]) < 2e-3
    assert abs(out["rel_err_Lz_lever"]) < 1e-5
    assert abs(out["Lz_own_axis_t0"]) < 1e-6 * abs(b * P0)


# ---------------------------------------------------------------------------
# 3. angular momentum of a spinning l = 1 Gaussian: L_z = omega a^2 pi^{3/2} R^5 / 2
# ---------------------------------------------------------------------------

def test_angular_momentum_spinning(rec):
    a, R, w, N, L = 0.5, 2.5, 0.9, 64, 40.0
    c = np.array([2.3, -1.7, 0.9])
    ev = Evolver(N, L, M, G4, G6, threads=4)
    x = ev.coords
    X, Y, Z = np.meshgrid(x - c[0], x - c[1], x - c[2], indexing="ij")
    G = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * R * R))
    ev.set_state(a * X * G, w * a * Y * G)          # phi = a rho G cos(phi_az - w t)
    cfg = lump_config({"centers": [c.tolist()], "count_every": 1e9,
                       "axes": [{"origin": c.tolist(), "dir": [0, 0, 1]},
                                {"origin": [0, 0, 0], "dir": [0, 0, 1]},
                                {"origin": c.tolist(), "dir": [1, 0, 0]},
                                {"origin": c.tolist(), "dir": [0, 0, -2]}]})
    ld = LumpDiagnostics(ev, cfg)
    ld.sample(0.0)
    Lx = ld.series["L_axes"][0]
    exact = w * a * a * np.pi ** 1.5 * R ** 5 / 2
    out = {"a": a, "R": R, "omega": w, "center": c.tolist(), "N": N, "L": L, "L_exact_z": exact,
           "L_z_about_c": Lx[0], "L_z_about_origin": Lx[1], "L_x_about_c": Lx[2], "L_minus_z_about_c": Lx[3],
           "rel_err_z_c": Lx[0] / exact - 1, "rel_err_z_origin": Lx[1] / exact - 1,
           "P_total": ld.series["P_total"][0], "target": 1e-8}
    rec("angular_momentum_spinning", out)
    assert abs(out["rel_err_z_c"]) < 1e-8
    assert abs(out["rel_err_z_origin"]) < 1e-8
    assert abs(Lx[2]) < 1e-8 * exact
    assert abs(Lx[3] / exact + 1) < 1e-8
    assert np.max(np.abs(out["P_total"])) < 1e-8 * exact


# ---------------------------------------------------------------------------
# 4. relative phase: estimator on synthetic signals; exact symmetric case; far-apart pair
# ---------------------------------------------------------------------------

def test_relative_phase_estimator_synthetic(rec):
    """phi_k = cos(w t + th_k) (1 + 0.1 t/T) + 0.05 cos(3 (w t + th_k)): dtheta = th_1 - th_0."""
    w, th0, th1, dt = 0.9093, 0.4, 0.4 + 1.234, 0.04
    t = np.arange(0, 60 + 1e-9, dt)
    sig = lambda th: np.cos(w * t + th) * (1 + 0.1 * t / 60) + 0.05 * np.cos(3 * (w * t + th))
    ts = t[::5]
    zero = np.zeros((ts.size, 2))
    st = {"centers": [[0, 0, 0], [1, 0, 0]],
          "series": {"t": ts.tolist(), "E": zero.tolist(), "X": np.zeros((ts.size, 2, 3)).tolist(),
                     "It": zero.tolist(), "Ip": zero.tolist(), "P": np.zeros((ts.size, 2, 3)).tolist(),
                     "P_total": [[0.0, 0.0, 0.0]] * ts.size, "P_half": [[]] * ts.size, "L_axes": [[]] * ts.size},
          "point": {"t": t.tolist(), "phi": np.stack([sig(th0), sig(th1)], 1).tolist()}}
    av = summarize(st, np.arange(20, 60.1, 5.0), 20.0)
    d = np.array([x for x in av["relative_phase"][0]["dtheta"] if x is not None])
    err = float(np.max(np.abs(d - (th1 - th0))))
    om = np.array([x for x in av["lumps"][0]["omega"] if x is not None])
    rec("relative_phase_synthetic", {"dtheta_set": th1 - th0, "max_abs_err": err,
                                     "omega_max_abs_err": float(np.max(np.abs(om - w))), "target": 1e-3})
    assert d.size >= 5 and err < 1e-3


def _pair_run(d, th, N, L, T, every=4, extra=None):
    cen = [[-d / 2, 0, 0], [d / 2, 0, 0]] if len(th) == 2 else [[-d / 2, 0, 0]]
    cfg = {"N": N, "L": L, "m": M, "g4": G4, "g6": G6, "dt": L / N / 10, "t_end": T,
           "init": lumps_init(PROF, cen, th, OM), "absorber": True, "r_c": [10.0],
           "center": cen[0], "sample_every": 1.0, "threads": 4,
           "lumps": {"centers": cen, "every": every, "planes": [{"axis": 0, "pos": 0.0}],
                     "count_every": 10.0, **(extra or {})}}
    return run(cfg)


def _cross_run_phase(ra, rb, W=14.0):
    """theta_b - theta_a from lump 0 of two runs on the same time axis (same estimator)."""
    pa, pb = ra["lumps"]["raw"]["point"], rb["lumps"]["raw"]["point"]
    assert pa["t"] == pb["t"]
    phis = np.stack([np.array(pa["phi"])[:, 0], np.array(pb["phi"])[:, 0]], 1)
    s = ra["lumps"]["raw"]["series"]
    n = len(s["t"])
    st = {"centers": [[0, 0, 0], [0, 0, 0]],
          "series": {"t": s["t"], "E": [[0.0, 0.0]] * n, "X": [[[0.0] * 3] * 2] * n, "It": [[0.0, 0.0]] * n,
                     "Ip": [[0.0, 0.0]] * n, "P": [[[0.0] * 3] * 2] * n, "P_total": [[0.0] * 3] * n,
                     "P_half": [[]] * n, "L_axes": [[]] * n},
          "point": {"t": pa["t"], "phi": phis.tolist()}}
    av = summarize(st, ra["series"]["t"], W)
    return av["relative_phase"][0]["dtheta"]


def test_relative_phase_pde(rec):
    """Anti-phase pair: phi(-x) = -phi(x) is exact, so dtheta = pi exactly. A d = 20 pair
    at -45/+45 degrees against the independent value from two single-lump runs started
    at -45 and +45 degrees at the same positions (the settling of the single-frequency
    start shifts each lump's phase, so the pair is not at exactly 90 degrees; the
    interaction adds K t^2/2 ~ 3e-3 rad by t = 24 at d = 20)."""
    d, N, L, T = 20.0, 96, 82.0, 24.0
    r1 = _pair_run(d, [0.0, np.pi], N, L, T)
    r2 = _pair_run(d, [-np.pi / 4, np.pi / 4], N, L, T)
    sa = _pair_run(d, [-np.pi / 4], N, L, T)
    sb = run({**sa["params"], "center": [d / 2, 0, 0],
              "init": lumps_init(PROF, [[d / 2, 0, 0]], [np.pi / 4], OM),
              "lumps": {**sa["params"]["lumps"], "centers": [[d / 2, 0, 0]]}})
    d1 = np.array([x for x in r1["lumps"]["averages"]["relative_phase"][0]["dtheta"] if x is not None])
    d2 = r2["lumps"]["averages"]["relative_phase"][0]["dtheta"]
    ds = _cross_run_phase(sa, sb)
    both = [(x, y) for x, y in zip(d2, ds) if x is not None and y is not None]
    d2v = np.array([x for x, _ in both])
    dsv = np.array([y for _, y in both])
    e1 = float(np.max(np.abs(np.abs(d1) - np.pi)))
    e2 = float(np.max(np.abs(d2v - dsv)))
    rec("relative_phase_pde", {"d": d, "N": N, "L": L, "T": T,
                               "antiphase_max_abs_err": e1, "pm45_pair_dtheta": d2v.tolist(),
                               "pm45_from_singles": dsv.tolist(), "pm45_max_abs_pair_minus_singles": e2,
                               "pm45_max_abs_minus_90deg": float(np.max(np.abs(d2v - np.pi / 2))),
                               "targets": {"antiphase": 1e-8, "pair_vs_singles": 5e-3},
                               "note": "first version asserted |dtheta - pi/2| < 0.02 and measured 0.045: "
                                       "the 0.02 was a guess; the offset is the start-phase dependent settling "
                                       "of the single-frequency ansatz, measured here by the single-lump runs"})
    assert d1.size > 3 and e1 < 1e-8
    assert d2v.size > 3 and e2 < 5e-3


# ---------------------------------------------------------------------------
# 5. lump count from the period-averaged energy density of the superposition
# ---------------------------------------------------------------------------

def _ansatz_avg_density(ev, d, th, nph=32):
    acc = np.zeros(ev.shape)
    for k in range(nph):
        tau = 2 * np.pi * (k + 0.5) / nph
        init = lumps_init(PROF, [[-d / 2, 0, 0], [d / 2, 0, 0]], [t + tau for t in th], OM)
        phi, pi = build_initial(ev.coords, ev.L, init)
        ev.set_state(phi, pi)
        acc += densities(ev)[0]
    return acc / nph


def test_lump_count(rec):
    ev = Evolver(104, 78.0, M, G4, G6, threads=4)
    out = {}
    for name, d, th in (("in_phase_d2", 2.0, [0, 0]), ("in_phase_d6", 6.0, [0, 0]),
                        ("in_phase_d8", 8.0, [0, 0]), ("anti_phase_d6", 6.0, [0, np.pi]),
                        ("in_phase_d16", 16.0, [0, 0])):
        res = find_lumps(_ansatz_avg_density(ev, d, th), ev.coords, ev.L, ev.dx)
        out[name] = {k: res[k] for k in ("count", "count_distinct", "peaks", "pairs")}
    rec("lump_count", out)
    assert out["in_phase_d2"]["count"] == 1 and out["in_phase_d2"]["count_distinct"] == 1
    assert out["in_phase_d16"]["count_distinct"] == 2
    assert out["anti_phase_d6"]["count_distinct"] == 2
    xs = sorted(p["pos"][0] for p in out["in_phase_d16"]["peaks"])   # separated: maxima at the centres
    assert abs(xs[0] + 8.0) < 0.1 and abs(xs[1] - 8.0) < 0.1


# ---------------------------------------------------------------------------
# 6. pair energy, ball energies and N at t = 0 against an independent quadrature
# ---------------------------------------------------------------------------

def test_pair_t0_against_quadrature(rec):
    d, th1, th2, N, L = 10.0, 0.0, np.deg2rad(60.0), 128, 72.0
    R, tap, ell = 10.0, 2.0, 1.0
    ev = _evolver_with(lumps_init(PROF, [[-d / 2, 0, 0], [d / 2, 0, 0]], [th1, th2], OM), N, L)
    ld = LumpDiagnostics(ev, lump_config({"centers": [[-d / 2, 0, 0], [d / 2, 0, 0]], "count_every": 1e9,
                                          "R_ball": R, "taper": tap, "split_width": ell}))
    ld.sample(0.0)
    E_eng = ev.energy()

    def fld(X, Rr, ths):
        r1 = np.sqrt((X + d / 2) ** 2 + Rr ** 2)
        r2 = np.sqrt((X - d / 2) ** 2 + Rr ** 2)
        f1, f2, g1, g2 = Phi(r1), Phi(r2), dPhi(r1), dPhi(r2)
        c1, c2, s1, s2 = np.cos(ths[0]), np.cos(ths[1]), np.sin(ths[0]), np.sin(ths[1])
        phi = f1 * c1 + f2 * c2
        pt = -OM * (f1 * s1 + f2 * s2)
        px = g1 * (X + d / 2) / np.maximum(r1, 1e-12) * c1 + g2 * (X - d / 2) / np.maximum(r2, 1e-12) * c2
        pr = g1 * Rr / np.maximum(r1, 1e-12) * c1 + g2 * Rr / np.maximum(r2, 1e-12) * c2
        return phi, pt, px, pr, r1, r2

    def dens(X, Rr, ths=(th1, th2)):
        phi, pt, px, pr, _, _ = fld(X, Rr, ths)
        return 0.5 * pt ** 2 + 0.5 * (px ** 2 + pr ** 2) + V(phi)

    def wts(X, Rr, k):
        r1 = np.sqrt((X + d / 2) ** 2 + Rr ** 2)
        r2 = np.sqrt((X - d / 2) ** 2 + Rr ** 2)
        if k == 0:
            return ball_weight(r1, R, tap) * 0.5 * (1 - np.tanh(X / ell))
        return ball_weight(r2, R, tap) * 0.5 * (1 - np.tanh(-X / ell))

    E_q = axisym_quad(dens)
    big = {"E": [], "It": [], "Ip": []}
    for k in (0, 1):
        big["E"].append(axisym_quad(lambda X, Rr: dens(X, Rr) * wts(X, Rr, k)))
        big["It"].append(axisym_quad(lambda X, Rr: fld(X, Rr, (th1, th2))[1] ** 2 * wts(X, Rr, k)))
        big["Ip"].append(axisym_quad(lambda X, Rr: fld(X, Rr, (th1, th2))[0] ** 2 * wts(X, Rr, k)))
    # overlap (cross) energy: pair minus the two singles, engine against quadrature
    single = []
    for c, th in (([-d / 2, 0, 0], th1), ([d / 2, 0, 0], th2)):
        single.append(_evolver_with(lumps_init(PROF, [c], [th], OM), N, L).energy())
    Ecross_eng = E_eng - single[0] - single[1]

    def dens_single(X, Rr, k):
        r = np.sqrt((X + (d / 2 if k == 0 else -d / 2)) ** 2 + Rr ** 2)
        th = th1 if k == 0 else th2
        f, g = Phi(r), dPhi(r)
        return 0.5 * (OM * f * np.sin(th)) ** 2 + 0.5 * (g * np.cos(th)) ** 2 + V(f * np.cos(th))
    Ecross_q = E_q - axisym_quad(lambda X, Rr: dens_single(X, Rr, 0)) - axisym_quad(lambda X, Rr: dens_single(X, Rr, 1))
    s = ld.series
    out = {"d": d, "theta_deg": [0.0, 60.0], "N": N, "L": L, "dx": ev.dx, "R_ball": R, "taper": tap, "split_width": ell,
           "E_engine": E_eng, "E_quadrature": E_q, "rel_err_E": E_eng / E_q - 1,
           "E_cross_engine": Ecross_eng, "E_cross_quadrature": Ecross_q, "rel_err_E_cross": Ecross_eng / Ecross_q - 1,
           "ball": {k: {"engine": [s[k][0][0], s[k][0][1]], "quadrature": big[k],
                        "rel_err": [s[k][0][i] / big[k][i] - 1 for i in (0, 1)]} for k in ("E", "It", "Ip")},
           "targets": {"E": 1e-7, "E_cross": 1e-4, "ball": 1e-6}}
    rec("pair_t0_quadrature", out)
    assert abs(out["rel_err_E"]) < 1e-7
    assert abs(out["rel_err_E_cross"]) < 1e-4
    for k in ("E", "It", "Ip"):
        assert max(abs(x) for x in out["ball"][k]["rel_err"]) < 1e-6


# ---------------------------------------------------------------------------
# 7. half-space momentum flux at early times against the derived force
# ---------------------------------------------------------------------------

def test_halfspace_force_d16(rec):
    """F = d/dt P(x < 0), whole-period averaged, pair minus twice the single-lump control
    (the control carries the self term and the radiation crossing the plane). Derived:
    F_asym = 2 pi A^2 e^{-kappa d}(kappa/d + 1/d^2); superposition plane integral F_sup."""
    d, N, L, T = 16.0, 96, 78.0, 30.0
    pair = _pair_run(d, [0.0, 0.0], N, L, T, every=3)
    sing = _pair_run(d, [0.0], N, L, T, every=3)

    def avg_series(res):
        av = res["lumps"]["averages"]
        ok = [i for i, x in enumerate(av["P_half"]) if x is not None]
        return (np.array([av["t_center_lump0"][i] for i in ok]), np.array([av["P_half"][i][0] for i in ok]))
    tp, Pp = avg_series(pair)
    ts_, Ps = avg_series(sing)
    Ps_i = np.interp(tp, ts_, Ps)
    F_pair_only = float(np.polyfit(tp, Pp, 1)[0])
    F_meas = float(np.polyfit(tp, Pp - 2 * Ps_i, 1)[0])
    pred = json.load(open(os.path.join(ROOT, "outputs/session2/wpC/pair_law_predictions.json")))
    row = [r for r in pred["table"] if r["d"] == 16][0]
    F_sup = row["F_sup_by_dphi_deg"]["0"]
    dth = [x for x in pair["lumps"]["averages"]["relative_phase"][0]["dtheta"] if x is not None]
    counts = [c["count"] for c in pair["lumps"]["counts"]]
    out = {"d": d, "N": N, "L": L, "T": T, "fit_window_t_center": [float(tp[0]), float(tp[-1])],
           "F_measured": F_meas, "F_pair_only_no_control": F_pair_only,
           "rel_pair_only_vs_F_asym": F_pair_only / F_asym(d) - 1,
           "note": "F_measured subtracts twice the single-lump plane momentum (incoherent radiation). "
                   "The two lumps radiate coherently and, for the mirror-symmetric pair, d_x phi = 0 on the "
                   "plane, so the radiation flux there is not the sum of the singles; the radiation share "
                   "of the early-time slope lies between 0 and 2x the single value (here 0 to 0.08).",
           "F_asym": F_asym(d), "F_sup": F_sup, "rel_vs_F_asym": F_meas / F_asym(d) - 1,
           "rel_vs_F_sup": F_meas / F_sup - 1, "max_abs_dtheta": float(np.max(np.abs(dth))),
           "lump_counts": counts, "target_rel": 0.10}
    rec("halfspace_force_d16", out)
    assert F_meas > 0
    assert abs(out["rel_vs_F_asym"]) < 0.10
    assert out["max_abs_dtheta"] < 1e-8
    assert all(c == 2 for c in counts)


# ---------------------------------------------------------------------------
# 8. N in a ball, whole-period average, for a uniform field (exact)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m", [1.113, 0.93])
def test_ball_N_uniform_field(m, rec):
    """phi = A cos(m t) everywhere (linear): N_ball = A^2 m W / 2 and E_ball = A^2 m^2 W / 2,
    W = sum of the ball weights dV on the grid (W against 4 pi int b r^2 dr also recorded)."""
    a, L, N = 0.1, 30.0, 24
    cfg = {"N": N, "L": L, "m": m, "g4": 0.0, "g6": 0.0, "dt": 0.05, "t_end": 60.0,
           "init": {"gaussians": [{"center": [0, 0, 0], "amp": a, "R": 1e6}]}, "r_c": [5.0],
           "sample_every": 1.0, "lumps": {"centers": [[0.3, -0.2, 0.1]], "every": 5, "R_ball": 8.0,
                                          "count_every": 1e9, "avg_window": 20.0}}
    res = run(cfg)
    ev = res["_evolver"]
    x = ev.coords
    X, Y, Z = np.meshgrid(x - 0.3, x + 0.2, x - 0.1, indexing="ij")
    Wg = float(ball_weight(np.sqrt(X ** 2 + Y ** 2 + Z ** 2), 8.0, 2.0).sum()) * ev.dV
    r = np.linspace(0, 8, 80001)
    Wc = float(4 * np.pi * np.trapezoid(ball_weight(r, 8.0, 2.0) * r * r, r))
    lu = res["lumps"]["averages"]["lumps"][0]
    Nv = np.array([v for v in lu["N"] if v is not None])
    Ev = np.array([v for v in lu["E"] if v is not None])
    dN = float(np.max(np.abs(Nv / (a * a * m * Wg / 2) - 1)))
    dE = float(np.max(np.abs(Ev / (a * a * m * m * Wg / 2) - 1)))
    dR = float(np.max(np.abs(Nv * m / Ev - 1)))       # N/E = 1/m exactly, whatever the ball volume
    Xs = np.array(res["lumps"]["raw"]["series"]["X"])[:, 0, :]
    rec("ball_N_uniform_m%s" % m, {"m": m, "A": a, "N": N, "L": L, "dx": L / N, "lump_every_steps": 5, "dt": 0.05,
                                   "W_grid_at_start_centre": Wg, "W_continuum": Wc,
                                   "rel_W_grid_vs_continuum": Wg / Wc - 1,
                                   "max_rel_dev_N_vs_A2mW/2": dN, "max_rel_dev_E_vs_A2m2W/2": dE,
                                   "max_rel_dev_N_m_over_E": dR, "n_values": int(Nv.size),
                                   "ball_centre_wander": float(np.max(np.linalg.norm(Xs - Xs[0], axis=1))),
                                   "target_N_m_over_E": 1e-4,
                                   "note": "dN and dE are equal: the ball follows the centroid, which wanders in a "
                                           "uniform field, and the grid sum of the ball weight depends on the ball "
                                           "position at this dx (1.25). The ratio N m / E removes the volume."})
    assert Nv.size > 30
    assert dR < 1e-4
    assert dN < 1e-4


# ---------------------------------------------------------------------------
# 9. checkpoint resume with the new diagnostics is bit-identical
# ---------------------------------------------------------------------------

CFG_R = {
    "N": 32, "L": 30.0, "m": 1.0, "g4": 0.30, "g6": 0.055, "dt": 0.1, "t_end": 12.0,
    "init": {"gaussians": [{"center": [-3.1, 0.2, 0], "amp": 1.5, "R": 2.0},
                           {"center": [3.0, 0, 0.3], "amp": 1.5, "R": 2.0, "phase": 1.0, "omega": 0.9}]},
    "absorber": {"gamma": 2.0, "width": 6.0, "power": 3},
    "center": [-3.1, 0.2, 0.0], "r_c": [3.0, 6.0], "sample_every": 0.5,
    "avg_window": 4.0, "checkpoint_every": 2.0, "threads": 4,
    "lumps": {"centers": [[-3.1, 0.2, 0], [3.0, 0, 0.3]], "R_ball": 4.0, "taper": 1.0, "every": 3,
              "count_every": 3.5, "count_periods": 1, "omega_nominal": 2.0, "avg_window": 5.0,
              "planes": [{"axis": 0, "pos": 0.0}], "axes": [{"origin": [0, 0, 0], "dir": [0, 0, 1]}]},
}


@pytest.mark.parametrize("first_chunk", [37, 40])
def test_resume_bit_identical_with_lumps(tmp_path, first_chunk):
    strip = lambda r: {k: v for k, v in r.items() if k not in ("_evolver", "wall_seconds",
                                                               "steps_per_second_this_call")}
    ref = run(CFG_R)
    ck = str(tmp_path / "ck")
    part = run(CFG_R, checkpoint_dir=ck, max_steps=first_chunk)
    assert part["completed"] is False
    run(CFG_R, checkpoint_dir=ck, max_steps=23)
    res = run(CFG_R, checkpoint_dir=ck)
    assert res["completed"] is True
    assert np.array_equal(ref["_evolver"].phi, res["_evolver"].phi)
    assert len(ref["lumps"]["counts"]) >= 3
    assert json.dumps(strip(ref)) == json.dumps(strip(res))


# ---------------------------------------------------------------------------
# 10. guards
# ---------------------------------------------------------------------------

def test_lump_guards():
    absorber = {"gamma": 2.0, "width": 12.0, "power": 3}
    ok = lumps_init(PROF, [[-12, 0, 0], [12, 0, 0]], [0, 0], OM)
    g = check_lumps(ok, 88.0, absorber)
    assert all(x["Phi_at_sponge_over_Phi0"] < 1e-3 for x in g)
    with pytest.raises(ValueError):       # tail in the sponge (L = 70, lump at 12: Phi(11) ~ 8e-3 Phi0)
        check_lumps(lumps_init(PROF, [[12, 0, 0]], [0], OM), 70.0, absorber)
    with pytest.raises(ValueError):       # profile not negligible at the box face
        check_lumps(lumps_init(PROF, [[20, 0, 0]], [0], OM), 70.0, None)
    with pytest.raises(ValueError):
        lumps_init(PROF, [[0, 0, 0]], [0, 1], OM)
