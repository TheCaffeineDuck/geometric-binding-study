"""Diagnostics against exact answers, the high-k filter as true damping, and initial data."""

import numpy as np
import pytest
from scipy.integrate import quad

from swirl.engine.evolver import Evolver
from swirl.engine.initial import TAIL_TOL, build_initial, radial_checks, radial_spline
from swirl.engine.run import frequency_from_point, run

# A uniform field: a Gaussian with R = 1e6 (constant to 1e-10 across the box)
def _uniform(A):
    return {"gaussians": [{"center": [0, 0, 0], "amp": A, "R": 1e6}]}


def test_zero_crossing_frequency_exact():
    t = np.arange(0, 200, 0.1)
    w = 0.9137
    f = frequency_from_point(t, np.cos(w * t + 0.3))
    assert np.max(np.abs(np.array(f["omega"]) - w)) < 1e-6


def test_uniform_mode_omega_E_Eloc():
    """Uniform field A cos(m t) in the linear theory: omega = m, E = A^2 m^2 L^3 / 2,
    E_loc over a sphere containing the box = E_total."""
    A, L, m = 0.1, 20.0, 1.3
    cfg = {"N": 16, "L": L, "m": m, "g4": 0.0, "g6": 0.0, "dt": 0.02, "t_end": 30.0,
           "init": _uniform(A), "r_c": [5.0, 40.0], "sample_every": 0.5}
    res = run(cfg)
    s = res["series"]
    E_ex = A ** 2 * m ** 2 * L ** 3 / 2
    assert abs(s["E_total"][0] - E_ex) < 1e-9 * E_ex
    assert np.max(np.abs(np.array(res["frequency"]["omega"]) - m)) < 1e-6
    Eloc = np.array(s["E_loc"])
    assert np.max(np.abs(Eloc[:, 1] / np.array(s["E_total"]) - 1)) < 1e-9
    assert abs(Eloc[0, 0] / (E_ex / L ** 3 * 4 / 3 * np.pi * 5 ** 3) - 1) < 0.1
    assert s["nonlinear_share"][0] == 0.0


@pytest.mark.parametrize("m", [1.113, 0.93])
def test_N_whole_period_average(m, record):
    """N = <int phi_t^2> / omega for A cos(m t): exact value A^2 m L^3 / 2.
    avg_window = 20 is not a whole number of periods for either m; the window
    used is cut at zero crossings. (The earlier fixed-span window gave a
    ripple of 1.19e-2 at m = 1.113 and 1.33e-2 at m = 0.93.)"""
    A, L = 0.1, 20.0
    cfg = {"N": 16, "L": L, "m": m, "g4": 0.0, "g6": 0.0, "dt": 0.05, "t_end": 80.0,
           "init": _uniform(A), "r_c": [5.0], "sample_every": 0.5, "avg_window": 20.0}
    res = run(cfg)
    N = np.array([x for x in res["series"]["N"] if x is not None])
    om = np.array([x for x in res["series"]["omega_window"] if x is not None])
    dev = float(np.max(np.abs(N / (A ** 2 * m * L ** 3 / 2) - 1)))
    record("diag_N_whole_period_m%s" % m, {"params": {k: v for k, v in res["params"].items()},
                                           "N_exact": A ** 2 * m * L ** 3 / 2,
                                           "max_rel_dev_N": dev, "n_values": int(N.size),
                                           "max_dev_omega_window": float(np.max(np.abs(om - m))),
                                           "fixed_span_method_ripple_before_fix":
                                               {"1.113": 1.19e-2, "0.93": 1.33e-2}[str(m)],
                                           "target": 1e-3})
    assert N.size > 100
    assert dev < 1e-3


def test_diagnostics_follow_center(record):
    """Off-centre Set B Gaussian; diagnostic centre on it. E_loc(r_c) against the
    radial integral of the analytic energy density, and phi(center, 0) = A to
    the trilinear interpolation bound. With the centre ignored (origin, 6.5
    away) E_loc(9) would hold well under the analytic value."""
    A, R, m, g4, g6 = 1.5, 3.0, 1.0, 0.30, 0.055
    c = [3.3, -2.1, 5.2]
    cfg = {"N": 80, "L": 40.0, "m": m, "g4": g4, "g6": g6, "dt": 0.05, "t_end": 0.1,
           "init": {"gaussians": [{"center": c, "amp": A, "R": R}]},
           "center": c, "r_c": [4.0, 9.0], "sample_every": 0.05}
    res = run(cfg)

    def e(r):
        g = np.exp(-r ** 2 / R ** 2)
        return (0.5 * A ** 2 * g * (r ** 2 / R ** 4 + m ** 2) - g4 / 24 * A ** 4 * g ** 2
                + g6 / 720 * A ** 6 * g ** 3) * 4 * np.pi * r ** 2

    exact = [quad(e, 0, rc, epsabs=1e-13)[0] for rc in (4.0, 9.0)]
    got = res["series"]["E_loc"][0]
    rel = [abs(a / b - 1) for a, b in zip(got, exact)]
    dx = 0.5
    fr = [((x + 20.0) / dx) % 1 for x in c]
    bound = sum(f * (1 - f) / 2 for f in fr) * dx ** 2 / R ** 2 * A
    p0 = res["point"]["phi"][0]
    wrong = run({**cfg, "center": [0.0, 0.0, 0.0]})  # control: centre at the origin
    rel_wrong = [abs(a / b - 1) for a, b in zip(wrong["series"]["E_loc"][0], exact)]
    record("diag_center", {"params": res["params"], "E_loc_t0": got, "E_loc_exact": exact,
                           "rel_err": rel, "rel_err_if_centre_at_origin": rel_wrong,
                           "phi_center_t0": p0, "phi_origin_t0": wrong["point"]["phi"][0],
                           "A": A, "trilinear_bound": bound})
    assert rel[1] < 1e-3
    assert rel[0] < 3e-3  # sphere staircase where the density is still large
    assert abs(p0 - A) <= 1.1 * bound
    assert min(rel_wrong) > 0.05 and abs(wrong["point"]["phi"][0] - A) > 0.5


def test_energy_density_sums_to_total():
    ev = Evolver(48, 30.0, 1.0, 0.30, 0.055)
    phi, pi = build_initial(ev.coords, ev.L, {"gaussians": [
        {"center": [1, 0, -2], "amp": 2.0, "R": 2.0, "phase": 0.7, "omega": 0.9}]})
    ev.set_state(phi, pi)
    assert abs(ev.energy_density().sum() * ev.dV / ev.energy() - 1) < 1e-10


def test_kfilter_is_true_damping(record):
    """Filter on, absorber off, noisy Set B lump: E is non-increasing step by
    step, and E + removed is conserved to the RK4 error (measured ~dt^4)."""
    seed = 1
    rng = np.random.default_rng(seed)
    params = {"N": 32, "L": 25.0, "m": 1.0, "g4": 0.30, "g6": 0.055,
              "kfilter": {"nu": 1.0, "power": 4}, "dt": 0.05, "T": 15.0,
              "init": {"gaussians": [{"center": [0, 0, 0], "amp": 2.0, "R": 2.0}],
                       "noise": {"seed": seed, "amp": 0.05, "band": "|k| < 0.6 pi/dx",
                                 "fields": ["phi", "phi_t"]}}}
    ev = Evolver(32, 25.0, 1.0, 0.30, 0.055, kfilter=params["kfilter"])
    phi, pi = build_initial(ev.coords, ev.L, {"gaussians": params["init"]["gaussians"]})
    band = np.sqrt(ev.K2) < 0.6 * np.pi / ev.dx  # band-limited noise, so RK4's own
    for f in (phi, pi):                            # loss on grid-scale modes stays small
        f += 0.05 * ev.irfft(ev.rfft(rng.standard_normal(phi.shape)) * band)
    ev.set_state(phi, pi)
    E0 = ev.energy()
    Es, tot = [E0], []
    for _ in range(300):
        ev.step(0.05)
        Es.append(ev.energy())
        tot.append(Es[-1] + ev.absorbed)
    dE = np.diff(Es)
    assert np.all(dE <= 1e-13 * E0)
    assert ev.absorbed > 1e-3 * E0
    dev = float(np.max(np.abs(np.array(tot) - E0)) / E0)
    record("kfilter_damping", {"params": params, "removed_fraction": ev.absorbed / E0,
                               "max_step_dE_over_E0": float(dE.max() / E0),
                               "max_rel_dev_E_plus_removed": dev})
    assert dev < 1e-5


def test_quadrupolar_distortion():
    """Stretch is volume preserving (int Phi^2 unchanged) and elongates along z."""
    ev = Evolver(64, 30.0, 1.0, 0.0, 0.0)
    r = np.linspace(0, 15, 3001)
    prof = {"r": r, "Phi": 3.0 * np.exp(-r ** 2 / 8.0)}
    p0, _ = build_initial(ev.coords, ev.L, {"radial": [prof]})
    p1, _ = build_initial(ev.coords, ev.L, {"radial": [{**prof, "quad_eps": 0.1}]})
    assert abs(np.sum(p1 ** 2) / np.sum(p0 ** 2) - 1) < 1e-6
    c = ev.coords
    q = 2 * c[None, None, :] ** 2 - c[:, None, None] ** 2 - c[None, :, None] ** 2
    assert abs(np.sum(q * p0 ** 2)) < 1e-8 * np.sum(p0 ** 2)
    assert np.sum(q * p1 ** 2) > 0.05 * np.sum((c[None, None, :] ** 2) * p1 ** 2)


def test_radial_guards():
    r = np.linspace(0, 15, 301)
    good = {"r": r, "Phi": 3.0 * np.exp(-r ** 2 / 8.0)}
    radial_checks(good, 30.0)
    with pytest.raises(ValueError):          # table must start at r = 0
        radial_checks({"r": r[1:], "Phi": good["Phi"][1:]}, 30.0)
    with pytest.raises(ValueError):          # not decayed at r_max
        radial_checks({"r": r, "Phi": 3.0 * np.exp(-r ** 2 / 50.0)}, 60.0)
    with pytest.raises(ValueError):          # not negligible beyond the box face
        radial_checks({"r": np.linspace(0, 30, 601),
                       "Phi": 3.0 * np.exp(-np.linspace(0, 30, 601) ** 2 / 50.0)}, 20.0)
    with pytest.raises(ValueError):          # same, because the centre is off the origin
        radial_checks({**good, "center": [5.0, 0, 0]}, 30.0)
    # clamped spline: Phi'(0) = 0 even on a coarse table of a cosh-like profile
    rc = np.linspace(0, 12, 25)
    spl = radial_spline(rc, 2.0 / np.cosh(rc))
    assert spl(0.0, 1) == 0.0
    # the run records the guard numbers
    res = run({"N": 16, "L": 30.0, "t_end": 0.1, "dt": 0.05, "init": {"radial": [good]},
               "r_c": [5.0]})
    assert res["init_checks"][0]["Phi_rmax_over_Phi0"] <= TAIL_TOL
