"""Diagnostics against exact answers, the high-k filter as true damping, and initial data."""

import numpy as np

from swirl.engine.evolver import Evolver
from swirl.engine.initial import build_initial
from swirl.engine.run import frequency_from_point, run


def test_zero_crossing_frequency_exact():
    t = np.arange(0, 200, 0.1)
    w = 0.9137
    f = frequency_from_point(t, np.cos(w * t + 0.3))
    assert np.max(np.abs(np.array(f["omega"]) - w)) < 1e-6


def test_uniform_mode_N_omega_Eloc():
    """Uniform field A cos(m t) in the linear theory: omega = m exactly,
    <int phi_t^2> = A^2 m^2 L^3 / 2, N = A^2 m L^3 / 2, E = A^2 m^2 L^3 / 2,
    E_loc over a sphere containing the box = E_total."""
    A, L, m = 0.1, 20.0, 1.3
    cfg = {"N": 16, "L": L, "m": m, "g4": 0.0, "g6": 0.0, "dt": 0.02, "t_end": 30.0,
           "init": {"radial": [{"r": [0.0, 50.0], "Phi": [A, A]}]},
           "r_c": [5.0, 40.0], "sample_every": 0.5, "avg_window": 2 * np.pi / m * 3}
    res = run(cfg)
    s = res["series"]
    E_ex = A ** 2 * m ** 2 * L ** 3 / 2
    assert abs(s["E_total"][0] - E_ex) < 1e-12 * E_ex
    om = np.array(res["frequency"]["omega"])
    assert np.max(np.abs(om - m)) < 1e-6
    N = np.array([x for x in s["N"] if x is not None])
    assert np.max(np.abs(N / (A ** 2 * m * L ** 3 / 2) - 1)) < 1e-3
    Eloc = np.array(s["E_loc"])
    assert np.max(np.abs(Eloc[:, 1] / np.array(s["E_total"]) - 1)) < 1e-9
    # sphere r_c = 5 vs continuum volume (grid-count discretisation, dx = 1.25)
    assert abs(Eloc[0, 0] / (E_ex / L ** 3 * 4 / 3 * np.pi * 5 ** 3) - 1) < 0.1
    assert s["nonlinear_share"][0] == 0.0


def test_energy_density_sums_to_total():
    ev = Evolver(48, 30.0, 1.0, 0.30, 0.055)
    phi, pi = build_initial(ev.coords, ev.L, {"gaussians": [
        {"center": [1, 0, -2], "amp": 2.0, "R": 2.0, "phase": 0.7, "omega": 0.9}]})
    ev.set_state(phi, pi)
    assert abs(ev.energy_density().sum() * ev.dV / ev.energy() - 1) < 1e-10


def test_kfilter_is_true_damping(record):
    """Filter on, absorber off, noisy Set B lump: E is non-increasing step by
    step, and E + removed is conserved to the RK4 error (measured ~dt^4)."""
    rng = np.random.default_rng(1)
    ev = Evolver(32, 25.0, 1.0, 0.30, 0.055, kfilter={"nu": 1.0, "power": 4})
    phi, pi = build_initial(ev.coords, ev.L, {"gaussians": [{"center": [0, 0, 0], "amp": 2.0, "R": 2.0}]})
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
    record("kfilter_damping", {"N": 32, "L": 25.0, "kfilter": {"nu": 1.0, "power": 4}, "dt": 0.05,
                               "T": 15.0, "removed_fraction": ev.absorbed / E0,
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
