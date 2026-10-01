"""WP-A check b and the rate-assembly pieces: path integral tau = int dE/(E Gamma(E)),
window fits, floor handling, and the Schmitt-trigger frequency."""

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from swirl.family import wpa_rate as rate
from swirl.family.radial import core_frequency
from swirl.family.wpa_run import schmitt_offline, window_fits


def test_tau_exact_power_law():
    """Gamma = G0 (E/E0)^-p: tau(E_a -> E_b) = ((E_a/E0)^p - (E_b/E0)^p)/(p G0), exactly
    representable by two nodes (ln Gamma linear in ln E)."""
    G0, p, E0 = 3e-5, 4.0, 1000.0
    Ea, Eb = 990.0, 500.0
    nodes = [dict(x=np.log(E), lc=np.log(G0 * (E / E0) ** -p), lhi=0.0, llo=0.0, n=1, below_floor=0)
             for E in (400.0, 1100.0)]
    got = rate.path_integral(nodes, Ea, Eb)["tau"]
    exact = ((Ea / E0) ** p - (Eb / E0) ** p) / (p * G0)
    rel = abs(got - exact) / exact
    print("\npower law: tau %.10g exact %.10g rel %.1e (target 1e-10)" % (got, exact, rel))
    assert rel < 1e-10


def _gamma_dip(E, G0=1e-5, a=5.0, d=0.8, xd=-0.2, w=0.05, Eref=1000.0):
    x = np.log(E / Eref)
    return G0 * np.exp(-a * x) * (1.0 - d * np.exp(-0.5 * ((x - xd) / w) ** 2))


def _exact_tau(Ea, Eb):
    return quad(lambda E: 1.0 / (E * _gamma_dip(E)), Eb, Ea, epsabs=0, epsrel=1e-12, limit=500,
                points=[1000.0 * np.exp(-0.2)])[0]


def test_tau_dense_nodes_vs_quad():
    """Linear interpolation of ln Gamma: error O(h^2), so doubling the nodes cuts it 4x."""
    ex = _exact_tau(1000.0, 600.0)
    rels = []
    for n in (2001, 4001, 8001):
        xs = np.linspace(np.log(300.0), np.log(1100.0), n)
        nodes = [dict(x=x, lc=np.log(_gamma_dip(np.exp(x))), lhi=0.0, llo=0.0, n=1, below_floor=0)
                 for x in xs]
        got = rate.path_integral(nodes, 1000.0, 600.0)["tau"]
        rels.append(abs(got - ex) / ex)
    print("\ndip, 2001/4001/8001 nodes: rel %.2e %.2e %.2e, ratios %.2f %.2f (expect 4)"
          % (*rels, rels[0] / rels[1], rels[1] / rels[2]))
    assert rels[2] < 1e-6 and 3.5 < rels[0] / rels[1] < 4.5 and 3.5 < rels[1] / rels[2] < 4.5


def _synthetic_runs(starts, T=2000.0, dt=0.5):
    runs = []
    for k, E0 in enumerate(starts):
        sol = solve_ivp(lambda t, y: [-_gamma_dip(np.exp(y[0]))], (0, T), [np.log(E0)],
                        t_eval=np.arange(0, T + 1e-9, dt), rtol=1e-12, atol=1e-14, method="DOP853")
        t, E = sol.t, np.exp(sol.y[0])
        wins = window_fits(t, E, 0.0, 300.0, T)
        runs.append(dict(alpha2=float(k), windows=wins))
    return runs


def test_synthetic_pipeline_lifetime():
    """Check b: synthetic E(t) trajectories from dE/dt = -Gamma(E) E with an analytic
    Gamma that has a dip (depth 0.8, width 5% in E) -> windows -> bins -> interpolation ->
    integral, against the exact integral. Target 1e-3."""
    starts = 1000.0 * np.exp(0.02 - 0.01 * np.arange(60))
    runs = _synthetic_runs(starts)
    nodes = rate.assemble(rate.points_from_runs(runs))
    worst = 0.0
    for Ea, Eb in ((1000.0, 900.0), (1000.0, 600.0), (850.0, 650.0)):
        pi = rate.path_integral(nodes, Ea, Eb)
        ex = _exact_tau(Ea, Eb)
        rel = abs(pi["tau"] - ex) / ex
        worst = max(worst, rel)
        print("\nsynthetic %g -> %g: tau %.6g exact %.6g rel %.2e; band %.6g..%s; nodes %d gap %.4f"
              % (Ea, Eb, pi["tau"], ex, rel, pi["tau_lo"], pi["tau_hi"], pi["n_nodes_in_range"],
                 pi["largest_gap_rel"]))
        assert pi["tau_lo"] <= pi["tau"] * (1 + 1e-12)
    print("synthetic pipeline worst rel error %.2e (target 1e-3)" % worst)
    assert worst < 1e-3


def test_window_fit_exact_exponential():
    t = np.arange(0, 2000.5, 0.5)
    E = 500.0 * np.exp(-2.5e-6 * t)
    w = window_fits(t, E, 100.0, 300.0, 2000.0)
    err = max(abs(x["Gamma"] - 2.5e-6) for x in w)
    print("\nexponential: %d windows, worst |Gamma - 2.5e-6| = %.1e" % (len(w), err))
    assert len(w) == 6 and err < 1e-15 * 1e3


def test_floor_flags_lower_bound():
    """A rate below 2 sigma is floored (central = 2 sigma) and has no upper tau."""
    rng = np.random.default_rng(1)
    t = np.arange(0, 2000.5, 0.5)
    E = 500.0 * (1 + 1e-4 * rng.standard_normal(t.size))      # Gamma = 0 with noise
    runs = [dict(alpha2=0.0, windows=window_fits(t, E, 0.0, 300.0, 2000.0))]
    nodes = rate.assemble(rate.points_from_runs(runs))
    pi = rate.path_integral(nodes, 500.0, 499.0)
    print("\nfloor: nodes %d, below_floor %s, tau %.3g, tau_hi %s, lower bound %s"
          % (len(nodes), [n["below_floor"] for n in nodes], pi["tau"], pi["tau_hi"], pi["is_lower_bound"]))
    assert pi["is_lower_bound"] and pi["tau_hi"] is None


def test_schmitt_frequency_with_harmonic():
    """phi(0,t) = cos(w t) + 0.9 cos(3 w t) has extra zero crossings; the trigger counts one
    per period and returns w to the O(dt^2) crossing interpolation error (plain crossings do not)."""
    w = 0.9091
    t = np.arange(0, 400.0, 0.0125)
    v = np.cos(w * t + 0.3) + 0.9 * np.cos(3 * w * t + 0.9)
    tc = schmitt_offline(t, v)
    wt = 2 * np.pi / np.diff(tc)
    _, wplain = core_frequency(t, v)
    err = np.max(np.abs(wt - w))
    print("\nSchmitt: %d periods, max |omega - w| = %.1e; plain crossings give median %.4f"
          % (wt.size, err, np.median(wplain)))
    assert err < 1e-5          # linear interpolation of the crossing at dt = 0.0125
    assert abs(np.median(wplain) - w) > 0.1


def test_refine_finds_dip():
    runs = [dict(alpha2=0.02 + 0.002 * k, Gamma_first=g, sigma_first=1e-9, T_death=None)
            for k, g in enumerate([1e-4, 8e-5, 1e-5, 9e-5, 1e-4])]
    new = rate.refine_alpha2(runs)
    print("\nrefine around alpha2=0.024: %s" % new)
    assert len(new) == 4 and all(abs(a - 0.024) <= 0.002 * 2 / 3 + 1e-12 for a in new)
