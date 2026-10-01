"""WP1A check 5 and the validation of the radial solver."""

import numpy as np

from swirl.family.family import BETA_SET_B, continue_to
from swirl.family.profile import profile_curve
from swirl.family.radial import Radial
from swirl.family.run_radial import GAUSS, R_BOX, R_SPONGE, evolve

G4 = 0.30
G6 = 0.055


def _setb_profile(w=0.8854):
    p = continue_to(BETA_SET_B, w)
    return p, (lambda r: profile_curve(p, r) / np.sqrt(G4))


def test_dalembert_massless_exact():
    """m = g4 = g6 = 0: u = r phi obeys u_tt = u_rr, exact d'Alembert solution."""
    Rg, T, dt = 2.0, 40.0, 0.0125
    s = Radial(100.0, 0.05, 0.0, 0.0, 0.0)
    s.set_phi(lambda r: np.exp(-r**2 / (2 * Rg**2)))
    s.step(dt, int(round(T / dt)))
    U = lambda x: x * np.exp(-x**2 / (2 * Rg**2))
    ue = 0.5 * (U(s.r - T) + U(s.r + T))
    rel = np.linalg.norm(s.u - ue) / np.linalg.norm(ue)
    print("\nd'Alembert rel L2 at T=40: %.2e (target 1e-5)" % rel)
    assert rel < 1e-5


def test_energy_conservation_absorber_off():
    p, phi = _setb_profile()
    for T in (100.0, 500.0):
        ts = evolve(phi, G4, G6, [3 * p["half_width"]], T=T, sigma_max=0.0)
        Eh = np.array(ts["E_discrete"])
        drift = np.max(np.abs(Eh - Eh[0])) / Eh[0]
        print("\ncheck5 Set B profile w=%.4f, absorber off, T=%g: max rel energy drift %.2e "
              "(target 1e-5); nonlinear share t=0 %.3f" % (p["omega"], T, drift,
                                                            ts["nonlinear_share_t0"]))
        assert drift < 1e-5


def test_sponge_only_removes_energy():
    p, phi = _setb_profile()
    ts = evolve(phi, G4, G6, [3 * p["half_width"]], T=300.0)
    Eh = np.array(ts["E_discrete"])
    rise = np.max(np.diff(Eh)) / Eh[0]
    print("\nsponge on: largest energy rise between records %.2e, E(300)/E(0) = %.4f"
          % (rise, Eh[-1] / Eh[0]))
    assert rise < 1e-8


def test_sponge_box_doubling():
    """Absorber on at R_sponge=100/R_box=200 vs a box twice the size, T=500."""
    p, phi = _setb_profile()
    Rg = GAUSS["R"]
    gphi = lambda r: GAUSS["phi0"] * np.exp(-r**2 / (2 * Rg**2))
    for name, ph, g4, g6, rc in (("oscillon", phi, G4, G6, 3 * p["half_width"]),
                                 ("gaussian linear", gphi, 0.0, 0.0, 3 * Rg)):
        a = evolve(ph, g4, g6, [rc])
        b = evolve(ph, g4, g6, [rc], R_sponge=2 * R_SPONGE, R_box=2 * R_BOX)
        fa = np.array([e[0] for e in a["E_inside"]]) / a["E0"]
        fb = np.array([e[0] for e in b["E_inside"]]) / b["E0"]
        d = np.max(np.abs(fa - fb))
        print("\nbox doubling %s: max |frac_L - frac_2L| over T=500: %.2e" % (name, d))
        assert d < 1e-4


def test_beta43_literature_oscillon_survives():
    """External validation: Ibe 2019 (beta = 4/3, omega_ini = 0.910, profile-started) finds
    E and Q flat from t = 100 until a slow decline, lifetime ~1e4. Expect E(r<3hw) to hold."""
    beta, g4 = 4.0 / 3.0, 6.0
    p = continue_to(beta, 0.91)
    phi = lambda r: profile_curve(p, r) / np.sqrt(g4)
    ts = evolve(phi, g4, beta * g4**2, [3 * p["half_width"]], T=500.0)
    fr = np.array([e[0] for e in ts["E_inside"]]) / ts["E0"]
    print("\nbeta=4/3 w=0.91 (g4=6): E0=%.2f (Ibe ~140), E(r<3hw)/E0 at T=500 = %.4f "
          "(linear twins for this case are in gamma_map.json)" % (ts["E0"], fr[-1]))
    assert fr[-1] > 0.95


def test_legacy_gaussian_disperses():
    Rg = GAUSS["R"]
    gphi = lambda r: GAUSS["phi0"] * np.exp(-r**2 / (2 * Rg**2))
    for name, g4, g6 in (("Set B", G4, G6), ("linear twin", 0.0, 0.0)):
        ts = evolve(gphi, g4, g6, [3 * Rg], T=500.0)
        t = np.array(ts["times"])
        fr = np.array([e[0] for e in ts["E_inside"]]) / ts["E0"]
        f = {T: fr[np.argmin(np.abs(t - T))] for T in (0, 50, 100, 500)}
        print("\ncheck5 Gaussian %s: E(r<3R)/E0 at t=0,50,100,500: %.4f %.4f %.2e %.2e; "
              "nonlinear share t=0 %.2e" % (name, f[0], f[50], f[100], f[500],
                                            ts["nonlinear_share_t0"]))
        assert f[0] > 0.99
        assert f[100] < 0.05 * f[0]
