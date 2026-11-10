"""Start-profile tool: exact and analytic references first (standard 6).

- the l = 0 NLS ground state (Townes, R(0) = 4.3374), target 1e-4;
- the generic cycle-averaged solver reproduces swirl/family/profile.py for O bump, 1e-9;
- small-amplitude (NLS) limit of the R families at omega = 0.9999, target 1%;
- on-shell identities: Derrick 3 Lbar + int T A = 0, and dE/domega = omega dN/domega, 1e-6;
- theta quadrature converged (K = 96 against 192), 1e-10;
- the energy of the start f = F, f_t = 0 (analytic, before any PDE) equals the PDE's discrete
  energy at t = 0 on the production grid, all four combinations, 1e-6.
"""

import numpy as np
import pytest

from swirl.family.profile import solve_profile
from swirl.radial5 import evolve as EV
from swirl.radial5 import profiles as P
from swirl.radial5.solver_o import SESSION1

STARTS = dict(O_bump=0.909149280617113, O_pinch=0.95, R_bump=0.995, R_pinch=0.99)


def model(combo, K=P.K_THETA):
    m, o = combo.split("_")
    return P.Model(m, o, K=K, **(SESSION1 if m == "O" else {}))


def test_nls_ground_state():
    g = P.nls_ground("bump")
    rel = abs(g["F_peak"] - 4.3374) / 4.3374
    g1 = P.nls_ground("pinch")
    print("\nprofiles NLS ground: l=0 R(0) %.8f (4.3374, rel %.1e, target 1e-4), I2 %.4f; l=1 R'(0) %.6f, "
          "peak %.6f at x %.4f, I2 %.4f" % (g["F_peak"], rel, g["I2"], g1["shoot_param"], g1["F_peak"],
                                            g1["r_peak"], g1["I2"]))
    assert rel < 1e-4


@pytest.mark.parametrize("w", [0.909149280617113, 0.95, 0.85])
def test_generic_matches_family_profile(w):
    q = solve_profile(w, 1.0, 1.0, 4.0 / 3.0, tol=1e-10)
    g = P.solve(model("O_bump"), w)
    d = dict(F0=abs(g["F_peak"] - q["P0"] / np.sqrt(0.3)) / g["F_peak"],
             E=abs(g["E"] - q["E"] / 0.3) / g["E"], N=abs(g["N"] - q["N"] / 0.3) / g["N"],
             hw=abs(g["hw"] - q["half_width"]) / g["hw"])
    print("\nprofiles O bump omega %.6f generic vs swirl/family/profile.py: rel diffs %s (target 1e-9; "
          "hw 1e-8, profile.py reads hw by linear interpolation on a 200001-point grid)"
          % (w, {k: "%.1e" % v for k, v in d.items()}))
    assert max(d["F0"], d["E"], d["N"]) < 1e-9 and d["hw"] < 1e-8


@pytest.mark.parametrize("obj", ["bump", "pinch"])
def test_R_small_amplitude_limit(obj):
    w = 0.9999
    r = P.solve(P.Model("R", obj), w)
    pr = P.small_amplitude(obj, 0.375, w)
    ratios = {k: r[k] / pr[k] for k in ("F_peak", "hw", "N", "E")}
    print("\nprofiles R %s omega 0.9999 / NLS leading order: %s (target within 1%%)"
          % (obj, {k: "%.4f" % v for k, v in ratios.items()}))
    assert all(abs(v - 1) < 0.01 for v in ratios.values())


@pytest.mark.parametrize("combo", list(STARTS))
def test_on_shell_identities(combo):
    w = STARTS[combo]
    m = model(combo)
    r0 = P.solve(m, w)
    dw = 1e-4 * (1 - w)
    rp, rm = P.solve(m, w + dw), P.solve(m, w - dw)
    dE = (rp["E"] - rm["E"]) / (2 * dw)
    dN = (rp["N"] - rm["N"]) / (2 * dw)
    ident = abs(dE - w * dN) / abs(dE)
    print("\nprofiles %s omega %.6f: Derrick rel %.1e, dE/dw %.6g, w dN/dw %.6g, identity rel %.1e "
          "(target 1e-6 each)" % (combo, w, r0["derrick_rel"], dE, w * dN, ident))
    assert abs(r0["derrick_rel"]) < 1e-6 and ident < 1e-6


def test_theta_quadrature_converged():
    a = P.solve(model("R_bump"), 0.995)
    b = P.solve(model("R_bump", K=192), 0.995)
    rel = abs(a["E"] - b["E"]) / a["E"]
    print("\nprofiles R bump 0.995: E with K = 96 vs 192 theta nodes, rel %.1e (target 1e-10)" % rel)
    assert rel < 1e-10


@pytest.mark.parametrize("combo", list(STARTS))
def test_start_energy_t0_before_pde(combo):
    m, o = combo.split("_")
    st = EV.profile_start(m, o, STARTS[combo])
    s = EV.make_solver(m, o)
    s.set_f(st["f"])
    Eh = s.energy_discrete()
    Ea = st["info"]["E_start_t0"]
    rel = abs(Eh - Ea) / Ea
    print("\nprofiles %s: start energy analytic %.9g, PDE grid t = 0 %.9g, rel %.1e (target 1e-6)"
          % (combo, Ea, Eh, rel))
    assert rel < 1e-6
