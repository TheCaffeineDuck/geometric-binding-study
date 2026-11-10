"""Check 3. Energy.

(a) Closed box (reflecting wall at R = 200, no sponge), production amplitude starts, T = 100:
    the discrete energy E_h (O: the D2 form; R: H_h, the exact Hamiltonian) drifts by at most
    1e-8 relative at dt = 0.0125; the drift at dt = 0.025 / 0.0125 / 0.00625 shows the RK4
    scaling (ratio ~16 per halving if the error is the integrator's dt^4). Failure would
    look like a drift that does not fall with dt (an energy that misses a term), or one that
    falls as dt^1 or dt^2.
(b) The energy contains every term: dE_h/dt along the kernel's right-hand side, by complex
    step (exact to round-off), is round-off for random smooth data, sponge off. Negative
    control: the same with one term dropped from the energy (O: the g6 term; R: the mass
    term) must be O(1), so the test can fail.
"""

import numpy as np
import pytest

from swirl.radial5 import evolve as EV
from swirl.radial5.solver_o import SolverO
from swirl.radial5.solver_r import SolverR

STARTS = dict(O_bump=0.909149280617113, O_pinch=0.95, R_bump=0.995, R_pinch=0.99)


def drift(combo, dt, T=100.0):
    model, obj = combo.split("_")
    st = EV.profile_start(model, obj, STARTS[combo])
    s = EV.make_solver(model, obj, closed=True)
    s.set_f(st["f"])
    E0 = s.energy_discrete()
    worst = 0.0
    n = int(round(5.0 / dt))
    for _ in range(int(round(T / 5.0))):
        s.step(dt, n)
        worst = max(worst, abs(s.energy_discrete() - E0) / E0)
    return worst, E0


@pytest.mark.parametrize("combo", list(STARTS))
def test_closed_box_conservation(combo):
    d = {dt: drift(combo, dt) for dt in (0.025, 0.0125, 0.00625)}
    r1 = d[0.025][0] / d[0.0125][0]
    r2 = d[0.0125][0] / d[0.00625][0]
    print("\ncheck3 %s closed box T=100, E0 %.6f: max rel drift dt 0.025 %.2e, 0.0125 %.2e (target 1e-8), "
          "0.00625 %.2e; ratios per halving %.1f, %.1f (dt^4: 16)"
          % (combo, d[0.0125][1], d[0.025][0], d[0.0125][0], d[0.00625][0], r1, r2))
    assert d[0.0125][0] < 1e-8


def _random_smooth(r, rng, scale):
    out = np.zeros_like(r)
    for _ in range(6):
        out += rng.normal() * np.exp(-((r - rng.uniform(0, 15)) / rng.uniform(1.0, 4.0)) ** 2)
    return scale * out


@pytest.mark.parametrize("model", ["O", "R"])
@pytest.mark.parametrize("obj", ["bump", "pinch"])
def test_energy_has_every_term(model, obj):
    rng = np.random.default_rng(5 if obj == "bump" else 7)
    eps = 1e-30
    if model == "O":
        s = SolverO(obj, 40.0, 0.05, 0.05, 0.001)
        f = _random_smooth(s.r, rng, 3.0) * (s.r if obj == "pinch" else 1.0) / (1 + s.r) ** (0 if obj == "bump" else 1)
        s.set_f(f, _random_smooth(s.r, rng, 2.0) * (s.r / (1 + s.r) if obj == "pinch" else 1.0))
        ut, vt = s.rhs()
        dE = s.energy_discrete(s.u + 1j * eps * ut, s.v + 1j * eps * vt).imag / eps
        scale = 4 * np.pi * s.h * np.sum(np.abs(s.v * vt))
        bad = SolverO(obj, 40.0, 0.05, 0.05, 0.0)           # energy missing the g6 term
        dE_bad = bad.energy_discrete(s.u + 1j * eps * ut, s.v + 1j * eps * vt).imag / eps
    else:
        s = SolverR(obj, 40.0, 0.05)
        par = s.r / (1 + s.r) if obj == "pinch" else 1.0
        s.set_f(_random_smooth(s.r, rng, 1.0) * par, _random_smooth(s.r, rng, 0.4) * par)
        ft, pt = s.rhs()
        dE = s.energy_discrete(s.f + 1j * eps * ft, s.p + 1j * eps * pt).imag / eps
        scale = 4 * np.pi * np.sum(s.w * np.abs(s.p * pt))
        f1, p1 = s.f + 1j * eps * ft, s.p + 1j * eps * pt
        Sb = s.S(f1)
        e_bad = np.sqrt((1 + Sb) * (1 + p1 * p1)) - 1.0      # energy missing the mass term
        dE_bad = (4 * np.pi * np.sum(s.w * e_bad)).imag / eps
    rel, rel_bad = abs(dE) / scale, abs(dE_bad) / scale
    print("\ncheck3 %s %s random smooth data: |dE_h/dt| / sum|power| = %.2e (target round-off 1e-12); "
          "negative control (one term dropped) %.2e" % (model, obj, rel, rel_bad))
    assert rel < 1e-12
    assert rel_bad > 1e-3
