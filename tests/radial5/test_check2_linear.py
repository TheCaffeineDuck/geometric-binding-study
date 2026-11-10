"""Check 2. Linear twin: at small amplitude omega^2 = k^2 + 1 for all four combinations.

Closed box R = 40 (reflecting wall, f(R) = 0), no sponge. Box modes:
  bump  f = A sin(k r)/(k r), k = n pi / R;
  pinch f = A j1(k r), the l = 1 radial mode, k = x_n / R with tan x_n = x_n.
omega from upward zero crossings of the core coordinate (cubic interpolation), first to
last crossing over T = 300. Target 1e-6 relative. Amplitude 1e-4 (R: X ~ 1e-8; O: g4 f^2
~ 5e-10). Failure would look like a relative error at the level of the dispersion error
of a wrong stencil (1e-3) or of a wrong l (the j0 frequency instead of j1).
"""

import numpy as np
import pytest

from swirl.radial5.solver_o import SolverO
from swirl.radial5.solver_r import SolverR

RB, H, DT, T = 40.0, 0.05, 0.0125, 300.0
J1_ZEROS = [4.493409457909064, 7.725251836937707, 10.904121659428899, 14.066193912831473,
            17.220755271930768]


def j1(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    sm = np.abs(x) < 1e-3
    out[sm] = x[sm] / 3.0 - x[sm] ** 3 / 30.0
    y = x[~sm]
    out[~sm] = np.sin(y) / y ** 2 - np.cos(y) / y
    return out


def crossings_rate(t, c):
    i = np.where((c[:-1] < 0) & (c[1:] >= 0))[0]
    tc = []
    for j in i:
        sl = slice(max(j - 1, 0), j + 3)
        p = np.polyfit(t[sl] - t[j], c[sl], 3)
        roots = np.roots(p)
        roots = roots[np.isreal(roots)].real
        roots = roots[(roots >= -1e-9) & (roots <= t[j + 1] - t[j] + 1e-9)]
        tc.append(t[j] + roots[0])
    tc = np.array(tc)
    return 2 * np.pi * (tc.size - 1) / (tc[-1] - tc[0])


def run_mode(model, obj, n, A=1e-4, kg=False):
    if obj == "bump":
        k = n * np.pi / RB
        f = lambda r: A * np.sinc(k * r / np.pi)
    else:
        k = J1_ZEROS[n - 1] / RB
        f = lambda r: A * j1(k * r)
    s = SolverO(obj, RB, H, 0.0 if kg else 0.05, 0.0 if kg else 0.001) if model == "O" else \
        SolverR(obj, RB, H, kg=kg)
    s.set_f(f)
    c0 = s.core()
    core, _ = s.step(DT, int(round(T / DT)))
    t = DT * np.arange(core.size + 1)
    w = crossings_rate(t, np.concatenate([[c0], core]))
    return k, w, np.sqrt(k * k + 1)


@pytest.mark.parametrize("model", ["O", "R"])
@pytest.mark.parametrize("obj", ["bump", "pinch"])
def test_box_modes(model, obj):
    worst = 0.0
    for n in (2, 5):
        k, w, wex = run_mode(model, obj, n)
        rel = abs(w - wex) / wex
        worst = max(worst, rel)
        print("\ncheck2 %s %s mode n=%d k=%.6f: omega %.10f, sqrt(k^2+1) %.10f, rel %.2e (target 1e-6)"
              % (model, obj, n, k, w, wex, rel))
    assert worst < 1e-6


def test_R_klein_gordon_twin_matches_O_linear():
    """The two discretisations (u = r f nodes; cell-centred Hamiltonian) agree on a
    linear Gaussian pluck; the measure is the core series over T = 50."""
    out = {}
    for model in ("O", "R"):
        s = SolverO("bump", 60.0, H, 0.0, 0.0) if model == "O" else SolverR("bump", 60.0, H, kg=True)
        s.set_f(lambda r: 0.3 * np.exp(-r * r / 8.0))
        core, _ = s.step(DT, 4000)
        out[model] = core
    d = np.max(np.abs(out["O"] - out["R"])) / np.max(np.abs(out["O"]))
    print("\ncheck2 cross: R Klein-Gordon twin vs O with g4 = g6 = 0, Gaussian, core f(0,t) to T = 50: "
          "max rel diff %.2e (discretisation error; target 1e-5)" % d)
    assert d < 1e-5
