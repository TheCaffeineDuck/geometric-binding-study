"""Check 4. Resolution: the headline quantities hold on a grid 1.5 times finer (h and dt
divided by 1.5), absorber on, T = 300, for all four combinations.

Targets stated in advance: E at t = 0 within 1e-6 relative; the settled rate (median over
t = 50 to 150) within 1e-4 relative; frac at t = 100 and t = 300 within 1e-4 absolute.
Failure would look like a change of the size of the effect being measured (a shift in a
death time, a rate change comparable to its drift), i.e. an under-resolved lump.
"""

import numpy as np
import pytest

from swirl.radial5 import evolve as EV

STARTS = dict(O_bump=0.909149280617113, O_pinch=0.95, R_bump=0.995, R_pinch=0.99)
T = 300.0


@pytest.mark.parametrize("combo", list(STARTS))
def test_resolution(combo):
    model, obj = combo.split("_")
    st = EV.profile_start(model, obj, STARTS[combo])
    g1 = dict(EV.GRID)
    g2 = dict(EV.GRID, h=EV.GRID["h"] / 1.5, dt=EV.GRID["dt"] / 1.5)
    a = EV.run(model, obj, st, T, grid=g1)
    b = EV.run(model, obj, st, T, grid=g2)
    dE = abs(a["E0"] - b["E0"]) / a["E0"]
    wa, wb = a["regime"]["rate_settled_50_150"], b["regime"]["rate_settled_50_150"]
    dw = abs(wa - wb) / wa
    ta, tb = np.array(a["series"]["t"]), np.array(b["series"]["t"])
    fa, fb = np.array(a["series"]["frac"]), np.array(b["series"]["frac"])
    n = min(fa.size, fb.size)
    dfr_max = float(np.max(np.abs(fa[:n] - fb[:n])))
    d100 = abs(a["frac_at"]["100"] - b["frac_at"]["100"])
    d300 = abs(fa[-1] - fb[-1])
    print("\ncheck4 %s h 0.05 -> 0.0333: E0 %.8g vs %.8g (rel %.1e, target 1e-6); rate %.7f vs %.7f "
          "(rel %.1e, target 1e-4); frac(100) %.6f vs %.6f, frac(300) %.6f vs %.6f (diff %.1e, %.1e, "
          "target 1e-4); max |dfrac| over the run %.1e; T_death %s vs %s"
          % (combo, a["E0"], b["E0"], dE, wa, wb, dw, a["frac_at"]["100"], b["frac_at"]["100"],
             fa[-1], fb[-1], d100, d300, dfr_max, a["T_death"], b["T_death"]))
    assert dE < 1e-6 and dw < 1e-4 and d100 < 1e-4 and d300 < 1e-4
