"""Check 5. Boundary: the production absorbing layer (O: sponge 100 to 200; R: 200 to 400,
reviewer condition 5) against a box twice the size (twice, not 1.33 times), all four
combinations and their linear twins,
T = 1000. Target 1e-4 in frac over the whole run (Session 2 measured 3.3e-4 at beta 0.75,
F-008). Failure would look like a late rise or step in frac in the small box only (waves
returning from the sponge), most visible in the twins, which radiate everything.
"""

import numpy as np
import pytest

from swirl.radial5 import evolve as EV

STARTS = dict(O_bump=0.909149280617113, O_pinch=0.95, R_bump=0.995, R_pinch=0.99)
T = 1000.0


@pytest.mark.slow
@pytest.mark.parametrize("twin", [False, True])
@pytest.mark.parametrize("combo", list(STARTS))
def test_box_doubling(combo, twin):
    model, obj = combo.split("_")
    st = EV.profile_start(model, obj, STARTS[combo])
    g1 = EV.default_grid(model)
    g2 = dict(g1, R_sponge=2 * g1["R_sponge"], R_box=2 * g1["R_box"])
    a = EV.run(model, obj, st, T, twin=twin)
    b = EV.run(model, obj, st, T, grid=g2, twin=twin)
    fa, fb = np.array(a["series"]["frac"]), np.array(b["series"]["frac"])
    d = np.abs(fa - fb)
    i = int(np.argmax(d))
    verdict = "hit" if d[i] < 1e-4 else ("pass with note" if d[i] < 3e-4 else "MISS")
    print("\ncheck5 %s%s box %g vs %g, T=%g: max |frac_L - frac_2L| = %.2e at t = %.1f (target 1e-4: %s); "
          "frac end %.6f vs %.6f"
          % (combo, " twin" if twin else "", g1["R_box"], g2["R_box"], T, d[i], a["series"]["t"][i], verdict, fa[-1], fb[-1]))
    assert d[i] < 3e-4          # CLAUDE.md: within 3 times the target is a pass with a note
