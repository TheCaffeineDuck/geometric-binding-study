"""WP3 task C: engine t=0 energy vs independent 1D quadrature of int [Phi'^2/2 + V(Phi)] 4 pi r^2 dr."""
import json, sys, numpy as np
from scipy.interpolate import CubicSpline
sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL")
from swirl.engine.run import run
D = "/Users/aaronramos/Desktop/SWIRL/outputs/step2"
for name, prof in (("run1_primary", "profile_beta1.3333_w0.9091"), ("run1b_secondary", "profile_beta0.8500_w0.7793")):
    cfg = json.load(open("%s/configs/%s.json" % (D, name)))
    cfg["threads"] = 2
    res = run(cfg, max_steps=0)
    p = json.load(open("%s/profiles/%s.json" % (D, prof)))
    r, Phi = np.array(p["r"]), np.array(p["Phi"])
    m, g4, g6 = 1.0, cfg["g4"], cfg["g6"]
    spl = CubicSpline(r, Phi, bc_type=((1, 0.0), "not-a-knot"))
    rf = np.linspace(0, r[-1], 340001)   # dr = 1e-4 on the spline, Simpson
    P, dP = spl(rf), spl(rf, 1)
    dens = (0.5 * dP**2 + 0.5 * m**2 * P**2 - g4 * P**4 / 24 + g6 * P**6 / 720) * 4 * np.pi * rf**2
    from scipy.integrate import simpson
    Eq = float(simpson(dens, x=rf))
    # also quartic/sextic parts for nonlinear share (same definition as engine: |E4|+|E6| over sum of |.| parts)
    parts = [0.5 * dP**2, 0.5 * m**2 * P**2, g4 * P**4 / 24, g6 * P**6 / 720]
    I = [float(simpson(q * 4 * np.pi * rf**2, x=rf)) for q in parts]
    share = (I[2] + I[3]) / sum(I)
    E0 = res["summary"]["E0"]; ns = res["summary"]["nonlinear_share_0"]
    print(name, "engine E0 %.8g  quad E %.8g  rel diff %.3e | nonlinear share engine %.5f quad %.5f | table-quad-of-profile-family E %.8g (time-avg, different)" % (E0, Eq, E0 / Eq - 1, ns, share, p["E"]))
