"""Scan the oscillon family across omega for one beta (rescaled units), and
locate its features: minimum E and N, the dN/domega sign change, E = mN.

Exact relation used as a check: dE/domega = omega dN/domega (E = omega N - L_bar,
with L_bar stationary in Phi), so E and N have their minimum at the same omega.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from swirl.family.profile import (omega_min_rescaled, solve_profile, next_guess, public)

SET_B = dict(m=1.0, g4=0.30, g6=0.055)
BETA_SET_B = 0.055 / 0.09


def scan(beta, w_start=0.95, w_top=0.999, dw=0.005, hw_cap=40.0, dw_min=2e-5,
         tol=1e-8):
    """Return list of solutions (dicts with '_sol'), sorted by decreasing omega."""
    first = solve_profile(w_start, 1.0, 1.0, beta, tol=tol)
    up = [first]
    # upward to w_top
    grid_up = list(np.arange(w_start + dw, w_top - 1e-12, dw)) + [0.996, 0.997, 0.998, w_top]
    grid_up = sorted(set(np.round(grid_up, 6)))
    for w in grid_up:
        if w <= w_start:
            continue
        up.append(solve_profile(w, 1.0, 1.0, beta, guess=next_guess(up[-1]), tol=tol))
    # downward with adaptive step until failure, step underflow or huge width
    down = [first]
    w_end = omega_min_rescaled(beta)
    step = dw
    w = w_start
    while True:
        w_new = w - step
        if w_new <= w_end + 1e-9:
            step *= 0.5
            if step < dw_min:
                break
            continue
        try:
            res = solve_profile(w_new, 1.0, 1.0, beta, guess=next_guess(down[-1]), tol=tol)
            if res["half_width"] > 1.5 * down[-1]["half_width"] + 1.0:
                raise RuntimeError("width jump")
        except RuntimeError:
            step *= 0.5
            if step < dw_min:
                break
            continue
        down.append(res)
        w = w_new
        if res["half_width"] > hw_cap:
            break
        # keep the width change per step between ~2% and ~10% near the thin wall
        growth = down[-1]["half_width"] / down[-2]["half_width"]
        if growth > 1.10:
            step *= 0.5
        elif growth < 1.02:
            step = min(dw, 1.5 * step)
    sols = sorted(up + down[1:], key=lambda s: -s["omega"])
    return sols


def nearest(sols, w):
    return min(sols, key=lambda s: abs(s["omega"] - w))


def at(sols, beta, w, tol=1e-8):
    """Solve at w, continuing (with adaptive sub-steps) from the nearest solution."""
    s = nearest(sols, w)
    return continue_to(beta, w, w_start=s["omega"], dw=abs(w - s["omega"]) + 1e-12,
                       tol=tol, start=s)


def continue_to(beta, w_target, w_start=0.95, dw=0.005, tol=1e-8, start=None):
    """Solve at w_target by continuation from w_start with adaptive steps."""
    s = start if start is not None else solve_profile(w_start, 1.0, 1.0, beta, tol=tol)
    w, step = w_start, dw
    sgn = 1.0 if w_target > w_start else -1.0
    while abs(w_target - w) > 1e-14:
        w_new = w + sgn * min(step, abs(w_target - w))
        try:
            s = solve_profile(w_new, 1.0, 1.0, beta, guess=next_guess(s), tol=tol)
            w = w_new
        except RuntimeError:
            step *= 0.5
            if step < 1e-5:
                raise
    return s


def features(sols, beta, tol=1e-8):
    """Refined family features in rescaled units."""
    w = np.array([s["omega"] for s in sols])
    N = np.array([s["N"] for s in sols])
    E = np.array([s["E"] for s in sols])
    r = E / N
    i = int(np.argmin(N))
    lo, hi = w[min(i + 1, len(w) - 1)], w[max(i - 1, 0)]
    fN = lambda x: at(sols, beta, x, tol)["N"]
    fE = lambda x: at(sols, beta, x, tol)["E"]
    mN = minimize_scalar(fN, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-7))
    mE = minimize_scalar(fE, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-7))
    sN = at(sols, beta, mN.x, tol)
    sE = at(sols, beta, mE.x, tol)
    out = dict(beta=beta,
               omega_minN=float(mN.x), N_min=sN["N"], E_at_minN=sN["E"],
               omega_minE=float(mE.x), E_min=sE["E"], N_at_minE=sE["N"],
               P0_at_minE=sE["P0"], hw_at_minE=sE["half_width"],
               E_over_mN_at_minE=sE["E_over_mN"],
               omega_dNdw_sign_change=float(mN.x),
               omega_end_analytic=omega_min_rescaled(beta),
               omega_lowest_computed=float(w.min()),
               hw_at_lowest=float(sols[-1]["half_width"]),
               P0_at_lowest=float(sols[-1]["P0"]),
               P0_thin_wall_limit=float(np.sqrt(18.0 / beta)))
    # E = mN crossings (E/N - 1 changes sign)
    cross = []
    d = r - 1.0
    for k in range(len(w) - 1):
        if d[k] * d[k + 1] < 0:
            g = lambda x: (lambda s: s["E"] / s["N"] - 1.0)(at(sols, beta, x, tol))
            cross.append(float(brentq(g, w[k + 1], w[k], xtol=1e-9)))
    out["omega_E_eq_mN"] = cross
    return out


def to_setB_units(d, g4=0.30, m=1.0):
    """Convert a rescaled solution/feature dict to 'Set-B-like' units
    (m = 1, g4 = 0.30, g6 = beta g4^2/m^2)."""
    fac_field = m / np.sqrt(g4)
    out = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if k in ("P0", "P0_at_minE", "P0_at_lowest", "P0_thin_wall_limit", "A"):
            out[k.replace("P0", "Phi0")] = v * fac_field if k != "A" else v / np.sqrt(g4)
        elif k in ("E", "E_min", "E_at_minN", "T", "G_int"):
            out[k] = v * m / g4
        elif k in ("N", "N_min", "N_at_minE"):
            out[k] = v / g4
        elif k in ("half_width", "hw_at_minE", "hw_at_lowest", "R_box"):
            out[k] = v / m
        elif k in ("kappa", "kappa_fit"):
            out[k] = v * m
        elif k == "m":
            out[k] = m
        elif k == "g4":
            out[k] = g4
        elif k == "g6":
            out[k] = d["beta"] * g4**2 / m**2
        elif k.startswith("omega") and isinstance(v, float):
            out[k] = v * m
        elif k.startswith("omega") and isinstance(v, list):
            out[k] = [x * m for x in v]
        else:
            out[k] = v
    out["units"] = "Set-B-like: m=%g, g4=%g, g6=beta*g4^2/m^2=%.6g" % (
        m, g4, d.get("beta", np.nan) * g4**2 / m**2)
    return out


def table(sols):
    return [public(s) for s in sols]


def n_minimum(beta, lo=0.88, hi=0.99, tol=1e-8):
    """Minimum of N(omega) (rescaled) by bounded minimization with continuation."""
    sols = [continue_to(beta, 0.5 * (lo + hi), tol=tol)]

    def fN(x):
        s = at(sols, beta, x, tol)
        sols.append(s)
        return s["N"]
    r = minimize_scalar(fN, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-7))
    s = at(sols, beta, r.x, tol)
    return s


# docs/PRIOR_ART.md (main, 7b6e71b), sections 2.4 and 3
LIT_BETA43 = dict(m=1.0, g4=6.0, g6=48.0)          # beta = 4/3, the Ibe/Mukaida/Li potential
LIT_BETA43_ROWS = [  # quantity, omega, literature value in g4 = 6 units, source
    ("N_min", None, 80.3, "Li et al. 2025 (lambda I_death = 20.07); omega_death ~ 0.97"),
    ("N", 0.94, 96.2, "Li et al. 2025, Fig. 1 legend"),
    ("N", 0.85, 525.0, "Li et al. 2025, Fig. 1 legend"),
]
LIT_G4NMIN = {0.611: 314.0, 0.625: 318.0, 0.70: 339.0, 0.85: 378.0, 1.0: 413.0, 4.0 / 3.0: 482.0}


def literature_comparison(tol=1e-8):
    rows = []
    g4 = LIT_BETA43["g4"]
    beta43 = LIT_BETA43["g6"] * LIT_BETA43["m"] ** 2 / g4**2
    smin = n_minimum(beta43, tol=tol)
    for q, w, val, src in LIT_BETA43_ROWS:
        if q == "N_min":
            meas = smin["N"] / g4
            rows.append(dict(quantity=q, omega_lit=0.97, omega_measured=smin["omega"],
                             literature=val, measured=meas, rel_diff=(meas - val) / val,
                             E_over_mN_measured=smin["E_over_mN"], source=src))
        else:
            s = solve_profile(w, guess=None, tol=tol, **LIT_BETA43)
            meas = s["N"]
            rows.append(dict(quantity=q, omega=w, literature=val, measured=meas,
                             rel_diff=(meas - val) / val, E_over_mN_measured=s["E_over_mN"],
                             source=src))
    minima = []
    for b, val in LIT_G4NMIN.items():
        bb = BETA_SET_B if abs(b - 0.611) < 1e-9 else b
        s = smin if abs(bb - beta43) < 1e-12 else n_minimum(bb, tol=tol)
        w_exp = float(np.sqrt(1 - 0.0788 / bb))
        minima.append(dict(beta=bb, g4N_min_expected=val, g4N_min_measured=s["N"],
                           rel_diff=(s["N"] - val) / val, omega_expected=w_exp,
                           omega_measured=s["omega"], omega_diff=s["omega"] - w_exp))
    return dict(
        source="docs/PRIOR_ART.md sections 2.4 and 3 (commit 7b6e71b on main)",
        units_beta43="m=1, g4=6, g6=48 (beta=4/3); N in these units (rescaled N = 6 x)",
        note_minima="expected g4*N_min and omega_min = sqrt(1-0.0788/beta) are DERIVED in the "
                    "brief from a universal-curve argument (accuracy 2-3% stated), not published "
                    "for these betas",
        solver_tol=tol, beta43_rows=rows, per_beta_minima=minima)


# Plan section 3, prior estimate (Claude Desktop, Set B units, unverified)
PRIOR_TABLE = [
    # omega, Phi0, half_width, N, E, E/(mN)
    (0.990, 3.1, 4.2, 1895.0, 1912.0, 1.009),
    (0.964, 5.6, 2.5, 1187.0, 1217.0, 1.026),
    (0.922, 7.7, 2.1, 1062.0, 1098.0, 1.034),
    (0.837, 9.7, 2.1, 1332.0, 1333.0, 1.000),
    (0.707, 10.5, 2.8, 2796.0, 2443.0, 0.874),
]


def prior_comparison(tol=1e-8):
    """Solve Set B (physical units) at the prior-table omegas and compare."""
    rows = []
    guess = None
    for w, P0, hw, N, E, r in PRIOR_TABLE:
        s = solve_profile(w, guess=guess, tol=tol, **SET_B)
        guess = next_guess(s)
        meas = dict(Phi0=s["P0"], half_width=s["half_width"], N=s["N"], E=s["E"],
                    E_over_mN=s["E_over_mN"])
        prior = dict(Phi0=P0, half_width=hw, N=N, E=E, E_over_mN=r)
        rel = {k: (meas[k] - prior[k]) / prior[k] for k in meas}
        rows.append(dict(omega=w, prior=prior, measured=meas, rel_diff=rel,
                         above_2pct=[k for k in rel if abs(rel[k]) > 0.02]))
    return dict(units="Set B: m=1, g4=0.30, g6=0.055 (beta=0.6111)",
                half_width_definition="radius where Phi = Phi(0)/2",
                prior_source="docs/SWIRL_PLAN_2026-10-01.md section 3 (unverified estimate)",
                solver_tol=tol, rows=rows)
