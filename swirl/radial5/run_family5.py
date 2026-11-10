"""Families of single-frequency profiles for the four combinations (O, R) x (bump, pinch).

Run:  <venv python> -m swirl.radial5.run_family5 [combo ...]      combo like O_bump
Writes outputs/radial5/family_<model>_<obj>.json (atomic, full parameters).

Small-amplitude expectation, stated before solving (standard 6), leading order in
eps = sqrt(1 - omega^2): F = (eps / sqrt(lam)) R_l(eps r) with R_l the l = 0 or l = 1
ground state of R'' + 2R'/x - s2 R/x^2 = R - R^3 (R_0(0) = 4.3374; R_1'(0) = 3.7782),
lam = 3 g4 / 4 = 0.0375 for O and 3/8 for R (from <X^2>/8 with <sin^4> = 3/8). So
F_peak ~ eps, hw ~ 1/eps, E ~ N ~ I_l / (2 lam eps) diverge as omega -> 1, dE/domega > 0
and dN/domega > 0 near omega = 1 (no energy minimum there). The pinch has the larger
I_l (110.13 against 18.90): at the same rate its energy is higher, by 5.83 at leading
order. In R the ansatz needs omega F < 1, which the leading order puts at
eps = sqrt(lam) / 4.3374 (bump, omega ~ 0.990) and sqrt(lam) / 2.7098 (pinch, omega ~ 0.974).
"""

import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from swirl.radial5 import profiles as P
from swirl.radial5.evolve import write_json
from swirl.radial5.solver_o import SESSION1

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "radial5")
KEYS = ("omega", "F_peak", "r_peak", "hw", "E", "N", "E_per_cycle", "nonlinear_share_signed",
        "X_max_cycle", "omega_F_peak", "derrick_rel", "E_start_t0", "shoot_param", "kappa")


def lam(model):
    return 0.75 * SESSION1["g4"] if model == "O" else 0.375


def omega_grid(model):
    if model == "O":
        return list(np.round(np.arange(0.995, 0.5, -0.005), 6))
    eps = np.geomspace(0.01, 0.4, 40)
    return list(np.round(np.sqrt(1 - eps ** 2), 8))


def family(model, obj):
    mdl = P.Model(model, obj, **(SESSION1 if model == "O" else {}))
    rows, fails = [], []
    t0 = time.time()
    last_ok = None
    for w in omega_grid(model):
        try:
            r = P.solve(mdl, w)
            rows.append({k: r[k] for k in KEYS})
            last_ok = w
        except RuntimeError as e:
            fails.append(dict(omega=float(w), reason=str(e)))
            break
    # locate the end of the family between the last success and the first failure
    end = None
    if fails and last_ok is not None:
        lo, hi = fails[0]["omega"], last_ok
        while hi - lo > 1e-5:
            mid = 0.5 * (lo + hi)
            try:
                r = P.solve(mdl, mid)
                rows.append({k: r[k] for k in KEYS})
                hi = mid
            except RuntimeError as e:
                lo = mid
                last_reason = str(e)
        end = dict(omega_last_solved=hi, omega_first_failed=lo, reason=last_reason if lo != fails[0]["omega"]
                   else fails[0]["reason"])
    rows.sort(key=lambda x: -x["omega"])
    w = np.array([x["omega"] for x in rows])
    E = np.array([x["E"] for x in rows])
    N = np.array([x["N"] for x in rows])
    dE = np.gradient(E, w)
    dN = np.gradient(N, w)
    ident = np.abs(dE - w * dN) / np.maximum(np.abs(dE), 1e-300)
    for x, a, b, c in zip(rows, dE, dN, ident):
        x.update(dE_domega=float(a), dN_domega=float(b), identity_rel=float(c))
    i_min = int(np.argmin(E))
    interior_min = 0 < i_min < len(E) - 1
    g = {o: P.nls_ground(o) for o in ("bump",)} if obj == "bump" else {"pinch": P.nls_ground("pinch")}
    ground = g[obj]
    small = []
    for x in rows[:3]:
        pr = P.small_amplitude(obj, lam(model), x["omega"], ground)
        small.append(dict(omega=x["omega"], F_peak_ratio=x["F_peak"] / pr["F_peak"],
                          hw_ratio=x["hw"] / pr["hw"], E_ratio=x["E"] / pr["E"], N_ratio=x["N"] / pr["N"]))
    out = dict(
        model=model, obj=obj, coupling=SESSION1 if model == "O" else None,
        lagrangian=("O: (1/2)f_t^2 - (1/2)S - f^2/2 + g4 f^4/4 - g6 f^6/6" if model == "O"
                    else "R: 1 - sqrt(1 - X) - f^2/2, X = f_t^2 - S"),
        S="f_r^2" if obj == "bump" else "f_r^2 + 2 f^2/r^2",
        method="single-frequency ansatz f = F(r) cos(omega t); cycle-averaged full Lagrangian by "
               "midpoint quadrature in theta (K = %d); bisection shooting; exact linear tail" % P.K_THETA,
        definitions=dict(N="dLbar/domega = int 4 pi r^2 omega F^2 <2 sin^2 P'(X)> (adiabatic invariant)",
                         E="omega N - Lbar (cycle-averaged energy, on shell)",
                         E_per_cycle="2 pi E / omega",
                         hw="radius where F falls to half its peak (pinch: outer half maximum)",
                         nonlinear_share_signed="(E - E_quadratic) / E, E_quadratic the cycle-averaged "
                                                "Klein-Gordon energy of the same profile",
                         X_max_cycle="max over r and the cycle of X = omega^2 F^2 (R: must be < 1)",
                         derrick_rel="(3 Lbar + int 4 pi r^2 T A) / int 4 pi r^2 T A, zero on shell",
                         identity_rel="|dE/domega - omega dN/domega| / |dE/domega| (finite differences "
                                      "on the grid; zero exactly)",
                         E_start_t0="energy of the evolution start f = F, f_t = 0"),
        small_amplitude_expectation=dict(
            lam=lam(model), ground_state=dict(F_peak=ground["F_peak"], r_peak=ground["r_peak"],
                                              hw=ground["hw"], I2=ground["I2"],
                                              shoot_param=ground["shoot_param"]),
            statement="F_peak = eps R(0 or peak)/sqrt(lam), hw = hw_R/eps, E = N = I2/(2 lam eps); "
                      "ratios measured/expected at the three points nearest omega = 1 below",
            ratios=small),
        rate_range=[float(w.min()), float(w.max())], end=end, first_failure=fails[0] if fails else None,
        energy_min=dict(index=i_min, omega=float(w[i_min]), E=float(E[i_min]), interior=bool(interior_min)),
        sign_dE_domega=sorted(set(np.sign(dE).astype(int).tolist())),
        sign_dN_domega=sorted(set(np.sign(dN).astype(int).tolist())),
        max_identity_rel=float(np.max(ident)), max_abs_derrick=float(max(abs(x["derrick_rel"]) for x in rows)),
        rows=rows, wall_s=time.time() - t0)
    return out


def job(combo):
    model, obj = combo.split("_")
    out = family(model, obj)
    path = write_json(os.path.join(OUT, "family_%s.json" % combo), out)
    print(combo, "points", len(out["rows"]), "rate range", out["rate_range"], "E min", out["energy_min"],
          "end", out["end"], "wall %.0f s" % out["wall_s"], path, flush=True)
    return combo


if __name__ == "__main__":
    combos = sys.argv[1:] or ["O_bump", "O_pinch", "R_bump", "R_pinch"]
    with Pool(min(4, len(combos))) as pool:
        pool.map(job, combos)
