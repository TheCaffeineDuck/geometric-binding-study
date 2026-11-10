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

The family is followed as a curve (continuation through folds in omega and in the central
amplitude), not scanned in omega: a scan in omega stops at the first fold (the 2026-11-10
tables did; reviewer condition 1).
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


def end_class(model, obj, last, reason):
    """Why the computed curve ends."""
    if model == "R" and last["omega_F_peak"] > 0.999:
        return "ansatz wall: omega F_peak -> 1 (%.6f at the last point)" % last["omega_F_peak"]
    if "runaway" in reason:
        return "runaway value F_out reached (O, g6 > 0): the thin-wall-like end of the ansatz"
    if "not resolved" in reason:
        return "solver limit: shooting precision (thin wall)"
    return "solver limit: " + reason


def family(model, obj, max_points=300, wall_cap=None, partial_path=None):
    """The family as a curve, by continuation in (ln eps, ln x) through folds in omega and
    in x = F(0) (bump) or F'(0) (pinch). Rows in order along the curve from omega -> 1."""
    mdl = P.Model(model, obj, **(SESSION1 if model == "O" else {}))
    t0 = time.time()
    def save_partial(pts, ds):
        if partial_path:
            write_json(partial_path, dict(model=model, obj=obj, partial=True, step=ds,
                                          rows=[{k: q[k] for k in KEYS} | {"fixed": q.get("fixed", "omega")}
                                                for q in pts], wall_s=time.time() - t0))
    pts, reason = P.continue_family(mdl, max_points=max_points, wall_cap=wall_cap,
                                    on_point=save_partial, verbose=partial_path is not None)
    rows = [{k: p[k] for k in KEYS} | {"fixed": p.get("fixed", "omega")} for p in pts]
    w = np.array([x["omega"] for x in rows])
    x = np.array([x["shoot_param"] for x in rows])
    E = np.array([x["E"] for x in rows])
    N = np.array([x["N"] for x in rows])
    gw, gE, gN, gx = (np.gradient(v) for v in (w, E, N, x))
    dE = gE / gw
    dN = gN / gw
    ident = np.abs(dE - w * dN) / np.maximum(np.abs(dE), np.abs(w * dN))
    for row, a, b, c in zip(rows, dE, dN, ident):
        row.update(dE_domega=float(a), dN_domega=float(b), identity_rel=float(c))
    folds_w = [dict(index=int(i), omega=float(w[i]), x=float(x[i]), E=float(E[i]))
               for i in range(1, len(w) - 1) if (w[i] - w[i - 1]) * (w[i + 1] - w[i]) < 0]
    folds_x = [dict(index=int(i), omega=float(w[i]), x=float(x[i]), E=float(E[i]))
               for i in range(1, len(x) - 1) if (x[i] - x[i - 1]) * (x[i + 1] - x[i]) < 0]
    E_min = [dict(index=int(i), omega=float(w[i]), x=float(x[i]), E=float(E[i]))
             for i in range(1, len(E) - 1) if E[i] < E[i - 1] and E[i] < E[i + 1]]
    # signs of dE/drate, dN/drate along the curve, as runs of equal sign
    runs = []
    for i, (a, b) in enumerate(zip(np.sign(dE).astype(int), np.sign(dN).astype(int))):
        if runs and runs[-1]["sign_dE"] == a and runs[-1]["sign_dN"] == b:
            runs[-1]["to_index"] = i
            runs[-1]["omega_to"] = float(w[i])
        else:
            runs.append(dict(sign_dE=int(a), sign_dN=int(b), from_index=i, to_index=i,
                             omega_from=float(w[i]), omega_to=float(w[i])))
    ground = P.nls_ground(obj)
    small = []
    for r in rows[:3]:
        pr = P.small_amplitude(obj, lam(model), r["omega"], ground)
        small.append(dict(omega=r["omega"], F_peak_ratio=r["F_peak"] / pr["F_peak"],
                          hw_ratio=r["hw"] / pr["hw"], E_ratio=r["E"] / pr["E"], N_ratio=r["N"] / pr["N"]))
    out = dict(
        model=model, obj=obj, coupling=SESSION1 if model == "O" else None,
        lagrangian=("O: (1/2)f_t^2 - (1/2)S - f^2/2 + g4 f^4/4 - g6 f^6/6" if model == "O"
                    else "R: 1 - sqrt(1 - X) - f^2/2, X = f_t^2 - S"),
        S="f_r^2" if obj == "bump" else "f_r^2 + 2 f^2/r^2",
        method="single-frequency ansatz f = F(r) cos(omega t); cycle-averaged full Lagrangian by "
               "midpoint quadrature in theta (K = %d); shooting; continuation in (ln eps, ln x), "
               "x = %s, through folds (fixed omega or fixed x, whichever the tangent favours)"
               % (P.K_THETA, "F(0)" if obj == "bump" else "F'(0)"),
        definitions=dict(N="dLbar/domega = int 4 pi r^2 omega F^2 <2 sin^2 P'(X)> (adiabatic invariant)",
                         E="omega N - Lbar (cycle-averaged energy, on shell)",
                         E_per_cycle="2 pi E / omega",
                         hw="radius where F falls to half its peak (pinch: outer half maximum)",
                         nonlinear_share_signed="1 - <E_quad>/E, E_quad = cycle average of "
                                                "int 4 pi r^2 (f_t^2/2 + S/2 + f^2/2) in (f, f_t) "
                                                "(the same definition as the run logs)",
                         X_max_cycle="max over r and the cycle of X = omega^2 F^2 (R: must be < 1)",
                         derrick_rel="(3 Lbar + int 4 pi r^2 T A) / int 4 pi r^2 T A, zero on shell",
                         dE_domega="along the curve, (dE/ds)/(domega/ds) by central differences",
                         identity_rel="|dE/domega - omega dN/domega| / max(...) (finite differences along "
                                      "the curve; zero exactly; large only at folds)",
                         E_start_t0="energy of the evolution start f = F, f_t = 0"),
        small_amplitude_expectation=dict(
            lam=lam(model), ground_state=dict(F_peak=ground["F_peak"], r_peak=ground["r_peak"],
                                              hw=ground["hw"], I2=ground["I2"],
                                              shoot_param=ground["shoot_param"]),
            ratios=small),
        rate_range=[float(w.min()), float(w.max())], x_range=[float(x.min()), float(x.max())],
        end=dict(reason_raw=reason, why=end_class(model, obj, rows[-1], reason or ""),
                 last=dict(omega=float(w[-1]), x=float(x[-1]), E=float(E[-1]),
                           omega_F_peak=rows[-1]["omega_F_peak"])),
        folds_in_omega=folds_w, folds_in_x=folds_x, energy_minima=E_min,
        sign_runs=runs, median_identity_rel=float(np.median(ident)),
        max_abs_derrick=float(max(abs(r["derrick_rel"]) for r in rows)),
        rows=rows, wall_s=time.time() - t0)
    return out


def job(combo):
    model, obj = combo.split("_")
    out = family(model, obj)
    path = write_json(os.path.join(OUT, "family_%s.json" % combo), out)
    print(combo, "points", len(out["rows"]), "rate range", out["rate_range"], "folds w", out["folds_in_omega"],
          "folds x", out["folds_in_x"], "E minima", out["energy_minima"], "end", out["end"],
          "wall %.0f s" % out["wall_s"], path, flush=True)
    return combo


def serial(combo, wall_cap=1200.0, max_points=150):
    """One family, no Pool, partial file after every point (outputs/radial5/family_<combo>.partial.json)."""
    model, obj = combo.split("_")
    out = family(model, obj, max_points=max_points, wall_cap=wall_cap,
                 partial_path=os.path.join(OUT, "family_%s.partial.json" % combo))
    path = write_json(os.path.join(OUT, "family_%s.json" % combo), out)
    print(combo, "done", len(out["rows"]), out["end"], path, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serial":
        serial(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1200.0)
        sys.exit(0)
    combos = sys.argv[1:] or ["O_bump", "O_pinch", "R_bump", "R_pinch"]
    with Pool(min(4, len(combos))) as pool:
        pool.map(job, combos)
