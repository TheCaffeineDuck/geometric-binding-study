"""WP1A part 1-3: the oscillon family for each beta, its features, the plots.

Run: <venv python> -m swirl.family.run_family
Writes outputs/family/family_beta*.json, family_summary.json, prior_comparison.json,
family_curves.png.
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swirl.family.family import (BETA_SET_B, at, features, literature_comparison,
                                 prior_comparison, scan, table, to_setB_units)
from swirl.family.io import INK2, OUT_DIR, SERIES, style, write_json
from swirl.family.profile import public

BETAS = [BETA_SET_B, 0.625, 0.70, 0.85, 1.0, 4.0 / 3.0]   # 4/3: literature validation row
TOL = 1e-8


def run_beta(beta):
    t0 = time.time()
    sols = scan(beta, tol=TOL)
    f = features(sols, beta, tol=TOL)
    # refine the grid around the N minimum and E = mN crossings
    extra = [f["omega_minN"] + d for d in np.arange(-0.004, 0.0041, 0.0005)]
    for wc in f["omega_E_eq_mN"]:
        extra += [wc + d for d in (-0.001, -0.0005, 0.0, 0.0005, 0.001)]
    sols += [at(sols, beta, w, TOL) for w in extra]
    sols = sorted(sols, key=lambda s: -s["omega"])
    w = np.array([s["omega"] for s in sols])
    N = np.array([s["N"] for s in sols])
    E = np.array([s["E"] for s in sols])
    # exact relation dE/domega = omega dN/domega, checked by finite differences
    dE = np.gradient(E, w)
    dN = np.gradient(N, w)
    rel = np.abs(dE - w * dN) / np.maximum(np.abs(dE), 1e-300)
    f["dEdw_eq_w_dNdw_median_rel"] = float(np.median(rel))
    rows = table(sols)
    out = dict(
        beta=beta,
        equation="Phi'' + (2/r) Phi' = (m^2-omega^2) Phi - (g4/8) Phi^3 + (g6/192) Phi^5",
        rescaled_units="m=1, g4=1, g6=beta; lengths 1/m, field m/sqrt(g4), E m/g4, N 1/g4",
        half_width_definition="radius where Phi = Phi(0)/2",
        tail_definition="Phi -> A exp(-kappa r)/r, kappa = sqrt(m^2-omega^2); A = mean of "
                        "r Phi exp(kappa r) over tail_window (1e-9 < Phi/Phi0 < 1e-3, r < 0.8 R_box); "
                        "A_rel_spread = std/mean there; kappa_fit = free log-linear fit",
        solver=dict(method="scipy solve_bvp collocation, Robin tail BC at R_box",
                    tol=TOL, continuation_from=0.95),
        features_rescaled=f,
        features_setB_units=to_setB_units(f),
        rows_rescaled=rows,
        rows_setB_units=[to_setB_units(r) for r in rows],
        wall_s=time.time() - t0,
    )
    write_json("family_beta%.4f.json" % beta, out)
    return sols, f


def plot(all_sols):
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    for k, (beta, sols, f) in enumerate(all_sols):
        c = SERIES[k]
        w = np.array([s["omega"] for s in sols])
        lab = "beta = %.4g" % beta
        for ax, key in zip(axs.flat[:5], ["P0", "half_width", "N", "E", "E_over_mN"]):
            y = np.array([s[key] for s in sols])
            ax.plot(w, y, color=c, lw=2, label=lab)
        N = np.array([s["N"] for s in sols])
        E = np.array([s["E"] for s in sols])
        axs.flat[5].plot(N, E / N, color=c, lw=2, label=lab)
        axs.flat[2].plot([f["omega_minN"]], [f["N_min"]], "o", ms=8, color=c, mec="white", mew=2)
        axs.flat[3].plot([f["omega_minE"]], [f["E_min"]], "o", ms=8, color=c, mec="white", mew=2)
    titles = ["Phi(0) (rescaled)", "half-width (Phi = Phi(0)/2), 1/m", "N (rescaled, x 1/g4)",
              "E (rescaled, x m/g4)", "E/(mN)", "E/(mN) against N (rescaled)"]
    for ax, t in zip(axs.flat, titles):
        style(ax)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("omega/m" if t != titles[5] else "N (rescaled)")
    for ax in (axs.flat[1], axs.flat[2], axs.flat[3]):
        ax.set_yscale("log")
    axs.flat[5].set_xscale("log")
    axs.flat[4].axhline(1.0, color=INK2, lw=1, ls="--")
    axs.flat[5].axhline(1.0, color=INK2, lw=1, ls="--")
    axs.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Single-frequency oscillon family (dots: minimum N and E, where dN/domega "
                 "changes sign)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR + "/family_curves.png", dpi=130)
    plt.close(fig)


def main():
    all_sols = []
    summary = []
    for beta in BETAS:
        sols, f = run_beta(beta)
        all_sols.append((beta, sols, f))
        summary.append(dict(rescaled=f, setB_units=to_setB_units(f)))
        print("beta=%.4f  minE=%.4f at w=%.6f  minN=%.4f at w=%.6f  E=mN at %s  lowest w=%.5f (end %.5f)"
              % (beta, f["E_min"], f["omega_minE"], f["N_min"], f["omega_minN"],
                 f["omega_E_eq_mN"], f["omega_lowest_computed"], f["omega_end_analytic"]))
    write_json("family_summary.json", dict(
        note="Set-B-like units: m=1, g4=0.30, g6=beta*g4^2; equals Set B only for beta=0.6111",
        features=summary))
    write_json("prior_comparison.json", prior_comparison(TOL))
    write_json("literature_comparison.json", literature_comparison(TOL))
    plot(all_sols)


if __name__ == "__main__":
    main()
