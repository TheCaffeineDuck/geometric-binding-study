"""Radiation-rate map along the dN/domega < 0 branch (orchestrator addition to WP1A).

For beta = 0.70, 0.85 and 4/3 (literature validation): profiles from omega_c - 0.01
down in steps of 0.01 while E/(mN) >= 0.80, evolved radially (full equation,
absorber on) to T = 500 with linear twins. Per run: shed fraction, settled omega,
E(r < 3 hw)/E0 at T = 100/250/500, Gamma = -d ln E(r<3hw)/dt over the last 200
time units and over the plateau. Plus two long beta = 4/3 runs (omega = 0.91,
0.94, T = 3000) to compare with Ibe 2019 / Li 2025 (lifetime ~1e4, Gamma 1e-7..4e-4).

Run: <venv python> -m swirl.family.run_gamma_map
Writes outputs/family/gamma_map.json and gamma_map.png.
"""

import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swirl.family.family import at, n_minimum
from swirl.family.io import OUT_DIR, SERIES, style, write_json
from swirl.family.profile import profile_curve
from swirl.family.run_radial import DR, G4, OMEGA_CAVEAT, R_BOX, evolve, summarize

BETAS = [0.70, 0.85, 4.0 / 3.0]
DW = 0.01
E_OVER_MN_FLOOR = 0.80


def job(args):
    beta, omega, phi, hw, eomn, linear, T = args
    g4, g6 = (0.0, 0.0) if linear else (G4, beta * G4**2)
    ts = evolve(phi, g4, g6, [3 * hw], T=T)
    sm = summarize(ts)
    t = np.array(ts["times"])
    fr = np.array([e[0] for e in ts["E_inside"]]) / ts["E0"]
    return dict(beta=beta, omega_profile=omega, half_width=hw, E_over_mN=eomn, linear=linear,
                T=T, summary=sm, times=t[::4].tolist(), frac=fr[::4].tolist(),
                omega_t=ts["omega_t"][::2], omega=ts["omega"][::2],
                params=ts["params"], grid=ts["grid"])


def main():
    t0 = time.time()
    r = np.arange(int(round(R_BOX / DR)) + 1) * DR
    jobs, wc = [], {}
    for beta in BETAS:
        sc = n_minimum(beta)
        wc[beta] = sc["omega"]
        sols = [sc]
        w = sc["omega"] - DW
        while True:
            p = at(sols, beta, w)
            sols.append(p)
            if p["E_over_mN"] < E_OVER_MN_FLOOR:
                break
            phi = profile_curve(p, r) / np.sqrt(G4)
            for lin in (False, True):
                jobs.append((beta, p["omega"], phi, p["half_width"], p["E_over_mN"], lin, 500.0))
            w -= DW
        if abs(beta - 4.0 / 3.0) < 1e-12:
            for wl in (0.94, 0.91):
                p = at(sols, beta, wl)
                phi = profile_curve(p, r) / np.sqrt(G4)
                jobs.append((beta, p["omega"], phi, p["half_width"], p["E_over_mN"], False, 3000.0))
    with Pool(4) as pool:
        res = pool.map(job, jobs)
    out = dict(description=__doc__,
               units="Set-B-like: m=1, g4=0.30, g6=beta*g4^2 (fractions, omega, Gamma are unit-free "
                     "in 1/m)",
               omega_dNdw_sign_change={str(b): wc[b] for b in BETAS},
               definitions="see summarize() in run_radial.py and radial_beta*.json 'definitions'",
               omega_caveat=OMEGA_CAVEAT,
               runs=res, wall_s=time.time() - t0)
    write_json("gamma_map.json", out)
    for x in res:
        s = x["summary"]
        if not x["linear"]:
            print("beta=%.4f w=%.3f E/mN=%.3f T=%g shed=%.3f w_set=%.4f frac100/250/500=%.3f/%.3f/%.3f "
                  "G200=%s Gpl=%s death=%s" % (
                      x["beta"], x["omega_profile"], x["E_over_mN"], x["T"], s["shedding_fraction"],
                      s["omega_settled"], s["frac_T100"], s["frac_T250"], s["frac_T500"],
                      "%.2e" % s["gamma_last200"] if s["gamma_last200"] is not None else None,
                      "%.2e" % s["gamma_plateau"] if s["gamma_plateau"] is not None else None,
                      s["death_time"]))
    plot(res, wc)
    print("wall %.1f s" % (time.time() - t0))


def plot(res, wc):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, beta in enumerate(BETAS):
        rs = sorted([x for x in res if x["beta"] == beta and not x["linear"] and x["T"] == 500.0],
                    key=lambda x: x["omega_profile"])
        w = np.array([x["omega_profile"] for x in rs])
        g = np.array([x["summary"]["gamma_plateau"] or np.nan for x in rs])
        alive = np.array([x["summary"]["alive_at_end"] for x in rs])
        f = np.array([x["summary"]["frac_T500"] for x in rs])
        c = SERIES[k]
        lab = "beta = %.4g" % beta
        axs[0].plot(w, np.abs(g), "-", color=c, lw=2, label=lab)
        axs[0].plot(w[alive], np.abs(g[alive]), "o", color=c, ms=7, mec="white", mew=1.5)
        axs[0].plot(w[~alive], np.abs(g[~alive]), "x", color=c, ms=7, mew=2)
        axs[1].plot(w, f, "-o", color=c, lw=2, ms=6, mec="white", mew=1.5, label=lab)
    style(axs[0]); style(axs[1])
    axs[0].set_yscale("log")
    axs[0].set_xlabel("omega/m of initial profile")
    axs[0].set_ylabel("|Gamma| on plateau, 1/m  (o: alive at T=500, x: died)")
    axs[1].set_xlabel("omega/m of initial profile")
    axs[1].set_ylabel("E(r < 3 hw) / E0 at T = 500")
    axs[1].set_ylim(0, 1.02)
    axs[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Radiation rate and retained energy along the dN/domega < 0 branch (radial, T = 500)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "gamma_map.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
