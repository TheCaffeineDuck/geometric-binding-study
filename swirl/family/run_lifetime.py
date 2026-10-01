"""Diagnostic for the radial result: lifetime of profiles along the family, all betas.

For each beta: profiles at the dN/domega sign change, 0.02 above it, and on the
dN/domega < 0 branch at E/(mN) = 1.0, 0.95, 0.9, 0.85, 0.8, 0.7 (where the
computed family reaches). Evolved (full radial equation, absorber on) to
T = 3000 or until E(r < 3 hw)/E0 < 0.05. Linear twins to T = 500.

lifetime = first t with E(r < 3 hw)/E0 < 0.5.
omega_before_death = median core omega over [lifetime - 30, lifetime - 10].

Run: <venv python> -m swirl.family.run_lifetime   (needs run_family outputs)
Writes outputs/family/lifetime_scan.json and lifetime_scan.png.
"""

import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import json
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from swirl.family.family import at, continue_to
from swirl.family.io import INK2, OUT_DIR, SERIES, style, write_json
from swirl.family.profile import profile_curve, solve_profile
from swirl.family.run_family import BETAS as BETAS_ALL
from swirl.family.run_radial import DR, G4, R_BOX, evolve, load_family

T_MAX = 3000.0
TARGETS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7]


def picks(beta):
    fam = load_family(beta)
    f = fam["features_rescaled"]
    rows = fam["rows_rescaled"]
    wc = f["omega_dNdw_sign_change"]
    w = np.array([r["omega"] for r in rows])
    q0 = np.array([r["E_over_mN"] for r in rows])
    out = [("sign change + 0.02", solve_profile(wc + 0.02, 1.0, 1.0, beta)),
           ("sign change (min E)", solve_profile(wc, 1.0, 1.0, beta))]
    for tgt in TARGETS:
        q = q0 - tgt
        k = np.where((q[:-1] > 0) & (q[1:] <= 0) & (w[1:] < wc))[0]
        if not k.size:
            continue
        k = int(k[0])
        sols = [_row_solution(beta, rows, k)]
        if abs(q[k + 1]) < 1e-8:
            x = w[k + 1]
        elif abs(q[k]) < 1e-8:
            x = w[k]
        else:
            x = brentq(lambda x: at(sols, beta, x)["E_over_mN"] - tgt, w[k + 1], w[k],
                       xtol=1e-7)
        out.append(("E/(mN) = %.2f" % tgt, at(sols, beta, x)))
    return f, out


def _row_solution(beta, rows, k):
    return continue_to(beta, rows[k]["omega"])


def job(args):
    beta, label, omega, phi, hw, linear = args
    g4, g6 = (0.0, 0.0) if linear else (G4, beta * G4**2)
    T = 500.0 if linear else T_MAX
    ts = evolve(phi, g4, g6, [3 * hw], T=T, stop_below=0.05)
    t = np.array(ts["times"])
    fr = np.array([e[0] for e in ts["E_inside"]]) / ts["E0"]
    below = np.where(fr < 0.5)[0]
    life = float(t[below[0]]) if below.size else None
    ot, om = np.array(ts["omega_t"]), np.array(ts["omega"])
    w_before = None
    if life is not None:
        sel = (ot >= life - 30) & (ot <= life - 10)
        w_before = float(np.median(om[sel])) if sel.any() else None
    sel = ot <= 50
    return dict(beta=beta, label=label, omega_profile=omega, half_width=hw, linear=linear,
                lifetime=life, t_end=float(t[-1]), frac_end=float(fr[-1]),
                frac_t0=float(fr[0]), frac_T50=float(fr[np.argmin(np.abs(t - 50))]),
                omega_early=float(np.median(om[sel])) if sel.any() else None,
                omega_before_death=w_before, nonlinear_share_t0=ts["nonlinear_share_t0"],
                times=t[::4].tolist(), frac=fr[::4].tolist(),
                omega_t=ot[::2].tolist(), omega=om[::2].tolist())


def main():
    t0 = time.time()
    r = np.arange(int(round(R_BOX / DR)) + 1) * DR
    jobs, feats = [], {}
    for beta in BETAS_ALL:
        f, pk = picks(beta)
        feats[beta] = f
        for label, p in pk:
            phi = profile_curve(p, r) / np.sqrt(G4)
            for lin in (False, True):
                jobs.append((beta, label + " [E/(mN)=%.3f]" % p["E_over_mN"], p["omega"], phi,
                             p["half_width"], lin))
    with Pool(4) as pool:
        res = pool.map(job, jobs)
    out = dict(
        description=__doc__,
        units="Set-B-like: m=1, g4=0.30, g6=beta*g4^2; absorber on (R_sponge=100, R_box=200)",
        features={str(b): dict(omega_dNdw_sign_change=feats[b]["omega_dNdw_sign_change"],
                               omega_E_eq_mN=feats[b]["omega_E_eq_mN"]) for b in feats},
        runs=res, wall_s=time.time() - t0)
    write_json("lifetime_scan.json", out)
    for x in res:
        if not x["linear"]:
            print("beta=%.4f w=%.4f %-40s life=%s w_early=%.4f w_before_death=%s frac_end=%.3f"
                  % (x["beta"], x["omega_profile"], x["label"], x["lifetime"], x["omega_early"],
                     x["omega_before_death"], x["frac_end"]))
    plot(res, feats)
    print("wall %.1f s" % (time.time() - t0))


def plot(res, feats):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, beta in enumerate(BETAS_ALL):
        rs = [x for x in res if x["beta"] == beta and not x["linear"]]
        w = [x["omega_profile"] for x in rs]
        life = [x["lifetime"] if x["lifetime"] is not None else T_MAX for x in rs]
        c = SERIES[k]
        axs[0].plot(w, life, "-o", color=c, lw=2, ms=6, mec="white", mew=1.5,
                    label="beta = %.4g" % beta)
        axs[0].axvline(feats[beta]["omega_dNdw_sign_change"], color=c, lw=1, ls=":")
        wb = [(x["omega_profile"], x["omega_before_death"]) for x in rs
              if x["omega_before_death"] is not None]
        if wb:
            axs[1].plot(*zip(*wb), "o", color=c, ms=7, mec="white", mew=1.5,
                        label="beta = %.4g" % beta)
            axs[1].axhline(feats[beta]["omega_dNdw_sign_change"], color=c, lw=1, ls=":")
    style(axs[0]); style(axs[1])
    axs[0].set_xlabel("omega/m of initial profile"); axs[0].set_yscale("log")
    axs[0].set_ylabel("lifetime: E(r<3hw)/E0 < 0.5 (T_max = %g if never)" % T_MAX)
    axs[1].set_xlabel("omega/m of initial profile")
    axs[1].set_ylabel("core omega just before death")
    axs[1].set_ylim(0.7, 1.1)
    axs[1].set_title("points above 1.1 (deep profiles, phi(0,t) not a single tone) are off-scale",
                     fontsize=8, color=INK2)
    axs[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Radial lifetime along the family (dotted: dN/domega sign change)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "lifetime_scan.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
