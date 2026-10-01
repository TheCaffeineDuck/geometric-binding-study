"""WP1A part 4: radial time evolution of family profiles, linear twins, the
legacy Gaussian control, absorber validation and energy conservation.

Run: <venv python> -m swirl.family.run_radial   (needs run_family outputs)
Writes outputs/family/radial_beta*.json, radial_checks.json, radial_beta*.png.
Units: Set-B-like (m=1, g4=0.30, g6=beta*g4^2); for beta=0.6111 this is Set B.
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from swirl.family.family import BETA_SET_B, at
from swirl.family.io import INK2, OUT_DIR, SERIES, style, write_json
from swirl.family.profile import profile_curve, solve_profile
from swirl.family.radial import Radial, core_frequency

M, G4 = 1.0, 0.30
DR, DT, T_END = 0.05, 0.0125, 500.0
R_SPONGE, R_BOX, SIGMA_MAX = 100.0, 200.0, 1.0
REC_DT = 0.5
T_SHED = 50.0
GAUSS = dict(phi0=0.5, R=2.5)   # legacy studies/01_single_reference.py: phi0 exp(-r^2/(2R^2)), phi_t = 0


def evolve(phi_init, g4, g6, rcs, T=T_END, dr=DR, dt=DT, R_sponge=R_SPONGE, R_box=R_BOX,
           sigma_max=SIGMA_MAX, stop_below=None):
    """Evolve from phi(r,0) = phi_init(r), phi_t = 0. Returns a time-series dict.

    stop_below: stop early once E(r < rcs[0]) / E0 < stop_below (after t = 50).
    """
    s = Radial(R_box, dr, M, g4, g6, sponge_start=R_sponge, sigma_max=sigma_max)
    s.set_phi(phi_init)
    nrec = int(round(REC_DT / dt))
    nchunks = int(round(T / REC_DT))
    Ein0, E0 = s.energy_inside(rcs)
    Eh0 = s.energy_discrete()
    nl0 = s.nonlinear_energy()
    phi, _ = s.fields()
    times, Ein, Etot, Eh = [0.0], [list(Ein0)], [E0], [Eh0]
    p0 = [phi[0]]
    for _ in range(nchunks):
        p0.extend(s.step(dt, nrec))
        a, b = s.energy_inside(rcs)
        times.append(s.t)
        Ein.append(list(a))
        Etot.append(b)
        Eh.append(s.energy_discrete())
        if stop_below is not None and s.t > 50 and a[0] / E0 < stop_below:
            break
    t_full = dt * np.arange(len(p0))
    p0 = np.array(p0)
    tc, wc = core_frequency(t_full, p0)
    return dict(times=times, E_inside=Ein, E_total=Etot, E_discrete=Eh, r_c=list(rcs),
                E0=E0, Eh0=Eh0, nonlinear_share_t0=nl0 / E0,
                phi_center_t=t_full[::20].tolist(), phi_center=p0[::20].tolist(),
                omega_t=tc.tolist(), omega=wc.tolist(),
                grid=dict(dr=s.dr, dt=dt, R_box=R_box, R_sponge=R_sponge,
                          sigma_max=sigma_max, sigma_profile="sigma_max((r-R_sponge)/(R_box-R_sponge))^2",
                          scheme="u=r phi, 4th-order FD, odd reflection at r=0 and r=R_box, RK4"),
                params=dict(m=M, g4=g4, g6=g6, T=T))


def window(ts, t0, t1, key="omega"):
    t = np.array(ts["omega_t"])
    w = np.array(ts[key])
    sel = (t >= t0) & (t <= t1)
    return float(np.median(w[sel])) if sel.any() else float("nan")


def frac_at(ts, t, k=0):
    i = int(np.argmin(np.abs(np.array(ts["times"]) - t)))
    return ts["E_inside"][i][k] / ts["E0"]


def gamma_fit(ts, t0, t1, k=0):
    """Gamma = -d ln E_loc / dt from a straight-line fit of ln E(r < r_c) over [t0, t1]."""
    t = np.array(ts["times"])
    E = np.array([e[k] for e in ts["E_inside"]])
    sel = (t >= t0) & (t <= t1) & (E > 0)
    if sel.sum() < 20 or t1 - t0 < 30:
        return None
    return float(-np.polyfit(t[sel], np.log(E[sel]), 1)[0])


def death_time(ts, k=0, t_ref=T_SHED):
    """First t > t_ref with E(r<r_c) below half its value at t_ref (None if never)."""
    t = np.array(ts["times"])
    E = np.array([e[k] for e in ts["E_inside"]])
    Eref = E[np.argmin(np.abs(t - t_ref))]
    i = np.where((t > t_ref) & (E < 0.5 * Eref))[0]
    return float(t[i[0]]) if i.size else None


def summarize(ts, k=0):
    T = ts["times"][-1]
    td = death_time(ts, k)
    plateau_end = T if td is None else td - 20.0
    return dict(
        nonlinear_share_t0=ts["nonlinear_share_t0"],
        omega_early=window(ts, 0, 50), omega_late=window(ts, T - 50, T),
        omega_settled=window(ts, 50, 100),
        death_time=td, alive_at_end=td is None,
        gamma_last200=gamma_fit(ts, T - 200, T, k),
        gamma_plateau=gamma_fit(ts, T_SHED, plateau_end, k),
        plateau_window=[T_SHED, plateau_end],
        frac_t0=frac_at(ts, 0, k), frac_T100=frac_at(ts, 100, k),
        frac_T250=frac_at(ts, 250, k), frac_T500=frac_at(ts, 500, k),
        shedding_fraction=1.0 - frac_at(ts, T_SHED, k) / frac_at(ts, 0, k),
        E_total_T500_over_E0=ts["E_total"][-1] / ts["E0"])


def load_family(beta):
    with open(os.path.join(OUT_DIR, "family_beta%.4f.json" % beta)) as f:
        return json.load(f)


def load_features(beta):
    return load_family(beta)["features_rescaled"]


def pick_profiles(beta, f):
    """One each side of the dN/domega sign change (+-0.04) and a deeper one at E/(mN) = 0.95."""
    wc = f["omega_dNdw_sign_change"]
    rows = load_family(beta)["rows_rescaled"]
    w = np.array([r["omega"] for r in rows])
    q = np.array([r["E_over_mN"] for r in rows]) - 0.95
    k = int(np.where((q[:-1] > 0) & (q[1:] <= 0))[0][0])   # rows sorted by decreasing omega
    sols = [solve_profile(w[k], 1.0, 1.0, beta), solve_profile(wc, 1.0, 1.0, beta)]
    g = lambda x: at(sols, beta, x)["E_over_mN"] - 0.95
    w95 = brentq(g, w[k + 1], w[k], xtol=1e-7)
    picks = [("dN/dw>0 (above sign change)", wc + 0.04),
             ("dN/dw<0 (below sign change)", wc - 0.04),
             ("dN/dw<0, deep, E/(mN)=0.95", w95)]
    out = []
    for label, w in picks:
        p = at(sols, beta, w)
        sols.append(p)
        out.append((label, p))
    return out


def run_beta(beta, gauss=False):
    g6 = beta * G4**2
    f = load_features(beta)
    runs = []
    for label, p in pick_profiles(beta, f):
        hw = p["half_width"]
        phi = (lambda pp: (lambda r: profile_curve(pp, r) / np.sqrt(G4)))(p)
        rcs = [3 * hw]
        nl = evolve(phi, G4, g6, rcs)
        lin = evolve(phi, 0.0, 0.0, rcs)
        runs.append(dict(label=label, omega_profile=p["omega"], E_over_mN=p["E_over_mN"],
                         half_width=hw, Phi0_setB=p["P0"] / np.sqrt(G4),
                         E_profile_setB=p["E"] / G4, N_profile_setB=p["N"] / G4,
                         summary=summarize(nl), summary_linear=summarize(lin),
                         series=nl, series_linear=lin))
        print("beta=%.4f %s w=%.4f: frac T500 %.4f (lin %.4f), w early %.4f late %.4f"
              % (beta, label, p["omega"], runs[-1]["summary"]["frac_T500"],
                 runs[-1]["summary_linear"]["frac_T500"], runs[-1]["summary"]["omega_early"],
                 runs[-1]["summary"]["omega_late"]))
    if gauss:
        Rg = GAUSS["R"]
        phi = lambda r: GAUSS["phi0"] * np.exp(-r**2 / (2 * Rg**2))
        hw = Rg * np.sqrt(2 * np.log(2))
        rcs = [3 * Rg, 3 * hw]
        nl = evolve(phi, G4, g6, rcs)
        lin = evolve(phi, 0.0, 0.0, rcs)
        runs.append(dict(label="legacy Gaussian control (phi0=0.5, R=2.5)", omega_profile=None,
                         E_over_mN=None, half_width=hw, Phi0_setB=GAUSS["phi0"],
                         r_c_note="r_c[0] = 3R = 7.5 (primary), r_c[1] = 3 half-widths = 8.83",
                         summary=summarize(nl), summary_linear=summarize(lin),
                         summary_3hw=summarize(nl, 1), summary_linear_3hw=summarize(lin, 1),
                         series=nl, series_linear=lin))
        print("gauss: frac(3R) t0 %.4f T100 %.4f T500 %.4f, w early %.4f"
              % (runs[-1]["summary"]["frac_t0"], runs[-1]["summary"]["frac_T100"],
                 runs[-1]["summary"]["frac_T500"], runs[-1]["summary"]["omega_early"]))
    out = dict(beta=beta, units="Set-B-like: m=1, g4=0.30, g6=beta*g4^2=%.6g" % g6,
               equation="phi_tt - lap phi + m^2 phi - (g4/6) phi^3 + (g6/120) phi^5 = 0 (full, radial)",
               initial_data="phi = Phi(r) (single-frequency profile), phi_t = 0",
               definitions=dict(
                   frac="E(r < r_c, t) / E_total(t=0), r_c = 3 half-widths of the initial profile",
                   shedding_fraction="1 - E(r<r_c, t=50) / E(r<r_c, t=0)",
                   omega="2 pi / period between consecutive upward zero crossings of phi(0,t); "
                         "early = median over t in [0,50], late = median over t in [450,500]",
                   nonlinear_share_t0="int(-(g4/24) phi^4 + (g6/720) phi^6) / E_total at t=0",
                   linear_twin="same initial data, g4 = g6 = 0",
                   omega_settled="median core omega over t in [50,100] (after initial shedding)",
                   death_time="first t > 50 with E(r<r_c) < 0.5 E(r<r_c, t=50); None if never",
                   gamma_last200="-d ln E(r<r_c)/dt, line fit over [T-200, T] (1/m); only meaningful "
                                 "if alive_at_end",
                   gamma_plateau="same fit over [50, death_time - 20] (or [50, T] if alive)"),
               runs=runs)
    write_json("radial_beta%.4f.json" % beta, out)
    plot(beta, runs)
    return runs


def plot(beta, runs):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, run in enumerate(runs):
        c = SERIES[k]
        lab = run["label"] + ("" if run["omega_profile"] is None else
                              ", omega=%.3f" % run["omega_profile"])
        for ser, ls, suffix in ((run["series"], "-", ""), (run["series_linear"], "--", " (linear twin)")):
            t = np.array(ser["times"])
            fr = np.array([e[0] for e in ser["E_inside"]]) / ser["E0"]
            axs[0].plot(t, fr, color=c, lw=2 if ls == "-" else 1.2, ls=ls, label=lab + suffix)
        axs[1].plot(run["series"]["omega_t"], run["series"]["omega"], color=c, lw=1.5, label=lab)
    style(axs[0]); style(axs[1])
    axs[0].set_xlabel("t (1/m)"); axs[0].set_ylabel("E(r < 3 half-widths) / E(0)")
    axs[0].set_ylim(0, 1.02)
    axs[1].set_xlabel("t (1/m)"); axs[1].set_ylabel("core omega / m (zero crossings)")
    axs[1].axhline(1.0, color=INK2, lw=1, ls=":")
    axs[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Radial evolution, beta = %.4g (Set-B-like units), absorber on" % beta, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "radial_beta%.4f.png" % beta), dpi=130)
    plt.close(fig)


def checks():
    """Absorber off: energy conservation. Absorber on: box-doubling and resolution checks."""
    beta = BETA_SET_B
    g6 = beta * G4**2
    f = load_features(beta)
    p = solve_profile(f["omega_dNdw_sign_change"] - 0.04, 1.0, 1.0, beta)
    phi = lambda r: profile_curve(p, r) / np.sqrt(G4)
    rcs = [3 * p["half_width"]]
    out = dict(profile_omega=p["omega"], beta=beta)
    # energy conservation, absorber off (reflecting wall at R_BOX, energy must be exact)
    for T in (100.0, 500.0):
        ts = evolve(phi, G4, g6, rcs, T=T, sigma_max=0.0)
        Eh = np.array(ts["E_discrete"])
        Ep = np.array(ts["E_total"])
        out["conservation_T%d" % T] = dict(
            max_rel_drift_discrete=float(np.max(np.abs(Eh - Eh[0])) / Eh[0]),
            max_rel_drift_physical_density=float(np.max(np.abs(Ep - Ep[0])) / Ep[0]))
    # box doubling with absorber on (oscillon and Gaussian, nonlinear and linear)
    Rg = GAUSS["R"]
    gphi = lambda r: GAUSS["phi0"] * np.exp(-r**2 / (2 * Rg**2))
    cases = [("oscillon", phi, G4, g6, rcs), ("oscillon_linear", phi, 0.0, 0.0, rcs),
             ("gaussian", gphi, G4, g6, [3 * Rg]), ("gaussian_linear", gphi, 0.0, 0.0, [3 * Rg])]
    for name, ph, g4, gg6, rc in cases:
        a = evolve(ph, g4, gg6, rc)
        b = evolve(ph, g4, gg6, rc, R_sponge=2 * R_SPONGE, R_box=2 * R_BOX)
        fa = np.array([e[0] for e in a["E_inside"]]) / a["E0"]
        fb = np.array([e[0] for e in b["E_inside"]]) / b["E0"]
        pa, pb = np.array(a["phi_center"]), np.array(b["phi_center"])
        out["box_doubling_" + name] = dict(
            max_abs_diff_frac=float(np.max(np.abs(fa - fb))),
            max_abs_diff_phi_center_over_max=float(np.max(np.abs(pa - pb)) / np.max(np.abs(pb))),
            frac_T500=[float(fa[-1]), float(fb[-1])])
    # resolution: dr and dt halved
    a = evolve(phi, G4, g6, rcs)
    b = evolve(phi, G4, g6, rcs, dr=DR / 2, dt=DT / 2)
    fa = np.array([e[0] for e in a["E_inside"]]) / a["E0"]
    fb = np.array([e[0] for e in b["E_inside"]]) / b["E0"]
    out["resolution_half_dr"] = dict(
        max_abs_diff_frac=float(np.max(np.abs(fa - fb))),
        omega_late=[window(a, 450, 500), window(b, 450, 500)],
        frac_T500=[float(fa[-1]), float(fb[-1])])
    out["settings"] = dict(dr=DR, dt=DT, R_sponge=R_SPONGE, R_box=R_BOX, sigma_max=SIGMA_MAX,
                           doubled=dict(R_sponge=2 * R_SPONGE, R_box=2 * R_BOX))
    write_json("radial_checks.json", out)
    print(json.dumps(out, indent=1))
    return out


def main():
    t0 = time.time()
    run_beta(BETA_SET_B, gauss=True)
    run_beta(0.70)
    checks()
    print("wall %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
