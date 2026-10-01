"""WP3 diagnostics D1-D3 for the run 1b / run 1 prediction miss. 1D only, existing radial solver.
Run: NUMBA_NUM_THREADS=1 .venv/bin/python outputs/step2/diagnostics/radial_T1000.py
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from swirl.family.run_radial import evolve, death_time, gamma_fit, G4
from swirl.family.io import SERIES

D = os.path.join(ROOT, "outputs", "step2")
OUT = os.path.join(D, "diagnostics")
T_END = 2000.0
TS = [250, 500, 600, 700, 750, 800, 900, 1000, 1500, 2000]
CASES = {  # name: (beta, profile file, 3D result file, 3D r_c index 2)
    "1b": (0.85, "profile_beta0.8500_w0.7793.json", "run1b_secondary.json"),
    "1": (4.0 / 3.0, "profile_beta1.3333_w0.9091.json", "run1_primary.json"),
}


def load(f):
    with open(os.path.join(D, f)) as fh:
        return json.load(fh)


def radial_case(key):
    beta, pf, rf = CASES[key]
    prof = load("profiles/" + pf)
    r, P, hw = np.array(prof["r"]), np.array(prof["Phi"]), prof["half_width"]
    phi = lambda x: np.interp(x, r, P, right=0.0)   # Phi in Set-B units, tail < 1e-7 Phi0 beyond r_max
    g6 = prof["g6"]
    rcs = [3 * hw, 15.0, 20.0]
    out = {}
    for tag, (g4, g6_) in (("nl", (G4, g6)), ("lin", (0.0, 0.0))):
        t0 = time.time()
        out[tag] = evolve(phi, g4, g6_, rcs, T=T_END)
        print(key, tag, "wall %.1fs" % (time.time() - t0), flush=True)
    return dict(beta=beta, omega_profile=prof["omega"], g6=g6, half_width=hw, r_c=rcs, nl=out["nl"], lin=out["lin"])


def frac_series(ts, k=0):
    return np.array(ts["times"]), np.array([e[k] for e in ts["E_inside"]]) / ts["E0"]


def nearest(t, y, tt):
    return float(y[int(np.argmin(np.abs(t - tt)))])


def d1(key, rad):
    d3 = load("results/" + CASES[key][2])
    rc3 = sorted(d3["params"]["r_c"])
    ci = 2
    t3 = np.array(d3["series"]["t"])
    f3 = np.array(d3["series"]["E_loc"])[:, ci] / d3["summary"]["E0"] if np.array(d3["series"]["E_loc"]).shape[1] == 5 else None
    tr, fr = frac_series(rad["nl"])
    fi = np.interp(t3, tr, fr)
    diff = f3 - fi
    m1 = t3 <= 500
    m2 = (t3 > 500) & (t3 <= 1000)
    tw = np.array(d3["frequency"]["t_mid"]); w3 = np.array(d3["frequency"]["omega"])
    rows = []
    trl, frl = frac_series(rad["lin"])
    for T in TS:
        w_r = nearest(np.array(rad["nl"]["omega_t"]), np.array(rad["nl"]["omega"]), T)
        rows.append(dict(t=T, frac_radial=nearest(tr, fr, T), frac_radial_linear=nearest(trl, frl, T),
                         frac_3D=nearest(t3, f3, T) if T <= 1000 else None,
                         omega_radial=w_r, omega_3D=nearest(tw, w3, T) if T <= tw[-1] + 10 else None))
    return dict(rows=rows,
                max_abs_diff_t_le_500=float(np.abs(diff[m1]).max()), t_at_max_le_500=float(t3[m1][np.abs(diff[m1]).argmax()]),
                max_abs_diff_500_1000=float(np.abs(diff[m2]).max()), t_at_max_500_1000=float(t3[m2][np.abs(diff[m2]).argmax()]),
                death_time_radial=death_time(rad["nl"]), death_time_3D_same_definition=death_time(
                    dict(times=t3.tolist(), E_inside=np.array(d3["series"]["E_loc"])[:, [ci]].tolist(), E0=d3["summary"]["E0"])),
                death_time_radial_linear=death_time(rad["lin"]),
                gamma_radial_200_500=gamma_fit(rad["nl"], 200, 500), gamma_radial_500_1000=gamma_fit(rad["nl"], 500, 1000),
                gamma_radial_1000_2000=gamma_fit(rad["nl"], 1000, 2000),
                r_c_3D=rc3[ci], t3=t3, f3=f3)


def d3(key="1b"):
    d = load("results/" + CASES[key][2])
    s = d["series"]
    t = np.array(s["t"]); E0 = d["summary"]["E0"]
    EL = np.array(s["E_loc"]); rc = sorted(d["params"]["r_c"])
    et = np.array(s["E_total"]); ab = np.array(s["absorbed"]); mp = np.abs(np.array(s["max_abs_phi"]))
    env = np.array([mp[max(0, i - 20):i + 1].max() for i in range(len(t))])   # trailing 20-unit window (samples at dt=1)
    f = EL[:, 2] / E0
    # windowed Gamma = -d ln f/dt, 50-unit linear fit
    def gam(t0, t1):
        sel = (t >= t0) & (t <= t1)
        return -np.polyfit(t[sel], np.log(f[sel]), 1)[0]
    gref = gam(200, 500)
    cen, g = [], []
    for c in np.arange(225, 976, 5):
        cen.append(c); g.append(gam(c - 25, c + 25))
    cen, g = np.array(cen), np.array(g)
    idx = np.where(g > 3 * gref)[0]
    t_decl = float(cen[idx[0]]) if idx.size else None
    rows = []
    for T in [200, 300, 400, 500, 600, 700, 750, 800, 900, 1000]:
        i = int(np.argmin(np.abs(t - T)))
        rows.append(dict(t=T, **{"Eloc_r%.3f/E0" % rc[2]: float(EL[i, 2] / E0), "Eloc_r15/E0": float(EL[i, 3] / E0),
                                 "Eloc_r20/E0": float(EL[i, 4] / E0), "E_total/E0": float(et[i] / E0),
                                 "absorbed/E0": float(ab[i] / E0), "E_plus_absorbed/E0": float((et[i] + ab[i]) / E0),
                                 "(E15-E10.9)/E0": float((EL[i, 3] - EL[i, 2]) / E0),
                                 "(E_total-E20)/E0": float((et[i] - EL[i, 4]) / E0),
                                 "max_abs_phi_env20": float(env[i])}))
    return dict(rows=rows, Gamma_ref_200_500=float(gref), decline_start_t=t_decl, window=50,
                gamma_windowed_at=dict(zip([int(c) for c in cen[::20]], [float(x) for x in g[::20]])),
                t=t, EL=EL, E0=E0, et=et, ab=ab, env=env, rc=rc)


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items() if not isinstance(v, np.ndarray)}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def eloc_plot():
    names = [("run1_primary.json", "run 1 primary", 0), ("run3_primary_quad.json", "run 3 quad", 1),
             ("run4_primary_res.json", "run 4 res (T=300)", 2), ("run1b_secondary.json", "run 1b secondary", 3),
             ("run2_primary_twin.json", "run 2 primary twin", 4), ("run2b_secondary_twin.json", "run 2b secondary twin", 5),
             ("run5_primary_generic.json", "run 5 generic (r_c 10.0321)", 6)]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    for f, lab, k in names:
        d = load("results/" + f); t = np.array(d["series"]["t"])
        y = np.array(d["series"]["E_loc"])[:, 2] / d["summary"]["E0"]
        for ax in axs:
            ax.plot(t, y, color=SERIES[k % len(SERIES)], lw=1.6, ls="--" if "twin" in f else ("-." if "run5" in f else "-"), label=lab)
    axs[0].set_ylim(0, 1.02); axs[1].set_yscale("log"); axs[1].set_ylim(1e-4, 1.5)
    axs[0].axhline(0.90, color="k", lw=0.8, ls=":")
    axs[0].text(axs[0].get_xlim()[1], 0.90, "Go line (0.90) ", ha="right", va="bottom", fontsize=8)
    for ax in axs:
        ax.axhline(0.10, color="gray", lw=0.8, ls=":")
        ax.text(ax.get_xlim()[1], 0.10, "control line (0.10) ", ha="right", va="bottom", fontsize=8, color="gray")
        ax.set_xlabel("t (1/m)"); ax.set_ylabel("E(r<3 hw)/E0"); ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(frameon=False, fontsize=7, loc="center right"); axs[0].set_title("linear scale"); axs[1].set_title("log scale (twins decay)")
    axs[0].text(0.5, 0.60, "runs 1, 3, 4, 5 overlap at ~0.995\n(indistinguishable on this scale)", transform=axs[0].transAxes,
                ha="center", fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="gray", lw=0.5))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "eloc_curves.png"), dpi=130); plt.close(fig)


def plots(rads, d1r, d3r):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, lab in zip(axs, ["1b", "1"], ["run 1b: beta=0.85, omega=0.7793", "run 1: beta=4/3, omega=0.9091"]):
        tr, fr = frac_series(rads[key]["nl"])
        trl, frl = frac_series(rads[key]["lin"])
        ax.plot(d1r[key]["t3"], d1r[key]["f3"], color=SERIES[0], lw=2, label="3D, r_c=%.3f" % d1r[key]["r_c_3D"])
        ax.plot(tr, fr, color=SERIES[1], lw=1.5, ls="--", label="radial (to T=2000)")
        ax.plot(trl, frl, color=SERIES[2], lw=1, ls=":", label="radial linear twin")
        ax.set_xlabel("t (1/m)"); ax.set_ylabel("E(r<3 hw)/E0"); ax.set_title(lab, fontsize=10)
        ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=8); ax.axvline(500, color="gray", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "run1b_vs_radial.png"), dpi=130); plt.close(fig)

    eloc_plot()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eloc":
        eloc_plot(); sys.exit(0)
    rads = {k: radial_case(k) for k in CASES}
    d1r = {k: d1(k, rads[k]) for k in CASES}
    d3r = d3("1b")
    plots(rads, d1r, d3r)
    res = dict(description="WP3 diagnostics D1 (radial 1b), D2 (radial primary), D3 (3D run1b series)",
               params=dict(m=1.0, g4=G4, T_end=T_END, dr=0.05, dt=0.0125, R_sponge=100, R_box=200, sigma_max=1.0,
                           initial="phi=Phi(r) (profile file, interpolated), phi_t=0", r_c="3 half-widths, 15, 20",
                           cases={k: dict(beta=v[0], profile=v[1]) for k, v in CASES.items()}),
               D1_run1b=jsonable(d1r["1b"]), D2_primary=jsonable(d1r["1"]), D3=jsonable(d3r),
               radial_summary={k: dict(omega_profile=rads[k]["omega_profile"], half_width=rads[k]["half_width"],
                                       E0=rads[k]["nl"]["E0"], E0_lin=rads[k]["lin"]["E0"], nonlinear_share_t0=rads[k]["nl"]["nonlinear_share_t0"])
                               for k in CASES},
               radial_series={k: dict(times=rads[k]["nl"]["times"], E_inside=rads[k]["nl"]["E_inside"], E_total=rads[k]["nl"]["E_total"],
                                      omega_t=rads[k]["nl"]["omega_t"], omega=rads[k]["nl"]["omega"],
                                      lin_times=rads[k]["lin"]["times"], lin_E_inside=rads[k]["lin"]["E_inside"]) for k in CASES})
    p = os.path.join(OUT, "radial_T1000.json")
    with open(p + ".tmp", "w") as fh:
        json.dump(res, fh)
    os.replace(p + ".tmp", p)
    print(json.dumps(dict(D1=jsonable(d1r["1b"])["rows"], D2=jsonable(d1r["1"])["rows"]), indent=0))
