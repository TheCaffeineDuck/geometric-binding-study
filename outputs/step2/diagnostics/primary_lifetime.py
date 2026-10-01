"""B: radial primary lifetime to T=12000 (omega 0.909149280617113 and 0.910). Run: NUMBA_NUM_THREADS=1 .venv/bin/python <this>"""
import json, os, sys, time
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from swirl.family.run_radial import evolve, death_time, gamma_fit, G4
from swirl.family.profile import solve_profile, profile_curve

BETA = 4.0 / 3.0
TS = [1000, 2000, 3000, 5000, 7500, 10000, 12000]
T_END = 12000.0

def prof_primary():
    d = json.load(open(os.path.join(ROOT, "outputs/step2/profiles/profile_beta1.3333_w0.9091.json")))
    r, P = np.array(d["r"]), np.array(d["Phi"])
    return d["omega"], d["half_width"], d["g6"], (lambda x: np.interp(x, r, P, right=0.0)), "table profile_beta1.3333_w0.9091.json"

def prof_solve(omega):
    res = solve_profile(omega, 1.0, 1.0, BETA, tol=1e-10)
    r = np.arange(0.0, 34.0 + 1e-9, 0.01)
    Phi = profile_curve(res, r) / np.sqrt(G4)
    cut = np.nonzero(np.abs(Phi) < 1e-9 * Phi[0])[0]
    n = (cut[0] + 1) if cut.size else r.size
    r, Phi = r[:n], Phi[:n]
    return omega, res["half_width"], BETA * G4**2, (lambda x: np.interp(x, r, Phi, right=0.0)), "solve_profile, tabulated dr=0.01 as make_profiles.py"

def first_below(t, f, thr):
    i = np.where(f < thr)[0]
    return float(t[i[0]]) if i.size else None

def analyse(ts):
    t = np.array(ts["times"]); f = np.array([e[0] for e in ts["E_inside"]]) / ts["E0"]
    ot, ow = np.array(ts["omega_t"]), np.array(ts["omega"])
    def w_at(T):
        return float(np.median(ow[(ot >= T - 25) & (ot <= T + 25)])) if ((ot >= T - 25) & (ot <= T + 25)).any() else None
    out = dict(T_reached=float(t[-1]),
               frac_at={str(T): (float(f[int(np.argmin(np.abs(t - T)))]) if T <= t[-1] + 0.5 else None) for T in TS},
               omega_at_T_median_pm25={str(T): (w_at(T) if T <= t[-1] + 0.5 else None) for T in TS},
               t_frac_below_0p90=first_below(t, f, 0.90), t_frac_below_0p50=first_below(t, f, 0.50),
               death_time_radial_def_rel_t50=death_time(ts),
               gamma_windows={"200-2000": gamma_fit(ts, 200, 2000), "2000-5000": gamma_fit(ts, 2000, 5000),
                              "5000-10000": gamma_fit(ts, 5000, 10000)})
    for k in ("t_frac_below_0p90", "t_frac_below_0p50"):
        T = out[k]
        out[k.replace("t_frac", "omega_at_t_frac")] = w_at(T) if T is not None else None
    return out, t, f

cases = {}
for name, pf in (("omega_0.909149280617113_table", prof_primary), ("omega_0.910_solved", lambda: prof_solve(0.910))):
    om, hw, g6, phi, src = pf()
    rcs = [3 * hw]
    t0 = time.time()
    ts = evolve(phi, G4, g6, rcs, T=T_END, stop_below=0.5)
    an, t, f = analyse(ts)
    print(name, "wall %.1fs" % (time.time() - t0), json.dumps(an), flush=True)
    c = dict(omega_profile=om, half_width=hw, r_c=rcs[0], g6=g6, profile_source=src, wall_s=time.time() - t0,
             E0=ts["E0"], nonlinear_share_t0=ts["nonlinear_share_t0"], grid=ts["grid"], analysis=an,
             series_every_50=dict(t=t[::100].tolist(), frac=f[::100].tolist()))
    # box check on the sponge/box: R_sponge=200, R_box=400 to T=3000 (frac comparison)
    tb = evolve(phi, G4, g6, rcs, T=3000.0, R_sponge=200.0, R_box=400.0)
    fb = np.array([e[0] for e in tb["E_inside"]]) / tb["E0"]
    n = min(len(fb), len(f))
    c["box_doubling_T3000"] = dict(R_sponge=200.0, R_box=400.0, max_abs_diff_frac=float(np.max(np.abs(fb[:n] - f[:n]))),
                                   frac_T3000=[float(f[min(n - 1, 6000)]), float(fb[min(n - 1, 6000)])])
    print(name, "box", c["box_doubling_T3000"], flush=True)
    cases[name] = c
gm = json.load(open(os.path.join(ROOT, "outputs/family/gamma_map.json")))
r910 = [r for r in gm["runs"] if abs(r["beta"] - BETA) < 1e-9 and abs(r["omega_profile"] - 0.91) < 1e-12 and r["T"] == 3000.0][0]
tt = np.array(r910["times"]); ff = np.array(r910["frac"])
gm_val = float(ff[int(np.argmin(np.abs(tt - 3000)))])
res = dict(description="B: radial primary lifetime, frac = E(r<3hw)/E0, stop when frac<0.5",
           params=dict(m=1.0, g4=G4, beta=BETA, g6=BETA * G4**2, T_end=T_END, dr=0.05, dt=0.0125, R_sponge=100.0, R_box=200.0,
                       sigma_max=1.0, initial="phi=Phi(r), phi_t=0", r_c="3 half-widths", stop_below=0.5),
           cases=cases, gamma_map_omega_0p910_frac_T3000=gm_val,
           gamma_map_omega_0p910_summary={k: r910["summary"][k] for k in ("death_time", "gamma_last200", "frac_T500")})
p = os.path.join(ROOT, "outputs/step2/diagnostics/primary_lifetime.json")
json.dump(res, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
print("gamma_map 0.910 frac T3000", gm_val)
