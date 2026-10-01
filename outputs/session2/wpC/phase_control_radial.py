"""WP-C item 5: single-lump phase control in the radial solver.

Initial data for phase theta (SESSION_02_PREDICTIONS.md, WP-C block):
    phi = Phi(r) cos(theta),  phi_t = -omega Phi(r) sin(theta)
at beta = 4/3, omega = 0.909149 (profile_beta1.3333_w0.9091.json), Set-B-like units.
Same grid as Session 1's radial runs: dr = 0.05, dt = 0.0125, R_box = 200,
sponge from r = 100, sigma_max = 1 (swirl/family/run_radial.py constants).

This file adds the phi_t option without editing swirl/family/ (task rule): it
drives swirl.family.radial.Radial directly, mirroring run_radial.evolve.

Pass criterion (fixed in SESSION_02_PREDICTIONS.md before this run): for theta =
45, 90, 135, 180 against theta = 0, |frac_theta(T) - frac_0(T)| <= 0.01 at
T = 300, 500, 1000; measured omega within 0.005; no death.

Run: NUMBA_NUM_THREADS=1 <venv python> outputs/session2/wpC/phase_control_radial.py
Writes outputs/session2/wpC/phase_control_radial.json (atomic).
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from swirl.family.radial import Radial, core_frequency  # noqa: E402
from swirl.family.run_radial import DR, DT, R_SPONGE, R_BOX, SIGMA_MAX, REC_DT, G4, death_time  # noqa: E402

PROFILE = os.path.join(ROOT, "outputs/step2/profiles/profile_beta1.3333_w0.9091.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase_control_radial.json")
THETAS_DEG = [0.0, 45.0, 90.0, 135.0, 180.0]
T_END = 1000.0
T_CHECK = [300.0, 500.0, 1000.0]
TOL_FRAC, TOL_OMEGA = 0.01, 0.005
OMEGA_WIN = 50.0  # omega at T = median over core periods with midpoints in [T - 50, T]


def evolve_phase(prof, theta, T=T_END):
    r_tab, P_tab = np.array(prof["r"]), np.array(prof["Phi"])
    om, hw = prof["omega"], prof["half_width"]
    Phi = lambda x: np.interp(x, r_tab, P_tab, right=0.0)
    s = Radial(R_BOX, DR, prof["m"], prof["g4"], prof["g6"], sponge_start=R_SPONGE, sigma_max=SIGMA_MAX)
    s.set_phi(lambda x: np.cos(theta) * Phi(x), lambda x: -om * np.sin(theta) * Phi(x))
    rcs = [3.0 * hw]
    nrec = int(round(REC_DT / DT))
    Ein0, E0 = s.energy_inside(rcs)
    times, Ein, Etot, Eh = [0.0], [float(Ein0[0])], [float(E0)], [float(s.energy_discrete())]
    phi, _ = s.fields()
    p0 = [phi[0]]
    for _ in range(int(round(T / REC_DT))):
        p0.extend(s.step(DT, nrec))
        a, b = s.energy_inside(rcs)
        times.append(s.t)
        Ein.append(float(a[0]))
        Etot.append(float(b))
        Eh.append(float(s.energy_discrete()))
    t_full = DT * np.arange(len(p0))
    tc, wc = core_frequency(t_full, np.array(p0))
    return dict(times=times, E_inside=Ein, E_total=Etot, E_discrete=Eh, E0=float(E0), r_c=rcs[0],
                omega_t=tc.tolist(), omega=wc.tolist(), phi_center_t=t_full[::40].tolist(),
                phi_center=np.array(p0)[::40].tolist())


def analyse(ts, E0_ref):
    t = np.array(ts["times"])
    Ein = np.array(ts["E_inside"])
    ot, ow = np.array(ts["omega_t"]), np.array(ts["omega"])
    out = {"E0": ts["E0"], "frac_at": {}, "frac_vs_theta0_E0_at": {}, "omega_at": {}}
    for T in T_CHECK:
        i = int(np.argmin(np.abs(t - T)))
        out["frac_at"][str(int(T))] = float(Ein[i] / ts["E0"])
        out["frac_vs_theta0_E0_at"][str(int(T))] = float(Ein[i] / E0_ref)
        sel = (ot >= T - OMEGA_WIN) & (ot <= T)
        out["omega_at"][str(int(T))] = float(np.median(ow[sel])) if sel.any() else None
    sel = (ot >= 50) & (ot <= 100)
    out["omega_settled_50_100"] = float(np.median(ow[sel]))
    sel = ot <= 20
    out["omega_first_20"] = float(np.median(ow[sel])) if sel.any() else None
    out["frac_min_t_gt_50"] = float(np.min(Ein[t > 50]) / ts["E0"])
    out["death_time"] = death_time({"times": ts["times"], "E_inside": [[e] for e in ts["E_inside"]]}, 0)
    out["shedding_fraction_t50"] = float(1 - Ein[int(np.argmin(np.abs(t - 50)))] / Ein[0])
    Eh = np.array(ts["E_discrete"])
    out["E_discrete_t0"] = float(Eh[0])
    return out


def main():
    prof = json.load(open(PROFILE))
    t0 = time.time()
    runs, series = {}, {}
    for th in THETAS_DEG:
        t1 = time.time()
        ts = evolve_phase(prof, np.deg2rad(th))
        series[str(th)] = ts
        print("theta %5.1f done in %.1fs  E0 %.6f" % (th, time.time() - t1, ts["E0"]), flush=True)
    E0_ref = series["0.0"]["E0"]
    for th in THETAS_DEG:
        runs[str(th)] = analyse(series[str(th)], E0_ref)
    ref = runs["0.0"]
    verdict, worst = {}, {"dfrac": 0.0, "domega": 0.0}
    for th in THETAS_DEG[1:]:
        r = runs[str(th)]
        df = {T: abs(r["frac_at"][T] - ref["frac_at"][T]) for T in r["frac_at"]}
        dw = {T: abs(r["omega_at"][T] - ref["omega_at"][T]) for T in r["omega_at"]}
        # full record: max |frac difference| over all samples (same time grid)
        fa = np.array(series[str(th)]["E_inside"]) / series[str(th)]["E0"]
        fb = np.array(series["0.0"]["E_inside"]) / series["0.0"]["E0"]
        dfull = float(np.max(np.abs(fa - fb)))
        ok = (max(df.values()) <= TOL_FRAC and max(dw.values()) <= TOL_OMEGA and r["death_time"] is None)
        verdict[str(th)] = {"abs_dfrac_at": df, "abs_domega_at": dw, "max_abs_dfrac_whole_record": dfull,
                            "t_of_max_dfrac_whole_record": float(series["0.0"]["times"][int(np.argmax(np.abs(fa - fb)))]),
                            "dead": r["death_time"] is not None, "pass": bool(ok)}
        worst["dfrac"] = max(worst["dfrac"], max(df.values()))
        worst["domega"] = max(worst["domega"], max(dw.values()))
    res = {
        "description": "WP-C item 5: radial single-lump phase control (phi = Phi cos theta, phi_t = -omega Phi sin theta)",
        "params": {"profile_file": os.path.relpath(PROFILE, ROOT), "beta": prof["beta"], "omega": prof["omega"],
                   "m": prof["m"], "g4": prof["g4"], "g6": prof["g6"], "half_width": prof["half_width"],
                   "r_c": 3 * prof["half_width"], "thetas_deg": THETAS_DEG, "T_end": T_END,
                   "dr": DR, "dt": DT, "R_box": R_BOX, "R_sponge": R_SPONGE, "sigma_max": SIGMA_MAX,
                   "sigma_profile": "sigma_max((r-R_sponge)/(R_box-R_sponge))^2", "rec_dt": REC_DT,
                   "scheme": "swirl.family.radial.Radial: u = r phi, 4th-order FD, RK4"},
        "definitions": {"frac": "E(r < 3 hw, t) / E_total(t = 0) of the same run",
                        "frac_vs_theta0_E0": "E(r < 3 hw, t) / E_total(t = 0) of the theta = 0 run",
                        "omega_at_T": "median of 2 pi / (upward zero-crossing period of phi(0,t)) over periods with midpoint in [T-50, T]",
                        "death_time": "first t > 50 with E(r<3hw) < 0.5 E(r<3hw, t=50)"},
        "criterion": {"text": "SESSION_02_PREDICTIONS.md WP-C block: |frac_theta(T) - frac_0(T)| <= 0.01 at T = 300, 500, 1000; omega within 0.005; no death",
                      "tol_frac": TOL_FRAC, "tol_omega": TOL_OMEGA, "T_check": T_CHECK},
        "runs": runs, "verdict": verdict, "worst_over_thetas": worst,
        "all_pass": all(v["pass"] for v in verdict.values()),
        "series": {k: {"times": v["times"][::4], "frac": (np.array(v["E_inside"][::4]) / v["E0"]).tolist(),
                       "omega_t": v["omega_t"], "omega": v["omega"]} for k, v in series.items()},
        "wall_seconds": time.time() - t0,
    }
    with open(OUT + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(OUT + ".tmp", OUT)
    print(json.dumps({"runs": runs, "verdict": verdict, "all_pass": res["all_pass"]}, indent=1))


if __name__ == "__main__":
    main()
