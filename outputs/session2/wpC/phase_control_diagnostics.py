"""Cheap diagnostics of the theta = 90 phase-control miss (CLAUDE.md: up to three, no fix).

D1. The start phases a symmetric pair assignment theta_{1,2} = -/+ dtheta/2 would use:
    theta = +-22.5, +-45, +-67.5 degrees, judged by the same criterion against theta = 0.
    (-theta has the same E0 as theta; it is the time-reversed slice, not a symmetry of the run.)
D2. Is the theta = 90 lump just another member of the family? Start theta = 0 from the
    family profile at the theta = 90 run's settled omega (median over t = 50..100) and
    compare omega(t) and the loss rate.
D3. The slice energy of the single-frequency ansatz, E0(theta) = int [omega^2 Phi^2 sin^2/2
    + Phi'^2 cos^2/2 + V(Phi cos)], against the time-averaged profile energy E.

Run: NUMBA_NUM_THREADS=1 <venv python> outputs/session2/wpC/phase_control_diagnostics.py
Writes outputs/session2/wpC/phase_control_diagnostics.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from phase_control_radial import PROFILE, T_CHECK, TOL_FRAC, TOL_OMEGA, analyse, evolve_phase  # noqa: E402
from swirl.family.profile import profile_curve, solve_profile  # noqa: E402

OUT = os.path.join(HERE, "phase_control_diagnostics.json")


def verdict(r, ref):
    df = {T: abs(r["frac_at"][T] - ref["frac_at"][T]) for T in r["frac_at"]}
    dw = {T: abs(r["omega_at"][T] - ref["omega_at"][T]) for T in r["omega_at"]}
    return {"abs_dfrac_at": df, "abs_domega_at": dw,
            "pass": bool(max(df.values()) <= TOL_FRAC and max(dw.values()) <= TOL_OMEGA and r["death_time"] is None)}


def main():
    prof = json.load(open(PROFILE))
    base = json.load(open(os.path.join(HERE, "phase_control_radial.json")))
    ref = base["runs"]["0.0"]
    out = {"description": __doc__, "params": base["params"], "criterion": base["criterion"]}
    # D1
    d1 = {}
    for th in (22.5, 67.5, -22.5, -45.0, -67.5):
        ts = evolve_phase(prof, np.deg2rad(th))
        a = analyse(ts, ref["E0"])
        d1[str(th)] = {**a, "verdict": verdict(a, ref)}
        print("D1 theta %6.1f E0 %.3f frac %s omega %s pass %s" % (th, a["E0"], a["frac_at"], a["omega_at"],
                                                                   d1[str(th)]["verdict"]["pass"]), flush=True)
    out["D1_symmetric_assignment_phases"] = d1
    # D2
    w90 = base["runs"]["90.0"]["omega_settled_50_100"]
    g4 = prof["g4"]
    res = solve_profile(w90, 1.0, 1.0, prof["beta"], tol=1e-10)
    r = np.arange(0.0, 34.0 + 1e-9, 0.01)
    Phi = profile_curve(res, r) / np.sqrt(g4)
    p2 = dict(prof)
    p2.update(omega=w90, r=r.tolist(), Phi=Phi.tolist(), half_width=prof["half_width"])  # same r_c as theta = 90
    ts = evolve_phase(p2, 0.0)
    a2 = analyse(ts, ts["E0"])
    r90 = base["runs"]["90.0"]
    s90 = base["series"]["90.0"]
    t90, f90 = np.array(s90["times"]), np.array(s90["frac"])
    t2 = np.array(ts["times"])
    f2 = np.array(ts["E_inside"]) / ts["E0"]

    def loss_rate(t, f, t0, t1):
        sel = (t >= t0) & (t <= t1)
        return float(-np.polyfit(t[sel], np.log(f[sel]), 1)[0])
    out["D2_family_member_at_theta90_omega"] = {
        "omega_profile": w90, "N_profile_setB": res["N"] / g4, "E_profile_setB": res["E"] / g4,
        "E0_theta0_slice": ts["E0"], "theta90_run_E0": r90["E0"],
        "omega_at": a2["omega_at"], "theta90_omega_at": r90["omega_at"],
        "loss_rate_200_1000": loss_rate(t2, f2, 200, 1000),
        "theta90_loss_rate_200_1000": loss_rate(t90, f90, 200, 1000),
        "theta0_loss_rate_200_1000": loss_rate(np.array(base["series"]["0.0"]["times"]),
                                               np.array(base["series"]["0.0"]["frac"]), 200, 1000),
        "frac_at": a2["frac_at"], "theta90_frac_at": r90["frac_at"]}
    print("D2", json.dumps(out["D2_family_member_at_theta90_omega"]), flush=True)
    # D3
    rr = np.array(prof["r"])
    P = np.array(prof["Phi"])
    dP = np.gradient(P, rr, edge_order=2)
    om = prof["omega"]

    def V(p):
        q = p * p
        return 0.5 * prof["m"] ** 2 * q - prof["g4"] / 24 * q * q + prof["g6"] / 720 * q ** 3
    d3 = {}
    for th in (0, 22.5, 45, 67.5, 90):
        c, s = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))
        e = 0.5 * (om * P * s) ** 2 + 0.5 * (dP * c) ** 2 + V(P * c)
        d3[str(th)] = float(4 * np.pi * np.trapezoid(e * rr * rr, rr))
    out["D3_slice_energy"] = {"E0_theta": d3, "E_profile_time_averaged": prof["E"], "N_profile": prof["N"],
                              "omega_N": om * prof["N"]}
    print("D3", d3, prof["E"], flush=True)
    with open(OUT + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(OUT + ".tmp", OUT)


if __name__ == "__main__":
    main()
