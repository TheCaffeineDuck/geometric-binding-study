"""Test B run driver: the radial solver (swirl.family.radial.Radial, unchanged) with probe
recording, energy bookkeeping, checkpoint/resume and atomic writes.

Probes (read-only on the solver state, every rec_dt = 0.2): phi, phi_t and phi_r at
r = 0, 10, 15, 20, 30, 45, 60, 80, 120 (grid points). phi = u/r, phi_t = v/r (exact state
variables); phi_r = (u_r - u/r)/r with u_r the 4th-order central difference of u; at
r = 0 phi and phi_t use the solver's own limit (4 f_1 - f_2)/3 and phi_r = 0 (phi even).

Energy (every 1.0): E_in(r_c) for r_c = 3 hw and each probe radius (the Session 1
physical-density quadrature, Radial.energy_inside), E_total of the same density,
E_h (the exactly conserved discrete energy) and absorbed = E_h(0) - E_h(t) (exact up
to the RK4 error, since E_h is conserved by the semi-discrete system without sponge).
Every 10: the nonlinear share (E4 + E6)/E_total.

Runs (Set-B-like units, m = 1, g4 = 0.30, g6 = 0.09 beta, beta = 4/3):
  b1_primary   Session 1 start (table profile, np.interp, phi_t = 0), T = 20000
  b2_dip       Session 2 dip start (solve_profile 0.9065045072253043, tol 1e-10), T = 20000
  b1_linear    b1_primary with g4 = g6 = 0, T = 20000
  b1_box2      b1_primary in the doubled box (R_s 400, R_box 800), T = 5000
  b1_half_dr   b1_primary with dr and dt halved, T = 2000
  b1_smoke     b1_primary to T = 2000 (smoke)
Box: the Session 2 validated form (box2 run of the dip to 300,000): R_s = 200, R_box = 400,
sigma = 1 ((r - R_s)/(R_box - R_s))^2; every probe (r <= 120) is in the undamped interior.

Run: NUMBA_NUM_THREADS=1 <venv python> -m swirl.session4.testB.run <name> [<name> ...]
"""

import json
import os
import sys
import time

import numpy as np

from swirl.family.profile import profile_curve, solve_profile
from swirl.family.radial import Radial
from swirl.family.wpa_run import energy_parts, nonlinear_shares, omega_series, schmitt_offline

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "outputs", "session4", "testB")
CKPT = os.path.join(ROOT, "checkpoints", "session4", "testB")

M, G4, BETA = 1.0, 0.30, 4.0 / 3.0
PROBES = [0.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 80.0, 120.0]
REC_DT, E_DT, NL_DT = 0.2, 1.0, 10.0
BASE_GRID = dict(dr=0.05, dt=0.0125, R_sponge=200.0, R_box=400.0, sigma_max=1.0)
PRIMARY_TABLE = "outputs/step2/profiles/profile_beta1.3333_w0.9091.json"
OMEGA_DIP = 0.9065045072253043

RUNS = {
    "b1_primary": dict(start="primary", T=20000.0),
    "b2_dip": dict(start="dip", T=20000.0),
    "b1_linear": dict(start="primary", T=20000.0, linear=True),
    # Orchestrator, 2026-11-09, before production: doubled boxes cover the full window,
    # and the dip run gets one too, so a later-epoch line can be checked against the box.
    "b1_box2": dict(start="primary", T=20000.0, R_sponge=400.0, R_box=800.0),
    "b2_box2": dict(start="dip", T=20000.0, R_sponge=400.0, R_box=800.0),
    "b1_half_dr": dict(start="primary", T=2000.0, dr=0.025, dt=0.00625),
    "b1_smoke": dict(start="primary", T=2000.0),
}


PRODUCTION = ["b1_primary", "b2_dip", "b1_linear", "b1_box2", "b2_box2", "b1_half_dr"]


def start_profile(start):
    """(phi(r) callable, info dict) for the named start."""
    if start == "primary":
        d = json.load(open(os.path.join(ROOT, PRIMARY_TABLE)))
        r, P = np.array(d["r"]), np.array(d["Phi"])
        info = dict(omega_profile=d["omega"], half_width=d["half_width"], A=d["A"],
                    kappa=d["kappa"], source=PRIMARY_TABLE + " (np.interp, right=0), as "
                    "outputs/step2/diagnostics/primary_lifetime.py")
        return (lambda x: np.interp(x, r, P, right=0.0)), info
    if start == "dip":
        p = solve_profile(OMEGA_DIP, 1.0, 1.0, BETA, tol=1e-10)
        info = dict(omega_profile=OMEGA_DIP, half_width=p["half_width"], A=p["A"] / np.sqrt(G4),
                    kappa=p["kappa"], source="solve_profile(%.16g, 1, 1, 4/3, tol=1e-10) / sqrt(g4), "
                    "as swirl.family.wpa_run" % OMEGA_DIP)
        return (lambda x: profile_curve(p, x) / np.sqrt(G4)), info
    raise ValueError(start)


def params_for(name, **over):
    cfg = dict(RUNS[name], **over)
    grid = dict(BASE_GRID)
    for k in grid:
        if k in cfg:
            grid[k] = cfg[k]
    linear = bool(cfg.get("linear", False))
    _, info = start_profile(cfg["start"])
    return dict(name=name, start=cfg["start"], T=float(cfg["T"]), linear=linear, m=M,
                beta=BETA, g4_run=0.0 if linear else G4, g6_run=0.0 if linear else BETA * G4**2,
                probes=PROBES, rec_dt=REC_DT, energy_dt=E_DT, nl_dt=NL_DT, r_c=3.0 * info["half_width"],
                initial_data="phi = Phi(r), phi_t = 0", profile=info,
                sigma_profile="sigma_max((r-R_sponge)/(R_box-R_sponge))^2",
                scheme="swirl.family.radial.Radial: u=r phi, 4th-order FD, odd reflection at r=0 and "
                       "r=R_box, RK4", **grid)


def probe_index(s, radii):
    idx = np.array([int(round(r / s.dr)) for r in radii])
    if np.max(np.abs(idx * s.dr - np.array(radii))) > 1e-9:
        raise ValueError("probe radii must be grid points")
    return idx


def read_probes(s, idx):
    """phi, phi_t, phi_r at grid indices idx from the state (no mutation)."""
    u, v, h = s.u, s.v, s.dr
    phi = np.empty(idx.size)
    pt = np.empty(idx.size)
    pr = np.empty(idx.size)
    for j, i in enumerate(idx):
        if i == 0:
            phi[j] = (4.0 * u[1] / s.r[1] - u[2] / s.r[2]) / 3.0
            pt[j] = (4.0 * v[1] / s.r[1] - v[2] / s.r[2]) / 3.0
            pr[j] = 0.0
        else:
            r = s.r[i]
            ur = (-u[i + 2] + 8.0 * u[i + 1] - 8.0 * u[i - 1] + u[i - 2]) / (12.0 * h)
            phi[j] = u[i] / r
            pt[j] = v[i] / r
            pr[j] = (ur - u[i] / r) / r
    return phi, pt, pr


def _atomic_savez(path, **arrs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def run(name, out_dir=OUT, ckpt_dir=CKPT, ckpt_every=2000.0, stop_after=None, verbose=True, **over):
    """Run (or resume) one named run. Returns the summary dict (also written as JSON).

    stop_after: checkpoint and return at this time (used to test resume)."""
    P = params_for(name, **over)
    pkey = json.dumps(P, sort_keys=True)
    phi_init, info = start_profile(P["start"])
    s = Radial(P["R_box"], P["dr"], M, P["g4_run"], P["g6_run"], sponge_start=P["R_sponge"],
               sigma_max=P["sigma_max"])
    dt = P["dt"]
    nsub = int(round(REC_DT / dt))
    n_rec = int(round(P["T"] / REC_DT))
    e_every = int(round(E_DT / REC_DT))
    nl_every = int(round(NL_DT / REC_DT))
    idx = probe_index(s, PROBES)
    rcs = np.array([P["r_c"]] + PROBES[1:])
    np_ = len(PROBES)
    ck = os.path.join(ckpt_dir, name + ".npz")
    wall0 = time.time()
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=False)
        if str(z["pkey"]) != pkey:
            raise RuntimeError("checkpoint %s has different parameters" % ck)
        s.u, s.v = z["u"].copy(), z["v"].copy()
        k0 = int(z["k"])
        rec = {key: z[key].copy() for key in ("phi", "phit", "phir", "E_in", "E_tot", "E_h", "nl")}
        Eh0, E0 = float(z["Eh0"]), float(z["E0"])
        wall_prior = float(z["wall"])
        if verbose:
            print("[%s] resume at t = %.1f" % (name, k0 * REC_DT), flush=True)
    else:
        s.set_phi(phi_init)
        k0 = 0
        rec = dict(phi=np.zeros((n_rec + 1, np_)), phit=np.zeros((n_rec + 1, np_), np.float32),
                   phir=np.zeros((n_rec + 1, np_), np.float32),
                   E_in=np.zeros((n_rec // e_every + 1, rcs.size)), E_tot=np.zeros(n_rec // e_every + 1),
                   E_h=np.zeros(n_rec // e_every + 1), nl=np.zeros(n_rec // nl_every + 1))
        a, b = s.energy_inside(rcs)
        Eh0, E0 = s.energy_discrete(), float(b)
        rec["E_in"][0], rec["E_tot"][0], rec["E_h"][0] = a, b, Eh0
        rec["nl"][0] = nonlinear_shares(energy_parts(s))["signed"]
        f, g, h = read_probes(s, idx)
        rec["phi"][0], rec["phit"][0], rec["phir"][0] = f, g, h
        wall_prior = 0.0
    last_ck = k0
    for k in range(k0 + 1, n_rec + 1):
        s.step(dt, nsub)
        f, g, h = read_probes(s, idx)
        rec["phi"][k], rec["phit"][k], rec["phir"][k] = f, g, h
        if k % e_every == 0:
            a, b = s.energy_inside(rcs)
            j = k // e_every
            rec["E_in"][j], rec["E_tot"][j], rec["E_h"][j] = a, b, s.energy_discrete()
        if k % nl_every == 0:
            rec["nl"][k // nl_every] = nonlinear_shares(energy_parts(s))["signed"]
        done = k == n_rec
        stop_now = stop_after is not None and k * REC_DT >= stop_after - 1e-9
        if (k - last_ck) * REC_DT >= ckpt_every - 1e-9 or stop_now or done:
            _atomic_savez(ck, u=s.u, v=s.v, k=k, Eh0=Eh0, E0=E0, pkey=pkey,
                          wall=wall_prior + time.time() - wall0, **rec)
            last_ck = k
            if verbose:
                print("[%s] t = %.0f  E_in(3hw)/E0 = %.6f  wall %.0f s" % (
                    name, k * REC_DT, rec["E_in"][k // e_every, 0] / E0 if k % e_every == 0 else np.nan,
                    wall_prior + time.time() - wall0), flush=True)
        if stop_now and not done:
            return dict(partial=True, t=k * REC_DT)
    wall = wall_prior + time.time() - wall0
    t = REC_DT * np.arange(n_rec + 1)
    te = E_DT * np.arange(rec["E_tot"].size)
    tn = NL_DT * np.arange(rec["nl"].size)
    series = os.path.join(out_dir, name + "_series.npz")
    _atomic_savez(series, t=t, probes=np.array(PROBES), phi=rec["phi"], phit=rec["phit"], phir=rec["phir"],
                  t_energy=te, r_c_energy=rcs, E_in=rec["E_in"], E_tot=rec["E_tot"], E_h=rec["E_h"],
                  absorbed=Eh0 - rec["E_h"], t_nl=tn, nl_share=rec["nl"], params=pkey)
    tc = schmitt_offline(t, rec["phi"][:, 0])
    tw, w = omega_series(tc)
    sel = (tw >= 50) & (tw <= 150)
    summ = dict(params=P, wall_s=wall, wall_s_per_time_unit=wall / P["T"], E0=E0, Eh0=Eh0,
                series_file=os.path.relpath(series, ROOT), series_bytes=os.path.getsize(series),
                regime=dict(omega_over_m_settled_50_150=float(np.median(w[sel])) if sel.any() else None,
                            omega_over_m_end_last100=float(np.median(w[tw >= t[-1] - 100])) if tw.size else None,
                            nonlinear_share_t0_signed=float(rec["nl"][0]),
                            nonlinear_share_mean_signed=float(np.mean(rec["nl"]))),
                frac_end=float(rec["E_in"][-1, 0] / E0), absorbed_end=float(Eh0 - rec["E_h"][-1]),
                E_h_rise_max_rel=float(np.max(np.diff(rec["E_h"])) / Eh0),
                probe_storage="phi float64; phi_t, phi_r float32 (relative precision 6e-8, used only "
                              "for flux)")
    _atomic_json(os.path.join(out_dir, name + "_run.json"), summ)
    if verbose:
        print("[%s] done, wall %.1f s" % (name, wall), flush=True)
    return summ


def main(argv):
    names = PRODUCTION if argv == ["all"] else argv
    for n in names:
        if n not in RUNS:
            raise SystemExit("unknown run %s; known: %s" % (n, ", ".join(RUNS)))
    for n in names:
        run(n)


if __name__ == "__main__":
    main(sys.argv[1:])
