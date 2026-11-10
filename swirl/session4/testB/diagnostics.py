"""Test B diagnostics D-B1 to D-B3 (docs/PRIOR_ART.md, "Session 4, Test B against the
literature", V5). Set-B-like units, m = 1, g4 = 0.30, g6 = 0.12 (beta = 4/3).

D-B1 (no PDE): harmonic radiation amplitudes from the single-frequency profile psi(r).
  With phi = psi cos(w t) + phi_3 cos(3 w t) + phi_5 cos(5 w t) + ..., the projections of
  phi_tt - lap phi + m^2 phi = (g4/6) phi^3 - (g6/120) phi^5 onto cos 3wt and cos 5wt give
      (-lap + m^2 - 9 w^2 + U) phi_3 = S_3 = (g4/24) psi^3 - (g6/384) psi^5
      (-lap + m^2 - 25 w^2 + U) phi_5 = S_5 = -(g6/1920) psi^5  [+ S_5' = (g4/8 psi^2 - g6/96 psi^4) phi_3]
  (cos^3 = (3 cos + cos 3)/4, cos^5 = (10 cos + 5 cos 3 + cos 5)/16), with the lump's effective
  potential U = -(g4/4) psi^2 + (g6/64) psi^4 (the cos 3 / cos 5 diagonal of V''(psi cos)).
  The family profile solves the cos-projection of the same equation, so S_n is exactly the
  residual at n w. Levels:
    (i)   Born: U = 0, S_n only; outgoing amplitude r a_n = |int_0^inf S_n r sin(k_n r)/k_n dr|,
          k_n = sqrt(n^2 w^2 - m^2) (the form of Li et al. 2025 eq. 2.28 / Zhang et al. 2020 eq. 5.1
          as the scholar describes it, with Xi_n(k) = 4 pi int psi^n r sin(k r)/k dr);
    (ii)  with U (radial ODE for u = r phi_n, outgoing condition u' = i k u at R);
    (iii) 5 w only: (ii) plus the second-order source S_5' from the complex phi_3 of (ii).
  Also phi_n(0), against the core spectrum's harmonic amplitudes at r = 0.
D-B2: sideband spacing against m - omega (radial runs db2_*, and the dip and primary series).
D-B3 (no PDE): omega(t) from the measured lump energy mapped onto the family E(omega).

Run: <venv python> -m swirl.session4.testB.diagnostics db1|db2|db3
"""

import json
import math
import os
import sys

import numpy as np
from scipy.integrate import simpson
from scipy.linalg import solve_banded

from swirl.family.profile import profile_curve, solve_profile
from swirl.session4.testB import analysis as A

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "outputs", "session4", "testB", "diagnostics")
M, G4, BETA = 1.0, 0.30, 4.0 / 3.0
G6 = BETA * G4**2
RMAX, H = 80.0, 0.005

MEASURED = dict(  # r times a at the far probes, epoch [0, 5000] (orchestrator's numbers)
    primary=dict(omega_profile=0.909149280617113, omega_measured=0.9100, a3=0.0167, a5=2.49e-4),
    dip=dict(omega_profile=0.9065045072253043, omega_measured=0.906836, a3=3.74e-4, a5=1.575e-4))


def _write(name, obj):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    with open(p + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(p + ".tmp", p)


# ------------------------------------------------------------------ D-B1
def profile_on_grid(omega):
    p = solve_profile(omega, 1.0, 1.0, BETA, tol=1e-10)
    r = np.arange(0.0, RMAX + H / 2, H)
    return r, profile_curve(p, r) / np.sqrt(G4), p


def born(S, r, k):
    f = S * r * np.sin(k * r) / k
    return float(simpson(f, x=r))


def outgoing(Ueff, rhs_u, r, k):
    """Solve -u'' + (Ueff - k^2) u = rhs_u on (0, R], u(0) = 0, u'(R) = i k u(R); 2nd-order FD.
    Returns complex u on the grid."""
    n = r.size - 1                       # unknowns u_1..u_n
    h = r[1] - r[0]
    d = 2.0 / h**2 + Ueff[1:] - k * k + 0j
    lo = np.full(n, -1.0 / h**2, complex)
    up = np.full(n, -1.0 / h**2, complex)
    b = rhs_u[1:].astype(complex)
    # ghost u_{n+1} = u_{n-1} + 2 h i k u_n
    lo[-1] = -2.0 / h**2                  # coefficient of u_{n-1} in the last row
    d[-1] = d[-1] - 2j * k / h
    ab = np.zeros((3, n), complex)
    ab[0, 1:] = up[:-1]
    ab[1] = d
    ab[2, :-1] = lo[1:]
    u = np.zeros(r.size, complex)
    u[1:] = solve_banded((1, 1), ab, b)
    return u


def harmonics(omega, psi_r=None):
    r, psi, p = profile_on_grid(omega) if psi_r is None else (*psi_r, None)
    k3, k5 = math.sqrt(9 * omega**2 - M * M), math.sqrt(25 * omega**2 - M * M)
    S3 = G4 / 24 * psi**3 - G6 / 384 * psi**5
    S5 = -G6 / 1920 * psi**5
    U = -G4 / 4 * psi**2 + G6 / 64 * psi**4
    out = dict(omega=omega, k3=k3, k5=k5, psi0=float(psi[0]))
    out["born_a3"] = born(S3, r, k3)
    out["born_a5"] = born(S5, r, k5)
    rs = r.copy()
    rs[0] = 1.0
    u3_0 = outgoing(np.zeros_like(r), r * S3, r, k3)          # (i) by the ODE, a check of the solver
    u3 = outgoing(U, r * S3, r, k3)
    u5 = outgoing(U, r * S5, r, k5)
    phi3 = u3 / rs
    phi3[0] = (4 * phi3[1] - phi3[2]) / 3
    S5b = (G4 / 8 * psi**2 - G6 / 96 * psi**4) * phi3
    u5b = outgoing(U, r * (S5 + S5b), r, k5)
    tail = slice(-200, None)                                  # |u| at r = 79 to 80 (U, S negligible)
    out.update(born_a3_by_ode=float(np.mean(np.abs(u3_0[tail]))),
               a3_with_U=float(np.mean(np.abs(u3[tail]))), a5_with_U=float(np.mean(np.abs(u5[tail]))),
               a5_with_U_and_second_order=float(np.mean(np.abs(u5b[tail]))),
               phi3_core_with_U=float(abs(phi3[0])),
               phi5_core_with_U_and_second_order=float(abs((4 * u5b[1] / r[1] - u5b[2] / r[2]) / 3)),
               profile_E_setB=None if p is None else p["E"] / G4, profile_N_setB=None if p is None else p["N"] / G4)
    return out


def db1():
    rows = []
    for w in np.round(np.arange(0.8950, 0.9250 + 1e-9, 0.0005), 4):
        rows.append(harmonics(float(w)))
    pts = {}
    for name, m in MEASURED.items():
        for key in ("omega_profile", "omega_measured"):
            h = harmonics(m[key])
            h.update(ratio_born_a3=abs(h["born_a3"]) / m["a3"], ratio_born_a5=abs(h["born_a5"]) / m["a5"],
                     ratio_U_a3=h["a3_with_U"] / m["a3"], ratio_U_a5=h["a5_with_U"] / m["a5"],
                     ratio_U2_a5=h["a5_with_U_and_second_order"] / m["a5"])
            pts["%s_%s" % (name, key)] = h
    w = np.array([x["omega"] for x in rows])
    b3 = np.array([x["born_a3"] for x in rows])
    i = np.where(np.sign(b3[:-1]) != np.sign(b3[1:]))[0]
    zeros_born = [float(w[j] - b3[j] * (w[j + 1] - w[j]) / (b3[j + 1] - b3[j])) for j in i]
    aU = np.array([x["a3_with_U"] for x in rows])
    jm = int(np.argmin(aU))
    # refine the minimum of |a3| with U by a parabola in omega
    if 0 < jm < len(w) - 1:
        c = np.polyfit(w[jm - 1:jm + 2], aU[jm - 1:jm + 2], 2)
        wmin = float(-c[1] / (2 * c[0]))
    else:
        wmin = float(w[jm])
    res = dict(description=__doc__.split("D-B2")[0], params=dict(m=M, g4=G4, g6=G6, beta=BETA, rmax=RMAX, h=H),
               measured=MEASURED, points=pts, scan=rows, zero_of_born_a3=zeros_born,
               min_of_a3_with_U=dict(omega=wmin, grid_min=float(aU[jm]), grid_omega=float(w[jm])))
    _write("db1_harmonic_amplitudes.json", res)
    return res


# ------------------------------------------------------------------ D-B2
def sidebands(series, t0, t1, radii=(0.0, 10.0, 30.0, 120.0), series_dir=A.OUT):
    """Lines within 0.02 to 0.25 of an odd harmonic n omega_c in one Hann window [t0, t1]."""
    z = np.load(os.path.join(series_dir, series + "_series.npz"))
    t, phi, pr = z["t"], z["phi"], list(z["probes"])
    dt = float(t[1] - t[0])
    twin = t1 - t0
    om, P0, _, _ = A.welch(phi[:, 0], dt, t0, t1, twin=twin)
    dom = om[1] - om[0]
    wc = A.peak_in(om, P0, 0.5, 1 - 3 * dom)
    out = dict(series=series, window=[t0, t1], d_omega=dom, omega_core=wc, gap=1 - wc, probes=[])
    for r in radii:
        ip = pr.index(r)
        om, P, _, _ = A.welch(phi[:, ip], dt, t0, t1, twin=twin)
        lines = A.lines_in(om, P, wc, None, near=(1 - 3 * dom, 1 + 3 * dom), max_lines=60)
        harm = {d["n"]: d for d in lines if d["label"] == "%d omega" % d["n"]}
        sb = []
        for d in lines:
            if d["label"] != "unidentified":
                continue
            n = int(round(d["omega"] / wc))
            for nn in (n - 1, n, n + 1):
                if nn in harm and 0.02 <= abs(d["omega"] - harm[nn]["omega"]) <= 0.25:
                    sb.append(dict(n=nn, omega=d["omega"], offset=d["omega"] - harm[nn]["omega"],
                                   amp=d["amp"], rel_to_harmonic=d["amp"] / harm[nn]["amp"]))
        sp = [abs(x["offset"]) for x in sb]
        nb = [d for d in lines if d["label"] == "near-threshold band"]
        out["probes"].append(dict(r=r, sidebands=sb, spacing_median=float(np.median(sp)) if sp else None,
                                  near_threshold_peak=nb[0]["omega"] if nb else None,
                                  spacings=sp, harmonics={str(n): harm[n]["amp"] for n in sorted(harm)}))
    allsp = [s for p in out["probes"] for s in p["spacings"]]
    out["spacing_median_all"] = float(np.median(allsp)) if allsp else None
    out["spacing_spread"] = [float(min(allsp)), float(max(allsp))] if allsp else None
    if allsp:
        out["omega_core_plus_spacing"] = wc + out["spacing_median_all"]
        out["spacing_minus_gap"] = out["spacing_median_all"] - (1 - wc)
    return out


def db2():
    cases = [("db2_w0.857370", 1000.0, 5000.0), ("db2_w0.9025", 1000.0, 5000.0), ("db2_w0.9115", 1000.0, 5000.0),
             ("b2_dip", 1000.0, 5000.0), ("b2_dip", 5000.0, 9000.0), ("b2_dip", 10000.0, 14000.0),
             ("b2_dip", 16000.0, 20000.0), ("b2_box2", 1000.0, 5000.0), ("b2_box2", 16000.0, 20000.0),
             ("b1_primary", 1000.0, 5000.0), ("b1_primary", 5000.0, 9000.0)]
    rows = [sidebands(*c) for c in cases]
    res = dict(method="one Hann window of 4000 time units (resolution 0.00157); lines as in analysis.lines_in; "
                      "sidebands = unidentified lines 0.02 to 0.25 from an identified odd harmonic n omega_core; "
                      "spacing = |omega_sideband - omega_harmonic| (refined peaks)",
               rows=rows)
    _write("db2_sidebands.json", res)
    return res


# ------------------------------------------------------------------ D-B3
def db3():
    fam = json.load(open(os.path.join(ROOT, "outputs", "family", "family_beta1.3333.json")))
    rows = sorted(fam["rows_setB_units"], key=lambda x: x["omega"])
    w = np.array([x["omega"] for x in rows])
    E = np.array([x["E"] for x in rows])
    wfold = fam["features_setB_units"]["omega_minE"] if "omega_minE" in fam["features_setB_units"] else None
    sel = (w >= 0.85) & (w <= (wfold if wfold else 0.969))
    ws, Es = w[sel], E[sel]
    an = json.load(open(os.path.join(A.OUT, "b1_primary_analysis.json")))
    z = np.load(os.path.join(A.OUT, "b1_primary_series.npz"))
    te, Ein = z["t_energy"], z["E_in"][:, 0]
    # anchor: first window (t 0 to 2000) omega 0.90965 at t = 1000
    ep0 = an["epochs"][0]
    w_anchor, t_anchor = ep0["omega_core_per_window"][0], ep0["window_starts"][0] + 1000.0
    E_anchor_fam = float(np.interp(w_anchor, ws, Es))
    Ein_anchor = float(np.mean(Ein[(te >= t_anchor - 10) & (te <= t_anchor + 10)]))
    Efam_t = E_anchor_fam + (Ein - Ein_anchor)              # family energy the lump has reached
    w_t = np.interp(Efam_t, Es[::-1], ws[::-1], left=np.nan, right=np.nan)   # E decreasing in omega
    out_w = []
    for ep in an["epochs"]:
        for s, wm in zip(ep["window_starts"], ep["omega_core_per_window"]):
            tm = s + 1000.0
            sel_t = (te >= s) & (te <= s + 2000)
            out_w.append(dict(t_mid=tm, omega_measured=wm, omega_family_from_E=float(np.nanmean(w_t[sel_t]))))
    # local rates per epoch: d omega/dt = L / |dE/domega|
    dEdw = np.gradient(Es, ws)
    rates = []
    for ep in an["epochs"]:
        wc = ep["omega_core"]
        g = float(np.interp(wc, ws, dEdw))
        st, wm = np.array(ep["window_starts"]) + 1000.0, np.array(ep["omega_core_per_window"])
        meas = float(np.polyfit(st, wm, 1)[0]) if st.size > 1 else None
        rates.append(dict(t=[ep["t0"], ep["t1"]], omega_core=wc, loss_rate=ep["loss_rate_Ein_3hw"],
                          dE_domega=g, domega_dt_pred=ep["loss_rate_Ein_3hw"] / abs(g),
                          domega_dt_measured_window_slope=meas))
    epoch_w = [ep["omega_core"] for ep in an["epochs"]]
    pred_epoch = [float(np.nanmean(w_t[(te >= ep["t0"]) & (te <= ep["t1"])])) for ep in an["epochs"]]
    E_fold = float(np.interp(wfold, ws, Es)) if wfold else float(Es.min())
    need = E_anchor_fam - E_fold
    dropped = Ein_anchor - Ein
    j = np.where(dropped >= need)[0]
    res = dict(method="omega(t) = E_family^-1(E_family(omega_anchor) + E_in(3hw, t) - E_in(3hw, t_anchor)); "
                      "anchor: first window omega at t = 1000; family table outputs/family/family_beta1.3333.json "
                      "(Set-B units, rows omega 0.85 to the fold)",
               omega_fold=wfold, E_fold=E_fold, anchor=dict(t=t_anchor, omega=w_anchor, E_family=E_anchor_fam,
                                                               E_in=Ein_anchor),
               windows=out_w, epochs=dict(measured=epoch_w, predicted=pred_epoch,
                                          steps_measured=list(np.diff(epoch_w)), steps_predicted=list(np.diff(pred_epoch))),
               local_rates=rates,
               fold_reached_at_t=float(te[j[0]]) if j.size else None,
               energy_to_fold=need, energy_dropped_by_T=float(dropped[-1]))
    _write("db3_fold_kinematics.json", res)
    return res


if __name__ == "__main__":
    f = dict(db1=db1, db2=db2, db3=db3)[sys.argv[1]]
    r = f()
    print(json.dumps({k: v for k, v in r.items() if k not in ("scan", "description", "rows", "windows")}, indent=1)[:6000])
