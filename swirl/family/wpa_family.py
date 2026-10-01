"""WP-A stage 1: the single-frequency family at a new beta, with the Session 1 checks.

Same solver and output format as swirl.family.run_family (family_beta*.json), written
to outputs/session2/wpA/. The scan grid is mapped from the beta = 4/3 grid in the
universal variable alpha^2 = 0.3 beta (1 - omega^2) (PRIOR_ART 1.3): above 4/3 the
stable branch in omega shrinks as 1/beta, so the start omega and the omega steps scale
as (4/3)/beta; at or below 4/3 the Session 1 grid (start 0.95, step 0.005) is kept.

Checks stored in the file under 'checks_wpA':
- Derrick identity (T + 3 int G)/T for every row (target 1e-6).
- dE/domega = omega dN/domega: finite differences on the rows (Session 1 form) and
  central differences (h = 1e-4 x (4/3)/beta) at five omegas on the stable branch.
- Universal curve: Psi0 = P0/a and N_AS = 2N/(omega a^2 L^3), a^2 = 20/beta,
  L^2 = 0.3 beta, against the beta = 4/3 family solved at the same alpha^2.
"""

import os
import time

import numpy as np

from swirl.family.family import at, features, scan, table, to_setB_units
from swirl.family.profile import solve_profile
from swirl.family.wpa_run import write_json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "outputs", "session2", "wpA")
OLD = os.path.join(ROOT, "outputs", "family")
TOL = 1e-8
BETA43 = 4.0 / 3.0


def scan_params(beta):
    s = min(1.0, BETA43 / beta)
    w_start = 0.95 if beta <= BETA43 + 1e-12 else float(np.sqrt(1.0 - 0.0975 * s))
    return w_start, 0.005 * s, s


def family_path(beta):
    """Session 2 file if present, else the Session 1 file."""
    for d in (OUT, OLD):
        p = os.path.join(d, "family_beta%.4f.json" % beta)
        if os.path.exists(p):
            return p
    return None


def universal(s, beta):
    a, L = np.sqrt(20.0 / beta), np.sqrt(0.3 * beta)
    return s["P0"] / a, 2.0 * s["N"] / (s["omega"] * a * a * L**3)


def run_beta(beta, sols43=None, n_universal=20):
    t0 = time.time()
    w_start, dw, sc = scan_params(beta)
    sols = scan(beta, w_start=w_start, dw=dw, tol=TOL)
    f = features(sols, beta, tol=TOL)
    extra = [f["omega_minN"] + d for d in np.arange(-0.004, 0.0041, 0.0005) * sc]
    for wc in f["omega_E_eq_mN"]:
        extra += [wc + d for d in np.array([-0.001, -0.0005, 0.0, 0.0005, 0.001]) * sc]
    sols += [at(sols, beta, w, TOL) for w in extra]
    sols = sorted(sols, key=lambda s: -s["omega"])
    w = np.array([s["omega"] for s in sols])
    N = np.array([s["N"] for s in sols])
    E = np.array([s["E"] for s in sols])
    dE = np.gradient(E, w)
    dN = np.gradient(N, w)
    rel = np.abs(dE - w * dN) / np.maximum(np.abs(dE), 1e-300)
    f["dEdw_eq_w_dNdw_median_rel"] = float(np.median(rel))
    # central differences at five points of the stable branch
    a2c = 0.3 * beta * (1 - f["omega_minN"] ** 2)
    cd = []
    h = 1e-4 * sc
    for a2 in np.linspace(a2c + 0.01, 0.14, 5):
        wc = float(np.sqrt(1 - a2 / (0.3 * beta)))
        sp, sm = at(sols, beta, wc + h, TOL), at(sols, beta, wc - h, TOL)
        dEc, dNc = (sp["E"] - sm["E"]) / (2 * h), (sp["N"] - sm["N"]) / (2 * h)
        cd.append(dict(omega=wc, alpha2=float(a2), rel=float(abs(dEc - wc * dNc) / abs(wc * dNc))))
    # universal curve against beta = 4/3 at the same alpha^2
    uni = []
    if sols43 is not None and abs(beta - BETA43) > 1e-12:
        a2rows = 0.3 * beta * (1 - w**2)
        a2_43 = np.array([0.3 * BETA43 * (1 - s["omega"] ** 2) for s in sols43])
        ok = (a2rows > a2_43.min() * 1.001) & (a2rows < a2_43.max() * 0.999) & (a2rows > 1e-3)
        idx = np.where(ok)[0]
        idx = idx[np.linspace(0, idx.size - 1, min(n_universal, idx.size)).astype(int)]
        for i in idx:
            s = sols[i]
            w43 = float(np.sqrt(1 - a2rows[i] / (0.3 * BETA43)))
            s43 = at(sols43, BETA43, w43, TOL)
            p, n = universal(s, beta)
            p43, n43 = universal(s43, BETA43)
            uni.append(dict(omega=s["omega"], alpha2=float(a2rows[i]), Psi0=p, N_AS=n,
                            Psi0_43=p43, N_AS_43=n43, rel_Psi0=float(abs(p - p43) / p43),
                            rel_N_AS=float(abs(n - n43) / n43)))
    checks = dict(
        derrick_worst=float(max(abs(s["derrick_rel"]) for s in sols)),
        derrick_target=1e-6, n_rows=len(sols),
        dEdw_rows_median_rel=f["dEdw_eq_w_dNdw_median_rel"],
        dEdw_rows_worst_rel=float(np.max(rel)),
        dEdw_rows_note="finite differences on the non-uniform row grid; the relative form "
                       "diverges where dE/domega -> 0 at the minimum, so the worst row value is "
                       "a grid statement, the central differences are the check",
        dEdw_central=cd, dEdw_central_worst=float(max(c["rel"] for c in cd)),
        universal=uni,
        universal_worst=float(max([max(u["rel_Psi0"], u["rel_N_AS"]) for u in uni] or [np.nan])),
        universal_target=1e-5,
        scan=dict(w_start=w_start, dw=dw, scale=sc))
    out = dict(
        beta=beta,
        equation="Phi'' + (2/r) Phi' = (m^2-omega^2) Phi - (g4/8) Phi^3 + (g6/192) Phi^5",
        rescaled_units="m=1, g4=1, g6=beta; lengths 1/m, field m/sqrt(g4), E m/g4, N 1/g4",
        half_width_definition="radius where Phi = Phi(0)/2",
        tail_definition="Phi -> A exp(-kappa r)/r, kappa = sqrt(m^2-omega^2); A = mean of "
                        "r Phi exp(kappa r) over tail_window (1e-9 < Phi/Phi0 < 1e-3, r < 0.8 R_box); "
                        "A_rel_spread = std/mean there; kappa_fit = free log-linear fit",
        solver=dict(method="scipy solve_bvp collocation, Robin tail BC at R_box", tol=TOL,
                    continuation_from=w_start, dw=dw),
        features_rescaled=f,
        features_setB_units=to_setB_units(f),
        rows_rescaled=table(sols),
        rows_setB_units=[to_setB_units(r) for r in table(sols)],
        checks_wpA=checks,
        wall_s=time.time() - t0,
    )
    write_json(os.path.join(OUT, "family_beta%.4f.json" % beta), out)
    return checks


def beta43_scan():
    return scan(BETA43, tol=TOL)
