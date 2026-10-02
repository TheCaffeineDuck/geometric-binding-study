"""E11 start in the point-lump model: dependence of the merger time on the seed size.

Imports swirl.arrange (unchanged). Same start as WP-E E11: ring6+c, six lumps at phase 180 on a ring of
radius R = 15.767 (nearest-neighbour spacing R), centre lump at phase 0. Measured pair-law parameters.
Perturbation: position and phase sigma s (same standard-normal draws as scan.run_one, scaled), seed SEED.
Writes e11_seed_scan.json next to this file. Run: .venv/bin/python outputs/session2/wpD_measured/e11_seed_scan.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from swirl.arrange import geometry as geo  # noqa: E402
from swirl.arrange import model as mdl  # noqa: E402
from swirl.arrange import scan  # noqa: E402
from swirl.arrange.pairlaw import load_params, make_laws  # noqa: E402

PARAMS = os.path.join(ROOT, "outputs/session2/wpD/params_measured.json")
FSUP = os.path.join(ROOT, "outputs/session2/wpD_measured/fsup_table.json")
R_RUN = 15.767                  # E11 ring radius (WP-E)
T_RUN = 2000.0
SIGMAS = [1e-3, 1e-5, 1e-7, 1e-10, 0.0]
T_REPORT = [100, 300, 500, 700]
ASYM_FRAC = 0.01


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run(R, law, p, model_name, sigma, T=T_RUN):
    X1 = geo.arrangements()["ring6+c"]
    th = np.deg2rad([180.0] * 6 + [0.0])
    n = len(X1)
    rng = np.random.default_rng(scan.SEED)
    X0 = R * X1 + sigma * rng.standard_normal((n, 3))
    th0 = th + sigma * rng.standard_normal(n)
    d0 = scan.min_dist(R * X1)
    d_stop = 2 * p["hw"]
    if model_name == "B":
        d_stop = max(d_stop, law.d_min + 1e-6)
    t_eval = np.linspace(0, T, int(2 * T) + 1)
    sol = mdl.integrate(X0, th0, law, p["E"], p["domega_dN"], T, d_stop=d_stop, rtol=1e-12, atol=1e-12,
                        t_eval=t_eval)
    t = sol.t
    Xs = sol.y[:3 * n].T.reshape(-1, n, 3)
    ph = sol.y[6 * n:7 * n].T
    rad = np.linalg.norm(Xs[:, :6] - Xs[:, 6:7], axis=2)              # six distances to the centre lump
    rmean = rad.mean(1)
    spread = rad.max(1) - rad.min(1)
    dph = np.rad2deg(wrap(ph[:, :6].mean(1) - ph[:, 6]))
    asym = spread / rmean
    ia = np.nonzero(asym > ASYM_FRAC)[0]
    contact = float(sol.t_events[0][0]) if sol.status == 1 else None
    out = {"model": model_name, "R": R, "sigma": sigma, "nn_spacing0": d0, "stop_distance": d_stop,
           "outcome": "merger (contact: a pair reaches the stop distance)" if contact is not None else None,
           "contact_time": contact, "t_end": float(t[-1]),
           "asymmetry_passes_1pct_at": float(t[ia[0]]) if ia.size else None,
           "asymmetry_at_end_over_radius": float(asym[-1]), "asymmetry_max_over_radius": float(asym.max()),
           "ring_radius": {}, "ring_centre_phase_diff_deg": {}}
    for tr in T_REPORT:
        k = np.searchsorted(t, tr)
        ok = k < len(t) and abs(t[k] - tr) < 1e-6
        out["ring_radius"][str(tr)] = float(rmean[k]) if ok else None
        out["ring_centre_phase_diff_deg"][str(tr)] = float(dph[k]) if ok else None
    out["ring_radius_end"] = float(rmean[-1])
    if contact is None:
        rr = rmean / rmean[0]
        out["outcome"] = ("flies apart" if rr[-1] > 1.5 else "bounded (no contact by T)")
    # for the exactly symmetric case: ring radius when the stop distance (pair separation) is reached
    out["ring_radius_last_sample"] = float(rmean[-1])
    out["t_ring_radius_below"] = {}
    for lev in (14.0, 12.0, 10.0, 8.0, 6.0):
        kk = np.nonzero(rmean < lev)[0]
        out["t_ring_radius_below"][str(lev)] = float(t[kk[0]]) if kk.size else None
    return out


def main():
    p = load_params(PARAMS)
    fsup = json.load(open(FSUP))
    laws = make_laws(p, fsup)
    res = {"params_file": os.path.relpath(PARAMS, ROOT), "fsup_table": os.path.relpath(FSUP, ROOT),
           "pair_law": p, "seed": scan.SEED, "T": T_RUN, "R_E11": R_RUN, "sigmas": SIGMAS,
           "reference_3D": {"ring_radius": {"300": 15.36, "500": 14.48, "690": 13.13},
                            "first_pair_contact_extrapolated": 700},
           "note": ("same standard-normal draws as scan.run_one scaled by sigma; sigma applies to position and phase; "
                    "stop at 2 hw (model B: not below the table end d_min); T_run = 2000"),
           "runs": []}
    # reproduction check: the stored scan run (sigma 1e-3, T 1000, R at the model-B balance) must give its outcome time
    Rb = 6.5253 / p["kappa"]
    chk = {}
    for m in ("A", "B"):
        r = run(Rb, laws[m], p, m, 1e-3, T=1000.0)
        chk[m] = {"R": Rb, "contact_time": r["contact_time"]}
    res["reproduction_check_balance_R_sigma1e-3_T1000"] = {
        "runs": chk, "stored_scan_B_kappaR_6.5253_merger": 440.0226443579101,
        "note": "balance R = kappaR / kappa from the stored candidate row"}
    t0 = time.time()
    for m in ("A", "B"):
        for R in (R_RUN, Rb):
            for s in SIGMAS:
                r = run(R, laws[m], p, m, s)
                r["R_label"] = "E11 ring radius" if R == R_RUN else "model-B balance radius (6.5253/kappa)"
                res["runs"].append(r)
                print("%s R=%.3f s=%g contact=%s asym1pct=%s r300=%s r500=%s r700=%s (%.0f s)" % (
                    m, R, s, r["contact_time"], r["asymmetry_passes_1pct_at"], r["ring_radius"]["300"],
                    r["ring_radius"]["500"], r["ring_radius"]["700"], time.time() - t0), flush=True)
    path = os.path.join(HERE, "e11_seed_scan.json")
    with open(path + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
