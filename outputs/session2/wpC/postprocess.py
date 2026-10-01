"""WP-C post-processing: result JSON -> per-run analysis, batch summary and plots.

Usage: <venv python> outputs/session2/wpC/postprocess.py [--results DIR] [--out DIR]
Defaults: results = outputs/session2/wpC/results, out = outputs/session2/wpC/analysis.
Works on whatever results exist (complete or partial).

Per run (all quantities are whole-period averages from the engine, placed at the
window centre t_center):
  centroid tracks X_k(t); separation d(t) = |X_1 - X_0| on lump 0's times;
  force from the half-space momentum through x = 0: F_early = slope of <P(x<0)> over
    t_center in EARLY_WINDOW, and F(t) as a sliding slope over SLIDE time units;
  fitted early acceleration: quadratic fit of d(t) over t_center in [10, t_fit_end],
    t_fit_end = min(150, T/2, first t with |d - d0| > 1): a_per_lump = -d''/2
    (> 0 means attraction);
  N_k(t), omega_k(t), relative phase dtheta(t) (unwrapped, lump 1 minus lump 0);
  lump counts per count window (raw and distinct);
  predictions: F_asym(d0, dtheta) with the profile A, kappa, and with A, kappa of the
    family at the lumps' measured omega (median over t_center in [30, 80]);
    F_pred_t = F_asym(d(t), dtheta(t)) at the measured separation and phase, and the
    ratio F_slide / F_pred_t (median over EARLY_WINDOW reported for |cos dtheta0| > 0.5);
  outcome by OUTCOME_CRITERIA.
Batch: kappa fitted from F_early at dtheta = 0 and 180 across separations; F_early /
F_early(0) against cos(dtheta) per separation; single-lump drift (sponge control).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

PROFILE = os.path.join(ROOT, "outputs/step2/profiles/profile_beta1.3333_w0.9091.json")
FAMILY = os.path.join(ROOT, "outputs/family/family_beta1.3333.json")
EARLY_WINDOW = (10.0, 60.0)
SLIDE = 40.0
OUTCOME_CRITERIA = {
    "merged": "the last count window has count_distinct == 1, or d(T_end) < 2 hw",
    "separated": "not merged, d(T_end) > d0 + 1.0 and mean dd/dt over the last 100 time units > 0",
    "bounded": "neither: count_distinct == 2 at the end and 2 hw <= d(T_end) <= d0 + 1.0, or d decreasing "
               "but not merged by T_end (reported with the closest approach and the final dd/dt)",
    "note": "bounded at T_end is not 'bonded': bonded needs distances bounded to the end of a run long "
            "against the predicted merge time (SESSION_02_PLAN WP-E)",
}


def load_profile():
    prof = json.load(open(PROFILE))
    fam = json.load(open(FAMILY))
    rows = fam["rows_rescaled"]
    w = np.array([r["omega"] for r in rows])
    A = np.array([r["A"] for r in rows]) / np.sqrt(0.30)
    o = np.argsort(w)
    return prof, (w[o], A[o])


def F_asym(d, dphi, A, kap):
    return 2 * np.pi * A ** 2 * np.cos(dphi) * np.exp(-kap * d) * (kap / d + 1 / d ** 2)


def series(lu, key, comp=None):
    t = lu["t_center"]
    v = lu[key]
    ok = [i for i in range(len(t)) if t[i] is not None and v[i] is not None]
    tt = np.array([t[i] for i in ok])
    vv = np.array([v[i] if comp is None else v[i][comp] for i in ok])
    tt, idx = np.unique(tt, return_index=True)       # consecutive outputs can share a window
    return tt, vv[idx]


def analyse_run(name, res, prof, fam):
    p = res["params"]
    lp = p.get("lumps") or {}
    K = len(lp.get("centers", []))
    hw = prof["half_width"]
    E_l = prof["E"]
    out = {"name": name, "completed": res["completed"], "t_reached": res["t"], "K": K,
           "N": p["N"], "L": p["L"], "dt": p["dt"], "g4": p["g4"], "g6": p["g6"],
           "linear_twin": p["g4"] == 0 and p["g6"] == 0,
           "phases": [r.get("phase", 0.0) for r in p["init"]["radial"]],
           "centers": lp.get("centers"), "steps_per_second": res.get("steps_per_second_this_call"),
           "wall_seconds": res.get("wall_seconds"), "E0": res["summary"]["E0"],
           "absorbed_end": res["summary"]["absorbed_end"],
           "max_rel_dev_E_plus_absorbed": res["summary"]["max_rel_dev_E_plus_absorbed"]}
    av = res["lumps"]["averages"]
    lus = av["lumps"]
    t0, X0 = series(lus[0], "X", 0)
    out["lump_tracks"] = []
    for k in range(K):
        tk = series(lus[k], "X", 0)[0]
        Xk = np.stack([series(lus[k], "X", a)[1] for a in range(3)], 1)
        Nk = series(lus[k], "N")
        Ek = series(lus[k], "E")
        wk = series(lus[k], "omega")
        out["lump_tracks"].append({"t": tk.tolist(), "X": Xk.tolist(), "N_t": Nk[0].tolist(), "N": Nk[1].tolist(),
                                   "E_t": Ek[0].tolist(), "E": Ek[1].tolist(), "omega_t": wk[0].tolist(),
                                   "omega": wk[1].tolist()})
    sel = lambda t: (t >= 30) & (t <= 80)
    om = [float(np.median(np.array(tr["omega"])[sel(np.array(tr["omega_t"]))]))
          if np.any(sel(np.array(tr["omega_t"]))) else None for tr in out["lump_tracks"]]
    out["omega_measured_30_80"] = om
    # E retained near each lump (fraction of the lump's t = 0 ball energy)
    out["E_ball_end_over_start"] = [tr["E"][-1] / tr["E"][0] if tr["E"] else None for tr in out["lump_tracks"]]
    counts = res["lumps"]["counts"]
    out["counts"] = [{"t_mid": c["t_mid"], "count": c["count"], "count_distinct": c.get("count_distinct", c["count"]),
                      "saddle_ratios": [q["saddle_ratio"] for q in c["pairs"]],
                      "peaks_x": [q["pos"][0] for q in c["peaks"]]} for c in counts]
    if K == 1:
        X = np.array(out["lump_tracks"][0]["X"])
        t = np.array(out["lump_tracks"][0]["t"])
        out["single"] = {"x_start": float(X[0, 0]), "x_end": float(X[-1, 0]),
                         "drift_x_end_minus_start": float(X[-1, 0] - X[0, 0]),
                         "max_abs_drift": float(np.max(np.linalg.norm(X - X[0], axis=1))),
                         "fit_acceleration_x": float(2 * np.polyfit(t, X[:, 0], 2)[0]) if t.size > 5 else None,
                         "N_start": out["lump_tracks"][0]["N"][0], "N_end": out["lump_tracks"][0]["N"][-1]}
        rc = p["r_c"]
        Eloc = np.array(res["series"]["E_loc"])
        out["frac_r10_end"] = float(Eloc[-1, 0] / res["summary"]["E0"])
        return out
    # pair
    t1 = np.array(out["lump_tracks"][1]["t"])
    X1 = np.array(out["lump_tracks"][1]["X"])
    X0v = np.array(out["lump_tracks"][0]["X"])
    tt = np.array(out["lump_tracks"][0]["t"])
    X1i = np.stack([np.interp(tt, t1, X1[:, a]) for a in range(3)], 1)
    dsep = np.linalg.norm(X1i - X0v, axis=1)
    d0 = float(np.linalg.norm(np.array(p["lumps"]["centers"][1]) - np.array(p["lumps"]["centers"][0])))
    dphi0 = out["phases"][1] - out["phases"][0]
    out.update(d0=d0, dtheta0=dphi0, sep_t=tt.tolist(), sep=dsep.tolist())
    # early acceleration from d(t)
    far = np.nonzero(np.abs(dsep - d0) > 1.0)[0]
    t_end_fit = min(150.0, 0.5 * out["t_reached"], tt[far[0]] if far.size else np.inf)
    m = (tt >= 10.0) & (tt <= t_end_fit)
    if m.sum() > 5:
        c2 = np.polyfit(tt[m], dsep[m], 2)
        out["fit_d"] = {"window": [10.0, float(t_end_fit)], "d_ddot": float(2 * c2[0]),
                        "a_per_lump": float(-c2[0]), "d_at_10": float(np.polyval(c2, 10.0))}
    # half-space momentum through x = 0 (plane 0)
    if av["P_half"] is not None:
        tp = np.array([x for x in av["t_center_lump0"]], dtype=object)
        ph = av["P_half"]
        ok = [i for i in range(len(ph)) if ph[i] is not None and tp[i] is not None]
        tq = np.array([tp[i] for i in ok], float)
        Pq = np.array([ph[i][0] for i in ok])
        tq, idx = np.unique(tq, return_index=True)
        Pq = Pq[idx]
        m = (tq >= EARLY_WINDOW[0]) & (tq <= EARLY_WINDOW[1])
        out["P_half_t"], out["P_half"] = tq.tolist(), Pq.tolist()
        if m.sum() > 5:
            out["F_early_halfspace"] = float(np.polyfit(tq[m], Pq[m], 1)[0])
            out["a_early_from_F"] = out["F_early_halfspace"] / E_l
        Fs = []
        for tc in tq:
            mm = (tq >= tc - SLIDE / 2) & (tq <= tc + SLIDE / 2)
            Fs.append(float(np.polyfit(tq[mm], Pq[mm], 1)[0]) if mm.sum() > 5 else None)
        out["F_slide"] = Fs
    # relative phase
    rp = av["relative_phase"][0]["dtheta"] if av["relative_phase"] else []
    ok = [i for i in range(len(rp)) if rp[i] is not None and av["t_center_lump0"][i] is not None]
    tr = np.array([av["t_center_lump0"][i] for i in ok])
    dr = np.unwrap(np.array([rp[i] for i in ok]))
    tr, idx = np.unique(tr, return_index=True)
    out["dtheta_t"], out["dtheta"] = tr.tolist(), dr[idx].tolist()
    # predictions
    A, kap = prof["A"], prof["kappa"]
    out["F_pred_profile"] = float(F_asym(d0, dphi0, A, kap))
    # force against the prediction at the instantaneous separation and measured phase
    if "F_slide" in out and len(tr) > 3:
        dq = np.interp(tq, tt, dsep)
        thq = np.interp(tq, tr, dr[idx])
        Fp = F_asym(dq, thq, A, kap)
        out["F_pred_t"] = Fp.tolist()
        Fsl = np.array([np.nan if x is None else x for x in out["F_slide"]])
        ratio = Fsl / Fp
        out["F_slide_over_F_pred_t"] = [None if not np.isfinite(x) else float(x) for x in ratio]
        m = (tq >= EARLY_WINDOW[0]) & (tq <= EARLY_WINDOW[1]) & np.isfinite(ratio)
        if m.sum() > 3 and abs(np.cos(dphi0)) > 0.5:
            out["F_ratio_early_median"] = float(np.median(ratio[m]))
    if all(o is not None for o in om):
        wm = float(np.mean(om))
        Am = float(np.interp(wm, fam[0], fam[1]))
        km = float(np.sqrt(1 - wm ** 2))
        out["F_pred_measured_omega"] = float(F_asym(d0, dphi0, Am, km))
        out["A_kappa_at_measured_omega"] = [Am, km, wm]
    # outcome
    last = out["counts"][-1] if out["counts"] else None
    d_end = float(dsep[-1])
    late = tt >= tt[-1] - 100
    v_late = float(np.polyfit(tt[late], dsep[late], 1)[0]) if late.sum() > 5 else None
    if (last and last["count_distinct"] == 1) or d_end < 2 * hw:
        oc = "merged"
    elif d_end > d0 + 1.0 and v_late is not None and v_late > 0:
        oc = "separated"
    else:
        oc = "bounded"
    out["outcome"] = {"label": oc, "d_end": d_end, "d_min": float(dsep.min()), "d_max": float(dsep.max()),
                      "dd_dt_last100": v_late, "last_count": last, "criteria": OUTCOME_CRITERIA,
                      "run_complete": res["completed"]}
    return out


def batch_fits(runs, prof):
    out = {}
    pairs = [r for r in runs if r["K"] == 2 and not r["linear_twin"] and "F_early_halfspace" in r]
    # kappa from F at dtheta = 0 and 180: ln(|F| d^2 / (kappa0 d + 1)) = c - kappa d
    k0 = prof["kappa"]
    for lab, sel in (("dtheta_0", lambda r: abs(np.cos(r["dtheta0"]) - 1) < 1e-9),
                     ("dtheta_180", lambda r: abs(np.cos(r["dtheta0"]) + 1) < 1e-9)):
        rs = [r for r in pairs if sel(r)]
        if len(rs) >= 2:
            d = np.array([r["d0"] for r in rs])
            F = np.array([abs(r["F_early_halfspace"]) for r in rs])
            y = np.log(F * d ** 2 / (k0 * d + 1))
            c = np.polyfit(d, y, 1)
            out["kappa_fit_" + lab] = {"kappa": float(-c[0]), "kappa_profile": k0, "d": d.tolist(), "F": F.tolist(),
                                       "signs": [float(np.sign(r["F_early_halfspace"])) for r in rs]}
    out["cos_law"] = {}
    for d in sorted({r["d0"] for r in pairs}):
        rs = [r for r in pairs if r["d0"] == d]
        ref = [r for r in rs if abs(np.cos(r["dtheta0"]) - 1) < 1e-9]
        if ref:
            F0 = ref[0]["F_early_halfspace"]
            out["cos_law"][str(d)] = [{"dtheta_deg": float(np.rad2deg(r["dtheta0"])), "F_over_F0": r["F_early_halfspace"] / F0,
                                       "cos": float(np.cos(r["dtheta0"]))} for r in rs]
    out["pair_table"] = [{k: r.get(k) for k in ("name", "d0", "dtheta0", "F_early_halfspace", "F_ratio_early_median",
                                                 "F_pred_profile",
                                                 "F_pred_measured_omega", "a_early_from_F", "fit_d", "outcome",
                                                 "omega_measured_30_80")} for r in pairs]
    out["singles"] = [{k: r.get(k) for k in ("name", "phases", "single", "omega_measured_30_80", "frac_r10_end")}
                      for r in runs if r["K"] == 1]
    return out


def plot_run(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for k, tr in enumerate(r["lump_tracks"]):
        ax[0, 0].plot(tr["t"], np.array(tr["X"])[:, 0], label="lump %d x" % k)
        ax[1, 0].plot(tr["N_t"], tr["N"], label="lump %d N" % k)
    ax[0, 0].set_xlabel("t"); ax[0, 0].set_ylabel("centroid x"); ax[0, 0].legend(frameon=False, fontsize=8)
    ax[1, 0].set_xlabel("t"); ax[1, 0].set_ylabel("N in ball"); ax[1, 0].legend(frameon=False, fontsize=8)
    if r["K"] == 2:
        ax[0, 1].plot(r["sep_t"], r["sep"], color="k")
        ax[0, 1].set_ylabel("separation d(t)")
        if "P_half" in r:
            a2 = ax[0, 1].twinx()
            a2.plot(r["P_half_t"], r["P_half"], color="#eb6834", lw=1)
            a2.set_ylabel("P(x<0)", color="#eb6834")
        ax[1, 1].plot(r["dtheta_t"], np.rad2deg(r["dtheta"]))
        ax[1, 1].set_ylabel("dtheta (deg, unwrapped)")
        ax[0, 1].set_title("%s: outcome %s" % (r["name"], r["outcome"]["label"]), fontsize=9)
    for a in ax.ravel():
        a.set_xlabel("t")
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "analysis"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prof, fam = load_profile()
    runs = []
    for f in sorted(glob.glob(os.path.join(a.results, "*.json"))):
        name = os.path.basename(f)[:-5]
        res = json.load(open(f))
        if "lumps" not in res:
            continue
        r = analyse_run(name, res, prof, fam)
        runs.append(r)
        plot_run(r, os.path.join(a.out, name + ".png"))
        print(name, r.get("outcome", {}).get("label"), "F_early", r.get("F_early_halfspace"),
              "pred", r.get("F_pred_profile"), r.get("single"), flush=True)
    summary = {"description": __doc__, "outcome_criteria": OUTCOME_CRITERIA, "early_window": EARLY_WINDOW,
               "slide": SLIDE, "profile": os.path.relpath(PROFILE, ROOT), "batch": batch_fits(runs, prof),
               "runs": runs}
    path = os.path.join(a.out, "summary.json")
    with open(path + ".tmp", "w") as f:
        json.dump(summary, f)
    os.replace(path + ".tmp", path)
    print(json.dumps(summary["batch"], indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
