"""WP-E post-processing: arrangement result JSON -> per-run series, outcome, plots.

Usage (from the repo root; works on partial results):
  <python> outputs/session2/wpE/postprocess.py [--results DIR] [--criteria FILE] [--out DIR] [--no-plots]
Defaults: results = outputs/session2/wpE/results, criteria = outputs/session2/wpE/criteria.json
(written by the orchestrator; without it the outcome is "not evaluated"), out = outputs/session2/wpE/analysis.
Writes <out>/<run>.json per run, <out>/summary.json and <out>/<run>.png.

Per run (stored series only; nothing is re-run):
  regime        omega/m (lump 0, window omega) and the nonlinear share at t = 0 (standard 1)
  counts        lump count from coordinates per count window: t_mid, count, count_distinct
  distances     tracked: whole-period averaged ball centroids of every lump, interpolated onto lump 0's
                window centres; peaks: positions of the distinct maxima of each count window (highest
                peak per distinct group, at most K). For each: min and max over pairs, nearest-neighbour
                mean, min and max over lumps of the nearest-neighbour distance
  lumps         per-lump N (whole-period average), omega
  phases        theta_k - theta_0 (lump 0's windows) and the differences across the neighbour edges of
                the initial geometry (pairs within 1.01 x the initial nearest spacing), wrapped to (-pi, pi]
  group         E_group / E_group(0) (raw samples and one-period running mean), group centroid drift
  momentum      L about the pattern axis through the group centroid (box and sphere), |P_total|
  outcome       from the criteria file (below)

Criteria file (JSON; the orchestrator fixes the numbers; null = not set):
  {"min_lump_count": int or "K",        # distinct lumps required ("K" = the run's lump number)
   "count_key": "count_distinct",
   "distance_source": "peaks" | "tracked",
   "nn_distance_min": float or null,    # bounds on every lump's nearest-neighbour distance
   "nn_distance_max": float or null,
   "pair_distance_max": float or null,  # optional bound on the largest pair distance
   "relative_to_initial": false,        # true: distance bounds are multiples of the initial nearest spacing
   "eval_from": 15.0,                   # samples (count windows) with t_mid >= eval_from are evaluated
   "hold_fraction": float,              # fraction of evaluated samples at which every condition must hold
   "require_completed": true}
Outcome: "bonded" if the conditions hold on at least hold_fraction of the samples (and the run is
complete when required); otherwise the state at the last sample: "merged" (count below the minimum),
"flew_apart" (nearest-neighbour or pair distance above its maximum), "too_close" (below the minimum),
or "unresolved" (all conditions hold at the end but not on enough samples).
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from swirl.engine.lumps import pair_distances, summarize  # noqa: E402

REQUIRED = ("min_lump_count", "nn_distance_min", "nn_distance_max", "hold_fraction")


def dist_stats(X, L):
    K = X.shape[0]
    if K < 2:
        return None
    D = pair_distances(X, L)
    iu = np.triu_indices(K, 1)
    Dn = D + np.diag(np.full(K, np.inf))
    nn = Dn.min(1)
    return {"min": float(D[iu].min()), "max": float(D[iu].max()), "nn_mean": float(nn.mean()),
            "nn_min": float(nn.min()), "nn_max": float(nn.max())}


def wrap_angle(x):
    return np.angle(np.exp(1j * np.asarray(x, float)))


def running_mean(t, y, span):
    t, y = np.asarray(t), np.asarray(y)
    out = np.empty_like(y)
    for i in range(t.size):
        sel = np.abs(t - t[i]) <= span / 2
        out[i] = y[sel].mean()
    return out


def analyse(res):
    p = res["params"]
    cand = p.get("_candidate", {})
    L = p["L"]
    raw = res["lumps"]["raw"]
    K = len(raw["centers"])
    st = {k: raw[k] for k in ("centers", "series", "point")}
    ts = np.array(raw["series"]["t"])
    t_out = np.arange(0.0, (ts[-1] if ts.size else 0.0) + 1e-9, 1.0)
    av = summarize(st, t_out, p["lumps"]["avg_window"])
    out = {"name": cand.get("name"), "candidate": cand, "completed": res["completed"], "t_end": res["t"],
           "T_design": p["t_end"], "K": K, "N": p["N"], "L": L, "dx": L / p["N"], "linear": p["g4"] == 0 and p["g6"] == 0,
           "start": cand.get("start")}
    lu0 = av["lumps"][0]
    om = [x for x in lu0["omega"] if x is not None]
    out["regime"] = {"omega_over_m_median": float(np.median(om)) / p["m"] if om else None,
                     "nonlinear_share_t0": res["series"]["nonlinear_share"][0]}
    # counts
    cnt = res["lumps"]["counts"]
    out["counts"] = {"t_mid": [c["t_mid"] for c in cnt], "count": [c["count"] for c in cnt],
                     "count_distinct": [c["count_distinct"] for c in cnt]}
    peak_stats = []
    for c in cnt:
        best = {}
        for lab, pk in zip(c["labels"], c["peaks"]):
            if lab not in best or pk["value"] > best[lab]["value"]:
                best[lab] = pk
        pos = sorted(best.values(), key=lambda q: -q["value"])[:K]
        peak_stats.append(dist_stats(np.array([q["pos"] for q in pos]), L) if len(pos) > 1 else None)
    out["distances_peaks"] = {"t_mid": out["counts"]["t_mid"], "stats": peak_stats}
    # tracked distances on lump 0's window centres
    tc0 = np.array([x if x is not None else np.nan for x in lu0["t_center"]])
    ok0 = ~np.isnan(tc0)
    Xs = []
    for k in range(K):
        lk = av["lumps"][k]
        tk = np.array([x for x in lk["t_center"] if x is not None])
        Xk = np.array([x for x in lk["X"] if x is not None])
        if Xk.size == 0 or tk.size < 2:
            Xs = None
            break
        Xs.append(np.stack([np.interp(tc0[ok0], tk, Xk[:, a]) for a in range(3)], 1))
    tr = []
    if Xs is not None and K > 1:
        Xs = np.stack(Xs, 1)
        tr = [dist_stats(Xs[i], L) for i in range(Xs.shape[0])]
    out["distances_tracked"] = {"t": tc0[ok0].tolist(), "stats": tr}
    # per-lump N and omega
    out["lumps"] = [{"t_center": av["lumps"][k]["t_center"], "N": av["lumps"][k]["N"], "omega": av["lumps"][k]["omega"]}
                    for k in range(K)]
    # phases
    rel = [np.zeros(len(t_out))] + [np.array([np.nan if x is None else x for x in r["dtheta"]])
                                    for r in av["relative_phase"]]
    c0 = np.asarray(cand.get("positions", raw["centers"]), float)
    edges = []
    if K > 1:
        D0 = pair_distances(c0, L)
        nn0 = float(np.min(D0[np.triu_indices(K, 1)]))
        edges = [(i, j) for i in range(K) for j in range(i + 1, K) if D0[i, j] <= 1.01 * nn0]
    th_set = np.deg2rad(cand.get("phases_deg", [0.0] * K))
    out["phases"] = {"t_center_lump0": lu0["t_center"],
                     "theta_minus_theta0": [r.tolist() for r in rel[1:]],
                     "edges": edges,
                     "edge_dtheta": [wrap_angle(rel[j] - rel[i]).tolist() for i, j in edges],
                     "edge_dtheta_set": [float(wrap_angle(th_set[j] - th_set[i])) for i, j in edges]}
    # group, momentum
    s = raw["series"]
    if s.get("E_group"):
        Eg = np.array(s["E_group"])
        G = np.array(s["G"])
        P_nom = 2 * np.pi / p["lumps"]["omega_nominal"]
        out["group"] = {"t": ts.tolist(), "E_group_over_0": (Eg / Eg[0]).tolist(),
                        "E_group_over_0_period_mean": (running_mean(ts, Eg, P_nom) / Eg[0]).tolist(),
                        "centroid_drift": np.linalg.norm(G - G[0], axis=1).tolist(),
                        "L_axis_box": s["L_axis_box"], "L_axis_sphere": s["L_axis_sphere"],
                        "P_total_abs": np.linalg.norm(np.array(s["P_total"]), axis=1).tolist()}
    return out


def outcome(r, crit):
    if crit is None:
        return {"outcome": "not evaluated", "reason": "no criteria file"}
    missing = [k for k in REQUIRED if crit.get(k) is None]
    if missing:
        return {"outcome": "not evaluated", "reason": "criteria not set: %s" % missing}
    K = r["K"]
    kmin = K if crit["min_lump_count"] == "K" else int(crit["min_lump_count"])
    scale = 1.0
    if crit.get("relative_to_initial"):
        scale = r["candidate"]["nearest_spacing"]
    dmin, dmax = crit["nn_distance_min"] * scale, crit["nn_distance_max"] * scale
    pmax = crit.get("pair_distance_max")
    pmax = None if pmax is None else pmax * scale
    key = crit.get("count_key", "count_distinct")
    t_mid = np.array(r["counts"]["t_mid"])
    counts = np.array(r["counts"][key])
    if crit.get("distance_source", "peaks") == "peaks":
        stats = r["distances_peaks"]["stats"]
    else:
        tt = np.array(r["distances_tracked"]["t"])
        tr = r["distances_tracked"]["stats"]
        stats = []
        for t in t_mid:
            if tt.size == 0:
                stats.append(None)
                continue
            i = int(np.argmin(np.abs(tt - t)))
            stats.append(tr[i])
    sel = t_mid >= crit.get("eval_from", 15.0)
    rows = []
    for t, c, st in zip(t_mid[sel], counts[sel], [s for s, k in zip(stats, sel) if k]):
        okc = c >= kmin
        okd = st is not None and st["nn_min"] >= dmin
        oku = st is not None and st["nn_max"] <= dmax and (pmax is None or st["max"] <= pmax)
        rows.append({"t": float(t), "count": int(c), "nn_min": st and st["nn_min"], "nn_max": st and st["nn_max"],
                     "pair_max": st and st["max"], "count_ok": bool(okc), "min_ok": bool(okd), "max_ok": bool(oku)})
    if not rows:
        return {"outcome": "not evaluated", "reason": "no samples after eval_from"}
    held = np.mean([x["count_ok"] and x["min_ok"] and x["max_ok"] for x in rows])
    complete_ok = r["completed"] or not crit.get("require_completed", True)
    last = rows[-1]
    if held >= crit["hold_fraction"] and complete_ok:
        oc = "bonded"
    elif not last["count_ok"]:
        oc = "merged"
    elif not last["max_ok"]:
        oc = "flew_apart"
    elif not last["min_ok"]:
        oc = "too_close"
    else:
        oc = "unresolved" if complete_ok else "incomplete"
    return {"outcome": oc, "held_fraction": float(held), "n_samples": len(rows), "min_lump_count": kmin,
            "nn_distance_bounds": [dmin, dmax], "pair_distance_max": pmax, "completed": r["completed"],
            "last_sample": last, "criteria": crit}


def plot(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 2, figsize=(11, 9))
    a = ax[0, 0]
    a.step(r["counts"]["t_mid"], r["counts"]["count_distinct"], where="mid", label="distinct")
    a.step(r["counts"]["t_mid"], r["counts"]["count"], where="mid", ls=":", label="maxima")
    a.set_title("lump count")
    a.legend()
    a = ax[0, 1]
    tr = r["distances_tracked"]
    if tr["stats"]:
        for k in ("nn_min", "nn_mean", "nn_max", "max"):
            a.plot(tr["t"], [s[k] for s in tr["stats"]], label=k)
    pk = [(t, s) for t, s in zip(r["distances_peaks"]["t_mid"], r["distances_peaks"]["stats"]) if s]
    if pk:
        a.plot([t for t, _ in pk], [s["nn_mean"] for _, s in pk], "ko", ms=3, label="nn mean (peaks)")
    a.set_title("distances")
    a.legend(fontsize=7)
    a = ax[1, 0]
    for k, lk in enumerate(r["lumps"]):
        tt = [t for t, n in zip(lk["t_center"], lk["N"]) if n is not None]
        a.plot(tt, [n for n in lk["N"] if n is not None], lw=0.8, label="lump %d" % k)
    a.set_title("N per lump")
    a = ax[1, 1]
    tc = r["phases"]["t_center_lump0"]
    for (i, j), d, d0 in zip(r["phases"]["edges"], r["phases"]["edge_dtheta"], r["phases"]["edge_dtheta_set"]):
        a.plot([t if t is not None else np.nan for t in tc], np.rad2deg(d), lw=0.8)
    a.set_title("neighbour phase differences (deg)")
    if "group" in r:
        g = r["group"]
        ax[2, 0].plot(g["t"], g["E_group_over_0_period_mean"])
        ax[2, 0].set_title("group energy retained (period mean)")
        ax[2, 1].plot(g["t"], g["L_axis_box"], label="L axis (box)")
        ax[2, 1].plot(g["t"], g["L_axis_sphere"], label="L axis (sphere)")
        ax[2, 1].plot(g["t"], g["P_total_abs"], label="|P total|")
        ax[2, 1].legend(fontsize=7)
        ax[2, 1].set_title("angular and linear momentum")
    for a in ax.ravel():
        a.set_xlabel("t")
    fig.suptitle("%s  (K = %d, %s, %s)" % (r["name"], r["K"], r["start"], "linear" if r["linear"] else "nonlinear"))
    fig.tight_layout()
    fig.savefig(path + ".tmp.png", dpi=90)
    plt.close(fig)
    os.replace(path + ".tmp.png", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--criteria", default=os.path.join(HERE, "criteria.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "analysis"))
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    crit = json.load(open(a.criteria)) if os.path.exists(a.criteria) else None
    os.makedirs(a.out, exist_ok=True)
    summary = {"criteria_file": a.criteria if crit is not None else None, "runs": {}}
    for f in sorted(glob.glob(os.path.join(a.results, "*.json"))):
        res = json.load(open(f))
        if "lumps" not in res:
            continue
        r = analyse(res)
        r["outcome"] = outcome(r, crit)
        name = os.path.basename(f)[:-5]
        path = os.path.join(a.out, name + ".json")
        with open(path + ".tmp", "w") as fh:
            json.dump(r, fh)
        os.replace(path + ".tmp", path)
        if not a.no_plots:
            plot(r, os.path.join(a.out, name + ".png"))
        last = lambda key: (r["distances_tracked"]["stats"][-1] or {}).get(key) if r["distances_tracked"]["stats"] else None
        summary["runs"][name] = {
            "K": r["K"], "start": r["start"], "linear": r["linear"], "completed": r["completed"], "t_end": r["t_end"],
            "outcome": r["outcome"]["outcome"], "held_fraction": r["outcome"].get("held_fraction"),
            "count_distinct_last": r["counts"]["count_distinct"][-1] if r["counts"]["count_distinct"] else None,
            "nn_mean_tracked_last": last("nn_mean"), "pair_max_tracked_last": last("max"),
            "E_group_retained_last": r["group"]["E_group_over_0_period_mean"][-1] if "group" in r else None,
            "L_axis_box_last": r["group"]["L_axis_box"][-1] if "group" in r else None,
            "regime": r["regime"]}
        print(name, summary["runs"][name]["outcome"])
    path = os.path.join(a.out, "summary.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(summary, fh, indent=1)
    os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
