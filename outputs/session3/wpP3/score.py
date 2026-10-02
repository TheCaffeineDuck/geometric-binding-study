"""WP-P3 scoring: the corrected bonded rule (swirl/engine/bonded.py, unchanged) and Block P's labels
(SESSION_03_PREDICTIONS.md, "Decisions fixed before the scan"; the same function as the P2 scan,
swirl.arrange.p2scan.label).

  unit      the in-phase control's 3D contact time (inphase_contact): first time the tracked centroid distance of
            P3_inphase falls below 2 hw0 (Session 2 C7 convention); failing that, the first count window with fewer
            than 2 distinct lumps; failing that, undefined (every candidate is then at best "too short").
  control   CONSTRUCTED from the two isolated single-lump runs of the candidate's lumps (a deviation from a far-apart
            pair, for cost; README): K = 2, count 2 in every window; distance = the start distance plus the two
            lumps' measured centroid drifts; per-lump N from each run; neighbour step = the start phase difference
            plus the integral of the two runs' own omega difference.
  rule      bonded_rule(series_from_result(candidate), constructed control, unit). Clause 5 is on the lab step, which
            equals D_eff for lumps at rest (speeds here are about 1e-4 or less).
  label     unequal lumps: "locked" (step drift x run length and excursion about the drift within 0.25 deg, or 3
            times the control's excursion) or "bounded, beating" (clauses 1-4 hold and the step drifts at the measured
            omega difference of the two lumps within 10%); otherwise the rule's verdict.
  flag      closest tracked approach below 3.6 hw (mean hw of the two lumps): labelled, not counted.

  <python> outputs/session3/wpP3/score.py [results_dir]      -> outputs/session3/wpP3/score.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

HW = {"N100": 3.344038183621855, "N175": 4.080599366623473, "N250": 4.636015304149339}
CANDS = {"P3_c0230": ("N100", "N175"), "P3_c0242": ("N100", "N250")}


def lump_track(res, k):
    lu = res["lumps"]["averages"]["lumps"][k]
    ok = [(t, X, N, w) for t, X, N, w in zip(lu["t_center"], lu["X"], lu["N"], lu["omega"])
          if t is not None and X is not None and N is not None and w is not None]
    t = np.array([o[0] for o in ok])
    _, idx = np.unique(t, return_index=True)
    return (t[idx], np.array([o[1] for o in ok])[idx], np.array([o[2] for o in ok])[idx],
            np.array([o[3] for o in ok])[idx])


def tracked(res):
    t0, X0, _, w0 = lump_track(res, 0)
    t1, X1, _, w1 = lump_track(res, 1)
    X1i = np.stack([np.interp(t0, t1, X1[:, a]) for a in range(3)], 1)
    return t0, np.linalg.norm(X1i - X0, axis=1), w0, np.interp(t0, t1, w1)


def inphase_contact(res):
    t, d, _, _ = tracked(res)
    below = np.where(d < 2 * HW["N100"])[0]
    if below.size:
        return float(t[below[0]])
    bad = [c["t_mid"] for c in res["lumps"]["counts"] if c["count_distinct"] < 2]
    return float(min(bad)) if bad else None


def lock_from_steps(t, U):
    c = np.polyfit(t, U, 1)
    return float(c[0]), float(np.max(np.abs(U - np.polyval(c, t))))


def lock(res):
    rp = res["lumps"]["averages"]["relative_phase"][0]
    t0 = res["lumps"]["averages"]["lumps"][0]["t_center"]
    ok = [(t, d) for t, d in zip(t0, rp["dtheta"]) if t is not None and d is not None]
    t = np.array([o[0] for o in ok])
    _, idx = np.unique(t, return_index=True)
    t = t[idx]
    U = np.rad2deg(np.unwrap(np.array([o[1] for o in ok])[idx]))
    drift, exc = lock_from_steps(t, U)
    _, _, w0, w1 = tracked(res)
    return {"drift_deg_per_t": [drift], "excursion_deg": [exc],
            "expected_beat_deg_per_t": [float(np.rad2deg(np.nanmean(w1 - w0)))],
            "total_change_deg": [float(U[-1] - U[0])], "t_span": float(t[-1] - t[0])}


def constructed_control(iso_a, iso_b, d0, dtheta0_deg):
    """Control series (bonded_rule input) from two isolated runs: lump 0 = iso_a at -d0/2, lump 1 = iso_b at +d0/2."""
    ta, Xa, Na, wa = lump_track(iso_a, 0)
    tb, Xb, Nb, wb = lump_track(iso_b, 0)
    t = ta[(ta >= tb[0]) & (ta <= tb[-1])]
    Xa_ = np.stack([np.interp(t, ta, Xa[:, k]) for k in range(3)], 1) - Xa[0]
    Xb_ = np.stack([np.interp(t, tb, Xb[:, k]) for k in range(3)], 1) - Xb[0]
    d = np.linalg.norm((Xb_ + [d0 / 2, 0, 0]) - (Xa_ - [d0 / 2, 0, 0]), axis=1)
    N = np.stack([np.interp(t, ta, Na), np.interp(t, tb, Nb)], 1)
    dw = np.interp(t, tb, wb) - np.interp(t, ta, wa)
    U = dtheta0_deg + np.rad2deg(np.concatenate([[0.0], np.cumsum(0.5 * (dw[1:] + dw[:-1]) * np.diff(t))]))
    steps = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(U))))[:, None]
    ctl = {"K": 2, "t_end": float(min(iso_a["t"], iso_b["t"])), "t_count": t.tolist(), "count": [2] * t.size,
           "t": t, "nn": np.stack([d, d], 1), "N": N, "steps": steps, "edges": [(0, 1)]}
    drift, exc = lock_from_steps(t, U)
    lk = {"drift_deg_per_t": [drift], "excursion_deg": [exc], "expected_beat_deg_per_t": [float(np.rad2deg(np.mean(dw)))],
          "total_change_deg": [float(U[-1] - U[0])], "t_span": float(t[-1] - t[0])}
    return ctl, lk


def main():
    from swirl.arrange.p2scan import label
    from swirl.engine.bonded import bonded_rule, series_from_result
    rd = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
    load = lambda n: json.load(open(os.path.join(rd, n + ".json")))
    unit = inphase_contact(load("P3_inphase"))
    out = {"what": "WP-P3 scoring (corrected rule, Block P labels; control constructed from isolated runs)",
           "interaction_unit": unit}
    for name, (ka, kb) in CANDS.items():
        r = load(name)
        cand = r["params"]["_candidate"]
        d0 = float(np.linalg.norm(np.subtract(cand["positions"][1], cand["positions"][0])))
        ctl, clk = constructed_control(load("P3_iso_" + ka), load("P3_iso_" + kb), d0,
                                       cand["phases_deg"][1] - cand["phases_deg"][0])
        rule = bonded_rule(series_from_result(r), ctl, unit) if unit else None
        tt, dd, _, _ = tracked(r)
        closest = float(np.min(dd) / (0.5 * (HW[ka] + HW[kb])))
        lk = lock(r)
        lab, info = label(rule, lk, clk, True, closest < 3.6) if rule else ("too short", {"reason": "no interaction unit"})
        out[name] = {"label": lab, "label_info": info, "rule": rule, "closest_approach_hw": closest,
                     "d_start": float(dd[0]), "d_end": float(dd[-1]), "phase_lock": lk, "control_phase_lock": clk,
                     "control": "constructed from P3_iso_%s and P3_iso_%s" % (ka, kb)}
        print(name, lab, "closest %.2f hw" % closest, "d %.2f -> %.2f" % (dd[0], dd[-1]))
    with open(os.path.join(HERE, "score.json.tmp"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    os.replace(os.path.join(HERE, "score.json.tmp"), os.path.join(HERE, "score.json"))


if __name__ == "__main__":
    main()
