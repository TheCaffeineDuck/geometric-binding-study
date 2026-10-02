"""WP-P3 scoring: the corrected bonded rule (swirl/engine/bonded.py, unchanged) and Block P's labels
(SESSION_03_PREDICTIONS.md, "Decisions fixed before the scan"; the same function as the P2 scan, swirl.arrange.p2scan.label).

  unit      the in-phase control's contact time: first time the tracked centroid distance of P3_inphase falls
            below 2 hw0 (Session 2 C7 convention); if it never does, the first count window with fewer than 2
            distinct lumps; if neither, the unit is undefined and every candidate is "too short" at best.
  rule      bonded_rule(series_from_result(candidate), series_from_result(its far control), unit). Clause 5 is
            on the lab step, which equals D_eff for lumps at rest (velocities here are about 1e-4 or less).
  label     unequal lumps: "locked" (each edge's step drift x run length and excursion about the drift within
            0.25 deg, or 3 times the far control's excursion) or "bounded, beating" (clauses 1-4 hold and the step
            drifts at the measured frequency difference of the two lumps, within 10%); otherwise the rule's verdict.
            The relative phase is the engine's whole-period estimate (lumps.summarize), unwrapped.
  flag      closest tracked approach below 3.6 hw (mean of the two lumps' hw): labelled, not counted.

  <python> outputs/session3/wpP3/score.py [results_dir]      -> outputs/session3/wpP3/score.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from swirl.arrange.p2scan import label  # noqa: E402
from swirl.engine.bonded import bonded_rule, series_from_result  # noqa: E402

HW = {"N100": 3.344038183621855, "N175": 4.080599366623473, "N250": 4.636015304149339}
PAIRS = {"P3_c0230": "P3_c0230_far", "P3_c0242": "P3_c0242_far"}


def tracked(res):
    lu = res["lumps"]["averages"]["lumps"]
    tr = []
    for k in range(2):
        ok = [(t, X, w) for t, X, w in zip(lu[k]["t_center"], lu[k]["X"], lu[k]["omega"]) if t is not None and X is not None]
        t = np.array([o[0] for o in ok])
        _, idx = np.unique(t, return_index=True)
        tr.append((t[idx], np.array([o[1] for o in ok])[idx], np.array([np.nan if o[2] is None else o[2] for o in ok])[idx]))
    tt = tr[0][0]
    X1 = np.stack([np.interp(tt, tr[1][0], tr[1][1][:, a]) for a in range(3)], 1)
    w1 = np.interp(tt, tr[1][0], tr[1][2])
    return tt, np.linalg.norm(X1 - tr[0][1], axis=1), tr[0][2], w1


def lock(res):
    rp = res["lumps"]["averages"]["relative_phase"][0]
    t0 = res["lumps"]["averages"]["lumps"][0]["t_center"]
    ok = [(t, d) for t, d in zip(t0, rp["dtheta"]) if t is not None and d is not None]
    t = np.array([o[0] for o in ok])
    _, idx = np.unique(t, return_index=True)
    t = t[idx]
    U = np.rad2deg(np.unwrap(np.array([o[1] for o in ok])[idx]))
    c = np.polyfit(t, U, 1)
    tt, d, w0, w1 = tracked(res)
    return {"drift_deg_per_t": [float(c[0])], "excursion_deg": [float(np.max(np.abs(U - np.polyval(c, t))))],
            "expected_beat_deg_per_t": [float(np.rad2deg(np.nanmean(w1 - w0)))],
            "total_change_deg": [float(U[-1] - U[0])], "t_span": float(t[-1] - t[0])}


def main():
    rd = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
    load = lambda n: json.load(open(os.path.join(rd, n + ".json")))
    out = {"what": "WP-P3 scoring (corrected rule, Block P labels)"}
    ip = load("P3_inphase")
    t, d, _, _ = tracked(ip)
    below = np.where(d < 2 * HW["N100"])[0]
    unit = float(t[below[0]]) if below.size else None
    if unit is None:
        bad = [c["t_mid"] for c in ip["lumps"]["counts"] if c["count_distinct"] < 2]
        unit = float(min(bad)) if bad else None
    out["interaction_unit"] = unit
    keys = {"P3_c0230": ("N100", "N175"), "P3_c0242": ("N100", "N250")}
    for name, far in PAIRS.items():
        r, c = load(name), load(far)
        rule = bonded_rule(series_from_result(r), series_from_result(c), unit) if unit else None
        tt, dd, _, _ = tracked(r)
        hwm = 0.5 * sum(HW[k] for k in keys[name])
        closest = float(np.min(dd) / hwm)
        lk, clk = lock(r), lock(c)
        lab, info = label(rule, lk, clk, True, closest < 3.6) if rule else ("too short", {"reason": "no interaction unit"})
        out[name] = {"label": lab, "label_info": info, "rule": rule, "closest_approach_hw": closest,
                     "d_start": float(dd[0]), "d_end": float(dd[-1]), "phase_lock": lk, "control_phase_lock": clk}
        print(name, lab, "closest %.2f hw" % closest, "d %.2f -> %.2f" % (dd[0], dd[-1]))
    with open(os.path.join(HERE, "score.json.tmp"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    os.replace(os.path.join(HERE, "score.json.tmp"), os.path.join(HERE, "score.json"))


if __name__ == "__main__":
    main()
