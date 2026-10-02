"""Summary of the P2 scan: counts by family and class, every candidate with its parameters, the Block P numbers.

    /Users/aaronramos/Desktop/SWIRL/.venv/bin/python -m swirl.arrange.p2summary
writes outputs/session3/wpP/p2/summary.json.
"""
import collections
import glob
import json
import os

import numpy as np

from . import unequal as un

P2 = os.path.join(un.ROOT, "outputs", "session3", "wpP", "p2")


def group(spec):
    if spec["family"] == "static":
        return "unequal static pairs" if spec["K"] == 2 else "unequal static rings"
    return {"equal": "equal orbits", "pair": "unequal orbiting pairs", "alternating": "alternating-size rings"}[spec["size_kind"]]


def main():
    R = []
    errors = []
    for f in sorted(glob.glob(os.path.join(P2, "p2_*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        (errors if "error" in r else R).append(r)
    counts = collections.defaultdict(collections.Counter)
    labels = collections.defaultdict(collections.Counter)
    labels_valid = collections.defaultdict(collections.Counter)
    cands = []
    for r in R:
        g = group(r["spec"])
        counts[g][r["outcome"]] += 1
        labels[g][r["label"]] += 1
        if not r["flag_below_3p6hw"]:
            labels_valid[g][r["label"]] += 1
        if r["label"] in ("bonded", "bounded, beating", "too short", "no interaction"):
            s = r["spec"]
            cands.append({"id": s["id"], "group": g, "label": r["label"], "outcome": r["outcome"], "K": s["K"],
                          "spacing_hw": s["spacing_hw"], "size_kind": s["size_kind"], "N_ratio": s["ratio"],
                          "pattern": s["pattern"], "speed_frac": s["speed_frac"], "T": r["T"],
                          "closest_approach_hw": r["closest_approach_hw"], "flag_below_3p6hw": r["flag_below_3p6hw"],
                          "failed_clauses": [k for k, v in r["bonded_rule"]["clauses"].items() if not v["pass"]],
                          "label_info": r["label_info"],
                          "mean_pair_distance_change_rel": r.get("mean_pair_distance_change_rel")})
    # equal orbits at 3.6 to 6 hw: ended (merged, flew apart, left) within 2 reference orbital periods
    eo = [r for r in R if group(r["spec"]) == "equal orbits" and r["spec"]["spacing_hw"] <= 6.0 + 1e-9]
    ended2 = [r for r in eo if r["t_event"] is not None and r["reference_orbit"]
              and r["t_event"] <= 2 * r["reference_orbit"]["T_orb"]]
    by_s = {}
    for s in sorted(set(r["spec"]["spacing_hw"] for r in eo)):
        sub = [r for r in eo if r["spec"]["spacing_hw"] == s]
        by_s[str(s)] = sum(r in ended2 for r in sub) / len(sub)
    # unequal static pairs: contact time over the equal in-phase contact time, by N ratio
    sp = [r for r in R if group(r["spec"]) == "unequal static pairs"]
    ratio_tab = {}
    for q in sorted(set(r["spec"]["ratio"] for r in sp)):
        sub = [r for r in sp if r["spec"]["ratio"] == q]
        m = [r["t_event"] / r["contact_time_unit"] for r in sub if r["outcome"] == "merged" and r["contact_time_unit"]]
        m0 = [r["t_event"] / r["contact_time_unit"] for r in sub if r["outcome"] == "merged" and r["contact_time_unit"]
              and r["spec"]["theta_deg"][1] == 0.0]
        ratio_tab[str(q)] = {"runs": len(sub), "merged": len(m),
                             "outcomes": dict(collections.Counter(r["outcome"] for r in sub)),
                             "contact_over_equal_inphase_all_phases": {
                                 "median": float(np.median(m)) if m else None, "min": float(min(m)) if m else None,
                                 "max": float(max(m)) if m else None},
                             "contact_over_equal_inphase_start_dtheta0": {
                                 "values_by_spacing": {str(r["spec"]["spacing_hw"]): r["t_event"] / r["contact_time_unit"]
                                                       for r in sub if r["spec"]["theta_deg"][1] == 0.0
                                                       and r["outcome"] == "merged" and r["contact_time_unit"]},
                                 "not_merged_spacings": [r["spec"]["spacing_hw"] for r in sub
                                                         if r["spec"]["theta_deg"][1] == 0.0 and r["outcome"] != "merged"]}}
        del m0
    out = {"what": "WP-P2 point-model scan, Block P (SESSION_03_PREDICTIONS.md)", "n_runs": len(R), "n_errors": len(errors),
           "errors": [{"id": e["spec"]["id"], "error": e["error"]} for e in errors],
           "settings": {"T_cap": 10000.0, "far_control_d": 45.0, "perturbation": "1e-3 positions and phases, seed 20261002 + run",
                        "boost": "default (co-moving exact, counter-moving first order)", "rtol": 1e-10,
                        "phase_clause": "D_eff for every run; unequal: locked / beating (p2scan.label)"},
           "outcome_counts": {g: dict(c) for g, c in counts.items()},
           "label_counts_all": {g: dict(c) for g, c in labels.items()},
           "label_counts_not_flagged": {g: dict(c) for g, c in labels_valid.items()},
           "flagged_below_3p6hw": {g: sum(1 for r in R if group(r["spec"]) == g and r["flag_below_3p6hw"]) for g in counts},
           "candidates": cands,
           "equal_orbits_3p6_to_6hw_ended_within_2_T_orb": {"fraction": len(ended2) / len(eo) if eo else None,
                                                             "n": len(eo), "by_spacing_hw": by_s},
           "unequal_static_pairs_contact_over_equal_inphase": ratio_tab,
           "wall_s_total": float(sum(r.get("wall_total_s", 0) for r in R))}
    un.dump(out, os.path.join(P2, "summary.json"))
    print(json.dumps({k: v for k, v in out.items() if k not in ("candidates", "unequal_static_pairs_contact_over_equal_inphase")}, indent=1))
    print("candidates:", len(cands))


if __name__ == "__main__":
    main()
