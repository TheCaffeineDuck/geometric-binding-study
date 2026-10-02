"""Reflag the P2 scan's 3.6 hw validity flag from the dense output (review fix).

The scan took the closest approach from 401 samples, so a run could pass below 3.6 hw between samples (five merged
runs were left "counted"). A run already flagged stays flagged (the dense minimum is at most the sampled one). Every
run not flagged is integrated again with the same start, tolerances and events (the trajectory is the same: the
added non-terminal 3.6 hw event and dense output do not change the steps) and its closest approach recomputed from
the dense output and the event times. Only the closest approach, the flag and the label's "counted" field change;
the outcome, the event time and the rule are checked to be unchanged and kept.

    nice -n 10 env NUMBA_NUM_THREADS=1 /Users/aaronramos/Desktop/SWIRL/.venv/bin/python -m swirl.arrange.p2reflag
"""
import glob
import json
import os
import time

from . import p2scan as p2
from . import unequal as un


def main():
    files = sorted(glob.glob(os.path.join(p2.OUT, "p2_*.json")))
    changed, checked, mism = [], 0, []
    t0 = time.time()
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        if "reflag" in r or r["flag_below_3p6hw"]:
            continue
        new, _ = p2.run_spec(r["spec"])
        checked += 1
        same = new["outcome"] == r["outcome"] and (
            (new["t_event"] is None and r["t_event"] is None)
            or (new["t_event"] is not None and r["t_event"] is not None and abs(new["t_event"] - r["t_event"]) < 1e-6))
        if not same:
            mism.append(r["spec"]["id"])
        r["reflag"] = {"old_closest_approach_hw": r["closest_approach_hw"], "old_flag": r["flag_below_3p6hw"],
                       "new_closest_approach_hw": new["closest_approach_hw"], "new_flag": new["flag_below_3p6hw"],
                       "method": "dense output (20001 points) plus event times, and a non-terminal 3.6 hw event",
                       "outcome_reproduced": same}
        r["closest_approach_hw"] = new["closest_approach_hw"]
        r["flag_below_3p6hw"] = new["flag_below_3p6hw"]
        r["label_info"]["counted"] = not new["flag_below_3p6hw"]
        if new["flag_below_3p6hw"]:
            changed.append({"id": r["spec"]["id"], "outcome": r["outcome"], "label": r["label"],
                            "old": r["reflag"]["old_closest_approach_hw"], "new": new["closest_approach_hw"]})
        un.dump(r, f)
    out = {"checked_unflagged_runs": checked, "newly_flagged": len(changed), "runs": changed,
           "outcome_not_reproduced": mism, "wall_s": time.time() - t0}
    un.dump(out, os.path.join(p2.OUT, "reflag.json"))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}))


if __name__ == "__main__":
    main()
