"""Stage 2 job lists, written after the B1-B4 results exist.
B9: the five longest-lived B1-B4 starts (ranked at h = 1/8 by frac fixed at the reference time, ties by
co-moving frac) rerun to 5,000 cycles of their late measured rate at h = 1/8.
B7b: extra push sizes (the stage-1 set had four of six sizes tripping the guard at once).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib                                   # noqa: E402
import an_smooth as AS                       # noqa: E402

JOBS = os.path.join(lib.TC, "jobs")


def main():
    rows, cv = AS.b_all()
    cand = []
    for tag, cls, obj, par in AS.bstarts():
        if tag in rows and len(rows[tag]) == 3:
            r = rows[tag][1]
            cand.append((r["frac_ref"], r["frac_comoving_ref"], tag, r))
    cand.sort(key=lambda x: (-round(x[0], 4), -x[1]))
    top = cand[:5]
    print("ranking at h = 1/8 (frac_ref, co-moving frac):")
    for c in cand:
        print("  %-34s %.5f %.5f" % (c[2], c[0], c[1]))
    jobs = []
    for _, _, tag, r in top:
        base = lib.J("%s_h8" % tag)["params"]
        rate = r["rate_late"] or 1.0
        cfg = {k: v for k, v in base.items() if k in ("kind", "medium", "law", "h", "N", "dt", "sample_dt", "start", "probes")}
        cfg.update({"name": "B9_%s_h8" % tag, "T": 5000 * 2 * np.pi / rate, "record_periods": 10, "probe_max": 400000,
                    "prio": 4, "est_s": 3000, "kind": "evolve"})
        jobs.append(cfg)
    json.dump(jobs, open(os.path.join(JOBS, "B9.json"), "w"), indent=1)
    print("B9:", [j["name"] for j in jobs])
    # B7b: extra push sizes
    b7 = json.load(open(os.path.join(JOBS, "B7.json")))
    extra = []
    for tag, lams in (("gsbend", (0.03, 0.2, 0.5, 0.7)), ("jsbend", (0.03, 0.15, 0.2, 0.5))):
        base = [j for j in b7 if j["name"] == "B7_%s_l1_h8" % tag][0]
        for lam in lams:
            j = dict(base, name="B7_%s_l%g_h8" % (tag, lam), push={"lambda": lam}, prio=4)
            extra.append(j)
    # extra A7 and nothing else
    json.dump(extra, open(os.path.join(JOBS, "B7b.json"), "w"), indent=1)
    print("B7b:", [j["name"] for j in extra])


if __name__ == "__main__":
    main()
