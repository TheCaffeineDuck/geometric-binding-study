"""Stage 4: the two most informative B7 pushes repeated at h = 1/16. 'Most informative' = the largest push
that does not trip the guard, for each of the two starts (gsbend lambda 0.3, jsbend lambda 0.1): the
object still exists there and is moved or not moved, and the next size up trips the guard."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib                                   # noqa: E402

JOBS = os.path.join(lib.TC, "jobs")
out = []
for tag, lam in (("gsbend", 0.3), ("jsbend", 0.1)):
    p = lib.J("B7_%s_l%g_h8" % (tag, lam))["params"]
    cfg = {k: v for k, v in p.items() if k in ("kind", "medium", "law", "dt", "sample_dt", "T", "record_periods", "probes", "probe_max", "profile_range", "start", "push")}
    cfg.update({"kind": "evolve", "h": 0.0625, "N": 13440, "name": "B7_%s_l%g_h16" % (tag, lam), "prio": 4, "est_s": 800})
    out.append(cfg)
json.dump(out, open(os.path.join(JOBS, "B7_h16.json"), "w"), indent=1)
print([c["name"] for c in out])
