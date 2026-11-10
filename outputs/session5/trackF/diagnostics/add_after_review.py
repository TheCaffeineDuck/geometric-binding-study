import json, glob, os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from diag_tables import *
RES = 2 * np.pi / 200
rows = []
for t in sorted(os.path.basename(f)[:-5] for pat in ("B_", "C_", "E_") for f in glob.glob(OUT + f"/runs/{pat}*.json")):
    m = metrics(t); rows.append(m)
def letter(m):
    tw_ok = m.get("twin_frac_ref") is not None and m["twin_frac_ref"] <= 0.5
    return (m["frac_cyc_ref"] >= 0.90, tw_ok, m["rate_at_ref"] < 1 - RES)
md = ["| run | cycles used | frac(0) | A: frac cycle-avg at cycle N | A: rate at cycle N | A: twin at same t | A: 2.1 by the letter (frac, twin, rate) | B: retained frac/frac(0) | B: zero-crossing rate | B: retained >= 0.90 and twin <= 0.5 |", "|---|---|---|---|---|---|---|---|---|---|"]
passing = []
for m in rows:
    l = letter(m); allok = all(l)
    if allok and m["cyc_ref"] >= 1000: passing.append(m["tag"])
    tw = "-" if m.get("twin_frac_ref") is None else "%.2e" % m["twin_frac_ref"]
    md.append("| %s | %d | %.3f | %.4f | %.4f | %s | %s (%s) | %.4f | %.4f | %s |" % (m["tag"], m["cyc_ref"], m["frac0"], m["frac_cyc_ref"], m["rate_at_ref"], tw,
              "yes" if allok else "no", ",".join("y" if x else "n" for x in l), m["retained_ref"], m["rate_at_ref"],
              "yes" if (m["retained_ref"] >= 0.9 and l[1]) else "no"))
open(D + "/rule21_columns.md", "w").write("\n".join(md))
json.dump(rows, open(D + "/rule21_columns.json", "w"), indent=1, default=float)
print("\n".join(md)); print("pass by letter, 1000 cycles reached:", passing)
