import os, sys, json
os.environ["NUMBA_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import numpy as np, lib
lib.RUNS = os.path.join(HERE, "runs")
out = []
refs = {"fd_h8 (B9, dt 0.004)": ("B9_B3_jump_sbend_w2_h8", os.path.join(lib.TC, "runs"))}
def row(name, runs, label):
    lib.RUNS = runs
    lib._cache.clear()
    res = lib.J(name)
    c = lib.curves(res)
    sm = res["summary"]
    d = {"run": label, "file": lib.REL(lib.path(name)), "t_end": sm["t_end"], "cycles_reached": sm["cycles_at_end"], "guard": sm["guard"] is not None}
    for n in (1000, 3000, 5000):
        t = lib.t_at_cycles(c, n)
        d["frac_%d" % n] = lib.val_at_t(c, "ff", t) if t is not None else None
        d["L_half_%d" % n] = lib.val_at_t(c, "lh", t) if t is not None else None
    d["frac_end"] = sm["frac_end"]
    d["rate_late"] = sm["rate_late"]
    d["L_half_late"] = lib.late(c, "lh")
    d["jump_max_late"] = lib.late(c, "jump_max")
    d["jump_distance_from_centre_late"] = float(np.nanmedian(np.abs(c["jump_x"][-max(1, len(c["t"]) // 10):])))
    d["jump_cells_late"] = lib.late(c, "jump_cells")
    d["h"] = res["params"]["h"]
    return d
out.append(row("B9_B3_jump_sbend_w2_h8", os.path.join(lib.TC, "runs"), "fd h=1/8 (B9)"))
for n, lab in (("D2_fd_h4", "fd h=1/4"), ("D2_fd_h16", "fd h=1/16")):
    if os.path.exists(os.path.join(HERE, "runs", n + ".json")):
        out.append(row(n, os.path.join(HERE, "runs"), lab))
json.dump(out, open(os.path.join(HERE, "D2.json"), "w"), indent=1)
for r in out:
    print(r)
