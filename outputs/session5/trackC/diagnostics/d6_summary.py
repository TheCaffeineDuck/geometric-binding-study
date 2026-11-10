import os, sys, json
os.environ["NUMBA_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import numpy as np, lib

def analyse(name, runs, label):
    lib.RUNS = runs
    lib._cache.clear()
    res = lib.J(name)
    c = lib.curves(res)
    cyc, ff = c["cyc"], c["ff"]
    loss = 1 - ff
    d = {"run": label, "file": lib.REL(lib.path(name)), "cycles_reached": float(cyc[-1]), "frac_end": float(ff[-1]), "h": res["params"]["h"]}
    i90 = np.where(ff < 0.9)[0]
    d["T_90_cycles"] = float(np.interp(i90[0], range(len(cyc)), cyc)) if i90.size else None
    for thr in (1e-4, 1e-2):
        i = np.where(loss > thr)[0]
        # first sustained crossing: loss stays above thr for the rest of the run
        sus = [k for k in i if np.all(loss[k:] > thr * 0.5)]
        d["onset_cycle_loss_%g" % thr] = float(cyc[sus[0]]) if sus else None
    a, b = d["onset_cycle_loss_0.0001"], d["onset_cycle_loss_0.01"]
    if a and b and b > a:
        d["growth_per_cycle_ln_loss_between_1e-4_and_1e-2"] = float(np.log(100) / (b - a))
    m = cyc > (d["T_90_cycles"] or cyc[-1]) * 0.5
    if m.sum() > 5:
        d["loss_slope_per_cycle_linear_fit_last_half"] = float(np.polyfit(cyc[m], loss[m], 1)[0])
    # loss series sampled
    for n in (500, 1000, 2000, 3000, 4000, 5000):
        t = lib.t_at_cycles(c, n)
        d["loss_at_%d_cycles" % n] = float(1 - lib.val_at_t(c, "ff", t)) if t is not None else None
    return d

out = [analyse("B9_B3_jump_sbend_w1.5_h8", os.path.join(lib.TC, "runs"), "fd h=1/8 (B9)"),
       analyse("D6_jump1.5_fd_h4", os.path.join(HERE, "runs"), "fd h=1/4")]
if os.path.exists(os.path.join(HERE, "runs", "D6_jump1.5_fd_h16.json")):
    out.append(analyse("D6_jump1.5_fd_h16", os.path.join(HERE, "runs"), "fd h=1/16"))
json.dump(out, open(os.path.join(HERE, "D6.json"), "w"), indent=1)
for r in out:
    print(r)
