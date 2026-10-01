"""Chunked driver for run 5 / 5r: run 2500 steps (t = 100) at a time, shape diagnostics after each chunk.
Usage: NUMBA_NUM_THREADS=2 python run5_driver.py <config name>   (resumable from checkpoint)"""
import json, os, sys
sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL"); sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL/outputs/step2/diagnostics")
import numpy as np
from swirl.engine.run import run, full_config
from swirl.engine.initial import build_initial
from shape import shape, evolver_from
R = "/Users/aaronramos/Desktop/SWIRL/"
name = sys.argv[1]
cfg = json.load(open(R + "outputs/step2/configs/%s.json" % name))
shp_path = R + "outputs/step2/diagnostics/%s_shape_partial.json" % name
rows = json.load(open(shp_path)) if os.path.exists(shp_path) else []
Rs = 3 * 3.3440381836218  # placeholder, replaced below
Rs = cfg["r_c"][2]
if not rows:
    f = full_config(cfg)
    ev0 = evolver_from(f, *build_initial(np.array([-f["L"] / 2 + f["L"] / f["N"] * i for i in range(f["N"])]), f["L"], f["init"]))
    r = shape(ev0, cfg["center"], Rs); r["t"] = 0.0; rows.append(r)
    json.dump(rows, open(shp_path, "w")); print("t=0", r["a2"], r["a3"], flush=True)
while True:
    res = run(cfg, out_path=R + "outputs/step2/results/%s.json" % name, checkpoint_dir=R + "checkpoints/step2/%s" % name, max_steps=2500, verbose=False)
    t = res["t"]
    if abs(t - rows[-1]["t"]) > 1e-9:
        r = shape(res["_evolver"], cfg["center"], Rs); r["t"] = t; rows.append(r)
        json.dump(rows, open(shp_path + ".tmp", "w")); os.replace(shp_path + ".tmp", shp_path)
        print("t=%.1f a2=%.3e a3=%.3e disp=%.3e Es/Et=%.5f steps/s=%.1f" % (t, r["a2"], r["a3"], r["centroid_disp"], r["E_sphere_over_total"], res["steps_per_second_this_call"]), flush=True)
    if res["completed"]:
        break
print("done", flush=True)
