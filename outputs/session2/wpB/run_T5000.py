"""WP-B driver: extend the Session 1 primary 3D run (run1_primary) from the T=950 checkpoint to T=5000.
Usage: NUMBA_NUM_THREADS=2 python run_T5000.py --ckpt DIR --out RESULT.json [--stop-t T] [--chunk-steps 6250]
First call copies checkpoints/step2/run1_primary/checkpoint.npz into DIR with ONLY params.t_end changed
(1000 -> 5000) so swirl.engine.run accepts it; the Session 1 checkpoint and result are never touched.
Re-running the same command resumes from DIR. A partial result JSON is written (atomically) after each chunk.
No solver changes: only swirl.engine.run.run is called."""
import argparse, json, os, shutil, sys
sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL")
import numpy as np
from swirl.engine.run import run
R = "/Users/aaronramos/Desktop/SWIRL/"
SRC = R + "checkpoints/step2/run1_primary/checkpoint.npz"
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--t-end", type=float, default=5000.0)
ap.add_argument("--stop-t", type=float, default=None)
ap.add_argument("--chunk-steps", type=int, default=6250)
a = ap.parse_args()
cfg = json.load(open(R + "outputs/step2/configs/run1_primary.json"))
cfg["t_end"] = a.t_end
dst = os.path.join(a.ckpt, "checkpoint.npz")
if not os.path.exists(dst):
    os.makedirs(a.ckpt, exist_ok=True)
    with np.load(SRC, allow_pickle=False) as z:
        phi, pi, meta = z["phi"].copy(), z["pi"].copy(), json.loads(str(z["meta"]))
    assert meta["params"]["t_end"] == 1000.0
    meta["params"]["t_end"] = a.t_end
    meta["extension_note"] = "copied from step2/run1_primary checkpoint (step %d); params.t_end patched to %g" % (meta["step"], a.t_end)
    with open(dst + ".tmp", "wb") as f:
        np.savez(f, phi=phi, pi=pi, meta=np.array(json.dumps(meta)))
    os.replace(dst + ".tmp", dst)
    print("copied checkpoint, step", meta["step"], flush=True)
dt = cfg["dt"]
stop_step = int(round((a.stop_t if a.stop_t else a.t_end) / dt))
while True:
    with np.load(dst, allow_pickle=False) as z:
        step = json.loads(str(z["meta"]))["step"]
    n = min(a.chunk_steps, stop_step - step)
    if n <= 0:
        break
    res = run(cfg, out_path=a.out, checkpoint_dir=a.ckpt, max_steps=n, verbose=False)
    print("t=%.2f steps_done=%d completed=%s steps/s=%.2f E+abs=%.8g omega_last=%s" % (
        res["t"], res["steps_done"], res["completed"], res["steps_per_second_this_call"],
        res["series"]["E_plus_absorbed"][-1], res["summary"]["omega_last"]), flush=True)
    if res["completed"] or res["steps_done"] >= stop_step:
        break
print("done", flush=True)
