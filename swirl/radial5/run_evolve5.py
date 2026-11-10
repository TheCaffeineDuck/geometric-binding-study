"""Driver: evolve one start and its linear twin, write one JSON (atomic, full parameters).

Examples (from the repo root, one process per run; NUMBA_NUM_THREADS=1 is set here):
  <venv python> -m swirl.radial5.run_evolve5 --model O --obj bump  --start profile:0.909149280617113 --T 10000
  <venv python> -m swirl.radial5.run_evolve5 --model R --obj pinch --start profile:0.99 --T 10000
  <venv python> -m swirl.radial5.run_evolve5 --model R --obj bump  --start gauss:1.5,3.0 --T 2000 --no-twin
Output: outputs/radial5/runs/<tag>.json (default tag from the arguments). Checkpoints:
checkpoints/radial5/<tag>[_twin].npz, every 500 time units; rerunning the same command
resumes. Grid options default to the production grid (h 0.05, dt 0.0125, sponge 100 to 200).
"""

import argparse
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from swirl.radial5 import evolve as EV  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_start(model, obj, spec):
    kind, _, args = spec.partition(":")
    if kind == "profile":
        return EV.profile_start(model, obj, float(args))
    if kind == "gauss":
        A, sigma = (float(x) for x in args.split(","))
        return EV.gauss_start(obj, A, sigma)
    raise SystemExit("start must be profile:<omega> or gauss:<A>,<sigma>")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["O", "R"], required=True)
    ap.add_argument("--obj", choices=["bump", "pinch"], required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--T", type=float, default=1000.0)
    ap.add_argument("--T-twin", type=float, default=None, help="twin length (default min(T, 2000))")
    ap.add_argument("--no-twin", action="store_true")
    ap.add_argument("--h", type=float, default=EV.GRID["h"])
    ap.add_argument("--dt", type=float, default=EV.GRID["dt"])
    ap.add_argument("--R-sponge", type=float, default=EV.GRID["R_sponge"])
    ap.add_argument("--R-box", type=float, default=EV.GRID["R_box"])
    ap.add_argument("--r-probe", type=float, default=EV.GRID["r_probe"])
    ap.add_argument("--stop-at-death", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "radial5", "runs"))
    a = ap.parse_args(argv)
    grid = dict(EV.GRID, h=a.h, dt=a.dt, R_sponge=a.R_sponge, R_box=a.R_box, r_probe=a.r_probe)
    tag = a.tag or "%s_%s_%s_T%g" % (a.model, a.obj, a.start.replace(":", "").replace(",", "_"), a.T)
    st = parse_start(a.model, a.obj, a.start)
    ck = os.path.join(ROOT, "checkpoints", "radial5", tag + ".npz")
    res = EV.run(a.model, a.obj, st, a.T, grid=grid, ckpt=ck, stop_at_death=a.stop_at_death)
    out = dict(tag=tag, command=vars(a), run=res)
    if not a.no_twin:
        Tt = a.T_twin if a.T_twin is not None else min(a.T, 2000.0)
        out["twin"] = EV.run(a.model, a.obj, st, Tt, grid=grid, twin=True,
                             ckpt=os.path.join(ROOT, "checkpoints", "radial5", tag + "_twin.npz"))
    path = EV.write_json(os.path.join(a.out_dir, tag + ".json"), out)
    s = EV.summary(res)
    print(tag, "->", path)
    print("  run :", {k: s[k] for k in ("T_end", "T_90", "T_death", "frac_end", "frac_at", "guard")})
    print("  rate:", s["regime"])
    if "twin" in out:
        print("  twin:", EV.summary(out["twin"])["frac_at"])
    return out


if __name__ == "__main__":
    main()
