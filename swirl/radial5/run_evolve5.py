"""Driver: evolve one start and its linear twin, write one JSON (atomic, full parameters).

Examples (from the repo root, one process per run; NUMBA_NUM_THREADS=1 is set here):
  <venv python> -m swirl.radial5.run_evolve5 --model O --obj bump  --start profile:0.909149280617113 --T 10000
  <venv python> -m swirl.radial5.run_evolve5 --model R --obj pinch --start profile:0.99 --T 10000
  <venv python> -m swirl.radial5.run_evolve5 --model R --obj bump  --start gauss:1.5,3.0 --T 2000 --no-twin
  <venv python> -m swirl.radial5.run_evolve5 --model R --obj bump  --start amp:0.95 --scale 1.03 --T 10000
  <venv python> -m swirl.radial5.run_evolve5 --model R --obj pinch --start shell:0.5,10,2,0.3 --T 1000
Starts: profile:<omega> (lowest-amplitude profile at that rate), amp:<x> (profile with
F(0) = x, bump, or F'(0) = x, pinch; reaches branches past a fold in omega), --scale
multiplies either; gauss:<A>,<sigma>[,<v0>] (f_t = v0 times the shape);
shell:<A>,<r0>,<sigma>[,<v_in>] (v_in > 0 moves inward).
Output: outputs/radial5/runs/<tag>.json (default tag from the arguments): run, twin (to the
same T unless --T-twin), and the run's end class (evolve.classify; trips refined at h/2, h/4
unless --no-refine). Checkpoints: checkpoints/radial5/<tag>[_twin].npz, every 500 time units;
rerunning the same command resumes. Grid defaults: h 0.05, dt 0.0125; sponge 100 to 200 for
O, 200 to 400 for R.
"""

import argparse
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from swirl.radial5 import evolve as EV  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_start(model, obj, spec, scale=1.0):
    kind, _, args = spec.partition(":")
    v = [float(x) for x in args.split(",")] if args else []
    if kind == "profile":
        return EV.profile_start(model, obj, omega=v[0], scale=scale)
    if kind == "amp":
        return EV.profile_start(model, obj, amplitude=v[0], scale=scale)
    if kind == "gauss":
        return EV.gauss_start(obj, v[0], v[1], v0=v[2] if len(v) > 2 else 0.0)
    if kind == "shell":
        return EV.shell_start(obj, v[0], v[1], v[2], v_in=v[3] if len(v) > 3 else 0.0)
    raise SystemExit("start must be profile:<omega>, amp:<x>, gauss:<A>,<sigma>[,<v0>] or "
                     "shell:<A>,<r0>,<sigma>[,<v_in>]")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["O", "R"], required=True)
    ap.add_argument("--obj", choices=["bump", "pinch"], required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--T", type=float, default=1000.0)
    ap.add_argument("--T-twin", type=float, default=None, help="twin length (default T)")
    ap.add_argument("--no-twin", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0, help="amplitude factor for profile/amp starts")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--h", type=float, default=None)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--R-sponge", type=float, default=None)
    ap.add_argument("--R-box", type=float, default=None)
    ap.add_argument("--r-probe", type=float, default=None)
    ap.add_argument("--stop-at-death", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "radial5", "runs"))
    a = ap.parse_args(argv)
    grid = EV.default_grid(a.model)
    for k, v in (("h", a.h), ("dt", a.dt), ("R_sponge", a.R_sponge), ("R_box", a.R_box),
                 ("r_probe", a.r_probe)):
        if v is not None:
            grid[k] = v
    tag = a.tag or "%s_%s_%s%s_T%g" % (a.model, a.obj, a.start.replace(":", "").replace(",", "_"),
                                      "" if a.scale == 1.0 else "_x%g" % a.scale, a.T)
    st = parse_start(a.model, a.obj, a.start, a.scale)
    ck = os.path.join(ROOT, "checkpoints", "radial5", tag + ".npz")
    res = EV.run(a.model, a.obj, st, a.T, grid=grid, ckpt=ck, stop_at_death=a.stop_at_death)
    out = dict(tag=tag, command=vars(a), run=res)
    if not a.no_twin:
        Tt = a.T_twin if a.T_twin is not None else a.T
        out["twin"] = EV.run(a.model, a.obj, st, Tt, grid=grid, twin=True,
                             ckpt=os.path.join(ROOT, "checkpoints", "radial5", tag + "_twin.npz"))
    c = EV.classify(res, out.get("twin"))
    if c["cls"] == "trip, unrefined" and not a.no_refine:
        c = EV.refine_trip(a.model, a.obj, st, res)
    out["end_class"] = c
    path = EV.write_json(os.path.join(a.out_dir, tag + ".json"), out)
    s = EV.summary(res)
    print(tag, "->", path)
    print("  run :", {k: s[k] for k in ("T_end", "T_90", "T_death", "frac_end", "frac_at", "guard")})
    print("  rate:", s["regime"])
    print("  end :", {k: v for k, v in c.items() if k != "runs"})
    if "twin" in out:
        print("  twin:", EV.summary(out["twin"])["frac_at"])
    return out


if __name__ == "__main__":
    main()
