"""Bisection driver: for one start shape, the amplitude boundary between "the guard does
not trip" (disperses, held, alive at cap) and "the guard trips" (singular, unresolved,
3D ill-posed), to a relative tolerance in the amplitude.

Run (one process; NUMBA_NUM_THREADS=1 is set here):
  <venv python> -m swirl.radial5.boundary5 --model R --obj bump --shape gauss --sigma 3 \
      --A-lo 0.2 --A-hi 1.0 --rtol 0.02 --T 1000
Shapes: gauss (--sigma, --v0 with f_t = v0 times the shape), shell (--r0, --sigma, --v-in),
profile (--omega or --amp; the amplitude is the --scale factor). Writes
outputs/radial5/boundary/<tag>.json after every step (atomic), with each step's amplitude,
class (evolve.CLASSES), T_death, trip time and threshold crossings, and the bracket.
Trips are refined (h/2, h/4) to tell 'singular' from 'unresolved' unless --no-refine.
"""

import argparse
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from swirl.radial5 import evolve as EV  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRIPS = ("singular", "unresolved", "3D ill-posed", "trip, unrefined")


def make_start_fn(model, obj, shape, sigma=None, v0=0.0, r0=None, v_in=0.0, omega=None, amp=None):
    if shape == "gauss":
        return lambda A: EV.gauss_start(obj, A, sigma, v0=v0)
    if shape == "shell":
        return lambda A: EV.shell_start(obj, A, r0, sigma, v_in=v_in)
    if shape == "profile":
        return lambda A: EV.profile_start(model, obj, omega=omega, amplitude=amp, scale=A)
    raise ValueError(shape)


def evaluate(model, obj, start, T, refine=True):
    res = EV.run(model, obj, start, T, stop_at_death=True)
    c = EV.classify(res)
    if c["cls"] == "trip, unrefined" and refine:
        c = EV.refine_trip(model, obj, start, res)
    gx = (res.get("guard_extremes") or {}).get("global_extremes", {})
    return dict(cls=c["cls"], detail={k: v for k, v in c.items() if k not in ("cls", "runs")},
                refine_runs=c.get("runs"), T_death=res.get("T_death"), T_end=res.get("T_end"),
                frac_end=res.get("frac_end"), guard=res.get("guard"),
                crossings=(res.get("guard_extremes") or {}).get("first_time_min_1mX_below"),
                min_one_minus_X=gx.get("min_one_minus_X"), max_abs_p=gx.get("max_abs_p"))


def bisect(model, obj, start_fn, A_lo, A_hi, rtol=0.02, T=1000.0, refine=True, out=None, meta=None):
    steps = []

    def save():
        if out:
            EV.write_json(out, dict(meta or {}, steps=steps, bracket=[A_lo, A_hi],
                                    rel_width=(A_hi - A_lo) / A_lo, done=(A_hi - A_lo) / A_lo <= rtol))

    def ev(A):
        r = evaluate(model, obj, start_fn(A), T, refine)
        r["A"] = float(A)
        steps.append(r)
        save()
        return r["cls"] in TRIPS

    lo_trips = ev(A_lo)
    hi_trips = ev(A_hi)
    if lo_trips or not hi_trips:
        save()
        return steps, (A_lo, A_hi), "bracket not valid: A_lo trips=%s, A_hi trips=%s" % (lo_trips, hi_trips)
    while (A_hi - A_lo) / A_lo > rtol:
        A = float(np.sqrt(A_lo * A_hi))
        if ev(A):
            A_hi = A
        else:
            A_lo = A
        steps[-1]["bracket_after"] = [A_lo, A_hi]
        steps[-1]["T_death_low_side"] = next((s["T_death"] for s in reversed(steps)
                                             if s["A"] == A_lo), None)
        steps[-1]["t_trip_high_side"] = next(((s["guard"] or {}).get("t") for s in reversed(steps)
                                             if s["A"] == A_hi), None)
        save()
    return steps, (A_lo, A_hi), "converged"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["O", "R"], default="R")
    ap.add_argument("--obj", choices=["bump", "pinch"], required=True)
    ap.add_argument("--shape", choices=["gauss", "shell", "profile"], required=True)
    ap.add_argument("--sigma", type=float)
    ap.add_argument("--v0", type=float, default=0.0)
    ap.add_argument("--r0", type=float)
    ap.add_argument("--v-in", type=float, default=0.0)
    ap.add_argument("--omega", type=float)
    ap.add_argument("--amp", type=float)
    ap.add_argument("--A-lo", type=float, required=True)
    ap.add_argument("--A-hi", type=float, required=True)
    ap.add_argument("--rtol", type=float, default=0.02)
    ap.add_argument("--T", type=float, default=1000.0)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--tag")
    a = ap.parse_args(argv)
    fn = make_start_fn(a.model, a.obj, a.shape, a.sigma, a.v0, a.r0, a.v_in, a.omega, a.amp)
    tag = a.tag or "%s_%s_%s_s%s_v%s_r%s_v%s" % (a.model, a.obj, a.shape, a.sigma, a.v0, a.r0, a.v_in)
    out = os.path.join(ROOT, "outputs", "radial5", "boundary", tag + ".json")
    steps, br, status = bisect(a.model, a.obj, fn, a.A_lo, a.A_hi, a.rtol, a.T, not a.no_refine,
                               out, dict(command=vars(a), classes=EV.CLASSES))
    print(status, "bracket", br)
    for s in steps:
        print("A=%.6g %-22s T_death=%s trip=%s" % (s["A"], s["cls"], s["T_death"],
                                                  (s["guard"] or {}).get("t")))
    return steps, br, status


if __name__ == "__main__":
    main()
