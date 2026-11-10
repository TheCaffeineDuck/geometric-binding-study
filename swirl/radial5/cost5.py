"""Cost table: wall time per 10,000 time units on one core at the production grid, for
each combination and its twin, measured over a short run, production grids (R: sponge 200 to 400) (default 200 time units after a
warm-up that compiles the kernels). Run: <venv python> -m swirl.radial5.cost5
Writes outputs/radial5/cost.json."""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import time  # noqa: E402

from swirl.radial5 import evolve as EV  # noqa: E402

STARTS = dict(O_bump=0.909149280617113, O_pinch=0.95, R_bump=0.995, R_pinch=0.99)


def measure(T=200.0):
    rows = []
    for combo, w in STARTS.items():
        model, obj = combo.split("_")
        st = EV.profile_start(model, obj, w)
        for twin in (False, True):
            EV.run(model, obj, st, 2.0, twin=twin)                 # compile
            t0 = time.perf_counter()
            res = EV.run(model, obj, st, T, twin=twin)
            wall = time.perf_counter() - t0
            rows.append(dict(combo=combo, twin=twin, start_omega=w, T_measured=T, wall_s=wall,
                             wall_s_per_10000=wall * 10000.0 / T,
                             hours_per_10000=wall * 10000.0 / T / 3600.0,
                             guard=res.get("guard")))
            print(rows[-1], flush=True)
    return dict(grid=dict(O=EV.GRID_O, R=EV.GRID_R), rec_dt=EV.REC_DT, note="one process, NUMBA_NUM_THREADS=1; includes the "
                "per-record diagnostics (energy inside, spectra every 10 time units)", rows=rows)


if __name__ == "__main__":
    out = measure()
    p = EV.write_json(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "radial5",
                                   "cost.json"), out)
    print(p)
