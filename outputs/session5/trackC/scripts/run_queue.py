"""WP-C job queue. Runs job lists with at most 3 worker processes, in priority order, through the
unchanged driver (swirl.string.driver). Results go to outputs/session5/trackC/runs/. Skips jobs whose
result already exists; evolve jobs resume from checkpoints (checkpoints/string/, not committed).

    NUMBA_NUM_THREADS=1 caffeinate -i python outputs/session5/trackC/scripts/run_queue.py jobs1.json [...]

Job kinds: the driver's (family, evolve, push) and 'custom' (an evolve whose start the driver cannot
build: a scaled exact state, or a staggered pattern under a Gaussian envelope). A custom job builds
u0 here and hands the start tuple to driver.evolve (an API the driver already has) and runs the linear
twin the same way the driver does. No solver code is added.
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import json
import multiprocessing as mp
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from swirl.string import driver as D          # noqa: E402
from swirl.string.core import band_top        # noqa: E402

RUNS = os.path.join(ROOT, "outputs", "session5", "trackC", "runs")
D.OUT = RUNS


def staggered(S, obj, env_sigma, peak_stretch):
    """Staggered (alternating sign) pattern under a Gaussian envelope of width env_sigma sites:
    bump u_n = (-1)^n E(n), S-bend u_n = (-1)^n E(n + 1/2) (odd about x = -1/2); scaled to the peak
    stretch."""
    n = np.round(S.x).astype(int)
    x0 = -0.5 if obj == "sbend" else 0.0
    E = np.exp(-0.5 * ((S.x - x0) / env_sigma) ** 2)
    u = ((-1.0) ** n) * E
    u = u * (peak_stretch / float(np.max(np.abs(S.stretch(u)))))
    return u, x0


def run_custom(job, log=print):
    name = job["name"]
    cfg = dict(job["cfg"], name=name)
    S = D.build(cfg)
    st = cfg["start"]
    v0 = np.zeros(S.N)
    if job["mode"] == "scale":
        base = dict(cfg, start=job["base_start"])
        u0, x0, om, info = D.build_start(S, base)
        u0 = u0 * job["scale"]
        info = dict(info, scale=job["scale"])
        period = 2 * np.pi / om
    else:
        u0, x0 = staggered(S, st["obj"], st["env_sigma"], st["peak_stretch"])
        rng = np.random.default_rng(0)
        u0 = u0 + 1e-8 * rng.standard_normal(S.N) * (np.abs(S.x - x0) <= 20)
        info = {"noise": 1e-8, "seed": 0, "env_sigma": st["env_sigma"],
                "peak_stretch": st["peak_stretch"]}
        period = D.measure_period(S, u0, v0, cfg["dt"], fallback=band_top(S.medium, S.h))
        if "T_cycles" in job:
            cfg["T"] = job["T_cycles"] * period
            info["period_measured"] = period
    W, lh0 = D.window_halflength(S, u0, v0, x0)
    W = cfg.get("W", W)
    out = D.evolve(cfg, name=name, start=(u0, v0, x0, period, W, info), log=log)
    if cfg.get("twin", True) and cfg["law"] != "lin":
        tw = dict(cfg, law="lin", twin=False)
        out["twin"] = D.evolve(tw, name=name + "_twin",
                               start=(u0, v0, x0, period, W, {"twin_of": name}), log=log)["summary"]
        D.write_json(os.path.join(RUNS, "%s.json" % name), out)
    return out


def is_done(job):
    p = os.path.join(RUNS, job["name"] + ".json")
    if not os.path.exists(p):
        return False
    if job["kind"] in ("family", "push"):
        return True
    cfg = job["cfg"] if job["kind"] == "custom" else job
    if cfg.get("twin", True) and cfg["law"] != "lin":
        try:
            return "twin" in json.load(open(p))
        except Exception:
            return False
    return True


def work(job):
    name = job["name"]
    if is_done(job):
        return name, "skipped (exists)", 0.0
    t0 = time.time()
    err = ""
    for attempt in (1, 2):
        try:
            lg = lambda *a: print("[%s]" % name, *a, flush=True)
            if job["kind"] == "custom":
                run_custom(job, log=lg)
            else:
                D.run_job(job, log=lg)
            return name, "ok", time.time() - t0
        except Exception:
            err = traceback.format_exc()
            print("[%s] attempt %d failed:\n%s" % (name, attempt, err), flush=True)
    D.write_json(os.path.join(RUNS, "FAILED_%s.json" % name), {"job": job, "error": err})
    return name, "FAILED", time.time() - t0


def main(argv):
    jobs = []
    for f in argv[1:]:
        j = json.load(open(f))
        jobs += j if isinstance(j, list) else [j]
    jobs.sort(key=lambda j: (j.get("prio", 9), -j.get("est_s", 0)))
    print("%d jobs, total est %.1f core-h" % (len(jobs), sum(j.get("est_s", 0) for j in jobs) / 3600),
          flush=True)
    with mp.Pool(3, maxtasksperchild=1) as pool:
        for name, st, w in pool.imap_unordered(work, jobs, chunksize=1):
            print("DONE %s: %s (%.0f s)" % (name, st, w), flush=True)


if __name__ == "__main__":
    main(sys.argv)
