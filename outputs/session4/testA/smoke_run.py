"""Test A smoke run (numerics build): primary pair r = 20, in phase, T = 100, and the matched isolated single
(same N = 144 box, lump at x = -10), through the production driver (pool of 2 workers, NUMBA_NUM_THREADS = 1).
Then: python -m swirl.session4.testA.analysis smoke
"""
import sys

from swirl.session4.testA import driver as D


def smoke_jobs(T=100.0):
    r, N = 20.0, D.pair_box("primary", "primary", 20.0)
    pair = dict(name="smoke_pA_r20_p000", group="smoke", kind="pair", deps=[], r=r, dtheta=0,
                types=["primary", "primary"], box=N,
                cfg=D.run_cfg([("primary", -r / 2, 0.0), ("primary", r / 2, 0.0)], N, T=T))
    single = dict(name="smoke_pA_single_primary_N%d_x-10.0" % N, group="smoke", kind="single", deps=[], r=r,
                  types=["primary"], box=N, cfg=D.run_cfg([("primary", -r / 2, 0.0)], N, T=T))
    return [pair, single]


if __name__ == "__main__":
    J = smoke_jobs()
    D.atomic_json(J, D.OUT + "/smoke_jobs.json")
    raise SystemExit(1 if D.run_jobs(J, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 2) else 0)
