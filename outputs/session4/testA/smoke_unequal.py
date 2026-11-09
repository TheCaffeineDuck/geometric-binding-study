"""Test A unequal-pair smoke run: dip + big at r = 20, dtheta = 0, T = 100, and the two matched singles
(same N = 144 box), through the production driver (3 workers). Exercises the multi-source start, the
relative-phase series and the amplitude fits. Needs src_dip and src_big built.
"""
import sys

from swirl.session4.testA import driver as D


def smoke_jobs(T=100.0, r=20.0):
    N = D.pair_box("dip", "big", r)
    base = dict(group="smoke", deps=[], r=r, box=N)
    return [
        dict(name="smoke_uA_r20_p000", kind="pair", dtheta=0, types=["dip", "big"],
             cfg=D.run_cfg([("dip", -r / 2, 0.0), ("big", r / 2, 0.0)], N, T=T), **base),
        dict(name="smoke_uA_single_dip_N%d_x-10.0" % N, kind="single", types=["dip"],
             cfg=D.run_cfg([("dip", -r / 2, 0.0)], N, T=T), **base),
        dict(name="smoke_uA_single_big_N%d_x+10.0" % N, kind="single", types=["big"],
             cfg=D.run_cfg([("big", r / 2, 0.0)], N, T=T), **base),
    ]


if __name__ == "__main__":
    J = smoke_jobs()
    D.atomic_json(J, D.OUT + "/smoke_unequal_jobs.json")
    raise SystemExit(1 if D.run_jobs(J, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 3) else 0)
