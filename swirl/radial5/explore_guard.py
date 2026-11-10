"""Exploration (not production): equation R from Gaussian starts beyond the single-frequency
family, and whether the guard trips are converged in the grid spacing.

Part A: production grid, T = 500, bump A in {1, 2, 4} x sigma in {1.5, 3}; pinch A in {1, 2}
x sigma in {2, 4}. Part B: three trips at h = 0.05, 0.025, 0.0125 (dt = h/4, sponge 60 to 90):
a trip time and place that converge with h is a property of the equation, one that moves
with h is a resolution artefact. Run: <venv python> -m swirl.radial5.explore_guard
Writes outputs/radial5/explore_guard.json."""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from swirl.radial5 import evolve as EV  # noqa: E402

KEYS = ("kind", "t", "r", "X", "q", "max_abs_p", "max_abs_fr", "pinch_margin", "energy_so_far", "high_k")


def one(obj, A, s, T, grid):
    r = EV.run("R", obj, EV.gauss_start(obj, A, s), T, grid=grid)
    g = r["guard"]
    ge = r["guard_extremes"]["global_extremes"]
    return dict(obj=obj, A=A, sigma=s, h=grid["h"], T=T, E0=r["E0"], T_end=r["T_end"],
                frac_at=r["frac_at"], T_90=r["T_90"], T_death=r["T_death"],
                rate_settled=r["regime"]["rate_settled_50_150"], rate_end=r["regime"]["rate_end_last100"],
                guard={k: g.get(k) for k in KEYS} if g else None, global_extremes=ge)


def main():
    A_runs = [one(o, A, s, 500.0, EV.GRID) for o, specs in
              (("bump", [(1, 1.5), (2, 1.5), (4, 1.5), (1, 3), (2, 3), (4, 3)]),
               ("pinch", [(1, 2), (2, 2), (1, 4), (2, 4)])) for A, s in specs]
    B_runs = []
    for o, A, s in (("bump", 1, 3), ("bump", 2, 3), ("pinch", 2, 4)):
        for h in (0.05, 0.025, 0.0125):
            g = dict(EV.GRID, h=h, dt=h / 4, R_sponge=60.0, R_box=90.0, r_probe=30.0)
            B_runs.append(one(o, A, s, 60.0, g))
            print(B_runs[-1]["obj"], A, s, h, B_runs[-1]["guard"], flush=True)
    out = dict(description=__doc__, grid_A=EV.GRID, part_A=A_runs, part_B=B_runs,
               note="guard diagnosis values are those of the last accepted state before the failing "
                    "RK stage; the extremes are over every stage")
    p = EV.write_json(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "radial5",
                                   "explore_guard.json"), out)
    print(p)


if __name__ == "__main__":
    main()
