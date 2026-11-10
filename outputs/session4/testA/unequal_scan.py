"""Radial scan of beta = 4/3 family members for the Test A unequal pair (plan A1).

For each start omega: radial run (swirl.family.wpa_run, Session 1-3 grid) to T = 1500 from the
single-frequency profile; settled omega (median whole-period omega over (t - 50, t]) at t = 200 ... 1500;
drift_t = |omega(t) - omega(t - 100)| (the snapshot settling guard compares whole-period omega over 20
time units 100 apart, tolerance 1e-4); frac = E(3 hw)/E0; loss rate (frac(300) - frac(1500))/1200.

Usage: python unequal_scan.py OUT.json omega [omega ...]
At most 3 processes (NUMBA_NUM_THREADS=1 is set before numba is imported in each worker).
"""
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

G4 = 0.30
TS = (200, 300, 400, 500, 1000, 1400, 1500)


def _init():
    os.environ["NUMBA_NUM_THREADS"] = "1"


def one(om):
    from swirl.family.wpa_run import profile, run
    t0 = time.time()
    prof = profile(4 / 3, om)
    r = run(4 / 3, om, 1500.0, stop_at_death=False, prof=prof)
    t, Ein, tw, w = r["series"]["t"], r["series"]["E_in"], r["omega_t"], r["omega"]

    def win(T, half=50.0):
        s = (tw > T - half) & (tw <= T)
        return float(np.median(w[s])) if s.any() else None

    def frac(T):
        return float(Ein[np.argmin(abs(t - T)), 0] / r["E0"])

    return dict(omega_profile=om, N=prof["N"] / G4, E=prof["E"] / G4, A=prof["A"] / np.sqrt(G4),
                kappa=prof["kappa"], hw=prof["half_width"],
                omega_settled={str(T): win(T) for T in TS},
                drift={str(T): abs(win(T) - win(T - 100)) for T in (300, 500, 1500)},
                frac={str(T): frac(T) for T in TS}, loss_rate_300_1500=(frac(300) - frac(1500)) / 1200.0,
                T_90=r["T_90"], T_death=r["T_death"], wall_s=time.time() - t0)


if __name__ == "__main__":
    out, oms = sys.argv[1], [float(a) for a in sys.argv[2:]]
    with get_context("spawn").Pool(min(3, len(oms)), initializer=_init) as p:
        rows = p.map(one, oms)
    res = dict(what="radial scan, beta 4/3, settling and loss of single-frequency starts", T=1500.0,
               grid="swirl.family.wpa_run.grid_for(4/3): dr 0.05, dt 0.0125, sponge 100 to 200",
               N_units="physical (rescaled N / g4)", rows=rows)
    with open(out + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(out + ".tmp", out)
    for q in rows:
        print("%.4f N=%7.1f hw=%.3f k=%.4f w300=%.6f w1500=%.6f d300=%.1e d500=%.1e d1500=%.1e frac1500=%.4f rate=%.2e"
              % (q["omega_profile"], q["N"], q["hw"], q["kappa"], q["omega_settled"]["300"],
                 q["omega_settled"]["1500"], q["drift"]["300"], q["drift"]["500"], q["drift"]["1500"],
                 q["frac"]["1500"], q["loss_rate_300_1500"]))
