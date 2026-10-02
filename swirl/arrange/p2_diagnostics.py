"""WP-P2 diagnostics of the three "bounded, beating" candidates and of the timing miss at 3.6 hw (point model).

D1  the candidates with the N34 phase-independent force added. DIAGNOSTIC VARIANT, NOT THE MODEL:
    U34(d) = -(g4/16) int Phi_1^2 Phi_2^2 for the two starting profiles (held fixed), added to H as a pair
    potential; same start, same T, same control, same rule and labels as the scan.
D2  the candidates without N34, run to 3 times their scan length (15 contact units).
D3  beat period 2 pi/|omega_1 - omega_2| against the equal in-phase contact time, N ratios 1.2, 1.5, 2 at
    3.6, 4.2, 4.8 hw.

    /Users/aaronramos/Desktop/SWIRL/.venv/bin/python -m swirl.arrange.p2_diagnostics
writes outputs/session3/wpP/p2_diagnostics.json.
"""
import json
import os

import numpy as np
from scipy.interpolate import CubicSpline

from swirl.family.profile import solve_profile

from . import p2scan as p2
from . import unequal as un
from .n34_check import U34

CANDIDATES = ["p2_0230", "p2_0234", "p2_0242"]
OUT = os.path.join(un.ROOT, "outputs", "session3", "wpP", "p2_diagnostics.json")


def n34_wrap(spec):
    """Model wrapper adding -dU34/dX to P' (pair only)."""
    F = p2.fam()
    Ws = [F.omega_of_N(p2.N0 * x) for x in spec["N_mult"]]
    pr = [solve_profile(w, m=un.M, g4=un.G4, g6=un.G6, tol=1e-10) for w in Ws]
    dg = np.arange(14.0, 40.01, 0.5)
    ug = np.array([U34(pr[0], pr[1], d) for d in dg])
    spl = CubicSpline(dg, np.log(-ug))
    dspl = spl.derivative()

    def wrap(m, z0):
        class M34(un.Model):
            def rhs(self, t, z, n):
                out = un.Model.rhs(self, t, z, n)
                X = z[:3 * n].reshape(n, 3)
                r = X[1] - X[0]
                d = float(np.linalg.norm(r))
                dU = float(-np.exp(spl(d)) * dspl(d))          # dU34/dd (U34 = -exp(spl))
                f = dU * r / d                                   # force on lump 0 toward lump 1 if dU > 0
                out[3 * n:3 * n + 3] += f          # P of lump 0
                out[3 * n + 3:3 * n + 6] -= f      # P of lump 1
                return out
        w = M34(m.fam, m.omega_ref, m.rel, m.boost, m.frozen)
        return w
    return wrap, {"d_grid": dg.tolist(), "U34": ug.tolist(), "omega": Ws}


def brief(res):
    cl = res["bonded_rule"]["clauses"]
    S = res["series"]
    d = [float(np.linalg.norm(np.asarray(x)[1] - np.asarray(x)[0])) for x in S["X"]]
    return {"outcome": res["outcome"], "label": res["label"], "T": res["T"], "t_event": res["t_event"],
            "closest_approach_hw": res["closest_approach_hw"],
            "d_start": d[0], "d_end": d[-1], "closing_rel": d[-1] / d[0] - 1,
            "trend": [cl["no_distance_trend"]["value"], cl["no_distance_trend"]["limit"], cl["no_distance_trend"]["pass"]],
            "N_held": [cl["N_held"]["value"], cl["N_held"]["limit"], cl["N_held"]["pass"]],
            "distance": [cl["distance_within_tol"]["value"], cl["distance_within_tol"]["limit"], cl["distance_within_tol"]["pass"]],
            "duration": [cl["duration"]["value"], cl["duration"]["limit"], cl["duration"]["pass"]],
            "phase_lock": res["phase_lock"]}


def main():
    specs = {s["id"]: s for s in p2.spec_list()}
    out = {"what": "P2 diagnostics D1-D3 (point model). D1 is a diagnostic variant with the N34 force added; "
                   "it is not the model", "D1_with_N34": {}, "D2_three_times_longer": {}, "D3_beat_vs_contact": []}
    for cid in CANDIDATES:
        spec = specs[cid]
        base = p2.run_with_rule(spec)
        wrap, info = n34_wrap(spec)
        r1 = p2.run_with_rule(spec, model_wrap=wrap)
        out["D1_with_N34"][cid] = {"variant": "DIAGNOSTIC: N34 force added (not the model)", "U34_table": info,
                                   "scan": brief(base), "with_N34": brief(r1)}
        r2 = p2.run_with_rule(spec, T_override=3 * base["T"])
        out["D2_three_times_longer"][cid] = brief(r2)
        print(cid, "base", brief(base)["closing_rel"], "N34", brief(r1)["label"], brief(r1)["closing_rel"],
              brief(r1)["trend"], "| 3T", brief(r2)["label"], brief(r2)["outcome"], brief(r2)["d_end"], brief(r2)["trend"])
    F = p2.fam()
    W0 = F.omega_of_N(p2.N0)
    for ratio in (1.2, 1.5, 2.0):
        W2 = F.omega_of_N(p2.N0 * ratio)
        beat = 2 * np.pi / abs(W0 - W2)
        for s in (3.6, 4.2, 4.8):
            tc = p2.contact_time_equal_in_phase(2, s * p2.HW0)
            out["D3_beat_vs_contact"].append({"N_ratio": ratio, "spacing_hw": s, "beat_period": beat,
                                              "equal_inphase_contact": tc, "contact_over_beat": tc / beat if tc else None})
            print("D3 ratio %.1f %.1f hw beat %.0f contact %s ratio %s" % (ratio, s, beat, tc, tc and tc / beat))
    un.dump(out, OUT)


if __name__ == "__main__":
    main()
