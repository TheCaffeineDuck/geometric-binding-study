"""WP-P diagnostic (not part of the model): size of the phase-independent quartic cross term of two unequal lumps
(PRIOR_ART.md N34), U34(d) = -(g4/16) int Phi_1^2 Phi_2^2 d^3x (the time mean of the quartic cross term when the
frequencies differ), as a force F34 = dU34/dd (positive = attraction), next to the model's in-phase first-order
force F0 and its beat-averaged force at held positions (beat_check). Ratios only; the model is unchanged.

    /Users/aaronramos/Desktop/SWIRL/.venv/bin/python -m swirl.arrange.n34_check
writes outputs/session3/wpP/n34_check.json.
"""
import os

import numpy as np

from swirl.family.profile import profile_curve, solve_profile

from . import unequal as un
from .beat_check import held_positions

N0 = 2865.545211117668
HW0 = 3.344038183621855


def U34(p1, p2, d, h=0.05, half=36.0):
    rho = np.arange(0, half, h) + h / 2
    z = np.arange(-half, d + half, h) + h / 2
    R, Z = np.meshgrid(rho, z, indexing="ij")
    F1 = profile_curve(p1, np.sqrt(R ** 2 + Z ** 2))
    F2 = profile_curve(p2, np.sqrt(R ** 2 + (Z - d) ** 2))
    return float(-(un.G4 / 16) * np.sum(2 * np.pi * R * F1 ** 2 * F2 ** 2) * h * h)


def main():
    fam = un.Family.from_table()
    W1 = fam.omega_of_N(N0)
    p1 = solve_profile(W1, m=un.M, g4=un.G4, g6=un.G6, tol=1e-10)
    rows = []
    for ratio in (1.1, 1.5, 2.0):
        W2 = fam.omega_of_N(N0 * ratio)
        p2 = solve_profile(W2, m=un.M, g4=un.G4, g6=un.G6, tol=1e-10)
        for s in (3.6, 4.8, 6.0, 7.2):
            d = s * HW0
            hd = 0.1
            F34 = (U34(p1, p2, d + hd) - U34(p1, p2, d - hd)) / (2 * hd)
            b = held_positions(fam, ratio, d)
            F0, Fb = b["F0_in_phase"], b["mean_force_over_F0"] * b["F0_in_phase"]
            rows.append({"N_ratio": ratio, "spacing_hw": s, "d": d, "omega": [W1, W2], "U34": U34(p1, p2, d),
                         "F34": F34, "F0_in_phase_first_order": F0, "F_beat_averaged_model": Fb,
                         "beat_over_sqrtK": b["beat_over_sqrtK"], "F34_over_F0": F34 / F0,
                         "F34_over_F_beat": F34 / Fb if Fb != 0 else None, "F_beat_over_F0": Fb / F0})
            r = rows[-1]
            print("ratio %.1f  %.1f hw  F34/F0 %.3g  Fbeat/F0 %.3g  F34/Fbeat %s  beat/sqrtK %.2f" % (
                ratio, s, r["F34_over_F0"], r["F_beat_over_F0"],
                "%.3g" % r["F34_over_F_beat"] if r["F34_over_F_beat"] is not None else "-", r["beat_over_sqrtK"]))
    un.dump({"what": "N34 quartic cross term (time mean, unequal frequencies) against the model's forces; diagnostic only",
             "definition": "U34 = -(g4/16) int Phi_1^2 Phi_2^2, F34 = dU34/dd (attraction positive), true profiles, "
                           "axisymmetric quadrature h = 0.05; F0 and F_beat from beat_check.held_positions (positions held, "
                           "20 beats, no boost phase)",
             "rows": rows}, os.path.join(un.ROOT, "outputs", "session3", "wpP", "n34_check.json"))


if __name__ == "__main__":
    main()
