"""WP-P diagnostic: the beat-averaged force between unequal lumps held at fixed positions.

With positions held, the relative phase of two unequal lumps runs at the beat dW (pendulum above its separatrix)
and the in-phase force F0 cos(D) averages to F0 <cos D>. Prediction (second order in the coupling): <cos D> =
K/(2 dW^2) with K = C_N (|w_1| + |w_2|), C_N the N-flow amplitude. This compares the prediction with the model's
own time average. Writes outputs/session3/wpP/beat_average_check.json.

    /Users/aaronramos/Desktop/SWIRL/.venv/bin/python -m swirl.arrange.beat_check
"""
import os

import numpy as np
from scipy.integrate import solve_ivp

from . import unequal as un

N0 = 2865.545211117668


def held_positions(fam, ratio, d, beats=20):
    W1 = fam.omega_of_N(N0)
    W2 = fam.omega_of_N(N0 * ratio)
    m = un.Model(fam, 0.5 * (W1 + W2), boost=False)
    X = np.array([[-d / 2, 0, 0], [d / 2, 0, 0]])
    z0 = m.pack(X, np.zeros((2, 3)), [0.0, 0.0], [W1, W2])
    T = beats * 2 * np.pi / abs(W1 - W2)

    def rhs(t, z):
        out = m.rhs(t, z, 2)
        out[:12] = 0.0
        return out
    t = np.linspace(0, T, 20001)
    sol = solve_ivp(rhs, [0, T], z0, method="DOP853", rtol=1e-11, atol=1e-11, t_eval=t)
    F = np.array([-m.grad(sol.y[:, k], 2)[0] for k in range(sol.y.shape[1])])     # force on lump 1 toward lump 2
    z00 = m.pack(X, np.zeros((2, 3)), [0.0, 0.0], [W1, W2])
    F0 = -m.grad(z00, 2)[0]
    U0 = abs(m.H(z00, 2) - float(np.sum(fam.E(np.array([W1, W2])) - m.omega_ref * fam.N(np.array([W1, W2])))))
    K = U0 * (abs(fam.domega_dN(W1)) + abs(fam.domega_dN(W2)))
    dW = abs(W1 - W2)
    # average over whole beats only
    D = sol.y[12] - sol.y[13]
    nb = int(np.floor(abs(D[-1] - D[0]) / (2 * np.pi)))
    k_end = int(np.argmax(np.abs(D - D[0]) >= 2 * np.pi * nb))
    avg = float(np.trapezoid(F[:k_end + 1], sol.t[:k_end + 1]) / sol.t[k_end])
    Ns = fam.N(sol.y[14:16])
    return {"N_ratio": ratio, "d": d, "omega": [W1, W2], "beat": dW, "sqrtK": float(np.sqrt(K)),
            "beat_over_sqrtK": dW / np.sqrt(K), "F0_in_phase": float(F0), "mean_force_over_F0": avg / float(F0),
            "predicted_K_over_2beat2": K / (2 * dW ** 2), "whole_beats_averaged": nb,
            "N_swing_rel": float(np.max(np.abs(Ns / Ns[:, :1] - 1)))}


def main():
    fam = un.Family.from_table()
    rows = [held_positions(fam, r, d) for r in (1.1, 1.25, 1.5, 2.0) for d in (16.0, 20.0, 24.0)]
    for r in rows:
        print("ratio %.2f d %.0f beat/sqrtK %6.2f  <F>/F0 %.4g  predicted %.4g" % (
            r["N_ratio"], r["d"], r["beat_over_sqrtK"], r["mean_force_over_F0"], r["predicted_K_over_2beat2"]))
    un.dump({"what": "beat-averaged force of unequal lumps at held positions (model, no boost phase: lumps at rest)",
             "rows": rows}, os.path.join(un.ROOT, "outputs", "session3", "wpP", "beat_average_check.json"))


if __name__ == "__main__":
    main()
