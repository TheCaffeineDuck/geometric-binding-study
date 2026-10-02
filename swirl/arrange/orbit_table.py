"""WP-P item 4: phase-instability growth against the orbital frequency, for orbiting pairs and rotating rings.

For each spacing and pattern: the rigidly rotating balance (unequal.rotating_ring), the rotating-frame
linearization of the full model, and its unstable modes split by position share (Session 2 convention: below 0.5
"phase", at or above 0.5 "position"). Also the analytic pair numbers, the same pattern without rotation (phase
sector only), and the beat of unequal pairs.

    /Users/aaronramos/Desktop/SWIRL/.venv/bin/python -m swirl.arrange.orbit_table
writes outputs/session3/wpP/instability_vs_orbit.json and .md (about a minute).
"""
import os
import time

import numpy as np

from . import unequal as un

OUT = os.path.join(un.ROOT, "outputs", "session3", "wpP")
N0 = 2865.545211117668
SPACINGS_HW = [3.6, 4.2, 4.8, 6.0, 7.2, 8.0]


def static_phase_rate(mdl, K, d, theta, W):
    """Fastest growth of the phase/N sector of a non-rotating ring with positions held (phases and N only)."""
    X, R = un.ring_positions(K, d)
    z = mdl.pack(X, np.zeros((K, 3)), theta, np.full(K, W))
    n = K
    J = np.zeros((2 * n, 2 * n))
    for k in range(2 * n):
        h = 1e-6 if k < n else 1e-8
        e = np.zeros(z.size)
        e[6 * n + k] = h
        J[:, k] = (mdl.rhs(0, z + e, n)[6 * n:] - mdl.rhs(0, z - e, n)[6 * n:]) / (2 * h)
    lam = np.linalg.eigvals(J)
    rate = float(lam.real.max())
    return rate if rate > 1e-7 else 0.0


def analyse(fam, K, d, theta, W, sense=+1):
    m = un.Model(fam, W)
    r = un.rotating_ring(m, K, d, theta, W, sense)
    stat = static_phase_rate(un.Model(fam, W, boost=False), K, d, theta, W)
    if r is None:
        return {"balance": False, "static_phase_rate": stat}
    m = un.Model(fam, W + r["phidot"])
    r = un.rotating_ring(m, K, d, theta, W, sense)
    J, res = un.rotating_frame_jacobian(m, r["z"], K, r["Omega_r"])
    kap = float(np.sqrt(1 - W * W))
    modes = un.classify_modes(J, K, kap)
    ph = [x["re"] for x in modes if x["position_share"] < 0.5]
    po = [x["re"] for x in modes if x["position_share"] >= 0.5]
    E = float(fam.E(W))
    # effective neighbour step: the boost phase of a rigidly rotating ring shifts every step by the same amount
    shift = float(W * r["p"] * d * np.cos(np.pi / K) / E) if K > 2 else 0.0
    out = {"balance": True, "v": r["v"], "T_orb": r["T_orb"], "Omega_r": r["Omega_r"],
           "effective_step_shift_deg": float(np.degrees(shift)),
           "fixed_point_residual": float(np.max(np.abs(res))), "n_unstable": len(modes),
           "phase_rate": max(ph) if ph else 0.0, "position_rate": max(po) if po else 0.0,
           "static_phase_rate": stat, "modes": modes}
    out["phase_efold"] = 1 / out["phase_rate"] if out["phase_rate"] > 0 else None
    out["position_efold"] = 1 / out["position_rate"] if out["position_rate"] > 0 else None
    out["phase_efold_over_T_orb"] = out["phase_efold"] / r["T_orb"] if out["phase_efold"] else None
    out["position_efold_over_T_orb"] = out["position_efold"] / r["T_orb"] if out["position_efold"] else None
    return out


def pair_analytic(fam, W, d):
    E, A = float(fam.E(W)), float(fam.A(W))
    k = np.sqrt(1 - W * W)
    C = 2 * np.pi * A * A
    F = C * np.exp(-k * d) * (k / d + 1 / d ** 2)
    Om = 2 * np.sqrt(F * d / (2 * E)) / d
    lam_p = np.sqrt(2 * C * np.exp(-k * d) / d * abs(fam.domega_dN(W)))
    rff = k * d + (k + 2 / d) / (k + 1 / d)
    return {"T_orb": 2 * np.pi / Om, "phase_efold": 1 / lam_p, "radial_efold": 1 / (Om * np.sqrt(rff - 3)),
            "phase_rate_over_Omega_orb_squared": abs(fam.domega_dN(W)) * E * d / (k + 1 / d),
            "circular_orbit_energy_over_C": np.exp(-k * d) / (2 * d) * (k * d - 1),
            "escape_speed_over_circular": float(np.sqrt(2 / (k * d + 1)))}


def unequal_pair(fam, W1, ratio, d):
    """Beat of an unequal pair against its frozen-phase circular orbit, and the beat-averaged attraction."""
    W2 = fam.omega_of_N(float(fam.N(W1)) * ratio)
    E1, E2 = float(fam.E(W1)), float(fam.E(W2))
    X = np.array([[[-d / 2, 0, 0], [d / 2, 0, 0]]])
    hc = lambda dd: un.pair_energy((X / d) * dd, np.zeros((1, 2, 3)), np.zeros((1, 2)), np.array([[W1, W2]]), fam)[0]
    F0 = float(hc(d + 1e-30j).imag / 1e-30)
    U0 = float(hc(d).real)
    mu = E1 * E2 / (E1 + E2)
    T_orb = 2 * np.pi / np.sqrt(F0 / (mu * d))
    dW = abs(W1 - W2)
    # phase pendulum about the beat: D'' = K sin D with K = C_N (|w1| + |w2|), C_N = |U0| (the N-flow amplitude)
    Keff = abs(U0) * (abs(fam.domega_dN(W1)) + abs(fam.domega_dN(W2)))
    favg = Keff / (2 * dW ** 2)
    return {"N_ratio": ratio, "omega_1": W1, "omega_2": W2, "beat_period": 2 * np.pi / dW,
            "T_orb_frozen_in_phase": T_orb, "beat_period_over_T_orb": 2 * np.pi / dW / T_orb,
            "pendulum_rate_sqrtK": float(np.sqrt(Keff)), "beat_over_sqrtK": dW / np.sqrt(Keff),
            "beat_averaged_force_fraction": favg,
            "T_orb_beat_averaged": T_orb / np.sqrt(favg) if favg > 0 else None,
            "N_swing_over_N": float(abs(U0) / dW / float(fam.N(W1)))}


def main():
    t0 = time.time()
    fam = un.Family.from_table()
    W0 = fam.omega_of_N(N0)
    hw = float(fam.hw(W0))
    rows = []
    for s in SPACINGS_HW:
        d = s * hw
        rows.append({"config": "pair", "K": 2, "pattern": "in phase", "spacing_hw": s, "d": d,
                     **analyse(fam, 2, d, np.zeros(2), W0), "analytic": pair_analytic(fam, W0, d)})
        for K in (3, 4, 5, 6):
            for name, mm, sense in (("in phase", 0, 1), ("stepped m=+1, rotating with the phase wave", 1, 1),
                                    ("stepped m=+1, rotating against the phase wave", 1, -1)):
                th = 2 * np.pi * mm * np.arange(K) / K
                rows.append({"config": "ring", "K": K, "pattern": name, "spacing_hw": s, "d": d,
                             **analyse(fam, K, d, th, W0, sense)})
    une = [unequal_pair(fam, W0, ratio, s * hw) for ratio in (1.1, 1.5, 2.0) for s in SPACINGS_HW]
    for u, s in zip(une, SPACINGS_HW * 3):
        u["spacing_hw"] = s
    res = {"what": "WP-P item 4: phase-instability e-folding against the circular-orbit period (equal lumps at "
                   "omega %.6f, N %.1f, hw %.4f; beta 4/3); unequal pairs: beat against orbit" % (W0, N0, hw),
           "model": "swirl.arrange.unequal, full family, relativistic, boost phase (default form)",
           "mode_split": "position share kappa^2|dx|^2/(kappa^2|dx|^2+|dphi|^2): < 0.5 phase, >= 0.5 position",
           "rows": rows, "unequal_pairs": une, "wall_s": time.time() - t0,
           "unequal_note": "beat-averaged force fraction K/(2 beat^2) holds only where beat/sqrtK > 2 (rotating phase); below that the relative phase can lock and the estimate does not apply"}
    un.dump(res, os.path.join(OUT, "instability_vs_orbit.json"))
    lines = ["| config | pattern | spacing (hw) | d | v | T_orb | fastest e-fold (any) / T_orb | phase e-fold | "
             "ratio phase/T_orb | position e-fold | ratio position/T_orb | static phase e-fold | "
             "effective step shift (deg) |", "|" + "---|" * 13]
    for r in rows:
        if not r["balance"]:
            lines.append("| %s K=%d | %s | %.1f | %.2f | no balance | | | | | | | %s | |" % (
                r["config"], r["K"], r["pattern"], r["spacing_hw"], r["d"],
                "%.1f" % (1 / r["static_phase_rate"]) if r["static_phase_rate"] > 0 else "stable"))
            continue
        f = lambda x: "%.1f" % x if x else "none"
        g = lambda x: "%.3f" % x if x else "-"
        fast = max(x["re"] for x in r["modes"]) if r["modes"] else None
        lines.append("| %s K=%d | %s | %.1f | %.2f | %.4f | %.0f | %s | %s | %s | %s | %s | %s | %.1f |" % (
            r["config"], r["K"], r["pattern"], r["spacing_hw"], r["d"], r["v"], r["T_orb"],
            g(1 / fast / r["T_orb"]) if fast else "stable", f(r["phase_efold"]),
            g(r["phase_efold_over_T_orb"]), f(r["position_efold"]), g(r["position_efold_over_T_orb"]),
            "%.1f" % (1 / r["static_phase_rate"]) if r["static_phase_rate"] > 0 else "stable",
            r["effective_step_shift_deg"]))
    lines += ["", "| N ratio | spacing (hw) | beat period | T_orb (frozen, in phase) | beat/T_orb | beat/sqrtK | "
              "beat-averaged force fraction | N swing/N |", "|" + "---|" * 8]
    for u in une:
        lines.append("| %.1f | %.1f | %.0f | %.0f | %.3f | %.2f | %.3g | %.3g |" % (
            u["N_ratio"], u["spacing_hw"], u["beat_period"], u["T_orb_frozen_in_phase"], u["beat_period_over_T_orb"],
            u["beat_over_sqrtK"], u["beat_averaged_force_fraction"], u["N_swing_over_N"]))
    path = os.path.join(OUT, "instability_vs_orbit.md")
    with open(path + ".tmp", "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(path + ".tmp", path)
    print("\n".join(lines))
    print("wall %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
