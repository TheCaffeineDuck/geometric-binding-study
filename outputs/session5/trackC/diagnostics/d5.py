"""D5: exact S-bend orbit of C-smooth (sigma 0.5 guess, peak stretch 0.8): Newton, jump, Floquet, at h = 1/4 .. 1/32 (1/64 energy only)."""
import os, sys, json, time
os.environ["NUMBA_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np
from swirl.string import orbits as O
from swirl.string import diag as G
from swirl.string.core import String

hs = [0.25, 0.125, 0.0625, 0.03125, 0.015625] if os.environ.get("D5_L", "16") == "16" else [0.125, 0.0625]
out = []
for h in hs:
    t0 = time.time()
    full = h >= 0.03125
    LBOX = float(os.environ.get("D5_L", "16"))
    r = O.smooth_attempt(h, "G1", "sbend", 0.5, 0.8, floquet=False, L=LBOX)
    row = {"h": h, "N": r["N"], "L": r["L"], "omega": r["omega"], "nq": r["nq"], "converged": r["newton"]["converged"],
           "newton_residual": r["newton"].get("residual"), "newton_iterations": r["newton"].get("iterations"),
           "newton_wall_s": time.time() - t0}
    if r["newton"]["converged"]:
        u = np.asarray(r["profile_u0"])
        S = String("fd", "G1", r["N"], h=h)
        row.update({"E": r["E"], "peak_stretch": r["peak_stretch"], "L_half": r["L_half"], "jacobian_sv_min_over_max": r["jacobian_sv_min_over_max"]})
        d = S.stretch(u)
        xs = S.x + 0.5 * h
        js = G.jump_stats(xs, d, h, "grid")      # xs already bond midpoints
        row.update({"jump_max": js["jump_max"], "jump_x": js["jump_x"], "jump_cells": js["jump_cells"], "stretch_total_variation": js["stretch_total_variation"]})
        if full:
            t1 = time.time()
            Mon, _, _ = O.monodromy(S, u, r["omega"], r["nq"])
            lam, vec = np.linalg.eig(Mon)
            a = np.abs(lam)
            order = np.argsort(-a)
            row["floquet_wall_s"] = time.time() - t1
            row["floquet_max_modulus"] = float(a[order[0]])
            row["growth_per_cycle_ln_max_modulus"] = float(np.log(a[order[0]]))
            row["n_outside_unit_circle_1e-6"] = int(np.sum(np.abs(a - 1) > 1e-6))
            row["n_growing_modulus_gt_1_plus_1e-6"] = int(np.sum(a > 1 + 1e-6))
            row["moduli_top5"] = [float(a[i]) for i in order[:5]]
            # fastest-growing mode: stretch of its displacement part
            N = r["N"]
            mode = vec[:, order[0]]
            mu = mode[:N]
            md = np.real(S.stretch(np.real(mu))) if False else None
            du = np.real(mu)
            dd = (np.roll(du, -1) - du) / h
            e = np.abs(dd) ** 2
            e2 = np.abs((np.roll(mu, -1) - mu) / h) ** 2
            xc = float(np.sum(xs * e2) / np.sum(e2))
            absx = np.abs(xs)
            row["mode_energy_fraction_within_0.5_of_centre"] = float(np.sum(e2[absx <= 0.5]) / np.sum(e2))
            row["mode_energy_fraction_within_1.0_of_centre"] = float(np.sum(e2[absx <= 1.0]) / np.sum(e2))
            row["mode_rms_position"] = float(np.sqrt(np.sum(xs ** 2 * e2) / np.sum(e2)))
            row["mode_eigenvalue"] = [float(lam[order[0]].real), float(lam[order[0]].imag)]
            ft = np.abs(np.fft.rfft(dd * 1.0)) ** 2
            k = 2 * np.pi * np.fft.rfftfreq(N, d=h)
            row["mode_dominant_k"] = float(k[np.argmax(ft[1:]) + 1])
            row["mode_k_over_pi_over_h"] = float(k[np.argmax(ft[1:]) + 1] / (np.pi / h))
            row["mode_first_moment_k_over_pi_over_h"] = float(np.sum(k * ft) / np.sum(ft) / (np.pi / h))
            sym = float(np.linalg.norm(dd + dd[::-1])); anti = float(np.linalg.norm(dd - dd[::-1]))
            row["mode_stretch_parity"] = "even about centre" if sym > anti else "odd about centre"
    out.append(row)
    print(row, flush=True)
    json.dump(out, open(os.path.join(HERE, "D5.json" if os.environ.get("D5_L", "16") == "16" else "D5_L%s.json" % os.environ["D5_L"]), "w"), indent=1)
