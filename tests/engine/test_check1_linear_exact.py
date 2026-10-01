"""Check 1: linear Gaussian vs exact spectral propagation, T = 50, target 1e-4.

Setup: N = 64, L = 50, m = 1, g4 = g6 = 0, no absorber. Gaussian amp 0.5,
R = 2.5, centred off-grid at (0.3, -0.2, 0.1), phase pi/3 with omega 1.1 so
both phi and phi_t are nonzero at t = 0. The exact solution rotates each
discrete Fourier mode of the same initial data at w = sqrt(K2 + m^2) with the
K2 the engine's Laplacian uses.
"""

import numpy as np
import pytest
import scipy.fft as sfft

from swirl.engine.evolver import Evolver
from swirl.engine.initial import build_initial

INIT = {"gaussians": [{"center": [0.3, -0.2, 0.1], "amp": 0.5, "R": 2.5,
                       "phase": np.pi / 3, "omega": 1.1}]}
T = 50.0
TARGET = 1e-4


@pytest.mark.parametrize("dt_label", ["0.05", "dx/8"])
def test_linear_gaussian_matches_exact(dt_label, record):
    ev = Evolver(64, 50.0, 1.0, 0.0, 0.0)
    dt = 0.05 if dt_label == "0.05" else ev.dx / 8
    n = int(round(T / dt))
    dt = T / n
    phi0, pi0 = build_initial(ev.coords, ev.L, INIT)
    ev.set_state(phi0, pi0)
    for _ in range(n):
        ev.step(dt)

    w = np.sqrt(ev.K2 + 1.0)
    ph, vh = sfft.rfftn(phi0), sfft.rfftn(pi0)
    phi_ex = sfft.irfftn(ph * np.cos(w * T) + vh * np.sin(w * T) / w, s=ev.shape)
    pi_ex = sfft.irfftn(-ph * w * np.sin(w * T) + vh * np.cos(w * T), s=ev.shape)
    err_phi = np.linalg.norm(ev.phi - phi_ex) / np.linalg.norm(phi_ex)
    err_pi = np.linalg.norm(ev.pi - pi_ex) / np.linalg.norm(pi_ex)
    record("check1_linear_exact_dt_" + dt_label.replace("/", "_"),
           {"check": "1", "N": 64, "L": 50.0, "dt": dt, "T": T, "init": {"gaussians": [
               {**INIT["gaussians"][0], "phase": float(np.pi / 3)}]},
            "rel_L2_phi": err_phi, "rel_L2_phi_t": err_pi, "target": TARGET})
    assert err_phi < TARGET
    assert err_pi < TARGET
