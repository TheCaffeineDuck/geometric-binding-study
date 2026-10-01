"""Check 2: Set B energy conservation, no absorber, T = 100, target 1e-5 relative.

Setup (legacy single Gaussian): N = 64, L = 50, m = 1, g4 = 0.30, g6 = 0.055,
phi0 = 0.5, R = 2.5 at the origin, phi_t = 0, dt = 0.05 (legacy dt).
dt = dx/8 = 0.0977 is also measured: RK4 loses energy as T w^6 dt^5 / 72 for a
mode of frequency w, about 2.4e-5 here, so it is held to 3x the target.

Also: the legacy evolver (read from legacy/, not edited) with its sigma = 0.01
"Kreiss-Oliger" term reproduces the 13.4618 -> 13.4637 energy step under its
own energy function, and the step is exactly the omitted stiffness energy
sigma dx^4 |lap phi|^2 / 2.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

from swirl.engine.evolver import Evolver
from swirl.engine.initial import build_initial

SETB = dict(m=1.0, g4=0.30, g6=0.055)
INIT = {"gaussians": [{"center": [0, 0, 0], "amp": 0.5, "R": 2.5}]}
TARGET = 1e-5
LEGACY_E0 = 13.461782355144003  # legacy/outputs/phase1/set_B_baseline.json E_series[0]


def _run_energy(dt, T):
    ev = Evolver(64, 50.0, **SETB)
    phi, pi = build_initial(ev.coords, ev.L, INIT)
    ev.set_state(phi, pi)
    n = int(round(T / dt))
    dt = T / n
    every = max(1, int(round(0.5 / dt)))
    ts, Es = [0.0], [ev.energy()]
    for i in range(1, n + 1):
        ev.step(dt)
        if i % every == 0 or i == n:
            ts.append(i * dt)
            Es.append(ev.energy())
    return dt, np.array(ts), np.array(Es)


@pytest.mark.parametrize("dt_label,limit", [("0.05", TARGET), ("dx/8", 3 * TARGET)])
def test_setB_energy_drift(dt_label, limit, record):
    dt = 0.05 if dt_label == "0.05" else 50.0 / 64 / 8
    dt, ts, Es = _run_energy(dt, 100.0)
    rel = np.abs(Es - Es[0]) / Es[0]
    early = rel[ts <= 10.0].max()
    record("check2_energy_dt_" + dt_label.replace("/", "_"),
           {"check": "2",
            "params": {"N": 64, "L": 50.0, **SETB, "init": INIT, "absorber": None,
                       "kfilter": None, "dt": dt, "T": 100.0, "energy_sample_every": 0.5},
            "E0": Es[0], "E_end": Es[-1], "max_rel_drift": rel.max(),
            "max_rel_drift_t_le_10": early, "end_rel_drift": (Es[-1] - Es[0]) / Es[0],
            "target": TARGET})
    assert abs(Es[0] - LEGACY_E0) < 1e-9 * LEGACY_E0
    assert rel.max() < limit
    assert early < 1e-5  # no legacy-like step (legacy: 1.46e-4 within t = 10)


def _load_legacy_evolver():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "legacy", "engine", "evolver.py")
    spec = importlib.util.spec_from_file_location("legacy_engine_evolver", path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # never write into legacy/
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


def test_legacy_ko_term_makes_the_step(record):
    mod = _load_legacy_evolver()
    ev = mod.SexticEvolver(64, 50.0, 1.0, 0.30, 0.055, dissipation_sigma=0.01)
    phi = 0.5 * np.exp(-(ev.X ** 2 + ev.Y ** 2 + ev.Z ** 2) / (2 * 2.5 ** 2))
    ev.set_initial_conditions(phi, np.zeros_like(phi))

    def stiff():
        ph = np.fft.fftn(ev.phi)
        return 0.5 * ev.sigma * ev.dx ** 4 * np.sum(ev.K4 * np.abs(ph) ** 2) * ev.dx ** 3 / ev.N ** 3

    E0, S0 = ev.compute_energy(), stiff()
    ts, Es, Ss = [], [], []
    for i in range(1, 201):
        ev.step_rk4(0.05)
        if i % 10 == 0:
            ts.append(i * 0.05)
            Es.append(ev.compute_energy())
            Ss.append(stiff())
    Es, Ss = np.array(Es), np.array(Ss)
    late = np.array(ts) >= 5.0
    step = Es[late].mean() - E0
    full_dev = np.abs((Es + Ss) - (E0 + S0)).max()
    record("check2_legacy_step", {
        "params": {"code": "legacy/engine/evolver.py SexticEvolver (read-only import)",
                   "N": 64, "L": 50.0, **SETB, "dissipation_sigma": 0.01, "dt": 0.05, "T": 10.0,
                   "init": INIT, "phi_t0": 0.0, "energy_sample_every": 0.5},
        "E0_legacy": E0, "E_legacy_mean_t5_10": Es[late].mean(),
        "step": step, "max_E_legacy_t_le_10": Es.max(),
        "max_dev_E_plus_stiffness": full_dev,
        "note": "legacy energy omits sigma dx^4 |lap phi|^2/2; with it included, energy is conserved"})
    assert abs(E0 - LEGACY_E0) < 1e-9
    assert 1e-3 < step < 3e-3          # LESSONS 7: 13.46178 -> 13.46374
    assert full_dev < 0.02 * step       # the step is the omitted stiffness energy
