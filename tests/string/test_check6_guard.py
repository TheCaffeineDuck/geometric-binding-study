"""Check 6. Guard: under G1 every force evaluation (every one of the 3 stages of every step) checks
|d| < 1 on every bond; on failure the step is undone and the run stops with a diagnosis
(time, place, stretch, energy so far, last-saved state). Nothing is clipped or softened.

Starts that must trip it:
  (i) |d| = 1.05 on one bond at t = 0 (outside the law's domain);
  (ii) dynamic: two neighbours thrown apart with speed 5 each, energy 25 in one bond. The exact
      flow keeps 1 - d^2 >= exp(-2E) = 2e-22, below double-precision resolution of 1 - d, so any
      finite step must land on |d| >= 1. This trips in the grid, in 'fd' and in 'spectral'.
Control that must not trip: an exact G1 S-bend state at peak stretch 0.95 run for 50 periods.

Failure would look like: no trip on (i) or (ii), a NaN instead of a diagnosis, a diagnosis whose
'last good' state itself violates |d| < 1, or a trip on the control."""

import json
import os

import numpy as np
import pytest

from swirl.string import driver as DR
from swirl.string import orbits as O
from swirl.string.core import GuardTrip, String


def test_guard_trips_at_start():
    S = String("grid", "G1", 32)
    u = np.zeros(32)
    u[17:] = 1.05            # one bond (16 -> 17) stretched to 1.05
    with pytest.raises(GuardTrip) as e:
        S.set_state(u)
    d = e.value.diag
    print("\ncheck6 start trip: t=%g x_bond=%g stretch=%.3f" % (d["t"], d["x_bond"], d["stretch"]))
    assert d["bond"] == 16 and abs(d["stretch"] - 1.05) < 1e-12


@pytest.mark.parametrize("medium,h", [("grid", 1.0), ("fd", 0.125), ("spectral", 0.125)])
def test_guard_trips_dynamically(medium, h, record):
    N = 64
    S = String(medium, "G1", N, h=h)
    c = N // 2
    v = np.zeros(N)
    if medium == "grid":
        v[c], v[c + 1] = -5.0, 5.0
    else:
        v = 5.0 * np.tanh((S.x - 0.5 * h) / (0.5 * h)) * np.exp(-S.x ** 2 / 0.5)
    S.set_state(np.zeros(N), v)
    E0 = S.energy()
    dt = 0.002
    with pytest.raises(GuardTrip) as e:
        S.advance(dt, 200000)
    d = e.value.diag
    ul, vl = d["u_last_good"], d["v_last_good"]
    dl = np.max(np.abs(S.stretch(ul)))
    rel = abs(d["energy_last_good"] / E0 - 1)
    print("\ncheck6 dynamic trip %s: E0=%.3f t=%.6f stage=%d x_bond=%.4f stretch=%.17g; last good "
          "state max|d|=%.17g, energy drift to last good %.2e; string left at last good: %s" % (
              medium, E0, d["t"], d["stage"], d["x_bond"], d["stretch"], dl, rel,
              bool(np.array_equal(S.u, ul) and np.array_equal(S.v, vl))))
    record("check6_dynamic_%s" % medium, {k: v for k, v in d.items() if "last_good" not in k}
           | {"E0": E0, "max_stretch_last_good": dl, "energy_rel_change_last_good": rel})
    assert abs(d["stretch"]) >= 1.0 or not np.isfinite(d["stretch"])
    assert dl < 1.0
    assert np.array_equal(S.u, ul) and np.array_equal(S.v, vl)
    assert d["t_last_good"] < d["t"]


def test_guard_driver_stops_with_diagnosis(tmp_path):
    cfg = {"kind": "evolve", "name": "_test_guard", "medium": "grid", "law": "G1", "N": 64,
           "sponge_len": 0.0, "profile_range": 20.0, "dt": 0.002, "T": 50.0, "start": {"type": "gauss", "obj": "bump", "sigma": 2.0,
                                             "peak_stretch": 0.3}, "twin": False}
    # replace the start velocity by the dynamic trip: patch through a push of huge speed
    cfg["push"] = {"lambda": 60.0}
    r = DR.evolve(cfg, log=lambda *a: None)
    g = r["summary"]["guard"]
    print("\ncheck6 driver: guard diagnosis %s" % g)
    assert g is not None and abs(g["stretch"]) >= 1.0
    rr = g["rerun_half_dt"]
    assert rr["label"].startswith("numerical") or rr["label"].startswith("persists")
    assert os.path.exists(os.path.join(DR.CKPT, "_test_guard_guard_state.npz"))
    for f in ("_test_guard.json", "_test_guard_series.npz"):
        os.remove(os.path.join(DR.OUT, f))
    os.remove(os.path.join(DR.CKPT, "_test_guard_guard_state.npz"))


def test_guard_no_false_trip_near_pole():
    N = 61
    S = String("grid", "G1", N)
    om = 3.0
    u = None
    # continue from 2.3 to a high-stretch exact state
    for omega in np.linspace(2.3, om, 15):
        g = u if u is not None else O.grid_pattern(N, "sbend", O.rwa_amplitude("sbend", "G1", 0, omega),
                                                   O.kappa_grid(omega))
        nq = O.choose_nq(S, g, omega)
        u, info = O.newton(S, g, omega, nq)
        assert info["converged"]
    dmax = np.max(np.abs(S.stretch(u)))
    S.set_state(u)
    T = 2 * np.pi / om
    S.advance(T / (4 * nq), 4 * nq * 50)
    print("\ncheck6 control: exact S-bend omega=%.2f peak stretch %.4f ran 50 periods, no trip; "
          "return error %.1e" % (om, dmax, np.max(np.abs(S.u - u))))
    assert dmax > 0.9
