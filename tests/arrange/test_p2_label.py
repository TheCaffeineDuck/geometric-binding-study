"""Driver tests for swirl.arrange.p2scan.label (Block P labels) on synthetic series through the corrected rule."""
import numpy as np
import pytest

from swirl.arrange.p2scan import label
from swirl.engine.bonded import bonded_rule

UNIT = 100.0


def series(T=1000.0, nn_jump=0.002, dN=0.0, step_rate=0.0, n=401):
    """Two lumps; the distance ratio jumps by nn_jump before t = 10 and stays; lump 0 gains dN (relative) linearly;
    the neighbour step runs at step_rate degrees per unit time."""
    t = np.linspace(0, T, n)
    nn = np.where(t < 10, 1.0, 1.0 + nn_jump)[:, None] * np.array([[20.0, 20.0]])
    N = np.stack([1000.0 * (1 + dN * t / T), 1000.0 * np.ones_like(t)], 1)
    U = 30.0 + step_rate * t
    steps = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(U))))[:, None]
    run = {"K": 2, "t_end": T, "t_count": t.tolist(), "count": [2] * n, "t": t, "nn": nn, "N": N, "steps": steps}
    c = np.polyfit(t, U, 1)
    lock = {"drift_deg_per_t": [float(c[0])], "excursion_deg": [float(np.max(np.abs(U - np.polyval(c, t))))],
            "expected_beat_deg_per_t": [step_rate], "total_change_deg": [float(U[-1] - U[0])], "t_span": T}
    return run, lock


@pytest.fixture
def control():
    return series(nn_jump=0.0)


def lab(run_lock, control, unequal, flagged=False):
    run, lk = run_lock
    ctl, clk = control
    rule = bonded_rule(run, ctl, UNIT)
    return label(rule, lk, clk, unequal, flagged), rule


def test_bonded_equal(control):
    (L, info), rule = lab(series(), control, unequal=False)
    assert L == "bonded" and info["counted"] and rule["verdict"] == "bonded"


def test_bonded_unequal_locked(control):
    (L, info), _ = lab(series(), control, unequal=True)
    assert L == "bonded" and info["locked"] and not info["beating"]


def test_bounded_beating(control):
    (L, info), rule = lab(series(step_rate=-1.7), control, unequal=True)
    assert L == "bounded, beating" and info["beating"] and not info["locked"]
    assert not rule["clauses"]["phase_steps_held_deg"]["pass"]


def test_beating_not_at_the_frequency_difference_is_not_bonded(control):
    run, lk = series(step_rate=-1.7)
    lk["expected_beat_deg_per_t"] = [-1.0]            # drift 70% away from the lumps' frequency difference
    (L, info), _ = lab((run, lk), control, unequal=True)
    assert L == "not bonded" and not info["beating"]


def test_equal_lumps_drifting_step_is_not_bonded(control):
    (L, _), _ = lab(series(step_rate=-1.7), control, unequal=False)
    assert L == "not bonded"


def test_too_short(control):
    run, lk = series(T=300.0)
    ctl, clk = series(T=300.0, nn_jump=0.0)
    (L, _), _ = lab((run, lk), (ctl, clk), unequal=False)
    assert L == "too short"


def test_no_interaction():
    run_lock = series(nn_jump=0.0)
    (L, _), _ = lab(run_lock, series(nn_jump=0.0), unequal=False)
    assert L == "no interaction"


def test_N_exchange_is_not_bonded(control):
    (L, _), rule = lab(series(dN=0.01), control, unequal=True)
    assert L == "not bonded" and not rule["clauses"]["N_held"]["pass"]


def test_flagged_is_labelled_not_counted(control):
    (L, info), _ = lab(series(), control, unequal=False, flagged=True)
    assert L == "bonded" and info["counted"] is False
