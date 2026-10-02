"""WP0 (Session 3): the corrected bonded rule (swirl/engine/bonded.py).

Synthetic series test every clause and every verdict. Two tests use the saved WP-E result files
(outputs/session2/wpE/results/, on the external drive; skipped if absent): the far-apart control E14 must
come out "no interaction" and E11's first run (T = 700) must not come out "bonded". The in-phase
control's contact time (the interaction time) is read from E02's own count windows."""

import importlib.util
import json
import os

import numpy as np
import pytest

from swirl.engine.bonded import DEFAULTS, bonded_rule, bonded_rule_session2, series_from_result

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "outputs", "session2", "wpE", "results")


def synth(K=5, t_end=800.0, ratio=lambda t: 1.0, N=lambda t: 0.0, step=lambda t: 0.0, count=None, dt=3.5):
    """A run dict with the layout of series_from_result: ring of K lumps, nearest-neighbour distance
    16 * ratio(t) for every lump, N = 2800 (1 + N(t) alternating in sign over the lumps), neighbour steps
    72 deg + step(t)."""
    t = np.arange(5.0, t_end, dt)
    nn = np.array([[16.0 * ratio(x)] * K for x in t])
    sgn = np.array([(-1.0) ** k for k in range(K)])
    Nk = np.array([2800.0 * (1.0 + N(x) * sgn) for x in t])
    steps = np.array([[72.0 + step(x)] * K for x in t])
    tc = np.arange(10.0, t_end, 25.0)
    cnt = [K if count is None else count(x) for x in tc]
    return {"K": K, "t_end": t_end, "completed": True, "t_count": tc.tolist(), "count": cnt,
            "t": t, "nn": nn, "N": Nk, "steps": steps, "edges": [(k, (k + 1) % K) for k in range(K)]}


def control(t_end=500.0):
    """Far-apart control: the E14 noise floor (distance drift 5e-4 over 500, N and phase constant)."""
    return synth(t_end=t_end, ratio=lambda t: 1.0 + 5e-4 * t / 500.0, N=lambda t: 1e-5, step=lambda t: 0.02)


def libration(t_end=800.0, t_from=50.0, amp=0.02, periods=3):
    """Distance ratio libration over whole periods, symmetric about the middle of [t_from, t_end], so that
    its linear fit over t >= t_from has no trend."""
    tc, span = 0.5 * (t_from + t_end), t_end - t_from
    return lambda t: 1.0 + amp * np.cos(2 * np.pi * periods * (t - tc) / span)


CONTACT = 100.0      # interaction time, so five of them is 500


def test_synthetic_bonded():
    v = bonded_rule(synth(ratio=libration(), N=lambda t: 2e-4, step=lambda t: 0.05), control(), CONTACT)
    assert v["verdict"] == "bonded", v["reason"]
    assert all(c["pass"] for c in v["clauses"].values())
    assert v["parameters"]["contact_time"] == CONTACT and v["parameters"]["dist_tol"] == DEFAULTS["dist_tol"]


def test_response_is_what_separates_bonded_from_no_interaction():
    v = bonded_rule(synth(ratio=lambda t: 1.0 + 4e-4 * t / 800.0), control(), CONTACT)
    assert v["verdict"] == "no interaction"
    assert not v["clauses"]["response_differs_from_control"]["pass"]


def test_too_short():
    v = bonded_rule(synth(t_end=400.0, ratio=libration(t_end=400.0)), control(), CONTACT)
    assert v["verdict"] == "too short" and not v["clauses"]["duration"]["pass"]
    assert v["clauses"]["duration"]["limit"] == 5 * CONTACT


def test_each_held_clause_can_fail():
    base = dict(ratio=libration())
    cases = {
        "count_equals_K": dict(base, count=lambda t: 5 if t < 400 else 4),
        "distance_within_tol": dict(ratio=lambda t: 1.0 + 0.15 * np.sin(np.pi * (t - 5.0) / 795.0) ** 2),
        "no_distance_trend": dict(ratio=lambda t: 1.0 + 0.05 * t / 800.0),
        "N_held": dict(base, N=lambda t: 0.02 * t / 800.0),
        "phase_steps_held_deg": dict(base, step=lambda t: 3.0 * t / 800.0),
    }
    for name, kw in cases.items():
        v = bonded_rule(synth(**kw), control(), CONTACT)
        assert v["verdict"] == "not bonded", (name, v["reason"])
        assert not v["clauses"][name]["pass"], name


def test_trend_is_against_the_control_not_absolute():
    """A 1e-4 total drift over 800 is below 3 times the control's 5e-4 over 500; 4e-3 is not."""
    ok = bonded_rule(synth(ratio=lambda t: 1.0 + 1e-4 * t / 800.0), control(), CONTACT)
    bad = bonded_rule(synth(ratio=lambda t: 1.0 + 4e-3 * t / 800.0), control(), CONTACT)
    assert ok["clauses"]["no_distance_trend"]["pass"]
    assert not bad["clauses"]["no_distance_trend"]["pass"] and bad["verdict"] == "not bonded"
    # a noisier control raises the limit
    noisy = synth(t_end=500.0, ratio=lambda t: 1.0 + 5e-3 * t / 500.0)
    assert bonded_rule(synth(ratio=lambda t: 1.0 + 4e-3 * t / 800.0), noisy, CONTACT)["clauses"]["no_distance_trend"]["pass"]


def test_common_n_drift_is_not_exchange():
    """Every lump losing the same N (isolated lumps do) is not an exchange between lumps."""
    run = synth(ratio=libration())
    run["N"] = run["N"] * (1.0 - 1.5e-3 * run["t"] / 500.0)[:, None]
    v = bonded_rule(run, control(), CONTACT)
    assert v["clauses"]["N_held"]["pass"] and abs(v["run_metrics"]["N_common_drift"]) > 1e-3


def test_arguments_override_defaults_and_unknown_is_refused():
    run = synth(ratio=libration(amp=0.06))
    assert bonded_rule(run, control(), CONTACT)["verdict"] == "bonded"
    assert bonded_rule(run, control(), CONTACT, dist_tol=0.05)["verdict"] == "not bonded"
    assert bonded_rule(run, control(), CONTACT, interaction_periods=10)["verdict"] == "too short"
    with pytest.raises(TypeError):
        bonded_rule(run, control(), CONTACT, dist_tolerance=0.1)
    with pytest.raises(ValueError):
        bonded_rule(run, None, CONTACT)
    with pytest.raises(ValueError):
        bonded_rule(run, control(), None)
    with pytest.raises(ValueError):
        bonded_rule(synth(K=1), control(), CONTACT)


def test_session2_rule_labels_a_motionless_run_bonded():
    """Why it is superseded: nothing moving passes it, and so does a slow collapse that ends inside the run."""
    tm = list(np.arange(10.0, 700.0, 25.0))
    still = bonded_rule_session2(tm, [5] * len(tm), [[1.0] * 5] * len(tm), 5)
    collapse = bonded_rule_session2(tm, [5] * len(tm), [[1.0 - 0.3 * t / 700.0] * 5 for t in tm], 5)
    assert still["bonded"] and collapse["bonded"]
    assert not bonded_rule_session2(tm, [5] * 20 + [4] * (len(tm) - 20), [[1.0] * 5] * len(tm), 5)["bonded"]


# ---------------------------------------------------------------------------
# Saved Session 2 runs
# ---------------------------------------------------------------------------

def _load(name):
    path = os.path.join(RES, name + ".json")
    if not os.path.exists(path):
        pytest.skip("saved WP-E result not available: " + path)
    return json.load(open(path))


@pytest.fixture(scope="module")
def saved():
    e14 = _load("E14_ring5_m1_a30")
    e02 = _load("E02_ring5_m0_a16")
    K = len(e02["lumps"]["raw"]["centers"])
    contact = next(c["t_mid"] for c in e02["lumps"]["counts"] if c["count_distinct"] < K)
    return {"E14": series_from_result(e14), "E11": series_from_result(_load("E11_ring6_c_R15.77")),
            "contact": contact, "e14_raw": e14}


def test_e14_far_apart_control_is_no_interaction(saved):
    """The far-apart control against itself: it holds everything and differs from nothing."""
    v = bonded_rule(saved["E14"], saved["E14"], saved["contact"])
    assert v["verdict"] == "no interaction", v
    assert abs(saved["contact"] - 160.36) < 0.1
    c = v["clauses"]
    assert c["count_equals_K"]["pass"] and c["distance_within_tol"]["pass"] and c["no_distance_trend"]["pass"]
    assert c["N_held"]["pass"] and c["phase_steps_held_deg"]["pass"]
    # measured, not assumed: distance 4.3e-4, N exchange 7e-5 (common drift 1.5e-3), steps 0.036 degrees
    m = v["run_metrics"]
    assert m["nn_dev"] < 5.5e-4 and m["steps_dev_deg"] < 0.05 and m["N_dev"] < 2e-4
    assert abs(m["N_common_drift"]) > 1e-3


def test_e11_first_run_is_not_bonded(saved):
    """T = 700 against five contact times of 160: too short, and it contracts (distance 41%, trend 3e-4
    per unit time against 3.4e-6) and exchanges N (21%)."""
    v = bonded_rule(saved["E11"], saved["E14"], saved["contact"])
    assert v["verdict"] != "bonded" and v["verdict"] == "not bonded"
    c = v["clauses"]
    assert not c["no_distance_trend"]["pass"] and not c["distance_within_tol"]["pass"]
    assert not c["duration"]["pass"] and "also too short" in v["reason"]


def test_session2_label_for_e14_and_e11_was_bonded(saved):
    """The old rule, on the same runs, gave "bonded" for both (the two failures this rule fixes)."""
    spec = importlib.util.spec_from_file_location("wpe_post", os.path.join(ROOT, "outputs/session2/wpE/postprocess.py"))
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    crit = json.load(open(os.path.join(ROOT, "outputs/session2/wpE/criteria.json")))
    for name in ("E14_ring5_m1_a30", "E11_ring6_c_R15.77"):
        r = pp.record(_load(name))
        old = bonded_rule_session2(r["peaks"]["t_mid"], r["counts"]["count_distinct"], r["peaks"]["ratios"], r["K"],
                                   completed=r["completed"], eval_from=crit["eval_from"], nn_min=crit["nn_distance_min"],
                                   nn_max=crit["nn_distance_max"], hold_fraction=crit["hold_fraction"])
        assert old["bonded"] and pp.outcome(r, crit)["outcome"] == "bonded"
