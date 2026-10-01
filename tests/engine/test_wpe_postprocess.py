"""WP-E post-processing (outputs/session2/wpE/postprocess.py) on synthetic result JSON: every Block 3
outcome class, the mechanical report, the guards after a merger, K = 1, the E1 to E15 table.

The synthetic results have the stored-series layout of a real run (raw lump series, per-step point
series, count windows with peaks and labels), with lumps on a ring of radius R(t) and phases
theta_k = 72 k degrees, so every expected value is known exactly."""

import importlib.util
import json
import os

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
spec = importlib.util.spec_from_file_location("wpe_post", os.path.join(ROOT, "outputs/session2/wpE/postprocess.py"))
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)
CRIT = json.load(open(os.path.join(ROOT, "outputs/session2/wpE/criteria.json")))
EMAP = json.load(open(os.path.join(ROOT, "outputs/session2/wpE/criteria_E_map.json")))
OM = 0.9095
WIN = 3 * 2 * np.pi / OM


def ring(K, R):
    a = 2 * np.pi * np.arange(K) / K
    return np.stack([R * np.cos(a), R * np.sin(a), np.zeros(K)], 1)


def fake(K=5, a0=16.0, T=500.0, R_of=None, count_of=None, completed=True, name="synthetic", L=200.0):
    """Synthetic result. R_of(t): ring radius / initial radius; count_of(t_mid): distinct count."""
    R0 = a0 / (2 * np.sin(np.pi / K)) if K > 1 else 0.0
    R_of = R_of or (lambda t: 1.0)
    count_of = count_of or (lambda t: K)
    th = np.deg2rad(72.0 * np.arange(K))
    ts = np.arange(0.0, T + 1e-9, 0.5)
    tp = np.arange(0.0, T + 1e-9, 0.1)
    X = np.array([ring(K, R0 * R_of(t)) for t in ts])
    N = 2800.0
    zK = np.zeros((ts.size, K))
    series = {"t": ts.tolist(), "E": (zK + 2800.0).tolist(), "X": X.tolist(), "It": (zK + N * OM).tolist(),
              "Ip": zK.tolist(), "P": np.zeros((ts.size, K, 3)).tolist(), "P_total": [[0.0, 0.0, 0.0]] * ts.size,
              "P_half": [[]] * ts.size, "L_axes": [[-100.0]] * ts.size, "centers_used": X.tolist(),
              "E_group": (1.0 - 1e-4 * ts).tolist(), "L_axis_box": (-100.0 + 0 * ts).tolist(),
              "L_axis_sphere": (-99.0 + 0 * ts).tolist()}
    phi = np.cos(OM * tp[:, None] + th[None, :])
    counts = []
    for t0 in np.arange(0.0, T - WIN + 1e-9, 25.0):
        tm = t0 + WIN / 2
        nd = int(count_of(tm))
        pos = ring(K, R0 * R_of(tm))
        labels = list(range(K)) if nd >= K else [min(i, nd - 1) for i in range(K)]
        counts.append({"t_mid": float(tm), "count": K, "count_distinct": nd, "labels": labels,
                       "peaks": [{"pos": pos[i].tolist(), "value": 10.0 - 0.1 * i} for i in range(K)]})
    return {"params": {"L": L, "N": 256, "m": 1.0, "g4": 0.3, "g6": 0.12, "t_end": T,
                       "lumps": {"avg_window": 20.0, "omega_nominal": OM, "count_periods": 3},
                       "_candidate": {"name": name, "positions": ring(K, R0).tolist(),
                                      "phases_deg": (72.0 * np.arange(K)).tolist(), "start": "snapshot", "K": K}},
            "completed": completed, "t": T,
            "lumps": {"raw": {"centers": X[-1].tolist(), "series": series, "point": {"t": tp.tolist(), "phi": phi.tolist()}},
                      "counts": counts},
            "series": {"nonlinear_share": [0.35], "t": ts.tolist(), "max_abs_phi": (6.9 * np.exp(-ts / 50)).tolist()}}


def run_all(res, crit=CRIT):
    r = pp.record(res)
    rep = pp.report(r)
    return r, rep, pp.outcome(r, crit)


def test_bonded():
    r, rep, oc = run_all(fake(R_of=lambda t: 1.0 + 0.02 * np.sin(t / 30)))
    assert oc["outcome"] == "bonded" and oc["held_fraction"] == 1.0 and oc["n_samples"] == 18
    assert rep["contact"]["time"] is None and rep["nn_passes_1.5"]["time"] is None
    assert abs(rep["nn_ratio"]["peaks_max_over_run"] - 1.02) < 1e-3
    assert rep["phase_steps"]["max_dev_deg_over_valid"] < 1e-2
    assert rep["N_spread_rel"]["max_over_valid"] < 1e-7


def test_merged_and_guards():
    # radius shrinks linearly to 0 at t = 300; one blob when R < 0.15 R0
    f = lambda t: max(1.0 - t / 300.0, 0.0)
    res = fake(R_of=f, count_of=lambda t: 5 if f(t) > 0.15 else 1)
    r, rep, oc = run_all(res)
    assert oc["outcome"] == "merged"
    R0 = 16.0 / (2 * np.sin(np.pi / 5))
    t_pair = 300.0 * (1 - pp.CONTACT / 16.0)          # pair spacing a(t) = 16 f(t) reaches 2 hw
    assert abs(rep["contact"]["pair_within_2hw_tracked"] - t_pair) < 0.5 * 2 * np.pi / OM + 0.5   # half-period window steps
    assert rep["contact"]["time"] == min(x for x in (rep["contact"]["count_drop_t_mid"],
                                                     rep["contact"]["pair_within_2hw_tracked"],
                                                     rep["contact"]["pair_within_2hw_peaks"]) if x is not None)
    tv = r["per_lump_valid_until"]
    assert tv is not None and all(t < tv for t in r["N_spread"]["t"]) and all(t < tv for t in r["tracked"]["t"])
    assert all(t < tv for t in r["phase_steps"]["t"])
    assert R0 > 0


def test_flew_apart():
    res = fake(R_of=lambda t: 1.0 + t / 500.0)
    r, rep, oc = run_all(res)
    assert oc["outcome"] == "flew apart"
    assert abs(rep["nn_passes_1.5"]["tracked"] - 250.0) < 0.5 * 2 * np.pi / OM + 0.5
    assert rep["contact"]["time"] is None


def test_not_distinct_at_start():
    res = fake(count_of=lambda t: 4 if t < 25 else 5)
    r, rep, oc = run_all(res)
    assert oc["outcome"] == "not distinct at the start" and oc["end_state"] == "bounded at T, not bonded"
    assert oc["held_fraction"] == 1.0          # all later samples pass, yet it is not bonded


def test_one_failing_sample_T500_not_bonded_T700_bonded():
    bad = lambda t: 4 if 200 < t < 225 else 5
    r, rep, oc = run_all(fake(count_of=bad))
    assert oc["n_samples"] == 18 and oc["outcome"] == "bounded at T, not bonded"
    assert "trend" in oc and abs(oc["trend"]["slope_nn_mean_peaks"]) < 1e-9
    r, rep, oc = run_all(fake(T=700.0, count_of=bad))
    assert oc["n_samples"] == 26 and oc["held_fraction"] > 0.95 and oc["outcome"] == "bonded"


def test_last_sample_fails_and_count_above_K_and_incomplete():
    r, rep, oc = run_all(fake(R_of=lambda t: 0.45 if t > 470 else 1.0))
    assert oc["outcome"] == "bounded at T, not bonded" and not oc["last_sample"]["ok"]
    r, rep, oc = run_all(fake(count_of=lambda t: 6))
    assert oc["outcome"] == "bounded at T, not bonded" and oc["held_fraction"] == 0.0
    r, rep, oc = run_all(fake(completed=False))
    assert oc["outcome"] == "bounded at T, not bonded" and oc["note"] == "run not completed"


def test_not_evaluated_without_parameters():
    res = fake()
    r = pp.record(res)
    for k in ("hold_fraction", "eval_from", "nn_distance_max", "min_lump_count"):
        c = dict(CRIT)
        c[k] = None
        assert pp.outcome(r, c)["outcome"] == "not evaluated"
        c.pop(k)
        assert pp.outcome(r, c)["outcome"] == "not evaluated"
    assert pp.outcome(r, None)["outcome"] == "not evaluated"
    assert pp.outcome(r, {**CRIT, "min_lump_count": 4})["outcome"] == "not evaluated"
    assert pp.outcome(r, {**CRIT, "min_lump_count": 5})["outcome"] == "bonded"


def test_K1_control():
    r, rep, oc = run_all(fake(K=1, a0=0.0))
    assert oc["outcome"] == "not evaluated" and "K = 1" in oc["reason"]
    assert rep["count_series"]["count_distinct"][0] == 1 and rep["contact"]["time"] is None
    assert rep["group"]["E_retained_end"] < 1.0


def test_criteria_E_table():
    f1 = lambda t: max(1.0 - t / 500.0, 0.0)     # E1: contact when 16 f = 2 hw: t = 291
    f2 = lambda t: max(1.0 - t / 260.0, 0.0)     # E2: t = 151
    runs = {}
    for name, f in (("E01_ring5_m1_a16", f1), ("E02_ring5_m0_a16", f2)):
        res = fake(T=600.0 if name.startswith("E01") else 400.0, R_of=f,
                   count_of=lambda t, f=f: 5 if f(t) > 0.15 else 1, name=name)
        r = pp.record(res)
        rep = pp.report(r)
        runs[name] = {"record": r, "report": rep, "outcome": pp.outcome(r, CRIT)}
    runs["E14_far_control_x"] = {"record": pp.record(fake(a0=30.0)), "report": None, "outcome": None}
    runs["E14_far_control_x"]["report"] = pp.report(runs["E14_far_control_x"]["record"])
    runs["E14_far_control_x"]["outcome"] = pp.outcome(runs["E14_far_control_x"]["record"], CRIT)
    tab = pp.criteria_E(runs, EMAP)
    e1 = {i["item"]: i for i in tab["rows"]["E1"]["items"]}
    assert e1["outcome merged"]["verdict"] == "hit"
    assert e1["contact time"]["verdict"] == "hit" and abs(e1["contact time"]["measured"] - 291.0) < 4.0
    assert e1["contact factor E1 / E2"]["verdict"] == "hit"
    assert abs(e1["contact factor E1 / E2"]["measured"] - 500.0 / 260.0) < 0.03
    e2 = {i["item"]: i for i in tab["rows"]["E2"]["items"]}
    assert e2["contact time"]["verdict"] == "hit"
    assert tab["mapping"]["E14"] == "E14_far_control_x"                 # prefix fallback
    e14 = {i["item"]: i["verdict"] for i in tab["rows"]["E14"]["items"]}
    assert set(e14.values()) == {"hit"}
    assert tab["rows"]["E3"]["items"][0]["verdict"] == "not evaluable"   # no result yet
    assert tab["rows"]["E12"]["items"][-1]["verdict"] in ("not evaluable",)
