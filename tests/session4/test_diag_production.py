"""Checks of swirl/session4/testA/diag_production.py: closed forms first, then synthetic fits, then the stored
record (reproduction of the analysis.json numbers it starts from)."""
import json
import os

import numpy as np
import pytest

from swirl.session4.testA import diag_production as D

OUT = D.OUT


def test_radial_ft_gaussian_closed_form():
    # int exp(-r^2/(2 s^2)) e^{-i k x} d^3x = (2 pi)^{3/2} s^3 exp(-k^2 s^2 / 2)
    s, k = 1.7, 2.5
    r = np.linspace(1e-8, 30.0, 300001)
    exact = (2 * np.pi) ** 1.5 * s ** 3 * np.exp(-k * k * s * s / 2)
    assert D.radial_ft(np.exp(-r * r / (2 * s * s)), r, k) == pytest.approx(exact, rel=1e-8)


def test_F3_pred_sign_flip_and_amplitude():
    om, Q, r = 0.9095, 0.2, np.array([28.0, 31.3, 37.7])
    a, b = D.F3_pred(r, 0, om, Q), D.F3_pred(r, 180, om, Q)
    np.testing.assert_allclose(a, -b, rtol=0, atol=1e-15)
    k3 = np.sqrt(9 * om * om - 1)
    np.testing.assert_allclose(a, k3 * Q * Q / (8 * np.pi * r) * np.sin(k3 * r), rtol=1e-12)


def test_lsq_neff_recovers_slope():
    t = np.arange(0.0, 400.0, 0.2)
    y = 0.3 + 1.234e-4 * t
    c, se = D.lsq_neff(t, y, 40.0, 390.0, 6.9, [np.ones_like(t), t])
    assert c[1] == pytest.approx(1.234e-4, rel=1e-10)


def test_beat_column_recovers_amplitude():
    # d = c0 + c1 t + F G(t) with G'' = cos(dw t + p): analytic G = -cos(dw t + p)/dw^2; the cumulative-trapezoid
    # column differs from it by a linear function (absorbed) and O(dt^2) terms
    dw, p, F = -0.048, 0.7, 3.0e-3
    t = np.arange(0.0, 400.0, 0.2)
    g = np.cos(dw * t + p)
    G = D.cumtrap(D.cumtrap(g, 0.2), 0.2)
    d = 30.0 + 1e-5 * t + F * (-np.cos(dw * t + p) / dw ** 2)
    tc = t - 215.0
    c, _ = D.lsq_neff(t, d, 40.0, 390.0, 6.9, [np.ones_like(t), tc, G])
    # two trapezoid integrations of cos(dw t): relative error 2 (dw dt)^2 / 12 = 1.5e-5; bound (dw dt)^2 / 4
    assert c[2] == pytest.approx(F, rel=(dw * 0.2) ** 2 / 4)


@pytest.fixture(scope="module")
def rec():
    if not os.path.exists(OUT):
        pytest.skip("diag_production.json not built")
    return json.load(open(OUT))


def test_record_carries_inputs(rec):
    i = rec["inputs"]
    for k in ("data_dir", "analysis_json_sha256", "code_commit", "RA3_TESTB"):
        assert i[k] is not None


def test_reproduces_stored_equal_force(rec):
    for q in rec["miss1"]["rows"]:
        assert q["F_recomputed"] == pytest.approx(q["F"], rel=1e-10)


def test_reproduces_stored_unequal_amplitude_and_static(rec):
    m2 = rec["miss2"]
    assert m2["static_split"]["U_amp_reproduction_max_rel"] < 1e-12
    for ch in m2["channels"].values():
        for q in ch["rows"]:
            assert q["F_amp_recomputed"] == pytest.approx(q["F_amp"], rel=1e-8)
            assert q["E1"]["F_amp_static_w"] == pytest.approx(q["F_amp_static_stored"], rel=1e-9)


def test_born_charge_matches_testB_order(rec):
    # Born r a3 from the profile against Test B's measured 0.0167: same order (not a physics target, a units check)
    d3 = rec["miss1"]["D3"]
    assert 0.5 < d3["ra3_born"] / d3["ra3_testB"] < 2.0
