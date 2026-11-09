"""Session 4 Test A: static quadrature, V_N bookkeeping, window rule, fits, verdict, driver guards.

Exact or closed-form answers first (Gaussian moments), then the Session 2 numbers, then synthetic data
through the fit and verdict code.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from swirl.session4.testA import analysis as A
from swirl.session4.testA import static as S

ROOT = S.ROOT


class Gauss:
    """f = a exp(-r^2 / (2 s^2)); attributes as static.Lump."""

    def __init__(self, a, s, kappa=1.0, omega=0.9, beta=4.0 / 3.0):
        self.a, self.s, self.kappa, self.omega, self.beta = a, s, kappa, omega, beta
        self.g6 = 0.09 * beta

    def ff(self, r):
        f = self.a * np.exp(-r * r / (2 * self.s ** 2))
        return f, -r / self.s ** 2 * f


def gauss_exact(a, s, d):
    base = (np.pi * s * s) ** 1.5 * np.exp(-d * d / (4 * s * s))
    out = dict(I_bil=a * a * base, I_dot=a * a * base * (1.5 * s * s - d * d / 4) / s ** 4)
    for p, q in S.PQ4 + S.PQ6:
        out["M%d%d" % (p, q)] = a ** (p + q) * (2 * np.pi * s * s / (p + q)) ** 1.5 * np.exp(-p * q * d * d / (2 * (p + q) * s * s))
    return out


@pytest.mark.parametrize("d", [3.0, 6.0, 10.0])
def test_moments_gaussian_exact(d):
    """Cross moments of two Gaussians against the closed form (target 1e-10 relative)."""
    g = Gauss(2.0, 1.7)
    mo = S.moments(g, g, d, **S.BASE)
    ex = gauss_exact(2.0, 1.7, d)
    worst = max(abs(mo[k] / ex[k] - 1) for k in ex)
    print("d=%g worst rel err %.2e" % (d, worst))
    assert worst < 1e-10


def test_F_sup_table_and_session2_t0():
    """WPC_DERIVATION F_sup/F_asym table (3-decimal rounding) and Session 2's t = 0 cross energy."""
    v = S.validation()
    print("F_sup table max abs diff %.2e" % v["F_sup_table_max_abs_diff"])
    assert v["F_sup_table_max_abs_diff"] < 5e-4
    t0 = v["t0_pair"]
    rel = t0["E_cross_axisym_cut34"] / t0["E_cross_session2_quadrature"] - 1
    print("t0 E_cross rel diff vs Session 2 quadrature %.2e, vs engine %.2e" % (rel, t0["rel_diff_vs_engine"]))
    assert abs(rel) < 1e-9
    assert abs(t0["rel_diff_vs_engine"]) < 1e-8


@pytest.fixture(scope="module")
def primary():
    b, w = S.LUMPS["primary"]
    return S.Lump(b, w)


def test_U_static_tail_large_r(primary):
    """U_static -> U_tail at large r (both channels); U_amp (first-order law) = U_static there."""
    l = primary
    for r in (36.0, 40.0):
        mo = S.moments(l, l, r, **S.BASE)
        for dth in (0.0, np.pi):
            U = S.U_static(mo, l, dth)
            Ut = S.U_tail(r, l.A, l.kappa, dth)
            Ua = S.U_amp(mo, l, l, dth)
            print("r=%g dth=%g U/U_tail-1 = %.2e  U_amp/U-1 = %.2e" % (r, dth, U / Ut - 1, Ua / U - 1))
            assert abs(U / Ut - 1) < 1e-4
            assert abs(Ua / U - 1) < 1e-4


def test_unequal_amp_equals_session3_law_at_large_r():
    """U_amp with true profiles against the Session 3 point law (charges Q) at r = 36, unequal pair."""
    l1, l2 = S.Lump(4 / 3, 0.9065045072253043), S.Lump(4 / 3, 0.857370255463254)
    mo = S.moments(l1, l2, 36.0, **S.BASE)
    Ua = S.U_amp(mo, l1, l2, 0.0)
    Ul = S.unequal_law(36.0, l1, l2, 0.0)
    print("U_amp %.6e law %.6e rel %.2e" % (Ua, Ul, Ua / Ul - 1))
    assert abs(Ua / Ul - 1) < 1e-3


def test_static_single_shifted_against_itself(primary):
    """A single lump's slice energy on a pair grid (centred at z1) against its own grid: zero to 1e-10 E."""
    l = primary
    E0 = S.self_slice_energy(l, 0.0, [-80.0, 0.0, 80.0])
    for r in (12.0, 30.0):
        mo = S.moments(l, l, r, **S.BASE)
        Zc = mo["Zc"]
        E1 = S.self_slice_energy(l, -r / 2, [-r / 2 - Zc, -r / 2, 0.0, r / 2, r / 2 + Zc], rho_max=max(mo["rho_max"], 60.0))
        print("r=%g dE/E = %.2e" % (r, E1 / E0 - 1))
        assert abs(E1 / E0 - 1) < 1e-10
    assert abs(E0 / 2843.8 - 1) < 1e-4       # WPC_DERIVATION section 7: theta = 0 slice energy 2843.8


# ------------------------------------------------------------------------------------------- V_N bookkeeping
def test_V_N_equal_zero_for_two_singles():
    """A 'pair' that is exactly two singles gives V_N = 0 (with and without box energy)."""
    s = dict(E=2838.4, E_box=2838.1, N=2860.3, omega=0.9095)
    p = dict(E=2 * s["E"], E_box=2 * s["E_box"], N=2 * s["N"])
    v = A.V_N_equal(p, s, -2.228e-5)
    assert v["V_N"] == 0.0 and v["V_N_box"] == 0.0


def test_V_N_equal_superposition_algebra():
    """Non-circular: lumps on a family E(N) = E0 + w0 (N - N0) + w1 (N - N0)^2 / 2 (w1 = d omega/dN). The pair holds
    2 N_s + dN and energy 2 E(N_s + dN/2) + U; the single sits at N_s with omega_s = dE/dN(N_s). V_N must return U
    to O(dN^3); a sign error in the omega or in the second-order term would miss by > 1e-3."""
    E0, w0, w1, N0 = 2838.4, 0.9095, -2.228e-5, 2860.3
    E = lambda N: E0 + w0 * (N - N0) + 0.5 * w1 * (N - N0) ** 2
    Ns = N0 + 7.0
    s = dict(E=E(Ns), E_box=E(Ns), N=Ns, omega=w0 + w1 * (Ns - N0))
    for U, dN in ((-0.25, 10.6), (0.243, -9.4), (-11.4, 300.0)):
        Ep = 2 * E(Ns + dN / 2) + U
        p = dict(E=Ep, E_box=Ep, N=2 * Ns + dN)
        v = A.V_N_equal(p, s, w1)
        print("U %.4f V_N %.10f" % (U, v["V_N"]))
        assert abs(v["V_N"] - U) < 1e-9
        assert v["V_N_minus_box"] == 0.0


def test_V_N_single_against_shifted_copy_pde():
    """3D: one settled lump (WP-E test source, dx = 0.8) at x = 0 and at x = -2.7 in the same box; the V_N-type
    combination 2[(E_b - E_a) - omega (N_b - N_a)] over the Hann window is zero within the WP-E sub-grid shift
    level (target 1e-5 E; shift test: ball E 4.1e-6 on this grid)."""
    src = _test_source()
    from swirl.engine.run import run
    from swirl.engine.snapshot import load_meta
    meta = load_meta(src)
    vals = []
    for x in (0.0, -2.7):
        cfg = dict(N=80, L=64.0, dt=0.08, t_end=40.0, m=1.0, g4=0.3, g6=0.12, absorber=True, threads=2,
                   init={"snapshot": dict(source=src, sha256=meta["sha256"], centers=[[x, 0.0, 0.0]], phases=[0.0],
                                          window=[14.0, 18.0])},
                   center=[x, 0.0, 0.0], r_c=[10.0], sample_every=0.8, point_every=1, avg_window=20.0,
                   lumps=dict(centers=[[x, 0.0, 0.0]], R_ball=10.0, every=5, it_every_step=True,
                              it_half=[{"axis": 0, "pos": 0.0}]))
        res = run(cfg)
        res.pop("_evolver")
        vals.append(A.run_values(res, 10.0, 40.0))
    a, b = vals
    V = 2 * ((b["E"] - a["E"]) - a["omega"] * (b["N"] - a["N"]))
    Vb = 2 * ((b["E_box"] - a["E_box"]) - a["omega"] * (b["N"] - a["N"]))
    print("shifted single: V %.3e  V_box %.3e  (E %.1f) dN %.2e domega %.2e" % (V, Vb, a["E"], b["N"] - a["N"],
                                                                                 b["omega"] - a["omega"]))
    assert abs(V) < 1e-5 * a["E"] and abs(Vb) < 1e-5 * a["E"]


# ------------------------------------------------------------------------------------------- window, force
def fake_run(r, a, T=400.0, omega=0.9095, dt=0.2):
    """A result-like dict whose lump separation is d = r + a t^2 / 2 (lumps at -d/2, +d/2)."""
    t = np.arange(0.0, T + 1e-9, dt)
    d = r + 0.5 * a * t * t
    X = np.zeros((t.size, 2, 3))
    X[:, 0, 0], X[:, 1, 0] = -d / 2, d / 2
    return {"t": T, "lumps": {"raw": {"series": {"t": t.tolist(), "X": X.tolist()}}}}, t, d


def test_window_rule_and_acceleration():
    kap, om = 0.41647, 0.9095
    for r, a in ((24.0, -1.26e-5), (16.0, -5.5e-4), (36.0, -2e-8)):
        res, t, d = fake_run(r, a)
        w = A.window_for(res, r, kap, om)
        t_exact = np.sqrt(2 * 0.10 / kap / abs(a))
        P = 2 * np.pi / om
        exp_t2 = min(400 - 1.5 * P, t_exact)
        print("r=%g t2 %.2f expected %.2f moving %s" % (r, w["t2"], exp_t2, w["moving"]))
        assert abs(w["t2"] - exp_t2) < 1.0
        assert w["moving"] == ((exp_t2 - 40.0) < 35.0)
        c, se, rms = A.accel_fit(w["t_d"], w["d"], w["t1"], w["t2"], P)
        assert abs(2 * c[2] / a - 1) < 1e-6


def fake_full(name, r, a, x=None, T=100.0, omega=0.9095, dt=0.2, dtheta=0, typ="primary", E=100.0):
    """A full result-like dict for pair_equal / pair_unequal: d = r + a t^2 / 2 for a pair (x None), a fixed lump at
    x for a single; phi = cos(omega t) at each lump centre, I = 10 omega^2 sin^2(omega t), constant energies."""
    t = np.arange(0.0, T + 1e-9, dt)
    nl = 1 if x is not None else 2
    X = np.zeros((t.size, nl, 3))
    if nl == 2:
        d = r + 0.5 * a * t * t
        X[:, 0, 0], X[:, 1, 0] = -d / 2, d / 2
        cen = [[-r / 2, 0.0, 0.0], [r / 2, 0.0, 0.0]]
    else:
        X[:, 0, 0] = x
        cen = [[x, 0.0, 0.0]]
    phi = np.cos(omega * t)
    I = 10 * omega ** 2 * np.sin(omega * t) ** 2 * nl
    C = np.concatenate([[0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * dt)])
    Es = np.full(t.size, E * nl)
    return {"t": T, "job": {"name": name, "r": r, "dtheta": dtheta, "types": [typ] * nl, "box": 128},
            "params": {"m": 1.0, "lumps": {"centers": cen}},
            "summary": {"nonlinear_share_0": 0.1},
            "point": {"t": t.tolist(), "phi": phi.tolist(), "C": C.tolist(), "I": I.tolist()},
            "series": {"t": t.tolist(), "E_plus_absorbed": Es.tolist(), "E_total": Es.tolist(),
                       "absorbed": np.zeros(t.size).tolist(), "nonlinear_share": np.full(t.size, 0.1).tolist()},
            "lumps": {"raw": {"series": {"t": t.tolist(), "X": X.tolist()},
                              "point": {"t": t.tolist(), "phi": np.stack([phi] * nl, axis=1).tolist(),
                                        "It_half": (0.5 * I)[:, None].tolist()}}}}


def test_pair_equal_no_window_is_flagged_not_a_crash():
    """Near-overlap production case (pA_r07..r13, T = 100): the 3-period smoothed d(t) starts at 1.5 P = 10.4 and is
    already past kappa |d - r| = 0.10 there, so t2 = t_hit = t_d[0] and [t1, t2] holds < 6 smoothed samples. The row
    comes back with F = None and a reason; the slow case still gives F = -(E_s/2) d'' exactly."""
    kap, om = 0.41647, 0.9095
    P = 2 * np.pi / om
    singles = [A.single_record(fake_full("s1", 0.0, 0.0, x=-3.5)), A.single_record(fake_full("s2", 0.0, 0.0, x=-12.0))]
    fast = A.pair_equal(fake_full("fast", 7.0, -0.05), singles, kap, om, -2.2e-5)
    print("fast: window", fast["window"], "reason", fast["force_reason"])
    assert fast["force_excluded"] and fast["F"] is None and fast["sigma_F"] is None and fast["F_se"] is None
    assert abs(fast["window"]["t_hit"] - 1.5 * P) < 0.3 and fast["window"]["moving"]
    assert "fewer than 6" in fast["force_reason"] and set(fast["d_raw"]) == {"0", "5", "10", "20", "40"}
    assert np.isfinite(fast["V_N"])                         # V_N still reported (moving: out of the verdict fits)
    a = -1.26e-5
    slow = A.pair_equal(fake_full("slow", 24.0, a, T=400.0), singles, kap, om, -2.2e-5)
    print("slow: F %.6e expected %.6e" % (slow["F"], -0.5 * 100.0 * a))
    assert not slow["force_excluded"] and slow["force_reason"] is None
    assert abs(slow["F"] / (-0.5 * 100.0 * a) - 1) < 1e-6


def test_pair_unequal_no_window_is_flagged_not_a_crash():
    lump1, lump2 = dict(omega=0.9065, kappa=0.4222), dict(omega=0.8574, kappa=0.5147)
    s1 = [A.single_record(fake_full("d1", 0.0, 0.0, x=-5.0, typ="dip", omega=0.9065))]
    s2 = [A.single_record(fake_full("b1", 0.0, 0.0, x=5.0, typ="big", omega=0.8574))]
    row = A.pair_unequal(fake_full("fast", 10.0, -0.05, omega=0.9065), s1, s2, lump1, lump2)
    print("unequal fast:", row["force_reason"])
    assert row["force_excluded"] and row["F_amp"] is None and row["sigma_F_amp"] is None and not row["use_in_fits"]
    assert row["V_N_amp_cos"] is None


PROD = os.environ.get("SWIRL_TESTA_RESULTS", os.path.join(ROOT, A.OUT, "results"))


@pytest.mark.skipif(not os.path.exists(os.path.join(PROD, "pA_r07_p000.json")),
                    reason="production Test A results not present (set SWIRL_TESTA_RESULTS)")
def test_analysis_all_on_production(tmp_path):
    """End to end on the production results: runs, and every row without a force is listed and kept out of the
    force route."""
    rec = A.analyze_all(results=PROD, out=str(tmp_path / "analysis.json"))
    for sec in ("primary", "unequal"):
        for lab, ch in rec[sec]["channels"].items():
            ex = {e["name"] for e in ch["force_excluded"]}
            names = (ch["F"] if "F" in ch else ch["F_amp"])["names"]
            print(sec, lab, "excluded", sorted(ex))
            assert ex.isdisjoint(names)
            assert all(q["force_reason"] for q in ch["rows"] if q["name"] in ex)
    assert "pA_r07_p000" in {e["name"] for e in rec["primary"]["channels"]["p000"]["force_excluded"]}


# ------------------------------------------------------------------------------------------- fits and verdict
R_FIT = np.arange(12.0, 40.01, 2.0)
KAP, AMP = 0.41647, 2 * np.pi * 56.7 ** 2


def noise(n):
    return np.sin(2.39996 * np.arange(1, n + 1) ** 1.3)      # deterministic, |.| <= 1, no RNG


def synth(form, rel=1e-3, floor=1e-7):
    r = R_FIT
    if form == "E1":
        V = -500 * np.exp(-0.45 * r)
    elif form == "E2":
        V = -AMP * np.exp(-KAP * r) / r
    elif form == "G":
        V = -30 * np.exp(-(r / 14.0) ** 2)
    elif form == "P":
        V = -4e4 * r ** -4.5
    sig = rel * np.abs(V) + floor
    return r, V + 0.5 * sig * noise(r.size), sig


@pytest.mark.parametrize("form", ["E1", "E2", "G", "P"])
def test_fits_recover_synthetic(form):
    r, V, sig = synth(form)
    fits = A.fit_all(r, V, sig)
    f = fits[form]
    print(form, {k: (round(v["chi2_red"], 3) if v.get("chi2_red") is not None else None) for k, v in fits.items()})
    assert f["ok"] and f["chi2_red"] <= 3.0
    others = [g for g in A.DECAYING if g != form]
    assert all(f["AIC"] + 10 < fits[g]["AIC"] for g in others)
    if form == "E2":
        assert abs(f["params"]["k"] - KAP) < 3 * f["se"]["k"] + 1e-6


def test_verdict_decays_on_yukawa():
    r, V, sig = synth("E2")
    r = np.append(r, [44.0, 48.0])                 # two points below the floor
    V = np.append(V, [1e-8, -2e-8])
    sig = np.append(sig, [1e-7, 1e-7])
    F16v = AMP * np.exp(-KAP * 16) * (KAP / 16 + 1 / 256)       # tail-law force at 16 as the measured F
    v = A.verdict(r, V, sig, F16v)
    print(v["verdict"], v["clauses"], v["preferred_form"], v["AIC_order"])
    assert v["verdict"] == "decays to zero"
    # L contains E2 (V_inf = sigma = 0), so on Yukawa data L can trail E2 by at most 2 x 2 = 4 in AIC: the A3
    # rule (beat every other form by > 10) cannot name E2 "preferred". E2 ranks first and beats E1, G, P by > 10.
    aic = {f: v["fits"][f]["AIC"] for f in v["fits"]}
    assert v["AIC_order"][0] == "E2" and v["preferred_form"] is None
    assert all(aic["E2"] + 10 < aic[f] for f in ("E1", "G", "P"))
    assert aic["L"] - aic["E2"] <= 4.0 + 1e-6


def test_verdict_grows_fires_on_linear_growth():
    """Synthetic confining case: Yukawa plus a linear rise sigma r (sigma = 1e-3): the 'grows' branch fires."""
    r = np.arange(12.0, 40.01, 1.0)
    V = -AMP * np.exp(-KAP * r) / r + 1e-3 * (r - 12.0)        # |V| rises at large r (A3 grows clause 2 uses |V|)
    sig = 1e-4 * np.ones_like(r) + 1e-3 * np.abs(-AMP * np.exp(-KAP * r) / r)
    V = V + 0.5 * sig * noise(r.size)
    v = A.verdict(r, V, sig, 1.0)
    print(v["verdict"], v["clauses"]["grows"], v["fits"]["L"]["params"], v["fits"]["L"]["se"])
    assert v["verdict"] == "grows"


def test_verdict_undetermined_few_decades():
    r = np.arange(12.0, 20.01, 1.0)
    V = -1.0 - 0.001 * r
    sig = 1e-3 * np.ones_like(r)
    v = A.verdict(r, V, sig, 1.0)
    assert v["verdict"] == "undetermined"


# ------------------------------------------------------------------------------------------- A3' (Addendum 1)
F16_TAIL = AMP * np.exp(-KAP * 16) * (KAP / 16 + 1 / 256)


def yukawa_departure(frac=0.02, floor_rel=1e-9):
    """Yukawa data with a smooth 2% model departure (as the static quadrature shows: exact data, residual is
    model departure, not noise) and a quadrature-like floor."""
    r = np.arange(12.0, 45.01, 0.5)
    U = -AMP * np.exp(-KAP * r) / r
    V = U * (1 + frac * np.exp(-(r - 12.0) / 6.0))
    sig = floor_rel * np.abs(V) + 1e-14
    return r, V, sig


def test_A3prime_decays_on_yukawa_with_2pct_departure():
    """Exact Yukawa data with a 2% model departure: A3' reads 'decays to zero' (not 'grows'); A3 does not decay."""
    r, V, sig = yukawa_departure()
    b = A.verdicts_both(r, V, sig, F16_TAIL)
    a3, ap = b["A3"], b["A3prime"]
    print("A3:", a3["verdict"], "A3' 12:", ap["r_lo_12"]["verdict"], "A3' 16:", ap["verdict"],
          ap["r_lo_16"]["clauses"], ap["r_lo_16"]["preferred_form"])
    assert ap["verdict"] == "decays to zero"
    assert a3["verdict"] != "decays to zero"
    assert ap["r_lo_16"]["preferred_form"] == "E2"


def test_A3prime_grows_fires():
    """Yukawa plus a confining linear rise (sigma = 1e-3 > 1% of |F(16)| = 5e-4): A3' reads 'grows'."""
    r = np.arange(12.0, 40.01, 1.0)
    V = -AMP * np.exp(-KAP * r) / r + 1e-3 * (r - 12.0)
    sig = 1e-4 * np.ones_like(r)
    V = V + 0.5 * sig * noise(r.size)
    v = A.verdict_prime(r, V, sig, 0.05, r_lo=16.0)
    print(v["verdict"], v["clauses"]["grows"])
    assert v["verdict"] == "grows"


def test_A3prime_small_sigma_below_1pct_is_not_growth():
    """A significant but tiny linear term (|sigma| below 1% of |F(16)|) does not fire 'grows' under A3'."""
    r = np.arange(12.0, 40.01, 1.0)
    V = -AMP * np.exp(-KAP * r) / r + 1e-3 * (r - 12.0)
    sig = 1e-4 * np.ones_like(r)
    v = A.verdict_prime(r, V, sig, 1.0, r_lo=16.0)       # 1% of |F(16)| = 1e-2 > sigma = 1e-3
    assert v["verdict"] != "grows"
    assert v["clauses"]["grows"]["sigma_confining_5se_and_above_1pct_F16"] is False


def test_A3prime_undetermined_cases():
    """Not falling 100x (two decades missing) and too few resolved points: undetermined."""
    r = np.arange(16.0, 24.01, 1.0)
    V = -AMP * np.exp(-KAP * r) / r
    v = A.verdict_prime(r, V, 1e-6 * np.abs(V), F16_TAIL)
    assert v["verdict"] == "undetermined" and v["clauses"]["decays"]["fall_ge_100"] is False
    r = np.arange(16.0, 19.01, 1.0)
    v = A.verdict_prime(r, -AMP * np.exp(-KAP * r) / r, 1e-6 + 0 * r, F16_TAIL)
    assert v["verdict"] == "undetermined" and v["n_fit"] < 5


def test_A3prime_form_preference_ignores_L():
    """On clean Yukawa data the A3' preference (E1, E2, G, P only) names E2."""
    r, V, sig = synth("E2")
    v = A.verdict_prime(r, V, sig, F16_TAIL, r_lo=16.0)
    assert v["preferred_form"] == "E2"


def test_runs_test_and_force_signs():
    assert A.runs_test(np.array([1, -1] * 10, float))["z"] > 3
    assert A.runs_test(np.array([1] * 10 + [-1] * 10, float))["z"] < -3
    fs = A.force_signs([10, 12, 14, 16], [-1.0, -0.5, 0.4, 0.2], [0.01, 0.01, 0.01, 0.01])
    assert fs["label"] == ["repulsive", "repulsive", "attractive", "attractive"]
    assert fs["sign_changes"][0]["kind"] == "minimum" and abs(fs["sign_changes"][0]["r_zero"] - (12 + 2 * 0.5 / 0.9)) < 1e-12


PROD_R = {"in phase": np.array([22, 24, 26, 28, 30, 32, 34, 36, 40.]),
          "anti phase": np.array([12, 13, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40.])}


@pytest.mark.parametrize("grid", list(PROD_R))
def test_V_F_integration_yukawa_production_spacing(grid):
    """V_F from the tail-law force at the production spacing: exact to 1% (Addendum 2), without tail against
    U - U(r_max), with the fitted tail against U itself."""
    r = PROD_R[grid]
    F = AMP * np.exp(-KAP * r) * (KAP / r + 1 / r ** 2)
    U = -AMP * np.exp(-KAP * r) / r
    rr, V0, s, _ = A.V_F_from_force(r, F, 1e-3 * F, tail=None)
    e0 = np.max(np.abs((V0[:-1] - (U[:-1] - U[-1])) / U[:-1]))
    rr, V, s, tail = A.V_F_from_force(r, F, 1e-3 * F, tail=True)
    e1 = np.max(np.abs(V / U - 1))
    print("%s: no tail max rel err %.2e; with fitted tail %.2e (tail %.3e vs U(r_max) %.3e, sigma used %s)"
          % (grid, e0, e1, tail and A.fit_force_tail(r, F, 1e-3 * F)[0], U[-1], tail["sigma_used"]))
    assert e0 < 0.01 and e1 < 0.01


def test_constant_force_reads_grows_through_V_F():
    """End to end through V_F: tail-law force plus a constant confining force (5% of F(16)) at the production
    in-phase spacing: V_F (fitted tail) reads 'grows' under A3' and meets every A3 grows clause; without the
    constant it reads 'decays to zero' under A3'."""
    r = PROD_R["in phase"]
    F16 = AMP * np.exp(-KAP * 16) * (KAP / 16 + 1 / 256)
    Fy = AMP * np.exp(-KAP * r) * (KAP / r + 1 / r ** 2)
    out = {}
    for lab, sc, fl in (("confining", 0.05 * F16, 1e-3), ("yukawa", 0.0, 1e-6)):
        F = Fy + sc
        sF = fl * F16 * np.ones_like(r) + 0.02 * np.abs(Fy)
        blk, v = A._vf_block(r, F, sF, F16)
        out[lab] = (v["A3"]["verdict"], v["A3prime"]["verdict"])
        out[lab + "_A3_reason"] = v["A3"].get("reason")
        out[lab + "_A3_grows"] = (v["A3"].get("clauses") or {}).get("grows")
        print(lab, out[lab], v["A3prime"]["r_lo_16"].get("clauses", v["A3prime"]["r_lo_16"].get("reason")))
    assert out["confining"][1] == "grows"
    # A3 as written: every grows clause holds, but "fewer than two decades of resolved |V|" (sigma r over r = 22..40
    # spans a factor ~2) makes it undetermined -- a structural limit of A3, reported, not changed
    g = out["confining_A3_grows"]
    assert out["confining"][0] == "undetermined" and out["confining_A3_reason"] == "fewer than two decades of resolved |V|"
    assert g["sigma_confining_gt_5se"] and g["V_hi_resolved_and_not_decreasing"] and g["L_beats_decaying_by_dAIC_10_and_L_chi2red_le_3"]
    assert out["yukawa"][1] == "decays to zero"


def _vf_trial(truth, f0, phase=0):
    """Force data truth + f0 * deterministic noise (|.| <= 1) on the production in-phase grid; V_F block."""
    r = PROD_R["in phase"]
    F16 = AMP * np.exp(-KAP * 16) * (KAP / 16 + 1 / 256)
    nz = np.sin(2.39996 * (np.arange(1, r.size + 1) + phase) ** 1.3)
    sF = f0 * np.ones_like(r)
    return A._vf_block(r, truth(r) + f0 * nz, sF, F16)


F16_T = AMP * np.exp(-KAP * 16) * (KAP / 16 + 1 / 256)
FY = lambda r: AMP * np.exp(-KAP * r) * (KAP / r + 1 / r ** 2)


@pytest.mark.parametrize("f0", [1e-5, 1e-4])
@pytest.mark.parametrize("sign", [1, -1], ids=["attractive", "repulsive"])
def test_A3prime_grows_for_yukawa_plus_2pct_constant(sign, f0):
    """Attractive and repulsive Yukawa force plus a constant confining force of 2% of F(16): A3' reads 'grows'
    (reviewer's points A and B: clause 2 on |F| is 'resolved' only; the tail fit survives a sign change)."""
    out = []
    for ph in range(4):
        blk, v = _vf_trial(lambda r: sign * FY(r) + 0.02 * F16_T, f0, ph)
        out.append(v["A3prime"]["verdict"])
        assert blk["tail"]["ok"] and blk["tail"]["sigma_used"] and 0.01 <= blk["tail"]["k"] <= 3.0
    print(sign, f0, out)
    assert out == ["grows"] * 4


@pytest.mark.parametrize("sign", [1, -1], ids=["attractive", "repulsive"])
def test_A3prime_pure_yukawa_decays(sign):
    out = [_vf_trial(lambda r: sign * FY(r), 1e-5, ph)[1]["A3prime"]["verdict"] for ph in range(4)]
    print(sign, out)
    assert out == ["decays to zero"] * 4


def test_A3prime_small_constant_not_grows():
    """Yukawa plus 0.3% of F(16) (below A3' grows-clause 1's 1%): never 'grows'."""
    for f0 in (1e-5, 1e-4):
        out = [_vf_trial(lambda r: FY(r) + 0.003 * F16_T, f0, ph)[1]["A3prime"]["verdict"] for ph in range(4)]
        print(f0, out)
        assert "grows" not in out


def test_L_singular_covariance_guard():
    """Data that is exactly V_inf + sigma r (C -> 0): the full L covariance is singular; the s.e. are reported
    NaN, and sigma's s.e. comes from the k-fixed refit (flagged), so a confining sigma is still read."""
    r = np.arange(16.0, 40.01, 2.0)
    V = -0.3 + 0.01 * r
    sig = 1e-4 * np.ones_like(r)
    L = A.fit_form("L", r, V, sig, k_ref=0.4165)
    print({k: L.get(k) for k in ("cov_singular", "cond", "k_fixed")}, L["se"], L.get("se_full"))
    assert L.get("cov_singular") is True
    assert np.isnan(list(L["se_full"].values())).any()
    assert np.isfinite(L["se"]["sigma"]) and abs(L["params"]["sigma"] - 0.01) < 5 * L["se"]["sigma"] + 1e-6


def fake_osc_run(I0, theta, E_tot, omega=0.9095, amp=0.3, T=100.0, dt=0.04, absorbed=0.0):
    """Result-like dict: phi(centre) = cos(omega t + theta), I(t) = I0 (1 + amp cos(2 omega t + 2 theta)) per step,
    C its exact integral; E series constant."""
    t = np.arange(0.0, T + 1e-9, dt)
    I = I0 * (1 + amp * np.cos(2 * omega * t + 2 * theta))
    C = I0 * (t + amp * (np.sin(2 * omega * t + 2 * theta) - np.sin(2 * theta)) / (2 * omega))
    ts = np.arange(0.0, T + 1e-9, 1.0)
    return {"point": {"t": t.tolist(), "phi": np.cos(omega * t + theta).tolist(), "C": C.tolist(), "I": I.tolist()},
            "series": {"t": ts.tolist(), "E_plus_absorbed": [E_tot] * ts.size, "E_total": [E_tot - absorbed] * ts.size,
                       "absorbed": [absorbed] * ts.size},
            "lumps": {"raw": {"point": {}}}}


def test_V_N_anti_phase_whole_period_regression():
    """Anti-phase start (lump 1 at -90 deg) against a single at 0 deg: the 2 omega modulation of I has opposite
    signs. Whole-period averages recover V_N = U to 1e-6; the old Hann average misses by > 1e-2 (the bug)."""
    om, w1 = 0.9095, -2.228e-5
    Is, U, dN = 2600.0, 0.2428, -9.6
    Ns = Is / om
    Es = 2838.4
    pair = fake_osc_run(om * (2 * Ns + dN), -np.pi / 2, 2 * Es + U + om * dN + w1 * dN * dN / 4)
    single = fake_osc_run(Is, 0.0, Es)
    pv, sv = A.run_values(pair, 40.0, 90.0), A.run_values(single, 40.0, 90.0)
    v = A.V_N_equal(pv, sv, w1)
    hp = A.hann_avg(pair["point"]["t"], pair["point"]["I"], 40.0, 74.4) / om
    hs = A.hann_avg(single["point"]["t"], single["point"]["I"], 40.0, 74.4) / om
    v_hann = (U + om * dN + w1 * dN * dN / 4) - om * (hp - 2 * hs) - w1 * (hp - 2 * hs) ** 2 / 4
    print("whole-period V_N %.8f (U %.4f); Hann-window V_N %.4f" % (v["V_N"], U, v_hann))
    assert abs(v["V_N"] - U) < 1e-6
    assert abs(v_hann - U) > 1e-2


def test_unequal_filters():
    """Half-beat rule and moving filter (A1, Addendum 2); the T = 100 unequal smoke run is excluded."""
    assert A.half_beat_rule(40.0, 400.0, 0.0479) and not A.half_beat_rule(40.0, 89.6, 0.0482)
    assert A.half_beat_rule(40.0, 60.0, 1e-3)                      # Delta omega t2 < 0.3
    assert not A.unequal_usable(True, 40.0, 400.0, 0.0479)
    assert A.unequal_usable(False, 40.0, 400.0, 0.0479)
    p = os.path.join(ROOT, A.OUT, "results", "smoke_uA_r20_p000.json")
    if not os.path.exists(p):
        pytest.skip("unequal smoke results not present")
    from swirl.session4.testA import driver as D
    res = A.load("smoke_uA_r20_p000")
    s1 = A.single_record(A.load("smoke_uA_single_dip_N144_x-10.0"))
    s2 = A.single_record(A.load("smoke_uA_single_big_N144_x+10.0"))
    q = A.pair_unequal(res, [s1], [s2], D.numbers("dip"), D.numbers("big"))
    print({k: q[k] for k in ("beats_in_window", "half_beat_ok", "use_in_fits", "F_amp", "F_mean")})
    assert q["half_beat_ok"] is False and q["use_in_fits"] is False


# ------------------------------------------------------------------------------------------- driver
def _test_source():
    for base in (ROOT, os.path.abspath(os.path.join(ROOT, "..", "..", ".."))):
        p = os.path.join(base, "outputs/session2/wpE/source/src_test_N80_L64.npz")
        if os.path.exists(p):
            return p
    pytest.skip("WP-E test source not present")


def test_t0_check_engine_vs_independent():
    """t = 0 energy of a two-copy snapshot start (test grid): engine vs independent evaluation (target 1e-10)
    and the cross energy vs the independent cross-term quadrature (target 1e-8)."""
    from swirl.engine.snapshot import load_meta
    from swirl.session4.testA.driver import t0_check
    src = _test_source()
    sha = load_meta(src)["sha256"]
    cfg = dict(N=80, L=64.0, dt=0.08, t_end=1.0, m=1.0, g4=0.3, g6=0.12, absorber=True, threads=2,
               init={"snapshot": dict(source=src, sha256=sha, centers=[[-6.2, 0.0, 0.0], [6.2, 0.0, 0.0]],
                                      phases=[-np.pi / 2, np.pi / 2], window=[14.0, 18.0])},
               center=[-6.2, 0.0, 0.0], r_c=[10.0],
               lumps=dict(centers=[[-6.2, 0.0, 0.0], [6.2, 0.0, 0.0]], R_ball=10.0))
    t = t0_check(cfg)
    print({k: v for k, v in t.items() if k != "note"})
    assert abs(t["rel_diff"]) < 1e-10
    assert abs(t["cross_rel_diff"]) < 1e-8
    assert t["superposition_residual_max"] < 1e-12


def test_driver_imports_no_numba():
    """The pool parent must not load numba (Session 3 report section 13)."""
    code = "import sys; import swirl.session4.testA.driver as D; print('numba' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_driver_boxes_pass_engine_guards():
    """Every primary job's box passes the snapshot/sponge/face guards and the plan's 3 hw + sponge rule."""
    from swirl.engine.lumps import check_lumps
    from swirl.session4.testA import driver as D
    if not os.path.exists(D.primary_source()):
        pytest.skip("primary source not present")
    J = [j for j in D.jobs() if j["kind"] in ("pair", "single", "twin") and set(j["types"]) == {"primary"}]
    hw = D.numbers("primary")["hw"]
    for j in J:
        c = j["cfg"]
        check_lumps(c["init"], c["L"], {"gamma": 2.0, "width": 12.0, "power": 3})
        for ctr in c["lumps"]["centers"]:
            assert c["L"] / 2 - abs(ctr[0]) >= 3 * hw + 12.0
    assert len({j["cfg"]["N"] for j in J}) <= 8
