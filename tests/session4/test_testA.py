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
    """Pair = 2 singles + (U + omega dN) in E and + dN in N: V_N = U - (domega/dN) dN^2/4 exactly."""
    s = dict(E=2838.4, E_box=2838.4, N=2860.3, omega=0.9095)
    U, dN, w = -0.25, 10.6, -2.228e-5
    p = dict(E=2 * s["E"] + U + s["omega"] * dN, E_box=2 * s["E"] + U + s["omega"] * dN, N=2 * s["N"] + dN)
    v = A.V_N_equal(p, s, w)
    assert abs(v["V_N"] - (U - w * dN * dN / 4)) < 1e-9


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
        c2, se, rms = A.accel_fit(w["t_d"], w["d"], w["t1"], w["t2"], P)
        assert abs(2 * c2 / a - 1) < 1e-6


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


def test_runs_test_and_force_signs():
    assert A.runs_test(np.array([1, -1] * 10, float))["z"] > 3
    assert A.runs_test(np.array([1] * 10 + [-1] * 10, float))["z"] < -3
    fs = A.force_signs([10, 12, 14, 16], [-1.0, -0.5, 0.4, 0.2], [0.01, 0.01, 0.01, 0.01])
    assert fs["label"] == ["repulsive", "repulsive", "attractive", "attractive"]
    assert fs["sign_changes"][0]["kind"] == "minimum" and abs(fs["sign_changes"][0]["r_zero"] - (12 + 2 * 0.5 / 0.9)) < 1e-12


def test_V_F_integration_exact_for_tail_law():
    """V_F = -int F dr of the tail-law force reproduces U_tail(r) - U_tail(r_max) to the trapezoid error."""
    r = np.arange(12.0, 40.01, 0.25)
    F = AMP * np.exp(-KAP * r) * (KAP / r + 1 / r ** 2)
    rr, V, s = A.V_F_from_force(r, F, 0 * F)
    U = -AMP * np.exp(-KAP * rr) / rr
    err = np.max(np.abs((V - (U - U[-1])) / U))
    print("V_F trapezoid rel err %.2e" % err)
    assert err < 2e-3


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
