"""WP1A checks 1-4 on the single-frequency profile solver."""

import glob
import json
import math
import os

import numpy as np
import pytest

from swirl.family.family import BETA_SET_B, SET_B, continue_to, prior_comparison
from swirl.family.io import OUT_DIR, write_json
from swirl.family.profile import omega_min_rescaled, shoot, solve_profile

BETAS = [BETA_SET_B, 0.625, 0.70, 0.85, 1.0]
U0_CUBIC = 4.3374   # 3D cubic ground state, U'' + (2/s) U' = U - U^3


# ---- check 1: g6 = 0 gives the known cubic ground state -----------------------

def test_cubic_ground_state_direct():
    # a = m^2 - omega^2 = 1, b = g4/8 = 1, c = 0  ->  U'' + 2U'/s = U - U^3
    s = solve_profile(1.0, m=math.sqrt(2.0), g4=8.0, g6=0.0)
    rel = abs(s["P0"] - U0_CUBIC) / U0_CUBIC
    print("\ncheck1 direct: U(0) = %.8f, rel diff to 4.3374 = %.2e (target 1e-3)" % (s["P0"], rel))
    assert rel < 1e-3


@pytest.mark.parametrize("w", [0.99, 0.9, 0.6])
def test_cubic_ground_state_rescaled(w):
    # rescaled family with beta = 0: P(s) = 2 sqrt(2) eps U(eps s), eps^2 = 1 - w^2
    s = solve_profile(w, 1.0, 1.0, 0.0)
    eps = math.sqrt(1 - w * w)
    U0 = s["P0"] / (2 * math.sqrt(2) * eps)
    rel = abs(U0 - U0_CUBIC) / U0_CUBIC
    print("\ncheck1 rescaled w=%.2f: U(0) = %.8f, rel = %.2e" % (w, U0, rel))
    assert rel < 1e-3


def test_shooting_agrees_with_collocation():
    for beta in (BETA_SET_B, 1.0):
        P0s, *_ = shoot(0.9, 1.0, 1.0, beta)
        P0b = solve_profile(0.9, 1.0, 1.0, beta)["P0"]
        print("\nshoot vs bvp beta=%.4f: %.10f %.10f" % (beta, P0s, P0b))
        assert abs(P0s - P0b) / P0b < 1e-7


# ---- check 2: Derrick / Pohozaev identity -------------------------------------

@pytest.mark.parametrize("beta", BETAS)
def test_derrick_identity(beta):
    ws = [0.995, 0.95, 0.9, 0.8, 0.7]
    worst = 0.0
    for w in ws:
        if w < omega_min_rescaled(beta) + 0.03:
            continue
        s = continue_to(beta, w)
        worst = max(worst, abs(s["derrick_rel"]))
        print("\ncheck2 beta=%.4f w=%.3f: (T + 3 int G)/T = %.2e" % (beta, w, s["derrick_rel"]))
    assert worst < 1e-6


def test_derrick_identity_saved_family():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "family_beta*.json")))
    if not files:
        pytest.skip("run swirl.family.run_family first")
    for fn in files:
        with open(fn) as f:
            d = json.load(f)
        worst = max(abs(r["derrick_rel"]) for r in d["rows_rescaled"])
        print("\ncheck2 saved %s: %d profiles, worst |rel| = %.2e"
              % (os.path.basename(fn), len(d["rows_rescaled"]), worst))
        assert worst < 1e-6


# ---- check 3: equal beta gives the same rescaled family -----------------------

def _rescaled(s, m, g4):
    return dict(P0=s["P0"] * math.sqrt(g4) / m, hw=s["half_width"] * m, N=s["N"] * g4,
                E=s["E"] * g4 / m, A=s["A"] * math.sqrt(g4))


@pytest.mark.parametrize("w", [0.97, 0.9, 0.8])
def test_equal_beta_same_rescaled_family(w):
    ref = solve_profile(w, 1.0, 1.0, BETA_SET_B)
    refr = _rescaled(ref, 1.0, 1.0)
    sets = [SET_B, dict(m=1.0, g4=0.6, g6=0.22),
            dict(m=2.0, g4=0.6, g6=BETA_SET_B * 0.36 / 4.0)]
    worst = 0.0
    for p in sets:
        s = solve_profile(w * p["m"], **p)
        sr = _rescaled(s, p["m"], p["g4"])
        for k in refr:
            worst = max(worst, abs(sr[k] - refr[k]) / abs(refr[k]))
    print("\ncheck3 w/m=%.2f: worst rel diff over P0, hw, N, E, A = %.2e (target 1e-6)" % (w, worst))
    assert worst < 1e-6


# ---- check 4: comparison with the prior-estimate table -------------------------

def test_prior_comparison_recorded():
    d = prior_comparison()
    path = write_json("prior_comparison.json", d)
    print("\ncheck4 written to %s" % path)
    for r in d["rows"]:
        print("  w=%.3f " % r["omega"] + " ".join(
            "%s %.4g/%.4g (%+.1f%%)" % (k, r["measured"][k], r["prior"][k], 100 * r["rel_diff"][k])
            for k in r["measured"]))
    assert len(d["rows"]) == 5
    assert all(np.isfinite(v) for r in d["rows"] for v in r["measured"].values())


def test_literature_comparison_recorded():
    """Orchestrator addition: beta = 4/3 validation row and the brief's expected minima."""
    from swirl.family.family import literature_comparison
    d = literature_comparison()
    path = write_json("literature_comparison.json", d)
    print("\nliterature comparison written to %s" % path)
    for r in d["beta43_rows"]:
        print("  beta=4/3 %s at w=%s: measured %.4g, literature %.4g (%+.2f%%)"
              % (r["quantity"], r.get("omega", r.get("omega_measured")), r["measured"],
                 r["literature"], 100 * r["rel_diff"]))
    for r in d["per_beta_minima"]:
        print("  beta=%.4f g4*N_min %.2f vs %.0f (%+.2f%%); omega_min %.5f vs %.5f"
              % (r["beta"], r["g4N_min_measured"], r["g4N_min_expected"], 100 * r["rel_diff"],
                 r["omega_measured"], r["omega_expected"]))
    assert len(d["beta43_rows"]) == 3 and len(d["per_beta_minima"]) == 6


@pytest.mark.parametrize("alpha2", [0.02, 0.06, 0.1])
def test_universal_curve(alpha2):
    """PRIOR_ART 1.3: Phi = a Psi(rho), r = L rho, a^2 = 20/beta, L^2 = 0.3 beta (rescaled)
    maps every beta onto one equation with alpha^2 = 0.3 beta (1 - omega^2). So
    Psi(0) and N_AS = int Psi^2 d3rho = 2 N / (omega a^2 L^3) must agree across beta."""
    vals = []
    for beta in (BETA_SET_B, 0.85, 4.0 / 3.0):
        w = math.sqrt(1.0 - alpha2 / (0.3 * beta))
        s = continue_to(beta, w)
        a, L = math.sqrt(20.0 / beta), math.sqrt(0.3 * beta)
        vals.append((s["P0"] / a, 2 * s["N"] / (w * a * a * L**3)))
    v = np.array(vals)
    rel = np.max(np.abs(v - v[0]) / np.abs(v[0]))
    print("\nuniversal curve alpha^2=%.2f: Psi0=%.6f N_AS=%.4f, max rel spread %.1e"
          % (alpha2, v[0, 0], v[0, 1], rel))
    assert rel < 1e-6


# ---- exact relation dE/domega = omega dN/domega, and tail constants ------------

@pytest.mark.parametrize("w", [0.97, 0.85, 0.75])
def test_dE_equals_omega_dN(w):
    h = 1e-4
    sp = continue_to(BETA_SET_B, w + h)
    sm = continue_to(BETA_SET_B, w - h)
    dE = (sp["E"] - sm["E"]) / (2 * h)
    dN = (sp["N"] - sm["N"]) / (2 * h)
    rel = abs(dE - w * dN) / abs(w * dN)
    print("\ndE/dw = w dN/dw at w=%.2f: rel %.2e" % (w, rel))
    assert rel < 1e-5


def test_tail_constants():
    for w in (0.99, 0.9, 0.75):
        s = continue_to(BETA_SET_B, w)
        print("\ntail w=%.2f: A=%.6g spread %.1e, kappa_fit rel err %.1e"
              % (w, s["A"], s["A_rel_spread"], s["kappa_fit_rel_err"]))
        assert s["A_rel_spread"] < 1e-5
        assert s["kappa_fit_rel_err"] < 1e-5
