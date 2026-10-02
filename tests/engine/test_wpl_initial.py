"""WP-L L2 (Session 3): 3D initial data from a settled-lump snapshot. Checks: the harmonic decomposition
reproduces the samples and the states between them; the boosted slice carries gamma E0 and gamma v E0; the
quadrupolar stretch against its closed form; the random multiplier's Delta N; placement on a grid; two-lump
data against the single lumps; the engine accepts the entry; the recorded 3D gate."""

import json
import os

import numpy as np
import pytest

from swirl.engine import boost
from swirl.engine.snapshot import _radial_grid, load_source, phase_states, phase_taus, window_profile

SRC = "/Users/aaronramos/Desktop/SWIRL/outputs/session2/wpE/" + "sou" + "rce/src_testbig_N112_L89.6.npz"
HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(HERE, "..", "..", "outputs", "session3", "wpL", "gate.json")
M_, G4, G6 = 1.0, 0.30, 0.12

pytestmark = pytest.mark.skipif(not os.path.exists(SRC), reason="WP-E test source not available")


@pytest.fixture(scope="module")
def H():
    return boost.harmonics(SRC, threads=2)


@pytest.fixture(scope="module")
def ref(H):
    return boost.rest_reference(H, M_, G4, G6)


def test_harmonics_reproduce_and_interpolate(H):
    phi, pt = boost.reconstruct(H, 0.0)
    e0 = max(np.max(np.abs(phi - H["sample0"][0])), np.max(np.abs(pt - H["sample0"][1])))
    src = load_source(SRC)
    meta = src[2]
    Mc = H["Mc"]
    w = window_profile(_radial_grid(2 * Mc, H["dx"], Mc), *H["window"])
    c = meta["params"]["N"] // 2
    sl = slice(c - Mc, c + Mc)
    out = {}
    ths = [2 * np.pi * (k + 0.5) / H["M"] for k in (0, 5, 11, 20, 27)]
    taus = {k: t for k, t in enumerate(phase_taus(meta, ths))}
    phase_states(src, taus, lambda k, p, q, rec: out.__setitem__(k, (p[sl, sl, sl] * w, q[sl, sl, sl] * w)), 2)
    err_p, err_v = 0.0, 0.0
    for k, th in enumerate(ths):
        p, q = boost.reconstruct(H, th)
        err_p = max(err_p, np.max(np.abs(p - out[k][0])) / H["Phi0"])
        err_v = max(err_v, np.max(np.abs(q - out[k][1])) / (H["Phi0"] * H["omega"]))
    print("\nharmonics: at a sample %.1e; between samples phi %.2e Phi0, phi_t %.2e omega Phi0" % (e0, err_p, err_v))
    assert e0 < 1e-12 and err_p < 1e-4 and err_v < 1e-4


@pytest.mark.parametrize("v", [0.05, 0.2])
def test_boost_t0_energy_momentum(H, ref, v):
    g = 1 / np.sqrt(1 - v * v)
    worst_E, worst_P = 0.0, 0.0
    for th in (0.0, 1.0, 2.5):
        cv = boost.cube_values(*boost.lump_cube(H, th, v=v), H["dx"], M_, G4, G6, H["omega"])
        eE = cv["E_total"] / (g * ref["E0"]) - 1
        eP = cv["P"][0] / (g * v * ref["E0"]) - 1
        worst_E, worst_P = max(worst_E, abs(eE)), max(worst_P, abs(eP))
        assert abs(cv["P"][1]) < 1e-8 and abs(cv["P"][2]) < 1e-8
    print("\nv = %g: E / (gamma E0) - 1 worst %.2e; P_x / (gamma v E0) - 1 worst %.2e (rest E over phases %.2e)"
          % (v, worst_E, worst_P, (ref["E_max"] - ref["E_min"]) / ref["E0"]))
    assert worst_E < 1e-4 and worst_P < 3e-4


@pytest.mark.parametrize("eps", [0.05, 0.1, -0.1])
def test_stretch_closed_form(H, eps):
    base = boost.cube_values(*boost.lump_cube(H, 0.0), H["dx"], M_, G4, G6, H["omega"])
    cs = boost.cube_values(*boost.lump_cube(H, 0.0, quad_eps=eps), H["dx"], M_, G4, G6, H["omega"])
    ex = boost.stretch_expected(base, eps)
    dE = cs["E_total"] - base["E_total"]
    dN = (cs["N_inst"] - base["N_inst"]) / base["N_inst"]
    print("\nstretch %g: dE %.6f closed form %.6f (rel %.2e), dE/E %.2e, dN/N %.2e" % (
        eps, dE, ex["dE"], dE / ex["dE"] - 1, dE / base["E_total"], dN))
    # Delta N = 0 in the continuum (det S = 1); on the grid the stretched window edge carries 1e-8-level
    # interpolation error at dx = 0.8
    assert abs(dE / ex["dE"] - 1) < 1e-3 and abs(dN) < 1e-7


@pytest.mark.parametrize("amp", [0.01, 0.05])
def test_random_multiplier(H, amp):
    base = boost.cube_values(*boost.lump_cube(H, 0.0), H["dx"], M_, G4, G6, H["omega"])
    cs = boost.cube_values(*boost.lump_cube(H, 0.0, random=dict(amp=amp, ell=3.0, seed=1)), H["dx"], M_, G4, G6, H["omega"])
    dN = cs["N_inst"] / base["N_inst"] - 1
    print("\nrandom multiplier a = %g: dN/N %.6e (a^2 = %.1e), dE/E %.3e, P %s" % (
        amp, dN, amp ** 2, cs["E_total"] / base["E_total"] - 1, cs["P"]))
    assert abs(dN / amp ** 2 - 1) < 1e-8


def test_placement_and_two_lumps(H, ref):
    from swirl.engine.evolver import Evolver
    from swirl.engine.initial import build_initial
    ev = Evolver(112, 89.6, M_, G4, G6, threads=2)
    one = dict(source=SRC, lumps=[dict(center=[0.37, -0.21, 0.13], phase=0.4, v=0.1)])
    p1, v1 = build_initial(ev.coords, ev.L, {"wpl": one})
    cube = boost.cube_values(*boost.lump_cube(H, 0.4, v=0.1), H["dx"], M_, G4, G6, H["omega"])
    e1 = ev.energy_parts(p1, v1)["E_total"]
    print("\nplacement with a sub-grid shift: E grid / E cube - 1 = %.2e" % (e1 / cube["E_total"] - 1))
    assert abs(e1 / cube["E_total"] - 1) < 1e-6
    rows = []
    for d in (20.0, 40.0):
        for ph in (0.0, 90.0, 180.0):
            ent = boost.two_lump_entry(SRC, d, ph, 0.05)
            pa, va = build_initial(ev.coords, ev.L, {"wpl": dict(ent, lumps=ent["lumps"][:1])})
            pb, vb = build_initial(ev.coords, ev.L, {"wpl": dict(ent, lumps=ent["lumps"][1:])})
            pt, vt = build_initial(ev.coords, ev.L, {"wpl": ent})
            assert np.array_equal(pt, pa + pb) and np.array_equal(vt, va + vb)
            Et, Ea, Eb = (ev.energy_parts(a, b)["E_total"] for a, b in ((pt, vt), (pa, va), (pb, vb)))
            g = 1 / np.sqrt(1 - 0.05 ** 2)
            rows.append((d, ph, (Et - Ea - Eb) / (Ea + Eb), (Ea + Eb) / (2 * g * ref["E0"]) - 1))
    for r in rows:
        print("two lumps d = %g, phase %g: cross term / (E_A + E_B) = %+.2e; (E_A + E_B)/(2 gamma E0) - 1 = %+.2e" % r)
    assert all(abs(r[2]) < 1e-4 for r in rows if r[0] == 40.0)
    assert all(abs(r[3]) < 1e-4 for r in rows)


def test_engine_accepts_wpl_entry():
    from swirl.engine.evolver import SPONGE_DEFAULT
    from swirl.engine.run import run
    cfg = dict(N=112, L=89.6, m=M_, g4=G4, g6=G6, dt=0.08, t_end=0.8, absorber=dict(SPONGE_DEFAULT),
               init={"wpl": dict(source=SRC, lumps=[dict(center=[-6.0, 0.0, 0.0], phase=0.0, v=0.05)])},
               lumps=dict(centers=[[-6.0, 0.0, 0.0]], omega_nominal=0.9095), threads=2)
    res = run(cfg)
    assert res["completed"] and "wpl" in res["init_info"] and res["lumps"]["guards"][0]["wpl"]["lumps"]
    with pytest.raises(ValueError):
        run(dict(cfg, init={"wpl": dict(source=SRC, lumps=[dict(center=[-30.0, 0.0, 0.0], v=0.05)])},
                 lumps=dict(centers=[[-30.0, 0.0, 0.0]])))


def test_gate_recorded():
    if not os.path.exists(GATE):
        pytest.skip("gate.json not made (outputs/session3/wpL/l2_gate.py)")
    g = json.load(open(GATE))
    for k, r in g["results"].items():
        print("\n%s: speed %.6f (rel err %.2e); E/(gamma E0) - 1 %.2e (vs rest run %.2e); omega ratio %.6f vs 1/gamma "
              "%.6f (err %.2e); ball-energy drift minus rest %.2e; I_xx/I_yy drift %.2e (rest %.2e)"
              % (k, r["speed_measured"], r["speed_rel_err"], r["E_ratio_vs_E0ref_err"], r["E_ratio_vs_rest_run_err"],
                 r["omega_ratio"], 1 / r["gamma"], r["omega_ratio_err"], r["E_ball_drift_minus_rest"],
                 r["contraction_drift"], r["contraction_drift_rest"]))
        assert abs(r["speed_rel_err"]) < 1e-2
        assert abs(r["E_ratio_vs_E0ref_err"]) < 1e-4
        assert abs(r["omega_ratio_err"]) < 1e-4
        assert abs(r["E_ball_drift_minus_rest"]) < 1e-3
