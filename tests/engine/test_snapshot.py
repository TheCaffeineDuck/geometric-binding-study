"""WP-E snapshot starts: checks 2a to 2d of the WP-E task, the guards and the resume check.

Sources (built by outputs/session2/wpE/make_source.py, cached on disk, never committed), both at
dx = 0.8, dt = 0.08, t_s = 300, default sponge:
  test     N = 80,  L = 64   (sponge from 20: windows up to r_out = 20)
  testbig  N = 112, L = 89.6 (sponge from 32.8: windows up to r_out = 30)
Every test records its measured numbers to outputs/session2/wpE/checks/<name>.json.
"""

import importlib.util
import json
import os

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from swirl.engine.evolver import Evolver
from swirl.engine.initial import build_initial
from swirl.engine.lumps import LumpDiagnostics, lump_config, summarize
from swirl.engine.run import run, zero_crossings
from swirl.engine.snapshot import check, load_source, place, spectral_shift

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECK_DIR = os.path.join(ROOT, "outputs", "session2", "wpE", "checks")
PROF = json.load(open(os.path.join(ROOT, "outputs/step2/profiles/profile_beta1.3333_w0.9091.json")))


def rec(name, data):
    os.makedirs(CHECK_DIR, exist_ok=True)
    path = os.path.join(CHECK_DIR, name + ".json")
    with open(path + ".tmp", "w") as f:
        json.dump(data, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


def _build(key):
    spec = importlib.util.spec_from_file_location("wpe_make_source",
                                                  os.path.join(ROOT, "outputs/session2/wpE/make_source.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build(key, verbose=False)


@pytest.fixture(scope="session")
def src():
    return _build("test")


@pytest.fixture(scope="session")
def src_big():
    return _build("testbig")


def snap(path, meta, centers, window="meta", **kw):
    s = {"source": path, "sha256": meta["sha256"], "centers": centers,
         "window": meta["window"] if window == "meta" else window, "margin": 2.0}
    s.update(kw)
    return s


def cfg_for(meta, sn, T, N=None, L=None, lumps=None, absorber=True, threads=4, **extra):
    p = meta["params"]
    cfg = {"N": N or p["N"], "L": L or p["L"], "m": p["m"], "g4": p["g4"], "g6": p["g6"], "dt": p["dt"],
           "t_end": T, "init": {"snapshot": sn}, "absorber": p["absorber"] if absorber else None,
           "kfilter": None, "center": [0.0, 0.0, 0.0], "r_c": [10.03], "sample_every": 1.0,
           "point_every": 1, "avg_window": 20.0, "threads": threads}
    if lumps:
        cfg["lumps"] = lumps
    cfg.update(extra)
    return cfg


def source_evolver(meta, absorber=True):
    p = meta["params"]
    return Evolver(p["N"], p["L"], p["m"], p["g4"], p["g6"], absorber=p["absorber"] if absorber else None,
                   threads=4)


# ---------------------------------------------------------------------------
# 2a. continuation: bit-identical without crop; crop loss level with the crop
# ---------------------------------------------------------------------------

def _continue_source(meta, path, T):
    p = meta["params"]
    dt, N = p["dt"], p["N"]
    c = N // 2
    phi_ref, pi_ref, _ = load_source(path)
    ev = source_evolver(meta)
    ev.set_state(phi_ref, pi_ref)
    ser = [float(ev.phi[c, c, c])]
    for _ in range(int(round(T / dt))):
        ev.step(dt)
        ser.append(float(ev.phi[c, c, c]))
    return ev, ser


LUMP1 = {"centers": [[0.0, 0.0, 0.0]], "every": 5, "count_every": 1e9, "avg_window": 20.0, "it_every_step": True}


def test_snapshot_continuation_bit_identical(src):
    """Unshifted, uncropped snapshot (tau = 0, the reference step) continues bit for bit."""
    path, meta = src
    T = 100.0
    ev, ser = _continue_source(meta, path, T)
    ru = run(cfg_for(meta, snap(path, meta, [[0.0, 0.0, 0.0]], window=None, taus=[0.0]), T))
    identical_series = ru["point"]["phi"] == ser
    identical_field = bool(np.array_equal(ru["_evolver"].phi, ev.phi) and np.array_equal(ru["_evolver"].pi, ev.pi))
    rec("snapshot_continuation_bit_identical", {"T": T, "N": meta["params"]["N"], "dx": meta["dx"],
                                                "identical_point_series": identical_series,
                                                "identical_final_field": identical_field, "target": "bit-identical"})
    assert identical_series and identical_field


def _matched(a, b):
    """Pairs of whole-period averages of two single-lump runs with the same window centres."""
    out = []
    for i in range(len(a["t_center"])):
        if a["N"][i] is None or b["N"][i] is None or abs(a["t_center"][i] - b["t_center"][i]) > 1e-3:
            continue
        out.append((a["t_center"][i], a["N"][i], b["N"][i], a["E"][i], b["E"][i], a["omega"][i], b["omega"][i]))
    return np.array(out)


CROP_RECORD = {}


def test_snapshot_continuation_crop(src):
    """Cropped copy (tau = 0) against the uncropped one over T = 100, for each measured window."""
    path, meta = src
    T = 100.0
    lumps = {**LUMP1, "omega_nominal": meta["omega"]}
    tout = np.arange(0.0, T + 1e-9, 0.5)
    ru = run(cfg_for(meta, snap(path, meta, [[0.0, 0.0, 0.0]], window=None, taus=[0.0]), T, lumps=lumps))
    au = summarize({k: ru["lumps"]["raw"][k] for k in ("centers", "series", "point")}, tout, 20.0)["lumps"][0]
    Eu = np.array(ru["series"]["E_loc"])[:, 0]
    yu = np.array(ru["point"]["phi"])
    ou = np.array(ru["frequency"]["omega"])
    out = {"T": T, "N": meta["params"]["N"], "dx": meta["dx"], "windows": {},
           "targets": "frac, N, omega match the uncropped run to about the crop loss (pass within 3x)"}
    for key, cr in meta["crop"].items():
        rc = run(cfg_for(meta, snap(path, meta, [[0.0, 0.0, 0.0]], window=cr["window"], taus=[0.0]), T,
                         lumps=lumps))
        ac = summarize({k: rc["lumps"]["raw"][k] for k in ("centers", "series", "point")}, tout, 20.0)["lumps"][0]
        m = _matched(ac, au)
        Ec = np.array(rc["series"]["E_loc"])[:, 0]
        oc = np.array(rc["frequency"]["omega"])
        k = min(ou.size, oc.size)
        loss = max(cr["crop_loss_E_max"], abs(cr["crop_loss_N"]))
        dphi = np.array(rc["point"]["phi"]) - yu
        late = m[:, 0] > 60
        out["windows"][key] = {
            "crop_loss_E_max": cr["crop_loss_E_max"], "crop_loss_N": cr["crop_loss_N"], "crop_loss_max": loss,
            "max_abs_dfrac_r10": float(np.max(np.abs(Ec / Ec[0] - Eu / Eu[0]))),
            "max_rel_dN_ball_period_avg": float(np.max(np.abs(m[:, 1] / m[:, 2] - 1))),
            "max_rel_dE_ball_period_avg": float(np.max(np.abs(m[:, 3] / m[:, 4] - 1))),
            "max_rel_domega_period_avg_W20": float(np.max(np.abs(m[:, 5] / m[:, 6] - 1))),
            "max_rel_domega_per_period": float(np.max(np.abs(oc[:k] / ou[:k] - 1))),
            "max_abs_dphi_center_over_Phi0": float(np.max(np.abs(dphi))) / meta["Phi0"],
            "max_abs_dphi_center_over_Phi0_t_gt_60": float(np.max(np.abs(
                dphi[np.array(rc["point"]["t"]) > 60]))) / meta["Phi0"],
            "t_center_gt_60": {"max_rel_dN_ball_period_avg": float(np.max(np.abs(m[late, 1] / m[late, 2] - 1))),
                               "max_rel_dE_ball_period_avg": float(np.max(np.abs(m[late, 3] / m[late, 4] - 1))),
                               "max_rel_domega_period_avg_W20": float(np.max(np.abs(m[late, 5] / m[late, 6] - 1)))},
            "rel_dN_ball_last_window": float(m[-1, 1] / m[-1, 2] - 1),
            "n_matched_windows": int(m.shape[0])}
    rec("snapshot_continuation_crop", out)
    CROP_RECORD.update(out)
    w = out["windows"]["14,18"]
    assert w["n_matched_windows"] > 50
    assert w["max_abs_dfrac_r10"] < 3 * w["crop_loss_max"]


@pytest.mark.xfail(strict=True, reason="WP-E check 2a, measured miss (result, not a bug): the crop excites a "
                   "transient at first order in the cut tail amplitude Phi(r_in)/Phi0 (~1.7e-3 at r_in = 14), "
                   "while the crop loss is second order; ball N 3.5e-4 and period-averaged omega 2.2e-4 "
                   "against 3 x 5.7e-5 (window 14,18). See docs/WPE_BUILD_NOTES.md")
def test_snapshot_crop_matches_to_crop_loss(src):
    if not CROP_RECORD:
        pytest.skip("needs test_snapshot_continuation_crop in the same session")
    w = CROP_RECORD["windows"]["14,18"]
    for q in ("max_rel_dN_ball_period_avg", "max_rel_dE_ball_period_avg", "max_rel_domega_period_avg_W20",
              "max_rel_domega_per_period"):
        assert w[q] < 3 * w["crop_loss_max"], q


# ---------------------------------------------------------------------------
# 2b. off-lattice shifted copy against the unshifted copy (field shifted back)
# ---------------------------------------------------------------------------

def _evolve_compare(meta, path, delta, absorber, T=100.0, every=10.0):
    """Unshifted and delta-shifted cropped copies (same phase), evolved side by side. Field
    difference after shifting the second back by -delta; ball observables."""
    p = meta["params"]
    dt = p["dt"]
    evs, lds = [], []
    for ctr in ([0.0, 0.0, 0.0], list(delta)):
        ev = source_evolver(meta, absorber)
        phi, pi = build_initial(ev.coords, ev.L, {"snapshot": snap(path, meta, [ctr], taus=[0.0])})
        ev.set_state(phi, pi)
        evs.append(ev)
        lds.append(LumpDiagnostics(ev, lump_config({"centers": [ctr], "every": 5, "count_every": 1e9, "it_every_step": True,
                                                    "omega_nominal": meta["omega"]})))
    Phi0, om = meta["Phi0"], meta["omega"]
    rows = []
    n_every = int(round(every / dt))
    n_end = int(round(T / dt))
    for k in range(n_end + 1):
        for ld in lds:
            ld.read_points(k * dt)
            if k % 5 == 0:
                ld.sample(k * dt)
        if k % n_every == 0:
            back_phi, _ = spectral_shift(evs[1].phi, -np.asarray(delta), evs[1].dx)
            back_pi, _ = spectral_shift(evs[1].pi, -np.asarray(delta), evs[1].dx)
            dE = evs[0].energy_parts(back_phi - evs[0].phi, back_pi - evs[0].pi)["E_total"]
            rows.append({"t": k * dt,
                         "max_dphi_over_Phi0": float(np.max(np.abs(back_phi - evs[0].phi))) / Phi0,
                         "max_dpi_over_omega_Phi0": float(np.max(np.abs(back_pi - evs[0].pi))) / (om * Phi0),
                         "energy_of_difference_over_E": dE / evs[0].energy(),
                         "rel_dE": evs[1].energy() / evs[0].energy() - 1})
        if k < n_end:
            for ev in evs:
                ev.step(dt)
    tout = np.arange(0.0, T + 1e-9, 1.0)
    av = [summarize(ld.state(), tout, 20.0)["lumps"][0] for ld in lds]
    ok = [i for i in range(tout.size) if av[0]["N"][i] is not None and av[1]["N"][i] is not None]
    obs = {"max_rel_dN_ball": max(abs(av[1]["N"][i] / av[0]["N"][i] - 1) for i in ok),
           "max_rel_dE_ball": max(abs(av[1]["E"][i] / av[0]["E"][i] - 1) for i in ok),
           "max_domega": max(abs(av[1]["omega"][i] - av[0]["omega"][i]) for i in ok),
           "max_centroid_err": max(float(np.linalg.norm(np.array(av[1]["X"][i]) - np.asarray(delta)
                                                        - np.array(av[0]["X"][i]))) for i in ok),
           "n_windows": len(ok)}
    return rows, obs


def shift_record(meta, path, T=100.0, absorber=True):
    delta = np.array([0.31, -0.17, 0.23]) * meta["dx"]
    rows, obs = _evolve_compare(meta, path, delta, absorber, T=T)
    return {"delta": delta.tolist(), "dx": meta["dx"], "T": T, "rows": rows, "observables": obs,
            "t0_max_dphi_over_Phi0": rows[0]["max_dphi_over_Phi0"],
            "worst_max_dphi_over_Phi0": max(r["max_dphi_over_Phi0"] for r in rows),
            "worst_max_dpi_over_omega_Phi0": max(r["max_dpi_over_omega_Phi0"] for r in rows),
            "worst_energy_of_difference_over_E": max(r["energy_of_difference_over_E"] for r in rows),
            "worst_abs_rel_dE": max(abs(r["rel_dE"]) for r in rows), "target": 1e-8}


def test_snapshot_offlattice_shift(src):
    path, meta = src
    out = shift_record(meta, path)
    rec("snapshot_offlattice_shift", out)
    assert out["worst_max_dphi_over_Phi0"] < 1e-3      # sanity bound; the measured value is the result
    assert out["observables"]["max_centroid_err"] < 1e-3


@pytest.mark.xfail(strict=True, reason="WP-E check 2b, measured miss at the test grid (dx = 0.8): field "
                   "difference 1.1e-5 Phi0 at t = 0 and up to 1.7e-4 over T = 100 against the 1e-8 target; "
                   "see docs/WPE_BUILD_NOTES.md")
def test_snapshot_offlattice_shift_target():
    d = json.load(open(os.path.join(CHECK_DIR, "snapshot_offlattice_shift.json")))
    assert d["worst_max_dphi_over_Phi0"] < 3e-8


# ---------------------------------------------------------------------------
# 2c, 2d: equal lumps at different phases (testbig source, windows 14-18 and 26-30)
# ---------------------------------------------------------------------------

def _crossings(res, k):
    pt = res["lumps"]["raw"]["point"]
    return zero_crossings(pt["t"], np.array(pt["phi"])[:, k])


def _wavg(res, k, za, zb):
    """Averages of lump k's ball E and It over [za, zb] (exact integral of a cubic spline); It from
    the per-step record when the run has it (it_every_step)."""
    s = res["lumps"]["raw"]["series"]
    pt = res["lumps"]["raw"]["point"]
    out = []
    for key in ("E", "It"):
        if key == "It" and pt.get("It"):
            t, y = np.array(pt["t"]), np.array(pt["It"])[:, k]
        else:
            t, y = np.array(s["t"]), np.array(s[key])[:, k]
        cs = CubicSpline(t, y).antiderivative()
        out.append(float((cs(zb) - cs(za)) / (zb - za)))
    return out


def age_matched(ra, ka, rb, kb, shift, hp=6, tol=0.05):
    """Lump kb of run rb against lump ka of run ra at equal lump age: rb's lump is the source
    `shift` time units later than ra's, so its crossing z matches ra's crossing near z + shift.
    Windows of hp half periods between matched crossings. Rows:
    (t_center_b, N_a, N_b, E_a, E_b, omega_a, omega_b)."""
    za_all, zb_all = _crossings(ra, ka), _crossings(rb, kb)
    t_end = min(ra["lumps"]["raw"]["series"]["t"][-1] - shift, rb["lumps"]["raw"]["series"]["t"][-1])
    rows = []
    for i in range(zb_all.size - hp):
        z0, z1 = zb_all[i], zb_all[i + hp]
        if z1 > t_end:
            break
        j0 = int(np.argmin(np.abs(za_all - (z0 + shift))))
        if abs(za_all[j0] - (z0 + shift)) > tol or j0 + hp >= za_all.size:
            continue
        a0, a1 = za_all[j0], za_all[j0 + hp]
        Ea, Ia = _wavg(ra, ka, a0, a1)
        Eb, Ib = _wavg(rb, kb, z0, z1)
        wa, wb = np.pi * hp / (a1 - a0), np.pi * hp / (z1 - z0)
        rows.append((0.5 * (z0 + z1), Ia / wa, Ib / wb, Ea, Eb, wa, wb))
    return np.array(rows)


def _summ(m):
    return {"n_windows": int(m.shape[0]),
            "dN_over_N_first": float(m[0, 2] / m[0, 1] - 1),
            "dN_over_N_worst": float(np.max(np.abs(m[:, 2] / m[:, 1] - 1))),
            "dE_over_E_worst": float(np.max(np.abs(m[:, 4] / m[:, 3] - 1))),
            "domega_worst": float(np.max(np.abs(m[:, 6] - m[:, 5]))),
            "dfrac_worst": float(np.max(np.abs(m[:, 4] / m[0, 4] - m[:, 3] / m[0, 3])))}


WINDOWS_2CD = [[14.0, 18.0], [26.0, 30.0]]


@pytest.mark.parametrize("window", WINDOWS_2CD)
def test_snapshot_two_far_copies(src_big, window):
    path, meta = src_big
    d, th, T = 40.0, np.deg2rad(72.0), 100.0
    cen = [[-d / 2, 0.0, 0.0], [d / 2, 0.0, 0.0]]
    lumps = {"centers": cen, "every": 5, "R_ball": 10.0, "taper": 2.0, "count_every": 1e9, "avg_window": 20.0, "it_every_step": True,
             "omega_nominal": meta["omega"]}
    res = run(cfg_for(meta, snap(path, meta, cen, window=window, phases=[0.0, th]), T, N=130, L=104.0,
                      lumps=lumps, center=cen[0]))
    st = {k: res["lumps"]["raw"][k] for k in ("centers", "series", "point")}
    av = summarize(st, np.arange(0.0, T + 1e-9, 0.5), 20.0)
    dth = np.array([x for x in av["relative_phase"][0]["dtheta"] if x is not None])
    s = th / meta["omega"]
    m_age = age_matched(res, 0, res, 1, s)                 # lump 1 (72 deg) is the source s later
    A, kap = PROF["A"], PROF["kappa"]
    C_N = 2 * np.pi * A ** 2 * np.exp(-kap * d) / d
    out = {"d": d, "theta_deg": [0.0, 72.0], "window": window, "N": 130, "L": 104.0, "T": T,
           "omega_source": meta["omega"], "crop": meta["crop"]["%g,%g" % tuple(window)],
           "relative_phase_max_abs_err_rad": float(np.max(np.abs(dth - th))), "n_phase_windows": int(dth.size),
           "same_age": _summ(m_age),
           "d(N1-N0)/dt_same_age_fit": float(np.polyfit(m_age[:, 0], m_age[:, 2] - m_age[:, 1], 1)[0]),
           "d(N1-N0)/dt_predicted_interaction": -2 * C_N * np.sin(th),
           "ansatz_comparison": "WP-C: ansatz starts at -/+45 read 92.3 deg (2.3 deg offset); F-009: N up to 9.5% "
                                "lower at 90", "targets": {"relative_phase_rad": 1e-3, "dN_over_N": 1e-6}}
    rec("snapshot_two_far_copies_w%g_%g" % tuple(window), out)
    assert out["relative_phase_max_abs_err_rad"] < 3e-3
    assert out["same_age"]["n_windows"] > 10


@pytest.mark.parametrize("window", WINDOWS_2CD)
def test_snapshot_phase_control(src_big, window):
    path, meta = src_big
    T = 100.0
    lumps = {"centers": [[0.0, 0.0, 0.0]], "every": 5, "count_every": 1e9, "avg_window": 20.0, "it_every_step": True,
             "omega_nominal": meta["omega"]}
    runs = {deg: run(cfg_for(meta, snap(path, meta, [[0.0, 0.0, 0.0]], window=window, phases=[np.deg2rad(deg)]),
                             T, lumps=lumps)) for deg in (0, 72, 144, 216, 288)}
    out = {"T": T, "window": window, "omega_source": meta["omega"], "crop": meta["crop"]["%g,%g" % tuple(window)],
           "per_phase": {}}
    worst = {k: 0.0 for k in ("dN_over_N_worst", "dE_over_E_worst", "domega_worst", "dfrac_worst")}
    for deg in (72, 144, 216, 288):
        sm = _summ(age_matched(runs[0], 0, runs[deg], 0, np.deg2rad(deg) / meta["omega"]))
        out["per_phase"][str(deg)] = sm
        for k in worst:
            worst[k] = max(worst[k], sm[k])
    out["worst"] = worst
    out["F009_ansatz_for_comparison"] = {"theta90_dfrac_T1000": 2.46e-2, "theta90_domega": 7.9e-3,
                                         "criterion_dfrac": 0.01, "criterion_domega": 0.005}
    rec("snapshot_phase_control_w%g_%g" % tuple(window), out)
    assert all(v["n_windows"] > 10 for v in out["per_phase"].values())
    assert worst["dfrac_worst"] < 1e-2 and worst["domega_worst"] < 5e-3     # the F-009 criterion


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def test_snapshot_guards(src):
    path, meta = src
    p = meta["params"]
    ok = snap(path, meta, [[0.0, 0.0, 0.0]], phases=[0.0])
    g = check(ok, p["L"], p["absorber"], 1e-3)
    assert g["omega_drift"] < 1e-4 and g["crop_loss_E_max"] < 1e-4 and abs(g["crop_loss_N"]) < 1e-4
    with pytest.raises(ValueError):           # tail at the sponge (r_in = 20; lump at 8: Phi(12) ~ 4.6e-3)
        check(snap(path, meta, [[8.0, 0.0, 0.0]], phases=[0.0]), p["L"], p["absorber"], 1e-3)
    with pytest.raises(ValueError):           # tail at a face (lump at 20: Phi(12) ~ 4.6e-3 Phi0 > 1e-6)
        check(snap(path, meta, [[20.0, 0.0, 0.0]], phases=[0.0]), p["L"], None, 1e-3)
    with pytest.raises(ValueError):           # settling guard tighter than the measured drift
        check(ok, p["L"], p["absorber"], 1e-3, tol={"omega_drift": 1e-9, "crop_loss": 1e-4})
    with pytest.raises(ValueError):           # crop guard (raises above 3x its target) tighter than the loss
        check(ok, p["L"], p["absorber"], 1e-3, tol={"omega_drift": 1e-4, "crop_loss": 1e-6})
    with pytest.raises(ValueError):           # hash mismatch
        check({**ok, "sha256": "0" * 64}, p["L"], p["absorber"], 1e-3)
    coords = -p["L"] / 2 + meta["dx"] * np.arange(p["N"])
    with pytest.raises(ValueError):           # window other than the ones the crop loss was measured for
        place({**ok, "window": [12.0, 16.0]}, coords, p["L"])
    with pytest.raises(ValueError):           # different dx without resampling
        place(ok, -50.0 + 1.0 * np.arange(100), 100.0)


# ---------------------------------------------------------------------------
# resume with a snapshot start, K = 3 diagnostics and the group block
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("first_chunk", [37, 40])
def test_resume_bit_identical_snapshot_k3(src, tmp_path, first_chunk):
    path, meta = src
    a = 6.0 / np.sqrt(3.0)
    cen = [[a * np.cos(2 * np.pi * k / 3), a * np.sin(2 * np.pi * k / 3), 0.0] for k in range(3)]
    lumps = {"centers": cen, "every": 3, "R_ball": 6.0, "taper": 2.0, "count_every": 3.0, "count_periods": 1,
             "omega_nominal": 4.0, "avg_window": 5.0, "it_every_step": True,
             "axes": [{"origin": [0, 0, 0], "dir": [0, 0, 1]}], "group": {"R": 12.0, "axis": [0, 0, 1]}}
    cfg = cfg_for(meta, snap(path, meta, cen, phases=[0.0, 2 * np.pi / 3, 4 * np.pi / 3]), 8.0, lumps=lumps,
                  checkpoint_every=2.0, avg_window=4.0)
    strip = lambda r: {k: v for k, v in r.items() if k not in ("_evolver", "wall_seconds",
                                                               "steps_per_second_this_call")}
    ref = run(cfg)
    ck = str(tmp_path / "ck")
    part = run(cfg, checkpoint_dir=ck, max_steps=first_chunk)
    assert part["completed"] is False
    run(cfg, checkpoint_dir=ck, max_steps=23)
    res = run(cfg, checkpoint_dir=ck)
    assert res["completed"] is True
    assert np.array_equal(ref["_evolver"].phi, res["_evolver"].phi)
    assert len(ref["lumps"]["counts"]) >= 2
    assert len(ref["lumps"]["raw"]["series"]["E_group"]) > 10
    assert json.dumps(strip(ref)) == json.dumps(strip(res))
