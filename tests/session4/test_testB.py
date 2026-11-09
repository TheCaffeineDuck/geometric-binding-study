"""Session 4, Test B: dispersion (B1), probe recording, spectra, floors, flux, verdict (B4).

Measured values are written to outputs/session4/testB/test_values.json (one key per test).
"""

import glob
import json
import math
import os
import time

import numpy as np
import pytest

from swirl.family.radial import Radial
from swirl.session4.testB import analysis as A
from swirl.session4.testB import dispersion as D
from swirl.session4.testB import run as RUN

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VALS = os.path.join(ROOT, "outputs", "session4", "testB", "test_values.json")


def record(key, val):
    os.makedirs(os.path.dirname(VALS), exist_ok=True)
    d = json.load(open(VALS)) if os.path.exists(VALS) else {}
    d[key] = val
    tmp = VALS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, VALS)


# ------------------------------------------------------------------ B1
def test_rk4_formula_exact_closed_form():
    # arg R(i theta) series: theta - theta^5/120 + O(theta^7); |R|^2 = 1 - theta^6/72 + theta^8/576 exactly
    th = 1e-2
    assert abs(D.rk4_omega(1.0, th) * th - (th - th**5 / 120)) < 1e-15
    assert abs(D.rk4_amp2(1.0, th) - (1 - th**6 / 72 + th**8 / 576)) < 5e-16
    # 4th-order stencil symbol: k^2 (1 - (k h)^4 / 90) at small k h
    k, h = 0.3, 0.05
    assert abs(D.K2_radial(k, h) / k**2 - (1 - (k * h) ** 4 / 90)) < 1e-12


def test_threshold_is_m_in_all_solvers():
    rows = {}
    for name, dt in (("radial", 0.0125), ("radial_half", 0.00625), ("engine_v2", 0.04), ("legacy", 0.05)):
        w = D.threshold(1.0, dt)
        rows[name] = dict(dt=dt, omega_th=w, minus_m=w - 1)
        assert abs(w - 1) < 1e-6          # B-P1 target
    record("threshold", rows)


def test_radial_eigenmodes_match_dispersion():
    rows = []
    for n, nl in ((1, False), (1, True), (100, False), (400, False), (799, False)):
        r = D.measure_radial_mode(n, nonlinear=nl)
        rel = r["omega_meas"] / r["omega_pred"] - 1
        rows.append(dict(n=n, nonlinear=nl, kdr=r["kdr"], omega_pred=r["omega_pred"], omega_meas=r["omega_meas"],
                         rel=rel, amp2_pred=r["amp2_per_step_pred"], amp2_meas=r["amp2_per_step_meas"]))
        assert abs(rel) < 1e-12
        assert abs(r["amp2_per_step_meas"] - r["amp2_per_step_pred"]) < 1e-12
    record("radial_eigenmodes", rows)


def test_engine_v2_and_legacy_plane_waves_match_dispersion():
    rows = []
    for nv in ((0, 0, 0), (1, 0, 0), (3, 2, 1), (8, 0, 0), (8, 8, 8)):
        r = D.measure_engine_v2_mode(nv)
        rows.append(dict(solver="engine_v2", **{k: r[k] for k in ("n_vec", "k", "omega_pred", "omega_meas")}))
        assert abs(r["omega_meas"] / r["omega_pred"] - 1) < 1e-12
    t0 = time.time()
    for nv in ((0, 0, 0), (1, 0, 0), (8, 0, 0), (8, 8, 8)):
        r = D.measure_legacy_mode(nv)
        rows.append(dict(solver="legacy", Omega_no_sigma=r["Omega_semi_without_sigma"],
                         **{k: r[k] for k in ("n_vec", "k", "omega_pred", "omega_meas")}))
        assert abs(r["omega_meas"] / r["omega_pred"] - 1) < 1e-12
        if nv != (0, 0, 0):
            assert r["Omega_semi"] > r["Omega_semi_without_sigma"]      # stiffness raises omega at high k
    # importing the legacy module must not write into legacy/
    new = [f for f in glob.glob(os.path.join(ROOT, "legacy", "**", "*.pyc"), recursive=True)
           if os.path.getmtime(f) >= t0 - 1]
    assert not new
    record("plane_waves_3d", rows)


# ------------------------------------------------------------------ probes
def _tiny_params(tmp):
    return dict(out_dir=str(tmp / "out"), ckpt_dir=str(tmp / "ck"), verbose=False)


def test_probe_recording_does_not_change_dynamics(tmp_path):
    T = 40.0
    RUN.run("b1_smoke", T=T, ckpt_every=1e9, **_tiny_params(tmp_path))
    z = np.load(tmp_path / "ck" / "b1_smoke.npz")
    phi, _ = RUN.start_profile("primary")
    g = RUN.BASE_GRID
    finals = []
    for chunk in (int(round(T / g["dt"])), 40):          # one call; Session 1 chunking (0.5)
        s = Radial(g["R_box"], g["dr"], 1.0, RUN.G4, RUN.BETA * RUN.G4**2, sponge_start=g["R_sponge"],
                   sigma_max=g["sigma_max"])
        s.set_phi(phi)
        for _ in range(int(round(T / g["dt"])) // chunk):
            s.step(g["dt"], chunk)
        finals.append((s.u.copy(), s.v.copy()))
    for u, v in finals:
        assert np.array_equal(u, z["u"]) and np.array_equal(v, z["v"])
    record("probe_non_interference", dict(T=T, bit_identical=True, compared="driver with probes vs one "
                                          "Radial.step call vs 40-step chunks"))


def test_probe_values_against_analytic_field():
    s = Radial(60.0, 0.05, 1.0, 0.0, 0.0)
    # an even (regular) radial field
    f = lambda r: (np.exp(-((r - 20) / 6) ** 2) + np.exp(-((r + 20) / 6) ** 2) + 0.3 * np.exp(-r * r / 8))
    fr = lambda r: (-2 * (r - 20) / 36 * np.exp(-((r - 20) / 6) ** 2) - 2 * (r + 20) / 36 * np.exp(-((r + 20) / 6) ** 2)
                    - 0.3 * r / 4 * np.exp(-r * r / 8))
    s.set_phi(f, lambda r: 2 * f(r))
    radii = [0.0, 10.0, 15.0, 20.0, 30.0, 45.0]
    phi, pt, pr = RUN.read_probes(s, RUN.probe_index(s, radii))
    rr = np.array(radii)
    e_phi = np.max(np.abs(phi - f(rr)))
    e_pt = np.max(np.abs(pt - 2 * f(rr)))
    e_pr = np.max(np.abs(pr - fr(rr)))
    record("probe_values", dict(err_phi=e_phi, err_phit=e_pt, err_phir=e_pr))
    assert e_phi < 1e-6 and e_pt < 2e-6 and e_pr < 1e-6


def test_resume_is_bit_identical(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    RUN.run("b1_smoke", T=40.0, **_tiny_params(a))
    RUN.run("b1_smoke", T=40.0, stop_after=20.0, **_tiny_params(b))
    RUN.run("b1_smoke", T=40.0, **_tiny_params(b))
    za, zb = np.load(a / "out" / "b1_smoke_series.npz"), np.load(b / "out" / "b1_smoke_series.npz")
    for k in ("phi", "phit", "phir", "E_in", "E_h", "nl_share"):
        assert np.array_equal(za[k], zb[k]), k


def _energy_inside_4th(s, R):
    """4 pi int_0^R e r^2 dr with phi_r from the 4th-order u_r (as the probes) and Simpson."""
    i1 = int(round(R / s.dr))
    idx = np.arange(0, i1 + 1)
    idx[0] = 1                                   # r = 0 carries zero weight (r^2)
    phi, pt, pr = RUN.read_probes(s, idx)
    e = 0.5 * pt**2 + 0.5 * pr**2 + 0.5 * s.m**2 * phi**2
    f = 4 * np.pi * s.r[: i1 + 1] ** 2 * e
    f[0] = 0.0
    from scipy.integrate import simpson
    return float(simpson(f, dx=s.dr))


def test_probe_flux_closes_solver_energy_balance():
    """Linear outgoing packet in the radial solver: time integral of 4 pi r^2 (-phi_t phi_r)
    at r = 20 (trapezoid, every 0.2) against the drop of E_in(20) (4th-order density, Simpson)."""
    s = Radial(80.0, 0.05, 1.0, 0.0, 0.0, sponge_start=50.0, sigma_max=1.0)
    s.set_phi(lambda r: 0.2 * np.exp(-r * r / 4) * np.cos(2.5 * r))
    idx = RUN.probe_index(s, [20.0])
    Ein0 = [_energy_inside_4th(s, 20.0)]
    fl = []
    for k in range(int(40 / 0.2) + 1):
        if k:
            s.step(0.0125, 16)
        _, pt, pr = RUN.read_probes(s, idx)
        fl.append(4 * np.pi * 400 * (-pt[0] * pr[0]))
    out = np.trapezoid(np.array(fl), dx=0.2)
    Ein1 = [_energy_inside_4th(s, 20.0)]
    rel = out / (Ein0[0] - Ein1[0]) - 1
    record("solver_flux_balance", dict(flux_integral=out, energy_drop=float(Ein0[0] - Ein1[0]), rel=float(rel)))
    assert abs(rel) < 2e-3


# ------------------------------------------------------------------ spectra on synthetic signals
def test_welch_normalisation_and_lines():
    dt, T = 0.2, 5000.0
    t = dt * np.arange(int(T / dt) + 1)
    wc = 0.9091
    comps = [(1, 3.0, 0.1), (3, 1e-3, 1.0), (5, 1e-4, 2.0), (2, 1e-6, 0.5)]   # Hann skirt of the unit
    # line 290 bins away is ~ -155 dB, so an even line at 1e-6 (power 1e-13 of it) is resolvable
    x = sum(a * np.cos(n * wc * t + p) for n, a, p in comps) + 2e-6 * np.cos(1.37 * t)
    om, P, _, nseg = A.welch(x, dt, 0, T)
    assert nseg == 4
    assert abs(np.sum(P) / np.mean(x[:-1] ** 2) - 1) < 2e-3
    lines = A.lines_in(om, P, wc)
    got = {d["label"]: d for d in lines}
    rows = []
    for n, a, _ in comps:
        d = got["%d omega" % n]
        rows.append(dict(n=n, amp=a, amp_meas=d["amp"], omega_err_bins=(d["omega"] - n * wc) / A.DOMEGA))
        assert abs(d["amp"] / a - 1) < 0.01
        assert abs(d["omega"] - n * wc) < 0.1 * A.DOMEGA
    un = [d for d in lines if d["label"] == "unidentified"]
    assert len(un) == 1 and abs(un[0]["omega"] - 1.37) < 0.1 * A.DOMEGA
    # no sidelobe of the 3 omega line is reported as a line
    assert all(abs(d["omega"] - 3 * wc) > 10 * A.DOMEGA for d in lines if d["label"] != "3 omega")
    assert len(lines) == 5
    # a harmonic above the probe Nyquist frequency (pi / 0.2) is labelled as its alias; a
    # near-threshold frequency is not
    wn = np.pi / dt
    assert A.label(2 * wn - 19 * wc, wc, A.DOMEGA, wn)[1] == "19 omega aliased"
    assert A.label(1.0011, wc, A.DOMEGA, wn)[1] == "unidentified"
    assert A.label(1.0011, wc, A.DOMEGA, wn, near=(1 - 3 * A.DOMEGA, 1 + 3 * A.DOMEGA))[1] == "near-threshold band"
    record("synthetic_lines", rows)


def test_leakage_floor_estimator_on_synthetic_near_threshold_content():
    dt, T = 0.2, 2000.0
    t = dt * np.arange(int(T / dt) + 1)
    cut = 1 - 3 * A.DOMEGA
    om, wk, R, L = A.tone_response(dt, 0, T, t.size, cut)
    sigs = dict(pure_1005=np.cos(1.005 * t), pure_10003=np.cos(1.0003 * t),
                chirp=np.cos(1.0064 * t - 0.5 * (0.0061 / T) * t**2),
                kg_r45=np.cos(np.sqrt(np.maximum(t**2 - 45**2, 0))) * (t > 45) / np.maximum(t, 45) ** 1.5,
                kg_r120=np.cos(np.sqrt(np.maximum(t**2 - 120**2, 0))) * (t > 120) / np.maximum(t, 120) ** 1.5)
    rows = {}
    for k, x in sigs.items():
        _, P, _, _ = A.welch(x, dt, 0, T)
        true = float(np.sum(P[om < cut]))
        est = A.near_threshold_leak(om, P, R, L, cut)
        rows[k] = dict(true=true, est=est, ratio=est / true)
        assert 0.5 <= est / true <= 10
    sp, tp = A.synth_sub_power(1.005, 1.0, dt, 0, T, t.size, cut)
    rows["synthetic_1005_sub_fraction"] = sp / tp
    record("leakage_estimator", rows)


def test_flux_formula_on_analytic_outgoing_spherical_wave():
    dt, T = 0.2, 5000.0
    t = dt * np.arange(int(T / dt) + 1)
    rows = []
    for r, w, a in ((30.0, 2.7273, 0.015), (120.0, 2.7273, 0.015), (60.0, 4.5455, 0.002)):
        k = math.sqrt(w * w - 1)
        ph = w * t - k * r
        phit = -(a / r) * w * np.sin(ph)
        phir = (a / r) * k * np.sin(ph) - (a / r**2) * np.cos(ph)
        phi = (a / r) * np.cos(ph)
        exact = 2 * np.pi * a * a * w * k
        om, P, _, _ = A.welch(phi, dt, 0, T)
        _, _, C, _ = A.welch(phit, dt, 0, T, y=phir)
        d = A.lines_in(om, P, w, C=C)[0]
        m1 = -4 * np.pi * r * r * d["cross"]
        m2 = 2 * np.pi * (r * d["amp"]) ** 2 * w * k
        direct = 4 * np.pi * r * r * np.mean(-phit * phir)
        rows.append(dict(r=r, omega=w, exact=exact, m1=m1, m2=m2, direct=direct))
        assert abs(m1 / exact - 1) < 1e-3 and abs(m2 / exact - 1) < 1e-3 and abs(direct / exact - 1) < 1e-3
    # a standing wave and an evanescent field carry no flux
    r, w = 30.0, 2.7273
    k = math.sqrt(w * w - 1)
    st_t = -np.sin(k * r) / r * w * np.sin(w * t)
    st_r = (k * np.cos(k * r) / r - np.sin(k * r) / r**2) * np.cos(w * t)
    assert abs(np.mean(st_t * st_r)) * 4 * np.pi * r * r < 1e-4 * 2 * np.pi * w * k   # finite-record residual ~ 1/(2 w T)
    record("flux_formula", rows)


def test_drifting_and_split_lines_flux_adds_up():
    """Lines drifting within the epoch (3 omega by 0.04, 13 bins; 5 omega by 0.067), a line
    split into two tones 4 bins apart, and a near-threshold chirp: the bands are disjoint,
    each drifting line is one line, and the summed line flux equals the total flux to 1e-3."""
    dt, T = 0.2, 5000.0
    t = dt * np.arange(int(T / dt) + 1)
    wc0, wc1 = 0.9198, 0.9332                 # core omega drifting across the epoch
    wct = wc0 + (wc1 - wc0) * t / T
    ph_c = wc0 * t + 0.5 * (wc1 - wc0) * t * t / T
    comps = [(3, 1.5e-2, 3 * ph_c, 3 * wct), (5, 2.3e-4, 5 * ph_c, 5 * wct),
             ("split_a", 1e-4, 1.8 * t, 1.8 + 0 * t), ("split_b", 1e-4, (1.8 + 4 * A.DOMEGA) * t + 1.0,
                                                        1.8 + 4 * A.DOMEGA + 0 * t),
             ("near", 3e-3, 1.008 * t - 0.5 * 0.0075 * t * t / T, 1.008 - 0.0075 * t / T)]
    rows = []
    for r in (30.0, 120.0):
        phi = np.zeros_like(t)
        pt = np.zeros_like(t)
        pr = np.zeros_like(t)
        for _, a, ph, w in comps:
            k = math.sqrt(np.mean(w) ** 2 - 1)
            phi += a / r * np.cos(ph - k * r)
            pt += -a / r * w * np.sin(ph - k * r)
            pr += a / r * k * np.sin(ph - k * r) - a / r**2 * np.cos(ph - k * r)
        om, P, _, _ = A.welch(phi, dt, 0, T)
        _, _, C, _ = A.welch(pt, dt, 0, T, y=pr)
        lines = A.lines_in(om, P, 0.5 * (wc0 + wc1), (wc0, wc1), C=C,
                           near=(1 - 3 * A.DOMEGA, 1 + 3 * A.DOMEGA))
        bands = sorted(tuple(d["band"]) for d in lines)
        assert all(b[1] <= c[0] for b, c in zip(bands, bands[1:]))          # disjoint
        total = -4 * np.pi * r * r * float(np.sum(C))
        in_lines = -4 * np.pi * r * r * sum(d["cross"] for d in lines)
        labels = [d["label"] for d in lines]
        assert labels.count("3 omega") == 1 and labels.count("5 omega") == 1
        assert labels.count("near-threshold band") == 1
        split = [d for d in lines if abs(d["omega_centroid"] - (1.8 + 2 * A.DOMEGA)) < 3 * A.DOMEGA]
        assert len(split) == 1 and len(split[0]["merged_peaks"]) == 2
        rows.append(dict(r=r, total=total, in_lines=in_lines, rel=in_lines / total - 1, labels=labels,
                         bands=bands))
        assert abs(in_lines / total - 1) < 1e-3
    record("drifting_split_lines", rows)


# ------------------------------------------------------------------ verdict on synthetic data
def _synthetic_run(tmp, name, loss=0.01, sub=None, sub_amp_exp=-1.0, T=5000.0):
    """Write a series npz in the driver's format: primary-like core (r = 0), outgoing 3 omega
    and 5 omega lines a_n / r carrying `loss`, evanescent fundamental, E_in falling at `loss`.
    sub = (omega_s, b, k_s): an outgoing sub-threshold line (b / r^-sub_amp_exp) cos(omega_s t - k_s r)."""
    dt = 0.2
    t = dt * np.arange(int(T / dt) + 1)
    probes = np.array(RUN.PROBES)
    wc, A0, kap = 0.9096, 56.7, math.sqrt(1 - 0.9096**2)
    w3, w5 = 3 * wc, 5 * wc
    k3, k5 = math.sqrt(w3**2 - 1), math.sqrt(w5**2 - 1)
    # 3 omega carries 99%, 5 omega 1%
    a3 = math.sqrt(0.99 * loss / (2 * np.pi * w3 * k3))
    a5 = math.sqrt(0.01 * loss / (2 * np.pi * w5 * k5))
    phi = np.zeros((t.size, probes.size))
    pt = np.zeros_like(phi)
    pr = np.zeros_like(phi)
    for i, r in enumerate(probes):
        if r == 0:
            phi[:, i] = 6.7 * np.cos(wc * t) + 0.46 * np.cos(w3 * t) + 0.09 * np.cos(w5 * t)
            continue
        for a, w, k in ((a3, w3, k3), (a5, w5, k5)):
            ph = w * t - k * r
            phi[:, i] += a / r * np.cos(ph)
            pt[:, i] += -a / r * w * np.sin(ph)
            pr[:, i] += a / r * k * np.sin(ph) - a / r**2 * np.cos(ph)
        ev = A0 * math.exp(-kap * r) / r
        phi[:, i] += ev * np.cos(wc * t)
        pt[:, i] += -ev * wc * np.sin(wc * t)
        pr[:, i] += -(kap + 1 / r) * ev * np.cos(wc * t)
        if sub is not None:
            ws, b, ks = sub
            amp = b * r ** sub_amp_exp
            ph = ws * t - ks * r
            phi[:, i] += amp * np.cos(ph)
            pt[:, i] += -amp * ws * np.sin(ph)
            pr[:, i] += amp * ks * np.sin(ph)
    te = np.arange(0.0, T + 0.5, 1.0)
    E0 = 2843.77
    E_in = np.repeat((E0 - loss * te)[:, None], probes.size, axis=1)
    params = dict(m=1.0, dt=0.0125, profile=dict(A=A0, kappa=kap), name=name, synthetic=True)
    os.makedirs(tmp, exist_ok=True)
    np.savez(os.path.join(tmp, name + "_series.npz"), t=t, probes=probes, phi=phi, phit=pt.astype(np.float32),
             phir=pr.astype(np.float32), t_energy=te, r_c_energy=probes, E_in=E_in, E_tot=E_in[:, 0],
             E_h=E0 - loss * te, absorbed=loss * te, t_nl=te[::10], nl_share=np.zeros(te[::10].size),
             params=json.dumps(params))


def test_verdict_on_synthetic_data(tmp_path):
    d = str(tmp_path)
    out = {}
    # 1. clean: only above-threshold lines, flux = loss
    _synthetic_run(d, "clean")
    an = A.analyse("clean", series_dir=d)
    v = A.verdict(an)
    out["clean"] = dict(verdict=v["overall"], c=[v["per_epoch"][0][k] for k in ("c1", "c2", "c3")],
                        ratio=[r["ratio_m1"] for r in v["per_epoch"][0]["c3_rows"]])
    assert v["overall"] == "Emission only above threshold"
    # 2. a propagating sub-threshold line, 1/r, at 1e-3 of the 3 omega amplitude, with a box2 that shows it too
    sub = (0.95, 2e-5, 0.3)
    _synthetic_run(d, "sub", sub=sub)
    _synthetic_run(d, "sub_box2", sub=sub)
    an_s, an_b = A.analyse("sub", series_dir=d), A.analyse("sub_box2", series_dir=d)
    v = A.verdict(an_s, an_b)
    g = an_s["epochs"][0]["sub_groups"]
    out["sub_present"] = dict(verdict=v["overall"], groups=[(x["omega"], x["n_strong"], x["exponent"]) for x in g])
    assert v["overall"] == "Sub-threshold emission present"
    assert any(abs(x["omega"] - 0.95) < A.DOMEGA and abs(x["exponent"] + 1) < 0.05 for x in g)
    # 3. same line, no doubled box: cannot be confirmed -> undetermined
    v = A.verdict(an_s)
    out["sub_no_box2"] = v["overall"]
    assert v["overall"] == "Undetermined" and v["per_epoch"][0]["present_a_unconfirmed_by_box2"]
    # 4. a bound sub-threshold line 400 e^(-kappa r)/r, above the floors at the far probes:
    # criterion 2 passes (bound), it is never "present"; criterion 1 fails (its power is not
    # covered by the fundamental's evanescent floor), so B4 as written gives "Undetermined"
    _synthetic_run(d, "bound", sub=(0.95, 400.0, 0.0), sub_amp_exp=-1.0)
    z = dict(np.load(os.path.join(d, "bound_series.npz")))
    ks = math.sqrt(1 - 0.95**2)
    for i, r in enumerate(z["probes"]):
        if r > 0:
            extra = 400.0 * (math.exp(-ks * r) - 1) / r * np.cos(0.95 * z["t"])
            z["phi"][:, i] += extra
    np.savez(os.path.join(d, "bound_series.npz"), **z)
    an_b = A.analyse("bound", series_dir=d)
    v = A.verdict(an_b)
    gb = [g for g in an_b["epochs"][0]["sub_groups"] if abs(g["omega"] - 0.95) < A.DOMEGA]
    out["bound"] = dict(verdict=v["overall"], c1=v["per_epoch"][0]["c1"], c2_rows=v["per_epoch"][0]["c2_rows"])
    assert gb and gb[0]["n_probes"] >= 2 and gb[0]["exponent"] < -1.3 and gb[0]["consistent_evanescent"]
    assert v["per_epoch"][0]["c2"] and not v["per_epoch"][0]["c1"] and v["overall"] == "Undetermined"
    # 5. loss 10x the line flux, nothing below threshold -> undetermined (c3 fails)
    _synthetic_run(d, "short", loss=0.01)
    z = dict(np.load(os.path.join(d, "short_series.npz")))
    z["E_in"] = np.repeat((2843.77 - 0.1 * z["t_energy"])[:, None], z["probes"].size, axis=1)
    np.savez(os.path.join(d, "short_series.npz"), **z)
    v = A.verdict(A.analyse("short", series_dir=d))
    out["flux_short"] = dict(verdict=v["overall"], c3=v["per_epoch"][0]["c3"])
    assert v["overall"] == "Undetermined" and not v["per_epoch"][0]["c3"]
    # 6. clause B: line flux short by 10x and a sub-threshold line carries the deficit
    ws, ks = 0.95, 0.3
    b = math.sqrt(0.09 / (2 * np.pi * ws * ks))      # outgoing flux 0.09 = 0.1 - 0.01
    _synthetic_run(d, "carry", loss=0.01, sub=(ws, b, ks))
    z = dict(np.load(os.path.join(d, "carry_series.npz")))
    z["E_in"] = np.repeat((2843.77 - 0.1 * z["t_energy"])[:, None], z["probes"].size, axis=1)
    np.savez(os.path.join(d, "carry_series.npz"), **z)
    v = A.verdict(A.analyse("carry", series_dir=d))
    out["clause_b"] = dict(verdict=v["overall"], rows=v["per_epoch"][0]["present_b_rows"])
    assert v["overall"] == "Sub-threshold emission present"
    record("verdict_synthetic", out)
