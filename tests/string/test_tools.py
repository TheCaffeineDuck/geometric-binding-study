"""Smoke tests of the production tools (a to f) and the Block 4 additions.

Each states what failure would look like in its docstring."""

import json
import os

import numpy as np
import pytest

from swirl.string import driver as DR
from swirl.string import jump as J
from swirl.string import orbits as O


def _cleanup(*names):
    for n in names:
        for f in (n + ".json", n + "_series.npz", n + "_twin.json", n + "_twin_series.npz"):
            p = os.path.join(DR.OUT, f)
            if os.path.exists(p):
                os.remove(p)


def test_family_grid_G1_smoke(record):
    """(a) Short family of each object. Failure: Newton not converging, a periodic state that does
    not return to itself, the rate-component tail off O1 (2 cosh(kappa/2) = rate), the S-bend
    unstable or the bump stable."""
    out = {}
    for obj in ("sbend", "bump"):
        r = O.family("grid", "G1", obj, omega_start=2.3, omega_min=2.25, omega_stop=2.36, N=81)
        pts = r["points"]
        assert len(pts) >= 3
        worst_ret = max(p["period_return_error"] for p in pts)
        worst_tail = max(abs(p["tail_kappa_measured"] / p["tail_kappa_O1"] - 1) for p in pts)
        out[obj] = {"n": len(pts), "omega": [p["omega"] for p in pts],
                    "max_period_return_error": worst_ret, "max_tail_rel_dev_O1": worst_tail,
                    "floquet_max": [p["floquet_max_modulus"] for p in pts],
                    "parity": [p["floquet_unstable_mode_parity"] for p in pts]}
        print("\ntool a %s: %s" % (obj, out[obj]))
        assert worst_ret < 1e-9 and worst_tail < 1e-3
    assert max(out["sbend"]["floquet_max"]) < 1 + 1e-6
    assert min(out["bump"]["floquet_max"]) > 1 + 1e-3
    assert set(out["bump"]["parity"]) == {"antisymmetric"}
    record("tool_a_family_smoke", out)


def test_smooth_orbit_attempt_smoke(record):
    """(b) Newton on C-smooth at h = 1/4: returns what happens (converged or not, residuals).
    Failure: an exception instead of a report."""
    r = O.smooth_attempt(0.25, "G1", "sbend", 0.5, 0.8, floquet=True)
    keep = {k: v for k, v in r.items() if not k.startswith("profile")}
    print("\ntool b: %s" % keep)
    record("tool_b_smoke", keep)
    assert "newton" in r


@pytest.mark.parametrize("omega", [1.2, 1.5])
@pytest.mark.parametrize("obj", ["sbend", "bump"])
def test_jump_profile(omega, obj, record):
    """Block 4 item 2. The constructed single-harmonic profile satisfies its first integral
    (y'^2/2 + Phi(a) = Phi(a_top) inside the outer jump, 0 outside), y = g(a) is continuous across
    the jumps, the net stretch is zero (exact for the profile equation), and the tail falls as
    exp(-kappa x), kappa = omega / sqrt(omega^2 - 1). No continuous profile exists: A* < A_z.
    Failure: any of these off by more than discretisation error."""
    P = J.Profile(omega)
    a_top = 1.02 * P.A_z
    x = np.linspace(-12, 12, 480001)
    a, info = J.stretch_profile(obj, omega, a_top, x)
    assert a is not None
    y = P.g(a)
    dx = x[1] - x[0]
    yp = np.gradient(y, dx)
    inner = (np.abs(x) < info["x_s"] - 5 * dx) & (np.abs(x) > 5 * dx)
    outer = (np.abs(x) > info["x_s"] + 5 * dx) & (np.abs(x) < 6)
    e_in = np.max(np.abs(0.5 * yp[inner] ** 2 + P.Phi(a[inner]) - P.Phi(a_top))) if inner.any() else 0.0
    e_out = np.max(np.abs(0.5 * yp[outer] ** 2 + P.Phi(a[outer])))
    ycont = abs(float(P.g(info["a_s"]) - P.g(info["a_out"])))
    area = float(np.trapezoid(a, x))
    far = (x > 3) & (x < 8)
    kap = -np.polyfit(x[far], np.log(np.abs(a[far])), 1)[0]
    out = dict(info, first_integral_err_inner=float(e_in), first_integral_err_outer=float(e_out),
               y_jump=ycont, net_stretch=area, tail_kappa=float(kap),
               scale=float(abs(P.Phi(a_top))))
    print("\njump profile %s omega=%g: %s" % (obj, omega, out))
    record("jump_profile_%s_om%g" % (obj, omega), out)
    assert P.A_star < P.A_z
    assert e_in < 1e-4 * out["scale"] + 1e-8 and e_out < 1e-4 * out["scale"] + 1e-8
    jumps = info["jump_outer"] * 2 + info["jump_centre"]
    assert ycont < 1e-12 and abs(area) < jumps * dx      # trapezoid error across the jumps
    assert abs(kap / P.kappa - 1) < 1e-3


def test_jump_start_runs_with_jump_diagnostics(record):
    """(c) and Block 4 items 2-4 through the driver: a jump-profile start at h = 1/8, G1, rate 1.2,
    100 time units, with its twin; records carry the jump measure. Failure: missing fields."""
    name = "_test_jump_start"
    cfg = {"kind": "evolve", "name": name, "medium": "fd", "law": "G1", "h": 0.125,
           "N": int(round(720 / 0.125)), "profile_range": 50.0, "dt": 0.01, "T": 100.0,
           "start": {"type": "jump", "obj": "sbend", "omega": 1.2}, "twin": True}
    r = DR.evolve(cfg, log=lambda *a: None)
    s, rec = r["summary"], r["records"][-1]
    out = {k: s[k] for k in ("frac_end", "rate_early", "rate_late", "core_coordinate",
                             "tail_kappa_at_late_rate", "tail_kappa_O1_at_late_rate", "guard")}
    out.update({k: rec[k] for k in ("jump_max", "jump_x", "stretch_total_variation", "jump_cells")})
    out["twin_frac_end"] = r["twin"]["frac_end"]
    out["start_info"] = {k: v for k, v in r["start_info"].items()}
    print("\njump start h=1/8 omega=1.2 sbend: %s" % out)
    record("jump_start_smoke", out)
    _cleanup(name)
    assert out["core_coordinate"] == "stretch across centre"
    assert np.isfinite(out["frac_end"]) and out["start_info"]["x_s"] <= 0.25 + 1e-6
    assert out["start_info"]["plateau_cells"] >= 1


@pytest.mark.parametrize("case", ["grid_sbend", "fd_orbit", "fd_jump"])
def test_tail_fit_at_late_rate(case, record):
    """Item 2 of the review: the tail is fitted on the component of u at the rate measured over the
    fit interval (last 20% of the run, Hann-tapered), compared with O1 at that rate (5%).
    grid_sbend (asserted): trapped exact S-bend, G1, rate 2.5 (O1: 2 arccosh(rate/2)).
    fd_orbit (asserted): held exact C-smooth S-bend of tool b, G1, h = 1/8, from sigma 0.25 and
    peak stretch 0.8, rate 1.423 (Floquet stable; O1: rate / sqrt(rate^2 - 1)).
    fd_jump (reported): the held jump-profile S-bend from rate 1.2; its rate drifts (1.16 -> 1.07)
    and its component at the late rate is not a clean bound tail. Failure: > 5% off O1."""
    name = "_test_tail_%s" % case
    if case == "grid_sbend":
        cfg = {"kind": "evolve", "name": name, "medium": "grid", "law": "G1", "N": 800,
               "dt": 0.01, "T": 600.0, "start": {"type": "orbit", "obj": "sbend", "omega": 2.5},
               "twin": False}
    elif case == "fd_orbit":
        cfg = {"kind": "evolve", "name": name, "medium": "fd", "law": "G1", "h": 0.125,
               "N": int(round(720 / 0.125)), "profile_range": 50.0, "dt": 0.01, "T": 600.0,
               "start": {"type": "smooth_orbit", "obj": "sbend", "sigma": 0.25,
                         "peak_stretch": 0.8}, "twin": False}
    else:
        cfg = {"kind": "evolve", "name": name, "medium": "fd", "law": "G1", "h": 0.125,
               "N": int(round(720 / 0.125)), "profile_range": 50.0, "dt": 0.01, "T": 600.0,
               "start": {"type": "jump", "obj": "sbend", "omega": 1.2}, "twin": False}
    s = DR.evolve(cfg, log=lambda *a: None)["summary"]
    out = {k: s[k] for k in ("tail_kappa_at_late_rate", "tail_kappa_O1_at_late_rate",
                             "tail_fit_rate", "tail_fit_rate_source", "rate_early", "rate_late",
                             "rate_late_over_fit_interval", "frac_end", "nominal_rate")}
    out["rel_dev"] = out["tail_kappa_at_late_rate"] / out["tail_kappa_O1_at_late_rate"] - 1
    print("\ntail fit %s: %s" % (case, out))
    record("tail_fit_%s" % case, out)
    _cleanup(name)
    if case != "fd_jump":
        assert abs(out["rel_dev"]) < 0.05


def test_driver_cli_checkpoint_resume(tmp_path):
    """(d) Driver entry point, atomic JSON with full parameters, and checkpoint/resume: a run cut
    after its first checkpoint and resumed gives the same records as an uncut run.
    Failure: a different result after resume, or a missing parameter in the JSON."""
    name = "_test_resume"
    cfg = {"kind": "evolve", "name": name, "medium": "grid", "law": "G1", "N": 800, "dt": 0.01,
           "T": 60.0, "start": {"type": "orbit", "obj": "sbend", "omega": 2.4}, "twin": False,
           "ckpt_every_s": 0.0}
    p = tmp_path / "jobs.json"
    p.write_text(json.dumps([cfg]))
    DR.main(["driver", str(p)])
    full = json.load(open(os.path.join(DR.OUT, name + ".json")))
    assert full["params"]["dt"] == 0.01 and full["params"]["start"]["omega"] == 2.4

    class Cut(Exception):
        pass

    def log(msg):
        if "checkpoint" in msg and "t=3" in msg:
            raise Cut()
    with pytest.raises(Cut):
        DR.evolve(cfg, log=log)
    msgs = []
    res = DR.evolve(cfg, log=msgs.append)
    assert any("resumed" in m for m in msgs)
    a = np.array([r["E_fixed"] for r in full["records"]])
    b = np.array([r["E_fixed"] for r in res["records"]])
    print("\ndriver resume: %d records, max |diff| %.1e" % (len(a), np.max(np.abs(a - b))))
    assert a.shape == b.shape and np.max(np.abs(a - b)) < 1e-12
    _cleanup(name)


def test_push_smoke(record):
    """(e) Push a C-grid G1 S-bend (rate 2.1, near the band top, weakly pinned) with
    v = -lambda du/dx. Failure: no motion at lambda = 1, or summary fields missing."""
    job = {"kind": "push", "name": "_test_push", "medium": "grid", "law": "G1", "N": 800,
           "dt": 0.01, "T": 300.0, "record_periods": 4,
           "start": {"type": "orbit", "obj": "sbend", "omega": 2.1},
           "push": {"lambdas": [0.2, 1.0]}, "dist_target": 10.0}
    path = DR.run_job(job, log=lambda *a: None)
    rows = json.load(open(path))["rows"]
    print("\npush smoke: %s" % rows)
    record("tool_e_push_smoke", {"rows": rows})
    for r in rows:
        _cleanup(r["file"])
    os.remove(path)
    assert rows[-1]["speed"] > 0.01 and rows[-1]["crossed_target"]


def test_cost_smoke():
    """(f) The cost table runs. Failure: an exception."""
    rows = DR.cost_table([{"medium": "fd", "law": "G1", "h": 0.25, "N": 2880, "dt": 0.02,
                           "profile_range": 50.0,
                           "start": {"type": "gauss", "obj": "bump", "sigma": 0.5,
                                     "peak_stretch": 0.8}}], periods=2, log=lambda *a: None)
    assert rows[0]["wall_s_per_1000_cycles"] > 0
