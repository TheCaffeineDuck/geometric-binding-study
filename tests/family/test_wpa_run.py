"""WP-A checks g and h: the long-run driver (checkpoint resume, Session 1 reproduction,
end-to-end smoke of maps -> rank -> long), the saved family checks (a), and the
recorded validation (c, d) recomputed from the saved files."""

import glob
import json
import os

import numpy as np
import pytest

from swirl.family import wpa_rate as rate
from swirl.family.wpa_family import OUT
from swirl.family.wpa_run import run


def test_checkpoint_resume_identical(tmp_path):
    """Run to T = 60 straight, and to 30 with a checkpoint then resumed to 60: identical."""
    beta, w = 4.0 / 3.0, 0.91
    a = run(beta, w, 60.0)
    ck = str(tmp_path / "c.npz")
    part = run(beta, w, 60.0, ckpt=ck, ckpt_every=10.0, stop_after=30.0)
    assert part["partial"] and abs(part["t"] - 30.0) < 1e-9
    b = run(beta, w, 60.0, ckpt=ck, ckpt_every=10.0)
    same = all(np.array_equal(a["series"][k], b["series"][k]) for k in a["series"])
    same_w = np.array_equal(a["crossings"], b["crossings"])
    print("\nresume: series identical %s, crossings identical %s, frac_end %.15f %.15f"
          % (same, same_w, a["frac_end"], b["frac_end"]))
    assert same and same_w and a["T_90"] == b["T_90"] and a["T_death"] == b["T_death"]
    with pytest.raises(RuntimeError):
        run(beta, 0.92, 60.0, ckpt=ck)          # a checkpoint of another run is refused


def test_driver_reproduces_session1_death():
    """beta = 0.85, omega = 0.7793: Session 1 radial death time 1145.5 (radial_T1000.json)."""
    r = run(0.85, 0.7793064609301195, 1300.0)
    t = r["series"]["t"]
    f = r["series"]["E_in"][:, 0] / r["E0"]
    f1000 = f[np.argmin(np.abs(t - 1000))]
    print("\nbeta=0.85 w=0.7793: T_death %s (Session 1 1145.5), frac(1000) %.6f (0.795936), "
          "omega settled %.4f, nonlinear share t0 signed %.3f abs %.3f"
          % (r["T_death"], f1000, r["regime"]["omega_over_m_settled"],
             r["regime"]["nonlinear_share_t0_signed"], r["regime"]["nonlinear_share_t0_absolute"]))
    assert r["T_death"] == 1145.5
    assert abs(f1000 - 0.7959363646) < 1e-6


def test_end_to_end_smoke(tmp_path, monkeypatch):
    """maps -> rank -> long at beta = 4/3 with tiny settings, written to a temporary folder."""
    from swirl.family import wpa
    monkeypatch.setattr(wpa, "OUT", str(tmp_path))
    monkeypatch.setattr(rate, "OUT", str(tmp_path))
    monkeypatch.setattr(rate, "CACHE", str(tmp_path / "checkpoints" / "maps"))
    monkeypatch.setattr(rate, "T_MAP", 400.0)
    monkeypatch.setattr(rate, "W", 100.0)
    monkeypatch.setattr(rate, "D_A2", 0.04)
    beta = 4.0 / 3.0
    wpa.stage_maps([beta], workers=1)
    with open(rate.map_path(beta)) as f:
        m = json.load(f)
    x = m["runs"][0]
    print("\nsmoke map: %d runs, %d nodes, R_lump %.2f, t_s %.1f; run 0: w %.4f E_profile %.2f "
          "E0(grid) %.2f E_s(measured) %.2f, windows %d, twin frac T300 %.2e"
          % (m["n_runs"], len(m["nodes"]), m["settings"]["R_lump"], m["settings"]["t_s"],
             x["omega_profile"], x["E_profile_setB"], x["E0"], x["E_s"], len(x["windows"]),
             x["twin"]["frac_T300"]))
    assert m["n_runs"] >= 3 and len(m["nodes"]) > 0
    assert all(w_["E"] != x["E_profile_setB"] for w_ in x["windows"])      # measured, not label
    wpa.stage_rank([beta])
    with open(os.path.join(str(tmp_path), "ranking.json")) as f:
        rk = json.load(f)["per_beta"]["%.4f" % beta]
    print("smoke rank: best w %.5f T_death_pred %s" % (rk["best"]["omega_profile"], rk["best"]["T_death"]))
    wpa.stage_long([beta], workers=1, T=300.0)
    out = sorted(glob.glob(os.path.join(str(tmp_path), "long_*.json")))
    assert len(out) in (1, 2)          # best and best_by_measured_rate (may coincide)
    with open(out[0]) as f:
        d = json.load(f)
    print("smoke long: T_end %s T_90 %s T_death %s omega %.4f share %.3f twin T100 %.2e"
          % (d["T_end"], d["T_90"], d["T_death"], d["regime"]["omega_over_m_settled"],
             d["regime"]["nonlinear_share_t0_absolute"], d["linear_twin"]["frac_T100"]))
    for k in ("params", "regime", "T_90", "T_death", "osc_T_end", "linear_twin", "frac_at"):
        assert k in d
    assert d["T_end"] == 300.0 and d["params"]["g4_run"] == 0.30
    # resume from the finished checkpoint returns the same numbers
    wpa.stage_long([beta], workers=1, T=300.0)
    with open(out[0]) as f:
        d2 = json.load(f)
    assert d2["series"] == d["series"] and d2["T_end"] == d["T_end"]


def _saved(pattern):
    return sorted(glob.glob(os.path.join(OUT, pattern)))


def test_saved_new_families_checks():
    """Check a, from outputs/session2/wpA/family_beta*.json."""
    files = _saved("family_beta*.json")
    if not files:
        pytest.skip("run the families stage first")
    for fn in files:
        with open(fn) as f:
            c = json.load(f)["checks_wpA"]
        print("\n%s: rows %d, Derrick worst %.2e (1e-6), universal worst %.2e (1e-5), "
              "dE=w dN central worst %.2e, rows median %.2e"
              % (os.path.basename(fn), c["n_rows"], c["derrick_worst"], c["universal_worst"],
                 c["dEdw_central_worst"], c["dEdw_rows_median_rel"]))
        assert c["derrick_worst"] < 1e-6
        assert c["universal_worst"] < 1e-5
        assert c["dEdw_central_worst"] < 1e-5


def test_saved_validation_recomputes():
    """Checks c and d: the recorded predictions recompute from the saved map files. The
    30% target is a result, printed, not asserted."""
    p = os.path.join(OUT, "validation.json")
    if not os.path.exists(p):
        pytest.skip("run the validate stage first")
    with open(p) as f:
        vals = json.load(f)["validations"]
    for v in vals:
        with open(rate.map_path(v["beta"])) as f:
            m = json.load(f)
        kept = [x for x in m["runs"] if abs(x["omega_profile"] - v["omega"]) >= 0.002]
        nodes = rate.assemble(rate.points_from_runs(kept))
        pr = rate.predict(nodes, m["settings"]["t_s"], v["s_frac_interpolated"] * v["E0"], v["E0"],
                          v["rho0"], m["settings"]["E_min_setB"])
        print("\nvalidation %s beta=%.4f %s: predicted %.1f (band %s..%s), measured %.1f, rel %+.3f -> %s"
              % (v["name"], v["beta"], v["quantity"], v["predicted"], v["predicted_lo"],
                 v["predicted_hi"], v["measured"], v["rel_error"], v["verdict"]))
        assert abs(pr[v["quantity"]] - v["predicted"]) < 1e-6 * v["predicted"]


def test_saved_box_and_resolution_checks():
    """Check f, from outputs/session2/wpA/checks.json (target 1e-4; 3x is a pass with note)."""
    p = os.path.join(OUT, "checks.json")
    if not os.path.exists(p):
        pytest.skip("run the checks stage first")
    with open(p) as f:
        c = json.load(f)
    for k in ("box_doubling_beta0.7500", "box_doubling_beta5.0000", "dr_halving_beta5.0000"):
        print("\n%s: max |d frac| = %.2e over T = %g; while alive %s"
              % (k, c[k]["max_abs_diff_frac"], c[k]["T"], c[k].get("max_abs_diff_frac_while_alive")))
    assert c["box_doubling_beta0.7500"]["max_abs_diff_frac_while_alive"] < 1e-4
    assert c["box_doubling_beta5.0000"]["max_abs_diff_frac"] < 1e-4
    assert c["dr_halving_beta5.0000"]["max_abs_diff_frac"] < 1e-4


@pytest.mark.xfail(strict=True, reason="recorded miss: beta = 0.75 box doubling over the whole "
                   "T = 3000 span is 3.3e-4 (target 1e-4), all of it after death (t > 1500), "
                   "from energy returning to the emptied centre")
def test_box_doubling_beta075_full_span_miss():
    with open(os.path.join(OUT, "checks.json")) as f:
        c = json.load(f)
    assert c["box_doubling_beta0.7500"]["max_abs_diff_frac"] < 3e-4
