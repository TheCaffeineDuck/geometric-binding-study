"""Smoke tests of the production tools: the evolution driver (JSON with full parameters,
twin, checkpoint/resume bit-identical), the family driver, the cost table."""

import json
import os

import numpy as np

from swirl.radial5 import cost5
from swirl.radial5 import evolve as EV
from swirl.radial5 import run_evolve5, run_family5


def test_resume_is_bit_identical(tmp_path):
    st = EV.profile_start("R", "pinch", 0.99)
    ck = str(tmp_path / "c.npz")
    part = EV.run("R", "pinch", st, 40.0, ckpt=ck, stop_after=15.0)
    assert part.get("partial")
    b = EV.run("R", "pinch", st, 40.0, ckpt=ck)
    c = EV.run("R", "pinch", st, 40.0)
    same = (b["series"] == c["series"] and b["probe"]["f"] == c["probe"]["f"]
            and b["per_period"]["rows"] == c["per_period"]["rows"]
            and b["guard_extremes"]["rows"] == c["guard_extremes"]["rows"])
    print("\ntools resume at t = 15 of 40: identical %s" % same)
    assert same


def test_driver_writes_json(tmp_path):
    tag = "pytest_smoke_O_pinch"
    out = run_evolve5.main(["--model", "O", "--obj", "pinch", "--start", "profile:0.95", "--T", "20",
                            "--tag", tag, "--out-dir", str(tmp_path)])
    with open(tmp_path / (tag + ".json")) as fh:
        d = json.load(fh)
    for suffix in ("", "_twin"):
        p = os.path.join(run_evolve5.ROOT, "checkpoints", "radial5", tag + suffix + ".npz")
        if os.path.exists(p):
            os.remove(p)
    p = d["run"]["params"]
    print("\ntools driver JSON: keys %s; params %s" % (sorted(d), sorted(p)))
    assert p["grid"]["h"] == 0.05 and p["coupling"]["g4"] > 0 and d["twin"]["params"]["coupling"]["g4"] == 0
    assert len(d["run"]["probe"]["f"]) == 40 and d["run"]["T_end"] == 20.0


def test_family_driver_smoke():
    out = run_family5.family("R", "pinch", max_points=6)
    print("\ntools family R pinch, first 6 continuation points: rates %s, end %s"
          % (out["rate_range"], out["end"]["why"]))
    assert len(out["rows"]) == 6 and out["end"]["reason_raw"] == "max_points reached"


def test_cost_smoke(monkeypatch):
    monkeypatch.setattr(cost5, "STARTS", dict(O_bump=0.909149280617113))
    out = cost5.measure(T=5.0)
    print("\ntools cost smoke: %s" % [(r["combo"], r["twin"], round(r["wall_s_per_10000"], 1)) for r in out["rows"]])
    assert all(np.isfinite(r["wall_s_per_10000"]) for r in out["rows"])


def test_new_starts_are_regular_and_shaped():
    """scale, Gaussian with initial velocity, shell (moving inward): parity at r = 0 holds
    (the kernel would see a kink otherwise), and the velocity is the stated shape."""
    from swirl.radial5 import profiles as P
    r = np.linspace(1e-3, 30, 3001)
    p1 = EV.profile_start("R", "pinch", 0.99)
    p2 = EV.profile_start("R", "pinch", 0.99, scale=1.03)
    assert np.allclose(p2["f"](r), 1.03 * p1["f"](r))
    g = EV.gauss_start("bump", 0.5, 2.0, v0=0.3)
    assert np.allclose(g["ft"](r) / g["f"](r), 0.6)
    for obj, par in (("bump", 1.0), ("pinch", -1.0)):
        sh = EV.shell_start(obj, 0.4, 8.0, 1.5, v_in=0.2)
        assert np.allclose(sh["f"](-r), par * sh["f"](r)) and np.allclose(sh["ft"](-r), par * sh["ft"](r))
        # moving inward: f(r, dt) ~ f + dt f_t equals the shifted shell
        dt = 1e-6
        g = lambda x: np.exp(-x * x / (2 * 1.5 ** 2))
        moved = 0.4 * (g(r - 8.0 + 0.2 * dt) + par * g(r + 8.0 - 0.2 * dt))
        assert np.max(np.abs(sh["f"](r) + dt * sh["ft"](r) - moved)) < 1e-10
    a = EV.profile_start("R", "bump", amplitude=0.95)
    assert abs(a["info"]["F_peak"] - 0.95) < 1e-9 and a["info"]["omega"] < 0.9950
    print("\ntools starts: scale, gauss v0, shell parity and inward motion, amp start F0 0.95 -> omega %.6f "
          "(upper branch past the fold)" % a["info"]["omega"])


def test_classify_and_refine():
    """A dispersing start, a 3D-ill-posed pinch, and a centre collapse refined at h/2, h/4."""
    d = EV.run("R", "pinch", EV.gauss_start("pinch", 1.0, 2.0), 150.0, stop_at_death=True)
    c1 = EV.classify(d)
    ill = EV.run("R", "pinch", EV.gauss_start("pinch", 2.0, 4.0), 20.0)
    c2 = EV.classify(ill)
    st = EV.gauss_start("bump", 1.0, 3.0)
    col = EV.run("R", "bump", st, 20.0)
    c3 = EV.refine_trip("R", "bump", st, col)
    print("\ntools classify: %s (T_death %s); %s at t %.4f; collapse: %s, t* %s at threshold %s, times %s"
          % (c1["cls"], c1.get("T_death"), c2["cls"], c2["t"], c3["cls"], c3.get("t_star"),
             c3.get("threshold"), c3.get("t_at_threshold")))
    assert c1["cls"] == "disperses" and c2["cls"] == "3D ill-posed" and c3["cls"] == "singular"


def test_boundary_bisection_smoke(tmp_path):
    from swirl.radial5 import boundary5
    fn = boundary5.make_start_fn("R", "bump", "gauss", sigma=3.0)
    steps, br, status = boundary5.bisect("R", "bump", fn, 0.1, 1.0, rtol=0.5, T=60.0, refine=False,
                                         out=str(tmp_path / "b.json"))
    print("\ntools boundary bisection R bump gauss sigma 3 (T 60, rtol 0.5): %s, bracket %s, classes %s"
          % (status, br, [(round(s["A"], 3), s["cls"]) for s in steps]))
    assert status == "converged" and (br[1] - br[0]) / br[0] <= 0.5
    assert json.load(open(tmp_path / "b.json"))["done"]
