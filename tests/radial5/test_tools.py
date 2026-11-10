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


def test_family_driver_smoke(monkeypatch):
    monkeypatch.setattr(run_family5, "omega_grid", lambda model: [0.999, 0.995, 0.99, 0.9])
    out = run_family5.family("R", "bump")
    print("\ntools family R bump on a 4-point grid: rates %s, end %s" % (out["rate_range"], out["end"]))
    assert out["end"] is not None and out["rate_range"][1] == 0.999 and len(out["rows"]) >= 3


def test_cost_smoke(monkeypatch):
    monkeypatch.setattr(cost5, "STARTS", dict(O_bump=0.909149280617113))
    out = cost5.measure(T=5.0)
    print("\ntools cost smoke: %s" % [(r["combo"], r["twin"], round(r["wall_s_per_10000"], 1)) for r in out["rows"]])
    assert all(np.isfinite(r["wall_s_per_10000"]) for r in out["rows"])
