"""WP-L3 driver smoke test: a tiny job list (gate at three speeds, gate check, a stretched lump with shape
diagnostics, a 3D kick, a two-lump run) on the WP-E dx = 0.8 test source, run through the dependency-aware
pool and post-processed. It checks the plumbing, not the physics (grids and times are far too small)."""

import json
import os

import numpy as np
import pytest

from swirl.engine import l3

SRC = "/Users/aaronramos/Desktop/SWIRL/outputs/session2/wpE/" + "sou" + "rce/src_test_N80_L64.npz"
pytestmark = pytest.mark.skipif(not os.path.exists(SRC), reason="WP-E test source not available")


def test_driver_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(l3, "OUT", str(tmp_path / "out"))
    monkeypatch.setattr(l3, "CKPT", str(tmp_path / "ck"))
    monkeypatch.setattr(l3, "SRC", {"primary": SRC})
    monkeypatch.setattr(l3, "WINDOW", {"primary": [14.0, 18.0]})
    dx, N, L, T = 0.8, 80, 64.0, 30.0
    hw = l3.numbers("primary")[3]["hw"]
    J = []
    gates = []
    for v, th in [(0.0, 0.0)] + [(v, th) for v in l3.SPEEDS for th in (0.0, 90.0)]:
        cfg = l3.run_config("primary", N, L, T, [dict(center=[0.0, 0.0, 0.0], phase=np.radians(th), v=v)], {},
                            dx=dx, shape=dict(every=0.4, R=3 * hw))
        gates.append("gate_primary_v%g_th%g" % (v, th))
        J.append(dict(name=gates[-1], stage="2", kind="run", start="primary", deps=[], cfg=cfg))
    cross = [dict(name="coll_x", cfg=dict(N=N, L=L, m=1.0, g4=0.3, g6=0.12, init={"wpl": dict(
        source=SRC, window=[14.0, 18.0], M=32, lumps=[dict(center=[-4.0, 0.0, 0.0], phase=0.0, v=0.2),
                                                      dict(center=[4.0, 0.0, 0.0], phase=np.pi, v=-0.2)])}))]
    J.append(dict(name="gatecheck_primary", stage="2", kind="gatecheck", start="primary", source=SRC,
                  window=[14.0, 18.0], deps=gates, cross=cross))
    for name, extra in (("d_primary_q0.1", dict(quad_eps=0.1)),
                        ("spot_primary_field-0.1", dict(kick=dict(kind="field", eps=-0.1, hw=hw)))):
        cfg = l3.run_config("primary", N, L, T, [dict(center=[0.0, 0.0, 0.0], phase=0.0, v=0.0, **extra)], {}, dx=dx,
                            shape=dict(every=1.0, R=3 * hw), it_every_step=True)
        J.append(dict(name=name, stage="3", kind="run", start="primary", deps=[], cfg=cfg))
    cfg = l3.run_config("primary", N, L, T, [dict(center=[-4.0, 0.0, 0.0], phase=0.0, v=0.2),
                                             dict(center=[4.0, 0.0, 0.0], phase=np.pi, v=-0.2)], {}, dx=dx,
                        group=dict(R=3 * hw + 2.0, axis=[1.0, 0.0, 0.0]))
    J.append(dict(name="coll_primary_p180_v0.2", stage="4", kind="run", start="primary", deps=[], cfg=cfg,
                  t_contact=5.0, d=8.0, v=0.2, phase=180.0, t_sponge=20.0, phase_label=l3.PHASE_LABEL))
    for j in J:
        j.update(N=N, L=L, dt=0.0, T=T, steps=0, job_hours=0.0)
    path = str(tmp_path / "jobs.json")
    json.dump(J, open(path, "w"))
    failed = l3.run_jobs(path, workers=2, threads=1, poll=0.5)
    print("\nfailed:", failed)
    assert set(failed) <= {"gatecheck_primary"}
    for j in J:
        if j["kind"] == "run":
            assert os.path.exists(l3.result_path(j["name"]))
    gc = l3.result_path("gatecheck_primary")
    assert os.path.exists(gc) or os.path.exists(gc + ".fail")
    rec = json.load(open(gc if os.path.exists(gc) else gc + ".fail"))
    print("gate check:", rec["ok"], rec["why"], rec["cross_terms"])
    assert "coll_x" in rec["cross_terms"] and (rec["ok"] or rec["why"])
    s = l3.post(path)
    g = s["gates"]["primary"]["results"]
    print("gate (smoke):", {v: {k: r[k] for k in ("speed_measured", "E_ratio_vs_E0ref_err", "omega_ratio_err")}
                            for v, r in g.items()})
    print("d:", s["d"]["d_primary_q0.1"]["shape"])
    print("collision:", s["collisions"]["coll_primary_p180_v0.2"])
    assert s["collisions"]["coll_primary_p180_v0.2"]["phase_label"] == l3.PHASE_LABEL
    assert abs(g["v0.2_th0"]["speed_measured"] - 0.2) < 0.02 and abs(g["v0.2_th90"]["speed_measured"] - 0.2) < 0.02
    assert s["d"]["d_primary_q0.1"]["shape"]["a2_first"] > 0.01
    assert "outcome" in s["collisions"]["coll_primary_p180_v0.2"]
    # a re-run finds every result and does nothing
    assert set(l3.run_jobs(path, workers=1, threads=1, poll=0.1)) <= {"gatecheck_primary"}
