"""D1: S-bend rate 2.3 push: reproduce the build's speed (0.039 to 0.053 at lambda >= 0.5) and find the setting that differs."""
import os, sys, json
os.environ["NUMBA_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from swirl.string import driver as D
D.OUT = os.path.join(HERE, "runs")
D.CKPT = os.path.join(HERE, "ck")

# (label, overrides): base = the test_push_smoke job of the build (N 800, dt 0.01, T 300, record_periods 4, dist_target 10)
base = {"kind": "evolve", "medium": "grid", "law": "G1", "N": 800, "dt": 0.01, "sample_dt": 0.1, "T": 300.0, "record_periods": 4,
        "start": {"type": "orbit", "obj": "sbend", "omega": 2.3}, "twin": False, "probes": [0, 20, 50]}
variants = [
    ("build_like: old sponge 100/0.3, no noise, T 300", {"sponge_len": 100.0, "sigma_max": 0.3, "start": dict(base["start"], noise=0)}),
    ("old sponge, noise 1e-8, T 300", {"sponge_len": 100.0, "sigma_max": 0.3}),
    ("production sponge 300/0.05, no noise, T 300", {"start": dict(base["start"], noise=0)}),
    ("production sponge, noise 1e-8, T 300", {}),
    ("production sponge, noise, T 2000, record 20", {"T": 2000.0, "record_periods": 20}),
    ("production sponge, noise, T 2000, dt 0.0051 (production dt)", {"T": 2000.0, "record_periods": 20, "dt": 0.0051}),
    ("old sponge, no noise, T 2000", {"sponge_len": 100.0, "sigma_max": 0.3, "start": dict(base["start"], noise=0), "T": 2000.0, "record_periods": 20}),
]
rows = []
for lam in (0.2, 0.5, 1.0):
    for lab, ov in variants:
        cfg = dict(base, **ov)
        cfg["push"] = {"lambda": lam}
        cfg["name"] = "D1_tmp"
        r = D.evolve(cfg, name="D1_tmp", log=lambda *a: None)
        ps = D.push_summary(r, 10.0)
        s = r["summary"]
        row = {"lambda": lam, "variant": lab, "speed": ps["speed"], "distance": ps["distance"], "fco_end": ps["frac_comoving_end"],
               "centroid_end": s["centroid_end"], "centroid_start": s["centroid_start"], "T": cfg["T"], "dt": cfg["dt"], "guard": s["guard"] is not None}
        rows.append(row)
        print(row, flush=True)
json.dump(rows, open(os.path.join(HERE, "D1.json"), "w"), indent=1)
for f in ("D1_tmp.json", "D1_tmp_series.npz"):
    p = os.path.join(D.OUT, f)
    if os.path.exists(p):
        os.remove(p)
