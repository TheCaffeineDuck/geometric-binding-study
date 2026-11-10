import os, sys, json
os.environ["NUMBA_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from swirl.string import driver as D
D.OUT = os.path.join(HERE, "runs")
D.CKPT = os.path.join(HERE, "ck")
base = json.load(open(os.path.join(ROOT, "outputs/session5/trackC/runs/B9_B3_jump_sbend_w1.5_h8.json")))["params"]
keep = {k: base[k] for k in ("kind", "medium", "law", "dt", "sample_dt", "record_periods", "probes", "start", "T", "probe_max")}
j = dict(keep, name="D6_jump1.5_fd_h4", h=0.25, N=3360, twin=False)
D.run_job(j, log=lambda *a: print(*a, flush=True))
