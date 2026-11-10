"""D2 diagnostics: jump S-bend 2.0 at other h and with the spectral scheme (nonlinear runs only, wall-capped by T)."""
import os, sys, json
os.environ["NUMBA_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from multiprocessing import Pool
from swirl.string import driver as D

OUT = os.path.join(HERE, "runs")
D.OUT = OUT
base = json.load(open(os.path.join(ROOT, "outputs/session5/trackC/runs/B9_B3_jump_sbend_w2_h8.json")))["params"]
keep = {k: base[k] for k in ("kind", "medium", "law", "dt", "sample_dt", "record_periods", "probes", "start")}
keep["probe_max"] = 200000
per = 2 * 3.141592653589793 / 1.846
jobs = [
    dict(keep, name="D2_fd_h4", h=0.25, N=3360, T=5000 * per, twin=False),
    dict(keep, name="D2_fd_h16", h=0.0625, N=13440, T=2000 * per, twin=False),
    dict(keep, name="D2_spectral_h8", medium="spectral", h=0.125, N=6720, dt=0.008, sample_dt=0.1, T=1700.0, twin=False),
]


def work(j):
    D.OUT = OUT
    try:
        D.run_job(j, log=lambda *a: print(j["name"], *a, flush=True))
    except Exception as e:
        print("FAILED", j["name"], repr(e), flush=True)
    return j["name"]


if __name__ == "__main__":
    with Pool(2) as p:
        # first process: h4 then h16 sequentially via ordering; second: spectral
        for n in p.imap_unordered(work, [jobs[1], jobs[0]]):
            print("DONE", n, flush=True)
