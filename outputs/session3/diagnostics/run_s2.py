"""Run one wave of the stage-2 jobs: python run_s2.py STAGE   (stage A: gates, B: collision + stretch z, C: stretch x + control).
Two workers, two threads each; results to outputs/session3/diagnostics/l3/results (not committed: 12 to 18 MB each)."""
import os
import sys
from dcommon import HERE, ROOT
from swirl.engine import l3

l3.OUT = "outputs/session3/diagnostics/l3"
l3.CKPT = "checkpoints/session3/diagnostics"
if __name__ == "__main__":
    os.environ["NUMBA_NUM_THREADS"] = "2"
    fails = l3.run_jobs(os.path.join(HERE, "jobs_s2.json"), workers=2, threads=2, stages=[sys.argv[1]])
    print("failed:", fails)
