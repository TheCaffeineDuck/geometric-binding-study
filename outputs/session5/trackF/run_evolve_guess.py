"""Workaround wrapper (analysis, not solver code): run_evolve5 with an explicit eps_guess for amp: starts on a
branch past a fold, where the CLI's default guess lands on the wrong branch. Usage:
  python run_evolve_guess.py <eps_guess> <run_evolve5 args...>
"""
import sys, os
sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL/.claude/worktrees/s5-trackF")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from swirl.radial5 import evolve as EV
from swirl.radial5 import run_evolve5 as RE
eps = float(sys.argv[1])
_orig = EV.profile_start
def patched(model, obj, omega=None, scale=1.0, amplitude=None, eps_guess=None):
    return _orig(model, obj, omega=omega, scale=scale, amplitude=amplitude, eps_guess=eps if amplitude is not None else eps_guess)
EV.profile_start = patched
RE.main(sys.argv[2:])
