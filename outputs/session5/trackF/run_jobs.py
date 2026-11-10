"""Run each line of a jobs file through swirl.radial5.run_evolve5 (or boundary5 if line starts with 'B5 '), 2 at a time."""
import sys, subprocess, os, time, shlex
from concurrent.futures import ThreadPoolExecutor
ROOT = "/Users/aaronramos/Desktop/SWIRL/.claude/worktrees/s5-trackF"
PY = "/Users/aaronramos/Desktop/SWIRL/.venv/bin/python"
OUT = ROOT + "/outputs/session5/trackF"
env = dict(os.environ, NUMBA_NUM_THREADS="1", PYTHONPATH=ROOT)
def run(line):
    line = line.strip()
    if not line: return
    if line.startswith("B5 "):
        args = shlex.split(line[3:]); mod = "swirl.radial5.boundary5"
        tag = args[args.index("--tag") + 1] if "--tag" in args else "b5_" + str(abs(hash(line)))[:6]
        cmd = [PY, "-m", mod] + [a for a in args]
    elif line.startswith("G "):
        args = shlex.split(line[2:]); eps = args[0]; args = args[1:]; tag = args[args.index("--tag") + 1]
        cmd = [PY, OUT + "/run_evolve_guess.py", eps] + args + ["--out-dir", os.environ.get("RUNDIR", OUT + "/runs")]
    else:
        args = shlex.split(line); tag = args[args.index("--tag") + 1]
        cmd = [PY, "-m", "swirl.radial5.run_evolve5"] + args + ["--out-dir", os.environ.get("RUNDIR", OUT + "/runs")]
    with open(OUT + "/logs/progress.log", "a") as p: p.write(time.strftime("%T") + " start " + tag + "\n")
    with open(OUT + f"/logs/{tag}.log", "w") as f:
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    with open(OUT + "/logs/progress.log", "a") as p: p.write(time.strftime("%T") + f" done {tag} rc={rc}\n")
lines = open(sys.argv[1]).read().splitlines()
with ThreadPoolExecutor(int(os.environ.get('NW','2'))) as ex: list(ex.map(run, lines))
