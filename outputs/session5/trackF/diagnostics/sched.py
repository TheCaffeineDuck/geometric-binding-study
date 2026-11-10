import subprocess, time, shlex, os, sys
ROOT = "/Users/aaronramos/Desktop/SWIRL/.claude/worktrees/s5-trackF"; OUT = ROOT + "/outputs/session5/trackF"
PY = "/Users/aaronramos/Desktop/SWIRL/.venv/bin/python"
env = dict(os.environ, NUMBA_NUM_THREADS="1", PYTHONPATH=ROOT)
jobs = [l for l in open(OUT + "/diagnostics/jobs_DF_lean.txt").read().splitlines() if l.strip()]
procs = []
def running(): return len(subprocess.run("pgrep -f [r]adial5.run_evolve5", shell=True, capture_output=True, text=True).stdout.split())
while jobs:
    if running() < 3:
        args = shlex.split(jobs.pop(0)); tag = args[args.index("--tag") + 1]
        open(OUT + "/logs/progress.log", "a").write(time.strftime("%T") + " start " + tag + "\n")
        procs.append(subprocess.Popen([PY, "-m", "swirl.radial5.run_evolve5"] + args + ["--out-dir", OUT + "/diagnostics/runs"], cwd=ROOT, env=env,
                     stdout=open(OUT + f"/logs/{tag}.log", "w"), stderr=subprocess.STDOUT))
    time.sleep(5)
for p in procs: p.wait()
