import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = "/Users/aaronramos/Desktop/SWIRL/.claude/worktrees/s5-trackF"; OUT = ROOT + "/outputs/session5/trackF"
names = ["O_bump", "O_pinch", "R_bump", "R_pinch"]
col = {"O_bump": "C0", "O_pinch": "C1", "R_bump": "C2", "R_pinch": "C3"}
fam = {n: json.load(open(f"{ROOT}/outputs/radial5/family_{n}.json"))["rows"] for n in names}
fig, ax = plt.subplots(2, 4, figsize=(18, 7.5))
for j, n in enumerate(names):
    R = fam[n]; w = np.array([r["omega"] for r in R]); E = np.array([r["E"] for r in R]); hw = np.array([r["hw"] for r in R])
    epc = np.array([r["E_per_cycle"] for r in R])
    a = ax[0, j]; a.semilogy(w, E, "-o", ms=3, color=col[n], label="E"); a.semilogy(w, epc, "--", color="gray", label="2piE/omega")
    i = int(np.argmin(E)); a.plot(w[i], E[i], "k*", ms=12, label="E min"); a.set_title(n); a.set_xlabel("rate omega"); a.set_ylabel("E"); a.legend(fontsize=7)
    a = ax[1, j]; a.semilogy(w, hw, "-o", ms=3, color=col[n]); a.set_xlabel("rate omega"); a.set_ylabel("half-width hw")
    for k in range(len(w)-1):
        pass
fig.suptitle("Track F families (cycle-averaged); star = energy minimum; R curves fold in omega"); fig.tight_layout()
fig.savefig(OUT + "/plot_families.png", dpi=110)
# overlay E vs omega for bump/pinch per model
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for j, m in enumerate(["O", "R"]):
    for obj, ls in [("bump", "-"), ("pinch", "--")]:
        R = fam[f"{m}_{obj}"]; ax[j].semilogy([r["omega"] for r in R], [r["E"] for r in R], ls, label=obj)
    ax[j].set_title("Model " + m); ax[j].set_xlabel("rate"); ax[j].set_ylabel("E"); ax[j].legend()
fig.tight_layout(); fig.savefig(OUT + "/plot_bump_vs_pinch_E.png", dpi=110)
