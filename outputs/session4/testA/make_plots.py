"""Report plots and tables for Test A from analysis.json and static/ (no new runs).

Writes PNGs and TABLES.md under outputs/session4/testA/plots/.
Conventions: F > 0 attracts.  U_law = -2 pi A^2 cos(dtheta) e^(-kappa r)/r,
F_law = 2 pi A^2 cos(dtheta) e^(-kappa r) (kappa/r + 1/r^2); dtheta = 0 (p000) or pi (p180).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)
AN = json.load(open(os.path.join(HERE, "analysis.json")))
SP = json.load(open(os.path.join(HERE, "static", "static_primary.json")))
CH = ["p000", "p180"]
COS = {"p000": 1.0, "p180": -1.0}
COL = {"p000": "tab:blue", "p180": "tab:red"}


def U_law(r, A, k, c):
    r = np.asarray(r, float)
    return -2 * np.pi * A**2 * c * np.exp(-k * r) / r


def F_law(r, A, k, c):
    r = np.asarray(r, float)
    return 2 * np.pi * A**2 * c * np.exp(-k * r) * (k / r + 1 / r**2)


def static_dense(ch):
    r = np.array([row["r"] for row in SP["rows"]])
    U = np.array([row[ch]["U_static"] for row in SP["rows"]])
    return r, U, -np.gradient(U, r)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=140, bbox_inches="tight")
    plt.close(fig)


def force_plot(data, numbers, fname, title, chs, static=True):
    A, k = numbers["A"], numbers["kappa"]
    fig, ax = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    for ch in chs:
        F = data["channels"][ch]["F"]
        re, f, s = np.array(F["r_eff"]), np.array(F["F"]), np.array(F["sigma"])
        mv = np.array(F["moving"], bool)
        c = COS[ch]
        for m, fill in ((~mv, True), (mv, False)):
            if m.any():
                ax[0].errorbar(re[m], np.abs(f[m]), yerr=s[m], fmt="o", color=COL[ch],
                               mfc=COL[ch] if fill else "none", capsize=2, ms=5,
                               label=f"{ch} F {'(static window)' if fill else '(moving, hollow)'}")
        rr = np.linspace(re.min(), re.max(), 300)
        ax[0].plot(rr, np.abs(F_law(rr, A, k, c)), "-", color=COL[ch], lw=1, label=f"{ch} |F_law|")
        if static:
            r, U, Fs = static_dense(ch)
            msk = (r >= re.min()) & (r <= re.max())
            ax[0].plot(r[msk], np.abs(Fs[msk]), "--", color="k" if ch == "p000" else "gray", lw=1,
                       label=f"{ch} |F_static|")
        ax[1].errorbar(re[~mv], f[~mv] / F_law(re[~mv], A, k, c), yerr=np.abs(s[~mv] / F_law(re[~mv], A, k, c)),
                       fmt="o", color=COL[ch], capsize=2, ms=5, label=f"{ch}")
        if mv.any():
            ax[1].errorbar(re[mv], f[mv] / F_law(re[mv], A, k, c), yerr=np.abs(s[mv] / F_law(re[mv], A, k, c)),
                           fmt="o", color=COL[ch], mfc="none", capsize=2, ms=5)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("|F|  (F > 0 attracts)")
    ax[0].set_title(title)
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].axhline(1, color="k", lw=0.6)
    ax[1].set_ylabel("F / F_law (signed)")
    ax[1].set_xlabel("r_eff")
    ax[1].legend(fontsize=8)
    ax[1].set_ylim(-0.2, 2.2)
    save(fig, fname)


# ---------------- Plot 1 -------------------------------------------------
P = AN["primary"]
force_plot(P, P["numbers"], "1_primary_force.png",
           f"Primary |F| vs r_eff  (A={P['numbers']['A']:.3f}, kappa={P['numbers']['kappa']:.4f})", CH)

# ---------------- Plot 2 -------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(13, 9))
A, k = P["numbers"]["A"], P["numbers"]["kappa"]
for j, ch in enumerate(CH):
    c = COS[ch]
    cd = P["channels"][ch]
    r, U, Fs = static_dense(ch)
    rv = np.linspace(10, 45, 400)
    for i in range(2):
        a = axs[i, j]
        VF = cd["V_F"]
        VN = cd["V_N"]
        vf, sf, rf = np.array(VF["V"]), np.array(VF["sigma"]), np.array(VF["r"])
        vn, sn, rn = np.array(VN["V"]), np.array(VN["sigma"]), np.array(VN["r"])
        sgn = 1 if i == 0 else None
        for (rr_, v_, s_, lab, col, mk) in ((rf, vf, sf, "V_F", "tab:green", "o"),
                                            (rn, vn, sn, "V_N", "tab:orange", "s")):
            if i == 0:
                a.errorbar(rr_, v_, yerr=s_, fmt=mk, color=col, capsize=2, ms=4, label=lab)
            else:
                hollow = np.abs(v_) < 3 * s_
                a.errorbar(rr_[~hollow], np.abs(v_[~hollow]), yerr=s_[~hollow], fmt=mk, color=col,
                           capsize=2, ms=4, label=lab + " (>3 sigma)")
                a.errorbar(rr_[hollow], np.abs(v_[hollow]), yerr=s_[hollow], fmt=mk, color=col,
                           mfc="none", capsize=2, ms=4, label=lab + " (<3 sigma, hollow)")
        f = (lambda x: x) if i == 0 else np.abs
        a.plot(r, f(U), "k--", lw=1, label="U_static")
        a.plot(rv, f(U_law(rv, A, k, c)), "-", color="gray", lw=1.2, label="U_law")
        a.set_xlabel("r")
        if i == 0:
            a.set_xlim(12, 42)
            m = (r >= 14) & (r <= 42)
            lim = max(0.5, np.abs(U[m]).max() * 0.2)
            lim = 0.6
            a.set_ylim(-lim, lim)
            a.axhline(0, color="k", lw=0.5)
            a.set_ylabel("V (clipped to +-0.6)")
            a.set_title(f"{ch}: V(r)")
        else:
            a.set_yscale("log")
            a.set_xlim(12, 42)
            a.set_ylim(1e-5, 20)
            a.set_ylabel("|V|")
            a.set_title(f"{ch}: |V(r)|")
        a.legend(fontsize=6)
save(fig, "2_primary_potential.png")

# ---------------- Plot 3 : unequal ---------------------------------------
U = AN["unequal"]
fig, ax = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
for ch in CH:
    cd = U["channels"][ch]
    Fa = cd["F_amp"]
    re, f, s = np.array(Fa["r_eff"]), np.array(Fa["F"]), np.array(Fa["sigma"])
    used = np.array(Fa["used"], bool)
    st_re = np.array(cd["F_amp_static_at_r_eff"] and Fa["r_eff"][-len(cd["F_amp_static_at_r_eff"]):])
    st = np.array(cd["F_amp_static_at_r_eff"])
    for m, fill, lab in ((used, True, "used in fits"), (~used, False, "not used")):
        if m.any():
            ax[0].errorbar(re[m], np.abs(f[m]), yerr=s[m], fmt="o", color=COL[ch], capsize=2, ms=5,
                           mfc=COL[ch] if fill else "none", label=f"{ch} F_amp ({lab})")
    ax[0].plot(st_re, np.abs(st), "--", color=COL[ch], lw=1, marker="x", ms=4, label=f"{ch} static amplitude")
    # ratio at matching r_eff (static list covers the last len(st) points)
    n = len(st)
    st = np.abs(st)
    rr, ff, ss, uu = re[-n:], f[-n:], s[-n:], used[-n:]
    ax[1].errorbar(rr[uu], ff[uu] / st[uu], yerr=np.abs(ss[uu] / st[uu]), fmt="o", color=COL[ch], capsize=2, ms=5,
                   label=ch)
    ax[1].errorbar(rr[~uu], ff[~uu] / st[~uu], yerr=np.abs(ss[~uu] / st[~uu]), fmt="o", color=COL[ch], mfc="none",
                   capsize=2, ms=5)
ax[0].set_yscale("log")
ax[0].set_ylabel("F_amp")
ax[0].set_title("Unequal lumps: force amplitude vs static amplitude")
ax[0].legend(fontsize=7)
ax[1].axhline(1, color="k", lw=0.6)
ax[1].set_ylabel("F_amp / static")
ax[1].set_xlabel("r_eff")
ax[1].legend(fontsize=8)
save(fig, "3_unequal_force.png")

# ---------------- Plot 4 : b5 --------------------------------------------
B = AN["b5"]
force_plot(B, B["numbers"], "4_b5_force.png",
           f"b5 |F| vs r_eff  (A={B['numbers']['A']:.3f}, kappa={B['numbers']['kappa']:.5f})", ["p000"],
           static=False)

# ---------------- Plot 5 : sign strip ------------------------------------
fig, ax = plt.subplots(figsize=(11, 3.6))
cmap = {"attractive": "tab:blue", "repulsive": "tab:red", "unresolved": "lightgray"}
rows = []
for ch in CH:
    cd = P["channels"][ch]
    rows.append((f"{ch} static", cd["force_signs_static"]))
    rows.append((f"{ch} dynamic", cd["force_signs_dynamic"]))
for y, (lab, d) in enumerate(rows):
    for r, l in zip(d["r"], d["label"]):
        ax.scatter(r, y, c=cmap.get(l, "k"), s=28 if "static" in lab else 60, marker="s")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel("r (static) / r_eff (dynamic)")
ax.set_title("Force sign: blue attractive, red repulsive, gray unresolved")
save(fig, "5_force_signs.png")

# ---------------- Tables -------------------------------------------------
L = []


def fmt(x, p=4):
    return "n/a" if x is None else f"{x:.{p}g}"


def force_table(data, name, chs, nums):
    A, k = nums["A"], nums["kappa"]
    for ch in chs:
        cd = data["channels"][ch]
        F = cd["F"]
        sg = cd["force_signs_dynamic"]
        L.append(f"\n### {name} {ch}: force vs tail law\n")
        L.append("| r_eff | F +- sigma | F_law | F/F_law | sign | moving |")
        L.append("|---|---|---|---|---|---|")
        for re, f, s, mv, lab in zip(F["r_eff"], F["F"], F["sigma"], F["moving"], sg["label"]):
            fl = F_law(re, A, k, COS[ch])
            L.append(f"| {re:.3f} | {f:.5g} +- {s:.2g} | {fl:.5g} | {f/fl:.4f} | {lab} | {'moving' if mv else 'window'} |")
        ex = cd.get("force_excluded", [])
        if ex:
            L.append(f"\nExcluded from force (too short a window): " + ", ".join(e["name"] for e in ex))


force_table(P, "Primary", CH, P["numbers"])

# unequal force tables
for ch in CH:
    cd = U["channels"][ch]
    Fa = cd["F_amp"]
    n = len(cd["F_amp_static_at_r_eff"])
    L.append(f"\n### Unequal {ch}: F_amp vs static amplitude\n")
    L.append("| r_eff | F_amp +- sigma | static amp (signed) | ratio to abs | used in fits |")
    L.append("|---|---|---|---|---|")
    off = len(Fa["F"]) - n
    for i, (re, f, s, u) in enumerate(zip(Fa["r_eff"], Fa["F"], Fa["sigma"], Fa["used"])):
        st = cd["F_amp_static_at_r_eff"][i - off] if i >= off else None
        L.append(f"| {re:.3f} | {f:.5g} +- {s:.2g} | {fmt(st,5)} | {fmt(f/abs(st) if st else None)} | {'used' if u else 'not used'} |")
    L.append(f"\nForce-excluded: " + ", ".join(e["name"] for e in cd["force_excluded"]))

force_table(B, "b5", ["p000"], B["numbers"])

# V tables
for ch in CH:
    cd = P["channels"][ch]
    L.append(f"\n### Primary {ch}: V_F and V_N vs U_law (also U_static)\n")
    r_s, U_s, _ = static_dense(ch)
    for key in ("V_F", "V_N"):
        V = cd[key]
        L.append(f"\n**{key}**\n")
        L.append("| r | V +- sigma | U_law | V/U_law | U_static | resolved (>3 sigma) |")
        L.append("|---|---|---|---|---|---|")
        for r, v, s in zip(V["r"], V["V"], V["sigma"]):
            if key == "V_N" and r < 12:
                continue
            ul = float(U_law(r, A, k, COS[ch]))
            us = float(np.interp(r, r_s, U_s))
            L.append(f"| {r:.3f} | {v:.5g} +- {s:.2g} | {ul:.5g} | {fmt(v/ul)} | {us:.5g} | {'yes' if abs(v) > 3*s else 'no'} |")
    L.append("\nNote: V_F r values are r_eff; V_N r values are nominal r; U_static interpolated at r.")

# verdicts
ROUTES = ["V_F", "V_N", "V_N_box", "U_static", "V_raw"]
FORMS = ["E1", "E2", "G", "P"]


def verdict_rows(dsname, data, chs):
    for ch in chs:
        vd = data["channels"][ch]["verdicts"]
        L.append(f"\n### Verdicts {dsname} {ch}\n")
        L.append("| route | rule | verdict | n_fit | chi2red E1 | E2 | G | P | L | k E1 | k E2 | k L | preferred |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for rt in ROUTES:
            if rt not in vd:
                continue
            blocks = [("A3", vd[rt]["A3"])]
            a3p = vd[rt]["A3prime"]
            for key in ("r_lo_12", "r_lo_16"):
                if key in a3p:
                    blocks.append((f"A3' {key}", a3p[key]))
            for lab, b in blocks:
                cr = b.get("chi2_red") or {}
                kf = b.get("k_fit") or {}
                lchi = (b.get("clauses", {}).get("grows", {}) or {}).get("L_chi2_red")
                L.append(f"| {rt} | {lab} | {b['verdict']} | {b.get('n_fit')} | "
                         + " | ".join(fmt(cr.get(f), 3) for f in FORMS) + f" | {fmt(lchi,3)} | "
                         + " | ".join(fmt(kf.get(f), 4) for f in ("E1", "E2", "L"))
                         + f" | {b.get('preferred_form')} |")
            L.append(f"| {rt} | A3' deciding | {a3p['verdict']} (r_lo {a3p.get('deciding_r_lo')}) | | | | | | | | | | |")
        L.append(f"\nHeadline: {json.dumps(data['channels'][ch]['headline'])}")


verdict_rows("primary", P, CH)
verdict_rows("unequal", U, CH)
verdict_rows("b5", B, ["p000"])

# minimum in V
L.append("\n### Any minimum in V(r)\n")


def interior_min(r, v, s=None):
    r, v = np.asarray(r), np.asarray(v)
    out = []
    for i in range(1, len(v) - 1):
        if v[i] < v[i - 1] and v[i] < v[i + 1]:
            out.append((float(r[i]), float(v[i])))
    return out


for ch in CH:
    cd = P["channels"][ch]
    r_s, U_s, Fs = static_dense(ch)
    m = (r_s >= 8)
    mins_s = interior_min(r_s[m], U_s[m])
    sc = cd["force_signs_static"]["sign_changes"]
    dc = cd["force_signs_dynamic"]["sign_changes"]
    VF = cd["V_F"]
    # V_F points resolved at >3 sigma; local minima among them
    res = [(r, v) for r, v, s in zip(VF["r"], VF["V"], VF["sigma"]) if abs(v) > 3 * s]
    mins_f = interior_min([x[0] for x in res], [x[1] for x in res])
    L.append(f"- {ch}: U_static interior minima for r>=8: {mins_s if mins_s else 'none'}; static force sign changes: "
             f"{sc if sc else 'none'}; dynamic force sign changes: {dc if dc else 'none'}; V_F interior minima among "
             f"resolved points: {mins_f if mins_f else 'none'}.")
L.append("\nV_N is noisy beyond r ~ 26 (sigma comparable to V), so no minimum is read from it.")

open(os.path.join(OUT, "TABLES.md"), "w").write("# Test A report tables (generated by make_plots.py)\n" + "\n".join(L) + "\n")
print("done")
