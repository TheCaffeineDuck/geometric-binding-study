"""Build lifetime_table_v2.json and lifetime_vs_beta_v2.png (v1 files are kept). Sources named per block in the JSON."""
import os, sys, json, glob
os.environ["NUMBA_NUM_THREADS"] = "1"
import numpy as np
sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL")
from swirl.family.wpa_run import profile, write_json
import swirl.family.wpa_family as fam
OUT = "/Users/aaronramos/Desktop/SWIRL/outputs/session2/wpA"
BETAS = [0.75, 0.85, 1.0, 1.15, 4/3, 1.6, 2.0, 3.0, 5.0]
rk = json.load(open(f"{OUT}/ranking.json"))["per_beta"]
v1rows = json.load(open(f"{OUT}/lifetime_table.json"))["rows"]
v1set = {(round(r["beta"], 4), round(r["omega_start"], 6)): r["pick"] for r in v1rows}
v1 = json.load(open(f"{OUT}/lifetime_table.json"))
def bkey(b): return "%.4f" % b
Nmin = {b: json.load(open(fam.family_path(b)))["features_rescaled"]["N_min"] for b in BETAS}
def find_pi(b, w):
    for r in rk[bkey(b)]["ranked"]:
        if abs(r["omega_profile"] - w) < 1e-9: return r
    return None
def row_from(f, L, group):
    b = L["params"]["beta"]; w = L["params"]["omega_profile"]
    bb = min(BETAS, key=lambda x: abs(x - b))
    p = profile(bb, w); reg = L["regime"]; W = L["windows_1000"]
    fa = L["frac_at"]; last = str(int(max(float(k) for k in fa))) if fa else None
    pk = find_pi(bb, w)
    return dict(
        group=group, beta=bb, omega_start=w, alpha2=0.3 * bb * (1 - w * w), pick_v1=v1set.get((round(bb, 4), round(w, 6))),
        file=os.path.basename(f), E0=L["E0"], N_over_Nmin=p["N"] / Nmin[bb], E_over_mN=p["E_over_mN"], half_width=L["params"]["half_width"],
        omega_over_m_settled=reg["omega_over_m_settled"], omega_over_m_end=reg["omega_over_m_end"],
        nl_share_t0_abs=reg["nonlinear_share_t0_absolute"], nl_share_t0_signed=reg["nonlinear_share_t0_signed"],
        nl_share_mean_abs=reg["nonlinear_share_mean_absolute"],
        T_end=L["T_end"], T_90=L["T_90"], T_death=L["T_death"], osc_T_90=L["osc_T_90"], osc_T_death=L["osc_T_death"], osc_T_end=L["osc_T_end"],
        alive_at_end=L["T_death"] is None, frac_at=fa, frac_end=L["frac_end"], frac_t0=L["frac_t0"],
        t_s=L["t_s"], n_windows_1000=len(W), omega_first_window=(W[0]["omega"] if W else None), omega_last_window=(W[-1]["omega"] if W else None),
        Gamma_first_window=(W[0]["Gamma"] if W else None), sigma_first_window=(W[0]["sigma"] if W else None),
        Gamma_last_window=(W[-1]["Gamma"] if W else None), sigma_last_window=(W[-1]["sigma"] if W else None),
        pathint=None if pk is None else dict(T_90=pk["T_90"], T_death=pk["T_death"], T_death_lo=pk["T_death_lo"], T_death_hi=pk["T_death_hi"],
                                             flags_death=pk["flags_death"], flags_90=pk["flags_90"], source="ranking.json per_beta[%s].ranked" % bkey(bb)),
        twin=dict(frac_T100=L["linear_twin"]["frac_T100"], frac_T1000_end=L["linear_twin"]["frac_T1000"]),
        E_h_rise_max_rel=L["energy_check"]["E_h_rise_max_rel"])
rows, s6 = [], []
for f in sorted(glob.glob(f"{OUT}/long_beta*_w*.json")):
    L = json.load(open(f))
    if f.endswith("_T300k.json"):
        r = row_from(f, L, "S6"); s6.append((r, L)); continue
    g = L.get("groups")
    rows.append(row_from(f, L, ",".join(g) if g else "V1_pick"))
# starts extended to 300k: use the 300k result as the lifetime of that start
ext = {(round(r["beta"], 4), round(r["omega_start"], 6)): r for r, _ in s6}
def life(r, key):
    e = ext.get((round(r["beta"], 4), round(r["omega_start"], 6)))
    rr = e if e is not None else r
    return rr[key], rr
perbeta = []
for b in BETAS:
    rs = [r for r in rows if abs(r["beta"] - b) < 1e-3]
    out = dict(beta=b, n_long_runs=len(rs))
    for key in ("T_90", "T_death"):
        sc = []
        for r in rs:
            T, rr = life(r, key)
            sc.append((np.inf if T is None else T, rr["frac_end"], r, rr))
        sc.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top = sc[0]
        ties = [x for x in sc if x[0] == top[0]]
        out["longest_by_" + key] = dict(
            omega_start=top[2]["omega_start"], alpha2=top[2]["alpha2"], group=top[2]["group"], value=None if np.isinf(top[0]) else top[0],
            reached_horizon_without_event=bool(np.isinf(top[0])), horizon=top[3]["T_end"], frac_end=top[1],
            n_tied_at_horizon=len(ties) if np.isinf(top[0]) else 1,
            tied_starts=[(x[2]["omega_start"], x[2]["group"], x[1]) for x in ties] if np.isinf(top[0]) else None,
            found_by_v1_two_pick=any(x[2]["pick_v1"] is not None for x in ties) if np.isinf(top[0]) else (top[2]["pick_v1"] is not None),
            v1_best_value_same_key=max([(np.inf if life(r, key)[0] is None else life(r, key)[0]) for r in rs if r["pick_v1"]]),
            note="horizon = 30,000 unless the start was extended to 300,000; ties among starts alive at the horizon are ordered by frac there")
    perbeta.append(out)
# matched size
matched = {}
for r in rows:
    for g in r["group"].split(","):
        if g.startswith("S5_N"):
            matched.setdefault(g, []).append(dict(beta=r["beta"], omega=r["omega_start"], N_over_Nmin=r["N_over_Nmin"], T_90=r["T_90"], T_death=r["T_death"],
                                                  T_end=r["T_end"], frac_end=r["frac_end"], E_over_mN=r["E_over_mN"], hw=r["half_width"], file=r["file"]))
for g in matched: matched[g].sort(key=lambda x: x["beta"])
# S6
s6out = []
for r, L in s6:
    W = L["windows_1000"]; t0s = np.array([w["t0"] for w in W]); G = np.array([w["Gamma"] for w in W]); sg = np.array([w["sigma"] for w in W])
    blocks = {}
    for lo, hi in ((0, 30000), (30000, 100000), (100000, 200000), (200000, 300001)):
        m = (t0s >= lo) & (t0s < hi)
        if m.any(): blocks["%d-%d" % (lo, hi)] = dict(mean_Gamma=float(G[m].mean()), mean_sigma=float(sg[m].mean()), n=int(m.sum()), max_Gamma=float(G[m].max()), min_Gamma=float(G[m].min()))
    fa = L["frac_at"]; oa = L["omega_at"]
    s6out.append(dict(beta=r["beta"], omega_start=r["omega_start"], pick_v1=r["pick_v1"], T_end=L["T_end"], T_90=L["T_90"], T_death=L["T_death"],
                      frac_at={k: fa.get(k) for k in ("30000", "100000", "200000", "300000")}, frac_at_all=fa, omega_at={k: oa.get(k) for k in ("30000", "100000", "200000", "300000")},
                      omega_at_all=oa, frac_min_over_series=float(np.min(L["series"]["frac"])), frac_max_after_30000=float(np.max(np.array(L["series"]["frac"])[np.array(L["series"]["t"]) >= 30000])),
                      Gamma_blocks=blocks, file=r["file"], source="long_*_T300k.json frac_at, omega_at, windows_1000"))
box2 = None
bf = glob.glob(f"{OUT}/box2_beta1.3333_w0.906505_T300k.json")
if bf:
    B = json.load(open(bf[0])); Ls = [L for r, L in s6 if abs(r["beta"] - 4/3) < 1e-3][0]
    tb = np.array(B["t"]); fb = np.array(B["frac"]); ts = np.array(Ls["series"]["t"]); fs = np.array(Ls["series"]["frac"])
    idx = {round(float(t), 3): i for i, t in enumerate(tb)}
    pairs = [(i, idx[round(float(t), 3)]) for i, t in enumerate(ts) if round(float(t), 3) in idx]
    d = np.array([abs(fs[i] - fb[j]) for i, j in pairs]); tt = np.array([ts[i] for i, j in pairs])
    box2 = dict(omega=B["omega"], grid=B["grid"], T_end_box2=B["T_end"], n_compared=len(pairs), max_abs_diff_frac=float(d.max()), t_at_max=float(tt[d.argmax()]), target=1e-4,
                max_abs_diff_by_50k={"%d-%d" % (lo, lo + 50000): float(d[(tt >= lo) & (tt < lo + 50000)].max()) for lo in range(0, 300000, 50000)},
                frac_end=[float(fs[-1]), float(fb[-1])], T_90=[Ls["T_90"], B["T_90"]], T_death=[Ls["T_death"], B["T_death"]], wall_s=B["wall_s"],
                note="standard (R_box 200) vs doubled (R_box 400), compared on a 50-unit grid", source="box2_*_T300k.json and long_*_T300k.json series")
# beta = 0.75 death shape
burst = []
for r in rows:
    if abs(r["beta"] - 0.75) > 1e-3 or r["T_death"] is None: continue
    L = json.load(open(f"{OUT}/{r['file']}")); t = np.array(L["series"]["t"]); fr = np.array(L["series"]["frac"]); Td = L["T_death"]
    cross = {}
    for lv in (0.9, 0.7, 0.5, 0.3, 0.1):
        i = np.argmax(fr < lv) if (fr < lv).any() else None
        cross[str(lv)] = None if i is None else float(t[i])
    at = lambda T_: float(np.interp(T_, t, fr))
    df = np.abs(np.diff(fr)) / np.diff(t)
    burst.append(dict(omega_start=r["omega_start"], group=r["group"], T_death=Td, T_90=r["T_90"], frac_cross_times=cross,
                      frac_at_offsets={str(o): at(Td + o) for o in (-300, -200, -100, -50, -25, 0, 25, 50) if 0 <= Td + o <= t[-1]},
                      max_abs_dfrac_dt_per_25=float(np.max(np.convolve(df, np.ones(5) / 5, mode="valid"))), t_at_max_rate=float(t[np.argmax(np.convolve(df, np.ones(5) / 5, mode="valid"))]),
                      time_from_frac_0p9_to_0p3=(None if cross["0.9"] is None or cross["0.3"] is None else cross["0.3"] - cross["0.9"]), file=r["file"],
                      source="series.frac (5-unit samples)"))
write_json(f"{OUT}/lifetime_table_v2.json", dict(description="WP-A lifetime table v2: 18 sweep runs + supplement. rows = one per long run (30,000 horizon); S6 = the same starts continued to 300,000.",
                                                 rows=rows, per_beta_longest=perbeta, matched_size=matched, S6=s6out, S6_doubled_box_beta4_3=box2, beta0p75_death_shape=burst, per_beta_v1=v1["per_beta"]))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
cols = {"V1_pick": "k", "S5": "C0", "S3": "C3", "S4": "C4", "S1": "C2", "S2": "C1"}
def col(g):
    for k, c in cols.items():
        if g.startswith(k): return c
    return "gray"
for r in rows:
    for key, a in (("T_90", ax[0, 0]), ("T_death", ax[0, 1])):
        T = r[key]; c = col(r["group"])
        if T is None: a.plot(r["beta"], 30000, "^", color=c, mfc="none", ms=8)
        else: a.plot(r["beta"], T, "o", color=c, ms=5)
for a, t in zip(ax[0], ("T_90 (open triangle: not reached at 30,000)", "T_death (open triangle: alive at 30,000)")):
    a.set_xscale("log"); a.set_yscale("log"); a.set_title(t, fontsize=9); a.set_xlabel("beta"); a.grid(alpha=.3)
for k, c in (("v1 picks", "k"), ("S5 matched", "C0"), ("S3", "C3"), ("S4", "C4"), ("S1", "C2"), ("S2", "C1")): ax[0, 0].plot([], [], "o", color=c, label=k)
ax[0, 0].legend(fontsize=7)
for (g, c) in (("S5_N1.3", "C0"), ("S5_N1.8", "C2"), ("S5_N2.5", "C3")):
    for key, ls, a in (("T_90", "-", ax[1, 0]), ("T_death", "--", ax[1, 0])):
        pts = matched.get(g, [])
        x = [p["beta"] for p in pts]; y = [p[key] if p[key] is not None else 30000 for p in pts]
        a.plot(x, y, ls + "o", color=c, label="%s %s" % (g[3:], key))
        for p_ in pts:
            if p_[key] is None: a.plot(p_["beta"], 30000, "^", color=c, mfc="none", ms=10)
ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log"); ax[1, 0].set_title("matched size N/Nmin (open triangle: not reached by 30,000)", fontsize=9); ax[1, 0].legend(fontsize=7); ax[1, 0].grid(alpha=.3); ax[1, 0].set_xlabel("beta")
for r, L in s6:
    ax[1, 1].plot(L["series"]["t"], L["series"]["frac"], lw=.7, label="beta %.2f w0 %.4f" % (r["beta"], r["omega_start"]))
ax[1, 1].set_ylim(0.6, 1.01); ax[1, 1].set_xlabel("T"); ax[1, 1].set_ylabel("frac"); ax[1, 1].set_title("S6: to T = 300,000", fontsize=9); ax[1, 1].legend(fontsize=6); ax[1, 1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/lifetime_vs_beta_v2.png", dpi=120)
print("rows", len(rows), "S6", len(s6))
