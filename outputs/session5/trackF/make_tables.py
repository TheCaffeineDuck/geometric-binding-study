"""Builds TABLES.md pieces, summary.json and plots from run JSON. Analysis only (no solver code)."""
import json, glob, os, shutil
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = "/Users/aaronramos/Desktop/SWIRL/.claude/worktrees/s5-trackF"; OUT = ROOT + "/outputs/session5/trackF"
RUNS = OUT + "/runs"
def L(path): return json.load(open(path))
fa = L(OUT + "/fa_fe_family.json")
fam = {n: L(f"{ROOT}/outputs/radial5/family_{n}.json") for n in ["O_bump", "O_pinch", "R_bump", "R_pinch"]}

def cyc_at(a, t):  # number of completed measured periods with t_mid <= t
    return int(np.sum(a[:, 0] <= t))
def series_at(s, t):
    ts = np.array(s["t"]); return float(np.array(s["frac"])[np.argmin(abs(ts - t))])

def life(tag):
    d = L(f"{RUNS}/{tag}.json"); r = d["run"]; tw = d.get("twin"); p = r["params"]; st = p["start"]
    a = np.array(r["per_period"]["rows"], float); E0 = r["E0"]
    w = r["regime"]["rate_end_last100"] or r["regime"]["rate_settled_50_150"]
    n = len(a)
    o = dict(tag=tag, model=p["model"], obj=p["obj"], start=d["command"]["start"], scale=d["command"]["scale"], T=r["T_end"],
             omega_start=st.get("omega"), hw=p["hw"], E=st.get("E"), frac_t0=r["frac_t0"], nl_profile=st.get("nonlinear_share_signed"),
             nl_run=r["regime"].get("nonlinear_share_cyc_50_150"), Xmax=r["regime"].get("X_max_all_stages"),
             late_rate=w, cls=d["end_class"]["cls"], guard=r.get("guard") is not None, n_periods=n)
    # cycle-averaged frac at 1000 cycles (rule) else at end
    if n >= 1000:
        o["frac_at"] = "1000 cyc"; o["t_ref"] = float(a[999, 0]); o["frac_cyc_ref"] = float(a[999, 2] / E0)
    else:
        o["frac_at"] = "end (%d cyc)" % n; o["t_ref"] = float(a[-1, 0]); o["frac_cyc_ref"] = float(a[-1, 2] / E0)
    o["frac_inst_end"] = r["frac_end"]
    o["T_90"] = r["T_90"]; o["T_death"] = r["T_death"]
    o["cyc_T90"] = cyc_at(a, r["T_90"]) if r["T_90"] is not None else None
    o["cyc_Tdeath"] = cyc_at(a, r["T_death"]) if r["T_death"] is not None else None
    o["cycles_end"] = r["cycles_T_end"]
    if tw:
        o["twin_T"] = tw["T_end"]; o["twin_T_death"] = tw["T_death"]
        o["twin_frac_ref"] = series_at(tw["series"], min(o["t_ref"], tw["T_end"]))
        o["twin_ref_time"] = min(o["t_ref"], tw["T_end"])
    # F4
    if n > 40:
        E = a[:, 3]; epc = a[:, 4]; W = a[:, 1]
        e0 = np.median(E[:20]); e1 = np.median(E[-20:]); p0 = np.median(epc[:20]); p1 = np.median(epc[-20:])
        o["dlnE"] = float(np.log(e1 / e0)); o["dlnEpc"] = float(np.log(p1 / p0)); o["dlnw"] = float(np.log(np.median(W[-20:]) / np.median(W[:20])))
        o["F4_ratio"] = o["dlnEpc"] / o["dlnE"] if abs(o["dlnE"]) >= 0.01 else None
    return o

# ---------------- lifetime rows
Bt = sorted(os.path.basename(f)[:-5] for f in glob.glob(RUNS + "/B_*.json"))
# drop the superseded T=2000 R bump a0.2 run (checkpoint conflict forced a new tag); keep both but flag
LT = {t: life(t) for t in Bt}
Ct = sorted(os.path.basename(f)[:-5] for f in glob.glob(RUNS + "/C_*.json"))
CT = {t: life(t) for t in Ct}
Et = sorted(os.path.basename(f)[:-5] for f in glob.glob(RUNS + "/E_*.json"))
ET = {t: life(t) for t in Et}

def fmt(x, f="%.4g"): return "-" if x is None else (f % x)
def lrow(o):
    td = "> %g" % o["T"] if o["T_death"] is None else "%.0f (%s cyc)" % (o["T_death"], fmt(o["cyc_Tdeath"], "%d"))
    t9 = "> %g" % o["T"] if o["T_90"] is None else ("%.4g (%s cyc)" % (o["T_90"], fmt(o["cyc_T90"], "%d")))
    tw = "-" if o.get("twin_frac_ref") is None else "%.2e @%g" % (o["twin_frac_ref"], o["twin_ref_time"])
    return "| %s | %s %s | %s | %.3g | %.4g | %.3f | %.4f (%s) | %s | %s | %s | %.4f | %+.3f | %s |" % (
        o["tag"].replace("B_", ""), o["model"], o["obj"], o["start"].replace("profile:", "w=").replace("amp:", "amp="), o["hw"], o["E"] or 0, o["frac_t0"],
        o["frac_cyc_ref"], o["frac_at"], t9, td, tw, o["late_rate"] or 0, o["nl_profile"] or 0, o["cls"])
hdr = ("| run | model obj | start | hw | E | frac(t=0) | frac, cycle-avg (at) | T_90 time (cycles) | T_death time (cycles) | twin frac (at t) | late rate | nonlinear share (profile) | end class |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
lines = [hdr] + [lrow(o) for o in LT.values()]
life_md = "\n".join(lines)

# ---------------- 1% table
def start_of(tag): return tag
c1 = []
pairs = [("O bump", "B_O_bump_w0.91", ["C_O_bump_w0.91_x0.99", "C_O_bump_w0.91_x1.01"]),
         ("O pinch", "B_O_pinch_w0.86", ["C_O_pinch_w0.86_x0.99", "C_O_pinch_w0.86_x1.01"])]
hdr1 = "| start | scale | late rate | T_90 | T_death | frac at 1000 cyc (cycle-avg) | twin frac | end class |\n|---|---|---|---|---|---|---|---|"
rows1 = [hdr1]
one_pct = {}
for name, base, nb in pairs:
    got = []
    for scl, tag in [(0.99, nb[0]), (1.0, base), (1.01, nb[1])]:
        o = (LT.get(tag) or CT.get(tag))
        if o is None: continue
        rows1.append("| %s | %.2f | %.4f | %s | %s | %.4f | %s | %s |" % (name, scl, o["late_rate"] or 0, fmt(o["T_90"], "%.0f") if o["T_90"] else "> %g" % o["T"],
                     fmt(o["T_death"], "%.0f") if o["T_death"] else "> %g" % o["T"], o["frac_cyc_ref"], "%.1e" % o["twin_frac_ref"], o["cls"]))
        got.append(o["cls"])
    one_pct[name] = got
extra = [("C_O_bump_w0.91_x0.98", 0.98), ("C_O_bump_w0.91_x1.02", 1.02), ("C_O_bump_w0.91_x1.05", 1.05), ("C_O_bump_w0.91_x1.02_T100k", 1.02)]
rows1.append("| (diagnostic: wider scan, O bump w=0.909, twin to T=2000 only) | | | | | | | |")
for tag, s in extra:
    o = CT.get(tag)
    if o: rows1.append("| O bump | %.2f | %.4f | %s | %s | %.4f (%s) | %.1e | %s |" % (s, o["late_rate"] or 0, fmt(o["T_90"], "%.0f") if o["T_90"] else "> %g" % o["T"], fmt(o["T_death"], "%.0f") if o["T_death"] else "> %g" % o["T"], o["frac_cyc_ref"], o["frac_at"], o.get("twin_frac_ref") or 0, o["cls"]))
onepct_md = "\n".join(rows1)

# ---------------- boundary table
bdir = OUT + "/boundary"; os.makedirs(bdir, exist_ok=True)
for f in glob.glob(ROOT + "/outputs/radial5/boundary/D_*.json"): shutil.copy(f, bdir)
brows = ["| shape | obj | A_lo (non-trip) | A_hi (trip) | rel width | low side: class, T_death (TU) | high side: class, trip time | collapse time, refined (h, h/2, h/4) at threshold | holds to 1000 cycles? |", "|---|---|---|---|---|---|---|---|---|"]
bsum = {}
for f in sorted(glob.glob(bdir + "/D_*.json")):
    d = L(f); tag = os.path.basename(f)[:-5]; S = d["steps"]; c = d["command"]
    if "bracket not valid" in "" : pass
    br = d["bracket"]
    valid = d["done"]
    lo = next((s for s in S if s["A"] == br[0]), None); hi = next((s for s in S if s["A"] == br[1]), None)
    if not valid:
        brows.append("| %s | %s | %s | %s | - | bracket not valid: %s | | | |" % (tag, c["obj"], br[0], br[1], "; ".join("A=%g %s" % (s["A"], s["cls"]) for s in S[:2]))); continue
    hd = hi["detail"] if hi else {}
    ref = ""
    if hi and hi["cls"] == "singular": ref = "t=%s (thr %s): %s" % ("%.3f" % hd["t_star"], hd["threshold"], ", ".join("%.3f" % x if x is not None else "-" for x in hd["t_at_threshold"]))
    elif hi: ref = "%s: trip t=%.3f (no h-refinement by the driver; see D_pinch_trip)" % (hi["cls"], (hi["guard"] or {}).get("t", float("nan")))
    shape = "%s%s" % (c["shape"], "" if c["shape"] == "shell" else " sigma=%g" % c["sigma"]) if c["shape"] != "shell" else "shell r0=%g sigma=%g v_in=%g" % (c["r0"], c["sigma"], c["v_in"])
    brows.append("| %s | %s | %.5f | %.5f | %.4f | %s, %.1f | %s, %.3f | %s | no: T_death %.1f TU = %.0f cycles at A_lo |" % (
        shape, c["obj"], br[0], br[1], d["rel_width"], lo["cls"], lo["T_death"] or -1, hi["cls"], (hi["guard"] or {}).get("t", float("nan")), ref, lo["T_death"] or -1, (lo["T_death"] or 0) / 6.2832))
    # trend of low-side T_death and high-side trip time along the bisection
    bsum[tag] = dict(bracket=br, rel_width=d["rel_width"], low=lo["cls"], T_death_low=lo["T_death"], high=hi["cls"], trip_t=(hi["guard"] or {}).get("t"),
                     steps=[dict(A=s["A"], cls=s["cls"], T_death=s["T_death"], trip=(s["guard"] or {}).get("t")) for s in S])
bound_md = "\n".join(brows)
json.dump(bsum, open(OUT + "/boundary_summary.json", "w"), indent=1, default=float)

# ---------------- boundary plot
fig, ax = plt.subplots(1, len(bsum), figsize=(4 * max(1, len(bsum)), 3.8), squeeze=False)
for k, (tag, b) in enumerate(bsum.items()):
    a = ax[0, k]
    for s in b["steps"]:
        trip = s["cls"] in ("singular", "unresolved", "3D ill-posed")
        y = s["trip"] if trip else s["T_death"]
        a.semilogy(s["A"], y, "rv" if trip else "bo")
    a.axvspan(*b["bracket"], color="gold", alpha=0.5); a.set_title(tag.replace("D_", ""), fontsize=8); a.set_xlabel("amplitude A"); a.set_ylabel("trip time (red) / T_death (blue), TU")
fig.tight_layout(); fig.savefig(OUT + "/plot_boundary_scan.png", dpi=110)

# ---------------- frac vs time plot
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for k, (m, o_) in enumerate([("O", "bump"), ("O", "pinch"), ("R", "bump"), ("R", "pinch")]):
    a = ax[k // 2, k % 2]
    for tag, o in LT.items():
        if o["model"] == m and o["obj"] == o_ and not tag.endswith("a0.2") or (tag.endswith("a0.2") is False and False): pass
    for tag, o in sorted(LT.items()):
        if o["model"] == m and o["obj"] == o_ and tag != "B_R_bump_a0.2":
            d = L(f"{RUNS}/{tag}.json"); s = d["run"]["series"]; lab = d["command"]["start"].replace("profile:", "w=").replace("amp:", "a=")
            ln, = a.semilogx(np.array(s["t"]) + 1, s["frac"], label=lab)
            if d.get("twin"): a.semilogx(np.array(d["twin"]["series"]["t"]) + 1, d["twin"]["series"]["frac"], ":", color=ln.get_color())
    a.set_title(f"{m} {o_} (dotted = linear twin)"); a.set_xlabel("t + 1"); a.set_ylabel("frac (3 hw sphere, / E at t=0)"); a.set_ylim(0, 1.05); a.legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(OUT + "/plot_frac_vs_time.png", dpi=110)

# ---------------- pinch vs bump (F-E)
def Erow(o): return o["E"]
fe_rows = ["| model | pairing | bump run: rate, E, hw | pinch run: rate, E, hw | E ratio pinch/bump | hw ratio | bump: T_90, T_death, frac_end | pinch: T_90, T_death, frac_end | bump end class | pinch end class |", "|---|---|---|---|---|---|---|---|---|---|"]
def lt(o):
    return "%s, %s, %.3f" % ("> %g" % o["T"] if o["T_90"] is None else "%.0f" % o["T_90"], "> %g" % o["T"] if o["T_death"] is None else "%.0f" % o["T_death"], o["frac_inst_end"])
ALL = {**LT, **CT, **ET}
def addpair(model, label, bt, pt):
    b = ALL.get(bt); p = ALL.get(pt)
    if not b or not p: return
    fe_rows.append("| %s | %s | w=%.4f, %.0f, %.2f | w=%.4f, %.0f, %.2f | %.2f | %.2f | %s | %s | %s | %s |" % (model, label, (b["omega_start"] or 0), b["E"], b["hw"], p["omega_start"] or 0, p["E"], p["hw"], p["E"] / b["E"], p["hw"] / b["hw"], lt(b), lt(p), b["cls"], p["cls"]))
for w, bt, pt in [("0.86", "B_O_bump_w0.86", "B_O_pinch_w0.86"), ("0.91", "B_O_bump_w0.91", "B_O_pinch_w0.91"), ("0.94", "B_O_bump_w0.94", "B_O_pinch_w0.94"),
                  ("0.965", "B_O_bump_w0.965", "B_O_pinch_w0.965"), ("0.98", "B_O_bump_w0.98", "B_O_pinch_w0.98"), ("0.995", "B_O_bump_w0.995", "B_O_pinch_w0.995"),
                  ("0.82", "B_O_bump_w0.82", "B_O_pinch_a1.828")]:
    addpair("O", "matched rate %s" % w, bt, pt)
addpair("O", "matched E ~ 2.0e4 (narrow)", "E_O_bump_E20k_w0.8312", "B_O_pinch_w0.86")
addpair("O", "matched E ~ 1.0e4 (narrow)", "E_O_bump_E10k_w0.8531", "E_O_pinch_E10k_w0.912")
addpair("O", "matched E ~ 1.0e4 (wide)", "E_O_bump_E10k_w0.99968", "E_O_pinch_E10k_w0.9854")
addpair("R", "matched E ~ 1.0e3", "E_R_bump_E1000_a0.17715_T6600", "E_R_pinch_E1000_a0.10892")
addpair("R", "matched E ~ 5.3e2", "B_R_bump_a0.33", "E_R_pinch_E533_a0.26158")
for w, bt, pt in [("~0.995 (bump 0.9950 a=0.79; pinch a=0.4384? rate .9932)", "B_R_bump_a0.79", "B_R_pinch_a0.4384")]:
    addpair("R", "near-matched rate " + w, bt, pt)
pinch_bump_md = "\n".join(fe_rows)

# family-level matched-rate and matched-energy
fam_md = ["| model | rate | bump E | bump hw | pinch E (branch) | pinch hw | E ratio pinch/bump |", "|---|---|---|---|---|---|---|"]
for m in ["O", "R"]:
    for r in fa["F_E_family"][m]["matched_rate"]:
        for pv in r["pinch"]:
            for bv in r["bump"]:
                fam_md.append("| %s | %.4f | %.1f | %.2f | %.1f (br %d) | %.2f | %.2f |" % (m, r["omega"], bv[1], bv[2], pv[1], pv[0], pv[2], pv[1] / bv[1]))
fam_md = "\n".join(fam_md)

# ---------------- family table
fhdr = "| family | points | rate range | amplitude param range | folds in rate (index, rate, E) | energy minima (rate, E) | hw range | E range | E/cycle range | nonlinear share range | X_max,cycle max | end of table |\n|---|---|---|---|---|---|---|---|---|---|---|---|"
frows = [fhdr]
for n, o in fa.items():
    if n == "F_E_family": continue
    frows.append("| %s%s | %d | %.5f to %.6f | %.4g to %.4g | %s | %s | %.2f to %.1f | %.1f to %.0f | %.0f to %.0f | %+.3f to %+.3f | %.3f | %s |" % (
        n, " (partial)" if o["partial"] else "", o["n_points"], o["rate_range"][0], o["rate_range"][1], o["amplitude_param_range"][0], o["amplitude_param_range"][1],
        "; ".join("(%d, %.5f, %.1f)" % (f["index"], f["omega"], f["E"]) for f in o["folds_in_omega"]) or "none in rate (fold in amplitude: %s)" % "; ".join("(%d, rate %.4f, amp %.3f)" % (f["index"], f["omega"], f["x"]) for f in o["folds_in_x"]),
        "; ".join("(%.5f, %.1f)" % (m_["omega"], m_["E"]) for m_ in o["energy_minima"]),
        o["hw_range"][0], o["hw_range"][1], o["E_range"][0], o["E_range"][1], o["Epc_range"][0], o["Epc_range"][1], o["nl_range"][0], o["nl_range"][1], o["Xmax_cycle_max"], o["end"][:70]))
fam_table_md = "\n".join(frows)
sp_rows = ["| family | branch (index range, rate range) | n | Spearman rho(rate, hw) | rule 2.2 |", "|---|---|---|---|---|"]
for n, o in fa.items():
    if n == "F_E_family": continue
    for k, v in o["spearman"].items():
        ver = o["spearman_verdict"].get(k, "n < 8: not scored" if v["n"] < 8 else "")
        sp_rows.append("| %s | %s | %d | %+.4f | %s |" % (n, k, v["n"], v["rho"], ver))
sp_md = "\n".join(sp_rows)

# O8-type: dlnw/dlnE along families (finite differences)
o8 = {}
for n, d in fam.items():
    R = d["rows"]; w = np.array([r["omega"] for r in R]); E = np.array([r["E"] for r in R])
    dlw = np.gradient(np.log(w)); dlE = np.gradient(np.log(E))
    with np.errstate(all="ignore"): rat = 1 - dlw / dlE  # d ln Epc / d ln E
    ok = np.abs(dlE) > 1e-3
    o8[n] = dict(ratio_min=float(np.nanmin(rat[ok])), ratio_max=float(np.nanmax(rat[ok])), ratio_median=float(np.nanmedian(rat[ok])), frac_points_ratio_le_0p2=float(np.mean(rat[ok] <= 0.2)))
# O12 small-amplitude slope
sa = {}
for n in ["R_bump", "O_bump"]:
    R = fam[n]["rows"][:6]; x = np.array([r["shoot_param"] for r in R]); w = np.array([r["omega"] for r in R])
    sa[n] = dict(c_1_minus_w_over_x2=[float((1 - ww) / xx ** 2) for ww, xx in zip(w, x)])
Rb = fam["R_bump"]["rows"]
sa["R_bump_dN_domega_signs"] = fam["R_bump"]["sign_runs"]
open(OUT + "/_tables_parts.json", "w").write(json.dumps(dict(o8=o8, small_amp=sa, one_pct=one_pct), indent=1, default=float))
open(OUT + "/_life.md", "w").write(life_md); open(OUT + "/_onepct.md", "w").write(onepct_md); open(OUT + "/_bound.md", "w").write(bound_md)
open(OUT + "/_pinchbump.md", "w").write(pinch_bump_md + "\n\nFamily-level (interpolated, from the family tables):\n\n" + fam_md)
open(OUT + "/_fam.md", "w").write(fam_table_md + "\n\n" + sp_md)
json.dump(dict(lifetime=list(LT.values()), diagnostics_C=list(CT.values()), matched_E=list(ET.values())), open(OUT + "/lifetime_all.json", "w"), indent=1, default=float)
print("ok", len(LT), len(CT), len(ET), len(bsum))
print(json.dumps(o8, indent=0)); print(json.dumps(sa, indent=0)[:1200])
