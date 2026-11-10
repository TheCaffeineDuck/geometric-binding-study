import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_tables import *
items = L(D + "/items.json"); r21 = open(D + "/rule21_columns.md").read()
def get(tag):
    p = path(tag)
    return metrics(tag) if p else None
def f(x, fmt="%.4g"): return "-" if x is None else fmt % x
def row(m, label):
    if m is None: return "| %s | not finished (no output) |" % label
    td = ("guard trip at %g" % m["T_end"] if m["guard"] else "> %g (T_end)" % m["T_end"]) if m["T_death"] is None else "%.1f" % m["T_death"]
    ttd = "> %g" % m["twin_T_end"] if m.get("twin_T_death") is None else "%.1f" % m["twin_T_death"]
    return "| %s | %.2f | %.4f | %.4f | %.4f | %.4f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (label, m["hw"], m["frac0"], m["frac_cyc_ref"], m["retained_ref"], m["rate_at_ref"], f(m["cycle_rate_above_1"], "%d"),
        f(m["T_90"], "%.1f"), td, f(m.get("Td_over_hw2"), "%.2f"), ttd, f(m.get("twin_Td_over_hw2"), "%.2f"), f(m.get("ratio_run_twin"), "%.2f"), "%.2e" % m["twin_frac_ref"] if m.get("twin_frac_ref") is not None else "-", "%+.5f" % m["nl_run"] if m["nl_run"] is not None else "-", "%s" % ("T_end %g%s; %d cycles used" % (m["T_end"], " (GUARD TRIP, class unresolved)" if m["guard"] else "", m["cyc_ref"])))
H = "| start | hw | frac(0) | frac cycle-avg at cycle 1000 (or last) | retained = frac/frac(0) | zero-crossing rate at cycle 1000 | cycle at which rate (21-period median) exceeds 1 | T_90 | T_death | T_death/hw^2 | twin T_death | twin T_death/hw^2 | run/twin | twin frac at same t | nonlinear share (run, t 50-150) | run ended |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
df1 = [H] + [row(get(t), l) for t, l in [("DF1_hw29_a0.13887", "DF1 amp 0.13887"), ("DF1_hw37_a0.107627", "DF1 amp 0.107627"), ("DF1_hw45_a0.0885", "DF1 amp 0.0885")]]
df2 = [H] + [row(get(t), l) for t, l in [("DF2_hw37_x0.99", "hw37 x0.99 (400/800)"), ("DF1_hw37_a0.107627", "hw37 x1.00 (400/800, DF1 run)"), ("DF2_hw37_x1.01", "hw37 x1.01 (400/800)"), ("DF2_hw37_box1600", "hw37 x1.00, sponge 800/1600")]]
df3 = [H] + [row(get("DF3_pinch_hw45_a0.008031"), "DF3 R pinch amp 0.008031 (hw ~45), sponge 400/800, T 6600")]
df4 = [H.replace("hw |", "hw |", 1)] + [row(get(t), l) for t, l in [("DF4_O_pinch_w0.94_x0.99", "O pinch 0.94 x0.99"), ("B_O_pinch_w0.94", "O pinch 0.94 x1.00 (production, twin to 20000)"), ("DF4_O_pinch_w0.94_x1.01", "O pinch 0.94 x1.01")]]
prod = items["production_R_bump_Td_over_hw2"]
extra_td = []
for t in ["B_R_bump_a0.2_T6600", "E_R_bump_E1000_a0.17715_T6600"]:
    m = metrics(t); extra_td.append("%s hw %.2f T_death %s (%.2f hw^2) twin %s (%.2f hw^2)" % (t, m["hw"], m["T_death"], m["T_death"] / m["hw"] ** 2, m["twin_T_death"], m["twin_T_death"] / m["hw"] ** 2))
lr = items["O_bump_loss_rates"]
lrt = "| scale | frac at T=20,000 (instantaneous) | loss rate -d ln frac/dt, t 2000 to 10000 | t 5000 to 15000 | t 0 to 20000 | reviewer's |\n|---|---|---|---|---|---|\n" + "\n".join("| %s | %.4f | %.2e | %.2e | %.2e | %s |" % (k, v["frac_20000"], v["rate_2000_10000"], v["rate_5000_15000"], v["rate_0_20000"], r_) for (k, v), r_ in zip(lr.items(), ["4.2e-5", "2.9e-5", "9.2e-6", "1.5e-6", "about 0", "2.2e-6"]))
sa = items["small_amp_R_bump"]; so = items["small_amp_O_bump"]
sat = "| F(0) (peak, = F_peak) | rate omega | c = (1 - omega)/F(0)^2 |\n|---|---|---|\n" + "\n".join("| %.5f | %.8f | %.6f |" % (s["F0"], s["omega"], s["c"]) for s in sa)
sot = "O bump: " + ", ".join("%.6f" % s["c"] for s in so)
body = f"""
## Added after review (follow-up to the reviewer's recomputation; numbers only, no new verdicts)

All production rows above stand. Diagnostics live in `diagnostics/` (runs, `diag_metrics.json`, `items.json`, `rule21_columns.md/json`, `DIAGNOSTICS.md`).

### A1. Rule 2.1 two ways, every lifetime row (B, C and E runs)

Column A is the rule by the letter: cycle-averaged frac at 1,000 integrated cycles (the 1,000th row of the per-period table; for runs shorter than 1,000 cycles the last row, marked by "cycles used"), the zero-crossing rate at that cycle (median of 21 periods), the twin at the same time; the flags are (frac >= 0.90, twin <= 0.5, rate below 1 by more than 0.0314). Column B is the retained-fraction reading frac/frac(0) with the same rate and twin.

Two classifier defects (in `evolve.classify`, not edited): (1) T_death short-circuit: any run with a T_death, however late, returns "disperses" before the 1,000-cycle test is made (O pinch 0.94: T_death 13,666 TU = 2,063 cycles, O bump 0.99 scale: 19,608, O pinch 0.995: 8,035). (2) The 1,000-cycle time is 1000 x 2 pi / (rate over the last 100 time units), the end-of-run rate, not the integrated count: O pinch 0.94 gives 6,258 against 6,652 integrated (late rate 1.0040 after dispersal); O bump 0.99 scale 6,280 against 6,882; O bump 0.909 6,558 against 6,903.

{r21}

By the letter at 1,000 integrated cycles the following meet rule 2.1: O bump 0.909, O pinch 0.86, **O pinch 0.94 (0.9643, rate 0.9487, twin 1.5e-7)**, O bump 0.909 at 0.99, 1.01, 1.02, 1.05 times amplitude, O pinch 0.86 at 0.99 and 1.01, and also E_O_bump_E10k_w0.8531 (0.9080, a matched-energy run). So the statement "only two starts meet rule 2.1" is replaced by "three family starts (O bump 0.909, O pinch 0.86, O pinch 0.94)", plus the amplitude-scaled neighbours and the matched-energy bump 0.8531. By the retained reading B_R_bump_a0.2 (T = 2000, 317 cycles) and E_R_bump_E1000_a0.17715 (T = 2000) look held, but both die before 1,000 cycles (T6600 rows).

### A2. O bump 0.909, 1% rule wording

At 0.99 times amplitude: frac 0.9003 at the threshold (marginal), T_90 6,880 (t at 1,000 integrated cycles 6,882, so T_90 falls at 999 cycles); at 0.98 clear fail (0.8187). Replaces "fails as written".

### A3. Loss-rate table (checked from my runs; "not a tuned peak" is removed)

Loss rate = -d ln(frac)/dt of the instantaneous 3 hw frac series, fitted between two times. My values for t 2000 to 10000 are within 8% of the reviewer's list except the 1.02 and 1.05 rows (the reviewer's "about 0" matches 4.6e-8; 1.05 is 2.0e-6 here against 2.2e-6).

{lrt}

Session 2 context: `docs/SESSION_02_REPORT.md` line 34 (the narrow dip at rate 0.9065) and lines 78 to 79 (the Session 1 primary on its wall): the baseline start is at that wall, so the rise of lifetime with amplitude above it is consistent with that record.

### A4. frac(0) below 0.90

R bump: {items['n_bump_below_0.90']} of 8 distinct starts (amp 0.55, 0.79, 0.90, 0.95, 1.0: 0.895, 0.851, 0.803, 0.766, 0.714; the others 0.910 to 0.915). R pinch: {items['n_pinch_below_0.90']} of 9 (amp 0.51: 0.884; the others 0.945 to 0.9997). Section 0 point 1 applies to those, not to all R starts.

### A5. Collapse-time thresholds

Each collapse time is the first crossing of min(1 - X) below a stated depth, common to the two finest grids: sigma 2: 3.822 at 1e-1; sigma 3: 12.847 at 1e-2 (A = 0.8500) and, at the first high-side point A = 1.0, 7.291 at 1e-7 with h/2 = 7.400 (**h/2 and h/4 differ by 1.5% at depth 1e-7**); sigma 5: 42.003 at 1e-1; shell: 36.131 at 1e-2 (finest two grids only). The 0.1% to 0.2% agreement quoted before holds for the shallow thresholds only.

### A6. Labels

The O pinch 0.86 100,000 TU run (`B_O_pinch_w0.86_T100k`) was added after the 20,000 TU result showed it alive. The "three diagnostics" after the 1% result are four runs (O bump 0.98, 1.02, 1.05 at T = 20,000, and 1.02 at T = 100,000); their twins ran to T = 2,000 only (none for the 100,000 run).

### A7. F2

The O energy minima (bump rate 0.9711, pinch 0.9558) sit at rates where starts disperse (O bump 0.965 and O pinch 0.965 die at 285 and 71 cycles); the held states (0.909, 0.86) are beyond the minimum, on the narrower side. The R pinch upper-branch energy crosses the R bump energy at rate {items['R_pinch_upper_vs_bump_E_crossing_rate'][0]:.5f} (previously quoted as 0.9975).

### A8. O12 small-amplitude constant

Definition: c = (1 - omega)/F(0)^2 with F(0) the peak amplitude of the radial profile F(r) in f = F(r) cos(omega t) (for the bump F(0) = F_peak, the column checked), omega the family's rate, six lowest-amplitude points of `family_R_bump.json`:

{sat}

Limit c = {items['R_bump_c_limit']:.6f}; 3/16 = 0.1875; ratio 3/16 over c = {items['ratio_3_16_over_c']:.2f}. {sot}.

### A9. Follow-up runs (DF1 to DF4), sponge 400/800 unless stated; T_death for DF1 is of the nonlinear run (stop-at-death), twin run to the stated T

DF1 (lifetime scaling; R bump family starts at hw 28.6, 37.0, 45.0):

{chr(10).join(df1)}

Production R bump starts with twins (T_death/hw^2): {'; '.join('%s hw %.2f: %s / %s' % (t.replace('B_R_bump_',''), v['hw'], f(v['Td_hw2'],'%.2f'), f(v['twin_Td_hw2'],'%.2f')) for t, v in prod.items())}; {'; '.join(extra_td)}.

DF2 (1% neighbours and box, hw 37):

{chr(10).join(df2)}

DF3 (R pinch, hw about 45, small-amplitude branch, amp 0.008031, interpolated from the family table):

{chr(10).join(df3)}

DF4 (O pinch 0.94 and its 1% neighbours, T = 20,000, twin to 2,000 for the neighbours):

{chr(10).join(df4)}
"""
open(D + "/DIAGNOSTICS.md", "w").write("# Track F follow-up diagnostics (DF1 to DF4) and added table text\n" + body)
t = open(OUT + "/TABLES.md").read()
if "## Added after review" in t: t = t.split("\n## Added after review")[0]
open(OUT + "/TABLES.md", "w").write(t + "\n" + body)
s = L(OUT + "/summary.json"); s["added_after_review"] = dict(items=items, diag_metrics=json.load(open(D + "/diag_metrics.json")), rule21_columns=json.load(open(D + "/rule21_columns.json")))
json.dump(s, open(OUT + "/summary.json", "w"), indent=1, default=float)
print(body[-6000:])
