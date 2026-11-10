import json, glob, os, numpy as np, csv
ROOT = "/Users/aaronramos/Desktop/SWIRL/.claude/worktrees/s5-trackF"; OUT = ROOT + "/outputs/session5/trackF"
L = lambda p: json.load(open(p))
res = {}
# item 4: frac(0) < 0.90 counts, distinct starts
rb = ["B_R_bump_a0.2", "B_R_bump_a0.33", "B_R_bump_a0.55", "B_R_bump_a0.79", "B_R_bump_a0.90", "B_R_bump_a0.95", "B_R_bump_a1.0", "E_R_bump_E1000_a0.17715"]
rp = ["B_R_pinch_a0.0384", "B_R_pinch_a0.0916", "B_R_pinch_a0.2", "B_R_pinch_a0.36", "B_R_pinch_a0.4384", "B_R_pinch_a0.4861", "B_R_pinch_a0.51", "E_R_pinch_E1000_a0.10892", "E_R_pinch_E533_a0.26158"]
f0 = lambda t: L(f"{OUT}/runs/{t}.json")["run"]["frac_t0"]
res["frac0_bump"] = {t: f0(t) for t in rb}; res["frac0_pinch"] = {t: f0(t) for t in rp}
res["n_bump_below_0.90"] = sum(v < 0.9 for v in res["frac0_bump"].values()); res["n_pinch_below_0.90"] = sum(v < 0.9 for v in res["frac0_pinch"].values())
# Td/hw^2 production R bump (reviewer's series)
td = {}
for t in rb[:7]:
    d = L(f"{OUT}/runs/{t}.json"); hw = d["run"]["params"]["hw"]; T = d["run"]["T_death"]; Tw = d["twin"]["T_death"]
    td[t] = dict(hw=hw, T_death=T, Td_hw2=None if T is None else T / hw ** 2, twin_T_death=Tw, twin_Td_hw2=None if Tw is None else Tw / hw ** 2)
res["production_R_bump_Td_over_hw2"] = td
# loss rates O bump
tags = {"0.98": "C_O_bump_w0.91_x0.98", "0.99": "C_O_bump_w0.91_x0.99", "1.00": "B_O_bump_w0.91", "1.01": "C_O_bump_w0.91_x1.01", "1.02": "C_O_bump_w0.91_x1.02", "1.05": "C_O_bump_w0.91_x1.05"}
lr = {}
for k, t in tags.items():
    p = f"{OUT}/runs/{t}.json"; r = L(p)["run"]; T = np.array(r["series"]["t"]); f = np.array(r["series"]["frac"])
    g = lambda x: f[np.argmin(abs(T - x))]
    lr[k] = dict(frac_20000=float(g(20000)), rate_2000_10000=float(-np.log(g(10000) / g(2000)) / 8000), rate_0_20000=float(-np.log(g(20000) / g(0)) / 20000), rate_5000_15000=float(-np.log(g(15000) / g(5000)) / 10000))
res["O_bump_loss_rates"] = lr
# item 7: R pinch upper-branch E vs bump E crossing in rate
fb = L(f"{ROOT}/outputs/radial5/family_R_bump.json")["rows"]; fp = L(f"{ROOT}/outputs/radial5/family_R_pinch.json")["rows"]
wb = np.array([r["omega"] for r in fb[:32]]); Eb = np.array([r["E"] for r in fb[:32]]); o = np.argsort(wb)
up = fp[58:]; wp = np.array([r["omega"] for r in up]); Ep = np.array([r["E"] for r in up])
ww = np.linspace(0.9995 if False else max(wb.min(), wp.min()), min(wb.max(), wp.max()), 20000)
d = np.interp(ww, wb[o], Eb[o]) - np.interp(ww, wp, Ep)
sc = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
res["R_pinch_upper_vs_bump_E_crossing_rate"] = [float(ww[i]) for i in sc]
# item 8 O12 constants
for n, key in [("R_bump", "R_bump"), ("O_bump", "O_bump")]:
    R = L(f"{ROOT}/outputs/radial5/family_{n}.json")["rows"][:6]
    res["small_amp_" + n] = [dict(F0=r["shoot_param"], F_peak=r["F_peak"], omega=r["omega"], c=(1 - r["omega"]) / r["shoot_param"] ** 2) for r in R]
res["R_bump_c_limit"] = res["small_amp_R_bump"][0]["c"]; res["ratio_3_16_over_c"] = 0.1875 / res["R_bump_c_limit"]
json.dump(res, open(OUT + "/diagnostics/items.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if not k.startswith("small_amp_")}, indent=0)[:3500])
print(res["small_amp_R_bump"][:2])
