"""Build lifetime_table.json and lifetime_vs_beta.png from long_*.json, ranking.json, ratemap_*.json, family_*.json."""
import os, sys, json, glob
os.environ["NUMBA_NUM_THREADS"] = "1"
import numpy as np
sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL")
from swirl.family.wpa_run import profile, write_json
import swirl.family.wpa_family as fam
OUT = "/Users/aaronramos/Desktop/SWIRL/outputs/session2/wpA"
BETAS = [0.75, 0.85, 1.0, 1.15, 4/3, 1.6, 2.0, 3.0, 5.0]
rk = json.load(open(f"{OUT}/ranking.json"))["per_beta"]
rows, perbeta = [], []
for b in BETAS:
    key = "%.4f" % b
    rm = json.load(open(f"{OUT}/ratemap_beta{key}.json"))
    st = rm["settings"]
    famj = json.load(open(fam.family_path(b)))
    Nmin = famj["features_rescaled"]["N_min"]
    # detection floor and deepest dip from map windows
    pts = []
    for r in rm["runs"]:
        for w in r["windows"]:
            pts.append((max(w["Gamma"], 2 * w["sigma"]), w["Gamma"], w["sigma"], w["omega"], r["omega_profile"], r["alpha2"], w["E"]))
    deep = min(pts, key=lambda x: (x[0], x[3]))
    pos = [p for p in pts if p[1] > 2 * p[2]]
    deep_meas = min(pos, key=lambda x: x[1]) if pos else None
    floors = np.array([2 * p[2] for p in pts])
    perbeta.append(dict(beta=b,
        deepest_dip_floored=dict(Gamma_floored=deep[0], Gamma_central=deep[1], sigma=deep[2], omega_window=deep[3], omega_profile=deep[4], alpha2=deep[5], E=deep[6],
                                 is_below_floor=bool(deep[1] <= 2 * deep[2]), source="ratemap_beta%s.json runs[].windows[]" % key),
        deepest_dip_detected=(None if deep_meas is None else dict(Gamma=deep_meas[1], sigma=deep_meas[2], omega_window=deep_meas[3], omega_profile=deep_meas[4], E=deep_meas[6])),
        detection_floor_2sigma=dict(min=float(floors.min()), median=float(np.median(floors)), max=float(floors.max()), source="ratemap windows, 2*sigma"),
        omega_c=st["omega_c"], E_min_setB=st["E_min_setB"], alpha2_c=st["alpha2_c"], omega_c_source="ratemap_beta%s.json settings.omega_c (alpha2_c = N minimum, energetic death)" % key,
        omega_end_analytic=famj["features_rescaled"].get("omega_end_analytic")))
    ws = {}
    for k in ("best", "best_by_measured_rate"):
        ws.setdefault(round(rk[key][k]["omega_profile"], 6), []).append(k)
    for w6, kinds in sorted(ws.items()):
        f = glob.glob(f"{OUT}/long_beta{key}_w{w6:.6f}.json")[0]
        L = json.load(open(f))
        pk = rk[key][kinds[0]]
        p = profile(b, L["params"]["omega_profile"])
        t = np.array(L["series"]["t"]); fr = np.array(L["series"]["frac"])
        runmin = np.minimum.accumulate(fr)
        rise = float((fr - runmin).max())
        W = L["windows_1000"]
        alive30k = L["T_death"] is None and L["T_end"] >= 30000 - 1
        reg = L["regime"]; om = reg["omega_over_m_settled"]
        rows.append(dict(
            beta=b, omega_start=L["params"]["omega_profile"], pick="+".join(kinds), long_file=os.path.basename(f),
            E0=L["E0"], N_over_Nmin=p["N"] / Nmin, E_over_mN=p["E_over_mN"], half_width=L["params"]["half_width"],
            omega_over_m_settled=om, omega_over_m_end=reg["omega_over_m_end"],
            nl_share_t0_signed=reg["nonlinear_share_t0_signed"], nl_share_t0_abs=reg["nonlinear_share_t0_absolute"],
            nl_share_mean_signed=reg["nonlinear_share_mean_signed"], nl_share_mean_abs=reg["nonlinear_share_mean_absolute"],
            T_end=L["T_end"], T_90=L["T_90"], T_death=L["T_death"], osc_T_90=L["osc_T_90"], osc_T_death=L["osc_T_death"], osc_T_end=L["osc_T_end"],
            alive_at_30000=alive30k, frac_at=L["frac_at"], frac_end=L["frac_end"], frac_t0=L["frac_t0"], max_frac_rise_above_running_min=rise,
            t_s=L["t_s"], n_windows_1000=len(W), omega_first_window=(W[0]["omega"] if W else None), omega_last_window=(W[-1]["omega"] if W else None),
            Gamma_first_window=(W[0]["Gamma"] if W else None), sigma_first_window=(W[0]["sigma"] if W else None),
            Gamma_last_window=(W[-1]["Gamma"] if W else None), sigma_last_window=(W[-1]["sigma"] if W else None),
            pathint=dict(T_90=pk["T_90"], T_90_lo=pk["T_90_lo"], T_90_hi=pk["T_90_hi"], T_death=pk["T_death"], T_death_lo=pk["T_death_lo"], T_death_hi=pk["T_death_hi"],
                         flags_90=pk["flags_90"], flags_death=pk["flags_death"], T_90_before_t_s=pk["T_90_before_t_s"], source="ranking.json per_beta[%s][%s]" % (key, kinds[0])),
            twin=dict(frac_T100=L["linear_twin"]["frac_T100"], frac_T300=L["linear_twin"]["frac_T300"], frac_T1000_end=L["linear_twin"]["frac_T1000"]),
            E_h_rise_max_rel=L["energy_check"]["E_h_rise_max_rel"], wall_s=L["wall_s"],
            sources="long_*.json: E0,regime,T_*,osc_*,frac_*,windows_1000,linear_twin,energy_check,params.half_width; N_over_Nmin,E_over_mN: profile() at start omega / family features_rescaled.N_min"))
write_json(f"{OUT}/lifetime_table.json", dict(description="WP-A lifetime against beta. One row per long run; per_beta has dip, omega_c, floor.", rows=rows, per_beta=perbeta))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for r in rows:
    mk = "o" if "best_by" not in r["pick"] or "best" == r["pick"] else "s"
    col = "C0" if r["pick"].startswith("best") and "by_measured" not in r["pick"] else "C1"
    if r["pick"] == "best+best_by_measured_rate": col = "C2"
    for T, name, ax_ in ((r["T_90"], "T90", ax[0]), (r["T_death"], "Tdeath", ax[1])):
        if T is None:
            ax_.plot(r["beta"], 30000, "^", color=col, mfc="none", ms=9)
        else:
            ax_.plot(r["beta"], T, mk, color=col, ms=6)
for a, ttl in zip(ax, ("T_90 (open triangle = not reached by 30,000)", "T_death (open triangle = alive at 30,000)")):
    a.set_yscale("log"); a.set_xscale("log"); a.set_xlabel("beta"); a.set_ylabel("T (m = 1)"); a.set_title(ttl, fontsize=9); a.grid(alpha=.3)
ax[0].plot([], [], "o", color="C0", label="best (path integral)"); ax[0].plot([], [], "s", color="C1", label="best by measured rate"); ax[0].plot([], [], "o", color="C2", label="same start")
ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/lifetime_vs_beta.png", dpi=130)
print("ok", len(rows))
