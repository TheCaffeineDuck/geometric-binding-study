"""Assemble the per-diagnostic result JSONs (F-023 ... F-029) from the partial files written by the d0xx scripts."""
import json
import os
import numpy as np
from dcommon import HERE, write_json

L = lambda n: json.load(open(os.path.join(HERE, n)))

# ---- F-023
rows = []
for kind in ("field", "velocity"):
    for r in L("d023_%s_partial.json" % kind):
        rows.append(dict(kind=kind, phase=0 if kind == "field" else 90, ball=r["ball"], eps_bracket=[r["unbroken"]["eps"], r["broken"]["eps"]],
                         t0_dE_over_E_threshold=r["dE_over_E_threshold"], t0_dN_over_N_threshold=r["dN_over_N_threshold"],
                         bracket_width_dE=r["bracket_width_dE"], t_first_below_half_of_broken_run=r["broken_time"], n_runs=len(r["history"])))
dE = np.array([r["t0_dE_over_E_threshold"] for r in rows]); dN = np.array([r["t0_dN_over_N_threshold"] for r in rows])
F23 = dict(diagnostic="F-023", status="run, with a substitution (see substitution)",
           substitution="Phase 45 was NOT run: wpl.run_one and t0_values support phase 0 and 90 only (the 'phase' spec key is read as 90 or not-90), so a 45 degree kick needs a code change. "
                        "Instead the kick's dE/dN ratio was varied through the kick's radial shape (spec key 'ball', a configuration key that wpl.disturb already reads), "
                        "for the field kick (phase 0) and the velocity kick (phase 90). Same question: is the t = 0 dE/E at the break threshold the same whatever the kick.",
           settings=dict(start="dip (beta 4/3, omega 0.906505)", T=2000.0, scale=1.0, side="negative eps", control="outputs/session3/wpL/runs/dip_control_T2000_s1.json",
                         break_criterion="wpl.compare()['broken']['broken_outright'] (E3 below half within 2000 of the kick)", balls_hw=[[2, 3], [1, 2], [3, 4]],
                         bracket_stop="width in t = 0 dE/E <= 0.004", drivers="d023.py field; d023.py velocity"),
           results=rows,
           summary=dict(dE_over_E_threshold_mean=float(dE.mean()), dE_over_E_min=float(dE.min()), dE_over_E_max=float(dE.max()),
                        dE_spread_over_smallest=float((np.abs(dE).max() - np.abs(dE).min()) / np.abs(dE).min()),
                        within_pm_0p01_of_minus_0p41=bool(np.all(np.abs(dE + 0.41) <= 0.01)),
                        dN_over_N_threshold_field=[r["t0_dN_over_N_threshold"] for r in rows if r["kind"] == "field"],
                        dN_over_N_threshold_velocity=[r["t0_dN_over_N_threshold"] for r in rows if r["kind"] == "velocity"],
                        dN_spread_over_smallest=float((np.abs(dN).max() - np.abs(dN).min()) / np.abs(dN).min()),
                        ledger_standard_ball_values="field -0.4155, velocity -0.4112 (scan_summary bracket midpoints); this run's standard ball: field -0.4124, velocity -0.4100"),
           ledger_causes_reading=[
               "Cause 2 (wrong coordinate: E, not N, is the threshold coordinate): all six thresholds sit at dE/E -0.4077 to -0.4138, inside the prediction's -0.41 +- 0.01, while dN/N at threshold runs -0.408 to -0.468 (13% spread).",
               "Cause 3 (N is only an adiabatic invariant): the field-kick dN/N thresholds exceed the velocity ones by 0.05 (dN/N != dE/E at phase 0), no universal dN/N; consistent, not separated from cause 2 by this test.",
               "Cause 1 (relaxation sheds more N, threshold inside the N31 ceiling): not tested here; no result bears on it.",
               "Caveat: the narrow-ball (1, 2 hw) breaks occur at 1794 to 1955 after the kick, close to the 2000 horizon, so those two thresholds depend on the horizon."])
write_json(os.path.join(HERE, "F-023_energy_threshold.json"), F23)

# ---- F-024
d = L("d024_partial.json")
fam = {s["N_over_Nmin"]: s for s in d["starts"]}
pairs = [("slow 0.2", 1.821), ("k3 0.05", 1.824), ("slow 0.05", 1.827)]
cmp_ = []
for name, tg in pairs:
    lab, ef = name.split()
    pr = next(p for p in d["packet_rows"] if p["packet"] == lab and abs(p["efrac"] - float(ef)) < 1e-9)
    f = fam[tg]
    cmp_.append(dict(packet=name, family_N_over_Nmin=tg, packet_dN_settled=pr["dN_settled"], packet_early=pr["early"], family_early=f["early_500_2000"],
                     early_ratio_packet_over_family=pr["early"] / f["early_500_2000"], packet_late=pr["late"], family_late=f["late_alive_mean"],
                     late_ratio_packet_over_family=pr["late"] / f["late_alive_mean"], packet_early_over_control=pr["early"] / pr["early_control"],
                     packet_late_over_control=pr["late"] / pr["late_control"]))
F24 = dict(diagnostic="F-024", status="run as proposed",
           settings=dict(start="dip family, beta 4/3, kind none, T = 30000, scale 1, three starts at profile N / N_min = 1.821, 1.824, 1.827 (N_min = N_profile(dip)/1.8295086 = %.4f; "
                                 "wpl.STARTS patched at run time, omega found by brentq on solve_profile)" % d["N_min"],
                         omega_profiles={str(k): v["omega"] for k, v in fam.items()}, omega_settled={str(k): v["omega_s"] for k, v in fam.items()},
                         early_rate="log fit of E3 over t_d + 500..2000", late_rate="mean of the 500-wide gamma windows after t_d + 2000 while frac > 0.5 (same as l1_summary.py)",
                         driver="d024.py then d024_post.py (packet rows and controls read from scan_summary.json)"),
           family_runs=d["starts"], packet_rows=d["packet_rows"], comparison=cmp_,
           summary=dict(max_ratio_packet_over_family_early=max(c["early_ratio_packet_over_family"] for c in cmp_),
                        max_ratio_packet_over_family_late=max(c["late_ratio_packet_over_family"] for c in cmp_),
                        family_early_rate_change_1p821_to_1p827=fam[1.821]["early_500_2000"] / fam[1.827]["early_500_2000"],
                        family_rate_per_percent_N="x%.0f per 1%% in N (from the end points, %.2f%% apart)" % (np.exp(np.log(fam[1.821]["early_500_2000"] / fam[1.827]["early_500_2000"]) / (0.006 / 1.824 * 100) ), 0.006 / 1.824 * 100)),
           ledger_causes_reading=[
               "Cause 2 (the family rate is so steep in N that a small N mismatch gives the excess): the family at the packets' settled N/N_min gives the packet rates to within 0.88 to 1.15 (early and late), against 4 to 24 times the control, so the excess is the family flank at that N.",
               "Cause 1 (a lasting excess at unchanged N, a state variable the family rule does not carry): not seen in these three packet cases (packet over family at most 1.15); the three starts are the k3 0.05 and the slow 0.05 and 0.2 packets only.",
               "Rule from the entry: excess falling to 3 times or less means an N-resolution artifact at the dip; it fell to 1.15 or less.",
               "Causes 3, 4 and 5 (late-clause design, kick transient, b5 floor) are not addressed by this run."])
write_json(os.path.join(HERE, "F-024_family_at_packet_N.json"), F24)

# ---- F-026
d = L("d026_partial.json")
rr = d["rows"]
dip = {r["efrac"]: r for r in rr if r["start"] == "dip"}
pri = {r["efrac"]: r for r in rr if r["start"] == "primary"}
efs = sorted(dip)
ratio = {ef: abs(pri[ef]["dN_after_pass"]) / abs(dip[ef]["dN_after_pass"]) for ef in efs if ef in pri}
ks = sorted(ratio)
cross = None
for a, b in zip(ks[:-1], ks[1:]):
    if ratio[a] >= 2 > ratio[b]:
        cross = float(np.exp(np.interp(np.log(2.0), [np.log(ratio[b]), np.log(ratio[a])], [np.log(b), np.log(a)])))
        cross_dip_dN = float(np.exp(np.interp(np.log(cross), np.log([a, b]), np.log([abs(dip[a]["dN_after_pass"]), abs(dip[b]["dN_after_pass"])])))) 
        cross_pri_dN = float(np.exp(np.interp(np.log(cross), np.log([a, b]), np.log([abs(pri[a]["dN_after_pass"]), abs(pri[b]["dN_after_pass"])]))))
dexp = {"%g-%g" % (a, b): float(np.log(abs(dip[b]["dN_after_pass"] / dip[a]["dN_after_pass"])) / np.log(b / a)) for a, b in zip(efs[:-1], efs[1:])}
pexp = {"%g-%g" % (a, b): float(np.log(abs(pri[b]["dN_after_pass"] / pri[a]["dN_after_pass"])) / np.log(b / a)) for a, b in zip(efs[:-1], efs[1:]) if a in pri and b in pri}
F26 = dict(diagnostic="F-026", status="run as proposed, with a note",
           note="The entry lists five dip efrac values (0.03, 0.04, 0.07, 0.15, 0.3) and says six runs. Ran those five at the dip and the same five at the primary (the ratio needs both): 10 new runs. The efrac 0.05 points are the existing runs (dip_pk_k2.53021_sig5_ef0.05 and primary), re-read; the other old grid points (0.01, 0.02, 0.1, 0.2, 0.5, 1, 2) are not in this worktree's wpL/runs (49 files), so the ratios for those are not recomputed here.",
           settings=dict(kind="packet k3, sigma 5, k0 = 2.530209 (dip), 2.539136 (primary), R0 default, T = 2000, scale 1, efrac new = 0.03, 0.04, 0.07, 0.15, 0.3",
                         windows="dN/N of the period-averaged N against the same-start control: (250, 750); (750, 2000); (t_pass + 200, 2000) with t_pass = 83.0 (wpl.free_packet)",
                         driver="d026.py"),
           rows=rr, ratio_primary_over_dip_after_pass_window=ratio,
           ratios_all_windows=d["ratios"],
           check_against_ledger="efrac 0.05: ratio %.3f (t_pass+200..2000), %.3f (250..750); ledger: reviewer 2.58, scholar 2.3 (250..750)" % (ratio[0.05], abs(pri[0.05]["dN_250_750"] / dip[0.05]["dN_250_750"])),
           summary=dict(efrac_where_ratio_passes_2=cross, dip_abs_dN_over_N_there=cross_dip_dN, primary_abs_dN_over_N_there=cross_pri_dN,
                        dip_response_exponent_between_points=dexp, primary_response_exponent_between_points=pexp, any_broken=bool(any(r["broken"] for r in rr))),
           ledger_causes_reading=[
               "Cause 1 (decoupled while the packet's own dN stays inside the dip width, coupling of the primary above): the ratio falls through 2 at efrac about %.3f where the dip's |dN/N| is %.4f (about %.0f%% of the 1.7%% dip width in N); the data are consistent with a crossover at about a third of the width. A fixed efrac against a fixed |dN/N| is not separated here (at the dip they are tied; the b5 comparison would separate them, not run)." % (cross, cross_dip_dN, 100 * cross_dip_dN / 0.017),
               "Cause 2 (the dip's response is steeply superlinear at small amplitude): the dip's exponent in efrac is %s, falling toward the primary's %s." % (list(np.round(list(dexp.values()), 2)), list(np.round(list(pexp.values()), 2))),
               "Cause 3 (ratio of 36 at 0.01 rests on an estimator near its floor): not tested by this grid (0.01 not rerun); the ratio at 0.03 is %.2f." % ratio[0.03]])
write_json(os.path.join(HERE, "F-026_efrac_grid.json"), F26)

# ---- F-028
d = L("d028_partial.json")
F28 = dict(diagnostic="F-028", status="run as proposed (calculation, no PDE); seeds 2 and 3 added because the calculation is cheap",
           settings=dict(source=d["source"], window=[26.0, 30.0], M=32, generator="swirl.engine.boost.lump_cube(H, 0.0, random={amp, ell, seed}) with the dip L3 job values ell = 3.4 (dip hw %.4f), amp 0.1 and 0.15, seeds 1, 2, 3" % d["hw"],
                         n=d["n"], dx=d["dx"], N0_inst=d["N0_inst"], E0=d["E0"], omega=d["omega"], kappa_gap=d["kappa_gap"], driver="d028.py",
                         projection="perturbation (f - 1)(phi, phi_t) Fourier decomposed; N_pert = int(dphi_t^2 + omega^2 dphi^2)/(2 omega); fractions by |k| band and by radius"),
           runs=d["runs"],
           caveat="lowpass_added_N in each run adds a low-passed perturbation to the lump and reads N_inst; it contains the first-order cross term (the low-passed xi no longer has zero rho-weighted mean), so its values (above 1) are not a fraction of the added N and are not used.",
           ledger_causes_reading=[
               "Imposed t = 0: dN/N = a^2 exactly (1.0e-2, 2.25e-2) for every seed; dE/E at a = 0.1 is 1.04e-2, 0.87e-2, 0.91e-2 (seeds 1, 2, 3) and at 0.15 is 2.42e-2, 2.13e-2, 2.22e-2; seed 1 reproduces the ledger's 2.4e-2.",
               "The first-order overlap of the perturbation with the lump's own profile (phi) is zero by construction (the multiplier is normalised to zero rho-weighted mean), so the added N is entirely second order; all of it sits inside r < 3 hw (98% for seed 1).",
               "Cause 2 (the lump keeps what projects onto it, the rest is radiation): the cumulative share of the added N below |k| = 1/hw (0.295) is 9.9%, 7.9%, 5.3% (seeds 1, 2, 3); below the gap wavenumber kappa = 0.421 it is 24%, 19%, 17%; below 2/hw (0.589) it is 55%, 46%, 45%; below 1.0 it is 95%, 93%, 93%. The measured kept fraction (36% and 39%, seed 1) lies between the kappa and 2/hw cuts; a cut at 1/hw, the one the entry proposed, gives 10%, not 36 to 39%.",
               "Cause 4 (one seed): the shares differ by up to 2x between seeds at the low cuts (below 1/hw: 5.3% to 9.9%), so the kept fraction can differ by seed by this projection."])
write_json(os.path.join(HERE, "F-028_multiplier_projection.json"), F28)
print("ok", cross, cross_dip_dN)
