"""F-027 and F-029 result JSONs from d027_spectrum.json, d029_gate.json and d029_collision_v0.4.json."""
import json
import os
from dcommon import HERE, write_json
L = lambda n: json.load(open(os.path.join(HERE, n)))
jobs = {j["name"]: j for j in L("jobs_s2.json")}

sp = L("d027_spectrum.json")
S = sp["series"]
F27 = dict(diagnostic="F-027", status="run, with a substitution (see substitution)",
           substitution="The engine records phi(t) at ONE point per run (cfg 'center', the point of the 'point' series), not on a shell of several hundred points; a shell record needs a code change, so none was made. "
                        "Three runs: the stretch with the recording point on the stretch axis (z) and on a transverse axis (x), both at radius 25 from the lump, and the unstretched control at the x point. "
                        "The lump stays at the origin (lumps.centers); cfg 'center' only moves the recording point (and the centre of the unused r_c shells).",
           settings=dict(start="dip", source="outputs/session3/wpL3/sources/src_dip_N160_L88.npz window [26, 30] M 32", N=160, L=88.0, dx=0.55, dt=0.04, T=150.0, absorber="default sponge",
                         quad_eps=0.1, record="phi at the point, every step (dt 0.04), trilinear", points={"f027_q0.1_z": [0, 0, 25], "f027_q0.1_x": [25, 0, 0], "f027_control_x": [25, 0, 0]},
                         spectrum="phi(t) for t = 30 to 150, mean removed, Hann window, zero padded x8; periodogram. 'flux weighted' = omega k |phi|^2 with k = sqrt(omega^2 - 1)",
                         jobs=jobs_ if (jobs_ := {n: dict(T=j["T"], N=j["N"], center=j["cfg"]["center"]) for n, j in jobs.items() if n.startswith("f027")}) else None,
                         wall_s={"f027_q0.1_z": 530, "f027_q0.1_x": 546, "f027_control_x": 543}, drivers="make_jobs_s2.py, run_s2.py B / C, d027_post.py"),
           spectrum=sp,
           summary=dict(lump_omega=sp["omega_lump"], three_omega=sp["three_omega"],
                        stretch_minus_control_peak_omega_power={k: S[k]["flux_weighted"]["peak_omega_power"] for k in S if k.startswith("stretch_minus")},
                        stretch_minus_control_peak_omega_energy_flux={k: S[k]["flux_weighted"]["peak_omega_energy_flux"] for k in S if k.startswith("stretch_minus")},
                        stretch_minus_control_mean_omega_r={k: S[k]["flux_weighted"]["mean_omega_r_energy_flux_over_N_flux"] for k in S if k.startswith("stretch_minus")},
                        stretch_minus_control_flux_fraction_near_gap_1p15_1p40={k: S[k]["flux_weighted"]["flux_band_fraction"]["1.15-1.40 (near gap, 1.2-1.3 m)"] for k in S if k.startswith("stretch_minus")},
                        stretch_minus_control_flux_fraction_3omega_band={k: S[k]["flux_weighted"]["flux_band_fraction"]["2.40-3.10 (3 omega = 2.72)"] for k in S if k.startswith("stretch_minus")},
                        control_flux_fraction_3omega_band=S["control_x"]["flux_weighted"]["flux_band_fraction"]["2.40-3.10 (3 omega = 2.72)"],
                        control_mean_omega_r=S["control_x"]["flux_weighted"]["mean_omega_r_energy_flux_over_N_flux"]),
           ledger_causes_reading=[
               "Entry's reading A (near-gap radiation at 1.2 to 1.3 m carries the shed N): the stretch-minus-control spectrum peaks at omega = 1.07 (energy flux weighting 1.08), 0.07 above the mass gap and below the 1.2 to 1.3 the bookkeeping asked for; the flux-weighted mean omega_r is 1.08 (on both axes), and the 1.15 to 1.40 band holds 2 to 3% of the flux. Radiation just above the gap, not at 3 omega; the exact 1.2 to 1.3 value is not what was measured.",
               "Entry's reading B (a 3 omega = 2.72 peak means the shed N has another source): the stretch-minus-control signal has 0.3% of its flux in the 2.4 to 3.1 band (the unstretched control has 21% there, with its 3 omega peak at 2.723), so no 3 omega excess from the stretch is seen at this point in t = 30 to 150.",
               "Limits: one point per run, not a shell (an l = 2 pattern could be missed; the z and x points agree to 1% in peak and mean omega_r); the window sees only waves that have reached r = 25 by t = 150, which cuts off the slowest (omega - 1 below about 0.015, group speed below 0.17); frequency resolution 0.05 against a peak 0.07 above the gap."])
write_json(os.path.join(HERE, "F-027_stretch_spectrum.json"), F27)

g = L("d029_gate.json")
c = L("d029_collision_v0.4.json")
F29 = dict(diagnostic="F-029", status="run as proposed (gate passed, then the collision)",
           settings=dict(start="primary (source outputs/session2/wpE/source/src_prod_N160_L88.npz, window [26, 30], omega 0.909532 source)", v_per_lump=0.4, closing_speed=dict(galilean=0.8, relativistic=0.6897), gamma=1.0911,
                         phase_label_deg=0.0, d=30.0, gate=dict(N=160, L=88.0, T=60.0, theta=[0, 90], t_from=20.0, x0=-12.0, note="T = 60 not 80: a gate at x0 = -16 failed the box-face tail guard (l3 run_job: max tail beyond r_face / Phi0 1.27e-6) and was not run; the earlier v = 0.2 gate used x0 = -12"),
                         collision=dict(N=180, L=99.0, dx=0.55, dt=0.04, T=150.0, group_R=3 * 3.393 + 2.0, absorber="default sponge", t_contact_estimate=29.14, t_sponge_estimate=76.78, job="coll_primary_p0_v0.4"),
                         drivers="make_jobs_s2.py, run_s2.py A / B, gate_s2.py, d029_post.py"),
           gate=g, collision=c,
           summary=dict(gate_ok=g["ok"], speed_error=[v["speed_rel_err"] for v in g["results"].values()], omega_ratio_err=[v["omega_ratio_err"] for v in g["results"].values()],
                        outcome_by_l3_rules=c["summary"]["outcome"], outcome_last_three_windows=c["summary"]["outcome_end"], count_windows=[x["n"] for x in c["counts"]],
                        tracker_separation_extrema=c["tracker_separation_extrema"], E_group_over_E0_at_150=c["E_group_over_E0_at"]["150"], absorbed_over_E0=c["absorbed_over_E0_end"],
                        E_group_over_E0_since_free_contact=c["E_group_over_E0_since_free_contact_time"]),
           ledger_causes_reading=[
               "Does the merger give way to two out by 0.4 per lump: not by T = 150. The pair meets (tracker separation 2.8 at t = 35), rebounds to separation 10 (t = 65, count 2 at t = 60), falls back to 3.4 (t = 100 to 105) and sits at 3.0 to 3.2 at t = 150 with the count at 1 in the last three windows; no two free outgoing lumps (end speeds 0.053 per tracker, ball energy 0.20 E0 each).",
               "Cause 1 (dissipation about 30% of the pair, independent of speed): the energy in the group sphere (r < 12.2) is 0.41 E0 at T = 150 against 0.69 for the v = 0.2 run at the same time after the free contact (115), so this run has lost more by then, not the same; the run is 115 after contact and still falling (0.46 at 120, 0.41 at 150), so the long-time remnant is not measured.",
               "Cause 2 (the passage is not sudden at 0.2): at 0.4 the passage through 2 hw takes about 8.5 time units (1.2 periods), and the pair still does not separate, so a slower passage is not what holds it.",
               "Not tested: a speed above 0.4 per lump, the 1.5x box and resolution checks, the dip start; one run, one box, one resolution, T = 150."])
write_json(os.path.join(HERE, "F-029_collision_v0.4.json"), F29)
print("ok")
