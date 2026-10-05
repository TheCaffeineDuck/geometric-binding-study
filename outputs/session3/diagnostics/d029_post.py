"""F-029 post: classify the phase-0 collision at 0.4 per lump with l3.collision_summary, and add energy inside the group sphere over time."""
import json
import os
import numpy as np
from dcommon import HERE, write_json
from swirl.engine import l3

l3.OUT = "outputs/session3/diagnostics/l3"
name = "coll_primary_p0_v0.4"
j = [x for x in json.load(open(os.path.join(HERE, "jobs_s2.json"))) if x["name"] == name][0]
res = json.load(open(l3.result_path(name)))
cs = l3.collision_summary(res, j)
E0 = res["series"]["E_total"][0]
tl, X, E = l3._lumps_series(res)
dist = np.linalg.norm(X[:, 0] - X[:, 1], axis=1)
raw = res["lumps"]["raw"]["series"]
G = raw.get("E_group")
counts = res["lumps"]["counts"]
out = dict(name=name, settings=dict(start="primary", v_per_lump=0.4, closing_speed_galilean=0.8, closing_speed_relativistic=2 * 0.4 / (1 + 0.16),
           phase_label_deg=0.0, d=30.0, N=j["N"], L=j["L"], dx=0.55, dt=j["dt"], T=j["T"], t_contact_estimate=j["t_contact"], t_sponge_estimate=j["t_sponge"],
           source=res["params"]["init"]["wpl"]["source"], absorber="default sponge"), E0=E0,
           summary=cs, tracker_distance_min=float(dist.min()),
           tracker_x0_end=float(X[-1, 0, 0]), tracker_x1_end=float(X[-1, 1, 0]),
           counts=[dict(t=c["t_mid"], n=c["count_distinct"]) for c in counts],
           E_group_over_E0_at=({str(t): float(np.interp(t, tl, np.asarray(G))/E0) for t in (20, 30, 40, 50, 60, 70, 80, 100, 120, 150) if t <= tl[-1]} if G else None),
           E_plus_absorbed_rel_dev=res["summary"]["max_rel_dev_E_plus_absorbed"], absorbed_over_E0_end=res["summary"]["absorbed_end"] / E0,
           wall_s=res["wall_seconds"])
def egroup_rel(path, tc_free):
    r = json.load(open(path))
    t_, X_, E_ = l3._lumps_series(r)
    G_ = np.asarray(r["lumps"]["raw"]["series"]["E_group"]); E0_ = r["series"]["E_total"][0]
    return {str(d): float(np.interp(tc_free + d, t_, G_) / E0_) for d in (0, 10, 20, 30, 40, 60, 80, 100, 115) if tc_free + d <= t_[-1]}
W = os.path.join(HERE, "..", "wpL3", "results")
out["E_group_over_E0_since_free_contact_time"] = {
    "v0.4_this_run (free contact 29.14)": egroup_rel(l3.result_path(name), 29.14),
    "v0.2_existing (free contact 58.3)": egroup_rel(os.path.join(W, "coll_primary_p0_v0.2.json"), 58.3),
    "v0.02_existing (free contact 582.8)": egroup_rel(os.path.join(W, "coll_primary_p0_v0.02.json"), 582.8),
    "note": "group sphere radius 3 hw + 2 = 12.2 in all three runs (same run_config); times since the free estimate of contact"}
sel = tl >= 20
out["tracker_separation"] = {str(t): float(np.interp(t, tl, dist)) for t in range(0, 151, 10)}
out["tracker_separation_extrema"] = dict(first_min=[float(tl[np.argmin(dist[(tl > 25) & (tl < 50)]) + np.searchsorted(tl, 25)]), float(dist[(tl > 25) & (tl < 50)].min())],
                                         first_max=[float(tl[(tl > 50) & (tl < 85)][np.argmax(dist[(tl > 50) & (tl < 85)])]), float(dist[(tl > 50) & (tl < 85)].max())],
                                         second_min=[float(tl[(tl > 85) & (tl < 130)][np.argmin(dist[(tl > 85) & (tl < 130)])]), float(dist[(tl > 85) & (tl < 130)].min())])
out["lump_E_over_E0_by_tracker_ball"] = {str(t): float(np.interp(t, tl, E[:, 0]) / E0) for t in (30, 50, 70, 100, 150)}
out["note_remnant_E_group_over_E0"] = "summary.remnant_E_group_over_E0 averages over the last 300 time units, which here is the whole run (T = 150); use E_group_over_E0_at (value at t = 150 is the end state)"
write_json(os.path.join(HERE, "d029_collision_v0.4.json"), out)
print(json.dumps({k: v for k, v in out.items() if k != "counts"}, indent=1, default=float))
print(out["counts"])
