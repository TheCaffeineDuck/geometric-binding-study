"""Collect shape_run5.json and shape_run5.png from the run5 driver outputs. Run: .venv/bin/python outputs/step2/diagnostics/shape_run5_collect.py"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "/Users/aaronramos/Desktop/SWIRL/outputs/step2/"
names = {"run5": "run5_primary_generic", "run5r": "run5r_primary_offcentre", "run1": "run1_primary"}
out = {"parameters": {"spec": "docs/WP2_PARAMETER_PICK.md addendum run 5", "shape_script": "outputs/step2/diagnostics/shape.py",
                      "R_sphere": 3 * json.load(open(R + "profiles/profile_beta1.3333_w0.9091.json"))["half_width"],
                      "frac_definition": "E_loc(r < R_sphere about the diagnostic centre, t) / E_total(t=0) from the run result series",
                      "snapshot_note": "shape quantities are single snapshots at the checkpoint time (not time-averaged); phi_centre is the phase indicator",
                      "configs": {k: json.load(open(R + "configs/%s.json" % v)) for k, v in names.items() if k != "run1"}},
       "validation": json.load(open(R + "diagnostics/shape_validation.json"))}
for k in ("run5", "run5r"):
    for r in out["parameters"]["configs"][k]["init"]["radial"]:
        r.pop("r"); r.pop("Phi")  # tables are in the profile file
fr = {}
for k, v in names.items():
    res = json.load(open(R + "results/%s.json" % v))
    p = res["params"]; rc = p["r_c"]; i = int(np.argmin([abs(x - out["parameters"]["R_sphere"]) for x in rc]))
    t = np.array(res["series"]["t"]); E0 = res["series"]["E_total"][0]
    f = np.array(res["series"]["E_loc"])[:, i] / E0
    fr[k] = {"t": t, "frac": f, "rc": rc[i]}
    out[k] = {"completed": res["completed"], "E0": E0, "max_rel_dev_E_plus_absorbed": res["summary"]["max_rel_dev_E_plus_absorbed"],
              "frac_vs_T": {str(T): float(f[np.argmin(abs(t - T))]) for T in range(100, 1001, 100)}, "frac_rc": rc[i], "N": p["N"], "L": p["L"]}
    if k != "run1":
        rows = json.load(open(R + "diagnostics/%s_shape_partial.json" % v))
        out[k]["shape"] = [{"t": r["t"], "centroid": r["centroid"], "centroid_disp_from_start": float(np.linalg.norm(np.array(r["centroid"]) - np.array(p["center"]))),
                            "a2": r["a2"], "a3": r["a3"], "E_sphere_over_total": r["E_sphere_over_total"], "phi_centre": r["phi_centre"]} for r in rows]
json.dump(out, open(R + "diagnostics/shape_run5.json", "w"), indent=1)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for k, c in (("run5", "C3"), ("run5r", "C0")):
    s = out[k]["shape"]; t = [r["t"] for r in s]
    ax[0].semilogy(t, [r["a2"] for r in s], "o-", color=c, label=k); ax[1].semilogy(t, [r["a3"] for r in s], "o-", color=c, label=k)
ax[0].set_title("a2 (snapshots)"); ax[1].set_title("a3 (snapshots)")
for k, c in (("run5", "C3"), ("run5r", "C0"), ("run1", "k")):
    ax[2].plot(fr[k]["t"], fr[k]["frac"], color=c, label=k, lw=1)
ax[2].set_title("frac = E_loc(r<3hw)/E0"); 
for a in ax: a.set_xlabel("t"); a.legend()
fig.tight_layout(); fig.savefig(R + "diagnostics/shape_run5.png", dpi=110)
for k in ("run5", "run5r", "run1"): print(k, out[k]["frac_vs_T"], out[k]["max_rel_dev_E_plus_absorbed"], out[k]["completed"])
for k in ("run5", "run5r"):
    for r in out[k]["shape"]:
        if r["t"] in (0, 100, 300, 500, 700, 1000): print(k, r["t"], "a2=%.3e a3=%.3e disp=%.3e Es/Et=%.5f phic=%.3f cen=%s" % (r["a2"], r["a3"], r["centroid_disp_from_start"], r["E_sphere_over_total"], r["phi_centre"], np.round(r["centroid"], 4)))
