"""A9 comparison: 3D confirm run vs the radial curve of the same start. Writes confirm3d/a9_comparison.json.
Matching: 3D samples are at integer t (sample_every 1.0); the stored radial series is at t = 0, 5, 10, ...;
the two are compared only at t that appear exactly in both lists (multiples of 5), no interpolation.
frac_3D = E_loc(r <= 3hw)/E_total(t=0), radial frac = E(r < 3hw)/E0 (stored series 'frac'), both 3hw = 10.17909.
"""
import json, os, numpy as np
R = "/Users/aaronramos/Desktop/SWIRL/outputs/session2/wpA/"
D = R + "confirm3d/"
rad = json.load(open(R + "long_beta1.3333_w0.906505.json"))
out = {"match": "exact common sample times (multiples of 5), no interpolation"}
rt = np.array(rad["series"]["t"]); rf = np.array(rad["series"]["frac"])
rom = rad["omega_series"]
def loc_frac(res):
    s = res["series"]; t = np.array(s["t"]); E = np.array(s["E_loc"])[:, 2]
    return t, E / s["E_total"][0]
if os.path.exists(D + "results/confirm3d_primary.json"):
    p = json.load(open(D + "results/confirm3d_primary.json")); s = p["series"]
    t, f = loc_frac(p)
    Tmax = t[-1]
    common = [x for x in t if x in set(rt.tolist()) and x <= Tmax]
    ti = {x: i for i, x in enumerate(t.tolist())}; ri = {x: i for i, x in enumerate(rt.tolist())}
    diffs = np.array([f[ti[x]] - rf[ri[x]] for x in common])
    k = int(np.argmax(np.abs(diffs)))
    out["primary"] = dict(T_reached=Tmax, completed=p["completed"], n_matched=len(common), max_abs_diff=float(abs(diffs[k])),
        t_of_max=common[k], diff_at_t0=float(diffs[0]), E0_3D=s["E_total"][0], E0_profile_quadrature_t0=2955.9297678959438, E0_radial=rad["E0"],
        frac_3D={str(T): float(f[ti[float(T)]]) for T in (100, 300, 1000, 2000, 3000) if float(T) in ti},
        frac_radial={str(T): float(rf[ri[float(T)]]) for T in (100, 300, 1000, 2000, 3000) if float(T) in ri},
        omega_end_3D=p["summary"]["omega_last"], omega_3D_last5_windows=float(np.mean(p["frequency"]["omega"][-5:])),
        omega_radial_last_window_leq_T=float(np.array(rom["omega"])[np.array(rom["t"]) <= Tmax][-1]),
        omega_radial_settled=rad["regime"]["omega_over_m_settled"],
        E_end=p["summary"]["E_end"], absorbed_end=p["summary"]["absorbed_end"], E_plus_absorbed_max_rel_dev=p["summary"]["max_rel_dev_E_plus_absorbed"],
        nonlinear_share_t0=s["nonlinear_share"][0], nonlinear_share_mean=float(np.mean(s["nonlinear_share"])),
        nonlinear_share_radial_t0_abs=rad["regime"]["nonlinear_share_t0_absolute"], nonlinear_share_radial_t0_signed=rad["regime"]["nonlinear_share_t0_signed"],
        steps_per_second=p.get("steps_per_second_this_call"), wall_seconds=p.get("wall_seconds"))
    q = out["primary"]
    q["A9_tracks_1e-3"] = q["max_abs_diff"] <= 1e-3
    q["A9_omega_within_0.0005_of_0.9069"] = abs(q["omega_end_3D"] - 0.9069) <= 5e-4
if os.path.exists(D + "results/confirm3d_twin.json"):
    w = json.load(open(D + "results/confirm3d_twin.json")); t, f = loc_frac(w); ti = {x: i for i, x in enumerate(t.tolist())}
    out["twin"] = dict(completed=w["completed"], T=t[-1], frac_T100=float(f[ti[100.0]]), frac_T300=float(f[ti[300.0]]) if 300.0 in ti else None,
                       E0=w["series"]["E_total"][0])
    out["twin"]["A9_twin_below_0.10"] = bool(f[ti[100.0]] < 0.10)
json.dump(out, open(D + "a9_comparison.json", "w"), indent=1)
print(json.dumps(out, indent=1))
