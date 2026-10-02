"""A10 comparison: 3D beta=5 run vs radial (dense rerun to T=1000, records every 0.5). 3D samples at integer t;
matched at the identical radial record (no interpolation). frac = E(r<=3hw=19.8684)/E(t=0) for each run."""
import json, numpy as np
R = "/Users/aaronramos/Desktop/SWIRL/outputs/session2/wpA/"; D = R + "confirm3d_beta5/"
st = json.load(open(R + "long_beta5.0000_w0.975598_T300k.json"))
rad = json.load(open(D + "radial_beta5_T1000_dense.json"))
rt = np.array(rad["t"]); rf = np.array(rad["frac"]); ri = {round(x, 6): i for i, x in enumerate(rt)}
def loc(res):
    s = res["series"]; return np.array(s["t"]), np.array(s["E_loc"])[:, 2] / s["E_total"][0]
out = {}
p = json.load(open(D + "results/confirm3d_b5_primary.json")); s = p["series"]; t, f = loc(p)
d = np.array([f[i] - rf[ri[round(x, 6)]] for i, x in enumerate(t)]); k = int(np.argmax(abs(d)))
fr = lambda T: float(rf[ri[float(T)]])
ti = {float(x): i for i, x in enumerate(t)}
omr = np.array(rad["omega"]); omt = np.array(rad["omega_t"])
out["primary"] = dict(completed=p["completed"], T=float(t[-1]), n_matched=len(t), max_abs_diff=float(abs(d[k])), t_of_max=float(t[k]),
    diff_signed_at_max=float(d[k]), diff_at_250_500_750_1000=[float(d[ti[float(T)]]) for T in (250, 500, 750, 1000)],
    frac_3D={str(T): float(f[ti[float(T)]]) for T in (250, 500, 750, 1000)}, frac_radial={str(T): fr(T) for T in (250, 500, 750, 1000)},
    frac_radial_stored_T300k_at_1000=st["frac_at"]["1000"],
    omega_end_3D=p["summary"]["omega_last"], omega_3D_last5_windows=float(np.mean(p["frequency"]["omega"][-5:])),
    omega_radial_last_window_T1000=float(omr[omt <= 1000][-1]), omega_radial_stored_1000=st["omega_at"]["1000"],
    omega_radial_settled=st["regime"]["omega_over_m_settled"],
    E0_3D=s["E_total"][0], E0_radial=rad["E0"], E0_radial_stored=st["E0"], E0_profile_quadrature=json.load(open(D + "t0_quadrature.json"))["E0_profile_quadrature_t0"],
    E_end=p["summary"]["E_end"], absorbed_end=p["summary"]["absorbed_end"], E_plus_absorbed_max_rel_dev=p["summary"]["max_rel_dev_E_plus_absorbed"],
    omega_over_m_start=st["params"]["omega_profile"], nonlinear_share_t0_3D_reported=s["nonlinear_share"][0],
    nonlinear_share_t0_radial_abs=st["regime"]["nonlinear_share_t0_absolute"], nonlinear_share_t0_radial_signed=st["regime"]["nonlinear_share_t0_signed"],
    steps_per_second=p.get("steps_per_second_this_call"), wall_seconds=p.get("wall_seconds"))
q = out["primary"]
q["A10_tracks_1e-3"] = bool(q["max_abs_diff"] <= 1e-3)
q["A10_frac1000_0.9986_pm0.001"] = bool(abs(q["frac_3D"]["1000"] - 0.9986) <= 1e-3)
q["A10_omega_within_0.0005"] = bool(abs(q["omega_end_3D"] - q["omega_radial_last_window_T1000"]) <= 5e-4)
w = json.load(open(D + "results/confirm3d_b5_twin.json")); tw, fw = loc(w); wi = {float(x): i for i, x in enumerate(tw)}
out["twin"] = dict(completed=w["completed"], T=float(tw[-1]), frac_T100=float(fw[wi[100.0]]), frac_T300=float(fw[wi[300.0]]),
    radial_twin_T100=st["linear_twin"]["frac_T100"], radial_twin_T300=st["linear_twin"]["frac_T300"], E0=w["series"]["E_total"][0], E0_radial_twin=st["linear_twin"]["E0"])
out["twin"]["diff_T100"] = out["twin"]["frac_T100"] - out["twin"]["radial_twin_T100"]; out["twin"]["diff_T300"] = out["twin"]["frac_T300"] - out["twin"]["radial_twin_T300"]
out["twin"]["A10_twin_within_0.05"] = bool(abs(out["twin"]["diff_T100"]) <= .05 and abs(out["twin"]["diff_T300"]) <= .05)
json.dump(out, open(D + "a10_comparison.json", "w"), indent=1); print(json.dumps(out, indent=1))
