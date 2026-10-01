"""WP-B: 3D run1_primary (extended to T=5000) vs the radial curve of the same profile.
Run: NUMBA_NUM_THREADS=1 .venv/bin/python outputs/session2/wpB/compare_radial.py
Set WPB_SESSION1=1 to test on the Session 1 T=1000 result.
Input: outputs/session2/wpB/run1_primary_T5000.json (partial or final); if absent, the series in the working
checkpoint's meta; if absent, the Session 1 result (T=1000).
Radial curve: swirl.family.run_radial.evolve on the same table profile, same parameters as
outputs/step2/diagnostics/primary_lifetime.py (dr=0.05, dt=0.0125, R_sponge=100, R_box=200), r_c = 3 half-widths,
full series cached in radial_primary_T5000.json (record every 0.5).
Time matching: 3D samples are at integer t (every 1.0); radial records every 0.5. Each 3D time is paired with the
radial record at the identical time (|dt| < 1e-6, asserted). No interpolation.
frac = E(r <= 3 half-widths = 10.0321)/E0 with each run's own E0 (3D r_c index 2 of [5,7.5,10.03,15,20])."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "/Users/aaronramos/Desktop/SWIRL/"
sys.path.insert(0, R)
D = R + "outputs/session2/wpB/"
TS = [1000, 2000, 3000, 4000, 5000]

def radial():
    p = D + "radial_primary_T5000.json"
    if os.path.exists(p):
        return json.load(open(p))
    from swirl.family.run_radial import evolve, G4
    prof = json.load(open(R + "outputs/step2/profiles/profile_beta1.3333_w0.9091.json"))
    r, P = np.array(prof["r"]), np.array(prof["Phi"])
    rc = 3 * prof["half_width"]
    ts = evolve(lambda x: np.interp(x, r, P, right=0.0), G4, prof["g6"], [rc], T=5000.0)
    out = dict(params=dict(profile="profile_beta1.3333_w0.9091.json", g6=prof["g6"], r_c=rc, T=5000.0, dr=0.05, dt=0.0125,
                           R_sponge=100.0, R_box=200.0, sigma_max=1.0, record_every=0.5),
               E0=ts["E0"], t=ts["times"], frac=(np.array([e[0] for e in ts["E_inside"]]) / ts["E0"]).tolist(),
               E_total=ts["E_total"], omega_t=ts["omega_t"], omega=ts["omega"])
    json.dump(out, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
    return out

def load3d():
    p = D + "run1_primary_T5000.json"
    if os.environ.get("WPB_SESSION1"):
        return json.load(open(R + "outputs/step2/results/run1_primary.json")), "session1 result (T=1000)"
    if os.path.exists(p):
        return json.load(open(p)), "result:" + os.path.basename(p)
    ck = R + "checkpoints/session2/wpB/checkpoint.npz"
    if os.path.exists(ck):
        with np.load(ck, allow_pickle=False) as z:
            m = json.loads(str(z["meta"]))
        return dict(params=m["params"], series=m["series"], point=m["point"], t=m["step"] * m["params"]["dt"],
                    frequency=None, summary={}), "checkpoint meta"
    return json.load(open(R + "outputs/step2/results/run1_primary.json")), "session1 result"

def main():
    rad = radial(); res, src = load3d()
    s = res["series"]; p = res["params"]
    rc = rad["params"]["r_c"]; assert abs(p["r_c"][2] - rc) < 1e-9
    t3 = np.array(s["t"]); E0 = s["E_total"][0]
    f3 = np.array([e[2] for e in s["E_loc"]]) / E0
    tr = np.array(rad["t"]); fr = np.array(rad["frac"])
    idx = np.rint(t3 * 2).astype(int)
    assert np.all(np.abs(t3 * 2 - idx) < 1e-6) and idx.max() < tr.size
    assert np.all(np.abs(tr[idx] - t3) < 1e-6)
    fr_m = fr[idx]; d = f3 - fr_m
    w = int(np.argmax(np.abs(d)))
    mask = t3 >= 50   # reported also excluding the initial transient (t<50)
    wl = int(np.argmax(np.where(mask, np.abs(d), -1)))
    at = {}
    for T in TS:
        j = np.where(np.abs(t3 - T) < 1e-6)[0]
        at[str(T)] = None if not j.size else dict(frac_3D=float(f3[j[0]]), frac_radial=float(fr_m[j[0]]), diff=float(d[j[0]]))
    # omega at end: 3D from zero crossings of phi(center,t) (engine), radial median +-25 around same t
    tend = float(t3[-1])
    om3 = res["frequency"]["omega"] if res.get("frequency") else None
    om3t = res["frequency"]["t_mid"] if res.get("frequency") else None
    def med(tt, ow, T): 
        ow = np.array(ow); tt = np.array(tt); m = (tt >= T - 25) & (tt <= T + 25)
        return float(np.median(ow[m])) if m.any() else None
    omega = dict(T=tend, omega_3D_median_pm25=med(om3t, om3, tend) if om3 else None,
                 omega_3D_last=res["summary"].get("omega_last"),
                 omega_radial_median_pm25=med(rad["omega_t"], rad["omega"], tend))
    out = dict(source_3D=src, T_3D_reached=tend, n_matched_times=int(t3.size), matching="exact equal times, 3D every 1.0 vs radial every 0.5, no interpolation",
               tolerance_target=1e-3, max_abs_diff=float(abs(d[w])), t_of_max=float(t3[w]), diff_at_max_signed=float(d[w]),
               max_abs_diff_t_ge_50=float(abs(d[wl])), t_of_max_t_ge_50=float(t3[wl]),
               passes_tol_1e3=bool(abs(d[w]) < 1e-3), frac_at_T=at, omega=omega,
               E_conservation=dict(E0=E0, E_end=s["E_total"][-1], absorbed_end=s["absorbed"][-1],
                                   max_rel_dev_E_plus_absorbed=float(np.max(np.abs(np.array(s["E_plus_absorbed"]) - s["E_plus_absorbed"][0])) / abs(s["E_plus_absorbed"][0])),
                                   rel_dev_E_plus_absorbed_end=float((s["E_plus_absorbed"][-1] - s["E_plus_absorbed"][0]) / abs(s["E_plus_absorbed"][0]))),
               E0_3D=E0, E0_radial=rad["E0"],
               energy_moment_isotropy="not stored by the engine series (only I_kin, C_kin); needs the field from the checkpoint with outputs/step2/diagnostics/shape.py",
               diff_series=dict(t=t3[::50].tolist(), diff=d[::50].tolist()))
    tmp = D + os.environ.get("WPB_TAG","") + "compare_radial.json.tmp"
    json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, D + os.environ.get("WPB_TAG","") + "compare_radial.json")
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(tr[tr <= tend], fr[tr <= tend], lw=1, label="radial"); ax[0].plot(t3, f3, lw=1, ls="--", label="3D (N=192)")
    ax[0].set_ylabel("frac = E(r<3hw)/E0"); ax[0].legend()
    ax[1].plot(t3, d, lw=0.8); ax[1].axhline(1e-3, c="r", ls=":"); ax[1].axhline(-1e-3, c="r", ls=":")
    ax[1].set_ylabel("frac_3D - frac_radial"); ax[1].set_xlabel("t")
    fig.suptitle("run1_primary: 3D vs radial, T reached %.0f" % tend); fig.tight_layout()
    fig.savefig(D + os.environ.get("WPB_TAG","") + "compare_radial.png", dpi=130)
    print(json.dumps({k: v for k, v in out.items() if k != "diff_series"}, indent=1))

main()
