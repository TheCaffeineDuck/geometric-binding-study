"""C: run4 vs run1 aligned; D: 3D vs radial twins early times. Run: .venv/bin/python <this>"""
import json, os
import numpy as np
from scipy.interpolate import CubicSpline
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
D = os.path.join(ROOT, "outputs/step2")
L = lambda f: json.load(open(os.path.join(D, f)))
J = lambda o, p: (json.dump(o, open(p + ".tmp", "w"), indent=1), os.replace(p + ".tmp", p))

r1, r4 = L("results/run1_primary.json"), L("results/run4_primary_res.json")
def fr(d):
    return np.array(d["series"]["t"]), np.array(d["series"]["E_loc"]) / d["summary"]["E0"]
t1, f1 = fr(r1); t4, f4 = fr(r4)
rc1, rc4 = sorted(r1["params"]["r_c"]), sorted(r4["params"]["r_c"])
g = np.arange(1, 300 + 1e-9, 0.25)
cols = {}
for j, rc in enumerate(rc4):
    a = CubicSpline(t1, f1[:, j])(g); b = CubicSpline(t4, f4[:, j])(g)
    dd = np.abs(b - a)
    cols["r_c=%.4g" % rc] = dict(max_abs_diff=float(dd.max()), t_at_max=float(g[dd.argmax()]))
tm1, w1 = np.array(r1["frequency"]["t_mid"]), np.array(r1["frequency"]["omega"])
tm4, w4 = np.array(r4["frequency"]["t_mid"]), np.array(r4["frequency"]["omega"])
sel = (tm1 >= 50) & (tm1 <= 300) & (tm1 <= tm4.max())
dw = np.abs(CubicSpline(tm4, w4)(tm1[sel]) - w1[sel])
C = dict(description="run 4 (dt 0.9879-sampled) vs run 1 (dt 1.0), cubic-spline aligned, t=1..300 step 0.25",
         r_c_run1=rc1, r_c_run4=rc4, frac_columns=cols,
         omega=dict(n=int(sel.sum()), t_range=[float(tm1[sel].min()), float(tm1[sel].max())], max_abs_diff=float(dw.max()),
                    t_at_max=float(tm1[sel][dw.argmax()]), median_abs_diff=float(np.median(dw))),
         sample_dt=dict(run1=float(np.median(np.diff(t1))), run4=float(np.median(np.diff(t4)))),
         note="omega_4 interpolated onto run 1 t_mid in [50,300]; frac = E_loc/summary.E0 per run")
J(C, os.path.join(D, "diagnostics/run4_vs_run1.json"))
print(json.dumps(C, indent=1))

rad = L("diagnostics/radial_T1000.json")
T = [25, 50, 75, 100, 150, 250, 300, 500, 700, 1000]
out = dict(description="frac = E(r<3hw)/E0, 3D twin (nearest sample, r_c index 2) vs radial twin (nearest sample)", t=T, picks={})
for pick, f3, rk in (("primary", "run2_primary_twin.json", "1"), ("secondary", "run2b_secondary_twin.json", "1b")):
    d = L("results/" + f3); t3, F3 = fr(d); rc = sorted(d["params"]["r_c"])
    rs = rad["radial_series"][rk]; tl = np.array(rs["lin_times"]); fl = np.array([e[0] for e in rs["lin_E_inside"]]) / rad["radial_summary"][rk]["E0_lin"]
    rows = []
    for t in T:
        i = int(np.argmin(np.abs(t3 - t))); k = int(np.argmin(np.abs(tl - t)))
        rows.append(dict(t=t, frac_3D_twin=float(F3[i, 2]), t3_sample=float(t3[i]), frac_radial_twin=float(fl[k]), t_rad_sample=float(tl[k])))
    out["picks"][pick] = dict(r_c_3D=rc[2], r_c_radial=3 * rad["radial_summary"][rk]["half_width"], rows=rows)
J(out, os.path.join(D, "diagnostics/twins_3d_vs_radial.json"))
for p, v in out["picks"].items():
    print(p)
    for r in v["rows"]: print(r)
