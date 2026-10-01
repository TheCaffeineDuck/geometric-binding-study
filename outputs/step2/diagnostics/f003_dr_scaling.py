"""F-003 cause test: physical-density energy variation vs dr (absorber off, T=100, Set B profile).
Run: NUMBA_NUM_THREADS=1 .venv/bin/python outputs/step2/diagnostics/f003_dr_scaling.py"""
import json, os, sys, time
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from swirl.family.run_radial import evolve, G4, load_features
from swirl.family.family import BETA_SET_B
from swirl.family.profile import solve_profile, profile_curve

beta = BETA_SET_B; g6 = beta * G4**2
f = load_features(beta)
p = solve_profile(f["omega_dNdw_sign_change"] - 0.04, 1.0, 1.0, beta)
phi = lambda r: profile_curve(p, r) / np.sqrt(G4)
rcs = [3 * p["half_width"]]
rows = []
for dr in (0.05, 0.025, 0.0125):
    dt = 0.0125 * dr / 0.05
    t0 = time.time()
    ts = evolve(phi, G4, g6, rcs, T=100.0, dr=dr, dt=dt, sigma_max=0.0)
    Eh = np.array(ts["E_discrete"]); Ep = np.array(ts["E_total"])
    rows.append(dict(dr=dr, dt=dt, max_rel_var_physical=float(np.max(np.abs(Ep - Ep[0])) / Ep[0]),
                     max_rel_var_physical_minmax=float((Ep.max() - Ep.min()) / Ep[0]),
                     max_rel_drift_discrete=float(np.max(np.abs(Eh - Eh[0])) / Eh[0]), wall_s=time.time() - t0))
    print(rows[-1], flush=True)
for a, b in zip(rows[:-1], rows[1:]):
    b["ratio_prev_over_this"] = a["max_rel_var_physical"] / b["max_rel_var_physical"]
res = dict(description="F-003 test: max rel variation of physical-density energy (np.gradient + trapezoid) vs dr",
           params=dict(profile_omega=p["omega"], beta=beta, m=1.0, g4=G4, g6=g6, T=100.0, absorber="off (sigma_max=0)",
                       R_box=200.0, dt_over_dr=0.25, rec_dt=0.5), rows=rows)
pth = os.path.join(ROOT, "outputs/step2/diagnostics/f003_dr_scaling.json")
json.dump(res, open(pth + ".tmp", "w"), indent=1); os.replace(pth + ".tmp", pth)
