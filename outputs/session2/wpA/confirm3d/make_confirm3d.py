"""A9 3D confirmation: profile + configs. Same recipe as outputs/step2/make_profiles.py and make_configs.py.
Run: .venv/bin/python outputs/session2/wpA/confirm3d/make_confirm3d.py
"""
import json, os, sys
import numpy as np
ROOT = "/Users/aaronramos/Desktop/SWIRL"
sys.path.insert(0, ROOT)
from swirl.family import profile as P
D = ROOT + "/outputs/session2/wpA/confirm3d"
src = json.load(open(ROOT + "/outputs/session2/wpA/long_beta1.3333_w0.906505.json"))
omega = src["params"]["omega_profile"]; beta = src["params"]["beta"]
G4, M, DR = 0.30, 1.0, 0.01
def trapz(y, x): return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))
res = P.solve_profile(omega, 1.0, 1.0, beta, tol=1e-10)
r = np.arange(0.0, 34.0 + 1e-9, DR)
Phi = P.profile_curve(res, r) / np.sqrt(G4)
cut = np.nonzero(np.abs(Phi) < 1e-9 * Phi[0])[0]
n = (cut[0] + 1) if cut.size else r.size
r, Phi = r[:n], Phi[:n]
g6 = beta * G4 ** 2 / M ** 2
dPhi = np.gradient(Phi, r, edge_order=2)
I2 = 4*np.pi*trapz(Phi**2*r**2, r); Id = 4*np.pi*trapz(dPhi**2*r**2, r)
I4 = 4*np.pi*trapz(Phi**4*r**2, r); I6 = 4*np.pi*trapz(Phi**6*r**2, r)
N_tab = 0.5*omega*I2
E_tab = 0.25*(omega**2+M**2)*I2 + 0.25*Id - G4/64*I4 + g6/2304*I6
hw = res["half_width"]/M
prof = dict(beta=beta, omega=omega, m=M, g4=G4, g6=g6, dr=DR, r_max=float(r[-1]), half_width=hw, kappa=res["kappa"],
            N=res["N"]/G4, E=res["E"]*M/G4, N_table_quadrature=N_tab, E_table_quadrature=E_tab,
            note="same recipe as outputs/step2/make_profiles.py; omega = omega_profile of long_beta1.3333_w0.906505.json",
            r=r.tolist(), Phi=Phi.tolist())
pp = D + "/profile_beta1.3333_w0.906505.json"
json.dump(prof, open(pp + ".tmp", "w")); os.replace(pp + ".tmp", pp)
print("omega", omega, "hw", hw, "3hw", 3*hw, "E_table", E_tab, "E_family", prof["E"], "N_table", N_tab, "N_family", prof["N"])
def cfg(linear, t_end):
    ent = {"r": r.tolist(), "Phi": Phi.tolist(), "center": [0.0, 0.0, 0.0], "phase": 0.0, "omega": 0.0}
    return {"N": 128, "L": 70.0, "m": 1.0, "g4": 0.0 if linear else G4, "g6": 0.0 if linear else g6, "dt": 0.04, "t_end": t_end,
            "init": {"radial": [ent], "gaussians": []}, "absorber": True, "kfilter": None, "center": [0.0, 0.0, 0.0],
            "r_c": [5.0, 7.5, 3*hw, 15.0, 20.0], "sample_every": 1.0, "point_every": 1, "avg_window": 20.0,
            "checkpoint_every": 50.0, "threads": 2,
            "_note": "A9 3D confirmation, beta=4/3 omega=%.12g, N=128 L=70 dt=0.04, E_loc shell at 3hw=%.6f" % (omega, 3*hw)}
for name, lin, T in [("confirm3d_primary", False, 3000.0), ("confirm3d_twin", True, 300.0)]:
    p = "%s/%s.json" % (D, name); json.dump(cfg(lin, T), open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
