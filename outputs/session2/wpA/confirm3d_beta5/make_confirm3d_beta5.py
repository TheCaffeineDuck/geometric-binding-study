"""A10 3D confirmation at beta = 5: profile + configs. Same recipe as confirm3d/make_confirm3d.py (A9).
Run: .venv/bin/python outputs/session2/wpA/confirm3d_beta5/make_confirm3d_beta5.py
Grid: dx = 0.55, N = 216 (2^3 3^3), L = N dx = 118.8. Box chosen from the engine guard
(profile at the box face <= 1e-6 of the peak, swirl.engine.initial.TAIL_TOL).
"""
import json, os, sys
import numpy as np
ROOT = "/Users/aaronramos/Desktop/SWIRL"
sys.path.insert(0, ROOT)
from swirl.family import profile as P
from swirl.engine.initial import radial_checks
D = ROOT + "/outputs/session2/wpA/confirm3d_beta5"
src = json.load(open(ROOT + "/outputs/session2/wpA/long_beta5.0000_w0.975598_T300k.json"))
omega = src["params"]["omega_profile"]; beta = src["params"]["beta"]
G4, M, DR = 0.30, 1.0, 0.01
DX, NGRID = 0.55, 216
def trapz(y, x): return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))
res = P.solve_profile(omega, 1.0, 1.0, beta, tol=1e-10)
r = np.arange(0.0, 100.0 + 1e-9, DR)
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
L = NGRID * DX
r_in = L/2 - 12.0
at = lambda x: float(abs(np.interp(x, r, Phi)) / Phi[0])
guards = radial_checks(dict(r=r, Phi=Phi), L)
info = dict(N=NGRID, L=L, dx=DX, r_max_table=float(r[-1]), sponge_inner=r_in, tail_at_sponge_inner=at(r_in), tail_at_face=at(L/2),
            points_per_half_width=hw/DX, three_hw=3*hw, guards=guards, E_table_quadrature=E_tab, E_family=res["E"]*M/G4,
            N_table_quadrature=N_tab, N_family=res["N"]/G4)
print(json.dumps(info, indent=1))
prof = dict(beta=beta, omega=omega, m=M, g4=G4, g6=g6, dr=DR, r_max=float(r[-1]), half_width=hw, kappa=res["kappa"],
            N=res["N"]/G4, E=res["E"]*M/G4, N_table_quadrature=N_tab, E_table_quadrature=E_tab,
            note="same recipe as confirm3d/make_confirm3d.py; omega = omega_profile of long_beta5.0000_w0.975598_T300k.json",
            r=r.tolist(), Phi=Phi.tolist())
pp = D + "/profile_beta5.0000_w0.975598.json"
json.dump(prof, open(pp + ".tmp", "w")); os.replace(pp + ".tmp", pp)
json.dump(info, open(D + "/grid_info.json", "w"), indent=1)
def cfg(linear, t_end):
    ent = {"r": r.tolist(), "Phi": Phi.tolist(), "center": [0.0, 0.0, 0.0], "phase": 0.0, "omega": 0.0}
    return {"N": NGRID, "L": L, "m": 1.0, "g4": 0.0 if linear else G4, "g6": 0.0 if linear else g6, "dt": 0.04, "t_end": t_end,
            "init": {"radial": [ent], "gaussians": []}, "absorber": True, "kfilter": None, "center": [0.0, 0.0, 0.0],
            "r_c": [5.0, 7.5, 3*hw, 25.0, 35.0], "sample_every": 1.0, "point_every": 1, "avg_window": 20.0,
            "checkpoint_every": 50.0, "threads": 2,
            "_note": "A10 3D confirmation, beta=5 omega=%.12g, N=%d L=%.1f dx=0.55 dt=0.04, E_loc shell at 3hw=%.6f" % (omega, NGRID, L, 3*hw)}
if __name__ == "__main__":
    T = float(sys.argv[1]) if len(sys.argv) > 1 else 1000.0
    for name, lin, TT in [("confirm3d_b5_primary", False, T), ("confirm3d_b5_twin", True, 300.0)]:
        p = "%s/%s.json" % (D, name); json.dump(cfg(lin, TT), open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
