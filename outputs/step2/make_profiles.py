"""WP3 task A: radial tables of Phi(r) in Set-B-like units from the WP1A profile solver.

The solver module is loaded by file path from the WP1A worktree (not on main yet).
Solves in rescaled units (m=1, g4=1, g6=beta) and converts Phi_setB = Phi/sqrt(g4).
Run: .venv/bin/python outputs/step2/make_profiles.py
"""
import importlib.util, json, os
import numpy as np

WT = "/Users/aaronramos/Desktop/SWIRL/.claude/worktrees/agent-a63e9fc3746934e76"
spec = importlib.util.spec_from_file_location("wp1a_profile", WT + "/swirl/family/profile.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

G4, M = 0.30, 1.0
PICKS = [("primary", 4.0 / 3.0, 0.909149280617113, "0.9091"),
         ("secondary", 0.85, 0.7793064609301195, "0.7793")]
OUT = "/Users/aaronramos/Desktop/SWIRL/outputs/step2/profiles"
DR = 0.01


def trapz(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


for name, beta, omega, wtag in PICKS:
    res = P.solve_profile(omega, 1.0, 1.0, beta, tol=1e-10)
    kappa = res["kappa"]
    r = np.arange(0.0, 34.0 + 1e-9, DR)
    Pr = P.profile_curve(res, r)
    fac = 1.0 / np.sqrt(G4)
    Phi = Pr * fac
    cut = np.nonzero(np.abs(Phi) < 1e-9 * Phi[0])[0]
    n = (cut[0] + 1) if cut.size else r.size
    r, Phi = r[:n], Phi[:n]
    g6 = beta * G4 ** 2 / M ** 2
    # quadrature of the table (Set-B units)
    dPhi = np.gradient(Phi, r, edge_order=2)
    I2 = 4 * np.pi * trapz(Phi ** 2 * r ** 2, r)
    Id = 4 * np.pi * trapz(dPhi ** 2 * r ** 2, r)
    I4 = 4 * np.pi * trapz(Phi ** 4 * r ** 2, r)
    I6 = 4 * np.pi * trapz(Phi ** 6 * r ** 2, r)
    N_tab = 0.5 * omega * I2
    E_tab = 0.25 * (omega ** 2 + M ** 2) * I2 + 0.25 * Id - G4 / 64 * I4 + g6 / 2304 * I6
    N_fam, E_fam = res["N"] / G4, res["E"] * M / G4
    hw = res["half_width"] / M
    out = dict(name=name, beta=beta, omega=omega, omega_tag=wtag, m=M, g4=G4, g6=g6,
               units="Set-B-like: m=1, g4=0.30, g6=beta*g4^2", solver="WP1A swirl.family.profile.solve_profile, rescaled (g4=1,g6=beta, tol 1e-10), Phi_setB=Phi_resc/sqrt(g4)",
               dr=DR, r_max=float(r[-1]), n_points=int(r.size),
               Phi0=float(Phi[0]), Phi0_rescaled=res["P0"], half_width=hw,
               N=N_fam, E=E_fam, E_over_mN=res["E_over_mN"], A=res["A"] / np.sqrt(G4), kappa=kappa,
               N_table_quadrature=N_tab, E_table_quadrature=E_tab,
               rel_diff_N=(N_tab - N_fam) / N_fam, rel_diff_E=(E_tab - E_fam) / E_fam,
               Phi_rmax_over_Phi0=float(abs(Phi[-1]) / Phi[0]),
               r=r.tolist(), Phi=Phi.tolist())
    path = os.path.join(OUT, "profile_beta%.4f_w%s.json" % (beta, wtag))
    json.dump(out, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
    print(name, path, {k: out[k] for k in ("beta", "omega", "Phi0", "Phi0_rescaled", "half_width", "N", "E", "E_over_mN",
          "N_table_quadrature", "E_table_quadrature", "rel_diff_N", "rel_diff_E", "r_max", "A", "kappa", "Phi_rmax_over_Phi0")})
