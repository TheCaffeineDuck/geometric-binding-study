import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL/outputs/audit"); sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL/outputs/audit/scratch")
from audit_common import *
from legacy_evolver_copy import SexticEvolver
# legacy Gaussian: phi = phi0*exp(-r^2/(2R^2)) (01_single_reference.py)
ev = SexticEvolver(N=N, L=L, m=M, g4=G4, g6=G6, dissipation_sigma=SIGMA)
r2 = ev.X**2 + ev.Y**2 + ev.Z**2
phi = PHI0 * np.exp(-r2 / (2 * R**2))
ev.set_initial_conditions(phi, np.zeros_like(phi))
E = ev.compute_energy(); dV = ev.dx**3
E_mass = 0.5*M**2*np.sum(phi**2)*dV
E_q4 = -(G4/24)*np.sum(phi**4)*dV
E_s6 = (G6/720)*np.sum(phi**6)*dV
# analytic
a2 = PHI0**2*(math.pi*R**2)**1.5
E_mass_an = 0.5*M**2*a2
E_grad_an = 0.5*PHI0**2*(math.pi*R**2)**1.5*3/(2*R**2)   # <k^2>=3/(2R^2)
E_q4_an = -(G4/24)*PHI0**4*(math.pi*R**2/2)**1.5
E_s6_an = (G6/720)*PHI0**6*(math.pi*R**2/3)**1.5
E_an = E_mass_an+E_grad_an+E_q4_an+E_s6_an
sb = load(LEG+"/outputs/phase1/set_B_baseline.json")
t = np.array(sb["t_series"]); mx = np.array(sb["max_phi_series"]); Es = np.array(sb["E_series"])
def at(tt): return float(mx[np.argmin(abs(t-tt))])
def win(t0,t1): s=(t>=t0)&(t<=t1); return float(mx[s].max())
free = lambda tt: (1+tt**2/(M**2*R**4))**-0.75
res = dict(
 E_total_grid=E, E_total_analytic=E_an, E_mass=E_mass, E_grad_analytic=E_grad_an, E_quartic=E_q4, E_sextic=E_s6,
 E_quartic_analytic=E_q4_an, E_sextic_analytic=E_s6_an,
 nonlinear_net_fraction_pct=100*(E_q4+E_s6)/E, quartic_abs_fraction_pct=100*abs(E_q4)/E, sextic_fraction_pct=100*E_s6/E,
 nonlinear_abs_sum_fraction_pct=100*(abs(E_q4)+abs(E_s6))/E,
 omega_formula=math.sqrt(M**2+3/(2*R**2)),
 omega_eff_calibrated_legacy=load(LEG+"/outputs/continuous_phase/continuous_phase_summary.json")["parameters"]["omega_eff"],
 omega_eff_hardcoded_phase12=1.113,
 baseline_E_t0=float(Es[0]),
 t_spacing=float(t[1]-t[0]), n_samples=len(t),
 maxphi_t0=at(0), maxphi_t50=at(50), maxphi_t50_ratio=at(50)/PHI0, maxphi_t50_window45_55_ratio=win(45,55)/PHI0,
 maxphi_t500=at(500), maxphi_t500_ratio=at(500)/PHI0, maxphi_t500_window495_505_ratio=win(495,505)/PHI0,
 maxphi_final_t=float(t[-1]), amplitude_retention_final=sb["amplitude_retention_final"],
 free_dispersal_t50_pct=100*free(50), free_dispersal_t500_pct=100*free(500),
 maxphi_min_over_run_ratio=float(mx.min())/PHI0, t_of_min=float(t[mx.argmin()]),
)
save_part("item01", res); print(json.dumps(res, indent=1))

# --- diagnostics for the nonlinear-share disagreement: other Gaussian conventions / quantities
diag = {}
for label, w in [("exp(-r^2/(2R^2)) [legacy]", 2*R**2), ("exp(-r^2/R^2)", R**2), ("exp(-r^2/(4R^2))", 4*R**2)]:
    p = PHI0*np.exp(-r2/w); ev.set_initial_conditions(p, np.zeros_like(p)); Et = ev.compute_energy()
    q = -(G4/24)*np.sum(p**4)*dV; s = (G6/720)*np.sum(p**6)*dV
    diag[label] = dict(E=Et, net_pct=100*(q+s)/Et, quartic_abs_pct=100*abs(q)/Et)
res["convention_diagnostic"] = diag
tt = t[t>=400]; mm = mx[t>=400]
res["maxphi_max_ratio_t400_500"] = float(mm.max())/PHI0
res["maxphi_max_ratio_t_after_60"] = float(mx[t>=60].max())/PHI0
res["maxphi_t_of_max_after_60"] = float(t[t>=60][mx[t>=60].argmax()])
save_part("item01", res); print(json.dumps(diag, indent=1), res["maxphi_max_ratio_t400_500"], res["maxphi_max_ratio_t_after_60"], res["maxphi_t_of_max_after_60"])
