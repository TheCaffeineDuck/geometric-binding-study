import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL/outputs/audit"); sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL/outputs/audit/scratch")
from audit_common import *
from legacy_evolver_copy import SexticEvolver
from numpy.fft import fftn
sb=load(LEG+"/outputs/phase1/set_B_baseline.json"); t=np.array(sb["t_series"]); E=np.array(sb["E_series"])
res=dict(E_t0=float(E[0]),
  E_series_first_values=[(float(t[i]),float(E[i])) for i in range(0,25)],
  E_at_t10=float(E[np.argmin(abs(t-10))]), E_at_t20=float(E[np.argmin(abs(t-20))]), E_final=float(E[-1]),
  E_t0_to_t10_step=float(E[np.argmin(abs(t-10))]-E[0]),
  final_rel_drift_pct=100*float((E[-1]-E[0])/E[0]),
  E_series_monotone_increments_after_t10=bool(np.all(np.diff(E[t>=10])>=-1e-6)),
  E_range_after_t10=[float(E[t>=10].min()),float(E[t>=10].max())])
# time of first reaching 13.46374 level
res["t_first_E_ge_13.46374"]=float(t[np.argmax(E>=13.46374)])
# scratch runs: legacy evolver copy, T=20, with sigma=0.01 and sigma=0
out={}
for sg in (0.01,0.0):
    ev=SexticEvolver(N=N,L=L,m=M,g4=G4,g6=G6,dissipation_sigma=sg)
    p=PHI0*np.exp(-(ev.X**2+ev.Y**2+ev.Z**2)/(2*R**2)); ev.set_initial_conditions(p,np.zeros_like(p))
    Es=[];Ec=[];ts=[]
    for step in range(401):
        if step%20==0:
            ph=fftn(ev.phi); extra=0.5*sg*ev.dx**4*np.sum(ev.K4*np.abs(ph)**2)*ev.dx**3/ev.N**3
            ts.append(ev.t);Es.append(ev.compute_energy());Ec.append(ev.compute_energy()+extra)
        ev.step_rk4(DT)
    out["sigma_%g"%sg]=dict(t=ts,E_legacy=Es,E_with_stiffness_term=Ec,
        step_legacy_t0_to_t10=Es[10]-Es[0],step_legacy_t0_to_t20=Es[-1]-Es[0],
        spread_with_term_t0_to_t20=max(Ec)-min(Ec))
res["scratch_runs_T20"]=out
save_part("item07",res)
for k,v in out.items(): print(k,v["E_legacy"][0],v["E_legacy"][10],v["E_legacy"][-1],v["spread_with_term_t0_to_t20"])
print({k:v for k,v in res.items() if k!="scratch_runs_T20" and k!="E_series_first_values"})
