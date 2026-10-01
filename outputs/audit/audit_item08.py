import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL/outputs/audit")
from audit_common import *
b0=[];bT=[];chg=[];S0=[];ST=[];ST_t0=[]
for sd in range(50):
    d=load(LEG+"/outputs/phase12/seed_%02d.json"%sd)
    e=np.array(d["binding_energy"]); b0.append(e[0]); bT.append(e[-1]); S0.append(d["order_parameter"][0]); ST.append(d["order_parameter"][-1])
b0=np.array(b0);bT=np.array(bT)
bound0=b0<0; boundT=bT<0
res=dict(n=50,t_first=0.0,t_last=d["times"][-1],n_bound_t0=int(bound0.sum()),n_bound_T1000=int(boundT.sum()),
  same_set=bool(np.array_equal(bound0,boundT)),n_sign_changes=int((np.sign(b0)!=np.sign(bT)).sum()),
  max_abs_change_E_bind=float(abs(bT-b0).max()),argmax_change=int(abs(bT-b0).argmax()),
  mean_abs_change=float(abs(bT-b0).mean()),
  min_abs_E_bind_t0=float(abs(b0).min()),
  # sign changes at any recorded time
  n_seeds_with_any_sign_change_over_series=int(sum(1 for sd in range(50) if len(set(np.sign(load(LEG+"/outputs/phase12/seed_%02d.json"%sd)["binding_energy"])))>1)),
  mean_E_bind_T1000=float(bT.mean()),
  order_param_mean_t0=float(np.mean(S0)),order_param_mean_T1000=float(np.mean(ST)),
  report_summary_n_bound=load(LEG+"/outputs/phase12/ensemble_summary.json")["n_seeds_bound_at_final"])
save_part("item08",res);print(json.dumps(res,indent=1))
