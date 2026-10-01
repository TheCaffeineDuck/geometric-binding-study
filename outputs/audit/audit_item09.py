import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL/outputs/audit")
from audit_common import *
S=[];T=None
for sd in range(50):
    d=load(LEG+"/outputs/phase12/seed_%02d.json"%sd); S.append(d["order_parameter"]); T=np.array(d["times"])
S=np.array(S); m=np.nanmean(S,axis=0)
n=len(T); dt=float(np.diff(T).mean()); win=n*dt
f=np.fft.rfftfreq(n,d=dt); A=np.abs(np.fft.rfft(m-m.mean()))
order=np.argsort(A[1:])[::-1][:6]+1
res=dict(n_samples=n,spacing=dt,t_first=float(T[0]),t_last=float(T[-1]),span_last_minus_first=float(T[-1]-T[0]),window_n_times_dt=win,
  nan_in_mean=int(np.isnan(m).sum()),
  periods_window_over_k={str(k):win/k for k in (1,2,4)},
  periods_span_over_k={str(k):(T[-1]-T[0])/k for k in (1,2,4)},
  top_fft_bins=[dict(k=int(k),period=float(win/k),amp=float(A[k])) for k in order],
  fundamental_freq=float(f[1]), nyquist_period=2*dt)
save_part("item09",res);print(json.dumps(res,indent=1))
