import json,os,numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from swirl.session4.testB import analysis as A
O=A.OUT
# Run from the repo root: PYTHONPATH=$PWD <venv python> outputs/session4/testB/make_plots.py
for n in ['b1_primary','b2_dip']:
    a=json.load(open(O+'/'+n+'_analysis.json')); z=np.load(O+'/'+n+'_series.npz'); t,phi,rad=z['t'],z['phi'],list(z['probes']); dt=t[1]-t[0]
    eps=a['epochs']; cols=plt.cm.viridis(np.linspace(0,.9,len(eps)))
    fig=plt.figure(figsize=(15,13)); gs=fig.add_gridspec(3,3)
    ax=fig.add_subplot(gs[0,:])
    for k,ep in enumerate(eps):
        om,P,_,_=A.welch(phi[:,rad.index(0.0)],dt,ep['t0'],ep['t1']); ax.semilogy(om,np.maximum(P,1e-40),c=cols[k],lw=.8,label='[%g,%g] omega_core=%.4f'%(ep['t0'],ep['t1'],ep['omega_core']))
    ax.axvline(1,c='k',ls='--',lw=.8); ax.set_xlim(0,14); ax.set_ylim(1e-30,1e2); ax.legend(fontsize=7); ax.set_title('(a) %s core (r=0) power spectrum; dashed omega=1'%n); ax.set_xlabel('omega/m')
    for i,r in enumerate([30,60,120]):
        ax=fig.add_subplot(gs[1,i])
        for k,ep in enumerate(eps):
            om,P,_,_=A.welch(phi[:,rad.index(float(r))],dt,ep['t0'],ep['t1']); ax.semilogy(om,np.maximum(P,1e-40),c=cols[k],lw=.7)
        for nn in range(1,10): ax.axvline(nn*eps[0]['omega_core'],c='0.75',lw=.5,zorder=0)
        ax.axvline(1,c='k',ls='--',lw=.8); ax.set_xlim(0,8); ax.set_ylim(1e-30,1e-3); ax.set_title('(b) r=%d; grey n*omega_core(epoch 0); dashed omega=1'%r,fontsize=9); ax.set_xlabel('omega/m')
    ax=fig.add_subplot(gs[2,:2])
    for k,ep in enumerate(eps):
        d={}
        for p in ep['probes']:
            if p['r']<30: continue
            for l in p['lines']:
                if l['label'] in('3 omega','5 omega','7 omega','9 omega'): d.setdefault(l['label'],[]).append((p['r'],p['r']*l['amp']))
        for j,(lab,L) in enumerate(sorted(d.items())):
            L.sort(); ax.semilogy([x[0] for x in L],[x[1] for x in L],'o-',c=cols[k],ms=3,lw=.8,alpha=.9)
            if k==0: ax.annotate(lab,L[-1],fontsize=7)
    ax.set_xlabel('r'); ax.set_ylabel('r * amplitude'); ax.set_title('(c) r*amp vs r per identified line (colour = epoch; flat = 1/r)',fontsize=9)
    ax=fig.add_subplot(gs[2,2]); x=np.arange(len(eps)); w=1/8
    ser=[('loss -dE_in/dt',[e['loss_rate_Ein_3hw'] for e in eps]),('absorbed rate',[e['absorbed_rate'] for e in eps])]
    for rr in (30,120):
        for key in ('flux_above_m1','flux_above_harmonics_m1'):
            ser.append(('%s r=%d'%(key,rr),[[p[key] for p in e['probes'] if p['r']==rr][0] for e in eps]))
    for i,(lab,v) in enumerate(ser): ax.bar(x+(i-2.5)*w,v,w,label=lab)
    ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(['[%g,%g]k'%(e['t0']/1000,e['t1']/1000) for e in eps],fontsize=7); ax.legend(fontsize=5); ax.set_title('(d) loss, absorbed, above-threshold and harmonics-only flux',fontsize=9)
    fig.suptitle(n); fig.tight_layout(); p=O+'/'+n+'_testB_plots.png'; fig.savefig(p+'.tmp.png',dpi=100); os.replace(p+'.tmp.png',p); plt.close(fig); print(p)
