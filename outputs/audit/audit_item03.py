import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL/outputs/audit"); sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL/outputs/audit/scratch")
from audit_common import *
from legacy_evolver_copy import SexticEvolver
GEO = dict(ico=ico_verts(6.0), cube=cube_verts(6.0), oct=oct_verts(6.0))
ev = SexticEvolver(N=N,L=L,m=M,g4=G4,g6=G6,dissipation_sigma=SIGMA)
E1 = load(LEG+"/outputs/phase1/set_B_baseline.json")["E_series"][0]
def grid_ebind(V,s):
    p=np.zeros((N,N,N))
    for k,pos in enumerate(V):
        p += s[k]*PHI0*np.exp(-((ev.X-pos[0])**2+(ev.Y-pos[1])**2+(ev.Z-pos[2])**2)/(2*R**2))
    ev.set_initial_conditions(p,np.zeros_like(p)); return ev.compute_energy()-len(V)*E1
def sig2(V,E,s):
    c=np.zeros(len(V))
    for i,j in E:
        if s[i]!=s[j]: c[i]+=1; c[j]+=1
    return float(np.var(c))

res={}
def fit(xs,ys):
    sl,ic=np.polyfit(xs,ys,1); return float(sl),float(ic),float(np.corrcoef(xs,ys)[0,1]**2)
def classes(V,E,ce_t,syms):
    n=len(V); seen={}
    for b in range(2**n):
        s=tuple(1 if (b>>k)&1 else -1 for k in range(n))
        if sum(1 for i,j in E if s[i]!=s[j])!=ce_t: continue
        can=min(min(tuple(np.array(s)[list(p)]),tuple(-np.array(s)[list(p)])) for p in syms)
        seen.setdefault(can,s)
    return list(seen.values())
for g,ce_t,proper in [("ico",15,True),("cube",6,None),("oct",6,None),("oct",8,None)]:
    V=GEO[g]; n=len(V); E=edges_from_coords(V)
    syms=list(symmetry_perms(V,proper=proper).keys())
    reps=classes(V,E,ce_t,syms)
    rows=[dict(s=list(s),sig2=sig2(V,E,s),closed=pair_sum_binary(V,s),grid_t0=grid_ebind(V,s)) for s in reps]
    xs=np.array([r["sig2"] for r in rows])
    D=np.linalg.norm(V[:,None]-V[None],axis=2); d2=sorted(set(np.round(D[np.triu_indices(n,1)],6)))[1]
    pred = (n if g!="oct" else n/2)*J(float(d2))
    sc=fit(xs,[r["closed"] for r in rows]); sg=fit(xs,[r["grid_t0"] for r in rows])
    res["%s_CE%d"%(g,ce_t)]=dict(n_classes=len(rows),symmetry="60 rotations + flip" if proper else "all (rot+refl+flip)",d2=float(d2),
        predicted_N_J_d2=pred,slope_closed_form_fit=sc[0],r2_closed=sc[2],slope_grid_t0_fit=sg[0],r2_grid=sg[2],rows=rows)
    print(g,ce_t,len(rows),"pred %.4f closed %.4f (r2 %.4f) grid_t0 %.4f (r2 %.4f)"%(pred,sc[0],sc[2],sg[0],sg[2]))
# paper Table 4 (legacy/paper/paper_draft_v9.md lines 289-296): measured PDE E_bind, sigma2 as printed
paper=[(0.917,-1.487),(0.917,-1.494),(0.917,-1.494),(1.250,-0.426),(1.250,-0.446),(2.250,3.557)]
sl,ic,r2=fit([p[0] for p in paper],[p[1] for p in paper])
res["paper_table4_regression"]=dict(slope=sl,intercept=ic,r2=r2,source="legacy/paper/paper_draft_v9.md Table 4 (not a legacy/outputs file)")
res["ico_ce15A_PDE_T500"]=load(LEG+"/outputs/phase2/icosahedron/ce_15.json").get("final_E_bind")
print("paper table slope",sl,ic,r2, res["ico_ce15A_PDE_T500"])
save_part("item03",res)
