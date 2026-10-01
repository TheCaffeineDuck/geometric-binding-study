import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL/outputs/audit"); sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL/outputs/audit/scratch")
from audit_common import *
res = {}
res["J_d6"] = J(6.0); res["J_d10"] = J(10.0)
ds = load(LEG+"/outputs/distance_sweep/distance_sweep_summary.json")
tab = {r["d"]: r for r in ds["table"]}
res["PDE_pair_same_d6_T500"] = tab[6.0]["E_pair_same"]; res["PDE_pair_same_d10_T500"] = tab[10.0]["E_pair_same"]
cv = cube_verts(6.0); iv = ico_verts(6.0)
res["cube_edges"] = len(edges_from_coords(cv)); res["ico_edges"] = len(edges_from_coords(iv))
# patterns: index bits x=1,y=2,z=4 (legacy ordering); compute classes from coordinates
bip = [-1,1,1,-1,1,-1,-1,1]            # legacy "checkerboard" in boundary_test (bipartite, CE=12)
p2 = load(LEG+"/outputs/phase2/cube/polarized_T1.json")["phases"]   # phase2 "polarized_T1" file, PDE -23.94
chk6 = [-1,-1,-1,1,1,1,-1,1]           # phase2 "checkerboard" file (CE=6), PDE -3.86
def ce(V,s): return sum(1 for i,j in edges_from_coords(V) if s[i]!=s[j])
cands = {}
for name,s in [("bipartite_boundary_test_checkerboard",bip),("phase2_polarized_T1_file",p2),("phase2_checkerboard_CE6_file",chk6)]:
    cands[name] = dict(phases=s, CE=ce(cv,s), closed_form=pair_sum_binary(cv,s))
res["cube_patterns"] = cands
res["cube_PDE"] = dict(
  boundary_test_checkerboard_L50_T500=load(LEG+"/outputs/boundary_test/comparison.json")["results"]["cube_checkerboard"]["E_bind_L50"],
  phase2_polarized_T1_T500=load(LEG+"/outputs/phase2/cube/polarized_T1.json")["final_E_bind"],
  phase2_checkerboard_CE6_T500=load(LEG+"/outputs/phase2/cube/checkerboard.json")["final_E_bind"],
  phase2_cube_polarized_T1_CE12_T500=load(LEG+"/outputs/phase2/cube/cube_polarized_T1_CE12.json")["final_E_bind"] if "final_E_bind" in load(LEG+"/outputs/phase2/cube/cube_polarized_T1_CE12.json") else None)
# all 2^8 patterns: which closed-form value is nearest -23.98 / what the tetrahedral (A={0,3,5,6}) patterns give
vals = {}
for b in range(256):
    s = [1 if (b>>k)&1 else -1 for k in range(8)]
    vals[b] = pair_sum_binary(cv,s)
res["cube_closed_form_distinct_values"] = sorted(set(round(v,3) for v in vals.values()))
# ico all-same
res["ico_all_same_closed_form"] = pair_sum_binary(iv,[1]*12)
i0 = load(LEG+"/outputs/phase2/icosahedron/ce_0.json")
res["ico_all_same_PDE_T500"] = i0["final_E_bind"] if "final_E_bind" in i0 else None
res["ico_all_same_PDE_phase3"] = 164.1058197879512
res["ico_pairs_by_distance"] = {}
D = np.linalg.norm(iv[:,None]-iv[None],axis=2)
for d in sorted(set(np.round(D[np.triu_indices(12,1)],6))):
    res["ico_pairs_by_distance"][str(d)] = int((np.abs(D[np.triu_indices(12,1)]-d)<1e-5).sum())
res["ico_edges_only_sum"] = 30*J(6.0)
# grid t=0 static E_bind via legacy compute_energy (copy of evolver), as a measured-on-grid cross-check
from legacy_evolver_copy import SexticEvolver
ev = SexticEvolver(N=N,L=L,m=M,g4=G4,g6=G6,dissipation_sigma=SIGMA)
def static(V, thetas=None, s=None, omega=1.113):
    p = np.zeros((N,N,N)); pd = np.zeros((N,N,N))
    for k,pos in enumerate(V):
        env = PHI0*np.exp(-((ev.X-pos[0])**2+(ev.Y-pos[1])**2+(ev.Z-pos[2])**2)/(2*R**2))
        if thetas is None: p += s[k]*env
        else: p += np.cos(thetas[k])*env; pd += -omega*np.sin(thetas[k])*env
    ev.set_initial_conditions(p,pd); return ev.compute_energy()
E1 = load(LEG+"/outputs/phase1/set_B_baseline.json")["E_series"][0]
res["E_single_t0"] = E1
res["grid_t0_cube_bipartite"] = static(cv,s=bip)-8*E1
res["grid_t0_cube_p2"] = static(cv,s=p2)-8*E1
res["grid_t0_ico_all_same"] = static(iv,s=[1]*12)-12*E1
pv = np.array([[-3,0,0],[3,0,0]],float); res["grid_t0_pair_d6"] = static(pv,s=[1,1])-2*E1
pv = np.array([[-5,0,0],[5,0,0]],float); res["grid_t0_pair_d10"] = static(pv,s=[1,1])-2*E1
# 50 seeds
cf=[];cf2=[];pde=[];pde0=[];seeds=[]
for sd in range(50):
    d = load(LEG+"/outputs/phase12/seed_%02d.json"%sd)
    th = np.array(d["initial_phases"]); 
    e=0.0; e2=0.0
    for i in range(12):
        for j in range(i+1,12):
            dd = np.linalg.norm(iv[i]-iv[j])
            e += J(dd)*math.cos(th[i])*math.cos(th[j]) + K(dd,1.113)*math.sin(th[i])*math.sin(th[j])
            e2 += J(dd)*math.cos(th[i])*math.cos(th[j]) + K(dd,0.5**0*1.1125610675103892)*math.sin(th[i])*math.sin(th[j])
    cf.append(e); cf2.append(e2); pde.append(d["binding_energy"][-1]); pde0.append(d["binding_energy"][0]); seeds.append(d["times"][-1])
cf=np.array(cf);pde=np.array(pde);cf2=np.array(cf2)
res["seeds"] = dict(n=50, t_final=seeds[0], corr_T1000=float(np.corrcoef(cf,pde)[0,1]), max_abs_miss=float(abs(cf-pde).max()),
   argmax_miss=int(abs(cf-pde).argmax()), mean_abs_miss=float(abs(cf-pde).mean()), pde_min=float(pde.min()), pde_max=float(pde.max()),
   corr_t0=float(np.corrcoef(cf,pde0)[0,1]), max_abs_miss_t0=float(abs(cf-np.array(pde0)).max()),
   corr_omega_1p11256=float(np.corrcoef(cf2,pde)[0,1]), max_abs_miss_omega_1p11256=float(abs(cf2-pde).max()),
   cf_min=float(cf.min()), cf_max=float(cf.max()))
save_part("item02", res); print(json.dumps(res, indent=1))
