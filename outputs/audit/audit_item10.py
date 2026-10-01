import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL/outputs/audit")
from audit_common import *
def orbits(V, E, ce_t, perms, flip):
    n=len(V); items=[]
    for b in range(2**n):
        s=tuple(1 if (b>>k)&1 else -1 for k in range(n))
        if sum(1 for i,j in E if s[i]!=s[j])==ce_t: items.append(s)
    S=set(items); seen=set(); orbs=[]
    for s in items:
        if s in seen: continue
        orb=set()
        for p in perms:
            t=tuple(np.array(s)[list(p)])
            orb.add(t)
            if flip: orb.add(tuple(-x for x in t))
        assert orb<=S
        seen|=orb; orbs.append(sorted(orb))
    return len(items),orbs
res={}
for g,V,ce_t in [("ico",ico_verts(6.0),15),("cube",cube_verts(6.0),6)]:
    E=edges_from_coords(V)
    allp=symmetry_perms(V); rot=[p for p,d in allp.items() if d>0]; refl=[p for p,d in allp.items() if d<0]
    ident=tuple(range(len(V)))
    out=dict(n_vertices=len(V),n_edges=len(E),n_group_total=len(allp),n_rot=len(rot),n_refl=len(refl))
    out["n_assignments_at_CE"],_=orbits(V,E,ce_t,[ident],False)
    for name,perms,flip in [("rotations_only",rot,False),("rotations_reflections",list(allp),False),("rotations_flip",rot,True),("all_three",list(allp),True)]:
        _,o=orbits(V,E,ce_t,perms,flip); out[name]=len(o)
    # which legacy classes (rot+flip) merge under reflections, with sigma^2_NS
    _,o_rf=orbits(V,E,ce_t,rot,True); _,o_all=orbits(V,E,ce_t,list(allp),True)
    def sig2(s):
        c=np.zeros(len(V))
        for i,j in E:
            if s[i]!=s[j]: c[i]+=1;c[j]+=1
        return round(float(np.var(c)),4)
    def energy(s): return round(pair_sum_binary(V,s),4)
    out["rot_flip_classes"]=[dict(size=len(o),sig2=sig2(o[0]),closed_form_E=energy(o[0]),n_pi_min=min(sum(1 for x in s if x<0) for s in o)) for o in o_rf]
    merged=[]
    for oa in o_all:
        subs=[i for i,o in enumerate(o_rf) if set(o)<=set(oa)]
        if len(subs)>1: merged.append(dict(rot_flip_class_indices=subs,sig2=[out["rot_flip_classes"][i]["sig2"] for i in subs],closed_form_E=[out["rot_flip_classes"][i]["closed_form_E"] for i in subs]))
    out["merged_by_reflection"]=merged
    # does a reflection ever map a rot+flip class to a different rot+flip class (chirality pair)?
    res[g]=out
save_part("item10",res);print(json.dumps({g:{k:v for k,v in d.items() if k!="rot_flip_classes"} for g,d in res.items()},indent=1))
for g in res: print(g,res[g]["rot_flip_classes"])
