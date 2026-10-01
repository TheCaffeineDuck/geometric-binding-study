import json,glob,os
r={}
for f in sorted(glob.glob("/Users/aaronramos/Desktop/SWIRL/outputs/audit/part_*.json")):
    r[os.path.basename(f)[5:-5]]=json.load(open(f))
r["_meta"]=dict(params=dict(m=1.0,g4=0.30,g6=0.055,phi0=0.5,R=2.5,d=6.0,N=64,L=50.0),scripts=sorted(os.path.basename(x) for x in glob.glob("/Users/aaronramos/Desktop/SWIRL/outputs/audit/audit_item*.py")))
tmp="/Users/aaronramos/Desktop/SWIRL/outputs/audit/audit_results.json.tmp"
json.dump(r,open(tmp,"w"),indent=1); os.replace(tmp,"/Users/aaronramos/Desktop/SWIRL/outputs/audit/audit_results.json")
