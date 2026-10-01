"""Builds candidates_wpE.json (Session 2 Block 3 run table E1-E14)."""
import json, math, os
HERE = os.path.dirname(os.path.abspath(__file__))
def ring(K, R):
    return [[R*math.cos(2*math.pi*k/K), R*math.sin(2*math.pi*k/K), 0.0] for k in range(K)]
def Ra(K, a): return a/(2*math.sin(math.pi/K))
kap = 0.41647
def step(K, d): return [d*k for k in range(K)]
c = []
def add(name, fam, pos, ph, T, **kw):
    c.append(dict(name=name, family=fam, positions=pos, phases_deg=ph, T=T, **kw))
add("E01_ring5_m1_a16","ring5_a16",ring(5,Ra(5,16)),[0,72,144,216,288],600,linear_twin=True,res_check=False,box_check=False)
add("E02_ring5_m0_a16","ring5_a16",ring(5,Ra(5,16)),[0]*5,400)
add("E03_ring5_m1_a8.11","ring5_a8",ring(5,2.8740/kap),[0,72,144,216,288],500,res_check=True,box_check=True)
add("E04_ring5_m1_a3.02","ring5_a3",ring(5,1.0695021/kap),[0,72,144,216,288],400)
add("E05_ring4_alt_a16","ring4_a16",ring(4,Ra(4,16)),[0,180,0,180],500)
add("E06_ring4_m1_a12","ring4_a12",ring(4,Ra(4,12)),[0,90,180,270],600)
add("E07_ring6_m1_a7.32","ring6_a7",ring(6,3.0465/kap),[0,60,120,180,240,300],400)
R=1.5909003/kap
add("E08_octa_c_R3.82","octa",[[R,0,0],[-R,0,0],[0,R,0],[0,-R,0],[0,0,R],[0,0,-R],[0,0,0]],[180]*6+[0],300)
R=3.5432/kap; s=R/math.sqrt(3)
cub=[[sx*s,sy*s,sz*s] for sx in(1,-1) for sy in(1,-1) for sz in(1,-1)]+[[0,0,0]]
add("E09_cube_c_R8.51","cube",cub,[180]*8+[0],400)
add("E10_ring8_m2_a12","ring8_a12",ring(8,Ra(8,12)),[0,90,180,270]*2,500)
add("E11_ring6_c_R15.77","ring6c",ring(6,15.767)+[[0,0,0]],[180]*6+[0],700)
add("E12_ring5_rounded_a16","ring5_a16r",ring(5,Ra(5,16)),[0,0,180,180,0],400)
add("E13_ring5_m2_a16","ring5_a16m2",ring(5,Ra(5,16)),[0,144,288,72,216],500)
add("E14_ring5_m1_a30","ring5_a30",ring(5,Ra(5,30)),[0,72,144,216,288],500)
spec={"defaults":{"window":[26.0,30.0],"start":"snapshot","dx":0.55,"dt":0.04,
  "source":"outputs/session2/wpE/source/src_prod_N160_L88.npz","box":"auto","res_check":False,"box_check":False,"linear_twin":False},
  "candidates":c}
json.dump(spec,open(os.path.join(HERE,"candidates_wpE.json"),"w"),indent=1)
for x in c: print(x["name"],len(x["positions"]))
