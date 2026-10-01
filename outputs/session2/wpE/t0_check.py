"""t = 0 check of the WP-E configs: total energy, N-proxy and instantaneous distinct-lump count of the
superposed initial field, next to K times the single-lump values (same dx, one lump at the origin, phase 0).
N-proxy = int (pi^2 + omega^2 phi^2) dV / (2 omega) (phase independent for a single-frequency lump)."""
import json, os, sys, copy
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from swirl.engine.evolver import Evolver, SPONGE_DEFAULT
from swirl.engine.initial import build_initial
from swirl.engine.lumps import find_lumps
from swirl.engine.snapshot import load_meta
CD = os.path.join(ROOT, "outputs/session2/wpE/configs")
def measure(cfg, om):
    ev = Evolver(cfg["N"], cfg["L"], cfg["m"], cfg["g4"], cfg["g6"], absorber=dict(SPONGE_DEFAULT), kfilter=None, threads=2)
    phi, pi = build_initial(ev.coords, ev.L, cfg["init"], info={}, threads=2)
    ev.set_state(phi, pi)
    E = ev.energy_parts()
    Np = float(np.sum(pi**2 + om**2 * phi**2) * ev.dV / (2 * om))
    r = find_lumps(ev.energy_density(), ev.coords, ev.L, ev.dx, 0.01, 1.0, 0.9)
    return E["E_total"], Np, r["count"], r["count_distinct"]
single = {}
out = {}
names = sorted(f[:-5] for f in os.listdir(CD) if f.endswith(".json") and f != "index.json")
for n in names:
    cfg = json.load(open(os.path.join(CD, n + ".json")))
    sn = cfg["init"]["snapshot"]; om = load_meta(os.path.join(ROOT, sn["source"]))["omega"]
    key = (round(cfg["L"] / cfg["N"], 6), cfg["g4"])
    if key not in single:
        c1 = copy.deepcopy(cfg); c1["init"]["snapshot"]["centers"] = [[0.0, 0.0, 0.0]]; c1["init"]["snapshot"]["phases"] = [0.0]
        single[key] = measure(c1, om)
    E, Np, cnt, cd = measure(cfg, om)
    K = cfg["_candidate"]["K"]; E1, N1 = single[key][:2]
    w = sn["window"]
    out[n] = dict(K=K, N=cfg["N"], L=cfg["L"], dx=cfg["L"]/cfg["N"], steps=int(round(cfg["t_end"]/cfg["dt"])), T=cfg["t_end"], window=w,
                  g4=cfg["g4"], g6=cfg["g6"], E0=E, K_E1=K*E1, E_ratio=E/(K*E1), N0=Np, K_N1=K*N1, N_ratio=Np/(K*N1),
                  E1=E1, N1=N1, count_t0=cnt, count_distinct_t0=cd)
    print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[n].items()}, flush=True)
json.dump(out, open(os.path.join(ROOT, "outputs/session2/wpE/t0_check.json.tmp"), "w"), indent=1)
os.replace(os.path.join(ROOT, "outputs/session2/wpE/t0_check.json.tmp"), os.path.join(ROOT, "outputs/session2/wpE/t0_check.json"))
