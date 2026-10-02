"""Rerun of the radial driver (swirl.family.wpa_run.run) for the A10 start to T = 1000, records every 0.5.
Same params as the stored T=300k run (dr 0.05, dt 0.0125, R_sponge 193.65, R_box 387.30). Run: NUMBA_NUM_THREADS=1 nice .venv/bin/python <this>"""
import json, os, sys
import numpy as np
R = "/Users/aaronramos/Desktop/SWIRL/"; sys.path.insert(0, R)
from swirl.family import wpa_run
src = json.load(open(R + "outputs/session2/wpA/long_beta5.0000_w0.975598_T300k.json"))
om = src["params"]["omega_profile"]
res = wpa_run.run(5.0, om, 1000.0, stop_at_death=False)
s = res["series"]
out = dict(params=res["params"], E0=float(s["E_in"][0, 0] if False else res["E0"]) if "E0" in res else None,
           t=np.asarray(s["t"]).tolist(), E_in=np.asarray(s["E_in"]).tolist(), E_tot=np.asarray(s["E_tot"]).tolist(),
           omega_t=np.asarray(res["omega_t"]).tolist(), omega=np.asarray(res["omega"]).tolist(),
           regime=res.get("regime"), keys=list(res.keys()))
E0 = out["E0"] if out["E0"] is not None else out["E_tot"][0]
out["E0"] = E0
out["frac"] = (np.asarray(out["E_in"])[:, 0] / E0).tolist()
p = R + "outputs/session2/wpA/confirm3d_beta5/radial_beta5_T1000_dense.json"
json.dump(out, open(p + ".tmp", "w")); os.replace(p + ".tmp", p); print("done", E0)
