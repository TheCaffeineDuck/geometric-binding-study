"""Add the production starts (the Session 2 dip and N/N_min 5.2 starts used by the driver), their radial settling
check at exactly those omegas, the 3D source guards and the static interaction numbers to unequal_choice.json.
Usage: python outputs/session4/testA/unequal_finalize.py
"""
import json
import os

import numpy as np

from swirl.session4.testA import driver as D
from swirl.session4.testA.static import dump

OUT = os.path.join(D.ROOT, D.OUT)


def main():
    ch = json.load(open(os.path.join(OUT, "unequal_choice.json")))
    rad = {round(q["omega_profile"], 10): q for q in json.load(open(os.path.join(OUT, "unequal_scan_production.json")))["rows"]}
    prod = {}
    for t in ("dip", "big"):
        w = D.TYPES[t]["omega"]
        n = D.numbers(t)
        q = rad[round(w, 10)]
        src = {}
        mp = os.path.join(OUT, "sources", os.path.basename(D.TYPES[t]["source"])[:-4] + ".json")
        if os.path.exists(mp):
            m = json.load(open(mp))
            src = {k: m[k] for k in ("omega", "omega_drift", "sha256", "crop_loss_E_max", "crop_loss_N", "t_s",
                                     "nonlinear_share_ref", "E_ref")}
        prod[t] = dict(omega_profile=w, N=n["N"], N_over_Nmin=n["N"] / 1633.571, E=n["E"], A=n["A"], kappa=n["kappa"],
                       hw=n["hw"], Phi0=n["Phi0"], D_sponge=n["D_sponge"],
                       radial_omega_settled_t300=q["omega_settled"]["300"], radial_drift_t300=q["drift"]["300"],
                       radial_drift_t500=q["drift"]["500"], radial_loss_rate=q["loss_rate_300_1500"],
                       source_3d=src)
    w1 = prod["dip"]["source_3d"].get("omega", prod["dip"]["radial_omega_settled_t300"])
    w2 = prod["big"]["source_3d"].get("omega", prod["big"]["radial_omega_settled_t300"])
    dw = abs(w1 - w2)
    st = json.load(open(os.path.join(OUT, "static", "static_unequal_0.906505_0.857370.json")))
    srow = {q["r"]: q["p000"] for q in st["rows"]}
    ch["production"] = dict(
        note="the driver uses the Session 2 starts of the recommended pair (dip 0.9065045, N/N_min 5.2 start 0.8573703), "
             "within 1.3e-4 in omega of the scanned members",
        dip=prod["dip"], big=prod["big"], N_ratio=prod["big"]["N"] / prod["dip"]["N"],
        delta_omega_settled=dw, beat_period=2 * np.pi / dw, beats_in_window_40_400=360 * dw / (2 * np.pi),
        static_U_amp_vs_law={str(r): dict(U_amp=srow[r]["U_amp"], U_law=srow[r]["U_law"], V_raw=srow[r]["V_raw"])
                             for r in (10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 40.0)})
    dump(ch, os.path.join(OUT, "unequal_choice.json"))
    print(json.dumps(ch["production"], indent=1, default=float))


if __name__ == "__main__":
    main()
