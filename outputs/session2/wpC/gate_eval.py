"""WP-C gate evaluation (SESSION_02_PREDICTIONS.md, Block 2 "Gate", criteria G1 and G2).

Usage: <python> outputs/session2/wpC/gate_eval.py [--results DIR] [--out FILE]
Reads the four 3D single-lump gate runs single_x12_t{00.0,22.5,45.0,67.5} and writes the
numbers and the verdicts. Criteria are fixed (not reinterpreted):

  G1  single_x12_t00.0: |centroid drift| at T = 500 below 0.01.
      Drift = period-averaged ball centroid at t_out = 500 minus the first window
      (the same definition as noise_floor.py); the 3-vector norm is the criterion,
      the x component is reported.
  G2  theta = 22.5, 45, 67.5 against theta = 0:
      |frac_theta - frac_0| <= 0.01 at T = 300 and at T = 500;
      |omega_theta - omega_0| <= 0.005 at T = 300 and at T = 500;
      no death: frac > 0.5 at every t_out of the run.
      frac = (ball energy) / (the run's own t = 0 ball energy). Ball = 3 half-widths
      (R_ball = 10). Primary: whole-period average of the ball energy (the engine's
      "averages", window ending at T) over the raw t = 0 value. omega: whole-period
      average frequency of phi(centre, t), window ending at T.
  Cross-checks reported, not used for the verdict: instantaneous raw ball energy at T over
  its t = 0 value; omega median over t_center in [30, 80].
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THETAS = ["00.0", "22.5", "45.0", "67.5"]
TOL_FRAC, TOL_OMEGA, TOL_DRIFT, DEATH = 0.01, 0.005, 0.01, 0.5
TIMES = (300.0, 500.0)


def load(path):
    with open(path) as f:
        return json.load(f)


def at(t_list, v_list, T):
    i = int(np.argmin(np.abs(np.array(t_list) - T)))
    return i, v_list[i]


def analyse(res):
    p = res["params"]
    raw = res["lumps"]["raw"]["series"]
    K = len(res["lumps"]["raw"]["centers"])
    rt = np.array(raw["t"])
    rE = np.array(raw["E"]).reshape(rt.size, K)[:, 0]
    rX = np.array(raw["X"]).reshape(rt.size, K, 3)[:, 0, :]
    av = res["lumps"]["averages"]
    lu = av["lumps"][0]
    t_out = np.array(av["t"])
    ok = [i for i in range(len(t_out)) if lu["E"][i] is not None and lu["omega"][i] is not None]
    E0 = float(rE[0])
    tt = t_out[ok]
    Eav = np.array([lu["E"][i] for i in ok])
    om = np.array([lu["omega"][i] for i in ok])
    X = np.array([lu["X"][i] for i in ok])
    tc = np.array([lu["t_center"][i] for i in ok])
    frac = Eav / E0
    out = {"completed": res["completed"], "t_reached": res["t"], "E_ball_t0_raw": E0,
           "N": p["N"], "L": p["L"], "dt": p["dt"], "threads": p["threads"],
           "min_frac_all_t": float(frac.min()), "t_at_min_frac": float(tt[int(np.argmin(frac))]),
           "frac": {}, "omega": {}, "frac_instantaneous": {}}
    for T in TIMES:
        i = int(np.argmin(np.abs(tt - T)))
        out["frac"][str(int(T))] = {"t_out": float(tt[i]), "value": float(frac[i])}
        out["omega"][str(int(T))] = {"t_out": float(tt[i]), "value": float(om[i])}
        j = int(np.argmin(np.abs(rt - T)))
        out["frac_instantaneous"][str(int(T))] = {"t": float(rt[j]), "value": float(rE[j] / E0)}
    sel = (tc >= 30) & (tc <= 80)
    out["omega_median_30_80"] = float(np.median(om[sel])) if sel.any() else None
    d = X[-1] - X[0]
    out["centroid"] = {"t_out_first": float(tt[0]), "t_out_last": float(tt[-1]),
                       "X_first": X[0].tolist(), "X_last": X[-1].tolist(),
                       "drift_vec": d.tolist(), "drift_norm": float(np.linalg.norm(d)),
                       "drift_x": float(d[0]),
                       "raw_X_t0": rX[0].tolist(), "raw_X_last": rX[-1].tolist(),
                       "raw_drift_norm": float(np.linalg.norm(rX[-1] - rX[0]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or os.path.join(a.results, "gate.json")
    runs = {}
    for th in THETAS:
        name = f"single_x12_t{th}"
        runs[th] = analyse(load(os.path.join(a.results, name + ".json")))
        runs[th]["name"] = name
    g = {"criteria": {"G1": f"|centroid drift| of single_x12_t00.0 at T = 500 below {TOL_DRIFT}",
                      "G2": f"theta = 22.5, 45, 67.5 vs 0: |d frac| <= {TOL_FRAC} at T = 300 and 500; "
                            f"|d omega| <= {TOL_OMEGA}; frac > {DEATH} throughout"},
         "definitions": __doc__, "runs": runs}
    r0 = runs["00.0"]
    c = r0["centroid"]
    g1 = {"drift_norm": c["drift_norm"], "drift_x": c["drift_x"], "threshold": TOL_DRIFT,
          "t_out_last": c["t_out_last"], "pass": bool(c["drift_norm"] < TOL_DRIFT and r0["completed"]
                                                      and c["t_out_last"] >= 500 - 1e-9)}
    g["G1"] = g1
    g2 = {"rows": [], "pass": True}
    for th in THETAS[1:]:
        r = runs[th]
        row = {"theta": float(th), "completed": r["completed"], "min_frac_all_t": r["min_frac_all_t"],
               "no_death": bool(r["min_frac_all_t"] > DEATH)}
        ok = r["completed"] and row["no_death"]
        for T in TIMES:
            k = str(int(T))
            df = r["frac"][k]["value"] - r0["frac"][k]["value"]
            dw = r["omega"][k]["value"] - r0["omega"][k]["value"]
            dfi = r["frac_instantaneous"][k]["value"] - r0["frac_instantaneous"][k]["value"]
            row[f"frac_{k}"] = r["frac"][k]["value"]
            row[f"dfrac_{k}"] = df
            row[f"omega_{k}"] = r["omega"][k]["value"]
            row[f"domega_{k}"] = dw
            row[f"dfrac_instantaneous_{k}"] = dfi
            ok = ok and abs(df) <= TOL_FRAC and abs(dw) <= TOL_OMEGA
            ok = ok and abs(r["frac"][k]["t_out"] - T) < 1.0 and abs(r0["frac"][k]["t_out"] - T) < 1.0
        row["domega_median_30_80"] = (None if r["omega_median_30_80"] is None or r0["omega_median_30_80"] is None
                                      else r["omega_median_30_80"] - r0["omega_median_30_80"])
        row["pass"] = bool(ok)
        g2["pass"] = g2["pass"] and row["pass"]
        g2["rows"].append(row)
    g2["reference_theta0"] = {"frac_300": r0["frac"]["300"]["value"], "frac_500": r0["frac"]["500"]["value"],
                              "omega_300": r0["omega"]["300"]["value"], "omega_500": r0["omega"]["500"]["value"],
                              "min_frac_all_t": r0["min_frac_all_t"],
                              "no_death": bool(r0["min_frac_all_t"] > DEATH)}
    g2["pass"] = bool(g2["pass"] and g2["reference_theta0"]["no_death"])
    g["G2"] = g2
    g["gate_pass"] = bool(g1["pass"] and g2["pass"])
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(g, f, indent=1)
    os.replace(tmp, out_path)
    print(json.dumps({"G1": g1, "G2": {"pass": g2["pass"], "rows": g2["rows"]}, "gate_pass": g["gate_pass"]}, indent=1))


if __name__ == "__main__":
    main()
