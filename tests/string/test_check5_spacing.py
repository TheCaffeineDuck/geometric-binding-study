"""Check 5. Numerical spacing in C-smooth ('fd' scheme) at h = 1/4, 1/8, 1/16.

Start: Gaussian displacement bump, sigma = 0.5, at rest, the same continuum data at every h:
A = 0.8 sigma sqrt(e) (continuum peak stretch 0.8). Runs: linear twin and G1, T = 300, window
W = 10 fixed for all h. Comparison quantities are the production ones from the driver: window
frac at t = 100, 200, 300; core rate (zero crossings of the projection on the start, and of the
stretch at the bond of initial peak stretch); L_half of the record-averaged window energy at the
end; and the pointwise displacement at the h = 1/4 points (shared by all three) at t = 50, 100,
200, 300 (snapshots under checkpoints/, not committed).

Convergence order between spacings, where differences shrink: p = log2(|q(1/4) - q(1/8)| /
|q(1/8) - q(1/16)|). The scheme is second order for smooth fields (check 1: dispersion error
O(h^2)), so p ~ 2 is expected for the linear run. Prediction O5 (orchestrator, not tuned toward):
for G1 the pointwise profiles stop agreeing after a time growing like 1/h while frac still agrees.

Failure would look like: the linear run not converging (p far from 2), which would be a code fault.
For G1, no order is a result, reported as such."""

import os

import numpy as np
import pytest

from swirl.string import driver as DR

HS = (0.25, 0.125, 0.0625)
SNAPS = (50.0, 100.0, 200.0, 300.0)


def _order(a, b, c):
    e1, e2 = abs(a - b), abs(b - c)
    if e2 == 0 or e1 == 0:
        return None, e1, e2
    return float(np.log2(e1 / e2)), float(e1), float(e2)


def _runs(law):
    out = {}
    A = 0.8 * 0.5 * np.sqrt(np.e)
    for h in HS:
        name = "check5_fd_%s_h%g" % (law, h)
        cfg = {"kind": "evolve", "name": name, "medium": "fd", "law": law, "h": h,
               "N": int(round(320 / h)), "sponge_len": 100.0, "sigma_max": 0.3, "dt": 0.02,
               "T": 300.0, "sample_dt": 0.1, "record_periods": 1, "W": 10.0,
               "start": {"type": "gauss", "obj": "bump", "sigma": 0.5, "A": A}, "twin": False,
               "snap_times": list(SNAPS), "probe_max": 4096}
        r = DR.evolve(cfg, log=lambda *a: None)
        snaps = {}
        for ts in SNAPS:
            p = os.path.join(DR.CKPT, "%s_snap_t%g.npz" % (name, ts))
            z = np.load(p)
            step = int(round(0.25 / h))
            c = len(z["x"]) // 2
            idx = c + step * np.arange(-int(20 / 0.25), int(20 / 0.25) + 1)
            snaps[ts] = (z["x"][idx], z["u"][idx])
            os.remove(p)
        out[h] = (r, snaps)
    return out


@pytest.mark.parametrize("law", ["lin", "G1"])
def test_spacing(law, record):
    R = _runs(law)
    rows = {}
    q = {}
    for h in HS:
        r, _ = R[h]
        recs = r["records"]
        t = np.array([x["t_mid"] for x in recs])
        fr = np.array([x["E_fixed"] for x in recs]) / r["summary"]["E_fixed0"]
        q[h] = {"frac_t100": float(np.interp(100, t, fr)), "frac_t200": float(np.interp(200, t, fr)),
                "frac_t300": float(fr[-1]),
                "rate_core": r["summary"]["rate_zero_crossings_core"],
                "rate_peak_stretch": r["summary"]["rate_zero_crossings_peak_stretch"],
                "L_half_end": recs[-1]["L_half_window"], "rms_width_end": recs[-1]["rms_width_window"],
                "k_mean_stretch_end": recs[-1].get("k_mean_stretch"),
                "E0": r["summary"]["E0"], "max_abs_balance": r["summary"]["max_abs_balance"]}
    for key in q[HS[0]]:
        vals = [q[h][key] for h in HS]
        if any(v is None for v in vals):
            rows[key] = {"values": vals, "order": None}
            continue
        p, e1, e2 = _order(*vals)
        rows[key] = {"values": vals, "diff_4_8": e1, "diff_8_16": e2, "order": p}
    for ts in SNAPS:
        u = [R[h][1][ts][1] for h in HS]
        assert np.allclose(R[HS[0]][1][ts][0], R[HS[2]][1][ts][0])
        e1 = float(np.max(np.abs(u[0] - u[1])))
        e2 = float(np.max(np.abs(u[1] - u[2])))
        scale = float(np.max(np.abs(u[2])))
        rows["pointwise_u_t%g" % ts] = {"diff_4_8": e1, "diff_8_16": e2, "scale": scale,
                                         "order": float(np.log2(e1 / e2)) if e2 > 0 else None}
    print("\ncheck5 fd %s, sigma=0.5, A=%.4f:" % (law, 0.8 * 0.5 * np.sqrt(np.e)))
    for k, v in rows.items():
        print("  %-22s %s" % (k, v))
    record("check5_fd_%s" % law, {"rows": rows, "h": HS})
    if law == "lin":
        po = [rows["pointwise_u_t%g" % ts]["order"] for ts in SNAPS]
        assert all(1.5 < p < 2.5 for p in po)
