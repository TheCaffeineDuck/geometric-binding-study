"""Readers and per-run metrics for the WP-C analysis (outputs/session5/trackC/runs/)."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from swirl.string import diag as G          # noqa: E402

TC = os.path.join(ROOT, "outputs", "session5", "trackC")
RUNS = os.path.join(TC, "runs")
REL = lambda p: os.path.relpath(p, ROOT)
_cache = {}


def path(name, ext="json"):
    return os.path.join(RUNS, "%s.%s" % (name, ext) if not name.endswith("." + ext) else name)


def exists(name):
    return os.path.exists(path(name))


def J(name):
    if name not in _cache:
        _cache[name] = json.load(open(path(name)))
    return _cache[name]


def S(name):
    z = np.load(path(name + "_series", "npz"))
    return {k: z[k] for k in z.files}


def _arr(recs, key):
    return np.array([np.nan if r.get(key) is None else r[key] for r in recs], dtype=float)


def curves(res):
    """Time series of the records: t (record mid), te (record end), cycles at record end, frac fixed
    and co-moving, centroid, widths, rates."""
    recs = res["records"]
    sm = res["summary"]
    te = _arr(recs, "t")
    t = _arr(recs, "t_mid")
    E0 = sm["E_fixed0"]
    rr = _arr(recs, "rate_zc")
    cyc = G.cycles(te, rr, sm["nominal_rate"])
    return {"t": t, "te": te, "cyc": cyc, "ff": _arr(recs, "E_fixed") / E0,
            "fc": _arr(recs, "E_comoving") / E0, "cen": _arr(recs, "centroid"),
            "lh": _arr(recs, "L_half_window"), "rms": _arr(recs, "rms_width_window"),
            "rate": rr, "absorbed": _arr(recs, "absorbed") / sm["E0"],
            "balance": _arr(recs, "balance"), "Ef": _arr(recs, "E_fixed"),
            "jump_max": _arr(recs, "jump_max"), "jump_x": _arr(recs, "jump_x"),
            "jump_cells": _arr(recs, "jump_cells"), "tv": _arr(recs, "stretch_total_variation"),
            "k1": _arr(recs, "k_first_moment_stretch"), "krms": _arr(recs, "k_rms_stretch"),
            "peak_u": _arr(recs, "peak_u_window"), "peak_d": _arr(recs, "peak_d_window")}


def t_at_cycles(c, n):
    """Time at which the (record-end) cycle count reaches n; None if the run ended earlier."""
    if c["cyc"][-1] < n:
        return None
    return float(np.interp(n, np.concatenate([[0.0], c["cyc"]]), np.concatenate([[0.0], c["te"]])))


def val_at_t(c, key, t):
    """Value of a per-record array at time t (interpolated on record mid-times); None outside."""
    if t is None or t > c["t"][-1] or t < c["t"][0]:
        return None
    x, y = c["t"], c[key]
    ok = np.isfinite(y)
    if not ok.any():
        return None
    return float(np.interp(t, x[ok], y[ok]))


def late(c, key, frac=0.1):
    n = max(1, int(len(c["t"]) * frac))
    y = c[key][-n:]
    y = y[np.isfinite(y)]
    return float(np.median(y)) if y.size else None


def early(c, key, frac=0.1):
    n = max(1, int(len(c["t"]) * frac))
    y = c[key][:n]
    y = y[np.isfinite(y)]
    return float(np.median(y)) if y.size else None


def twin_name(name):
    return name + "_twin"


def lifetime_row(name, ncyc_list=(300, 1000)):
    """The standard per-run numbers (Block 2.1 and the task's list). Returns a flat dict."""
    res = J(name)
    sm = res["summary"]
    c = curves(res)
    out = {"name": name, "source": REL(path(name)), "cap_t": sm["cap"], "t_end": sm["t_end"],
           "cycles_at_end": sm["cycles_at_end"], "guard": sm["guard"] is not None,
           "T_90": sm["T_90"], "T_death": sm["T_death"],
           "T_90_cycles": sm["T_90_cycles"], "T_death_cycles": sm["T_death_cycles"],
           "T_90_comoving": sm["T_90_comoving"], "T_death_comoving": sm["T_death_comoving"],
           "rate_early": sm["rate_early"], "rate_late": sm["rate_late"],
           "frac_end": sm["frac_end"], "frac_comoving_end": sm["frac_comoving_end"],
           "frac_min": sm["frac_min"], "nonlinear_share_t0": sm["nonlinear_share_t0"],
           "absorbed_over_E0": float(c["absorbed"][-1]), "max_abs_balance": sm["max_abs_balance"],
           "E0": sm["E0"], "W": sm["W"]}
    if sm["guard"]:
        out["guard_info"] = {k: sm["guard"][k] for k in ("t", "x_bond", "stretch", "energy_last_good")
                             if k in sm["guard"]}
        out["guard_rerun"] = sm["guard"].get("rerun_half_dt", {}).get("label")
    out["L_half_start"] = float(c["lh"][0])
    out["L_half_late"] = late(c, "lh")
    out["rms_start"] = float(c["rms"][0])
    out["rms_late"] = late(c, "rms")
    tw = None
    if exists(twin_name(name)):
        tw = curves(J(twin_name(name)))
    out["twin"] = res.get("twin") and {k: res["twin"][k] for k in
                                       ("frac_end", "frac_comoving_end", "T_90", "T_death", "t_end",
                                        "rate_late")}
    for n in ncyc_list:
        t = t_at_cycles(c, n)
        out["t_%dcyc" % n] = t
        out["frac_fixed_%dcyc" % n] = val_at_t(c, "ff", t)
        out["frac_comoving_%dcyc" % n] = val_at_t(c, "fc", t)
        out["twin_frac_%dcyc" % n] = val_at_t(tw, "ff", t) if tw else None
        for key in ("lh", "rms", "jump_max", "jump_x", "jump_cells", "tv", "k1", "krms", "rate"):
            out["%s_%dcyc" % (key, n)] = val_at_t(c, key, t)
    return out, c, tw


def spectrum_split(t, q, split=1.0):
    """Hann amplitude-squared spectrum of a probe series; fractions of the power below and above
    `split` (rate) and the peak."""
    q = np.asarray(q, dtype=float)
    q = q - q.mean()
    w = np.hanning(q.size)
    Q = np.abs(np.fft.rfft(q * w)) ** 2
    om = 2 * np.pi * np.fft.rfftfreq(q.size, d=float(t[1] - t[0]))
    tot = Q.sum()
    if tot <= 0:
        return None
    return {"power_below": float(Q[om < split].sum() / tot), "power_above": float(Q[om >= split].sum() / tot),
            "total_power": float(tot), "peak_rate": float(om[np.argmax(Q)])}


def lone_oscillator_rate(a):
    """O4: 2 pi / (4 int_0^a dw / sqrt(ln((1 - w^2)/(1 - a^2)))), by w = a sin(th) substitution."""
    th = np.linspace(0, np.pi / 2, 200001)[1:-1]
    w = a * np.sin(th)
    f = a * np.cos(th) / np.sqrt(np.log((1 - w * w) / (1 - a * a)))
    I = np.trapezoid(f, th)
    return float(2 * np.pi / (4 * I))
