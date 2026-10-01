"""Run driver for engine v2: config dict -> evolution -> result JSON.

Usage
-----
    from swirl.engine.run import run
    result = run(config, out_path="outputs/x.json", checkpoint_dir="checkpoints/x")

    python -m swirl.engine.run config.json --out result.json --checkpoint-dir dir

Config keys (defaults in DEFAULTS; `init` is required)
------------------------------------------------------
    N, L                 grid points per side, box side (x in [-L/2, L/2))
    m, g4, g6            couplings (Set B: 1, 0.30, 0.055)
    dt                   time step (default dx/8)
    t_end                final time
    init                 {"radial": [...], "gaussians": [...]}, see initial.py
    absorber             null/false (off), true (SPONGE_DEFAULT), or a dict overriding
                         {"gamma", "width", "power", "r_out"} (evolver.sponge_profile)
    kfilter              null (default, off), or {"nu", "power"}: damping rate
                         nu (|k|/k_max)^power on phi_t, k_max = sqrt(3) pi/dx
    center               diagnostic centre [x, y, z]
    r_c                  radii for E_loc (energy inside |x - center| <= r_c)
    sample_every         time between diagnostic samples
    point_every          steps between samples of phi(center) (frequency series)
    avg_window           longest span for the phi_t^2 average and N; the window used
                         is the largest whole number of measured periods (zero
                         crossings of phi(center, t)) that fits inside it
    checkpoint_every     time between checkpoints (if checkpoint_dir is given)
    threads              FFT workers and numba threads

Result JSON: {"params", "completed", "steps_done", "t", "wall_seconds",
"steps_per_second", "series": {...}, "point": {...}, "frequency": {...},
"summary": {...}}. Fields are never written to the result; checkpoints hold
them in a single .npz written atomically.
"""

import argparse
import json
import os
import time

import numpy as np

from swirl.engine.evolver import Evolver, SPONGE_DEFAULT
from swirl.engine.initial import build_initial, radial_checks

DEFAULTS = {
    "N": 64, "L": 50.0, "m": 1.0, "g4": 0.30, "g6": 0.055,
    "dt": None, "t_end": 100.0, "init": None,
    "absorber": None, "kfilter": None,
    "center": [0.0, 0.0, 0.0], "r_c": [5.0, 10.0, 15.0],
    "sample_every": 1.0, "point_every": 1, "avg_window": 20.0,
    "checkpoint_every": 50.0, "threads": 4,
}

SERIES_KEYS = ("t", "E_total", "E_kin", "E_grad", "E_mass", "E_quartic", "E_sextic",
               "nonlinear_share", "absorbed", "E_plus_absorbed", "max_abs_phi",
               "I_kin", "C_kin", "E_loc")


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def atomic_write_json(obj, path):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def full_config(config):
    cfg = dict(DEFAULTS)
    cfg.update(_jsonable(config))
    if cfg["init"] is None:
        raise ValueError("config['init'] is required")
    if cfg["dt"] is None:
        cfg["dt"] = cfg["L"] / cfg["N"] / 8.0
    cfg["r_c"] = sorted(float(r) for r in cfg["r_c"])
    if cfg["absorber"] is True:
        cfg["absorber"] = dict(SPONGE_DEFAULT)
    elif cfg["absorber"]:
        cfg["absorber"] = {**SPONGE_DEFAULT, **cfg["absorber"]}
    return cfg


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------

def _min_image_r(coords, L, center):
    d = []
    for c, a in zip(center, range(3)):
        x = coords - c
        x = x - L * np.round(x / L)
        shp = [1, 1, 1]
        shp[a] = -1
        d.append(x.reshape(shp))
    return np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)


def _trilinear(coords, L, dx, center):
    """Indices and weights to read phi at `center` by trilinear interpolation."""
    N = coords.size
    idx, wts = [], []
    f = [((c + L / 2) / dx) % N for c in center]
    i0 = [int(np.floor(v)) for v in f]
    fr = [v - i for v, i in zip(f, i0)]
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                idx.append(((i0[0] + a) % N, (i0[1] + b) % N, (i0[2] + c) % N))
                wts.append((fr[0] if a else 1 - fr[0]) * (fr[1] if b else 1 - fr[1])
                           * (fr[2] if c else 1 - fr[2]))
    return idx, wts


def zero_crossings(t, y):
    """Zero crossing times of a sampled signal (cubic refinement on 4 points)."""
    t = np.asarray(t)
    y = np.asarray(y)
    s = np.signbit(y)
    out = []
    for i in np.nonzero(s[1:] != s[:-1])[0]:
        lo, hi = max(i - 1, 0), min(i + 3, t.size)
        if hi - lo < 4:
            lo, hi = (0, 4) if lo == 0 else (t.size - 4, t.size)
        tt, yy = t[lo:hi], y[lo:hi]
        c = np.polyfit(tt - t[i], yy, 3)
        tc = t[i] - y[i] * (t[i + 1] - t[i]) / (y[i + 1] - y[i])
        for _ in range(8):
            x = tc - t[i]
            f = np.polyval(c, x)
            fp = np.polyval(np.polyder(c), x)
            if fp == 0:
                break
            tc -= f / fp
        if not (t[i] <= tc <= t[i + 1]):
            tc = t[i] - y[i] * (t[i + 1] - t[i]) / (y[i + 1] - y[i])
        out.append(float(tc))
    return np.array(out)


def frequency_from_point(t, y):
    """omega from full periods between alternate zero crossings."""
    tz = zero_crossings(t, y)
    if tz.size < 3:
        return {"t_mid": [], "omega": [], "zero_crossings": tz.tolist()}
    per = tz[2:] - tz[:-2]
    return {"t_mid": (0.5 * (tz[2:] + tz[:-2])).tolist(),
            "omega": (2 * np.pi / per).tolist(),
            "zero_crossings": tz.tolist()}


def _hermite(pt, C, I, z):
    """C at time z by cubic Hermite interpolation (C' = I) on the step grid."""
    i = min(max(int(np.searchsorted(pt, z, side="right")) - 1, 0), pt.size - 2)
    h = pt[i + 1] - pt[i]
    s = (z - pt[i]) / h
    return ((2 * s ** 3 - 3 * s ** 2 + 1) * C[i] + (s ** 3 - 2 * s ** 2 + s) * h * I[i]
            + (-2 * s ** 3 + 3 * s ** 2) * C[i + 1] + (s ** 3 - s ** 2) * h * I[i + 1])


def window_averages(ts, tz, pt, pC, pI, W):
    """Whole-period averages ending at each sample time t_j.

    The window runs between zero crossings z_a < z_b of phi(center, t) inside
    [t_j - W, t_j], with an even number of half periods between them, so it
    spans an integer number of measured periods. Returns, per sample,
    <int phi_t^2 d3x> = (C(z_b) - C(z_a)) / (z_b - z_a), the window-mean omega
    = pi * (half periods) / (z_b - z_a), and N = <...> / omega. C is the
    per-step trapezoid integral of int phi_t^2 d3x, read between steps by
    cubic Hermite interpolation with C' = I. None where fewer than two
    half periods fit.
    """
    pt, pC, pI = np.asarray(pt), np.asarray(pC), np.asarray(pI)
    avg, om, Nad = [], [], []
    for t in ts:
        i0 = int(np.searchsorted(tz, t - W - 1e-12, side="left"))
        i1 = int(np.searchsorted(tz, t + 1e-12, side="right")) - 1
        if (i1 - i0) % 2:
            i1 -= 1
        if i1 - i0 < 2:
            avg.append(None), om.append(None), Nad.append(None)
            continue
        za, zb = tz[i0], tz[i1]
        a = (_hermite(pt, pC, pI, zb) - _hermite(pt, pC, pI, za)) / (zb - za)
        w = np.pi * (i1 - i0) / (zb - za)
        avg.append(float(a))
        om.append(float(w))
        Nad.append(float(a / w))
    return avg, om, Nad


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _sample(ev, series, shell, n_rc, t):
    parts = ev.energy_parts()
    for k in ("E_total", "E_kin", "E_grad", "E_mass", "E_quartic", "E_sextic", "max_abs_phi"):
        series[k].append(float(parts[k]))
    absq = abs(parts["E_quartic"]) + abs(parts["E_sextic"])
    denom = parts["E_kin"] + parts["E_grad"] + parts["E_mass"] + absq
    series["nonlinear_share"].append(float(absq / denom) if denom > 0 else 0.0)
    series["t"].append(float(t))
    series["absorbed"].append(float(ev.absorbed))
    series["E_plus_absorbed"].append(float(parts["E_total"] + ev.absorbed))
    series["I_kin"].append(float(ev.I_kin))
    series["C_kin"].append(float(ev.C_kin))
    e = ev.energy_density()
    shells = np.bincount(shell, weights=e.ravel(), minlength=n_rc + 1)[:n_rc]
    series["E_loc"].append((np.cumsum(shells) * ev.dV).tolist())


def _save_checkpoint(ckpt_dir, ev, cfg, series, point, wall):
    os.makedirs(ckpt_dir, exist_ok=True)
    meta = {"params": cfg, "step": ev.step_count, "absorbed": ev.absorbed,
            "C_kin": ev.C_kin, "I_kin": ev.I_kin, "series": series,
            "point": point, "wall_seconds": wall}
    path = os.path.join(ckpt_dir, "checkpoint.npz")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, phi=ev.phi, pi=ev.pi, meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)


def _load_checkpoint(ckpt_dir):
    path = os.path.join(ckpt_dir, "checkpoint.npz")
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        return z["phi"].copy(), z["pi"].copy(), json.loads(str(z["meta"]))


def run(config, out_path=None, checkpoint_dir=None, max_steps=None, verbose=False):
    """Evolve `config`; resume from `checkpoint_dir` if a checkpoint is there.

    max_steps: stop after this many steps in this call (a checkpoint is written
    first when checkpoint_dir is set); the result then has completed=False.
    """
    cfg = full_config(config)
    ev = Evolver(cfg["N"], cfg["L"], cfg["m"], cfg["g4"], cfg["g6"],
                 absorber=cfg["absorber"], kfilter=cfg["kfilter"], threads=cfg["threads"])
    dt = cfg["dt"]
    n_total = int(round(cfg["t_end"] / dt))
    s_every = max(1, int(round(cfg["sample_every"] / dt)))
    p_every = max(1, int(cfg["point_every"]))
    c_every = max(1, int(round(cfg["checkpoint_every"] / dt)))

    r = _min_image_r(ev.coords, ev.L, cfg["center"])
    shell = np.searchsorted(np.array(cfg["r_c"]), r.ravel(), side="left").astype(np.int16)
    n_rc = len(cfg["r_c"])
    pidx, pw = _trilinear(ev.coords, ev.L, ev.dx, cfg["center"])

    def point_value():
        return float(sum(w * ev.phi[i] for i, w in zip(pidx, pw)))

    ck = _load_checkpoint(checkpoint_dir) if checkpoint_dir else None
    if ck is not None:
        phi, pi, meta = ck
        if meta["params"] != cfg:
            raise ValueError("checkpoint params differ from config")
        ev.phi[...] = phi
        ev.pi[...] = pi
        ev.step_count, ev.absorbed = meta["step"], meta["absorbed"]
        ev.C_kin, ev.I_kin = meta["C_kin"], meta["I_kin"]
        series, point, wall_prior = meta["series"], meta["point"], meta["wall_seconds"]
        resumed = True
    else:
        phi, pi = build_initial(ev.coords, ev.L, cfg["init"])
        ev.set_state(phi, pi)
        series = {k: [] for k in SERIES_KEYS}
        point = {"t": [], "phi": [], "C": [], "I": []}
        wall_prior = 0.0
        resumed = False

    start = ev.step_count
    stop = n_total if max_steps is None else min(n_total, start + int(max_steps))
    t0 = time.perf_counter()
    n = start
    while True:
        if not (resumed and n == start):
            if n % s_every == 0 or n == n_total:
                _sample(ev, series, shell, n_rc, n * dt)
            if n % p_every == 0:
                point["t"].append(n * dt)
                point["phi"].append(point_value())
                point["C"].append(ev.C_kin)
                point["I"].append(ev.I_kin)
            if checkpoint_dir and n % c_every == 0 and start < n < n_total:
                _save_checkpoint(checkpoint_dir, ev, cfg, series, point,
                                 wall_prior + time.perf_counter() - t0)
                if verbose:
                    print("step %d/%d t=%.2f E=%.10g absorbed=%.4g" % (
                        n, n_total, n * dt, series["E_total"][-1], ev.absorbed), flush=True)
        if n >= stop:
            break
        ev.step(dt)
        n += 1

    wall = time.perf_counter() - t0
    completed = n >= n_total
    if checkpoint_dir and not completed:
        _save_checkpoint(checkpoint_dir, ev, cfg, series, point, wall_prior + wall)

    freq = frequency_from_point(point["t"], point["phi"])
    tz = np.array(freq["zero_crossings"])
    avg, om_w, Nad = window_averages(series["t"], tz, point["t"], point["C"], point["I"],
                                     cfg["avg_window"])
    out_series = dict(series)
    out_series["I_kin_window_avg"] = avg
    out_series["omega_window"] = om_w
    out_series["N"] = Nad
    E = np.array(series["E_total"])
    Epa = np.array(series["E_plus_absorbed"])
    summary = {
        "E0": float(E[0]), "E_end": float(E[-1]),
        "absorbed_end": series["absorbed"][-1],
        "max_rel_dev_E_plus_absorbed": float(np.max(np.abs(Epa - Epa[0])) / abs(Epa[0])),
        "nonlinear_share_0": series["nonlinear_share"][0],
        "omega_first": freq["omega"][0] if freq["omega"] else None,
        "omega_last": freq["omega"][-1] if freq["omega"] else None,
        "omega_over_m_last": (freq["omega"][-1] / cfg["m"]) if freq["omega"] else None,
    }
    result = {
        "engine": "swirl.engine v2",
        "params": cfg,
        "completed": completed,
        "steps_done": n,
        "t": n * dt,
        "wall_seconds": wall_prior + wall,
        "steps_per_second_this_call": (n - start) / wall if wall > 0 else None,
        "series": out_series,
        "point": point,
        "frequency": {"t_mid": freq["t_mid"], "omega": freq["omega"]},
        "summary": summary,
        "init_checks": [radial_checks(p, cfg["L"]) for p in cfg["init"].get("radial", [])],
    }
    if out_path:
        atomic_write_json(result, out_path)
    result["_evolver"] = ev
    return result


def main():
    ap = argparse.ArgumentParser(description="Engine v2 run: config JSON -> result JSON")
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)
    res = run(cfg, out_path=a.out, checkpoint_dir=a.checkpoint_dir,
              max_steps=a.max_steps, verbose=a.verbose)
    print(json.dumps(res["summary"]), flush=True)


if __name__ == "__main__":
    main()
