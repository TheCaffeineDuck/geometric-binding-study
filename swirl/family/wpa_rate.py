"""WP-A stages 2-3: radiation-rate map along the stable branch and the path-integral
lifetime tau = int dE / (E Gamma(E)).

Rate map, one beta:
- Starts: uniform in alpha^2 = 0.3 beta (1 - omega^2), alpha^2_k = alpha^2_c + 0.001 +
  0.002 k (alpha^2_c at the N minimum), up to min(0.14, the alpha^2 where E/(mN) falls
  to 0.80). The cap 0.14 (thin-wall end 0.169 at every beta) keeps the near-thin-wall
  bubbles (hw 15-40 above 0.15) out: they would set the common lump radius for all runs,
  and at beta = 4/3 starts above alpha^2 = 0.13 die within ~400 (Session 1). One refinement pass: around every interior local minimum of the first-window
  rate that is below half of both neighbours, four more starts at +-1/3 and +-2/3 of the
  spacing.
- Each start: radial run (wpa_run.run) to T_MAP = 2000 or death, and a linear twin to 300.
- E coordinate: E_lump = E(r < R_lump), R_lump common to all starts at this beta,
  R_lump = max over starts of max(3 hw, hw + 4.6/kappa) (encloses the tail to ~1e-4).
  The coordinate is the MEASURED energy of the evolved lump, never the profile label
  (F-004 note iii). The measured core omega of each window is stored next to it.
- Transient excluded: t < t_s = 150 + 2 R_lump (shed radiation with v > 0.5 has left
  R_lump). Windows of W = 300 from t_s, up to the earlier of the run end and the first
  time E_lump < 0.5 E_lump(t_s). Gamma = -slope of ln E_lump (least squares), sigma =
  slope standard error (one independent sample per period).

Assembly of Gamma(E) from the windows of all runs at one beta:
- A window is "below floor" if Gamma < 2 sigma; its central value is set to 2 sigma
  (so the lifetime from it is a lower bound) and it has no lower bound.
- Bins of width 0.2% in ln E. Node per non-empty bin: ln E = median, ln Gamma_c =
  median of the central values, ln Gamma_hi = max of ln(Gamma + 2 sigma), ln Gamma_lo =
  min of ln(Gamma - 2 sigma) (-inf if any window in the bin is below floor or
  Gamma - 2 sigma <= 0). Scatter between runs at the same E therefore sets the band.
- Gamma(E) between nodes: linear interpolation of ln Gamma in ln E; constant beyond the
  outermost nodes (flagged as extrapolation). Dips are represented only where some run
  sampled them; a dip that falls between nodes is bridged by the interpolation and its
  lifetime contribution is lost (the integral then underestimates tau); the largest node
  gap inside each integration range is reported for this reason.
- tau(E_a -> E_b) = int_{ln E_b}^{ln E_a} dx / Gamma(x), integrated exactly for the
  piecewise-exponential interpolant. Band: tau_lo uses Gamma_hi, tau_hi uses Gamma_lo
  (None when a node in range has no lower bound).

Lifetimes predicted for a start: T = t_s + tau(E_s -> E_end), E_s = E_lump(t_s).
T_90: E_end = 0.90 E0 rho0, rho0 = E_lump(0) / E_3hw(0) (converts the frac threshold,
defined in r < 3 hw, to the lump coordinate). T_death: E_end = max(0.5 E_s, E_min of the
family) (the half-energy rule taken at t_s instead of t = 50, and never below energetic
death).
"""

import json
import os

import numpy as np

from swirl.family.wpa_family import OUT, family_path
from swirl.family.wpa_run import (G4, grid_for, profile, r_lump, run, window_fits,
                                  write_json)

CACHE = os.path.join(OUT, "checkpoints", "maps")
T_MAP, T_TWIN_MAP, W = 2000.0, 300.0, 300.0
D_A2, A2_HI, EOMN_FLOOR = 0.002, 0.14, 0.80
BIN_REL = 0.002
FLOOR_SIGMAS = 2.0


# ---------------------------------------------------------------- the map grid

def load_family(beta):
    p = family_path(beta)
    if p is None:
        raise FileNotFoundError("no family file for beta=%.4f; run the families stage" % beta)
    with open(p) as f:
        return json.load(f)


def branch(beta):
    """alpha^2 range of the map and per-row interpolants (hw, E/mN) from the family file."""
    fam = load_family(beta)
    f = fam["features_rescaled"]
    rows = sorted(fam["rows_rescaled"], key=lambda r: r["omega"])
    w = np.array([r["omega"] for r in rows])
    a2 = 0.3 * beta * (1 - w**2)
    eomn = np.array([r["E_over_mN"] for r in rows])
    hw = np.array([r["half_width"] for r in rows])
    a2c = 0.3 * beta * (1 - f["omega_minN"] ** 2)
    deep = np.where((a2 > a2c) & (eomn < EOMN_FLOOR))[0]
    a2_hi = A2_HI if not deep.size else min(A2_HI, float(a2[deep].min()))
    o = np.argsort(a2)
    return dict(a2c=float(a2c), a2_hi=float(a2_hi), omega_c=f["omega_minN"],
                E_min_setB=fam["features_setB_units"]["E_min"],
                hw_of_a2=lambda x: np.interp(x, a2[o], hw[o]),
                eomn_of_a2=lambda x: np.interp(x, a2[o], eomn[o]))


def a2_to_w(beta, a2):
    return float(np.sqrt(1.0 - a2 / (0.3 * beta)))


def coarse_grid(beta):
    b = branch(beta)
    n = int(np.floor((b["a2_hi"] - (b["a2c"] + 0.001)) / D_A2 + 1e-9)) + 1
    a2s = b["a2c"] + 0.001 + D_A2 * np.arange(n)
    return [float(x) for x in a2s], b


def map_settings(beta):
    a2s, b = coarse_grid(beta)
    hw = b["hw_of_a2"](np.array(a2s))
    kap = np.sqrt(np.array(a2s) / (0.3 * beta))
    R = max(r_lump(h, k) for h, k in zip(hw, kap))
    return dict(beta=beta, R_lump=float(R), t_s=float(150.0 + 2.0 * R), W=W, T_map=T_MAP,
                T_twin=T_TWIN_MAP, d_alpha2=D_A2, alpha2_c=b["a2c"], alpha2_hi=b["a2_hi"],
                omega_c=b["omega_c"], E_min_setB=b["E_min_setB"], grid=grid_for(beta),
                coarse_alpha2=a2s)


# ---------------------------------------------------------------- one map run

def map_run(beta, a2, st, T=None, use_cache=True):
    w0 = a2_to_w(beta, a2)
    T = st["T_map"] if T is None else T
    cpath = os.path.join(CACHE, "map_beta%.4f_a2%.6f_T%g_R%.3f.json" % (beta, a2, T, st["R_lump"]))
    if use_cache and os.path.exists(cpath):
        with open(cpath) as f:
            return json.load(f)
    p = profile(beta, w0)
    rcs = [3.0 * p["half_width"], st["R_lump"]]
    r = run(beta, w0, T, grid=st["grid"], rcs=rcs, prof=p)
    tw = run(beta, w0, st["T_twin"], linear=True, grid=st["grid"], rcs=rcs, prof=p,
             stop_at_death=False)
    t = r["series"]["t"]
    El = r["series"]["E_in"][:, 1]
    E3 = r["series"]["E_in"][:, 0]
    i_s = int(np.argmin(np.abs(t - st["t_s"])))
    E_s = float(El[i_s]) if t[-1] >= st["t_s"] else None
    if E_s is not None:
        dead = np.where((t > st["t_s"]) & (El < 0.5 * E_s))[0]
        t_stop = float(t[dead[0]]) if dead.size else float(t[-1])
        wins = window_fits(t, El, st["t_s"], st["W"], t_stop, r["omega_t"], r["omega"])
    else:
        t_stop, wins = None, []
    ft = tw["series"]["E_in"][:, 0] / tw["E0"]
    ttw = tw["series"]["t"]
    keep = slice(None, None, 20)
    out = dict(
        beta=beta, alpha2=a2, omega_profile=w0, half_width=p["half_width"], kappa=p["kappa"],
        E_over_mN=p["E_over_mN"], E_profile_setB=p["E"] / G4, rcs=rcs,
        E0=r["E0"], E_lump0=float(El[0]), rho0=float(El[0] / E3[0]),
        E_s=E_s, s_frac=None if E_s is None else E_s / r["E0"],
        shedding_50=float(1.0 - E3[int(np.argmin(np.abs(t - 50)))] / E3[0]),
        t_stop_windows=t_stop, windows=wins,
        Gamma_first=wins[0]["Gamma"] if wins else None,
        sigma_first=wins[0]["sigma"] if wins else None,
        T_end=r["T_end"], T_90=r["T_90"], T_death=r["T_death"], regime=r["regime"],
        frac_t0=r["frac_t0"], frac_end=r["frac_end"],
        twin=dict(frac_T100=float(ft[int(np.argmin(np.abs(ttw - 100)))]),
                  frac_T300=float(ft[-1]), E0=tw["E0"]),
        series=dict(t=t[keep], E_lump_over_E0=El[keep] / r["E0"], frac=E3[keep] / r["E0"]),
        omega_series=dict(t=r["omega_t"][::10], omega=r["omega"][::10]),
        params=r["params"])
    write_json(cpath, out)
    return _reload(cpath)      # same types whether fresh or cached


def _reload(p):
    with open(p) as f:
        return json.load(f)


def _gamma_eff(x):
    if x["Gamma_first"] is not None:
        return max(x["Gamma_first"], FLOOR_SIGMAS * x["sigma_first"], 1e-14)
    if x["T_death"]:
        return 1.0 / x["T_death"]
    return np.inf


def refine_alpha2(runs, d=D_A2):
    """Starts added around local minima of the first-window rate (dips)."""
    rs = sorted(runs, key=lambda x: x["alpha2"])
    g = np.array([_gamma_eff(x) for x in rs])
    a2 = np.array([x["alpha2"] for x in rs])
    new = []
    for i in range(1, len(rs) - 1):
        if g[i] < g[i - 1] and g[i] < g[i + 1] and g[i] < 0.5 * min(g[i - 1], g[i + 1]):
            new += [a2[i] + f * d for f in (-2 / 3, -1 / 3, 1 / 3, 2 / 3)]
    return [float(x) for x in new]


# ---------------------------------------------------------------- assembly and integral

def points_from_runs(runs):
    pts = []
    for x in runs:
        for wnd in x["windows"]:
            pts.append(dict(E=wnd["E"], Gamma=wnd["Gamma"], sigma=wnd["sigma"],
                            omega=wnd["omega"], alpha2=x["alpha2"]))
    return pts


def assemble(points, bin_rel=BIN_REL):
    """Nodes of ln Gamma(ln E): central, hi, lo. See module docstring."""
    E = np.array([p["E"] for p in points])
    G = np.array([p["Gamma"] for p in points])
    s = np.array([p["sigma"] for p in points])
    fl = np.maximum(FLOOR_SIGMAS * s, 1e-300)
    below = G < fl
    Gc = np.where(below, fl, G)
    Ghi = np.maximum(G + FLOOR_SIGMAS * s, fl)
    Glo = G - FLOOR_SIGMAS * s
    x = np.log(E)
    k = np.floor(x / bin_rel).astype(np.int64)
    nodes = []
    for kk in np.unique(k):
        sel = k == kk
        lo_ok = (not below[sel].any()) and (Glo[sel] > 0).all()
        nodes.append(dict(x=float(np.median(x[sel])), lc=float(np.median(np.log(Gc[sel]))),
                          lhi=float(np.max(np.log(Ghi[sel]))),
                          llo=float(np.min(np.log(Glo[sel]))) if lo_ok else -np.inf,
                          n=int(sel.sum()), below_floor=int(below[sel].sum())))
    nodes.sort(key=lambda n: n["x"])
    return nodes


def _lgamma(nodes, which, xq):
    xs = np.array([n["x"] for n in nodes])
    ls = np.array([n[which] for n in nodes])
    return np.interp(xq, xs, ls)


def tau_exact(xs, ls, xa, xb):
    """int_{xb}^{xa} exp(-l(x)) dx for l piecewise linear through (xs, ls), constant beyond."""
    if xa <= xb:
        return 0.0
    if np.any(~np.isfinite(ls)):
        return None
    bp = np.concatenate([[xb], xs[(xs > xb) & (xs < xa)], [xa]])
    lb = np.interp(bp, xs, ls)
    tot = 0.0
    for i in range(bp.size - 1):
        dx = bp[i + 1] - bp[i]
        l0, l1 = lb[i], lb[i + 1]
        dl = l1 - l0
        if abs(dl) < 1e-10:
            tot += dx * np.exp(-0.5 * (l0 + l1))
        else:
            tot += dx * (np.exp(-l0) - np.exp(-l1)) / dl
    return float(tot)


def path_integral(nodes, E_a, E_b):
    """tau from E_a down to E_b: central, lo (from Gamma_hi), hi (from Gamma_lo), flags."""
    xs = np.array([n["x"] for n in nodes])
    xa, xb = np.log(E_a), np.log(E_b)
    # nodes the interpolant uses on [xb, xa]: those inside plus one bracketing node each side
    lo_i = int(np.searchsorted(xs, xb, side="right")) - 1
    hi_i = int(np.searchsorted(xs, xa, side="left"))
    used = np.arange(max(lo_i, 0), min(hi_i, xs.size - 1) + 1)
    out = {}
    for name, which in (("tau", "lc"), ("tau_lo", "lhi"), ("tau_hi", "llo")):
        ls = np.array([n[which] for n in nodes])
        if not np.all(np.isfinite(ls[used])):
            out[name] = None
        else:
            out[name] = tau_exact(xs[used], ls[used], xa, xb)
    inr = (xs > xb) & (xs < xa)
    gaps = np.diff(np.concatenate([[xb], xs[inr], [xa]])) if xa > xb else np.array([0.0])
    out.update(
        n_nodes_in_range=int(inr.sum()),
        largest_gap_rel=float(np.max(gaps)) if gaps.size else None,
        extrapolated_low=bool(xb < xs.min() - BIN_REL), extrapolated_high=bool(xa > xs.max() + BIN_REL),
        floor_nodes_in_range=int(sum(n["below_floor"] > 0 for n in nodes
                                     if xb - BIN_REL <= n["x"] <= xa + BIN_REL)))
    out["is_lower_bound"] = out["floor_nodes_in_range"] > 0 or out["extrapolated_low"]
    return out


def predict(nodes, t_s, E_s, E0, rho0, E_min):
    e90 = 0.90 * E0 * rho0
    ed = max(0.5 * E_s, E_min)
    p90 = path_integral(nodes, E_s, e90)
    pd = path_integral(nodes, E_s, ed)
    add = lambda v: None if v is None else t_s + v
    return dict(
        T_90=add(p90["tau"]), T_90_lo=add(p90["tau_lo"]), T_90_hi=add(p90["tau_hi"]),
        T_death=add(pd["tau"]), T_death_lo=add(pd["tau_lo"]), T_death_hi=add(pd["tau_hi"]),
        E_s=E_s, E_end_90=e90, E_end_death=ed, death_end_is_E_min=bool(E_min > 0.5 * E_s),
        flags_90={k: v for k, v in p90.items() if not k.startswith("tau")},
        flags_death={k: v for k, v in pd.items() if not k.startswith("tau")})


# ---------------------------------------------------------------- map file and ranking

def map_path(beta):
    return os.path.join(OUT, "ratemap_beta%.4f.json" % beta)


def write_map(beta, st, runs):
    runs = sorted(runs, key=lambda x: x["alpha2"])
    nodes = assemble(points_from_runs(runs))
    out = dict(
        description=__doc__, beta=beta, units="Set-B-like: m=1, g4=0.30, g6=0.09 beta",
        settings=st, n_runs=len(runs), nodes=nodes,
        runs=[{k: v for k, v in x.items()} for x in runs])
    write_json(map_path(beta), out)
    return out


def rank(beta, m=None):
    if m is None:
        with open(map_path(beta)) as f:
            m = json.load(f)
    st, nodes = m["settings"], m["nodes"]
    rows = []
    for x in m["runs"]:
        if x["E_s"] is None:
            pr = dict(T_90=x["T_90"], T_death=x["T_death"], note="died before t_s")
        else:
            pr = predict(nodes, st["t_s"], x["E_s"], x["E0"], x["rho0"], st["E_min_setB"])
        rows.append(dict(alpha2=x["alpha2"], omega_profile=x["omega_profile"],
                         omega_settled=x["regime"]["omega_over_m_settled"],
                         Gamma_first=x["Gamma_first"], sigma_first=x["sigma_first"],
                         Gamma_min_windows=min([max(w["Gamma"], FLOOR_SIGMAS * w["sigma"]) for w in x["windows"]],
                                               default=None),
                         T_death_own=x["T_death"], T_90_own=x["T_90"], T_end_own=x["T_end"],
                         **pr))
        r_ = rows[-1]
        r_["T_90_before_t_s"] = bool(x["E_s"] is not None and r_.get("E_end_90") is not None
                                     and x["E_s"] <= r_["E_end_90"])
        if r_["T_90_before_t_s"]:
            r_["T_90"] = x["T_90"]       # frac fell below 0.90 during the initial shedding
    key = lambda r: (r["T_death"] if r["T_death"] is not None else -1.0)
    best = max(rows, key=key)
    alive = [r for r in rows if r["Gamma_min_windows"] is not None]
    by_rate = min(alive, key=lambda r: r["Gamma_min_windows"]) if alive else best
    return dict(beta=beta, best=best, best_by_measured_rate=by_rate,
                note="best: largest path-integral T_death. best_by_measured_rate: lowest measured "
                     "windowed Gamma over the map run (floored at 2 sigma), no interpolation. The long stage runs both.",
                ranked=sorted(rows, key=key, reverse=True))
