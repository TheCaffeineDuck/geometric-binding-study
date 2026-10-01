"""WP-D: arrangement and phase-pattern scan in the point-lump model, then point-model time evolution.

One command reruns everything (minutes) with the pair-law numbers in the params file:

    /Users/aaronramos/Desktop/SWIRL/.venv/bin/python -m swirl.arrange.scan \
        --params outputs/session2/wpD/params_default.json --out outputs/session2/wpD

Writes (atomically) scan_results.json, candidates.json, predictions_point_model.json, fsup_table.json
(reused if present and built from the same profile; --rebuild-fsup forces a rebuild).
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.optimize import brentq

from . import geometry as geo
from . import model as mdl
from .pairlaw import ROOT, build_fsup_table, load_params, make_laws

KR_MIN, KR_MAX, NR = 0.05, 12.0, 1500
SPACINGS = [12.0, 16.0, 20.0]
FAR_SPACING = 30.0
CENTRE_STEP_DEG = 30
FLAG_PLAN_HW, FLAG_TAIL_HW = 2.5, 3.6
T_RUN = 1000.0
PERT_X, PERT_TH, SEED = 1e-3, 1e-3, 20261002
N_DYN_CANDIDATES = 6
EV_TOL, LAM_TOL, F_TOL = 1e-8, 1e-6, 1e-8


def dump(obj, path):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    os.replace(path + ".tmp", path)


def min_dist(X):
    i, j = np.triu_indices(len(X), 1)
    return float(np.min(np.linalg.norm(X[i] - X[j], axis=1)))


# ------------------------------------------------------------------ pattern enumeration
def binary_label(X, bits):
    i, j = np.triu_indices(len(X), 1)
    d = np.linalg.norm(X[i] - X[j], axis=1)
    nn = np.isclose(d, d.min(), atol=1e-8)
    anti = bits[i] != bits[j]
    s = "0/180 bits=%s nn_anti=%d/%d" % ("".join(map(str, bits)), int(np.sum(anti & nn)), int(np.sum(nn)))
    if geo.has_centre(X):
        s += " centre_anti_to=%d/%d" % (int(np.sum(bits[:-1] != bits[-1])), len(X) - 1)
    return s


def enumerate_patterns(X, G):
    n = len(X)
    centre = geo.has_centre(X)
    pats = []
    for bits, size in geo.binary_classes(G, n):
        pats.append({"kind": "binary", "theta": np.pi * bits.astype(float), "orbit_size": size,
                     "label": binary_label(X, bits), "aliases": []})
    burn = geo.burnside_count(G, n)
    shell = X[:-1] if centre else X
    Gs = geo.point_group(shell)
    axes = geo.symmetry_axes(shell, Gs)
    buckets = {}
    for ax in axes:
        sp, capped = geo.stepped_patterns(shell, ax["axis"], ax["order"])
        for m, th in sp:
            tag = "stepped %s-axis(order %d) m=%d" % (ax["type"], ax["order"], m)
            if geo.is_binary(th):
                full = np.r_[th, 0.0] if centre else th
                for q in pats:
                    if q["kind"] == "binary" and geo.equivalent(G, full, q["theta"]):
                        q["aliases"].append(tag + (" centre=0" if centre else ""))
                        break
                continue
            for c in (range(0, 360, CENTRE_STEP_DEG) if centre else [None]):
                full = np.r_[th, np.deg2rad(c)] if centre else th.copy()
                key = geo.invariant(X, full)
                dup = next((q for q in buckets.get(key, []) if geo.equivalent(G, full, q["theta"])), None)
                lab = tag + ("" if c is None else " centre=%d" % c)
                if dup is not None:
                    dup["aliases"].append(lab)
                    continue
                q = {"kind": "stepped", "theta": full, "label": lab, "axis_type": ax["type"],
                     "axis_order": ax["order"], "m": m, "centre_deg": c, "m_capped": capped, "aliases": []}
                buckets.setdefault(key, []).append(q)
                pats.append(q)
    return pats, burn, axes


# ------------------------------------------------------------------ static analysis
def dUdR_grid(X, th, law, R, with_scale=False):
    i, j = np.triu_indices(len(X), 1)
    s = np.linalg.norm(X[i] - X[j], axis=1)
    D = th[i] - th[j]
    dd = np.outer(R, s)
    ud = law.pair(dd, np.broadcast_to(D, dd.shape))[1]
    if with_scale:
        return (ud * s).sum(1), (np.abs(ud) * s).sum(1)
    return (ud * s).sum(1)


def d2UdR2(X, th, law, R):
    i, j = np.triu_indices(len(X), 1)
    s = np.linalg.norm(X[i] - X[j], axis=1)
    udd = law.pair(R * s, th[i] - th[j])[2]
    return float(np.sum(udd * s * s))


def force_scale(Xb, th, law):
    i, j, d, rh, u, ud, *_ = mdl.pair_terms(Xb, th, law)
    sc = np.zeros(len(Xb))
    np.add.at(sc, i, np.abs(ud))
    np.add.at(sc, j, np.abs(ud))
    return float(sc.max())


def find_balances(X, th, law, kappa):
    smin = min_dist(X)
    Rlo = max(KR_MIN / kappa, (law.d_min + 1e-9) / smin)
    R = np.geomspace(Rlo, KR_MAX / kappa, NR)
    g, sc = dUdR_grid(X, th, law, R, with_scale=True)
    if not np.all(np.isfinite(g)):
        return [], []
    # U(R) can be identically constant when the pair energies cancel (e.g. tetrahedron 1 vs 3): no stationary
    # point carries information then; values below 1e-10 of the term scale are treated as zero
    g = np.where(np.abs(g) < 1e-10 * sc, 0.0, g)
    if not np.any(g):
        return [], [{"U_independent_of_R": True}]
    out, stationary = [], []
    f = lambda r: dUdR_grid(X, th, law, np.array([r]))[0]
    nz = np.nonzero(g)[0]
    for a, b in zip(nz[:-1], nz[1:]):
        if np.sign(g[a]) * np.sign(g[b]) >= 0:
            continue
        k = a if b == a + 1 else None
        if k is None:
            stationary.append({"R": float(R[a]), "kappa_R": float(R[a] * kappa), "flat_interval_to_kappa_R": float(R[b] * kappa)})
            continue
        Rb = brentq(f, R[k], R[k + 1], xtol=1e-14, rtol=1e-14)
        Xb = Rb * X
        res = float(np.max(np.linalg.norm(mdl.forces(Xb, th, law), axis=1)) / force_scale(Xb, th, law))
        (out if res < F_TOL else stationary).append({"R": Rb, "kappa_R": Rb * kappa, "force_residual_rel": res})
    return out, stationary


def classify_vec(dx, dph, kappa):
    a = kappa ** 2 * np.sum(np.abs(dx) ** 2)
    b = np.sum(np.abs(dph) ** 2)
    share = float(a / (a + b))
    return share, ("positional" if share > 0.9 else "phase/N" if share < 0.1 else "mixed")


def analyze_balance(X, th, Rb, law, p):
    kappa, E0, w, hw, om = p["kappa"], p["E"], p["domega_dN"], p["hw"], p["omega"]
    n = len(X)
    Xb = Rb * X
    Ui = mdl.single_energies(Xb, th, law)
    H = mdl.hessian(Xb, th, law)
    Hx = H[:3 * n, :3 * n]
    ev, V = np.linalg.eigh(Hx)
    sc = np.max(np.abs(ev))
    neg, zero = ev < -EV_TOL * sc, np.abs(ev) <= EV_TOL * sc
    rig = mdl.rigid_modes(Xb)
    sv = np.linalg.svd(rig.T @ V[:, zero], compute_uv=False) if np.any(zero) else np.array([])
    rig_rq = float(np.max(np.abs(rig.T @ Hx @ rig)) / sc)
    # N16: trace of each lump's own block = kappa^2 U_i (exact for Model A)
    tr = np.array([np.trace(Hx[3 * a:3 * a + 3, 3 * a:3 * a + 3]) for a in range(n)])
    n16 = np.abs(tr - kappa ** 2 * Ui)
    n16_rel = float(np.max(n16) / max(np.max(np.abs(kappa ** 2 * Ui)), 1e-300))
    blk_min = np.array([np.linalg.eigvalsh(Hx[3 * a:3 * a + 3, 3 * a:3 * a + 3]).min() for a in range(n)])
    n16_logic_viol = int(np.sum((Ui < 0) & (blk_min >= 0)))
    nd = mdl.ndot(Xb, th, law)
    nscale = force_scale(Xb, th, law)
    i, j, d, rh, u, *_ = mdl.pair_terms(Xb, th, law)
    nfix = bool(np.max(np.abs(nd)) < 1e-9 * max(np.max(np.abs(u)), 1e-300))
    out = {"R": Rb, "kappa_R": Rb * kappa, "nn_spacing": min_dist(Xb), "nn_spacing_hw": min_dist(Xb) / hw,
           "flag_below_2p5hw": min_dist(Xb) / hw < FLAG_PLAN_HW, "flag_below_3p6hw": min_dist(Xb) / hw < FLAG_TAIL_HW,
           "U_total": float(np.sum(u)), "breathing_d2U_dR2": d2UdR2(X, th, law, Rb),
           "radial_restoring": d2UdR2(X, th, law, Rb) > 0,
           "U_i": Ui, "U_i_min": float(Ui.min()), "any_U_i_negative": bool(np.any(Ui < 0)),
           "N16_trace_minus_k2Ui_rel": n16_rel, "N16_lumps_Ui_neg_but_block_stable": n16_logic_viol,
           "hessian_eigs": ev, "n_negative": int(neg.sum()), "n_zero": int(zero.sum()), "n_modes": 3 * n,
           "n_zero_rigid": int(np.sum(sv > 0.999)), "rigid_rayleigh_rel": rig_rq,
           "N_flow_max": float(np.max(np.abs(nd))), "N_fixed_point": nfix}
    Jv = mdl.angular_momentum_overlap(Xb, th, p["A"], p["kappa"], om)
    out["J_overlap"] = Jv
    out["J_over_nN0"] = float(np.linalg.norm(Jv) / (n * p["N"]))
    if getattr(law, "dOdd", None) is not None and min_dist(Xb) >= 3.0:
        Jt = mdl.angular_momentum_overlap(Xb, th, p["A"], p["kappa"], om, dOdd=law.dOdd)
        out["J_overlap_true_profile"] = Jt
        out["J_true_over_nN0"] = float(np.linalg.norm(Jt) / (n * p["N"]))
    nn = np.isclose(d, d.min(), atol=1e-9)
    out["nn_link_N_current_max"] = float(np.max(np.abs(2 * np.pi * p["A"] ** 2 * np.exp(-kappa * d[nn]) / d[nn]
                                                       * np.sin(th[i[nn]] - th[j[nn]]))))
    # frozen-phase position rates and phase-only rates
    out["pos_rate_max"] = float(np.sqrt(max(-ev.min(), 0) / E0))
    Htt = H[3 * n:, 3 * n:]
    mu = np.linalg.eigvalsh(Htt)
    out["phase_only_rate_max"] = float(np.sqrt(max(abs(w) * mu.max(), 0)))
    if nfix:
        J = mdl.jacobian(Xb, th, law, E0, w)
        lam, W = np.linalg.eig(J)
        ls = np.max(np.abs(lam))
        uns = lam.real > LAM_TOL * ls
        out["n_unstable_full"] = int(uns.sum())
        if uns.any():
            k = int(np.argmax(lam.real))
            share, kind = classify_vec(W[:3 * n, k], W[6 * n:7 * n, k], kappa)
            out.update({"growth_max": float(lam[k].real), "growth_max_imag": float(abs(lam[k].imag)),
                        "efold_time": float(1 / lam[k].real),
                        "efold_periods": float(1 / lam[k].real / (2 * np.pi / om)),
                        "fastest_mode": kind, "fastest_mode_position_share": share,
                        "unstable_kinds": sorted(classify_vec(W[:3 * n, q], W[6 * n:7 * n, q], kappa)[1]
                                                 for q in np.nonzero(uns)[0])})
        else:
            out.update({"growth_max": 0.0, "efold_time": None, "efold_periods": None, "fastest_mode": None})
        out["growth_rates_unstable"] = np.sort(lam.real[uns])[::-1]
    return out


def pattern_summary(X, th, G, laws, p):
    lawA = laws["A"]
    sym = geo.pattern_symmetry(G, th)
    radial, dims = geo.radial_by_symmetry(X, sym)
    mirror = geo.phase_exchanging_mirror(G, th, X)
    smin = min_dist(X)
    at = {}
    nfix = True
    for s in SPACINGS:
        Xb = (s / smin) * X
        F = mdl.forces(Xb, th, lawA)
        rhat = np.where(np.linalg.norm(Xb, axis=1)[:, None] > 0, Xb / np.maximum(np.linalg.norm(Xb, axis=1), 1e-300)[:, None], 0)
        Fr = np.sum(F * rhat, axis=1)
        Fnr = np.linalg.norm(F - Fr[:, None] * rhat, axis=1)
        nd = mdl.ndot(Xb, th, lawA)
        u = mdl.pair_terms(Xb, th, lawA)[4]
        nfix &= bool(np.max(np.abs(nd)) < 1e-9 * np.max(np.abs(u)))
        Jv = mdl.angular_momentum_overlap(Xb, th, p["A"], p["kappa"], p["omega"])
        Jt = mdl.angular_momentum_overlap(Xb, th, p["A"], p["kappa"], p["omega"], dOdd=lawA.dOdd)
        at[str(int(s))] = {"R": s / smin, "F_radial": Fr, "F_nonradial_max": float(Fnr.max()),
                           "F_max": float(np.linalg.norm(F, axis=1).max()), "N_flow": nd,
                           "N_flow_max": float(np.max(np.abs(nd))), "J_overlap": Jv,
                           "J_over_nN0": float(np.linalg.norm(Jv) / (len(X) * p["N"])),
                           "J_overlap_true_profile": Jt,
                           "J_true_over_nN0": float(np.linalg.norm(Jt) / (len(X) * p["N"]))}
    out = {"symmetry_order": len(sym), "radial_by_symmetry": radial, "fixed_dims": dims,
           "phase_exchanging_mirror_normal": mirror, "N15_control": mirror is not None,
           "N_fixed_point": nfix, "carries_J": bool(at["16"]["J_over_nN0"] > 1e-12), "at_spacing": at,
           "balances": {}, "R_stationary_not_balance": {}}
    for mname, law in laws.items():
        bal, stat = find_balances(X, th, law, p["kappa"])
        out["balances"][mname] = [dict(b, **analyze_balance(X, th, b["R"], law, p)) for b in bal]
        out["R_stationary_not_balance"][mname] = stat
    return out


def run_scan(p, laws):
    arrs = geo.arrangements()
    res = {}
    for name, X in arrs.items():
        G = geo.point_group(X)
        pats, burn, axes = enumerate_patterns(X, G)
        nbin = sum(q["kind"] == "binary" for q in pats)
        rows = []
        for q in pats:
            s = pattern_summary(X, q["theta"], G, laws, p)
            rows.append(dict({k: v for k, v in q.items() if k != "theta"},
                             theta_deg=np.round(np.rad2deg(q["theta"]), 6), **s))
        res[name] = {"n_lumps": len(X), "unit_coords": X, "group_order": len(G),
                     "group_order_rotations": sum(g[2] == 1 for g in G),
                     "binary_classes_bruteforce": nbin, "binary_classes_burnside": burn,
                     "axes": [{"type": a["type"], "order": a["order"], "n_equivalent": a["count"],
                               "axis": a["axis"]} for a in axes],
                     "n_stepped_patterns": len(pats) - nbin, "patterns": rows}
        print("%-9s group %3d  0/180 classes %3d (Burnside %g)  stepped %3d  balances A %d B %d" % (
            name, len(G), nbin, burn, len(pats) - nbin,
            sum(len(r["balances"]["A"]) for r in rows), sum(len(r["balances"]["B"]) for r in rows)), flush=True)
    return res


# ------------------------------------------------------------------ item 4: Claude Desktop numbers, normalized units
def claude_desktop_check():
    """C = 1, kappa = 1 (Model A is scale-free: balance kappa R does not depend on A or kappa)."""
    from .pairlaw import PairLaw
    law = PairLaw(1 / np.sqrt(2 * np.pi), 1.0, "A")
    p1 = {"A": 1 / np.sqrt(2 * np.pi), "kappa": 1.0, "E": 1.0, "N": 1.0, "domega_dN": -1.0, "hw": 1.0, "omega": 1.0}
    cases = {"ring5_step72": (geo.ring(5), 2 * np.pi * np.arange(5) / 5),
             "ring6_step60": (geo.ring(6), 2 * np.pi * np.arange(6) / 6),
             "octa_inphase_shell_antiphase_centre": (geo.with_centre(geo.octahedron()), np.r_[np.zeros(6), np.pi])}
    out = {}
    for name, (X, th) in cases.items():
        bal, _ = find_balances(X, th, law, 1.0)
        rows = []
        for b in bal:
            a = analyze_balance(X, th, b["R"], law, p1)
            Xb = b["R"] * X
            Hb = mdl.hessian(Xb, th, law)[:3, :3]
            x = Xb[0] / np.linalg.norm(Xb[0])
            t = np.cross([0, 0, 1.0], x)
            t = t / np.linalg.norm(t)
            z = np.cross(x, t)
            rows.append({"kappa_R": b["kappa_R"], "radial_restoring": a["radial_restoring"],
                         "breathing_d2U_dR2_total": a["breathing_d2U_dR2"], "U_i_lump0": float(a["U_i"][0]),
                         "U_i_centre": float(a["U_i"][-1]) if geo.has_centre(X) else None,
                         "n_negative": a["n_negative"], "n_modes": a["n_modes"], "n_zero": a["n_zero"],
                         "n_zero_rigid": a["n_zero_rigid"],
                         "lump0_curvature_radial": float(x @ Hb @ x), "lump0_curvature_tangential": float(t @ Hb @ t),
                         "lump0_curvature_third": float(z @ Hb @ z)})
        out[name] = rows
    return out


# ------------------------------------------------------------------ candidates and controls
RANK_FIELDS = ("kappa_R", "R", "nn_spacing", "nn_spacing_hw", "flag_below_2p5hw", "flag_below_3p6hw",
               "radial_restoring", "breathing_d2U_dR2", "n_negative", "n_modes", "n_zero_rigid", "U_i_min",
               "any_U_i_negative", "N16_trace_minus_k2Ui_rel", "n_unstable_full", "growth_max", "efold_time",
               "efold_periods", "fastest_mode", "unstable_kinds")


def rank_balances(res, hw):
    rows = []
    for arr, r in res.items():
        for qi, q in enumerate(r["patterns"]):
            for m in ("A", "B"):
                for b in q["balances"][m]:
                    row = {"model": m, "arrangement": arr, "pattern_index": qi, "pattern": q["label"],
                           "aliases": q["aliases"], "theta_deg": q["theta_deg"], "N15_control": q["N15_control"],
                           "N_fixed_point": b["N_fixed_point"], "carries_J": q["carries_J"]}
                    row.update({k: b.get(k) for k in RANK_FIELDS})
                    row["J_true_over_nN0"] = b.get("J_true_over_nN0")
                    row["efold_below_one_period"] = bool(b.get("efold_periods") is not None and b["efold_periods"] < 1)
                    th = np.deg2rad(np.asarray(q["theta_deg"], float))
                    ctl = controls_for(th)
                    row["controls"] = {
                        "far_apart": "same lumps and phases at nearest-neighbour spacing %g (%.2f hw)" % (
                            FAR_SPACING, FAR_SPACING / hw),
                        "plain_0_180": np.rad2deg(ctl["plain_0_180_rounded"]) if "plain_0_180_rounded" in ctl
                        else "pattern is already 0/180; use all in phase and far apart",
                        "all_in_phase": np.zeros(len(th))}
                    rows.append(row)
    key = lambda r: (r["N15_control"], not r["N_fixed_point"], r["n_unstable_full"] or 0,
                     -(r["efold_time"] or np.inf), r["flag_below_2p5hw"], r["flag_below_3p6hw"])
    rows.sort(key=key)
    cnt = {"A": 0, "B": 0}
    for k, r in enumerate(rows):
        r["rank"] = k + 1
        cnt[r["model"]] += 1
        r["rank_within_model"] = cnt[r["model"]]
    return rows


def literature_patterns():
    """Stepped rings with literature support (PRIOR_ART S8 items 8, 11): N >= 4m, step <= 90 degrees."""
    return [("ring%d" % K, m) for K, m in ((4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (8, 2))]


def stepped_ring_theta(K, m):
    return np.mod(2 * np.pi * m * np.arange(K) / K, 2 * np.pi)


def controls_for(theta):
    out = {"all_in_phase": np.zeros_like(theta)}
    if not geo.is_binary(theta):
        out["plain_0_180_rounded"] = np.where(np.cos(theta) >= -1e-12, 0.0, np.pi)
    return out


# ------------------------------------------------------------------ point-model time evolution
def run_one(X1, th, R, law, p, model_name):
    n = len(X1)
    rng = np.random.default_rng(SEED)
    X0 = R * X1 + PERT_X * rng.standard_normal((n, 3))
    th0 = th + PERT_TH * rng.standard_normal(n)
    d0 = min_dist(R * X1)
    inside = d0 <= 2 * p["hw"]
    d_stop = 0.5 * d0 if inside else 2 * p["hw"]
    if model_name == "B":
        d_stop = max(d_stop, law.d_min + 1e-6)
    t_eval = np.linspace(0, T_RUN, 2001)
    sol = mdl.integrate(X0, th0, law, p["E"], p["domega_dN"], T_RUN, d_stop=d_stop, rtol=1e-12, atol=1e-12,
                        t_eval=t_eval)
    Y = sol.y
    i, j = np.triu_indices(n, 1)
    Xs = Y[:3 * n].T.reshape(-1, n, 3)
    D = np.linalg.norm(Xs[:, i] - Xs[:, j], axis=2)
    Dfull = np.full((len(sol.t), n, n), np.inf)
    Dfull[:, i, j] = D
    Dfull[:, j, i] = D
    nnd = Dfull.min(2)
    ratio = nnd / nnd[0]
    nu = Y[7 * n:].T
    yend = sol.y_events[0][0] if sol.status == 1 else Y[:, -1]
    H0 = mdl.hamiltonian(Y[:, 0], n, law, p["E"], p["domega_dN"])
    H1 = mdl.hamiltonian(yend, n, law, p["E"], p["domega_dN"])
    # scale: sum of |pair energies| at start and at the end state (U itself can cancel to zero, e.g. 0,0,180,0)
    Uscale = max(np.sum(np.abs(mdl.pair_terms(R * X1, th, law)[4])),
                 np.sum(np.abs(mdl.pair_terms(yend[:3 * n].reshape(n, 3), yend[6 * n:7 * n], law)[4])))
    nbad = np.nonzero(np.max(np.abs(nu), 1) > 0.1 * p["N"])[0]
    dph = Y[6 * n:7 * n, -1] - th0
    out = {"model": model_name, "R": R, "nn_spacing0": d0, "nn_spacing0_hw": d0 / p["hw"],
           "starts_inside_2hw": bool(inside), "stop_distance": d_stop,
           "t_end": float(sol.t_events[0][0]) if sol.status == 1 else float(sol.t[-1]),
           "min_pair_distance": float(D.min()), "min_pair_distance_hw": float(D.min() / p["hw"]),
           "nn_ratio_last_sample_min": float(ratio[-1].min()), "nn_ratio_last_sample_max": float(ratio[-1].max()),
           "N_swing_max_over_N": float(np.max(np.abs(nu)) / p["N"]),
           "N_model_invalid_from_t": float(sol.t[nbad[0]]) if nbad.size else None,
           "phase_pattern_drift_deg": float(np.rad2deg(np.max(np.abs(np.angle(np.exp(1j * (dph - dph[0]))))))),
           "H_drift_rel": float(abs(H1 - H0) / Uscale)}
    if sol.status == 1:
        out["outcome"] = "collapse" if inside else "merger"
        out["outcome_time"] = float(sol.t_events[0][0])
    else:
        esc = ratio[-1] > 1.5
        if esc.all():
            out["outcome"] = "flies apart"
            out["outcome_time"] = float(sol.t[np.argmax(ratio.min(1) > 1.5)])
        elif esc.any():
            out["outcome"] = "partly dispersed"
            out["outcome_time"] = float(sol.t[np.argmax(ratio.max(1) > 1.5)])
        else:
            out["outcome"] = "bounded"
            out["outcome_time"] = None
        out["pair_distance_rel_change_max"] = float(np.max(np.abs(D / D[0] - 1)))
        out["mean_pair_distance_change_rel"] = float(D[-1].mean() / D[0].mean() - 1)
    return out


def run_dynamics(res, ranked, p, laws):
    arrs = geo.arrangements()
    todo = []

    def add(arr, th, role, parent, bal):
        G = geo.point_group(arrs[arr])
        for k in todo:
            if k["arrangement"] == arr and geo.equivalent(G, th, k["theta"]):
                k["roles"].append((role, parent))
                for m, Rs in bal.items():
                    k["balance_R"].setdefault(m, Rs)
                return
        todo.append({"arrangement": arr, "theta": th, "roles": [(role, parent)], "balance_R": dict(bal)})

    def q_theta(q):
        return np.deg2rad(np.asarray(q["theta_deg"], float))

    # dynamics set: the top N_DYN_CANDIDATES distinct patterns, plus every pattern with a Model A balance, plus
    # every balance with spacing at or above the plan's 2.5 hw limit
    seen, picked = set(), []
    for r in ranked:
        if r["N15_control"] or not r["N_fixed_point"]:
            continue
        key = (r["arrangement"], r["pattern_index"])
        if key in seen:
            continue
        if len(picked) < N_DYN_CANDIDATES or r["model"] == "A" or not r["flag_below_2p5hw"]:
            seen.add(key)
            picked.append(r)
    for r in picked:
        q = res[r["arrangement"]]["patterns"][r["pattern_index"]]
        bal = {m: [b["R"] for b in q["balances"][m]] for m in ("A", "B") if q["balances"][m]}
        name = "%s | %s" % (r["arrangement"], r["pattern"])
        add(r["arrangement"], q_theta(q), "candidate (rank %d)" % r["rank"], name, bal)
        for cn, cth in controls_for(q_theta(q)).items():
            add(r["arrangement"], cth, "control: " + cn, name, {})
    for arr, m in literature_patterns():
        th = stepped_ring_theta(int(arr[4:]), m)
        G = geo.point_group(arrs[arr])
        q = next((q for q in res[arr]["patterns"] if geo.equivalent(G, th, q_theta(q))), None)
        bal = {mm: [b["R"] for b in q["balances"][mm]] for mm in ("A", "B") if q is not None and q["balances"][mm]}
        name = "%s stepped m=%d (literature)" % (arr, m)
        add(arr, th, "literature candidate", name, bal)
        for cn, cth in controls_for(th).items():
            add(arr, cth, "control: " + cn, name, {})
    for r in ranked:
        if r["N15_control"]:
            q = res[r["arrangement"]]["patterns"][r["pattern_index"]]
            add(r["arrangement"], q_theta(q), "N15 mirror control", "%s | %s" % (r["arrangement"], r["pattern"]),
                {m: [b["R"] for b in q["balances"][m]] for m in ("A", "B") if q["balances"][m]})
    runs = []
    t0 = time.time()
    for k in todo:
        X1 = arrs[k["arrangement"]]
        smin = min_dist(X1)
        sizes = [("spacing %g" % s, s / smin) for s in SPACINGS + [FAR_SPACING]]
        entry = {"arrangement": k["arrangement"], "theta_deg": np.round(np.rad2deg(k["theta"]), 6),
                 "roles": [{"role": a, "for": b} for a, b in k["roles"]], "runs": []}
        for mname, law in laws.items():
            sz = list(sizes)
            if mname in k["balance_R"]:
                sz = [("balance kappaR=%.4f (model %s)" % (R * p["kappa"], mname), R)
                      for R in k["balance_R"][mname]] + sz
            for lab, R in sz:
                rr = run_one(X1, k["theta"], R, law, p, mname)
                rr["size"] = lab
                entry["runs"].append(rr)
        runs.append(entry)
        print("dyn %-9s %-26s %s (%.0f s)" % (
            k["arrangement"], k["roles"][0][0][:26],
            " ".join("%s/%s:%s" % (r["model"], r["size"].replace("spacing ", "")[:14], r["outcome"][:5])
                     for r in entry["runs"]), time.time() - t0), flush=True)
    return runs


def short(label):
    return (label.replace("0/180 bits=", "").replace("stepped ", "").replace("centre_anti_to", "c-anti")
            .replace("nn_anti", "nn-anti").replace("-axis(order ", "(").replace(")", ")"))


def tables(res, ranked, runs, p):
    L = ["## Class counts", "", "| arrangement | lumps | group order (rotations) | 0/180 classes, brute force | Burnside "
         "| stepped patterns (distinct) | radial by symmetry | N-flow fixed points | carry J | N15 mirror |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for a, r in res.items():
        pats = r["patterns"]
        L.append("| %s | %d | %d (%d) | %d | %g | %d | %d | %d | %d | %d |" % (
            a, r["n_lumps"], r["group_order"], r["group_order_rotations"], r["binary_classes_bruteforce"],
            r["binary_classes_burnside"], r["n_stepped_patterns"], sum(q["radial_by_symmetry"] for q in pats),
            sum(q["N_fixed_point"] for q in pats), sum(q["carries_J"] for q in pats),
            sum(q["N15_control"] for q in pats)))
    L += ["", "## Balances, ranked", "", "| rank | model | arrangement | pattern | kappa R | R | spacing | /hw | "
          "<2.5hw | <3.6hw | d2U/dR2 | restoring | neg / modes | U_i min | unstable eigs (full) | fastest rate | "
          "e-fold (t) | e-fold (periods) | fastest kind | J/(n N0) asym (true profile) | N15 control |",
          "|" + "---|" * 21]
    for r in ranked:
        b = next(b for b in res[r["arrangement"]]["patterns"][r["pattern_index"]]["balances"][r["model"]]
                 if abs(b["R"] - r["R"]) < 1e-9)
        L.append("| %d | %s | %s | %s | %.4f | %.3f | %.3f | %.3f | %s | %s | %.3g | %s | %d / %d | %.3g | %s | %.3g | "
                 "%.3g | %.3g | %s | %.2g (%s) | %s |" % (
                     r["rank"], r["model"], r["arrangement"], short(r["pattern"]), r["kappa_R"], r["R"],
                     r["nn_spacing"], r["nn_spacing_hw"], "yes" if r["flag_below_2p5hw"] else "no",
                     "yes" if r["flag_below_3p6hw"] else "no", r["breathing_d2U_dR2"],
                     "yes" if r["radial_restoring"] else "no", r["n_negative"], r["n_modes"], r["U_i_min"],
                     r["n_unstable_full"], r["growth_max"], r["efold_time"], r["efold_periods"], r["fastest_mode"],
                     b["J_over_nN0"], "%.2g" % b["J_true_over_nN0"] if "J_true_over_nN0" in b else "-",
                     "yes" if r["N15_control"] else "no"))
    if runs is not None:
        L += ["", "## Point-model time evolution, T = %g" % T_RUN, "",
              "| arrangement | phases (deg) | role | model | balance | spacing 12 | 16 | 20 | 30 |", "|---|---|---|---|---|---|---|---|---|"]
        def fmt(x):
            s = x["outcome"] + ("" if x["outcome_time"] is None else " %.0f" % x["outcome_time"])
            if x["outcome"] == "bounded":
                s += " (%+.1f%%)" % (100 * x["mean_pair_distance_change_rel"])
            if x["N_model_invalid_from_t"] is not None:
                s += " [N>10%% from %.0f]" % x["N_model_invalid_from_t"]
            return s
        for e in runs:
            for m in ("A", "B"):
                rr = [x for x in e["runs"] if x["model"] == m]
                bal = "; ".join("%s at %.2f hw" % (fmt(x), x["nn_spacing0_hw"]) for x in rr if x["size"].startswith("balance"))
                sp = {x["size"]: fmt(x) for x in rr if x["size"].startswith("spacing")}
                L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    e["arrangement"], ",".join("%g" % t for t in e["theta_deg"]), e["roles"][0]["role"], m,
                    bal or "-", sp["spacing 12"], sp["spacing 16"], sp["spacing 20"], sp["spacing 30"]))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join(ROOT, "outputs/session2/wpD/params_default.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/session2/wpD"))
    ap.add_argument("--rebuild-fsup", action="store_true")
    ap.add_argument("--no-dynamics", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    p = load_params(a.params)
    fpath = os.path.join(a.out, "fsup_table.json")
    fsup = None
    if os.path.exists(fpath) and not a.rebuild_fsup:
        with open(fpath) as f:
            fsup = json.load(f)
    if fsup is None or "overlap_true" not in fsup:
        dump(build_fsup_table(), fpath)
        with open(fpath) as f:
            fsup = json.load(f)
    laws = make_laws(p, fsup)
    t0 = time.time()
    run_params = {"params_file": os.path.relpath(os.path.abspath(a.params), ROOT), "pair_law": p,
                  "fsup_table": os.path.relpath(fpath, ROOT), "fsup_profile": fsup["profile"],
                  "kappaR_scan": [KR_MIN, KR_MAX, NR], "spacings": SPACINGS, "far_spacing": FAR_SPACING,
                  "centre_phase_step_deg": CENTRE_STEP_DEG, "flags_hw": [FLAG_PLAN_HW, FLAG_TAIL_HW],
                  "T_run": T_RUN, "perturbation": {"x_sigma": PERT_X, "theta_sigma_rad": PERT_TH, "N": 0.0,
                                                   "p": 0.0, "seed": SEED},
                  "tolerances": {"hessian_zero_rel": EV_TOL, "growth_rel": LAM_TOL, "force_balance_rel": F_TOL},
                  "model_B_note": "Model B = Model A(params) + (F_sup - F_asym) of the profile in fsup_table; "
                                  "incomplete below about d = 12 (core polarization missing), undefined below d = 3"}
    res = run_scan(p, laws)
    cd = claude_desktop_check()
    ranked = rank_balances(res, p["hw"])
    dump({"params": run_params, "claude_desktop_check_normalized": cd, "arrangements": res},
         os.path.join(a.out, "scan_results.json"))
    lit = []
    for arr, m in literature_patterns():
        th = stepped_ring_theta(int(arr[4:]), m)
        G = geo.point_group(geo.arrangements()[arr])
        q = next(q for q in res[arr]["patterns"] if geo.equivalent(G, th, np.deg2rad(np.asarray(q["theta_deg"], float))))
        lit.append({"arrangement": arr, "m": m, "step_deg": 360.0 * m / int(arr[4:]),
                    "balance_kappa_R_model_A": [b["kappa_R"] for b in q["balances"]["A"]],
                    "balance_kappa_R_model_B": [b["kappa_R"] for b in q["balances"]["B"]],
                    "source": "PRIOR_ART.md S8 items 8 and 11 (Desyatnikov & Kivshar 2002; Crasovan et al. 2003)"})
    dump({"params": run_params, "ranked_balances": ranked, "literature_stepped_rings": lit,
          "note": "Ranked by: N15 mirror controls last, N-flow fixed point, number of unstable eigenvalues of the "
                  "full linearization (positions, phases, N), slowest fastest growth, spacing flags. Nothing in "
                  "the scan is stable in the point model; every balance has unstable modes."},
         os.path.join(a.out, "candidates.json"))
    print("static scan %.0f s, %d balances" % (time.time() - t0, len(ranked)), flush=True)
    runs = None if a.no_dynamics else run_dynamics(res, ranked, p, laws)
    tpath = os.path.join(a.out, "tables.md")
    with open(tpath + ".tmp", "w") as f:
        f.write("# WP-D tables (generated by swirl.arrange.scan from %s)\n\n" % run_params["params_file"])
        f.write(tables(res, ranked, runs, p))
    os.replace(tpath + ".tmp", tpath)
    if runs is None:
        return
    dump({"params": run_params,
          "outcome_definitions": {
              "merger": "some pair below 2 hw (start above 2 hw)",
              "collapse": "start already at or below 2 hw: some pair below half the starting spacing "
                          "(model B: or below d = 3, where its table ends)",
              "flies apart": "every lump's nearest-neighbour distance above 1.5 times its start value at T = 1000; "
                             "time is when the last lump crossed 1.5",
              "partly dispersed": "some but not all lumps above 1.5 at T = 1000; time is the first crossing",
              "bounded": "none of the above by T = 1000 (mean_pair_distance_change_rel gives the drift; at spacing 30 "
                         "the forces are too weak for anything to resolve in T = 1000)",
              "N_model_invalid_from_t": "first time some |N_i - N0| > 0.1 N0, beyond which the linear omega(N) is "
                                        "not valid"},
          "runs": runs}, os.path.join(a.out, "predictions_point_model.json"))
    print("total %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
