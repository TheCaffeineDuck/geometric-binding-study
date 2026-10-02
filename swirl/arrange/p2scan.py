"""WP-P P2 scan driver (built in P1; the production scan is not run here).

(i)  Static starts with unequal lumps: pairs across N ratio, phase difference and spacing; rings of 3 to 6 with
     one big lump, alternating sizes or graded sizes, in phase / stepped / alternating phases.
(ii) Orbiting pairs and rotating rings: speed as a fraction of the reference circular speed, spacing 3.6 to 8 hw,
     phase pattern in phase / stepped (rotating with or against the phase wave) / alternating, equal lumps and
     unequal pairs and alternating-size rings.

Outcome per run: "merged" (a pair within hw_i + hw_j, the contact rule), "flew apart" (every lump's nearest-neighbour
distance above 1.5 times its start, or every pair beyond max(3 d0, d0 + 40)), "left the family" (a lump's omega
left the fitted range), or "bounded for the run"; with the event time and the closest approach. Runs that come
within 3.6 hw (1.8 (hw_i + hw_j)) at any time are flagged: the tail law is off by more than 10% there.
Every run also gets the corrected bonded rule (swirl/engine/bonded.py, unchanged) on its point-model series, with
its far-apart static control (d = 45) and the in-phase static contact time of equal lumps at the same K and d as
the interaction unit; and the variants of clause 5 proposed in docs/WPP_BUILD_NOTES.md (effective steps, phase lock),
reported next to the rule, not in place of it.

    /Users/aaronramos/Desktop/SWIRL/.venv/bin/python -m swirl.arrange.p2scan --list           # size and cost
    /Users/aaronramos/Desktop/SWIRL/.venv/bin/python -m swirl.arrange.p2scan --smoke 24       # smoke test
    /Users/aaronramos/Desktop/SWIRL/.venv/bin/python -m swirl.arrange.p2scan --workers 2      # the scan (P2)
Results: one JSON per run under --out (atomic; existing results are skipped, so the scan resumes).
"""
import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

from . import unequal as un  # noqa: E402

N0 = 2865.545211117668
HW0 = 3.344038183621855
T_MAX = 10000.0
D_FAR = 45.0           # far-apart control: 13.5 hw (in-phase contact well beyond T_MAX)
N_SAMPLES = 400
RTOL = 1e-10
PERT_X, PERT_TH, PERT_SEED = 1e-3, 1e-3, 20261002   # Session 2 WP-D values; seed + run number
OUT = os.path.join(un.ROOT, "outputs", "session3", "wpP", "p2")
SPACINGS_STATIC_HW = [3.6, 4.2, 4.8, 6.0, 7.2]
SPACINGS_RING_STATIC_HW = [3.6, 4.8, 6.0]
SPACINGS_ORBIT_HW = [3.6, 4.2, 4.8, 6.0, 7.2, 8.0]
RATIOS_PAIR = [1.0, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 2.95]
RATIOS_RING = [1.1, 1.5, 2.0, 2.95]
SPEEDS = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]

_FAM = None
_CACHE = {}


def fam():
    global _FAM
    if _FAM is None:
        _FAM = un.Family.from_table()
    return _FAM


# ------------------------------------------------------------------ the run list
def phases(K, pattern):
    if pattern == "in_phase":
        return np.zeros(K)
    if pattern == "stepped":
        return 2 * np.pi * np.arange(K) / K
    if pattern == "alternating":
        return np.pi * (np.arange(K) % 2)
    raise ValueError(pattern)


def sizes(K, kind, ratio):
    if kind == "equal":
        return [1.0] * K
    if kind == "one_big":
        return [ratio] + [1.0] * (K - 1)
    if kind == "alternating":
        return [ratio if k % 2 == 0 else 1.0 for k in range(K)]
    if kind == "graded":
        return list(np.linspace(1.0, ratio, K))
    raise ValueError(kind)


def spec_list():
    S = []
    # (i) static, unequal
    for s in SPACINGS_STATIC_HW:
        for ratio in RATIOS_PAIR:
            for deg in (0, 45, 90, 135, 180):
                S.append({"family": "static", "K": 2, "spacing_hw": s, "size_kind": "pair", "ratio": ratio,
                          "N_mult": [1.0, ratio], "pattern": "dtheta=%d" % deg, "theta_deg": [0.0, float(deg)],
                          "speed_frac": 0.0, "sense": 0})
    for K in (3, 4, 5, 6):
        kinds = ["one_big", "graded"] + (["alternating"] if K % 2 == 0 else [])
        pats = ["in_phase", "stepped"] + (["alternating"] if K % 2 == 0 else [])
        for s in SPACINGS_RING_STATIC_HW:
            for kind in kinds:
                for ratio in RATIOS_RING:
                    for pat in pats:
                        S.append({"family": "static", "K": K, "spacing_hw": s, "size_kind": kind, "ratio": ratio,
                                  "N_mult": sizes(K, kind, ratio), "pattern": pat,
                                  "theta_deg": np.degrees(phases(K, pat)).tolist(), "speed_frac": 0.0, "sense": 0})
    # (ii) orbits, equal lumps
    combos = [(2, "in_phase", 1), (2, "alternating", 1)]
    for K in (3, 4, 5, 6):
        combos += [(K, "in_phase", 1), (K, "stepped", 1), (K, "stepped", -1)]
        if K % 2 == 0:
            combos.append((K, "alternating", 1))
    for s in SPACINGS_ORBIT_HW:
        for K, pat, sense in combos:
            for f in SPEEDS:
                S.append({"family": "orbit", "K": K, "spacing_hw": s, "size_kind": "equal", "ratio": 1.0,
                          "N_mult": [1.0] * K, "pattern": pat + ("" if pat != "stepped" else
                                                                 (" (rotating with the phase wave)" if sense > 0 else
                                                                  " (rotating against the phase wave)")),
                          "theta_deg": np.degrees(phases(K, pat)).tolist(), "speed_frac": f, "sense": sense})
    # (ii) orbits, unequal pairs and alternating-size rings
    for s in SPACINGS_ORBIT_HW:
        for ratio in (1.1, 1.5, 2.0):
            for deg in (0, 90):
                for f in SPEEDS:
                    S.append({"family": "orbit", "K": 2, "spacing_hw": s, "size_kind": "pair", "ratio": ratio,
                              "N_mult": [1.0, ratio], "pattern": "dtheta=%d" % deg, "theta_deg": [0.0, float(deg)],
                              "speed_frac": f, "sense": 1})
        for K in (4, 6):
            for ratio in (1.5, 2.0):
                for pat, sense in (("in_phase", 1), ("stepped", 1)):
                    for f in SPEEDS:
                        S.append({"family": "orbit", "K": K, "spacing_hw": s, "size_kind": "alternating",
                                  "ratio": ratio, "N_mult": sizes(K, "alternating", ratio),
                                  "pattern": pat + (" (rotating with the phase wave)" if pat == "stepped" else ""),
                                  "theta_deg": np.degrees(phases(K, pat)).tolist(), "speed_frac": f, "sense": sense})
    for k, s in enumerate(S):
        s["id"] = "p2_%04d" % k
    return S


# ------------------------------------------------------------------ setup
def geometry(K, d):
    if K == 2:
        X = np.array([[-d / 2, 0, 0], [d / 2, 0, 0]])
        return X, np.array([[0, -1.0, 0], [0, 1.0, 0]])
    X, R = un.ring_positions(K, d)
    return X, un.tangents(X)


def contact_time_equal_in_phase(K, d):
    """Interaction unit of the bonded rule: in-phase static contact time of equal lumps (N0) at K and d."""
    key = ("tc", K, round(d, 6))
    if key not in _CACHE:
        F = fam()
        W0 = F.omega_of_N(N0)
        m = un.Model(F, W0)
        X, _ = geometry(K, d)
        z = m.pack(X, np.zeros((K, 3)), np.zeros(K), np.full(K, W0))
        sol = m.integrate(z, K, 5 * T_MAX, rtol=RTOL, atol=RTOL)
        _CACHE[key] = float(sol.t_events[0][0]) if sol.status == 1 and len(sol.t_events[0]) else None
    return _CACHE[key]


def reference_speed(spec, Ws):
    """Circular (rigid-rotation) speed of the pattern; if it has no balance, the in-phase one (labelled)."""
    F = fam()
    K, d = spec["K"], spec["spacing_hw"] * HW0
    if K == 2:
        W1, W2 = Ws
        Xu = np.array([[[-0.5, 0, 0], [0.5, 0, 0]]])
        hc = lambda dd: un.pair_energy(Xu * dd, np.zeros((1, 2, 3)), np.zeros((1, 2)), np.array([[W1, W2]]), F)[0]
        Fd = float(hc(d + 1e-30j).imag / 1e-30)                 # in-phase attraction (phases frozen at 0)
        E1, E2 = float(F.E(W1)), float(F.E(W2))
        if Fd <= 0:
            return None
        from scipy.optimize import brentq

        def g(wo):
            p = Fd / wo
            return (p / np.hypot(p, E1) + p / np.hypot(p, E2)) / wo - d
        wo = brentq(g, 1e-9, 1.0, xtol=1e-16, rtol=1e-15)
        p = Fd / wo
        return {"v": [p / np.hypot(p, E1), p / np.hypot(p, E2)], "T_orb": 2 * np.pi / wo,
                "reference": "in-phase circular orbit, phases frozen (relativistic two-body)"}
    W = float(np.mean(Ws))
    th = np.radians(spec["theta_deg"])
    key = ("vc", K, round(d, 6), tuple(np.round(th, 6)), spec["sense"], round(W, 9))
    if key not in _CACHE:
        m = un.Model(F, W)
        r = un.rotating_ring(m, K, d, th, W, spec["sense"] or 1)
        ref = "rigid rotation of this pattern (equal lumps at the mean omega)"
        if r is None:
            r = un.rotating_ring(m, K, d, np.zeros(K), W, spec["sense"] or 1)
            ref = "no balance for this pattern: in-phase rigid rotation of equal lumps at the mean omega"
        _CACHE[key] = None if r is None else {"v": [r["v"]] * K, "T_orb": r["T_orb"], "reference": ref}
    return _CACHE[key]


def setup(spec):
    F = fam()
    K, d = spec["K"], spec["spacing_hw"] * HW0
    Ws = np.array([F.omega_of_N(N0 * x) for x in spec["N_mult"]])
    th = np.radians(spec["theta_deg"])
    X, tg = geometry(K, d)
    # generic perturbation (as in Session 2 WP-D): an exactly symmetric start never leaves its symmetric subspace
    rng = np.random.default_rng(PERT_SEED + int(spec["id"].split("_")[1]))
    X = X + PERT_X * rng.standard_normal(X.shape)
    th = th + PERT_TH * rng.standard_normal(K)
    ref = None
    V = np.zeros((K, 3))
    if spec["family"] == "orbit":
        ref = reference_speed(spec, Ws)
        if ref is None:
            raise RuntimeError("no reference speed")
        V = (spec["sense"] or 1) * spec["speed_frac"] * np.asarray(ref["v"])[:, None] * tg
    m = un.Model(F, float(np.mean(Ws)))
    P = m.momenta_for_velocities(X, V, th, Ws)
    return m, m.pack(X, P, th, Ws), ref, d


# ------------------------------------------------------------------ one run
def edges_of(X):
    i, j = np.triu_indices(len(X), 1)
    D = np.linalg.norm(X[i] - X[j], axis=1)
    return [(int(a), int(b)) for a, b, dd in zip(i, j, D) if dd <= 1.01 * D.min()]


def series(m, sol, n, edges):
    F = m.fam
    Y = sol.y
    Xs = Y[:3 * n].T.reshape(-1, n, 3)
    Ps = Y[3 * n:6 * n].T.reshape(-1, n, 3)
    ph = Y[6 * n:7 * n].T
    Ws = Y[7 * n:].T
    Ns = F.N(Ws)
    Es = F.E(Ws)
    Dm = np.linalg.norm(Xs[:, :, None, :] - Xs[:, None, :, :], axis=-1)
    Dm[:, np.arange(n), np.arange(n)] = np.inf
    nn = Dm.min(2)
    raw, eff = [], []
    for a, b in edges:
        r = Xs[:, b] - Xs[:, a]
        da = Ws[:, a] * np.sum(Ps[:, a] * r, 1) / Es[:, a]
        db = Ws[:, b] * np.sum(Ps[:, b] * r, 1) / Es[:, b]
        raw.append(ph[:, b] - ph[:, a])
        eff.append(ph[:, b] - ph[:, a] + 0.5 * (da + db))
    wrap = lambda x: np.rad2deg(np.angle(np.exp(1j * np.asarray(x))))
    return {"t": sol.t, "X": Xs, "nn": nn, "N": Ns, "W": Ws, "steps_raw_unwrapped": np.array(raw).T,
            "steps_eff_unwrapped": np.array(eff).T, "edges": edges,
            "steps": wrap(np.array(raw).T), "steps_eff": wrap(np.array(eff).T), "dmin": Dm.min((1, 2))}


def run_spec(spec, control=False, T_override=None, model_wrap=None):
    """Integrate one spec. Returns (result dict, series dict). T_override and model_wrap (a function of the
    Model returning an object with .rhs, used for diagnostic variants) are not used by the scan."""
    F = fam()
    m, z0, ref, d = setup(spec)
    if model_wrap is not None:
        m = model_wrap(m, z0)
    K = spec["K"]
    tc = contact_time_equal_in_phase(K, d)
    T = T_MAX if tc is None else min(T_MAX, 5 * tc)
    if ref is not None:
        T = min(T_MAX, max(T, 3 * ref["T_orb"]))
    if control:
        T = spec["T_control"]
    elif T_override is not None:
        T = float(T_override)
    t_eval = np.linspace(0, T, N_SAMPLES + 1)
    far = max(3 * d, d + 40.0)
    i, j = np.triu_indices(K, 1)

    def left(t, z, *_):
        W = z[7 * K:]
        return min(np.min(W - F.W_lo), np.min(F.W_hi - W))
    left.terminal, left.direction = True, -1
    t0 = time.time()
    sol = solve_ivp(m.rhs, [0.0, T], z0, method="DOP853", rtol=RTOL, atol=RTOL, t_eval=t_eval, args=(K,),
                       events=[contact_event(m, K), far_event(K, far), left])
    wall = time.time() - t0
    edges = edges_of(z0[:3 * K].reshape(K, 3))
    S = series(m, sol, K, edges)
    hw = F.hw(z0[7 * K:])
    hwsum = hw[i] + hw[j]
    # outcome
    if sol.status == 1 and len(sol.t_events[0]):
        outcome, t_event = "merged", float(sol.t_events[0][0])
    elif sol.status == 1 and len(sol.t_events[2]):
        outcome, t_event = "left the family", float(sol.t_events[2][0])
    elif (sol.status == 1 and len(sol.t_events[1])) or np.all(S["nn"][-1] > 1.5 * S["nn"][0]):
        outcome = "flew apart"
        crossed = np.all(S["nn"] > 1.5 * S["nn"][0], axis=1)
        t_event = float(S["t"][np.argmax(crossed)]) if crossed.any() else float(sol.t[-1])
    else:
        outcome, t_event = "bounded for the run", None
    # closest approach, validity flag (3.6 hw = 1.8 (hw_i + hw_j))
    Xs = S["X"]
    pd = np.linalg.norm(Xs[:, i] - Xs[:, j], axis=-1)
    ratio = pd / hwsum
    closest = float(np.min(ratio) * 2)                     # in units of hw (mean of the pair)
    res = {"spec": spec, "T": T, "t_end": float(sol.t[-1]), "status": int(sol.status), "outcome": outcome,
           "t_event": t_event, "closest_approach_hw": closest, "flag_below_3p6hw": bool(closest < 3.6),
           "contact_time_unit": tc, "reference_orbit": ref, "nfev": int(sol.nfev), "wall_s": wall,
           "N_change_max_rel": float(np.max(np.abs(S["N"] / S["N"][0] - 1))),
           "omega_start": z0[7 * K:].tolist(), "N_start": F.N(z0[7 * K:]).tolist(), "d0": d,
           "perturbation": {"sigma_x": PERT_X, "sigma_theta_rad": PERT_TH, "seed": PERT_SEED + int(spec["id"].split("_")[1])},
           "model": {"family": "beta 4/3 profiles, outputs/session3/wpP/family_coupling_beta1.3333.json",
                     "relativistic": m.rel, "boost": m.boost, "rtol": RTOL, "coupling_scale": F.coupling_scale}}
    if outcome == "bounded for the run":
        res["mean_pair_distance_change_rel"] = float(np.mean(pd[-1] / pd[0] - 1))
    return res, S


def contact_event(m, K):
    F = m.fam
    i, j = np.triu_indices(K, 1)

    def hit(t, z, *_):
        X = z[:3 * K].reshape(K, 3)
        hw = F.hw(z[7 * K:])
        return np.min(np.linalg.norm(X[i] - X[j], axis=1) - (hw[i] + hw[j]))
    hit.terminal, hit.direction = True, -1
    return hit


def far_event(K, far):
    i, j = np.triu_indices(K, 1)

    def gone(t, z, *_):
        X = z[:3 * K].reshape(K, 3)
        return np.min(np.linalg.norm(X[i] - X[j], axis=1)) - far
    gone.terminal, gone.direction = True, 1
    return gone


def bonded_inputs(S, K):
    return {"K": K, "t_end": float(S["t"][-1]), "t_count": S["t"].tolist(), "count": [K] * len(S["t"]),
            "t": S["t"], "nn": S["nn"], "N": S["N"], "steps": S["steps"]}


def phase_lock(S):
    """Each neighbour edge's effective step D_eff (unwrapped): its mean drift rate (linear fit), its largest
    excursion about the fit, and the drift expected from the lumps' frequency difference (time-mean omega)."""
    t = S["t"]
    if t.size < 3:
        return None
    U = np.rad2deg(S["steps_eff_unwrapped"])       # continuous: the phases are integrated, never wrapped
    rate, exc, expect = [], [], []
    for k, (a, b) in enumerate(S["edges"]):
        c = np.polyfit(t, U[:, k], 1)
        rate.append(float(c[0]))
        exc.append(float(np.max(np.abs(U[:, k] - np.polyval(c, t)))))
        expect.append(float(np.rad2deg(np.mean(S["W"][:, b] - S["W"][:, a]))))
    return {"drift_deg_per_t": rate, "excursion_deg": exc, "expected_beat_deg_per_t": expect,
            "total_change_deg": [float(U[-1, k] - U[0, k]) for k in range(U.shape[1])], "t_span": float(t[-1] - t[0])}


def label(rule, lock, ctl_lock, unequal, flagged, steps_tol=0.25):
    """Block P (SESSION_03_PREDICTIONS.md, decisions fixed before the scan). rule: bonded_rule applied to the
    effective steps D_eff (clause 5 on D_eff for moving groups; equal to the raw step at rest). For unequal lumps
    clause 5 is "locked": every edge's drift times the run length and its excursion about the drift within the
    clause-5 tolerance, the drift limit raised to 3 times the far-apart control's detrended excursion if that is
    larger. A group meeting every other held clause whose steps drift at the lumps' frequency difference (within
    10% or the drift limit) is "bounded, beating". Runs closer than 3.6 hw at any time are labelled but not counted."""
    if rule is None:
        return "not bonded", {"reason": "too few samples"}
    cl = rule["clauses"]
    others = all(cl[k]["pass"] for k in ("count_equals_K", "distance_within_tol", "no_distance_trend", "N_held"))
    info = {}
    if unequal and lock is not None:
        noise = 3 * max(ctl_lock["excursion_deg"]) if ctl_lock else 0.0
        lim = max(steps_tol, noise)
        T = lock["t_span"]
        locked = all(abs(r) * T <= lim and e <= lim for r, e in zip(lock["drift_deg_per_t"], lock["excursion_deg"]))
        beating = all(abs(r - x) <= max(0.1 * abs(x), lim / T) and e <= lim and abs(x) * T > lim
                      for r, e, x in zip(lock["drift_deg_per_t"], lock["excursion_deg"], lock["expected_beat_deg_per_t"]))
        phase_ok = locked
        info = {"locked": locked, "beating": beating, "limit_deg": lim}
    else:
        phase_ok = cl["phase_steps_held_deg"]["pass"]
        beating = False
    if not others:
        lab = "not bonded"
    elif not phase_ok:
        lab = "bounded, beating" if beating else "not bonded"
    elif not cl["response_differs_from_control"]["pass"]:
        lab = "no interaction"
    elif not cl["duration"]["pass"]:
        lab = "too short"
    else:
        lab = "bonded"
    info["counted"] = not flagged
    return lab, info


def run_with_rule(spec, T_override=None, model_wrap=None):
    from swirl.engine.bonded import bonded_rule
    res, S = run_spec(spec, T_override=T_override, model_wrap=model_wrap)
    K = spec["K"]
    ctrl_spec = dict(spec, spacing_hw=D_FAR / HW0, speed_frac=0.0, family="static", T_control=res["T"])
    cres, CS = run_spec(ctrl_spec, control=True)
    tc = res["contact_time_unit"] or T_MAX
    run_in, ctl_in = bonded_inputs(S, K), bonded_inputs(CS, K)
    ok = S["t"].size >= 3 and CS["t"].size >= 3
    # the rule as written (raw lab steps), recorded
    res["bonded_rule_raw_steps"] = bonded_rule(run_in, ctl_in, tc) if ok else None
    # the rule with clause 5 on the effective step D_eff (Block P decision (a))
    eff, ctl_eff = dict(run_in, steps=S["steps_eff"]), dict(ctl_in, steps=CS["steps_eff"])
    res["bonded_rule"] = bonded_rule(eff, ctl_eff, tc) if ok else None
    res["phase_lock"] = phase_lock(S)
    res["control_phase_lock"] = phase_lock(CS)
    unequal = len(set(np.round(spec["N_mult"], 9))) > 1
    res["unequal"] = unequal
    res["label"], res["label_info"] = label(res["bonded_rule"], res["phase_lock"], res["control_phase_lock"],
                                            unequal, res["flag_below_3p6hw"])
    res["control"] = {"outcome": cres["outcome"], "t_end": cres["t_end"], "d0": cres["d0"]}
    k = np.linspace(0, len(S["t"]) - 1, 41).astype(int)
    res["series"] = {"t": S["t"][k], "X": S["X"][k], "N": S["N"][k], "steps_deg": S["steps"][k],
                     "steps_eff_deg": S["steps_eff"][k], "dmin": S["dmin"][k]}
    return res


def work(args):
    spec, out = args
    path = os.path.join(out, spec["id"] + ".json")
    if os.path.exists(path):
        return spec["id"], "skipped", 0.0
    t0 = time.time()
    try:
        res = run_with_rule(spec)
    except Exception as e:                       # recorded, not hidden
        res = {"spec": spec, "error": repr(e)}
    res["wall_total_s"] = time.time() - t0
    un.dump(res, path)
    return spec["id"], res.get("outcome", "error"), res["wall_total_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", type=int, default=0, help="run this many specs spread over the list")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    S = spec_list()
    fams = {}
    for s in S:
        key = (s["family"], s["K"] == 2, s["size_kind"])
        fams[key] = fams.get(key, 0) + 1
    print("scan size: %d runs (+ one far-apart control each)" % len(S))
    for k, v in sorted(fams.items()):
        print("  %-8s %-6s %-12s %d" % (k[0], "pair" if k[1] else "ring", k[2], v))
    if a.list:
        return
    if a.smoke:
        idx = np.unique(np.linspace(0, len(S) - 1, a.smoke).astype(int))
        S = [S[k] for k in idx]
        a.out = os.path.join(os.path.dirname(a.out), "p2_smoke")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    jobs = [(s, a.out) for s in S]
    if a.workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.workers) as pool:
            for rid, outc, w in pool.imap_unordered(work, jobs):
                print("%s %-22s %.1f s" % (rid, outc, w), flush=True)
    else:
        for j in jobs:
            rid, outc, w = work(j)
            print("%s %-22s %.1f s" % (rid, outc, w), flush=True)
    print("wall %.1f s for %d runs" % (time.time() - t0, len(S)))


if __name__ == "__main__":
    main()
