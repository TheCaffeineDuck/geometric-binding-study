"""Evolution driver for equations O and R, single bump and pinch (Track F production).

One run = one start evolved with the absorber on, plus (separately) its linear twin:
O: g4 = g6 = 0 in the O code; R: the Klein-Gordon density in the R code (same start,
same grid, same operators).

Definitions (Session 2, unchanged; SESSION_05_PREDICTIONS.md Block 2.1):
  frac(t)  = E_in(3 hw, t) / E_total(t = 0), instantaneous, recorded every rec_dt = 0.5
  frac_cyc = the same with E_in averaged over each measured period (Block 2.1)
  T_90     = first t with frac < 0.90
  T_death  = first t > 50 with E_in(3 hw) < 0.5 E_in(3 hw, t = 50)
  rate     = 2 pi / (time between upward zero crossings of the core coordinate), Schmitt
             trigger as Session 2 (swirl.family.wpa_run); core = f(0, t) for the bump and
             f_r(0, t) for the pinch (f(0) = 0); cross-checked by the spectral peak of the
             core over the last window.
  E per cycle = 2 pi E_in / rate, per measured period (prediction F4).
hw is the radius where the start profile falls to half its peak (for the pinch: the
outer half-maximum radius). E_in uses the 4th-order physical density (O) or the exact
discrete density (R), not np.gradient (F-003).
"""

import json
import os
import time

import numpy as np

from swirl.family.wpa_run import _schmitt, omega_series
from swirl.radial5 import profiles as P
from swirl.radial5.solver_o import SESSION1, SolverO
from swirl.radial5.solver_r import GuardTrip, SolverR

GRID = dict(h=0.05, dt=0.0125, R_sponge=100.0, R_box=200.0, sigma_max=1.0, r_probe=50.0)
REC_DT = 0.5
PROBE_DT = 0.5
T_REF_DEATH = 50.0
STRIDE_JSON = 5            # frac and energy series in the JSON: every 2.5 time units
CORE_TAIL = 200.0          # the last 200 time units of the core, at 0.1, kept for the spectrum


def coupling(model, twin):
    if model == "R":
        return dict(g4=None, g6=None)
    return dict(g4=0.0, g6=0.0) if twin else dict(SESSION1)


def make_solver(model, obj, grid=GRID, twin=False, closed=False, qmax=1e4, hk_max=1e-6):
    sm = 0.0 if closed else grid["sigma_max"]
    if model == "O":
        c = coupling("O", twin)
        return SolverO(obj, grid["R_box"], grid["h"], c["g4"], c["g6"],
                       sponge_start=grid["R_sponge"], sigma_max=sm, r_probe=grid.get("r_probe"))
    return SolverR(obj, grid["R_box"], grid["h"], kg=twin, sponge_start=grid["R_sponge"],
                   sigma_max=sm, r_probe=grid.get("r_probe"), qmax=qmax, hk_max=hk_max)


# ---------------- starts ----------------

def profile_start(model, obj, omega):
    """Single-frequency profile start (f = F(r), f_t = 0). O bump: the Session 1 family
    tool (swirl.family.profile, rescaled beta = 4/3, / sqrt(0.30)); the rest: the
    cycle-averaged full Lagrangian (swirl.radial5.profiles)."""
    g4, g6 = (SESSION1["g4"], SESSION1["g6"]) if model == "O" else (0.0, 0.0)
    res = P.solve(P.Model(model, obj, g4=g4, g6=g6), omega)
    info = P.public(res)
    if model == "O" and obj == "bump":
        from swirl.family.profile import profile_curve, solve_profile
        q = solve_profile(omega, 1.0, 1.0, 4.0 / 3.0, tol=1e-10)
        fn = lambda r: profile_curve(q, r) / np.sqrt(0.30)
        info["source"] = "swirl.family.profile.solve_profile(omega, 1, 1, 4/3, tol=1e-10) / sqrt(0.30)"
        info["hw"] = float(q["half_width"])
    else:
        fn = lambda r: P.curve(res, r)[0]
        info["source"] = "swirl.radial5.profiles.solve (cycle-averaged full Lagrangian, shooting)"
    info.update(kind="profile", model=model, obj=obj)
    return dict(f=fn, ft=None, hw=info["hw"], kappa=info["kappa"], info=info)


def gauss_start(obj, A, sigma, kappa_tail=None):
    """f = A exp(-r^2/2 sigma^2) (bump) or A (r/sigma) exp(-r^2/2 sigma^2) (pinch), f_t = 0.
    hw is measured on the start (outer half-maximum radius)."""
    if obj == "bump":
        fn = lambda r: A * np.exp(-r * r / (2 * sigma * sigma))
    else:
        fn = lambda r: A * (r / sigma) * np.exp(-r * r / (2 * sigma * sigma))
    rr = np.linspace(0, 20 * sigma, 200001)
    F = fn(rr)
    i = int(np.argmax(F))
    j = i + int(np.argmax(F[i:] < 0.5 * F[i]))
    hw = float(rr[j])
    return dict(f=fn, ft=None, hw=hw, kappa=kappa_tail,
                info=dict(kind="gauss", obj=obj, A=A, sigma=sigma, hw=hw))


def r_lump(hw, kappa):
    """Session 2: radius holding the lump with its tail, max(3 hw, hw + 4.6/kappa)."""
    if kappa is None:
        return 3.0 * hw
    return float(max(3.0 * hw, hw + 4.6 / kappa))


# ---------------- the run ----------------

def _params(model, obj, start, T, grid, twin, extra):
    p = dict(model=model, obj=obj, twin=bool(twin), T=float(T), rec_dt=REC_DT, probe_dt=PROBE_DT,
             start=start["info"], hw=start["hw"], rcs=[3.0 * start["hw"], r_lump(start["hw"], start["kappa"])],
             grid=dict(grid, sigma_profile="sigma_max((r-R_sponge)/(R_box-R_sponge))^2"),
             units="wave speed 1, gap (mass) 1, limit 1; plan normalisation")
    if model == "O":
        p["coupling"] = coupling("O", twin)
        p["equation"] = ("O: f_tt = f_rr + 2 f_r/r - s2 f/r^2 - f + g4 f^3 - g6 f^5 (s2 = 0 bump, 2 pinch); "
                         "u = r f, 4th-order FD, RK4" + (" [linear twin]" if twin else ""))
    else:
        p["equation"] = ("R: L = 1 - sqrt(1 - X) - f^2/2, Hamiltonian (f, p), cell-centred 4th-order D, RK4"
                         if not twin else "R twin: Klein-Gordon density p^2/2 + S/2 + f^2/2, same scheme")
    p.update(extra)
    return p


def run(model, obj, start, T, grid=GRID, twin=False, ckpt=None, ckpt_every=500.0,
        stop_at_death=False, stop_after=None, qmax=1e4, hk_max=1e-6, verbose=False):
    """Evolve one start. Returns the result dict (see _result). ckpt: .npz path for
    checkpoint/resume (resumes when the parameters match). stop_after: stop and
    checkpoint at this time (tests the resume). A guard trip (R) ends the run and is
    recorded in result['guard']."""
    params = _params(model, obj, start, T, grid, twin,
                     dict(qmax=qmax, hk_max=hk_max, stop_at_death=bool(stop_at_death)))
    pkey = json.dumps(params, sort_keys=True, default=float)
    s = make_solver(model, obj, grid, twin, qmax=qmax, hk_max=hk_max)
    rcs = params["rcs"]
    dt = grid["dt"]
    nrec = int(round(REC_DT / dt))
    nprobe = int(round(PROBE_DT / dt))
    ntail = max(1, int(round(0.1 / dt)))
    st = None
    if ckpt is not None and os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=False)
        if str(z["pkey"]) != pkey:
            raise RuntimeError("checkpoint %s has different parameters" % ckpt)
        s.put_state({k: z[k] for k in z.files if k in ("u", "v", "f", "p")} | {"t": float(z["t"])})
        st = json.loads(str(z["state"]))
    if st is None:
        guard0 = None
        try:
            s.set_f(start["f"], start["ft"])
        except GuardTrip as g:
            guard0 = g.diag
        if guard0 is not None:
            return dict(params=params, guard=guard0, T_end=0.0)
        Ein0, E0 = s.energy_inside(rcs)
        parts0 = s.energy_parts()
        c0 = float(s.core())
        st = dict(times=[0.0], E_in=[list(map(float, Ein0))], E_tot=[float(E0)],
                  E_h=[float(s.energy_discrete())], E0=float(E0), parts0=parts0,
                  E_ref50=None, T_90=None, T_death=None,
                  trig=dict(prev_v=c0, prev_t=0.0, armed=False, tc=[], amp=abs(c0),
                            amp_hist=[abs(c0)]),
                  probe=[], core_tail=[], hk=[[0.0, s.high_k_fraction()]],
                  extra=[[0.0, parts0["nonlinear_share_signed"], parts0.get("X_max"),
                          parts0.get("q_max")]],
                  guard=None, done=False, wall=0.0)
    nchunks = int(round(T / REC_DT))
    k = int(round(s.t / REC_DT))
    last_ck = s.t
    t_wall = time.time()
    while k < nchunks and not st["done"]:
        t_first = s.t + dt
        try:
            core, probe = (s.step(dt, nrec, ckpt_dir=os.path.dirname(ckpt) if ckpt else None)
                           if model == "R" else s.step(dt, nrec))
        except GuardTrip as g:
            st["guard"] = g.diag
            st["done"] = True
            if hasattr(s, "last_ext"):
                _log_extremes(st, s)
            break
        k += 1
        if abs(s.t - k * REC_DT) < 1e-6:
            s.t = k * REC_DT
        if model == "R":
            _log_extremes(st, s)
        _schmitt(st["trig"], t_first, dt, core)
        st["probe"].extend(float("%.6g" % x) for x in probe[nprobe - 1::nprobe])
        st["core_tail"].extend(float("%.6g" % x) for x in core[ntail - 1::ntail])
        del st["core_tail"][:-int(round(CORE_TAIL / (ntail * dt)))]
        a, b = s.energy_inside(rcs)
        st["times"].append(float(s.t))
        st["E_in"].append(list(map(float, a)))
        st["E_tot"].append(float(b))
        st["E_h"].append(float(s.energy_discrete()))
        if k % 20 == 0:
            pa = s.energy_parts()
            st["hk"].append([float(s.t), s.high_k_fraction()])
            st["extra"].append([float(s.t), pa["nonlinear_share_signed"], pa.get("X_max"),
                                pa.get("q_max")])
        frac = a[0] / st["E0"]
        if st["T_90"] is None and frac < 0.90:
            st["T_90"] = float(s.t)
        if st["E_ref50"] is None and s.t >= T_REF_DEATH - 1e-9:
            st["E_ref50"] = float(a[0])
        if (st["T_death"] is None and st["E_ref50"] is not None and s.t > T_REF_DEATH
                and a[0] < 0.5 * st["E_ref50"]):
            st["T_death"] = float(s.t)
            if stop_at_death:
                st["done"] = True
        stop_now = stop_after is not None and s.t >= stop_after - 1e-9
        if ckpt is not None and (s.t - last_ck >= ckpt_every - 1e-9 or stop_now or st["done"]
                                 or k >= nchunks):
            st["wall"] += time.time() - t_wall
            t_wall = time.time()
            _save_ckpt(ckpt, s, st, pkey)
            last_ck = s.t
        if verbose and k % 2000 == 0:
            print("t=%.0f frac=%.4f" % (s.t, frac), flush=True)
        if stop_now:
            return dict(params=params, partial=True, t=s.t)
    st["wall"] += time.time() - t_wall
    if st.get("ext_blk") is not None:                     # flush a partial block
        b = st["ext_blk"]
        st.setdefault("ext_rows", []).append(
            [float(s.t), b["min_one_minus_X"], b["r_min_one_minus_X"], b["max_abs_p"],
             b["r_max_abs_p"], b["min_pinch_margin"], b["r_min_pinch_margin"]])
        st["ext_blk"] = None
    if ckpt is not None:
        _save_ckpt(ckpt, s, st, pkey)
    return _result(params, st, s)


EXT_EVERY = 10             # R: extremes logged over blocks of 10 records (5 time units)


def _log_extremes(st, s):
    """R runs: min(1 - X), max|p| and the pinch margin min(1 - X - (f/r)^2), each with its
    radius, over every RK4 stage; one row per EXT_EVERY records holding the block
    extremes, and the global extremes with time and radius."""
    e = s.last_ext
    t = float(s.t)
    blk = st.setdefault("ext_blk", None)
    if blk is None:
        blk = dict(e)
    else:
        if e["min_one_minus_X"] < blk["min_one_minus_X"]:
            blk["min_one_minus_X"], blk["r_min_one_minus_X"] = e["min_one_minus_X"], e["r_min_one_minus_X"]
        if e["max_abs_p"] > blk["max_abs_p"]:
            blk["max_abs_p"], blk["r_max_abs_p"] = e["max_abs_p"], e["r_max_abs_p"]
        if e["min_pinch_margin"] is not None and e["min_pinch_margin"] < blk["min_pinch_margin"]:
            blk["min_pinch_margin"], blk["r_min_pinch_margin"] = e["min_pinch_margin"], e["r_min_pinch_margin"]
    st["ext_blk"] = blk
    g = st.setdefault("ext_global", {})
    for key, sign in (("min_one_minus_X", 1), ("max_abs_p", -1), ("min_pinch_margin", 1)):
        v = e[key]
        if v is None:
            continue
        if key not in g or sign * v < sign * g[key]["value"]:
            g[key] = dict(value=v, t=t, r=e["r_" + key])
    st["ext_n"] = st.get("ext_n", 0) + 1
    if st["ext_n"] % EXT_EVERY == 0:
        st.setdefault("ext_rows", []).append(
            [t, blk["min_one_minus_X"], blk["r_min_one_minus_X"], blk["max_abs_p"], blk["r_max_abs_p"],
             blk["min_pinch_margin"], blk["r_min_pinch_margin"]])
        st["ext_blk"] = None


def _save_ckpt(path, s, st, pkey):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **{k: v for k, v in s.get_state().items() if k != "t"}, t=s.t,
             state=json.dumps(st), pkey=pkey)
    os.replace(tmp, path)


def spectral_rate(core, dt):
    """Peak of the Hann-windowed spectrum of the core signal, parabolic interpolation."""
    x = np.asarray(core) - np.mean(core)
    n = x.size
    if n < 64:
        return None
    pad = 8 * n
    X = np.abs(np.fft.rfft(x * np.hanning(n), pad))
    om = 2 * np.pi * np.fft.rfftfreq(pad, dt)
    i = int(np.argmax(X[1:])) + 1
    if 0 < i < X.size - 1:
        a, b, c = np.log(X[i - 1] + 1e-300), np.log(X[i] + 1e-300), np.log(X[i + 1] + 1e-300)
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        return float(om[i] + d * (om[1] - om[0]))
    return float(om[i])


def _result(params, st, s):
    t = np.array(st["times"])
    Ein = np.array(st["E_in"])
    E0 = st["E0"]
    frac = Ein[:, 0] / E0
    tc = np.array(st["trig"]["tc"])
    tw, w = omega_series(tc)
    # per-period table: rate, cycle-mean E_in(3hw), E_in(r_lump), energy per cycle
    per = []
    for a, b in zip(tc[:-1], tc[1:]):
        sel = (t >= a) & (t < b)
        if sel.sum() == 0:
            continue
        rate = 2 * np.pi / (b - a)
        e3, eL = float(np.mean(Ein[sel, 0])), float(np.mean(Ein[sel, 1]))
        per.append([round(0.5 * (a + b), 4), rate, e3, eL, 2 * np.pi * eL / rate])
    per = np.array(per) if per else np.zeros((0, 5))
    frac_cyc = per[:, 2] / E0 if per.size else np.array([])
    t_cyc = per[:, 0] if per.size else np.array([])
    first = lambda tt, ff, thr: float(tt[np.argmax(ff < thr)]) if np.any(ff < thr) else None
    T_end = float(t[-1])
    sel = (tw >= 50) & (tw <= 150)
    w_settled = float(np.median(w[sel])) if sel.any() else None
    sel = tw >= T_end - 100
    w_end = float(np.median(w[sel])) if sel.any() else None
    ntail = max(1, int(round(0.1 / params["grid"]["dt"])))
    w_spec = spectral_rate(st["core_tail"], ntail * params["grid"]["dt"])
    cyc = lambda T: None if (T is None or w_settled is None) else T * w_settled / (2 * np.pi)
    Eh = np.array(st["E_h"])
    i50 = int(np.argmin(np.abs(t - 50.0)))
    out = dict(
        params=params,
        E0=E0, E_h0=float(Eh[0]), energy_parts_t0=st["parts0"],
        regime=dict(rate_settled_50_150=w_settled, rate_end_last100=w_end,
                    rate_spectral_last200=w_spec, floor=1.0,
                    nonlinear_share_signed_t0=st["parts0"]["nonlinear_share_signed"]),
        T_end=T_end, T_90=st["T_90"], T_death=st["T_death"],
        cycles_T_90=cyc(st["T_90"]), cycles_T_death=cyc(st["T_death"]), cycles_T_end=cyc(T_end),
        T_90_cyc=first(t_cyc, frac_cyc, 0.90),
        frac_t0=float(frac[0]), frac_t50=float(frac[i50]), frac_end=float(frac[-1]),
        frac_at={str(T): float(frac[int(np.argmin(np.abs(t - T)))]) for T in
                 (100, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000) if T <= T_end + 1e-9},
        guard=st["guard"],
        energy_check=dict(E_h_rise_max_rel=float(np.max(np.diff(Eh)) / Eh[0]) if Eh.size > 1 else 0.0,
                          note="E_h is exactly conserved without the sponge; with it, it may only fall"),
        wall_s=st["wall"],
        series=dict(t=t[::STRIDE_JSON].tolist(), frac=[float("%.7g" % x) for x in frac[::STRIDE_JSON]],
                    frac_rlump=[float("%.7g" % x) for x in (Ein[:, 1] / E0)[::STRIDE_JSON]],
                    E_tot=[float("%.9g" % x) for x in np.array(st["E_tot"])[::STRIDE_JSON]],
                    E_h=[float("%.10g" % x) for x in Eh[::STRIDE_JSON]]),
        per_period=dict(columns=["t_mid", "rate", "E_in_3hw_mean", "E_in_rlump_mean",
                                 "E_per_cycle_rlump"],
                        rows=[[float("%.8g" % x) for x in row] for row in per]),
        probe=dict(r=float(s.r_probe), dt=PROBE_DT, t0=PROBE_DT, f=st["probe"]),
        core_tail=dict(dt=ntail * params["grid"]["dt"], f=st["core_tail"]),
        high_k=st["hk"],
        nonlinear_share_X_q=dict(columns=["t", "nonlinear_share_signed", "X_max", "q_max"],
                                 rows=st["extra"]),
    )
    if params["model"] == "R":
        out["guard_extremes"] = dict(
            note="over every RK4 stage; rows are block extremes over %d records (%.1f time units)"
                 % (EXT_EVERY, EXT_EVERY * REC_DT),
            columns=["t_block_end", "min_1-X", "r", "max|p|", "r", "min_pinch_margin(1-X-(f/r)^2)", "r"],
            rows=[[None if x is None else float("%.7g" % x) for x in row] for row in st.get("ext_rows", [])],
            global_extremes=st.get("ext_global", {}))
    return out


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, default=_default)
    os.replace(tmp, path)
    return os.path.abspath(path)


def _default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def summary(res):
    """The headline numbers of one run (no series)."""
    keys = ("E0", "T_end", "T_90", "T_death", "cycles_T_90", "cycles_T_death", "T_90_cyc",
            "frac_t0", "frac_t50", "frac_end", "frac_at", "regime", "guard", "wall_s")
    return {k: res.get(k) for k in keys}
