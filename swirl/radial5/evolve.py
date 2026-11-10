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

GRID_O = dict(h=0.05, dt=0.0125, R_sponge=100.0, R_box=200.0, sigma_max=1.0, r_probe=50.0)
GRID_R = dict(GRID_O, R_sponge=200.0, R_box=400.0)    # reviewer condition 5: wide R tails
GRID = GRID_O
X_THRESHOLDS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]


def default_grid(model):
    return dict(GRID_R if model == "R" else GRID_O)
REC_DT = 0.5
NL_SHARE_DEF = ("1 - <E_quad>/<E> over each measured period inside r_lump, E_quad = int 4 pi r^2 "
                "(f_t^2/2 + S/2 + f^2/2) in the Lagrangian variables (f, f_t; R: f_t = p/q); the "
                "profile tables use the same definition (cycle average of the ansatz)")
PROBE_DT = 0.5
T_REF_DEATH = 50.0
STRIDE_JSON = 5            # frac and energy series in the JSON: every 2.5 time units
CORE_TAIL = 200.0          # the last 200 time units of the core, at 0.1, kept for the spectrum


def coupling(model, twin):
    if model == "R":
        return dict(g4=None, g6=None)
    return dict(g4=0.0, g6=0.0) if twin else dict(SESSION1)


def make_solver(model, obj, grid=None, twin=False, closed=False, qmax=1e4, hk_max=1e-6):
    grid = default_grid(model) if grid is None else grid
    sm = 0.0 if closed else grid["sigma_max"]
    if model == "O":
        c = coupling("O", twin)
        return SolverO(obj, grid["R_box"], grid["h"], c["g4"], c["g6"],
                       sponge_start=grid["R_sponge"], sigma_max=sm, r_probe=grid.get("r_probe"))
    return SolverR(obj, grid["R_box"], grid["h"], kg=twin, sponge_start=grid["R_sponge"],
                   sigma_max=sm, r_probe=grid.get("r_probe"), qmax=qmax, hk_max=hk_max)


# ---------------- starts ----------------

def profile_start(model, obj, omega=None, scale=1.0, amplitude=None, eps_guess=None):
    """Single-frequency profile start, f = scale F(r), f_t = 0. Given omega: the lowest-
    amplitude profile at that rate (O bump: the Session 1 family tool, swirl.family.profile,
    rescaled beta = 4/3, / sqrt(0.30)). Given amplitude (F(0) for the bump, F'(0) for the
    pinch): the profile with that central amplitude, which reaches branches past a fold in
    omega (eps_guess: where to look, default the NLS estimate)."""
    g4, g6 = (SESSION1["g4"], SESSION1["g6"]) if model == "O" else (0.0, 0.0)
    mdl = P.Model(model, obj, g4=g4, g6=g6)
    if amplitude is not None:
        if eps_guess is None:
            lam = 0.75 * g4 if model == "O" else 0.375
            eps_guess = (amplitude * np.sqrt(lam) / 4.3374 if obj == "bump"
                         else np.sqrt(amplitude * np.sqrt(lam) / 3.7782))
        res = P.solve_point(mdl, x=amplitude, guess=min(eps_guess, 0.9))
        omega = res["omega"]
    else:
        res = P.solve(mdl, omega)
    info = P.public(res)
    info["scale"] = float(scale)
    if model == "O" and obj == "bump" and amplitude is None:
        from swirl.family.profile import profile_curve, solve_profile
        q = solve_profile(omega, 1.0, 1.0, 4.0 / 3.0, tol=1e-10)
        fn = lambda r: scale * profile_curve(q, r) / np.sqrt(0.30)
        info["source"] = "swirl.family.profile.solve_profile(omega, 1, 1, 4/3, tol=1e-10) / sqrt(0.30)"
        info["hw"] = float(q["half_width"])
    else:
        fn = lambda r: scale * P.curve(res, r)[0]
        info["source"] = ("swirl.radial5.profiles.%s (cycle-averaged full Lagrangian, shooting)"
                          % ("solve_point, amplitude fixed" if amplitude is not None else "solve"))
    info.update(kind="profile", model=model, obj=obj)
    return dict(f=fn, ft=None, hw=info["hw"], kappa=info["kappa"], info=info)


def _hw_of(fn, r_max):
    rr = np.linspace(0, r_max, 200001)
    F = np.abs(fn(rr))
    i = int(np.argmax(F))
    j = i + int(np.argmax(F[i:] < 0.5 * F[i]))
    return float(rr[j])


def gauss_start(obj, A, sigma, v0=0.0, kappa_tail=None):
    """f = A g(r), f_t = v0 g(r), with the shape g = exp(-r^2/2 sigma^2) (bump) or
    (r/sigma) exp(-r^2/2 sigma^2) (pinch). hw is measured on f (outer half maximum)."""
    if obj == "bump":
        g = lambda r: np.exp(-r * r / (2 * sigma * sigma))
    else:
        g = lambda r: (r / sigma) * np.exp(-r * r / (2 * sigma * sigma))
    fn = lambda r: A * g(r)
    ft = None if v0 == 0.0 else (lambda r: v0 * g(r))
    hw = _hw_of(fn, 20 * sigma)
    return dict(f=fn, ft=ft, hw=hw, kappa=kappa_tail,
                info=dict(kind="gauss", obj=obj, A=A, sigma=sigma, v0=v0, hw=hw))


def shell_start(obj, A, r0, sigma, v_in=0.0, kappa_tail=None):
    """A Gaussian shell centred at r0, made regular at r = 0 by its mirror image:
    bump f = A [g(r - r0) + g(r + r0)] (even), pinch f = A [g(r - r0) - g(r + r0)] (odd),
    g(x) = exp(-x^2/2 sigma^2). v_in > 0 moves it inward: f(r, t) = A [g(r - r0 + v t)
    +- g(r + r0 - v t)], so f_t(t = 0) = A v_in [g'(r - r0) -+ g'(r + r0)]."""
    g = lambda x: np.exp(-x * x / (2 * sigma * sigma))
    dg = lambda x: -x / (sigma * sigma) * g(x)
    sg = 1.0 if obj == "bump" else -1.0
    fn = lambda r: A * (g(r - r0) + sg * g(r + r0))
    ft = None if v_in == 0.0 else (lambda r: A * v_in * (dg(r - r0) - sg * dg(r + r0)))
    hw = _hw_of(fn, r0 + 20 * sigma)
    return dict(f=fn, ft=ft, hw=hw, kappa=kappa_tail,
                info=dict(kind="shell", obj=obj, A=A, r0=r0, sigma=sigma, v_in=v_in, hw=hw))


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


def run(model, obj, start, T, grid=None, twin=False, ckpt=None, ckpt_every=500.0,
        stop_at_death=False, stop_after=None, qmax=1e4, hk_max=1e-6, verbose=False):
    """Evolve one start. Returns the result dict (see _result). ckpt: .npz path for
    checkpoint/resume (resumes when the parameters match). stop_after: stop and
    checkpoint at this time (tests the resume). A guard trip (R) ends the run and is
    recorded in result['guard']."""
    grid = default_grid(model) if grid is None else grid
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
        Ein0, E0, Eq0 = s.energy_inside_q(rcs)
        parts0 = s.energy_parts()
        c0 = float(s.core())
        st = dict(times=[0.0], E_in=[list(map(float, Ein0))], E_tot=[float(E0)], Eq_in=[float(Eq0)],
                  E_h=[float(s.energy_discrete())], E0=float(E0), parts0=parts0,
                  E_ref50=None, T_90=None, T_death=None,
                  trig=dict(prev_v=c0, prev_t=0.0, armed=False, tc=[], amp=abs(c0),
                            amp_hist=[abs(c0)]),
                  probe=[], core_tail=[], hk=[[0.0, s.high_k_fraction()]],
                  xcross={}, guard=None, done=False, wall=0.0)
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
                _crossings(st, s, t_first, dt)
            break
        k += 1
        if abs(s.t - k * REC_DT) < 1e-6:
            s.t = k * REC_DT
        if model == "R":
            _log_extremes(st, s)
            _crossings(st, s, t_first, dt)
        _schmitt(st["trig"], t_first, dt, core)
        st["probe"].extend(float("%.6g" % x) for x in probe[nprobe - 1::nprobe])
        st["core_tail"].extend(float("%.6g" % x) for x in core[ntail - 1::ntail])
        del st["core_tail"][:-int(round(CORE_TAIL / (ntail * dt)))]
        a, b, aq = s.energy_inside_q(rcs)
        st["times"].append(float(s.t))
        st["E_in"].append(list(map(float, a)))
        st["E_tot"].append(float(b))
        st["Eq_in"].append(float(aq))
        st["E_h"].append(float(s.energy_discrete()))
        if k % 20 == 0:
            st["hk"].append([float(s.t), s.high_k_fraction()])
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


def _crossings(st, s, t_first, dt):
    """First time the per-step min(1 - X) (over every RK4 stage) falls below each of
    X_THRESHOLDS; used to tell a converging collapse time t* from a resolution artefact."""
    om = s.last_om
    for thr in X_THRESHOLDS:
        key = "%.0e" % thr
        if key in st["xcross"]:
            continue
        i = np.where(om < thr)[0]
        if i.size:
            st["xcross"][key] = float(t_first + dt * i[0])


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
    Eq = np.array(st["Eq_in"])
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
        share = 1.0 - float(np.mean(Eq[sel])) / eL if eL > 0 else float("nan")
        per.append([round(0.5 * (a + b), 4), rate, e3, eL, 2 * np.pi * eL / rate, share])
    per = np.array(per) if per else np.zeros((0, 6))
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
    sh = per[:, 5] if per.size else np.array([])
    sel_sh = (per[:, 0] >= 50) & (per[:, 0] <= 150) if per.size else np.array([], bool)
    gx = st.get("ext_global", {}).get("min_one_minus_X")
    out = dict(
        params=params,
        E0=E0, E_h0=float(Eh[0]), energy_parts_t0=st["parts0"],
        regime=dict(rate_settled_50_150=w_settled, rate_end_last100=w_end,
                    rate_spectral_last200=w_spec, floor=1.0,
                    nonlinear_share_definition=NL_SHARE_DEF,
                    nonlinear_share_cyc_first_period=float(sh[0]) if sh.size else None,
                    nonlinear_share_cyc_50_150=float(np.median(sh[sel_sh])) if sel_sh.any() else None,
                    X_max_all_stages=None if gx is None else 1.0 - gx["value"],
                    q_max_all_stages=None if gx is None else gx["value"] ** -0.5),
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
                                 "E_per_cycle_rlump", "nonlinear_share_cyc_rlump"],
                        rows=[[float("%.8g" % x) for x in row] for row in per]),
        probe=dict(r=float(s.r_probe), dt=PROBE_DT, t0=PROBE_DT, f=st["probe"]),
        core_tail=dict(dt=ntail * params["grid"]["dt"], f=st["core_tail"]),
        high_k=st["hk"],
    )
    if params["model"] == "R":
        out["guard_extremes"] = dict(
            note="over every RK4 stage; rows are block extremes over %d records (%.1f time units)"
                 % (EXT_EVERY, EXT_EVERY * REC_DT),
            columns=["t_block_end", "min_1-X", "r", "max|p|", "r", "min_pinch_margin(1-X-(f/r)^2)", "r",
                     "X_max = 1 - min(1-X)", "q_max = min(1-X)^-1/2"],
            rows=[[None if x is None else float("%.7g" % x) for x in
                   row + [1.0 - row[1], row[1] ** -0.5 if row[1] > 0 else None]]
                  for row in st.get("ext_rows", [])],
            global_extremes=st.get("ext_global", {}),
            first_time_min_1mX_below=st.get("xcross", {}))
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


# ---------------- classification of a run's end ----------------

CLASSES = ("held", "held, but not by stiffening", "alive at cap", "disperses", "singular",
           "3D ill-posed", "unresolved", "invalid start", "trip, unrefined")


def classify(res, twin=None):
    """One of CLASSES for a finished run (rule 2.1 for 'held'). A resolution or momentum
    trip is 'trip, unrefined' until refine_trip decides 'singular' or 'unresolved'."""
    g = res.get("guard")
    if g is not None:
        kind = g["kind"]
        if "start" in kind:
            return dict(cls="invalid start", detail=kind)
        if "angular" in kind:
            return dict(cls="3D ill-posed", detail=kind, t=g["t"], r=g["r"])
        return dict(cls="trip, unrefined", detail=kind, t=g["t"], r=g["r"])
    if res["T_death"] is not None or res["frac_end"] < 0.5:
        return dict(cls="disperses", T_death=res["T_death"], frac_end=res["frac_end"])
    w = res["regime"]["rate_end_last100"] or res["regime"]["rate_settled_50_150"]
    if w is None:
        return dict(cls="disperses", detail="no measurable rate", frac_end=res["frac_end"])
    T1000 = 1000.0 * 2 * np.pi / w
    if res["T_end"] < T1000 - 1e-9:
        return dict(cls="alive at cap", cap=res["T_end"], cycles=res["T_end"] * w / (2 * np.pi),
                    frac_end=res["frac_end"])
    t = np.array(res["series"]["t"])
    i = int(np.argmin(np.abs(t - T1000)))
    fr = res["series"]["frac"][i]
    resolution = 2 * np.pi / CORE_TAIL                    # spectral resolution of the rate
    below = w < 1.0 - resolution
    out = dict(frac_1000_cycles=fr, rate=w, rate_resolution=resolution, below_floor=bool(below))
    if fr < 0.90 or not below:
        return dict(out, cls="disperses" if fr < 0.5 else "alive at cap",
                    detail="frac at 1000 cycles below 0.90 or rate not below the floor")
    if twin is None:
        return dict(out, cls="alive at cap", detail="held-candidate: twin needed for rule 2.1")
    tt = np.array(twin["series"]["t"])
    ftw = twin["series"]["frac"][int(np.argmin(np.abs(tt - T1000)))] if tt[-1] >= T1000 - 1e-9 else None
    out["twin_frac_1000_cycles"] = ftw
    if ftw is None:
        return dict(out, cls="alive at cap", detail="twin shorter than 1000 cycles")
    return dict(out, cls="held" if ftw <= 0.5 else "held, but not by stiffening")


def refine_trip(model, obj, start, res, levels=(2, 4), t_extra=5.0, tol=0.02):
    """Rerun a tripped start at h/2 and h/4 (dt = h/4, sponge 60 to 90; the trip comes long
    before anything reaches 60) to t_trip + t_extra, and compare the first times at which the
    per-stage min(1 - X) falls below 1e-1 ... 1e-8. If at the deepest threshold reached on the
    two finest grids the times agree to tol (relative): 'singular', t* = that time on the
    finest grid ('centre collapse' when it sits within 1 of r = 0). Otherwise 'unresolved'."""
    g = res["guard"]
    base = res["params"]["grid"]
    runs = [dict(h=base["h"], trip_t=g["t"], trip_r=g["r"], kind=g["kind"],
                 crossings=res.get("guard_extremes", {}).get("first_time_min_1mX_below", {}))]
    for m in levels:
        h = base["h"] / m
        grid = dict(base, h=h, dt=h / 4, R_sponge=60.0, R_box=90.0, r_probe=30.0)
        grid.pop("sigma_profile", None)
        r = run(model, obj, start, g["t"] + t_extra, grid=grid)
        gg = r.get("guard") or {}
        runs.append(dict(h=h, trip_t=gg.get("t"), trip_r=gg.get("r"), kind=gg.get("kind"),
                         crossings=r["guard_extremes"]["first_time_min_1mX_below"]))
    a, b = runs[-2]["crossings"], runs[-1]["crossings"]
    common = [k for k in a if k in b]
    if not common:
        return dict(cls="unresolved", runs=runs, detail="no common threshold crossing on the finest grids")
    deep = min(common, key=float)
    ta, tb = a[deep], b[deep]
    conv = abs(ta - tb) <= tol * tb
    r_trip = runs[-1]["trip_r"] if runs[-1]["trip_r"] is not None else runs[0]["trip_r"]
    if conv:
        where = "centre collapse" if (r_trip is not None and r_trip < 1.0) else "off-centre"
        return dict(cls="singular", where=where, t_star=tb, threshold=deep,
                    t_at_threshold=[r_["crossings"].get(deep) for r_ in runs], runs=runs)
    return dict(cls="unresolved", threshold=deep, t_at_threshold=[r_["crossings"].get(deep) for r_ in runs],
                runs=runs, detail="crossing times move with h")
