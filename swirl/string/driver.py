"""Track C driver: one entry point for families, evolutions (with linear twin), pushes and the cost
table. JSON written atomically to outputs/string/, with the full job parameters in every file.

    <python> -m swirl.string.driver jobs.json            # a JSON list of jobs (or one job)

Job kinds and their keys (defaults in brackets):
  family         C-grid: law, obj (bump|sbend), beta [0], omega_start [2.3], omega_min [2.01],
                 d_stop [0.99], omega_stop, floquet [true], N [auto]
  smooth_orbits  C-smooth ('fd') Newton attempts (tool b): law, obj, hs, sigmas, peak_stretches,
                 omega [O4 lone-oscillator rate], L [16]
  evolve         medium, law, beta, h, N, sponge_len [300], sigma_max [0.05], dt, T, start,
                 twin [true], sample_dt [0.1], record_periods [1], probes [[0, 20, 50]],
                 probe_max [65536], W [Block 2.1], push ({"lambda": c}), profile_range [100],
                 snap_times [[]] (field snapshots to checkpoints/), ckpt_every_s [300]
                 start: {"type": "orbit", "obj", "omega"}           C-grid exact state
                      | {"type": "rwa", "obj", "A" or "omega"}      C-grid rotating-wave pattern
                      | {"type": "gauss", "obj", "sigma", "A" or "peak_stretch"}
                      | {"type": "jump", "obj", "omega", "x_s" [0.25] or "a_top"}  C-smooth G1
                 every start: "noise" [1e-8] (0 = off), "seed" [0]; tail_fit_from [0.5]
  push           as evolve (twin on by default), with push.lambdas a list: one evolve per kick
                 (speed, energy lost per unit distance, broke, crossed dist_target [100])
  cost           cases: list of evolve-like dicts (+ omega_nominal); periods [20]
Core coordinate (Block 4): bump = displacement at the centre; S-bend = stretch across the centre.
Field arrays never go to outputs/; checkpoints and snapshots go to <repo>/checkpoints/string/.
Run under: NUMBA_NUM_THREADS=1 caffeinate -i <python> -m swirl.string.driver jobs.json
"""

import hashlib
import json
import os
import sys
import time

import numpy as np

from swirl.string import diag as G
from swirl.string import jump as JP
from swirl.string import orbits as O
from swirl.string import starts as ST
from swirl.string.core import GuardTrip, String, band_top

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "outputs", "string")
CKPT = os.path.join(ROOT, "checkpoints", "string")


def _default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_default)
    os.replace(tmp, path)
    return path


def write_npz(path, **arrs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrs)
    os.replace(tmp, path)
    return path


def cfg_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=_default).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- building
# Production sponge (matched, quadratic ramp): reflection of a grid band-edge packet (k = 0.9 pi)
# 6.0e-6 of its energy (check 4); 100 / 0.3 gave 3.5e-4. Long waves lose a factor e^-20 in energy
# over one traverse.
SPONGE_LEN = 300.0
SPONGE_SIGMA = 0.05
NOISE = 1e-8          # default seeded noise on every start (start.noise = 0 switches it off)


def build(cfg):
    if cfg["law"] == "G2" and "beta" not in cfg:
        raise ValueError("a G2 job needs beta (without it the law would be the linear twin)")
    return String(cfg["medium"], cfg["law"], cfg["N"], h=cfg.get("h", 1.0),
                  beta=cfg.get("beta", 0.0), sponge_len=cfg.get("sponge_len", SPONGE_LEN),
                  sigma_max=cfg.get("sigma_max", SPONGE_SIGMA))


def build_start(S, cfg):
    """Returns u0, x0 (object centre), omega_nominal (None if unknown), start info.
    Every start gets noise * N(0,1) on |x - x0| <= 20, noise = start.noise [1e-8], seed
    start.seed [0]; start.noise = 0 switches it off. An exactly symmetric start stays exactly
    symmetric in floating point (F is exactly odd), so an instability of the other parity is never
    seeded without it. The amplitude and seed used are written to start_info."""
    u0, x0, om, info = _build_start(S, cfg)
    st = cfg["start"]
    noise, seed = float(st.get("noise", NOISE)), int(st.get("seed", 0))
    if noise:
        rng = np.random.default_rng(seed)
        u0 = u0 + noise * rng.standard_normal(S.N) * (np.abs(S.x - x0) <= 20)
    info["noise"], info["seed"] = noise, seed
    return u0, x0, om, info


def _build_start(S, cfg):
    st = cfg["start"]
    if st["type"] == "orbit":
        assert S.medium == "grid", "orbit starts are C-grid (use orbits.smooth_attempt for C-smooth)"
        om = st["omega"]
        Ns = min(S.N, 2 * int(np.ceil(14.0 / O.kappa_grid(om))) + 41)
        Ns += Ns % 2
        us, nq, info = O.orbit_at(S.law, st["obj"], S.beta, om, Ns)
        u0 = np.zeros(S.N)
        off = S.N // 2 - Ns // 2
        u0[off:off + Ns] = us
        return u0, (-0.5 if st["obj"] == "sbend" else 0.0), om, {"N_orbit": Ns, "nq": nq,
                                                                  "newton": info}
    if st["type"] == "smooth_orbit":
        assert S.medium == "fd", "smooth_orbit starts are exact 'fd' states (tool b)"
        r = O.smooth_attempt(S.h, S.law, st["obj"], st["sigma"], st["peak_stretch"],
                             omega=st.get("omega"), L=st.get("L", 16.0), beta=S.beta,
                             floquet=False)
        if not r["newton"]["converged"]:
            raise RuntimeError("smooth_orbit start did not converge: %s" % r["newton"])
        us = np.asarray(r["profile_u0"])
        u0 = np.zeros(S.N)
        off = S.N // 2 - us.size // 2
        u0[off:off + us.size] = us
        info = {k: v for k, v in r.items() if not k.startswith("profile")}
        return u0, 0.0, r["omega"], info
    if st["type"] == "rwa":
        u0, x0 = ST.rwa(S, st["obj"], A=st.get("A"), omega=st.get("omega"))
        return u0, x0, None, {}
    if st["type"] == "jump":
        assert S.law == "G1" and S.smooth, "jump profiles are C-smooth G1 states"
        u0, x0, info = JP.start(S, st["obj"], st["omega"], st.get("a_top"), st.get("x_s", 0.25))
        return u0, x0, st["omega"], info
    if st["type"] == "gauss":
        u0, x0 = ST.gaussian(S, st["obj"], st["sigma"], A=st.get("A"),
                             peak_stretch=st.get("peak_stretch"))
        return u0, x0, None, {"A": float(np.max(np.abs(u0)))}
    raise ValueError(st["type"])


def measure_period(S, u0, v0, dt, T=60.0, fallback=None):
    """Nominal period from upward zero crossings of the projection on u0 over a short pre-run."""
    P = String(S.medium, S.law, S.N, h=S.h, beta=S.beta, sponge_len=S.sponge_len,
               sigma_max=S.sigma_max)
    P.set_state(u0, v0)
    nn = u0 @ u0
    ns = max(1, int(round(0.05 / dt)))
    ts, qs = [], []
    try:
        for _ in range(int(T / (ns * dt))):
            ts.append(P.t)
            qs.append(P.u @ u0 / nn)
            P.advance(dt, ns)
    except GuardTrip:
        pass
    r = G.rate_zero_crossings(np.array(ts), np.array(qs)) if len(ts) > 4 else None
    if r is None:
        r = fallback
    return 2 * np.pi / r


def window_halflength(S, u0, v0, x0):
    e = S.density(u0, v0)
    lh, _, _ = O.l_half(S.x, e, S.h)
    return max(10.0, 5.0 * lh), lh


# ---------------------------------------------------------------- evolution
def evolve(cfg, name=None, start=None, log=print):
    """One evolution with the Block 2.1 diagnostics. `start` = (u0, v0, x0, period, W, info) to
    reuse a start (the linear twin). Returns the result dict (also written to outputs/string/)."""
    name = name or cfg["name"]
    S = build(cfg)
    dt = float(cfg["dt"])
    Tcap = float(cfg["T"])
    if start is None:
        u0, x0, om_nom, sinfo = build_start(S, cfg)
        v0 = np.zeros(S.N)
        if cfg.get("push"):
            v0 = ST.push_velocity(S, u0, cfg["push"]["lambda"])
        period = 2 * np.pi / om_nom if om_nom else measure_period(
            S, u0, v0, dt, fallback=band_top(S.medium, S.h))
        W, lh0 = window_halflength(S, u0, v0, x0)
        W = cfg.get("W", W)
    else:
        u0, v0, x0, period, W, sinfo = start
        lh0 = None
    sdt_req = cfg.get("sample_dt", 0.1)
    ns = max(1, int(round(sdt_req / dt)))
    sdt = ns * dt
    nr = max(1, int(round(cfg.get("record_periods", 1) * period / sdt)))
    nsamp = int(np.ceil(Tcap / sdt))
    probes = cfg.get("probes", [0.0, 20.0, 50.0])
    pidx = [int(np.argmin(np.abs(S.x - (x0 + p)))) for p in probes]
    pmax = int(cfg.get("probe_max", 65536))
    fixed = np.abs(S.x - x0) <= W
    L = S.L
    prange = cfg.get("profile_range", 100.0)
    psel = np.where(np.abs(S.x - x0) <= prange)[0]
    pstep = max(1, int(np.ceil(psel.size / 2000)))
    psel = psel[::pstep]
    interior = S.L / 2 - (S.sponge_len if S.damping else 0.0)
    if interior < max(W, prange):
        raise ValueError("interior half-length %.1f is below max(W=%.1f, profile_range=%.1f)"
                         % (interior, W, prange))

    S.set_state(u0, v0)
    E0 = S.energy()
    Efix0 = float(S.density()[fixed].sum())
    nn = u0 @ u0
    params = dict(cfg, name=name, dt=dt, sample_dt=sdt, record_T=nr * sdt, W=W,
                  nominal_period=period, x0=x0, L=L, h=S.h,
                  band_top_scheme=band_top(S.medium, S.h),
                  ceiling_continuum=2.0 if S.medium == "grid" else 1.0)
    h0 = cfg_hash(params)
    ck = os.path.join(CKPT, "%s.npz" % name)
    st = {"k": 0, "c": x0, "cu": x0, "qprev": None, "tprev": None, "acc": [0.0, 0.0, 0.0, 0, 0.0, 0.0],
          "cross": [], "records": [], "sum_u": np.zeros(psel.size), "sum_d": np.zeros(psel.size),
          "sum_u2": np.zeros(psel.size), "nprof": 0,
          "probe": np.zeros((len(probes), min(pmax, nsamp + 1)), dtype=np.float32),
          "core": np.zeros(min(pmax, nsamp + 1)), "core_d": np.zeros(min(pmax, nsamp + 1)),
          "core_proj": np.zeros(min(pmax, nsamp + 1)), "guard": None, "stop": False, "nz": 0,
          "tail_rate": None, "tail_rate_source": None}
    ipk = int(np.argmax(np.abs(S.stretch(u0))))       # bond of the initial peak stretch
    xw = S.x[fixed]
    esum = np.zeros(xw.size)                           # record-summed density in the window
    # tail fit (review item 2): component of u over [t_tail, T] (Hann-tapered) at a bank of rates
    # around the record rate at t_tail; the one at the rate measured over that interval is fitted
    t_tail = cfg.get("tail_fit_from", 0.8) * Tcap
    bank = 1.0 + np.linspace(-0.1, 0.1, 81)
    zsum = np.zeros((bank.size, psel.size), dtype=complex)
    om_nom = 2 * np.pi / period
    obj = cfg["start"].get("obj", "bump")
    c0 = S.N // 2

    def core(S):
        """Block 4: bump = displacement at the centre; S-bend = stretch across the centre."""
        if obj == "bump":
            return S.u[int(np.argmin(np.abs(S.x - x0)))]
        if S.medium == "grid":
            return S.dd[c0 - 1]                        # u_c - u_{c-1}, centre at x = -1/2
        if S.medium == "fd":
            return 0.5 * (S.dd[c0 - 1] + S.dd[c0])     # (u_{c+1} - u_{c-1}) / 2h
        return S.dd[c0]
    snaps = sorted(cfg.get("snap_times", []))
    dpk = np.zeros(xw.size)                            # stretch at the record's peak sample
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=False)
        if str(z["hash"]) == h0:
            S.set_state(z["u"], z["v"], float(z["t"]))
            S.dd = np.array(z["d"])                # the carried stretch (differs in the sponge)
            S._accel_now()
            S.absorbed = float(z["absorbed"])
            meta = json.loads(str(z["meta"]))
            st.update(meta)
            for key in ("sum_u", "sum_d", "sum_u2", "probe", "core", "core_d", "core_proj"):
                st[key] = z[key]
            esum = np.array(z["esum"])
            zsum = np.array(z["zsum"])
            dpk = np.array(z["dpk"])
            log("%s: resumed at t=%.1f" % (name, S.t))
    snaps = [s for s in snaps if s > S.t - 0.5 * sdt or S.t == 0.0]
    t_wall = time.time()
    t_ck = time.time()
    ck_every = cfg.get("ckpt_every_s", 300)

    def track(e, c):
        for _ in range(2):
            dx = (S.x - c + L / 2) % L - L / 2
            m = np.abs(dx) <= W
            sw = e[m].sum()
            c = c + (e[m] @ dx[m]) / sw if sw > 0 else c
        return c

    while st["k"] <= nsamp:
        k = st["k"]
        e = S.density()
        efix = float(e[fixed].sum())
        cnew = track(e, st["c"])
        st["cu"] += (cnew - st["c"] + L / 2) % L - L / 2
        st["c"] = (cnew + L / 2) % L - L / 2
        dxc = (S.x - st["c"] + L / 2) % L - L / 2
        eco = float(e[np.abs(dxc) <= W].sum())
        q = float(core(S))
        if st["qprev"] is not None and st["qprev"] < 0 <= q:
            st["cross"].append(st["tprev"] - st["qprev"] * (S.t - st["tprev"]) / (q - st["qprev"]))
        st["qprev"], st["tprev"] = q, S.t
        d = S.stretch()
        if k < st["core"].size:
            st["core"][k] = q
            st["core_d"][k] = d[ipk]
            st["core_proj"][k] = S.u @ u0 / nn
            st["probe"][:, k] = S.u[pidx]
        st["sum_u"] += S.u[psel]
        st["sum_d"] += d[psel]
        st["sum_u2"] += S.u[psel] ** 2
        st["nprof"] += 1
        if S.t >= t_tail - 0.5 * sdt:
            if st["tail_rate"] is None:
                rr = [r["rate_zc"] for r in st["records"] if r["rate_zc"] is not None
                      and r["t"] >= 0.5 * t_tail]
                st["tail_rate"] = float(np.median(rr)) if rr else om_nom
                st["tail_rate_source"] = ("median record rate on [t/2, t] at t = %.1f" % S.t
                                          if rr else "nominal (no crossings)")
            wgt = np.sin(np.pi * min(1.0, (S.t - t_tail) / max(Tcap - t_tail, 1e-12))) ** 2
            ph = np.exp(-1j * st["tail_rate"] * bank * S.t) * wgt
            zsum += ph[:, None] * S.u[psel][None, :]
            st["nz"] += wgt
        a = st["acc"]
        a[0] += efix
        a[1] += eco
        a[2] += st["cu"]
        a[3] += 1
        a[4] = max(a[4], float(np.max(np.abs(S.u[fixed]))))
        pkd = float(np.max(np.abs(d[fixed])))
        if pkd >= a[5]:
            a[5] = pkd
            dpk[:] = d[fixed]
        esum += e[fixed]
        while snaps and S.t >= snaps[0] - 0.5 * sdt:
            os.makedirs(CKPT, exist_ok=True)
            write_npz(os.path.join(CKPT, "%s_snap_t%g.npz" % (name, snaps[0])), x=S.x, u=S.u,
                      v=S.v, d=S.dd, t=S.t)
            snaps.pop(0)
        if a[3] == nr or k == nsamp or st["stop"]:
            cr = st["cross"]
            lh, rms, _ = O.l_half(xw, esum / a[3], S.h)
            rec = {"t": S.t, "t_mid": S.t - 0.5 * (a[3] - 1) * sdt, "E_fixed": a[0] / a[3],
                   "E_comoving": a[1] / a[3], "centroid": a[2] / a[3], "E_total": S.energy(),
                   "absorbed": S.absorbed, "peak_u_window": a[4], "peak_d_window": a[5],
                   "L_half_window": lh, "rms_width_window": rms,
                   "rate_zc": 2 * np.pi * (len(cr) - 1) / (cr[-1] - cr[0]) if len(cr) > 1 else None}
            esum[:] = 0.0
            rec["balance"] = (rec["E_total"] + rec["absorbed"] - E0) / E0
            if S.smooth:
                rec["k_first_moment_stretch"], rec["k_rms_stretch"] = G.wavenumber_moments(
                    S, d, fixed)
                rec.update(G.jump_stats(xw, dpk, S.h, S.medium))
            st["records"].append(rec)
            st["acc"] = [0.0, 0.0, 0.0, 0, 0.0, 0.0]
            st["cross"] = cr[-1:]
        st["k"] += 1
        if st["k"] > nsamp or st["stop"]:
            break
        chunk = (S.u.copy(), S.v.copy(), S.dd.copy(), S.t, S.absorbed)
        try:
            S.advance(dt, ns)
        except GuardTrip as gt:
            dg = {kk: vv for kk, vv in gt.diag.items() if not kk.endswith("last_good")}
            dg["energy_last_good"] = gt.diag["energy_last_good"]
            dg["t_last_good"] = gt.diag["t_last_good"]
            dg["rerun_half_dt"] = rerun_half_dt(S, chunk, dt, gt.diag["t"])
            st["guard"] = dg
            st["stop"] = True          # one more sample (the last good state) closes the record
            os.makedirs(CKPT, exist_ok=True)
            write_npz(os.path.join(CKPT, "%s_guard_state.npz" % name), u=gt.diag["u_last_good"],
                      v=gt.diag["v_last_good"], t=gt.diag["t_last_good"])
            log("%s: GUARD TRIP %s" % (name, dg))
        if time.time() - t_ck > ck_every:
            meta = {kk: st[kk] for kk in ("k", "c", "cu", "qprev", "tprev", "acc", "cross",
                                          "records", "nprof", "guard", "stop", "nz",
                                          "tail_rate", "tail_rate_source")}
            write_npz(ck, hash=h0, u=S.u, v=S.v, d=S.dd, t=S.t, absorbed=S.absorbed, esum=esum,
                      zsum=zsum, dpk=dpk,
                      meta=json.dumps(meta, default=_default), sum_u=st["sum_u"],
                      sum_d=st["sum_d"], sum_u2=st["sum_u2"], probe=st["probe"], core=st["core"],
                      core_d=st["core_d"], core_proj=st["core_proj"])
            t_ck = time.time()
            log("%s: t=%.1f / %.1f checkpoint" % (name, S.t, Tcap))
    wall = time.time() - t_wall
    recs = st["records"]
    t = np.array([r["t_mid"] for r in recs])
    Ef = np.array([r["E_fixed"] for r in recs])
    Ec = np.array([r["E_comoving"] for r in recs])
    frac = Ef / Efix0
    fco = Ec / Efix0
    T90, Td = G.t90_tdeath(t, frac, Ef)
    T90c, Tdc = G.t90_tdeath(t, fco, Ec)
    nk = min(st["k"], st["core"].size)
    tq = np.arange(nk) * sdt
    rz = G.rate_zero_crossings(tq, st["core"][:nk])
    rs, rres = G.rate_spectrum(tq, st["core"][:nk]) if nk > 16 else (None, None)
    rzd = G.rate_zero_crossings(tq, st["core_d"][:nk])
    rsd = G.rate_spectrum(tq, st["core_d"][:nk])[0] if nk > 16 else None
    rzp = G.rate_zero_crossings(tq, st["core_proj"][:nk])
    rr = np.array([np.nan if r["rate_zc"] is None else r["rate_zc"] for r in recs])
    n10 = max(1, len(recs) // 10)
    early = float(np.nanmedian(rr[:n10])) if np.any(np.isfinite(rr[:n10])) else None
    late = float(np.nanmedian(rr[-n10:])) if np.any(np.isfinite(rr[-n10:])) else None
    t_end_rec = np.array([r["t"] for r in recs])
    cyc = G.cycles(t_end_rec, rr, 2 * np.pi / period)
    rate = late or rz or (2 * np.pi / period)
    in_fit = rr[t_end_rec >= t_tail]
    late_fit = float(np.nanmedian(in_fit)) if np.any(np.isfinite(in_fit)) else None
    r0 = st["tail_rate"] or om_nom
    jb = int(np.argmin(np.abs(r0 * bank - (late_fit or r0))))
    tail_rate = float(r0 * bank[jb])
    amp_rate = 2 * np.abs(zsum[jb]) / max(st["nz"], 1e-300)
    kap, kap_o1 = G.tail_kappa(S.x[psel] - x0, amp_rate, 2 * (lh0 or 1.0), tail_rate, S.medium, W)
    tail_bank_edge = jb in (0, bank.size - 1)
    S2 = String(S.medium, S.law, S.N, h=S.h, beta=S.beta)
    nl0 = (S2.energy(u0, v0) - S2.quadratic_energy(u0, v0)) / S2.energy(u0, v0)
    summary = {
        "E0": E0, "E_fixed0": Efix0, "L_half0_instant": lh0, "W": W, "cap": Tcap,
        "t_end": S.t, "guard": st["guard"],
        "frac_end": float(frac[-1]), "frac_comoving_end": float(fco[-1]),
        "frac_min": float(frac.min()), "frac_comoving_min": float(fco.min()),
        "T_90": T90, "T_death": Td, "T_90_comoving": T90c, "T_death_comoving": Tdc,
        "cycles_at_end": float(cyc[-1]),
        "cycles_note": "integral of the record rates over time (gaps filled from neighbours)",
        "T_90_cycles": G.at_time(T90, t_end_rec, cyc),
        "T_death_cycles": G.at_time(Td, t_end_rec, cyc),
        "rate_early": early, "rate_late": late,
        "rate_note": "early/late: median of the records' zero-crossing rates of the core "
                     "coordinate over the first/last 10% of records",
        "rate_zero_crossings_core_first_window": rz, "rate_spectrum_core_first_window": rs,
        "spectral_resolution_first_window": rres,
        "rate_zero_crossings_peak_stretch_first_window": rzd,
        "rate_spectrum_peak_stretch_first_window": rsd,
        "rate_zero_crossings_projection_first_window": rzp,
        "first_window_span": float(tq[-1]) if nk else None,
        "core_coordinate": "displacement at centre" if obj == "bump" else "stretch across centre",
        "tail_kappa_at_late_rate": kap, "tail_kappa_O1_at_late_rate": kap_o1,
        "tail_fit_rate": tail_rate, "tail_fit_rate_bank_centre": r0,
        "tail_fit_rate_source": "rate measured over the fit interval (median of its records), "
                                "nearest of 81 bank rates within +-10% of the record rate at "
                                "the start of the interval" + (" (AT BANK EDGE)" if tail_bank_edge
                                                               else ""),
        "tail_fit_interval": [t_tail, S.t], "rate_late_over_fit_interval": late_fit,
        "nominal_rate": om_nom,
        "rate_late_over_ceiling": rate / (2.0 if S.medium == "grid" else 1.0),
        "rate_early_over_ceiling": (early / (2.0 if S.medium == "grid" else 1.0)) if early else None,
        "rate_late_over_band_top_scheme": rate / band_top(S.medium, S.h),
        "nonlinear_share_t0": float(nl0),
        "max_abs_balance": float(max(abs(r["balance"]) for r in recs)),
        "centroid_start": float(recs[0]["centroid"]), "centroid_end": float(recs[-1]["centroid"]),
        "wall_s": wall,
    }
    for key in ("T_90", "T_death", "T_90_comoving", "T_death_comoving"):
        if summary[key] is None:
            summary[key] = "> cap"
    npf = max(st["nprof"], 1)
    out = {"params": params, "start_info": sinfo, "summary": summary,
           "records": recs,
           "profiles": {"x_rel": S.x[psel] - x0, "mean_u": st["sum_u"] / npf,
                        "mean_d": st["sum_d"] / npf, "rms_u": np.sqrt(st["sum_u2"] / npf),
                        "amp_u_at_late_rate": amp_rate},
           "series_file": "%s_series.npz" % name}
    write_npz(os.path.join(OUT, "%s_series.npz" % name), t=tq, core=st["core"][:nk],
              core_stretch=st["core_d"][:nk], core_projection=st["core_proj"][:nk],
              probes=st["probe"][:, :nk], probe_dist=np.array(probes))
    write_json(os.path.join(OUT, "%s.json" % name), out)
    if os.path.exists(ck):
        os.remove(ck)
    if cfg.get("twin", True) and S.law != "lin" and start is None:
        tw = dict(cfg, law="lin", twin=False)
        out["twin"] = evolve(tw, name=name + "_twin",
                             start=(u0, v0, x0, period, W, {"twin_of": name}), log=log)["summary"]
        write_json(os.path.join(OUT, "%s.json" % name), out)
    log("%s: done t=%.1f frac_end=%.4f wall %.1fs" % (name, S.t, summary["frac_end"], wall))
    return out


def rerun_half_dt(S, chunk, dt, t_fail):
    """Guard check: rerun the last stretch (from the state at the start of the sample chunk in which
    the guard tripped) at dt/2, up to t_fail + max(10 dt, 1). A trip that disappears or moves by
    more than dt is labelled numerical (it moves with dt)."""
    u, v, d, t0, ab = chunk
    R = String(S.medium, S.law, S.N, h=S.h, beta=S.beta, sponge_len=S.sponge_len,
               sigma_max=S.sigma_max)
    R.u, R.v, R.dd, R.t, R.absorbed = u.copy(), v.copy(), d.copy(), t0, ab
    R._accel_now()
    h2 = 0.5 * dt
    n = int(np.ceil((t_fail - t0 + max(10 * dt, 1.0)) / h2))
    out = {"from_t": float(t0), "dt_half": h2, "trip_t_dt": float(t_fail)}
    try:
        R.advance(h2, n)
        out.update({"trip_t_half_dt": None,
                    "label": "numerical (no trip at dt/2 up to t = %.4g)" % R.t})
    except GuardTrip as g2:
        shift = g2.diag["t"] - t_fail
        out.update({"trip_t_half_dt": g2.diag["t"], "trip_time_shift": shift,
                    "stretch_half_dt": g2.diag["stretch"], "x_bond_half_dt": g2.diag["x_bond"],
                    "label": "numerical (trip time moves with dt)" if abs(shift) > dt
                    else "persists at dt/2 (trip time within dt)"})
    return out


def push_summary(res, dist_target=100.0):
    recs = res["records"]
    t = np.array([r["t_mid"] for r in recs])
    c = np.array([r["centroid"] for r in recs])
    E0 = res["summary"]["E_fixed0"]
    fco = np.array([r["E_comoving"] for r in recs]) / E0
    n0 = len(t) // 10
    speed = float(np.polyfit(t[n0:], c[n0:], 1)[0]) if len(t) - n0 > 2 else None
    dist = abs(c[-1] - c[n0])
    loss = (fco[n0] - fco[-1]) * E0 / dist if dist > 1 else None
    ib = np.where(fco < 0.5)[0]
    ic = np.where(np.abs(c - c[0]) >= dist_target)[0]
    return {"speed": speed, "distance": float(dist), "energy_lost_per_unit_distance": loss,
            "broke": bool(ib.size), "t_break": float(t[ib[0]]) if ib.size else None,
            "crossed_target": bool(ic.size),
            "frac_comoving_at_target": float(fco[ic[0]]) if ic.size else None,
            "frac_comoving_end": float(fco[-1])}


def cost_table(cases, periods=20, log=print):
    rows = []
    for cs in cases:
        cfg = dict(cs, twin=False, T=1.0, name="_cost")
        S = build(cfg)
        u0, x0, om, _ = build_start(S, cfg)
        if om is None:
            om = cs.get("omega_nominal", 1.1 if S.smooth else 2.3)
        T = periods * 2 * np.pi / om
        cfg["T"] = T
        t0 = time.time()
        evolve(cfg, name="_cost", log=lambda *a: None)
        w = time.time() - t0
        rows.append({"case": cs, "omega_nominal": om, "N": S.N, "wall_s": w, "T": T,
                     "wall_s_per_1000_cycles": w / periods * 1000})
        log("cost %s %s h=%s N=%d: %.1f s per 1000 cycles" % (S.medium, S.law, S.h, S.N,
                                                               rows[-1]["wall_s_per_1000_cycles"]))
    for f in ("_cost.json", "_cost_series.npz"):
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            os.remove(p)
    return rows


def run_job(job, log=print):
    kind = job["kind"]
    if kind == "family":
        res = O.family(job.get("medium", "grid"), job["law"], job["obj"], beta=job.get("beta", 0.0),
                       N=job.get("N"), omega_start=job.get("omega_start", 2.3),
                       omega_min=job.get("omega_min", 2.01), d_stop=job.get("d_stop", 0.99),
                       omega_stop=job.get("omega_stop"), floquet=job.get("floquet", True),
                       log=log)
        out = {"params": job, "result": res}
        return write_json(os.path.join(OUT, "%s.json" % job["name"]), out)
    if kind == "smooth_orbits":
        rows = []
        for h in job.get("hs", [0.25, 0.125, 0.0625]):
            for sg in job.get("sigmas", [0.5]):
                for a in job.get("peak_stretches", [0.8]):
                    r = O.smooth_attempt(h, job["law"], job["obj"], sg, a, omega=job.get("omega"),
                                         L=job.get("L", 16.0), beta=job.get("beta", 0.0),
                                         floquet=job.get("floquet", True))
                    log("smooth %s %s h=%g sigma=%g a=%g: converged=%s L_half=%s" % (
                        job["law"], job["obj"], h, sg, a, r["newton"]["converged"],
                        r.get("L_half")))
                    rows.append(r)
        return write_json(os.path.join(OUT, "%s.json" % job["name"]), {"params": job, "rows": rows})
    if kind == "evolve":
        evolve(job, log=log)
        return os.path.join(OUT, "%s.json" % job["name"])
    if kind == "push":
        rows = []
        for lam in job["push"]["lambdas"]:
            cfg = dict(job, kind="evolve", push={"lambda": lam}, twin=job.get("twin", True))
            nm = "%s_l%g" % (job["name"], lam)
            r = evolve(cfg, name=nm, log=log)
            rows.append(dict(push_summary(r, job.get("dist_target", 100.0)), **{"lambda": lam,
                                                                               "file": nm}))
        return write_json(os.path.join(OUT, "%s.json" % job["name"]), {"params": job, "rows": rows})
    if kind == "cost":
        rows = cost_table(job["cases"], job.get("periods", 20), log=log)
        return write_json(os.path.join(OUT, "%s.json" % job["name"]), {"params": job, "rows": rows})
    raise ValueError(kind)


def main(argv):
    jobs = json.load(open(argv[1]))
    if isinstance(jobs, dict):
        jobs = [jobs]
    for j in jobs:
        print(run_job(j), flush=True)


if __name__ == "__main__":
    main(sys.argv)
