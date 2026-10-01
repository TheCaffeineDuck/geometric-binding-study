"""WP-A radial run driver (Session 2): one single-frequency profile evolved with the
full radial equation, absorber on, optional checkpoint/resume.

Equation, scheme and E_loc diagnostic are those of Session 1 (swirl.family.radial):
u = r phi, 4th-order FD, RK4, quadratic sponge sigma_max ((r - R_s)/(R_box - R_s))^2.
E_in(r_c) = 4 pi int_0^rc e r^2 dr with e from np.gradient and the trapezoid rule
(Radial.energy_inside). Its quadrature error is O(dr^2): about 1e-4 of E at dr = 0.05
(F-003; halving dr gave 3.2e-5 then 7.9e-6).

Units: Set-B-like, m = 1, g4 = 0.30, g6 = 0.09 beta. Initial data phi = Phi(r),
phi_t = 0, Phi from solve_profile (rescaled) / sqrt(g4).

Box and sponge scale with the lump size. At fixed alpha^2 = 0.3 beta (1 - omega^2)
the profile is universal with length scale L = sqrt(0.3 beta) (PRIOR_ART 1.3), so the
sponge start and the sponge width are scaled by max(1, sqrt(beta / (4/3))) from the
Session 1 values (R_s = 100, R_box = 200, validated at beta <= 4/3). dr and dt are
not scaled: the radiation wavenumber k_3 = sqrt(9 omega^2 - 1) <= 2.83 is set by
omega ~ m, not by beta (44 points per wavelength at dr = 0.05).

frac = E_in(3 hw) / E_total(t = 0). T_90 = first t with frac < 0.90. T_death = first
t > 50 with E_in(3 hw) < 0.5 E_in(3 hw, t = 50) (SESSION_02_PREDICTIONS.md, Block 1).
"""

import json
import os

import numpy as np

from swirl.family.profile import profile_curve, solve_profile
from swirl.family.radial import Radial

M, G4 = 1.0, 0.30
DR, DT, SIGMA_MAX = 0.05, 0.0125, 1.0
REC_DT = 0.5
T_REF_DEATH = 50.0
BETA_REF = 4.0 / 3.0
SCHMITT = 0.3          # arm the next upward crossing only after phi(0) < -0.3 x amplitude


def grid_for(beta, scale=1.0, dr=DR, dt=DT):
    """Box for this beta. scale = 2 gives the doubled box used by the box check."""
    s = max(1.0, np.sqrt(beta / BETA_REF))
    R_s = 100.0 * s * scale
    return dict(dr=dr, dt=dt, R_sponge=R_s, R_box=2.0 * R_s, sigma_max=SIGMA_MAX,
                sigma_profile="sigma_max((r-R_sponge)/(R_box-R_sponge))^2",
                scheme="u=r phi, 4th-order FD, odd reflection at r=0 and r=R_box, RK4",
                size_scale=float(s))


def profile(beta, omega, tol=1e-10):
    """Rescaled profile (dict from solve_profile) and its Set-B-like numbers."""
    p = solve_profile(omega, 1.0, 1.0, beta, tol=tol)
    return p


def r_lump(hw, kappa):
    """Radius enclosing the lump including its tail: tail energy ~ exp(-2 kappa (R - hw))
    below 1e-4 needs R - hw > 4.6 / kappa; at least 3 hw."""
    return float(max(3.0 * hw, hw + 4.6 / kappa))


def _schmitt(state, t_first, dt, rec):
    """Upward zero crossings of phi(0,t), counted only after the signal has gone below
    -SCHMITT x amplitude since the last counted crossing (removes the extra crossings
    that strong harmonics add). Appends crossing times to state['tc']."""
    amp = state["amp"]
    v = np.concatenate([[state["prev_v"]], rec])
    t = np.concatenate([[state["prev_t"]], t_first + dt * np.arange(rec.size)])
    h = SCHMITT * amp
    arm = np.where(v < -h)[0]
    up = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
    ev = sorted([(i, 0) for i in arm] + [(i, 1) for i in up])   # arm before a crossing at the same i
    armed = state["armed"]
    for i, kind in ev:
        if kind == 0:
            armed = True
        elif armed:
            tc = t[i] - v[i] * (t[i + 1] - t[i]) / (v[i + 1] - v[i])
            state["tc"].append(float(tc))
            armed = False
    state["armed"] = armed
    state["prev_v"] = float(rec[-1])
    state["prev_t"] = float(t[-1])
    # amplitude: max |phi(0)| over the last 20 records (10 time units, > one period)
    state["amp_hist"].append(float(np.max(np.abs(rec))))
    del state["amp_hist"][:-20]
    state["amp"] = max(state["amp_hist"])


def omega_series(tc):
    tc = np.asarray(tc)
    if tc.size < 2:
        return np.array([]), np.array([])
    return 0.5 * (tc[1:] + tc[:-1]), 2.0 * np.pi / np.diff(tc)


def schmitt_offline(t, v, amp=None):
    """Same trigger on a whole sampled signal (for tests). Returns crossing times."""
    st = dict(prev_v=float(v[0]), prev_t=float(t[0]), armed=False, tc=[],
              amp=float(np.max(np.abs(v))) if amp is None else amp, amp_hist=[])
    dt = t[1] - t[0]
    _schmitt(st, t[1], dt, np.asarray(v[1:], dtype=float))
    return np.array(st["tc"])


def energy_parts(s):
    """Whole-box energy split: kinetic, gradient, mass, E4, E6 (physical density)."""
    phi, pt = s.fields()
    pr = np.gradient(phi, s.dr, edge_order=2)
    pr[0] = 0.0
    p2 = phi * phi
    w = 4.0 * np.pi * s.r**2
    I = lambda f: float(np.sum(0.5 * (f[1:] + f[:-1])) * s.dr)
    return dict(kin=I(w * 0.5 * pt * pt), grad=I(w * 0.5 * pr * pr), mass=I(w * 0.5 * s.m**2 * p2),
                E4=I(w * (-s.g4 / 24.0) * p2 * p2), E6=I(w * (s.g6 / 720.0) * p2**3))


def nonlinear_shares(parts):
    E = parts["kin"] + parts["grad"] + parts["mass"] + parts["E4"] + parts["E6"]
    absum = parts["kin"] + parts["grad"] + parts["mass"] + abs(parts["E4"]) + abs(parts["E6"])
    return dict(signed=(parts["E4"] + parts["E6"]) / E,
                absolute=(abs(parts["E4"]) + abs(parts["E6"])) / absum)


def run(beta, omega, T, linear=False, grid=None, rcs=None, ckpt=None, ckpt_every=1000.0,
        stop_at_death=True, prof=None, stop_after=None):
    """Evolve the profile (beta, omega). Returns a dict with parameters, series and
    summary. rcs[0] is 3 hw (frac, T_90, T_death); rcs[1] is the lump radius r_lump.

    ckpt: path of an .npz checkpoint; if it exists and matches the parameters the run
    resumes from it. stop_after: stop (and checkpoint) at this time without finishing
    (used to test resume).
    """
    if prof is None:
        prof = profile(beta, omega)
    hw, kappa = prof["half_width"], prof["kappa"]
    if grid is None:
        grid = grid_for(beta)
    if rcs is None:
        rcs = [3.0 * hw, r_lump(hw, kappa)]
    g4, g6 = (0.0, 0.0) if linear else (G4, beta * G4**2)
    params = dict(beta=beta, omega_profile=float(omega), linear=bool(linear), m=M,
                  g4_run=g4, g6_run=g6, g4_units=G4, T=float(T), rcs=[float(x) for x in rcs],
                  half_width=hw, kappa=kappa, rec_dt=REC_DT, stop_at_death=bool(stop_at_death),
                  initial_data="phi = Phi(r) single-frequency profile (solve_profile tol 1e-10) / "
                               "sqrt(g4), phi_t = 0", **grid)
    pkey = json.dumps(params, sort_keys=True)
    s = Radial(grid["R_box"], grid["dr"], M, g4, g6, sponge_start=grid["R_sponge"],
               sigma_max=grid["sigma_max"])
    nrec = int(round(REC_DT / grid["dt"]))
    st = None
    if ckpt is not None and os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=False)
        if str(z["pkey"]) != pkey:
            raise RuntimeError("checkpoint %s has different parameters" % ckpt)
        s.u, s.v, s.t = z["u"].copy(), z["v"].copy(), float(z["t"])
        st = json.loads(str(z["state"]))
    if st is None:
        s.set_phi(lambda r: profile_curve(prof, r) / np.sqrt(G4))
        Ein0, E0 = s.energy_inside(rcs)
        phi, _ = s.fields()
        parts0 = energy_parts(s)
        st = dict(times=[0.0], E_in=[list(map(float, Ein0))], E_tot=[float(E0)],
                  E_h=[float(s.energy_discrete())], E0=float(E0), Eh0=float(s.energy_discrete()),
                  parts0=parts0, E_ref50=None, T_90=None, T_death=None,
                  trig=dict(prev_v=float(phi[0]), prev_t=0.0, armed=False, tc=[],
                            amp=float(abs(phi[0])), amp_hist=[float(abs(phi[0]))]),
                  nl_t=[0.0], nl_signed=[nonlinear_shares(parts0)["signed"]],
                  nl_abs=[nonlinear_shares(parts0)["absolute"]], done=False)
    nchunks = int(round(T / REC_DT))
    k = int(round(s.t / REC_DT))
    last_ck = s.t
    while k < nchunks and not st["done"]:
        t_first = s.t + grid["dt"]
        rec = s.step(grid["dt"], nrec)
        k += 1
        s.t = k * REC_DT if abs(s.t - k * REC_DT) < 1e-6 else s.t
        _schmitt(st["trig"], t_first, grid["dt"], rec)
        a, b = s.energy_inside(rcs)
        st["times"].append(float(s.t))
        st["E_in"].append(list(map(float, a)))
        st["E_tot"].append(float(b))
        st["E_h"].append(float(s.energy_discrete()))
        if k % 100 == 0:
            sh = nonlinear_shares(energy_parts(s))
            st["nl_t"].append(float(s.t))
            st["nl_signed"].append(sh["signed"])
            st["nl_abs"].append(sh["absolute"])
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
            _save_ckpt(ckpt, s, st, pkey)
            last_ck = s.t
        if stop_now:
            return dict(params=params, partial=True, t=s.t)
    return _result(params, st, s)


def _save_ckpt(path, s, st, pkey):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, u=s.u, v=s.v, t=s.t, state=json.dumps(st), pkey=pkey)
    os.replace(tmp, path)


def _result(params, st, s):
    t = np.array(st["times"])
    Ein = np.array(st["E_in"])
    tw, w = omega_series(st["trig"]["tc"])
    Eh = np.array(st["E_h"])
    T_end = float(t[-1])
    sel = (tw >= 50) & (tw <= 150)
    w_settled = float(np.median(w[sel])) if sel.any() else None
    sel = tw >= T_end - 100
    w_end = float(np.median(w[sel])) if sel.any() else None
    n_osc = len(st["trig"]["tc"])
    osc = lambda T: None if (T is None or w_settled is None) else T * w_settled / (2 * np.pi)
    periods_before = lambda T: None if T is None else int(np.sum(np.array(st["trig"]["tc"]) <= T))
    sh0 = nonlinear_shares(st["parts0"])
    return dict(
        params=params,
        E0=st["E0"], Eh0=st["Eh0"],
        regime=dict(omega_over_m_settled=w_settled, omega_over_m_end=w_end,
                    nonlinear_share_t0_signed=sh0["signed"],
                    nonlinear_share_t0_absolute=sh0["absolute"],
                    nonlinear_share_mean_signed=float(np.mean(st["nl_signed"])),
                    nonlinear_share_mean_absolute=float(np.mean(st["nl_abs"])),
                    energy_parts_t0=st["parts0"]),
        T_end=T_end, T_90=st["T_90"], T_death=st["T_death"],
        osc_T_90=osc(st["T_90"]), osc_T_death=osc(st["T_death"]), osc_T_end=osc(T_end),
        periods_counted_T_90=periods_before(st["T_90"]),
        periods_counted_T_death=periods_before(st["T_death"]), periods_counted_total=n_osc,
        frac_t0=float(Ein[0, 0] / st["E0"]), frac_end=float(Ein[-1, 0] / st["E0"]),
        E_ref50=st["E_ref50"],
        energy_check=dict(E_h_rise_max_rel=float(np.max(np.diff(Eh)) / Eh[0]) if Eh.size > 1 else 0.0,
                          note="E_h (exactly conserved without sponge) may only fall with the sponge on"),
        series=dict(t=t, E_in=Ein, E_tot=np.array(st["E_tot"]), E_h=Eh),
        omega_t=tw, omega=w, crossings=np.array(st["trig"]["tc"]),
        nl=dict(t=st["nl_t"], signed=st["nl_signed"], absolute=st["nl_abs"]),
    )


def window_fits(t, E, t0, W, t_stop, omega_t=None, omega=None):
    """Gamma = -d ln E/dt at the centre of consecutive windows [t0 + kW, t0 + (k+1)W]
    inside [t0, t_stop], from a least-squares quadratic in (t - t_centre); E is the fitted
    value at the centre. (A straight-line slope assigned to the window-mean E is biased by
    about (W dGamma/dt)^2-order terms; the centred quadratic removes that at no cost in
    slope variance, since the linear and quadratic terms are orthogonal on a symmetric
    window.) sigma = slope standard error with the residual scatter and one independent
    sample per oscillation period (n_eff = W / period)."""
    t, E = np.asarray(t), np.asarray(E)
    out = []
    a = t0
    while a + W <= t_stop + 1e-9:
        b = a + W
        sel = (t >= a - 1e-9) & (t <= b + 1e-9) & (E > 0)
        if sel.sum() >= 20:
            tt, y = t[sel] - 0.5 * (a + b), np.log(E[sel])
            c = np.polyfit(tt, y, 2)
            res = y - np.polyval(c, tt)
            c = c[1:]                     # slope and centre value
            wm = None
            if omega_t is not None and len(omega_t):
                ws = (np.asarray(omega_t) >= a) & (np.asarray(omega_t) <= b)
                wm = float(np.median(np.asarray(omega)[ws])) if ws.any() else None
            period = 2 * np.pi / wm if (wm and wm > 0) else 2 * np.pi
            n_eff = max(2.0, W / period)
            sig = float(np.std(res) * np.sqrt(12.0 / n_eff) / W)
            out.append(dict(t0=float(a), t1=float(b), E=float(np.exp(c[1])),
                            Gamma=float(-c[0]), sigma=sig, omega=wm))
        a = b
    return out


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(jsonable(obj), f, indent=1)
    os.replace(tmp, path)
    return os.path.abspath(path)
