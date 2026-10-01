"""Snapshot starts (WP-E): K identical lumps at arbitrary positions and phases, all cut
from one settled single lump.

Why: a single-frequency start Phi cos(theta), -omega Phi sin(theta) is a different lump for
each theta (F-009: slice energy 2843.8 at 0, 2605.2 at 90). A time shift of a settled lump
is the real-field counterpart of a phase rotation, so copies cut from one settled lump at
different times are the same lump at different phases.

Source (make_source)
--------------------
Stage 1: the engine (run.run) evolves one lump at the origin of a grid with the production
dx to t_s (checkpointed, resumable); the state at t_s is saved.
Stage 2 (cheap, from the t_s state): phi(0, t) is followed to the first upward zero crossing
t_up after t_s. omega = whole-period mean over [t_up - omega_span, t_up]. Settling guard:
|omega(window ending at t_up) - omega(window ending ~drift_lag earlier)|.

Phase convention. t_0 = t_up + pi/(2 omega) is the source time of phase 0, so that
phi(0, t) = Phi cos(omega (t - t_0)) + (odd harmonics in phase); phase 0 is a maximum of
phi(0, t). A copy at phase theta is the source state at time t_0 + theta/omega, so started
at t = 0 it oscillates as Phi cos(omega t + theta): the convention of the ansatz
(initial.py, lumps.lumps_init) and of the relative-phase estimator (lumps.summarize).

The stored reference is the source state at the step n_ref = floor(t_0/dt) (a step boundary
of the source run, reproduced bit for bit by rewinding from the t_s state). Any phase is
made on demand: from the reference, full RK4 steps of the source dt, then one final
partial step that lands exactly on tau = t_0 - n_ref dt + (theta mod 2 pi)/omega. Cost at
most one period of single-lump steps on the source grid per run start, for any number of
distinct phases (they are produced in one pass). No snapshot is stored per phase.

Crop loss is measured on the source over one period from the reference: E of the windowed
slice against E of the full box (worst over the period) and N = <int phi_t^2>/omega with
and without the window (window applied to phi and phi_t).

Placement (place)
-----------------
For each lump: the state at its phase is multiplied by the window w(r) (1 for r <= r_in,
cos^2 taper to 0 at r_out), cropped to a cube of half-size M = ceil((r_out + margin)/dx)
points about the source centre, shifted by the sub-grid remainder delta (|delta_a| <= dx/2)
with an exact spectral shift on the cube (periodic on the cube; the windowed field is zero
in the margin; Nyquist bins zeroed), and added into the target grid at the nearest grid
point (integer index shift, exact; periodic wrap). The target needs the same dx; a box of
another size is zero padding. With window None the whole source box is used unmultiplied,
and a zero shift on the same grid reproduces the source state bit for bit.
"""

import hashlib
import json
import os

import numpy as np
import scipy.fft as sfft

from swirl.engine.evolver import Evolver
from swirl.engine.initial import TAIL_TOL, radial_spline

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WINDOW_DEFAULT = [14.0, 18.0]
WINDOWS_MEASURED = [[14.0, 18.0], [16.0, 20.0], [18.0, 22.0], [20.0, 24.0], [22.0, 26.0], [24.0, 28.0],
                    [26.0, 30.0]]
MARGIN_DEFAULT = 2.0
GUARD_TOL = {"omega_drift": 1e-4, "crop_loss": 1e-4}


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def window_key(window):
    return "%g,%g" % tuple(window)


def window_profile(r, r_in, r_out):
    u = np.clip((r - r_in) / (r_out - r_in), 0.0, 1.0)
    return np.cos(0.5 * np.pi * u) ** 2


def state_hash(phi, pi):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(phi, dtype="<f8").tobytes())
    h.update(np.ascontiguousarray(pi, dtype="<f8").tobytes())
    return h.hexdigest()


def _atomic_npz(path, **arrays):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _atomic_json(obj, path):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def _evolver(p, threads):
    return Evolver(p["N"], p["L"], p["m"], p["g4"], p["g6"], absorber=p["absorber"],
                   kfilter=p["kfilter"], threads=threads)


def _radial_grid(n, dx, M):
    x = dx * np.arange(-M, n - M)
    return np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

def _whole_period_omega(tz, t_end, span):
    """omega over the zero crossings in [t_end - span, t_end] (even number of half periods)."""
    sel = tz[(tz >= t_end - span - 1e-9) & (tz <= t_end + 1e-9)]
    n = sel.size - 1
    if n % 2:
        sel = sel[1:]
        n -= 1
    if n < 2:
        raise ValueError("fewer than one period in the omega window")
    return float(np.pi * n / (sel[-1] - sel[0])), float(sel[0]), float(sel[-1])


def make_source(name, cfg, t_s, out_dir, windows=WINDOWS_MEASURED, checkpoint_dir=None,
                omega_span=20.0, drift_lag=100.0, verbose=False):
    """Make (or reuse) the source `name` in out_dir. cfg: an engine config of one lump at the
    origin (init radial[0] centred at 0, no lumps block). The crop loss is measured for each
    window in `windows` (the first is the default). Returns the meta dict."""
    from swirl.engine.run import full_config, run, zero_crossings
    os.makedirs(out_dir, exist_ok=True)
    c1 = full_config({**cfg, "t_end": float(t_s), "center": [0.0, 0.0, 0.0], "point_every": 1})
    if c1.get("lumps"):
        raise ValueError("source config must not have a lumps block")
    if c1["N"] % 2 or np.any(np.asarray(c1["init"]["radial"][0].get("center", [0, 0, 0])) != 0):
        raise ValueError("source: even N and a lump at the origin (a grid point) are required")
    ts_path = os.path.join(out_dir, name + "_ts.npz")
    st1 = None
    if os.path.exists(ts_path):
        with np.load(ts_path, allow_pickle=False) as z:
            m1 = json.loads(str(z["meta"]))
            if m1["params"] == c1:
                st1 = (z["phi"].copy(), z["pi"].copy(), m1)
    if st1 is None:
        res = run(c1, out_path=os.path.join(out_dir, name + "_run.json"),
                  checkpoint_dir=checkpoint_dir, verbose=verbose)
        ev = res["_evolver"]
        m1 = {"params": c1, "step": ev.step_count, "point": res["point"]}
        _atomic_npz(ts_path, phi=ev.phi, pi=ev.pi, meta=np.array(json.dumps(m1)))
        st1 = (ev.phi.copy(), ev.pi.copy(), m1)
        del res, ev
    phi_s, pi_s, m1 = st1
    dt, N = c1["dt"], c1["N"]
    n_s = m1["step"]
    ev = _evolver(c1, c1["threads"])
    ev.phi[...] = phi_s
    ev.pi[...] = pi_s
    c = N // 2
    tp = list(m1["point"]["t"])
    yp = list(m1["point"]["phi"])
    # follow phi(0, t) to the first upward crossing after t_s (plus 3 steps for the refinement)
    n, extra = n_s, None
    while True:
        ev.step(dt)
        n += 1
        tp.append(n * dt)
        yp.append(float(ev.phi[c, c, c]))
        if extra is None and yp[-2] < 0.0 <= yp[-1] and tp[-2] >= t_s:
            extra = 3
        elif extra is not None:
            extra -= 1
            if extra == 0:
                break
    tz = zero_crossings(tp, yp)
    tpa, ypa = np.array(tp), np.array(yp)
    up = [z for z in tz if z >= t_s and np.interp(z - 0.25 * dt, tpa, ypa) < 0.0]
    t_up = float(up[0])
    om, za, zb = _whole_period_omega(tz, t_up, omega_span)
    z_lag = float(tz[np.argmin(np.abs(tz - (t_up - drift_lag)))])
    om_lag, _, _ = _whole_period_omega(tz, z_lag, omega_span)
    sel = tz[(tz >= t_up - drift_lag - 1e-9) & (tz <= t_up + 1e-9)]
    per = 2 * np.pi / (sel[2:] - sel[:-2])
    t0 = t_up + np.pi / (2 * om)
    n_ref = int(np.floor(t0 / dt + 1e-9))
    # rewind to the reference step
    ev.phi[...] = phi_s
    ev.pi[...] = pi_s
    for _ in range(n_ref - n_s):
        ev.step(dt)
    phi_ref, pi_ref = ev.phi.copy(), ev.pi.copy()
    parts = ev.energy_parts()
    absq = abs(parts["E_quartic"]) + abs(parts["E_sextic"])
    share = absq / (parts["E_kin"] + parts["E_grad"] + parts["E_mass"] + absq)
    # crop loss over one period from the reference, for each window
    r = _radial_grid(N, ev.dx, c)
    ws = [window_profile(r, *wd) for wd in windows]
    P = 2 * np.pi / om
    nP = int(np.ceil(P / dt))
    absorber = c1["absorber"]
    r_in = (absorber.get("r_out") or c1["L"] / 2) - absorber["width"] if absorber else None
    shells = [(r > wd[1]) & (r < r_in) if r_in else (r > wd[1]) for wd in windows]
    lossE = [[] for _ in windows]
    Iwin = [[] for _ in windows]
    e_shell = [[] for _ in windows]
    Ifull = []
    for k in range(nP):
        Ef = ev.energy()
        ed = ev.energy_density()
        Ifull.append(float(np.sum(ev.pi ** 2)) * ev.dV)
        for i, w in enumerate(ws):
            lossE[i].append(1.0 - ev.energy_parts(w * ev.phi, w * ev.pi)["E_total"] / Ef)
            Iwin[i].append(float(np.sum((w * ev.pi) ** 2)) * ev.dV)
            e_shell[i].append(float(np.sum(ed[shells[i]])) * ev.dV / Ef)
        ev.step(dt)
    crop = {}
    for i, wd in enumerate(windows):
        crop[window_key(wd)] = {"window": list(map(float, wd)),
                                "crop_loss_E_max": float(np.max(np.abs(lossE[i]))),
                                "crop_loss_E_mean": float(np.mean(lossE[i])),
                                "crop_loss_N": float(1.0 - np.sum(Iwin[i]) / np.sum(Ifull)),
                                "shell_energy_fraction_max": float(np.max(e_shell[i]))}
    prof = c1["init"]["radial"][0]
    meta = {
        "name": name, "params": c1, "t_s": float(t_s), "dt": dt, "dx": ev.dx,
        "omega": om, "omega_window": [za, zb], "omega_span": omega_span,
        "omega_lag": om_lag, "omega_lag_window_end": z_lag, "omega_drift": abs(om - om_lag),
        "omega_per_period_last_lag": {"min": float(per.min()), "max": float(per.max())},
        "t_up": t_up, "t0": float(t0), "n_ref": n_ref, "t_ref": n_ref * dt, "tau0": float(t0 - n_ref * dt),
        "omega_over_m": om / c1["m"], "nonlinear_share_ref": share, "E_ref": parts["E_total"],
        "phi_center_ref": float(phi_ref[c, c, c]),
        "window": list(map(float, windows[0])), "crop": crop,
        **{k: v for k, v in crop[window_key(windows[0])].items() if k != "window"},
        "profile": {"r": list(map(float, prof["r"])), "Phi": list(map(float, prof["Phi"]))},
        "Phi0": float(abs(prof["Phi"][0])),
        "convention": "copy at theta = source state at t0 + theta/omega; phi_copy(t) ~ Phi cos(omega t + theta)",
    }
    meta["sha256"] = state_hash(phi_ref, pi_ref)
    path = os.path.join(out_dir, name + ".npz")
    _atomic_npz(path, phi=phi_ref, pi=pi_ref, meta=np.array(json.dumps(meta)))
    _atomic_json(meta, os.path.join(out_dir, name + ".json"))
    if verbose:
        print("source %s: omega %.7f drift %.2e crop E %.2e N %.2e" % (
            name, om, meta["omega_drift"], meta["crop_loss_E_max"], meta["crop_loss_N"]), flush=True)
    return meta


def load_source(path, sha256=None):
    with np.load(resolve(path), allow_pickle=False) as z:
        phi, pi, meta = z["phi"].copy(), z["pi"].copy(), json.loads(str(z["meta"]))
    h = state_hash(phi, pi)
    if h != meta["sha256"] or (sha256 is not None and h != sha256):
        raise ValueError("snapshot source hash mismatch: %s" % path)
    return phi, pi, meta


def load_meta(path):
    with np.load(resolve(path), allow_pickle=False) as z:
        return json.loads(str(z["meta"]))


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_taus(meta, thetas):
    """Times after the reference step at which the source has the phases `thetas`."""
    return [meta["tau0"] + float(np.mod(th, 2 * np.pi)) / meta["omega"] for th in thetas]


def phase_states(src, taus, consume, threads=4):
    """Evolve the reference once through the times `taus` (dict key -> time after the
    reference step, from phase_taus); consume(key, phi, pi, rec) is called for each key
    in order of time."""
    phi_ref, pi_ref, meta = src
    p, dt = meta["params"], meta["dt"]
    ev = _evolver(p, threads)
    ev.phi[...] = phi_ref
    ev.pi[...] = pi_ref
    n = 0
    for th, tau in sorted(taus.items(), key=lambda kv: kv[1]):
        if tau < 0:
            raise ValueError("snapshot time before the reference step")
        nj = int(np.floor(tau / dt + 1e-9))
        while n < nj:
            ev.step(dt)
            n += 1
        dlt = tau - nj * dt
        if dlt > 1e-9 * dt:
            keep = (ev.phi.copy(), ev.pi.copy())
            ev.step(dlt)
            consume(th, ev.phi, ev.pi, {"tau": tau, "full_steps": nj, "partial_step": dlt})
            ev.phi[...] = keep[0]
            ev.pi[...] = keep[1]
        else:
            consume(th, ev.phi, ev.pi, {"tau": tau, "full_steps": nj, "partial_step": 0.0})


# ---------------------------------------------------------------------------
# Window, shift, resample, placement
# ---------------------------------------------------------------------------

def spectral_shift(f, delta, dx):
    """f(x - delta) on a periodic cube (Nyquist bins zeroed). Returns (g, nyquist_fraction)."""
    n = f.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    kr = 2 * np.pi * np.fft.rfftfreq(n, d=dx)
    fh = sfft.rfftn(f)
    tot = float(np.sum(np.abs(fh) ** 2))
    ny = 0.0
    if n % 2 == 0:
        m = np.zeros(fh.shape, bool)
        m[n // 2, :, :] = True
        m[:, n // 2, :] = True
        m[:, :, -1] = True
        ny = float(np.sum(np.abs(fh[m]) ** 2))
        fh[m] = 0.0
    ph = np.exp(-1j * (k[:, None, None] * delta[0] + k[None, :, None] * delta[1] + kr[None, None, :] * delta[2]))
    return sfft.irfftn(fh * ph, s=f.shape), (np.sqrt(ny / tot) if tot > 0 else 0.0)


def resample(f, n_new):
    """Band-limited interpolation of a periodic cube onto n_new points per side (same box)."""
    n = f.shape[0]
    fh = sfft.fftn(f)
    if n % 2 == 0:
        fh[n // 2, :, :] = 0
        fh[:, n // 2, :] = 0
        fh[:, :, n // 2] = 0
    out = np.zeros((n_new,) * 3, complex)
    h = n // 2
    lo = [np.r_[0:h + (n % 2)], np.r_[n - h:n]]
    lo_new = [np.r_[0:h + (n % 2)], np.r_[n_new - h:n_new]]
    for a in range(2):
        for b in range(2):
            for c in range(2):
                out[np.ix_(lo_new[a], lo_new[b], lo_new[c])] = fh[np.ix_(lo[a], lo[b], lo[c])]
    return np.real(sfft.ifftn(out)) * (n_new / n) ** 3


def place(snap, coords, L, threads=4):
    """phi, pi on the target grid and an info dict, for the init entry `snap`:
    {source, sha256, centers, phases (rad), window ([r_in, r_out] or null), margin, resample};
    "taus" (source times after the reference step) may replace "phases" (tests)."""
    src = load_source(snap["source"], snap.get("sha256"))
    meta = src[2]
    window = snap.get("window", WINDOW_DEFAULT)
    margin = float(snap.get("margin", MARGIN_DEFAULT))
    N_t = coords.size
    dx_t = L / N_t
    dx_s, N_s, L_s = meta["dx"], meta["params"]["N"], meta["params"]["L"]
    n_grid = N_s
    if abs(dx_t - dx_s) > 1e-9 * dx_s:
        if not snap.get("resample"):
            raise ValueError("target dx %.6g differs from the source dx %.6g" % (dx_t, dx_s))
        n_grid = int(round(L_s / dx_t))
        if abs(L_s / n_grid - dx_t) > 1e-9 * dx_t or n_grid % 2:
            raise ValueError("resample needs L_source / dx_target to be an even integer")
    if window is not None and window_key(window) not in meta["crop"]:
        raise ValueError("crop loss was not measured for window %s (measured: %s)" % (window, list(meta["crop"])))
    centers = np.asarray(snap["centers"], float)
    if snap.get("taus") is not None:      # explicit source times after the reference (tests)
        phases = [("tau", float(t)) for t in snap["taus"]]
        taus = {k: k[1] for k in phases}
    else:
        phases = [("theta", float(t)) for t in snap["phases"]]
        taus = {k: phase_taus(meta, [k[1]])[0] for k in phases}
    if len(phases) != len(centers):
        raise ValueError("one phase per lump")
    if window is None:
        M = n_grid // 2
    else:
        M = int(np.ceil((window[1] + margin) / dx_t))
        if M > n_grid // 2:
            raise ValueError("source box too small for the window")
    if 2 * M > N_t:
        raise ValueError("target box smaller than the snapshot cube")
    r_cube = _radial_grid(2 * M, dx_t, M)
    w = None if window is None else window_profile(r_cube, *window)
    cubes, recs = {}, {}

    def consume(th, phi, pi, rec):
        if n_grid != N_s:
            phi, pi = resample(phi, n_grid), resample(pi, n_grid)
        c = n_grid // 2
        sl = slice(c - M, c + M)
        if w is None:
            cubes[th] = (phi[sl, sl, sl].copy(), pi[sl, sl, sl].copy())
        else:
            cubes[th] = (phi[sl, sl, sl] * w, pi[sl, sl, sl] * w)
        recs[th] = rec

    phase_states(src, taus, consume, threads)
    phi_t = np.zeros((N_t,) * 3)
    pi_t = np.zeros((N_t,) * 3)
    lumps, ring, nyq = [], 0.0, 0.0
    for ctr, th in zip(centers, phases):
        f = (ctr + L / 2) / dx_t
        ic = np.round(f).astype(int)
        delta = (f - ic) * dx_t
        cp, cv = cubes[th]
        if np.any(delta != 0.0):
            cp, n1 = spectral_shift(cp, delta, dx_t)
            cv, n2 = spectral_shift(cv, delta, dx_t)
            nyq = max(nyq, n1, n2)
            if window is not None:
                xs = dx_t * np.arange(-M, M)
                rr = np.sqrt((xs[:, None, None] - delta[0]) ** 2 + (xs[None, :, None] - delta[1]) ** 2
                             + (xs[None, None, :] - delta[2]) ** 2)
                far = rr > window[1] + dx_t
                if far.any():
                    ring = max(ring, float(np.max(np.abs(cp[far]))) / meta["Phi0"])
        idx = [(ic[a] + np.arange(-M, M)) % N_t for a in range(3)]
        sub = np.ix_(*idx)
        phi_t[sub] += cp
        pi_t[sub] += cv
        lumps.append({"center": ctr.tolist(), th[0]: th[1], "grid_index": ic.tolist(),
                      "subgrid_shift": delta.tolist(), **recs[th]})
    info = {"source": snap["source"], "sha256": meta["sha256"], "omega_source": meta["omega"],
            "t0_source": meta["t0"], "t_ref_source": meta["t_ref"], "window": window, "margin": margin,
            "cube_points": 2 * M, "resampled_to": n_grid if n_grid != N_s else None,
            "crop": meta["crop"][window_key(window)] if window is not None else None,
            "omega_drift": meta["omega_drift"], "shift_ringing_max_over_Phi0": ring,
            "nyquist_fraction_dropped_max": nyq, "lumps": lumps}
    return phi_t, pi_t, info


def check(snap, L, absorber, sponge_tol, tol=GUARD_TOL):
    """Guards for a snapshot init entry (no fields built); raises ValueError, else returns numbers."""
    meta = load_meta(snap["source"])
    if snap.get("sha256") and snap["sha256"] != meta["sha256"]:
        raise ValueError("snapshot sha256 in the config differs from the source")
    window = snap.get("window", WINDOW_DEFAULT)
    cr = None
    if window is not None:
        if window_key(window) not in meta["crop"]:
            raise ValueError("crop loss was not measured for window %s" % (window,))
        cr = meta["crop"][window_key(window)]
    out = {"source": snap["source"], "sha256": meta["sha256"], "omega_drift": meta["omega_drift"],
           "window": window, "crop_loss_E_max": cr and cr["crop_loss_E_max"],
           "crop_loss_N": cr and cr["crop_loss_N"], "lumps": []}
    if meta["omega_drift"] > tol["omega_drift"]:
        raise ValueError("source not settled: omega drift %.3g" % meta["omega_drift"])
    if cr is not None:
        loss = max(cr["crop_loss_E_max"], abs(cr["crop_loss_N"]))
        out["crop_loss_over_target"] = loss / tol["crop_loss"]
        if loss > 3 * tol["crop_loss"]:   # CLAUDE.md: within 3x the target is a pass with a note
            raise ValueError("crop loss too large: E %.3g N %.3g" % (cr["crop_loss_E_max"], cr["crop_loss_N"]))
    spl = radial_spline(meta["profile"]["r"], meta["profile"]["Phi"])
    rmax = meta["profile"]["r"][-1]
    rr = np.linspace(0.0, rmax, 3401)
    env = np.abs(spl(rr)) / meta["Phi0"]
    if window is not None:
        env = env * window_profile(rr, *window)
    for ctr in snap["centers"]:
        c = np.asarray(ctr, float)
        r_face = L / 2 - float(np.max(np.abs(c)))
        rec = {"center": c.tolist(), "r_face": r_face,
               "max_tail_beyond_r_face_over_Phi0": float(np.max(env[rr >= r_face])) if r_face < rmax else 0.0}
        if rec["max_tail_beyond_r_face_over_Phi0"] > TAIL_TOL:      # the radial_checks face guard
            raise ValueError("lump tail not negligible at a box face: %s" % rec)
        if absorber:
            r_in = (absorber.get("r_out") or L / 2) - absorber["width"]
            dist = r_in - float(np.linalg.norm(c))
            if dist <= 0:
                raise ValueError("lump centre inside the sponge: %s" % rec)
            rec.update(r_in=r_in, dist_to_sponge=dist,
                       Phi_at_sponge_over_Phi0=(float(abs(spl(dist))) if dist < rmax else 0.0) / meta["Phi0"])
            if rec["Phi_at_sponge_over_Phi0"] > sponge_tol:
                raise ValueError("lump tail not negligible at the sponge inner radius: %s" % rec)
        out["lumps"].append(rec)
    return out
