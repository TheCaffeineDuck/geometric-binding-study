"""WP-L (Session 3), L2: 3D initial data from a settled-lump snapshot (snapshot.py source): a harmonic
decomposition over one period, a Lorentz boost, a quadrupolar stretch, a smooth random multiplier, and
two-lump (collision) data. Init entry "wpl" (initial.build_initial), guards in lumps.check_lumps.

Harmonics. The source state is sampled at M equally spaced phases theta_k = 2 pi k / M over one period
(snapshot.phase_states, exact landing by a partial RK4 step), each multiplied by the window w(r)
(snapshot.window_profile) and cropped to the snapshot cube (half-size Mc = ceil((r_out + margin)/dx)).
a_n = (1/M) sum_k phi_k exp(-i n theta_k) and b_n likewise for phi_t, n = 0 .. M/2. Then
    phi(theta) = sum_n c_n Re(a_n exp(i n theta)),  c_0 = c_{M/2} = 1, c_n = 2 otherwise,
which reproduces the samples exactly and, between them, is exact for harmonics below M/2. The source
oscillates as Phi cos(omega t + theta) (snapshot convention), so a copy at phase theta is phi(theta).

Boost along x with speed v (gamma = 1/sqrt(1 - v^2)). Rest frame (x, t), lab (x', t'), x = gamma(x' - v t'),
t = gamma(t' - v x'). On the lab slice t' = 0, at lab offset xi from the lump centre:
    phi'(xi, y, z)   = phi(gamma xi, y, z; theta_c - omega gamma v xi)
    phi'_t'(xi, y, z) = gamma [phi_t - v phi_x](gamma xi, y, z; theta_c - omega gamma v xi)
with theta_c the lump's phase at its centre. The rest-frame values at x = gamma xi (off the grid) come
from the periodic trigonometric interpolant of each a_n, b_n along x on the cube (a dense n x n matrix;
Nyquist term dropped), phi_x from its derivative. Points with |gamma xi| beyond the cube are zero (the
windowed field is zero there). For an isolated time-periodic solution the lab energy is gamma E0 and the
momentum gamma v E0 exactly (the energy-momentum of a conserved T^{mu nu} does not depend on the slice);
the lab frequency at the moving centre is omega / gamma.

Quadrupolar stretch (volume preserving, the convention of initial.py): phi'(x) = phi(S x),
S = diag(sqrt(1 + eps), sqrt(1 + eps), 1/(1 + eps)), applied to phi and phi_t at the lump's phase, by the
same separable interpolation. det S = 1, so int phi^2, int phi_t^2 and int V(phi) are unchanged and the
gradient energy changes by sum_a (S_a^2 - 1) G_a, G_a = (1/2) int (d_a phi)^2 (stretch_expected).

Random multiplier (no symmetry): phi -> phi (1 + a xi), phi_t -> phi_t (1 + a xi), xi a smooth Gaussian
random field on the cube (seeded white noise filtered by exp(-k^2 ell^2 / 2)), shifted and scaled to zero
mean and unit rms with the weight rho = phi_t^2 + omega^2 phi^2 (the N_inst density at the lump's phase).
Then Delta N_inst / N_inst = a^2 exactly: the first-order (uniform) part, which is an amplitude kick, is
removed, and a is the rms relative modulation over the lump.

Placement as snapshot.place: integer grid shift plus an exact spectral sub-grid shift of the cube.
"""

import numpy as np
import scipy.fft as sfft

from swirl.engine.evolver import Evolver
from swirl.engine.snapshot import (MARGIN_DEFAULT, _radial_grid, load_source, phase_states, phase_taus,
                                   resample, spectral_shift, window_profile)

WINDOW = [26.0, 30.0]
M_HARM = 32
_CACHE = {}


# ---------------------------------------------------------------------------
# Harmonics
# ---------------------------------------------------------------------------

def harmonics(source, window=WINDOW, margin=MARGIN_DEFAULT, M=M_HARM, sha256=None, threads=2, dx_target=None):
    """Harmonic decomposition of the windowed source cube over one period. Cached per arguments.
    dx_target: resample each source state spectrally to this spacing first (resolution checks);
    L_source / dx_target must be an even integer."""
    key = (source, tuple(window), float(margin), int(M), dx_target)
    if key in _CACHE:
        return _CACHE[key]
    src = load_source(source, sha256)
    meta = src[2]
    dx, Ns = meta["dx"], meta["params"]["N"]
    n_res = None
    if dx_target is not None and abs(dx_target - dx) > 1e-9 * dx:
        n_res = int(round(meta["params"]["L"] / dx_target))
        if n_res % 2 or abs(meta["params"]["L"] / n_res - dx_target) > 1e-9 * dx_target:
            raise ValueError("L_source / dx_target must be an even integer")
        dx, Ns = meta["params"]["L"] / n_res, n_res
    Mc = int(np.ceil((window[1] + margin) / dx))
    if Mc > Ns // 2:
        raise ValueError("source box too small for the window")
    w = window_profile(_radial_grid(2 * Mc, dx, Mc), *window)
    thetas = 2 * np.pi * np.arange(M) / M
    taus = {k: t for k, t in enumerate(phase_taus(meta, thetas))}
    nh = M // 2 + 1
    a = np.zeros((nh,) + (2 * Mc,) * 3, complex)
    b = np.zeros_like(a)
    c = Ns // 2
    sl = slice(c - Mc, c + Mc)
    first = {}

    def consume(k, phi, pi, rec):
        if n_res is not None:
            phi, pi = resample(phi, n_res), resample(pi, n_res)
        p, q = phi[sl, sl, sl] * w, pi[sl, sl, sl] * w
        if k == 0:
            first["phi"], first["pi"] = p.copy(), q.copy()
        for j in range(nh):
            e = np.exp(-1j * j * thetas[k]) / M
            a[j] += e * p
            b[j] += e * q

    phase_states(src, taus, consume, threads)
    H = dict(a=a, b=b, M=M, omega=meta["omega"], dx=dx, Mc=Mc, window=list(window), margin=margin,
             source=source, sha256=meta["sha256"], Phi0=meta["Phi0"], params=meta["params"],
             coeff=np.array([1.0] + [2.0] * (nh - 2) + [1.0]), sample0=(first["phi"], first["pi"]))
    _CACHE.clear()
    _CACHE[key] = H
    return H


def reconstruct(H, theta):
    """phi, phi_t of the windowed source cube at phase theta."""
    e = H["coeff"] * np.exp(1j * np.arange(H["a"].shape[0]) * theta)
    phi = np.tensordot(e, H["a"], axes=(0, 0)).real
    pt = np.tensordot(e, H["b"], axes=(0, 0)).real
    return phi, pt


# ---------------------------------------------------------------------------
# Interpolation along one axis of the cube
# ---------------------------------------------------------------------------

def interp_matrices(n, dx, X):
    """Periodic trigonometric interpolation from the cube nodes x_i = (i - n/2) dx to the points X:
    B (len(X) x n) and its x-derivative D. Nyquist term dropped. Rows with |X| beyond the cube are zero."""
    xs = dx * (np.arange(n) - n // 2)
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    if n % 2 == 0:
        k = k[np.abs(np.abs(k) - np.pi / dx) > 1e-9 * np.pi / dx]
    E = np.exp(1j * np.outer(X, k))
    F = np.exp(-1j * np.outer(k, xs))
    B = (E @ F).real / n
    D = ((E * (1j * k)[None, :]) @ F).real / n
    out = np.abs(X) > xs[-1] + 1e-9
    B[out] = 0.0
    D[out] = 0.0
    return B, D


def _apply(Bs, f):
    """f'(i, j, k) = sum B0[i, a] B1[j, b] B2[k, c] f(a, b, c) (Bs[i] None = identity)."""
    for ax, B in enumerate(Bs):
        if B is not None:
            f = np.moveaxis(np.tensordot(B, f, axes=(1, ax)), 0, ax)
    return f


# ---------------------------------------------------------------------------
# One lump on its cube
# ---------------------------------------------------------------------------

def smooth_random(n, dx, ell, seed):
    rng = np.random.default_rng(seed)
    wn = rng.standard_normal((n, n, n))
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    K2 = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
    xi = np.real(np.fft.ifftn(np.fft.fftn(wn) * np.exp(-0.5 * K2 * ell * ell)))
    return xi / np.sqrt(np.mean(xi * xi))


def apply_kick(phi, pt, dx, kick):
    """The L1 radial disturbances on a cube centred at index n/2 (spot checks of radial points):
    kick = {kind: field | velocity | removal, eps or s, hw}; tapers as swirl.family.wpl."""
    from swirl.family.wpl import CORE, KICK_BALL, taper
    n = phi.shape[0]
    r = _radial_grid(n, dx, n // 2)
    hw = kick["hw"]
    if kick["kind"] in ("field", "velocity"):
        w = 1.0 + kick["eps"] * taper(r, KICK_BALL[0] * hw, KICK_BALL[1] * hw)
        return (phi * w, pt) if kick["kind"] == "field" else (phi, pt * w)
    if kick["kind"] == "removal":
        f = 1.0 - kick["s"] * taper(r, CORE[0] * hw, CORE[1] * hw)
        return phi * f, pt * f
    raise ValueError("unknown kick %s" % kick["kind"])


def lump_cube(H, theta=0.0, v=0.0, quad_eps=0.0, random=None, kick=None):
    """(phi, phi_t) of one lump on the cube (centre at index Mc), lab frame. theta: phase (rad) at the
    centre at t' = 0; v: boost speed along x; quad_eps: stretch along z; random: {amp, ell, seed};
    kick: an L1 radial disturbance (apply_kick)."""
    n = 2 * H["Mc"]
    dx = H["dx"]
    if v != 0.0 and (quad_eps or random or kick):
        raise ValueError("boost and distortion are not combined")
    if v == 0.0:
        phi, pt = reconstruct(H, theta)
        if kick:
            phi, pt = apply_kick(phi, pt, dx, kick)
        if quad_eps:
            xs = dx * (np.arange(n) - n // 2)
            s1, s3 = np.sqrt(1.0 + quad_eps), 1.0 / (1.0 + quad_eps)
            B1, _ = interp_matrices(n, dx, s1 * xs)
            B3, _ = interp_matrices(n, dx, s3 * xs)
            phi, pt = _apply([B1, B1, B3], phi), _apply([B1, B1, B3], pt)
        if random:
            xi = smooth_random(n, dx, random["ell"], random["seed"])
            rho = pt * pt + H["omega"] ** 2 * phi * phi
            xi = xi - np.sum(xi * rho) / np.sum(rho)
            xi = xi / np.sqrt(np.sum(xi * xi * rho) / np.sum(rho))
            f = 1.0 + random["amp"] * xi
            phi, pt = phi * f, pt * f
        return phi, pt
    g = 1.0 / np.sqrt(1.0 - v * v)
    xs = dx * (np.arange(n) - n // 2)
    B, D = interp_matrices(n, dx, g * xs)
    th = theta - H["omega"] * g * v * xs
    phi = np.zeros((n, n, n))
    pt = np.zeros((n, n, n))
    px = np.zeros((n, n, n))
    for j in range(H["a"].shape[0]):
        ph = (H["coeff"][j] * np.exp(1j * j * th))[:, None, None]
        A = np.tensordot(B, H["a"][j], axes=(1, 0))
        Ax = np.tensordot(D, H["a"][j], axes=(1, 0))
        Bt = np.tensordot(B, H["b"][j], axes=(1, 0))
        phi += (A * ph).real
        px += (Ax * ph).real
        pt += (Bt * ph).real
    return phi, g * (pt - v * px)


# ---------------------------------------------------------------------------
# Quadrature on a cube (periodic spectral, the evolver's energy)
# ---------------------------------------------------------------------------

def cube_values(phi, pt, dx, m, g4, g6, omega, threads=2):
    """E (and parts), momentum P = -int phi_t grad phi, N_inst = int (phi_t^2 + omega^2 phi^2)/(2 omega),
    per-axis gradient energies G_a, on the periodic cube."""
    n = phi.shape[0]
    ev = Evolver(n, n * dx, m, g4, g6, threads=threads)
    parts = ev.energy_parts(phi, pt)
    fh = sfft.rfftn(phi, workers=threads)
    G, P = [], []
    for kk in (ev.kx, ev.ky, ev.kz):
        kk = kk.copy()
        if n % 2 == 0:
            kk[np.isclose(np.abs(kk), np.pi / dx)] = 0.0
        d = sfft.irfftn(1j * kk * fh, s=phi.shape, workers=threads)
        G.append(0.5 * float(np.sum(d * d)) * dx ** 3)
        P.append(-float(np.sum(pt * d)) * dx ** 3)
    N = float(np.sum(pt * pt + omega * omega * phi * phi)) * dx ** 3 / (2 * omega)
    return dict(parts, G=G, P=P, N_inst=N)


def rest_reference(H, m, g4, g6, threads=2):
    """E and N_inst of the rest lump at the M sample phases: mean (the reference for gamma E0), min, max."""
    E, N = [], []
    for k in range(H["M"]):
        v = cube_values(*reconstruct(H, 2 * np.pi * k / H["M"]), H["dx"], m, g4, g6, H["omega"], threads)
        E.append(v["E_total"])
        N.append(v["N_inst"])
    return dict(E0=float(np.mean(E)), E_min=float(np.min(E)), E_max=float(np.max(E)), N0=float(np.mean(N)),
                N_min=float(np.min(N)), N_max=float(np.max(N)))


def stretch_expected(base, quad_eps):
    """Closed-form t = 0 change of the stretch: Delta E = sum_a (S_a^2 - 1) G_a, Delta N_inst = 0."""
    S2 = [1.0 + quad_eps, 1.0 + quad_eps, 1.0 / (1.0 + quad_eps) ** 2]
    return dict(dE=float(sum((s - 1.0) * g for s, g in zip(S2, base["G"]))), dN=0.0, dE_kin=0.0,
                dE_pot=0.0)


# ---------------------------------------------------------------------------
# Placement and the init entry
# ---------------------------------------------------------------------------

def place_cubes(cubes, coords, L):
    """Add cubes [(phi, phi_t, centre)] onto the target grid (same dx): integer shift plus a spectral
    sub-grid shift. Returns phi, pi, per-cube records."""
    N_t = coords.size
    dx = L / N_t
    phi_t = np.zeros((N_t,) * 3)
    pi_t = np.zeros((N_t,) * 3)
    recs = []
    for cp, cv, ctr in cubes:
        n = cp.shape[0]
        Mc = n // 2
        f = (np.asarray(ctr, float) + L / 2) / dx
        ic = np.round(f).astype(int)
        delta = (f - ic) * dx
        nyq = 0.0
        if np.any(delta != 0.0):
            cp, n1 = spectral_shift(cp, delta, dx)
            cv, n2 = spectral_shift(cv, delta, dx)
            nyq = max(n1, n2)
        idx = [(ic[a] + np.arange(-Mc, Mc)) % N_t for a in range(3)]
        sub = np.ix_(*idx)
        phi_t[sub] += cp
        pi_t[sub] += cv
        recs.append(dict(center=list(map(float, ctr)), grid_index=ic.tolist(), subgrid_shift=delta.tolist(),
                         nyquist_fraction_dropped=nyq))
    return phi_t, pi_t, recs


def place_wpl(entry, coords, L, threads=2):
    """init["wpl"] = {source, sha256, window, margin, M, lumps: [{center, phase (rad), v, quad_eps, random}]}"""
    H = harmonics(entry["source"], entry.get("window", WINDOW), entry.get("margin", MARGIN_DEFAULT),
                  entry.get("M", M_HARM), entry.get("sha256"), threads, entry.get("dx_target"))
    dx = L / coords.size
    if abs(dx - H["dx"]) > 1e-9 * dx:
        raise ValueError("target dx %.6g differs from the source dx %.6g" % (dx, H["dx"]))
    cubes = []
    for lp in entry["lumps"]:
        cp, cv = lump_cube(H, lp.get("phase", 0.0), lp.get("v", 0.0), lp.get("quad_eps", 0.0), lp.get("random"),
                           lp.get("kick"))
        cubes.append((cp, cv, lp["center"]))
    phi, pi, recs = place_cubes(cubes, coords, L)
    info = dict(source=H["source"], sha256=H["sha256"], omega_source=H["omega"], window=H["window"],
                margin=H["margin"], M=H["M"], cube_points=2 * H["Mc"], dx=H["dx"],
                lumps=[{**lp, **r} for lp, r in zip(entry["lumps"], recs)])
    return phi, pi, info


def check_wpl(entry, L, absorber, sponge_tol):
    """Guards: those of a snapshot entry with the same source, window and centres (snapshot.check)."""
    from swirl.engine.snapshot import check
    snap = dict(source=entry["source"], sha256=entry.get("sha256"), window=entry.get("window", WINDOW),
                centers=[lp["center"] for lp in entry["lumps"]])
    return check(snap, L, absorber, sponge_tol)


def two_lump_entry(source, d, phase_deg, v, window=WINDOW, axis_offset=(0.0, 0.0), sha256=None):
    """Collision data: lump A at x = -d/2 moving +v, lump B at x = +d/2 moving -v, phases 0 and phase_deg."""
    y, z = axis_offset
    return dict(source=source, sha256=sha256, window=list(window),
                lumps=[dict(center=[-0.5 * d, y, z], phase=0.0, v=float(v)),
                       dict(center=[0.5 * d, y, z], phase=float(np.radians(phase_deg)), v=-float(v))])


# ---------------------------------------------------------------------------
# Shape of a lump (Session 1 shape.py definitions)
# ---------------------------------------------------------------------------

def shape(ev, center, R):
    """Energy-density shape in a sphere of radius R: centroid c (two passes from `center`),
    M = int (x - c)(x - c) e, a2 = (lambda_max - lambda_min) / tr M, T = int (x - c)^3 e,
    a3 = ||T||_F / (E_s (tr M / E_s)^{3/2}). A single snapshot (it oscillates at 2 omega)."""
    from swirl.engine.lumps import _wrap
    e = ev.energy_density()
    L, dV = ev.L, ev.dV
    X = [_wrap(ev.coords - center[a], L).reshape([-1 if b == a else 1 for b in range(3)]) for a in range(3)]
    c = np.zeros(3)
    for _ in range(2):
        r2 = sum((X[a] - c[a]) ** 2 for a in range(3))
        w = np.where(r2 < R * R, e, 0.0)
        S = w.sum()
        c = np.array([(w * X[a]).sum() / S for a in range(3)])
    r2 = sum((X[a] - c[a]) ** 2 for a in range(3))
    w = np.where(r2 < R * R, e, 0.0)
    Es = w.sum() * dV
    Y = [np.broadcast_to(X[a] - c[a], e.shape) for a in range(3)]
    Mm = np.array([[(w * Y[i] * Y[j]).sum() for j in range(3)] for i in range(3)]) * dV
    T = np.array([[[(w * Y[i] * Y[j] * Y[k]).sum() for k in range(3)] for j in range(3)] for i in range(3)]) * dV
    lam = np.linalg.eigvalsh(Mm)
    trM = np.trace(Mm)
    return dict(centroid=(np.asarray(center, float) + c).tolist(), a2=float((lam[-1] - lam[0]) / trM),
                a3=float(np.sqrt((T ** 2).sum()) / (Es * (trM / Es) ** 1.5)), E_sphere=float(Es),
                eig_M=lam.tolist(), M=Mm.tolist())


# ---------------------------------------------------------------------------
# Tracking run for the boosted-lump gate
# ---------------------------------------------------------------------------

def track_run(grid, entry, T, every=5, R_ball=10.0, taper=2.0, threads=2, verbose=False):
    """Evolve one lump from init {"wpl": entry} on grid {N, L, m, g4, g6, dt, absorber} and track it:
    every `every` steps the energy-weighted centroid X in a ball (cos^2 taper) about the previous X, the ball
    energy, momentum and second moments of the energy density about X, E_total and absorbed; every step
    phi at X (trilinear). Returns the series."""
    import time as _time
    from swirl.engine.initial import build_initial
    from swirl.engine.lumps import _wrap, ball_weight, densities, trilinear
    ev = Evolver(grid["N"], grid["L"], grid["m"], grid["g4"], grid["g6"], absorber=grid["absorber"],
                 threads=threads)
    info = {}
    phi, pi = build_initial(ev.coords, ev.L, {"wpl": entry}, info=info, threads=threads)
    ev.set_state(phi, pi)
    X = np.array(entry["lumps"][0]["center"], float)
    dt = grid["dt"]
    nT = int(round(T / dt))
    S = {k: [] for k in ("t", "E_total", "absorbed", "E_ball", "X", "P_ball", "I_diag")}
    pt_t, pt_phi = [], []
    c = ev.coords
    t0 = _time.perf_counter()
    for n in range(nT + 1):
        if n % every == 0:
            e, p = densities(ev)
            rel = [_wrap(c - X[a], ev.L).reshape([-1 if b == a else 1 for b in range(3)]) for a in range(3)]
            W = ball_weight(np.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2), R_ball, taper)
            ew = e * W
            Eb = float(ew.sum()) * ev.dV
            d = [float((ew * rel[a]).sum()) * ev.dV / Eb for a in range(3)]
            I = [float((ew * (rel[a] - d[a]) ** 2).sum()) * ev.dV / Eb for a in range(3)]
            S["t"].append(n * dt)
            S["E_total"].append(ev.energy())
            S["absorbed"].append(ev.absorbed)
            S["E_ball"].append(Eb)
            S["X"].append((X + np.array(d)).tolist())
            S["P_ball"].append([float((p[a] * W).sum()) * ev.dV for a in range(3)])
            S["I_diag"].append(I)
            X = X + np.array(d)
        idx, wts = trilinear(c, ev.L, ev.dx, X)
        pt_t.append(n * dt)
        pt_phi.append(float(sum(w * ev.phi[i] for i, w in zip(idx, wts))))
        if n < nT:
            ev.step(dt)
        if verbose and n % 500 == 0:
            print("step %d/%d E %.8g" % (n, nT, S["E_total"][-1]), flush=True)
    return dict(series=S, point=dict(t=pt_t, phi=pt_phi), init_info=info, wall_s=_time.perf_counter() - t0,
                grid=grid, entry=entry, T=T, every=every, R_ball=R_ball)


def gate_analysis(run, run0, v, E0_ref, t_from=20.0):
    """Speed, energy ratio, frequency ratio and shape of a boosted run against the v = 0 run.
    Speed: least-squares slope of the tracked centroid x(t), t >= t_from. Frequency: pi / (slope of the zero
    crossing times of phi at the tracked centre against their index), t >= t_from. Energy: E_total at t = 0
    against gamma E0_ref (E0_ref: rest_reference mean over the M phases) and against gamma E_total(0) of the
    v = 0 run. Shape: the ball energy and the moment ratio I_xx / I_yy, both smoothed over three periods,
    from first to last window. Both runs are read over the same span [t_from, min(T, T0)]."""
    from swirl.engine.run import zero_crossings
    g = 1.0 / np.sqrt(1.0 - v * v)

    t_to = min(run["T"], run0["T"])

    def track(r):
        s = r["series"]
        t = np.asarray(s["t"])
        X = np.asarray(s["X"])
        sel = (t >= t_from) & (t <= t_to + 1e-9)
        tz = zero_crossings(r["point"]["t"], r["point"]["phi"])
        tz = tz[(tz >= t_from) & (tz <= t_to)]
        half = np.polyfit(np.arange(tz.size), tz, 1)[0]
        Eb = np.asarray(s["E_ball"])
        I = np.asarray(s["I_diag"])
        Epa = (np.asarray(s["E_total"]) + np.asarray(s["absorbed"]))[t <= t_to + 1e-9]
        nper = max(1, int(round(3 * 2 * half / (t[1] - t[0]))))
        sm = lambda y: np.convolve(y, np.ones(nper) / nper, mode="valid")
        Ebs, ratio = sm(Eb[sel]), sm(I[sel, 0]) / sm(I[sel, 1])
        return dict(vx=float(np.polyfit(t[sel], X[sel, 0], 1)[0]), vy=float(np.polyfit(t[sel], X[sel, 1], 1)[0]),
                    vz=float(np.polyfit(t[sel], X[sel, 2], 1)[0]), omega=float(np.pi / half), n_crossings=int(tz.size),
                    E_total0=float(s["E_total"][0]), E_ball_first=float(Ebs[0]), E_ball_last=float(Ebs[-1]),
                    E_ball_drift=float(Ebs[-1] / Ebs[0] - 1), Ixx_over_Iyy_first=float(ratio[0]),
                    Ixx_over_Iyy_last=float(ratio[-1]), Iyy_over_Izz_mean=float(np.mean(sm(I[sel, 1]) / sm(I[sel, 2]))),
                    E_plus_absorbed_rel_dev=float(np.max(np.abs(Epa - Epa[0])) / Epa[0]),
                    x_start=float(X[0, 0]), x_end=float(X[-1, 0]), wall_s=r["wall_s"])

    a, b = track(run), track(run0)
    out = dict(v=v, gamma=g, run=a, rest=b)
    out["speed_measured"] = a["vx"]
    out["speed_rel_err"] = a["vx"] / v - 1 if v else None
    out["E_ratio_vs_E0ref"] = a["E_total0"] / E0_ref
    out["E_ratio_vs_E0ref_err"] = a["E_total0"] / (g * E0_ref) - 1
    out["E_ratio_vs_rest_run"] = a["E_total0"] / b["E_total0"]
    out["E_ratio_vs_rest_run_err"] = a["E_total0"] / (g * b["E_total0"]) - 1
    out["omega_ratio"] = a["omega"] / b["omega"]
    out["omega_ratio_err"] = a["omega"] / b["omega"] * g - 1
    out["E_ball_drift_minus_rest"] = a["E_ball_drift"] - b["E_ball_drift"]
    out["contraction_Ixx_over_Iyy_first"] = a["Ixx_over_Iyy_first"] / b["Ixx_over_Iyy_first"]
    out["contraction_drift"] = a["Ixx_over_Iyy_last"] / a["Ixx_over_Iyy_first"] - 1
    out["contraction_drift_rest"] = b["Ixx_over_Iyy_last"] / b["Ixx_over_Iyy_first"] - 1
    out["one_over_gamma_sq"] = 1.0 / g ** 2
    return out
