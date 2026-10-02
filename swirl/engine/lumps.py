"""Multi-lump initial data and per-lump / field-momentum diagnostics (WP-C).

Initial data
------------
lumps_init(profile, centers, phases, omega) builds the `init` dict for K copies of
a radial profile:  phi = sum_i Phi(|x - x_i|) cos(theta_i),
phi_t = -omega sum_i Phi(|x - x_i|) sin(theta_i).  check_lumps() adds the guards:
the profile is negligible at the box face (initial.radial_checks, per lump) and
each lump sits in the sponge-free interior, Phi(r_in - |x_i|) <= sponge_tol Phi(0).

Diagnostics (sampled every `every` steps; all densities include every term of the
equation of motion, standard 3)
------------------------------------------------------------------------------
    e   = phi_t^2/2 + |grad phi|^2/2 + m^2 phi^2/2 - g4 phi^4/24 + g6 phi^6/720
    p_i = -phi_t d_i phi        (momentum density; also the energy flux)
Gradients are spectral with the Nyquist wavenumber zeroed (as Evolver.energy_density).

Per lump i, with weight W_i(x) = b(|x - c_i|) / sum_j exp(2 d_ij s_ij / (ell d_nn)):
    b: 1 inside R_ball - taper, cos^2 taper to 0 at R_ball (smooth, so no staircase);
    s_ij = signed distance past the bisector plane of c_i, c_j; j runs over i (term 1) and the
       lumps whose balls can overlap ball i; d_nn = smallest centre distance (floored at ell).
       Overlapping balls are split by the nearest tracked centre, smoothly over a width ell.
       For K = 2 this is b * 0.5 (1 - tanh(s/ell)) (the WP-C split); for any K the weights
       of the lumps covering a point sum to 1.
    E_i = int W e,  X_i = c_i + int W e (x - c_i) / E_i (minimum image),
    It_i = int W phi_t^2,  Ip_i = int W phi^2,  P_i = int W p.
The balls follow the lumps: c_i <- X_i after each sample.
phi(c_i) is read every step (trilinear) for the frequency, the whole-period windows
and the relative phase.

Field quantities: total momentum int p; angular momentum n . int (x - o) x p about
each axis (o, n) (minimum image about o; the grid plane at exactly L/2 from o has lever 0,
see lever_arms); half-space momentum int_{x_a < s} p_a for each plane (axis a, position s),
whose time derivative is the force on that half.

K lumps (WP-E): balls that overlap are split by the nearest tracked centre with a smooth
partition of unity (sample(); identical to the WP-C bisector split for K = 2). Optional
"group": {"R", "taper", "axis"}: energy inside a sphere about the energy-weighted centroid
of the lumps, and the angular momentum about that centroid (series G, E_group, L_group_box,
L_group_sphere, L_axis_box, L_axis_sphere). Absent = off (WP-C configs unchanged).

Lump count (standard 8, from coordinates): every `count_every` time units the
energy density is averaged over `count_periods` nominal periods with Hann weights
(suppresses the 2 omega oscillation); maxima of that average above
max(count_threshold * max, floor), at least count_min_sep apart, are the lumps (the floor is
absolute, see "Absolute floor" below). For each pair
of maxima the minimum of the averaged density along the segment between them,
over the smaller peak, is the saddle ratio; count_distinct joins maxima whose saddle
ratio is >= saddle_merge (default 0.9).

Absolute floor (Session 3, WP0; F-018): the relative threshold alone counts debris once the field has
emptied, because "1% of the current maximum" is a tiny number when the maximum is a few per cent of a lump
(Session 2: E3, E7 and the E3 checks were counted as "flew apart" or as 6 distinct debris maxima after two
lumps had left the box). The floor is count_floor_frac (default 0.05) times a reference density: cfg
"count_floor_ref" if given, else the maximum of the first count window (the initial lump scale; for
well-separated lumps it is the single-lump peak of the period-averaged density). It is not applied to the
first window itself (no reference yet) and the reference is read back from counts[0]["max"], so it needs
no extra checkpoint state. Why 0.05: in the 20 stored Session 2 arrangement runs the largest debris maximum
on an emptied field is 0.0073 of the first-window maximum (twin of E1) and the smallest maximum of anything
that is plainly a lump is 0.29 (a remnant leaving through the sponge); 0.05 sits 7 times above the first and
6 times below the second. count_floor_frac = 0 switches the floor off (the Session 2 behaviour). Each stored
count entry records "abs_floor". refloor() applies a floor to a stored entry, to recount saved results.

Period averages (post-processing, `summarize`): every per-lump series is averaged
over whole periods of that lump's phi(c_i, t), between zero crossings, by exact
integration of a cubic spline through the samples (Session 1: a fixed-span
average of N ripples at 1.2e-2). N_i = <It_i> / omega_i. The relative phase of
lumps j and 0 is arg(c_j conj(c_0)), c_k = int phi(c_k, t) exp(-i omega t) h(t) dt over
the window (h = Hann on the window, which removes the counter-rotating term even
when the amplitude drifts), so phi_k ~ cos(omega t + theta_k) gives theta_j - theta_0.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import maximum_filter

from swirl.engine.initial import radial_checks, radial_spline

SPONGE_TOL = 1e-3   # Phi at the sponge inner radius, relative to Phi(0)

LUMP_DEFAULTS = {
    "centers": None, "R_ball": 10.0, "taper": 2.0, "split_width": 1.0, "every": 5,
    "count_every": 25.0, "count_periods": 3, "omega_nominal": 0.909,
    "count_threshold": 0.01, "count_min_sep": 1.0, "saddle_merge": 0.9, "avg_window": 20.0,
    "count_floor_frac": 0.05, "count_floor_ref": None,
    "sponge_tol": SPONGE_TOL, "planes": [], "axes": [],
}


# ---------------------------------------------------------------------------
# Initial data
# ---------------------------------------------------------------------------

def lumps_init(profile, centers, phases, omega):
    """init dict with one radial entry per lump (profile: dict with r, Phi)."""
    if len(centers) != len(phases):
        raise ValueError("one phase per lump")
    r = [float(x) for x in profile["r"]]
    P = [float(x) for x in profile["Phi"]]
    return {"radial": [{"r": r, "Phi": P, "center": [float(c) for c in ctr], "phase": float(th),
                        "omega": float(omega)} for ctr, th in zip(centers, phases)],
            "gaussians": []}


def check_lumps(init, L, absorber, sponge_tol=SPONGE_TOL):
    """Guards for multi-lump data; raises ValueError, else returns the numbers.
    A snapshot entry is checked by snapshot.check (source settled, crop loss, faces, sponge)."""
    out = []
    if init.get("snapshot"):
        from swirl.engine.snapshot import check
        snaps = init["snapshot"]
        for sn in (snaps if isinstance(snaps, list) else [snaps]):     # a list: one entry per source (WP-P3)
            out.append({"snapshot": check(sn, L, absorber, sponge_tol)})
    for p in init.get("radial", []):
        g = radial_checks(p, L)
        c = np.asarray(p.get("center", (0, 0, 0)), float)
        rc = float(np.linalg.norm(c))
        rec = {"center": c.tolist(), **g}
        if absorber:
            r_out = absorber.get("r_out") or L / 2
            r_in = r_out - absorber["width"]
            dist = r_in - rc
            if dist <= 0:
                raise ValueError("lump centre inside the sponge: %s" % rec)
            r_tab = np.asarray(p["r"], float)
            phi_in = float(abs(radial_spline(r_tab, p["Phi"])(dist))) if dist < r_tab[-1] else 0.0
            rec.update(r_in=r_in, dist_to_sponge=dist, Phi_at_sponge_over_Phi0=phi_in / abs(p["Phi"][0]))
            if rec["Phi_at_sponge_over_Phi0"] > sponge_tol:
                raise ValueError("lump tail not negligible at the sponge inner radius: %s" % rec)
        out.append(rec)
    return out


def lump_config(cfg_lumps):
    c = dict(LUMP_DEFAULTS)
    c.update(cfg_lumps)
    if not c["centers"]:
        raise ValueError("lumps.centers is required")
    return c


# ---------------------------------------------------------------------------
# Densities
# ---------------------------------------------------------------------------

def densities(ev):
    """(e, (px, py, pz)) on the grid from the evolver state."""
    fh = ev.rfft(ev.phi)
    g = []
    for kk in (ev.kx, ev.ky, ev.kz):
        kk = kk.copy()
        if ev.N % 2 == 0:
            kk[np.isclose(np.abs(kk), np.pi / ev.dx)] = 0.0
        g.append(ev.irfft(1j * kk * fh))
    pi = ev.pi
    p2 = ev.phi ** 2
    e = 0.5 * pi ** 2 + 0.5 * (g[0] ** 2 + g[1] ** 2 + g[2] ** 2) \
        + p2 * (0.5 * ev.m ** 2 + p2 * (-ev.g4 / 24.0 + p2 * ev.g6 / 720.0))
    return e, (-pi * g[0], -pi * g[1], -pi * g[2])


def _wrap(x, L):
    return x - L * np.round(x / L)


def lever_arms(coords, L, origin):
    """Minimum-image coordinates about `origin`, one broadcastable array per axis. The grid plane
    at exactly L/2 from the origin has two images (+L/2 and -L/2) and gets lever 0, their mean
    (otherwise it carries an unpaired lever -L/2: the WP-C pair L_z of 4e-5 to 9e-5)."""
    out = []
    for a in range(3):
        x = _wrap(coords - origin[a], L)
        x[np.abs(np.abs(x) - L / 2) < 1e-9 * L] = 0.0
        out.append(x.reshape([-1 if b == a else 1 for b in range(3)]))
    return out


def pair_distances(X, L):
    """K x K minimum-image distance matrix of positions X (K x 3)."""
    X = np.asarray(X, float)
    dv = _wrap(X[None, :, :] - X[:, None, :], L)
    return np.sqrt(np.sum(dv ** 2, axis=-1))


def trilinear(coords, L, dx, center):
    N = coords.size
    f = [((c + L / 2) / dx) % N for c in center]
    i0 = [int(np.floor(v)) for v in f]
    fr = [v - i for v, i in zip(f, i0)]
    idx, wts = [], []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                idx.append(((i0[0] + a) % N, (i0[1] + b) % N, (i0[2] + c) % N))
                wts.append((fr[0] if a else 1 - fr[0]) * (fr[1] if b else 1 - fr[1])
                           * (fr[2] if c else 1 - fr[2]))
    return idx, wts


def ball_weight(r, R, taper):
    u = np.clip((r - (R - taper)) / taper, 0.0, 1.0)
    return np.cos(0.5 * np.pi * u) ** 2


def split_weight(rel, dvs, R, taper, ell, d_nn):
    """Weight of lump i at offsets rel (3 broadcastable arrays) from its centre; dvs: minimum-image
    vectors from c_i to the other tracked centres. W = b(r) / sum_j exp((r_i^2 - r_j^2)/(ell d_nn)),
    j over i (term 1) and the lumps closer than 2 R + 4 ell (balls that can overlap).
    r_i^2 - r_j^2 = 2 d_ij s_ij with s_ij the signed distance past the bisector plane, so for
    K = 2 (d_nn = d) W = b 0.5 (1 - tanh(s/ell)), the WP-C split; for any K the weights of the
    lumps covering a point sum to 1 (a product of pair factors would not)."""
    r = np.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2)
    S = np.ones_like(r)
    for dv in dvs:
        dn = float(np.linalg.norm(dv))
        if dn >= 2 * R + 4 * ell:
            continue
        arg = (2 * (rel[0] * dv[0] + rel[1] * dv[1] + rel[2] * dv[2]) - dn * dn) / (ell * d_nn)
        S = S + np.exp(np.minimum(arg, 700.0))
    return ball_weight(r, R, taper) / S


# ---------------------------------------------------------------------------
# Lump finder
# ---------------------------------------------------------------------------

def _interp_periodic(f, coords, L, dx, pts):
    out = []
    for p in pts:
        idx, w = trilinear(coords, L, dx, p)
        out.append(sum(wi * f[i] for i, wi in zip(idx, w)))
    return np.array(out)


def _join_labels(n, pairs, saddle_merge):
    """Union of maxima 0..n-1 whose saddle ratio is >= saddle_merge; returns the label list."""
    lab = list(range(n))
    for pr in pairs:
        if pr["saddle_ratio"] >= saddle_merge:
            la, lb = lab[pr["pair"][0]], lab[pr["pair"][1]]
            lab = [la if x == lb else x for x in lab]
    return lab


def find_lumps(eavg, coords, L, dx, threshold=0.01, min_sep=1.0, saddle_merge=0.9, abs_floor=0.0):
    """Maxima of a (period-averaged) energy density, from coordinates only.

    count: maxima above max(threshold * max(eavg), abs_floor), at least min_sep apart (min_sep only
    removes duplicates of one maximum on neighbouring grid points). count_distinct: maxima
    joined when the density along the segment between them never drops below
    saddle_merge times the smaller peak (a dimpled single lump counts once).

    abs_floor (default 0 = off): an absolute density below which a maximum is debris, not a lump. The
    relative threshold alone counts debris on a nearly empty field (F-018). LumpDiagnostics sets it to
    count_floor_frac (0.05) times the initial lump scale; see the module docstring for the reason."""
    mx = float(eavg.max())
    loc = (maximum_filter(eavg, size=3, mode="wrap") == eavg) & (eavg >= max(threshold * mx, abs_floor))
    cand = np.argwhere(loc)
    vals = eavg[loc]
    order = np.argsort(-vals)
    N = eavg.shape[0]
    peaks = []
    for k in order:
        i = cand[k]
        pos = []
        for a in range(3):
            im, ip = i.copy(), i.copy()
            im[a] = (i[a] - 1) % N
            ip[a] = (i[a] + 1) % N
            fm, f0, fp = eavg[tuple(im)], eavg[tuple(i)], eavg[tuple(ip)]
            den = fm - 2 * f0 + fp
            off = 0.5 * (fm - fp) / den if den != 0 else 0.0
            pos.append(coords[i[a]] + float(np.clip(off, -0.5, 0.5)) * dx)
        pos = np.array(pos)
        if all(np.linalg.norm(_wrap(pos - q["pos"], L)) >= min_sep for q in peaks):
            peaks.append({"pos": pos, "value": float(vals[k])})
    pairs = []
    for a in range(len(peaks)):
        for b in range(a + 1, len(peaks)):
            pa, pb = peaks[a]["pos"], peaks[b]["pos"]
            dv = _wrap(pb - pa, L)
            n = max(8, int(4 * np.linalg.norm(dv) / dx))
            pts = [pa + s * dv for s in np.linspace(0, 1, n + 1)]
            line = _interp_periodic(eavg, coords, L, dx, pts)
            pairs.append({"pair": [a, b], "distance": float(np.linalg.norm(dv)),
                          "saddle_ratio": float(line.min() / min(peaks[a]["value"], peaks[b]["value"]))})
    lab = _join_labels(len(peaks), pairs, saddle_merge)
    return {"count": len(peaks), "count_distinct": len(set(lab)), "labels": lab,
            "peaks": [{"pos": p["pos"].tolist(), "value": p["value"]} for p in peaks],
            "pairs": pairs, "max": mx, "threshold": threshold, "min_sep": min_sep, "saddle_merge": saddle_merge,
            "abs_floor": float(abs_floor)}


def refloor(entry, abs_floor):
    """Recount a stored find_lumps entry with an absolute floor (post-processing of saved results).

    Keeps the maxima at or above max(abs_floor, threshold * max), the same cut find_lumps applies, and
    recomputes labels and count_distinct from the stored saddle ratios (a saddle ratio belongs to the
    pair, so the others are unchanged). Exact: the stored maxima are a superset of what any cut at or
    above entry["threshold"] * entry["max"] keeps. Returns a new entry; the input is not modified."""
    cut = max(float(abs_floor), entry["threshold"] * entry["max"])
    keep = [i for i, p in enumerate(entry["peaks"]) if p["value"] >= cut]
    new_idx = {old: new for new, old in enumerate(keep)}
    pairs = [{**pr, "pair": [new_idx[pr["pair"][0]], new_idx[pr["pair"][1]]]} for pr in entry["pairs"]
             if pr["pair"][0] in new_idx and pr["pair"][1] in new_idx]
    lab = _join_labels(len(keep), pairs, entry["saddle_merge"])
    out = dict(entry)
    out.update(count=len(keep), count_distinct=len(set(lab)), labels=lab,
               peaks=[entry["peaks"][i] for i in keep], pairs=pairs, abs_floor=float(abs_floor))
    return out


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class LumpDiagnostics:
    """Samples per-lump and field quantities; state is checkpointable."""

    def __init__(self, ev, cfg):
        self.ev = ev
        self.cfg = cfg
        self.K = len(cfg["centers"])
        self.centers = np.array(cfg["centers"], float)
        self.series = {"t": [], "E": [], "X": [], "It": [], "Ip": [], "P": [],
                       "P_total": [], "L_axes": [], "P_half": [], "centers_used": []}
        self.point = {"t": [], "phi": []}
        self.counts = []
        self.acc = None
        self.acc_w = 0.0
        self.win_start = 0.0
        self._set_point_idx()
        L, c = ev.L, ev.coords
        self._axes = []
        for ax in cfg["axes"]:
            o = np.asarray(ax["origin"], float)
            n = np.asarray(ax["dir"], float)
            n = n / np.linalg.norm(n)
            self._axes.append((lever_arms(c, L, o), n))
        self._planes = []
        for pl in cfg["planes"]:
            a, s = int(pl["axis"]), float(pl["pos"])
            self._planes.append((a, int(np.searchsorted(c, s, side="left"))))
        self.win_len = cfg["count_periods"] * 2 * np.pi / cfg["omega_nominal"]

    # -- state ---------------------------------------------------------------
    def state(self):
        return {"centers": self.centers.tolist(), "series": self.series, "point": self.point,
                "counts": self.counts, "acc_w": self.acc_w, "win_start": self.win_start,
                "acc_present": self.acc is not None}

    def restore(self, st, acc):
        self.centers = np.array(st["centers"], float)
        self.series, self.point, self.counts = st["series"], st["point"], st["counts"]
        self.acc_w, self.win_start = st["acc_w"], st["win_start"]
        self.acc = acc if st["acc_present"] else None
        self._set_point_idx()

    def _set_point_idx(self):
        self._pidx = [trilinear(self.ev.coords, self.ev.L, self.ev.dx, c) for c in self.centers]
        self._wcache = None

    def _ball_weights(self):
        """[(index tuple, rel, W)] per lump for the current tracked centres (cached until they move)."""
        if self._wcache is None:
            ev, cfg = self.ev, self.cfg
            L, dx, N = ev.L, ev.dx, ev.N
            R, tap, ell = cfg["R_ball"], cfg["taper"], cfg["split_width"]
            nb = int(np.ceil(R / dx)) + 1
            d_nn = ell
            if self.K > 1:
                D = pair_distances(self.centers, L)
                d_nn = max(float(np.min(D[np.triu_indices(self.K, 1)])), ell)
            out = []
            for i in range(self.K):
                c = self.centers[i]
                i0 = np.round((c + L / 2) / dx).astype(int)
                ids = [(i0[a] + np.arange(-nb, nb + 1)) % N for a in range(3)]
                rel = [_wrap(ev.coords[ids[a]] - c[a], L).reshape([-1 if b == a else 1 for b in range(3)])
                       for a in range(3)]
                dvs = [_wrap(self.centers[j] - c, L) for j in range(self.K) if j != i]
                out.append((np.ix_(*ids), rel, split_weight(rel, dvs, R, tap, ell, d_nn)))
            self._wcache = out
        return self._wcache

    def count_floor(self):
        """Absolute floor for the lump finder: count_floor_frac * reference density. The reference is
        cfg["count_floor_ref"], else the maximum of the first count window (read from counts[0]); 0 until
        a reference exists (the first window), and 0 when count_floor_frac is 0."""
        frac = float(self.cfg.get("count_floor_frac") or 0.0)
        ref = self.cfg.get("count_floor_ref")
        if ref is None and self.counts:
            ref = self.counts[0]["max"]
        return frac * float(ref) if (frac > 0 and ref) else 0.0

    # -- every step ------------------------------------------------------------
    def read_points(self, t):
        phi = self.ev.phi
        self.point["t"].append(float(t))
        self.point["phi"].append([float(sum(w * phi[i] for i, w in zip(idx, wts))) for idx, wts in self._pidx])
        if self.cfg.get("it_every_step"):
            # int W phi_t^2 per lump every step (no FFT): the whole-period average of the ball N
            # then needs no interpolation of a signal with 2, 4 and 6 omega content
            pi = self.ev.pi
            self.point.setdefault("It", []).append(
                [float((pi[sub] ** 2 * W).sum()) * self.ev.dV for sub, _, W in self._ball_weights()])

    # -- every `every` steps -----------------------------------------------------
    def sample(self, t):
        ev, cfg = self.ev, self.cfg
        L, dx, N, dV = ev.L, ev.dx, ev.N, ev.dV
        e, p = densities(ev)
        rows = {k: [] for k in ("E", "X", "It", "Ip", "P")}
        new_c = self.centers.copy()
        for i, (sub, rel, W) in enumerate(self._ball_weights()):
            c = self.centers[i]
            es = e[sub] * W
            Ei = float(es.sum()) * dV
            Xi = c + np.array([float((es * rel[a]).sum()) * dV / Ei for a in range(3)])
            rows["E"].append(Ei)
            rows["X"].append(Xi.tolist())
            rows["It"].append(float((ev.pi[sub] ** 2 * W).sum()) * dV)
            rows["Ip"].append(float((ev.phi[sub] ** 2 * W).sum()) * dV)
            rows["P"].append([float((p[a][sub] * W).sum()) * dV for a in range(3)])
            new_c[i] = Xi
        for k in rows:
            self.series[k].append(rows[k])
        self.series["t"].append(float(t))
        self.series["centers_used"].append(self.centers.tolist())
        self.series["P_total"].append([float(p[a].sum()) * dV for a in range(3)])
        Lx = []
        for X, n in self._axes:
            cr = [X[1] * p[2] - X[2] * p[1], X[2] * p[0] - X[0] * p[2], X[0] * p[1] - X[1] * p[0]]
            Lx.append(float(sum(n[a] * cr[a].sum() for a in range(3))) * dV)
        self.series["L_axes"].append(Lx)
        ph = []
        for a, k in self._planes:
            sl = [slice(None)] * 3
            sl[a] = slice(0, k)
            ph.append(float(p[a][tuple(sl)].sum()) * dV)
        self.series["P_half"].append(ph)
        g = cfg.get("group")
        if g:
            self._sample_group(g, e, p, rows)
        # lump-count window (Hann weights over count_periods nominal periods)
        if t >= self.win_start:
            u = (t - self.win_start) / self.win_len
            if u <= 1.0:
                w = np.sin(np.pi * u) ** 2
                if self.acc is None:
                    self.acc = np.zeros_like(e)
                    self.acc_w = 0.0
                if w > 0:
                    self.acc += w * e
                    self.acc_w += w
            else:
                if self.acc is not None and self.acc_w > 0:
                    res = find_lumps(self.acc / self.acc_w, ev.coords, L, dx,
                                     cfg["count_threshold"], cfg["count_min_sep"], cfg["saddle_merge"],
                                     abs_floor=self.count_floor())
                    res["t_mid"] = self.win_start + 0.5 * self.win_len
                    self.counts.append(res)
                self.acc = None
                self.acc_w = 0.0
                while self.win_start < t:          # next window starts on the count_every grid
                    self.win_start += cfg["count_every"]
        self.centers = new_c
        self._set_point_idx()


    def _sample_group(self, g, e, p, rows):
        """Group sphere about the energy-weighted centroid G of the lump centroids: energy inside
        (ball weight, radius g["R"], taper g.get("taper", 2)), angular momentum about G of the
        whole box and of the sphere (3-vectors), and its component on the pattern axis g["axis"]."""
        ev = self.ev
        L, dV = ev.L, ev.dV
        Es = np.array(rows["E"])
        Xs = np.array(rows["X"])
        s = self.series
        # minimum image about the previous group centre (first sample: mean of the configured
        # centres), so a group wider than L/2 about any one lump is still handled
        G0 = np.array(s["G"][-1]) if s.get("G") else np.mean(np.array(self.cfg["centers"], float), axis=0)
        G = _wrap(G0 + (Es[:, None] * _wrap(Xs - G0, L)).sum(0) / Es.sum(), L)
        rel = lever_arms(ev.coords, L, G)
        Wg = ball_weight(np.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2), g["R"], g.get("taper", 2.0))
        cr = [rel[1] * p[2] - rel[2] * p[1], rel[2] * p[0] - rel[0] * p[2], rel[0] * p[1] - rel[1] * p[0]]
        Lbox = [float(x.sum()) * dV for x in cr]
        Lsph = [float((x * Wg).sum()) * dV for x in cr]
        n = np.asarray(g.get("axis", [0.0, 0.0, 1.0]), float)
        n = n / np.linalg.norm(n)
        for k in ("G", "E_group", "L_group_box", "L_group_sphere", "L_axis_box", "L_axis_sphere"):
            s.setdefault(k, [])
        s["G"].append(G.tolist())
        s["E_group"].append(float((e * Wg).sum()) * dV)
        s["L_group_box"].append(Lbox)
        s["L_group_sphere"].append(Lsph)
        s["L_axis_box"].append(float(np.dot(n, Lbox)))
        s["L_axis_sphere"].append(float(np.dot(n, Lsph)))


# ---------------------------------------------------------------------------
# Whole-period averages (post-processing)
# ---------------------------------------------------------------------------

def _windows(tz, ts, W):
    """(za, zb, half_periods) or None for each output time: the window between zero
    crossings inside [t - W, t] with an even number of half periods (>= 2)."""
    out = []
    for t in ts:
        i0 = int(np.searchsorted(tz, t - W - 1e-12, side="left"))
        i1 = int(np.searchsorted(tz, t + 1e-12, side="right")) - 1
        if (i1 - i0) % 2:
            i1 -= 1
        out.append(None if i1 - i0 < 2 else (tz[i0], tz[i1], i1 - i0))
    return out


def _avg(spl_int, za, zb):
    return (spl_int(zb) - spl_int(za)) / (zb - za)


def summarize(diag_state, t_out, W):
    """Whole-period averages of the lump series at the times t_out."""
    from swirl.engine.run import zero_crossings
    s, pt = diag_state["series"], diag_state["point"]
    ts = np.array(s["t"])
    K = len(diag_state["centers"])
    tp = np.array(pt["t"])
    phis = np.array(pt["phi"]).reshape(len(tp), K)
    out = {"t": list(map(float, t_out)), "lumps": [], "relative_phase": [], "P_total": None,
           "P_half": None, "L_axes": None, "omega_window_lump0": None, "t_center_lump0": None,
           "note": "each average is over whole periods of that lump's phi(c_i, t) inside [t - W, t] "
                   "(It from the per-step record when it_every_step is set); "
                   "plot it at t_center (window centre), not at t; P_total, P_half, L_axes and the "
                   "relative phase use lump 0's windows"}
    if ts.size < 4:
        return out
    E = np.array(s["E"]).reshape(ts.size, K)
    X = np.array(s["X"]).reshape(ts.size, K, 3)
    It = np.array(s["It"]).reshape(ts.size, K)
    Ip = np.array(s["Ip"]).reshape(ts.size, K)
    P = np.array(s["P"]).reshape(ts.size, K, 3)
    tzs, wins = [], []
    for k in range(K):
        tz = zero_crossings(tp, phis[:, k])
        tzs.append(tz)
        wins.append(_windows(tz, t_out, W))

    def per_window(y, win):
        cs = CubicSpline(ts, y, axis=0).antiderivative()
        res = []
        for wv in win:
            if wv is None or wv[0] < ts[0] or wv[1] > ts[-1]:
                res.append(None)
            else:
                v = _avg(cs, wv[0], wv[1])
                res.append(v.tolist() if np.ndim(v) else float(v))
        return res

    It_step = np.array(pt["It"]).reshape(len(tp), K) if pt.get("It") else None

    def per_window_step(y, win):
        cs = CubicSpline(tp, y).antiderivative()
        return [None if wv is None or wv[0] < tp[0] or wv[1] > tp[-1] else float(_avg(cs, wv[0], wv[1]))
                for wv in win]

    for k in range(K):
        win = wins[k]
        om = [None if wv is None else float(np.pi * wv[2] / (wv[1] - wv[0])) for wv in win]
        itav = per_window(It[:, k], win) if It_step is None else per_window_step(It_step[:, k], win)
        out["lumps"].append({
            "t_center": [None if wv is None else float(0.5 * (wv[0] + wv[1])) for wv in win],
            "omega": om, "E": per_window(E[:, k], win), "X": per_window(X[:, k, :], win),
            "It": itav, "Ip": per_window(Ip[:, k], win), "P": per_window(P[:, k, :], win),
            "N": [None if (a is None or w is None) else a / w for a, w in zip(itav, om)],
            "n_zero_crossings": int(tzs[k].size)})
    win0 = wins[0]
    out["omega_window_lump0"] = out["lumps"][0]["omega"]
    out["t_center_lump0"] = out["lumps"][0]["t_center"]
    for name, key in (("P_total", "P_total"), ("P_half", "P_half"), ("L_axes", "L_axes")):
        arr = np.array(s[key])
        if arr.size and arr.ndim == 2 and arr.shape[1] > 0:
            out[name] = per_window(arr, win0)
    # relative phase of lump k against lump 0: Fourier projection over lump 0's window
    if K > 1:
        for k in range(1, K):
            rp = []
            for wv in win0:
                if wv is None:
                    rp.append(None)
                    continue
                za, zb, hp = wv
                w = np.pi * hp / (zb - za)
                sel = (tp >= za) & (tp <= zb)
                tt = tp[sel]
                ker = np.exp(-1j * w * tt) * np.sin(np.pi * (tt - za) / (zb - za)) ** 2   # Hann
                c0 = np.trapezoid(phis[sel, 0] * ker, tt)
                ck = np.trapezoid(phis[sel, k] * ker, tt)
                rp.append(float(np.angle(ck * np.conj(c0))))
            out["relative_phase"].append({"lumps": [0, k], "dtheta": rp})
    return out
