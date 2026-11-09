"""Test B analysis (plan section 3, B3) and verdict (B4, implemented as written).

Spectra: Welch, Hann (periodic) windows of TWIN = 2000 time units, 50% overlap, inside
epochs of 5000 time units ([0,5000], [5000,10000], ...; one epoch [0, T] if T < 5000).
One-sided power per bin, normalised so that sum_j P_j = <x^2> (a line a cos(w t) has
total power a^2/2); cross spectra the same way, sum_j C_j = <x y>. Resolution
d_omega = 2 pi / TWIN = 0.00314.

Lines: bins that are the maximum over +-3 bins and at least CONTRAST times the median
power 10 to 40 bins away (Hann sidelobes fall monotonically, so they are never a +-3-bin
maximum). Frequency by parabolic interpolation of log P. A line is labelled n omega_core
when |omega - n omega_core| <= 2 bins (n >= 1), else "unidentified" (nearest n and offset
kept). Line power = sum of P over +-W bins, W = 3 + ceil(n * drift / (2 d_omega)) with
drift the spread of the per-window core omega in the epoch (W = 3 if unidentified).
Amplitude a = sqrt(2 P_line); outgoing amplitude a_n = r a.

Threshold omega_th from B1 (RK4 at the run's dt); cut omega_cut = omega_th - delta,
delta = 3 bins. "Sub-threshold" = omega < omega_cut; lines with omega >= omega_cut are the
"above-threshold" lines of B4 criterion 3.

Floors (per probe, in power):
  leakage    = sub-threshold power that windowing alone makes from the measured content at
               omega >= omega_cut: (a) every above-cut harmonic line (n omega) resynthesised
               as a pure line (measured omega and amplitude) through the same pipeline; (b) the
               content in [omega_cut, omega_cut + 0.5] (the near-threshold transient, a chirp)
               fitted by pure tones on a quarter-bin grid with the pipeline's own response
               (non-negative least squares, relative residuals), each tone's sub-threshold
               fraction summed. The plan's synthetic line at 1.005 is reported per epoch.
  evanescent = (A e^(-kappa r)/r)^2 / 2, A from the start profile, kappa = sqrt(1 - omega_c^2)
               with omega_c the epoch's measured core omega (the profile kappa is reported too).
Flux: method 1, 4 pi r^2 <-phi_t phi_r> (per line: minus the phi_t/phi_r cross spectrum summed
over the line band; total: the whole cross spectrum, and the plain time average); method 2,
2 pi a_n^2 omega k, k = sqrt(omega^2 - m^2).
Loss rate: -dE_in(3 hw)/dt and d(absorbed)/dt with the weight the spectra carry (Hann^2 per
window, mean over the epoch's windows), so that flux and loss are the same time average;
uniform endpoint rates are reported beside them.
Harmonics above the probe Nyquist frequency (pi / 0.2 = 15.7, n >= 19) fold back and are
labelled "n omega aliased".

Run: <venv python> -m swirl.session4.testB.analysis <name> [--box2 <name>] [--compare <name>[,<name>]]
"""

import json
import math
import os
import sys

import numpy as np
from scipy.signal.windows import hann

from swirl.session4.testB.dispersion import threshold

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "outputs", "session4", "testB")

TWIN = 2000.0
EPOCH = 5000.0
DOMEGA = 2 * np.pi / TWIN
DELTA_BINS = 3
ID_BINS = 2
CONTRAST = 30.0
HALF = 3
R_FAR = 30.0
FOUR_PI = 4 * np.pi
SYNTH_OMEGA = 1.005


# ------------------------------------------------------------------ spectra
def epochs_for(T):
    if T < EPOCH:
        return [(0.0, float(T))]
    n = int(math.floor(T / EPOCH + 1e-9))
    return [(k * EPOCH, (k + 1) * EPOCH) for k in range(n)]


def segments(dt, t0, t1, nmax, twin=TWIN):
    n = int(round(twin / dt))
    i0, i1 = int(round(t0 / dt)), min(int(round(t1 / dt)), nmax)
    return n, list(range(i0, i1 - n + 1, n // 2))


def welch(x, dt, t0, t1, y=None, twin=TWIN):
    """(omega, P, C or None, nseg). P one-sided, sum P = <x^2>; C: sum C = <x y>."""
    n, starts = segments(dt, t0, t1, len(x), twin)
    if not starts:
        raise ValueError("epoch [%g, %g] shorter than one window" % (t0, t1))
    w = hann(n, sym=False)
    s2 = float(np.sum(w * w))
    c = np.full(n // 2 + 1, 2.0)
    c[0] = 1.0
    if n % 2 == 0:
        c[-1] = 1.0
    P = np.zeros(n // 2 + 1)
    C = np.zeros(n // 2 + 1) if y is not None else None
    for s in starts:
        X = np.fft.rfft(w * np.asarray(x[s:s + n], dtype=float))
        P += c * (X.real**2 + X.imag**2) / (n * s2)
        if y is not None:
            Y = np.fft.rfft(w * np.asarray(y[s:s + n], dtype=float))
            C += c * (X.real * Y.real + X.imag * Y.imag) / (n * s2)
    P /= len(starts)
    if C is not None:
        C /= len(starts)
    om = 2 * np.pi * np.fft.rfftfreq(n, dt)
    return om, P, C, len(starts)


def refine(P, j):
    """Parabolic interpolation of log P about bin j; returns fractional bin."""
    if j <= 0 or j >= len(P) - 1 or min(P[j - 1], P[j], P[j + 1]) <= 0:
        return float(j)
    lm, l0, lp = np.log(P[j - 1]), np.log(P[j]), np.log(P[j + 1])
    den = lm - 2 * l0 + lp
    return float(j + (0.5 * (lm - lp) / den if den < 0 else 0.0))


def peak_in(om, P, lo, hi):
    sel = np.where((om >= lo) & (om <= hi))[0]
    j = int(sel[np.argmax(P[sel])])
    return refine(P, j) * (om[1] - om[0])


def find_peaks(P, contrast=CONTRAST, half=HALF):
    """Bins that are the maximum over +-half bins and >= contrast x the median 10-40 bins away."""
    out = []
    n = len(P)
    for j in range(n):
        if P[j] <= 0:
            continue
        lo, hi = max(0, j - half), min(n, j + half + 1)
        if P[j] < np.max(P[lo:hi]) or np.argmax(P[lo:hi]) != j - lo:
            continue
        bg = np.concatenate([P[max(0, j - 40):max(0, j - 9)], P[min(n, j + 10):min(n, j + 41)]])
        if bg.size == 0 or P[j] < contrast * np.median(bg):
            continue
        out.append(j)
    return out


def label(omega, omega_c, dom=DOMEGA, omega_nyq=None):
    n = int(round(omega / omega_c))
    if n >= 1 and abs(omega - n * omega_c) <= ID_BINS * dom:
        return n, "%d omega" % n
    if n == 0 and omega <= ID_BINS * dom:
        return 0, "static"
    if omega_nyq is not None:
        # first fold only: omega_nyq < m omega_c <= 2 omega_nyq (alias 2 omega_nyq - m omega_c)
        for m in range(int(omega_nyq / omega_c) + 1, int(2 * omega_nyq / omega_c) + 1):
            f = 2 * omega_nyq - m * omega_c
            if abs(omega - f) <= ID_BINS * dom:
                return m, "%d omega aliased" % m
    return n, "unidentified"


def lines_in(om, P, omega_c, drift, C=None, max_lines=25):
    """Identified spectral lines, strongest first, without double counting."""
    dom = om[1] - om[0]
    cand = sorted(find_peaks(P), key=lambda j: -P[j])
    taken = np.zeros(len(P), bool)
    out = []
    for j in cand:
        if taken[j]:
            continue
        w = refine(P, j) * dom
        n, lab = label(w, omega_c, dom, omega_nyq=om[-1])
        W = HALF + (int(math.ceil(n * drift / (2 * dom))) if lab.endswith("omega") else 0)
        lo, hi = max(0, j - W), min(len(P), j + W + 1)
        taken[lo:hi] = True
        pw = float(np.sum(P[lo:hi]))
        d = dict(omega=w, n=n, label=lab, offset_from_n_omega=w - n * omega_c, power=pw,
                 amp=math.sqrt(2 * pw), band=[int(lo), int(hi)], peak_bin_power=float(P[j]))
        if C is not None:
            d["cross"] = float(np.sum(C[lo:hi]))
        out.append(d)
        if len(out) >= max_lines:
            break
    return out


# ------------------------------------------------------------------ floors
def synth_sub_power(omega, amp, dt, t0, t1, nmax, omega_cut):
    t = dt * np.arange(nmax)
    om, P, _, _ = welch(amp * np.cos(omega * t), dt, t0, t1)
    return float(np.sum(P[om < omega_cut])), float(np.sum(P))


def tone_response(dt, t0, t1, nmax, omega_cut, span=0.5, extra=0.1, q=4):
    """Pipeline response to unit pure tones on a quarter-bin grid from omega_cut to
    omega_cut + span + extra: R[j, k] = Welch power at bin j of cos(omega_k t); L[k] = the
    sub-threshold fraction of tone k."""
    n, starts = segments(dt, t0, t1, nmax)
    om = 2 * np.pi * np.fft.rfftfreq(n, dt)
    dom = om[1] - om[0]
    wk = omega_cut + np.arange(int((span + extra) / dom * q) + 1) * dom / q
    w = hann(n, sym=False)
    s2 = float(np.sum(w * w))
    c = np.full(n // 2 + 1, 2.0)
    c[0] = 1.0
    if n % 2 == 0:
        c[-1] = 1.0
    tt = dt * np.arange(n)
    R = np.zeros((om.size, wk.size))
    for k, wl in enumerate(wk):
        for s in starts:
            X = np.fft.rfft(w * np.cos(wl * (tt + s * dt)))
            R[:, k] += c * (X.real**2 + X.imag**2) / (n * s2)
    R /= len(starts)
    L = R[om < omega_cut].sum(0) / R.sum(0)
    return om, wk, R, L


def near_threshold_leak(om, P, R, L, omega_cut, exclude=None, span=0.5):
    """Sub-threshold power that windowing alone makes from the content in
    [omega_cut, omega_cut + span]: that content is fitted (non-negative least squares,
    relative residuals) by pure tones with the pipeline's own response R, and their
    sub-threshold fractions L are summed. On synthetic pure tones, chirps and free
    Klein-Gordon transients this lands within 0.8x to 7x of the true leakage (biased
    high for a pure tone right at the cut); see tests/session4."""
    from scipy.optimize import nnls
    rows = (om >= omega_cut) & (om <= omega_cut + span)
    if exclude is not None:
        rows &= ~exclude
    rows = np.where(rows & (P > 0))[0]
    if rows.size == 0:
        return 0.0
    wq = 1.0 / P[rows]
    S, _ = nnls(R[rows] * wq[:, None], P[rows] * wq, maxiter=50 * R.shape[1])
    return float(S @ L)


def evanescent_power(A, kappa, r):
    a = A * np.exp(-kappa * r) / r
    return 0.5 * a * a


# ------------------------------------------------------------------ energy
def endpoint_mean(t, y, tc, half=10.0):
    sel = (t >= tc - half) & (t <= tc + half)
    return float(np.mean(y[sel]))


def rate(t, y, t0, t1):
    return (endpoint_mean(t, y, t1) - endpoint_mean(t, y, t0)) / (t1 - t0)


def weighted_rate(t, y, starts, twin=TWIN):
    """dy/dt averaged with the weight the spectra use (Hann^2 per window, mean over the
    windows), by parts: <y'>_w2 = -int (w^2)' y dt / int w^2 dt. Spectral fluxes are
    Hann^2-weighted time averages, so this is the loss rate to set against them."""
    out = []
    for s in starts:
        sel = (t >= s) & (t <= s + twin)
        tau = (t[sel] - s) / twin
        w = 0.5 * (1 - np.cos(2 * np.pi * tau))
        dw2 = 2 * w * (np.pi / twin) * np.sin(2 * np.pi * tau)
        out.append(-np.trapezoid(dw2 * y[sel], t[sel]) / np.trapezoid(w * w, t[sel]))
    return float(np.mean(out))


# ------------------------------------------------------------------ main analysis
def analyse(name, series_dir=OUT):
    z = np.load(os.path.join(series_dir, name + "_series.npz"), allow_pickle=False)
    P_run = json.loads(str(z["params"]))
    t = z["t"]
    dt = float(t[1] - t[0])
    radii = z["probes"]
    phi, phit, phir = z["phi"], z["phit"].astype(float), z["phir"].astype(float)
    te, E_in, absd = z["t_energy"], z["E_in"], z["absorbed"]
    nmax = t.size
    w_th = threshold(P_run["m"], P_run["dt"])
    w_cut = w_th - DELTA_BINS * DOMEGA
    prof = P_run["profile"]
    A = prof["A"]
    out = dict(name=name, params=P_run, omega_th=w_th, omega_cut=w_cut, delta=DELTA_BINS * DOMEGA,
               d_omega=DOMEGA, twin=TWIN, epoch_length=EPOCH, contrast=CONTRAST, epochs=[])
    for (t0, t1) in epochs_for(t[-1]):
        ep = dict(t0=t0, t1=t1)
        # core omega: epoch spectrum at r = 0 and per window
        om, P0, _, nseg = welch(phi[:, 0], dt, t0, t1)
        wc = peak_in(om, P0, 0.5, w_cut)
        n, starts = segments(dt, t0, t1, nmax)
        wseg = [peak_in(*welch(phi[:, 0], dt, s * dt, s * dt + TWIN)[:2], 0.5, w_cut) for s in starts]
        drift = float(max(wseg) - min(wseg))
        ep.update(omega_core=wc, omega_core_per_window=wseg, window_starts=[s * dt for s in starts],
                  drift=drift, n_windows=nseg)
        kap = math.sqrt(max(1 - wc * wc, 0.0))
        ep["kappa_epoch"] = kap
        # leakage response
        _, wk, Rt, Lt = tone_response(dt, t0, t1, nmax, w_cut)
        sp, tp = synth_sub_power(SYNTH_OMEGA, 1.0, dt, t0, t1, nmax, w_cut)
        ep["synthetic_1.005"] = dict(sub_fraction=sp / tp, L_grid_at_1005=float(np.interp(SYNTH_OMEGA, wk, Lt)))
        # core line through the pipeline: pure and with the measured linear drift
        a0 = math.sqrt(2 * np.sum(P0[np.abs(om - wc) <= HALF * DOMEGA]))
        tt = dt * np.arange(nmax)
        slope = (wseg[-1] - wseg[0]) / max(starts[-1] * dt - starts[0] * dt, 1.0) if len(wseg) > 1 else 0.0
        core_leak = {}
        for key, sig in (("pure", a0 * np.cos(wc * tt)),
                         ("drifting", a0 * np.cos(wc * tt + 0.5 * slope * (tt - 0.5 * (t0 + t1)) ** 2))):
            o2, Ps, _, _ = welch(sig, dt, t0, t1)
            tot = np.sum(Ps)
            core_leak[key] = {"frac_outside_pm%dbins" % b: float(np.sum(Ps[np.abs(o2 - wc) > b * DOMEGA]) / tot)
                              for b in (3, 10, 30)}
            core_leak[key]["sub_cut_fraction_excluding_pm3bins"] = float(
                np.sum(Ps[(o2 < w_cut) & (np.abs(o2 - wc) > 3 * DOMEGA)]) / tot)
        ep["core_line_leakage"] = dict(amp_r0=a0, drift_rate=slope, **core_leak)
        # core spectrum
        ep["core_lines"] = lines_in(om, P0, wc, drift, max_lines=30)
        # probes
        probes = []
        for ip, r in enumerate(radii):
            om, P, C, _ = welch(phi[:, ip], dt, t0, t1, y=None)
            _, Pt, C, _ = welch(phit[:, ip], dt, t0, t1, y=phir[:, ip])
            tot = float(np.sum(P))
            sub = float(np.sum(P[om < w_cut]))
            lines = lines_in(om, P, wc, drift, C=C)
            i0, i1 = int(round(t0 / dt)), min(int(round(t1 / dt)), nmax)
            flux_direct = float(FOUR_PI * r * r * np.mean(-phit[i0:i1, ip] * phir[i0:i1, ip]))
            harm_bands = np.zeros(len(P), bool)
            leak_lines = 0.0
            for d in lines:
                d["above"] = bool(d["omega"] >= w_cut)
                d["a_n"] = r * d["amp"]
                k = math.sqrt(max(d["omega"] ** 2 - 1.0, 0.0))
                d["k"] = k
                # method 1 per line: band of the phi_t / phi_r cross spectrum (same bins)
                d["flux_m1"] = float(-FOUR_PI * r * r * d.pop("cross"))
                d["flux_m2"] = float(2 * np.pi * d["a_n"] ** 2 * d["omega"] * k)
                if d["above"] and d["label"].endswith("omega"):
                    # stable harmonic lines: resynthesised as pure lines through the pipeline
                    harm_bands[d["band"][0]:d["band"][1]] = True
                    leak_lines += synth_sub_power(d["omega"], d["amp"], dt, t0, t1, nmax, w_cut)[0]
            leak_cont = near_threshold_leak(om, P, Rt, Lt, w_cut, exclude=harm_bands)
            leak = leak_lines + leak_cont
            ev = float(evanescent_power(A, kap, r)) if r > 0 else float("nan")
            evp = float(evanescent_power(A, prof["kappa"], r)) if r > 0 else float("nan")
            fl = max(leak, ev) if r > 0 else float("nan")
            probes.append(dict(
                r=float(r), total_power=tot, sub_power=sub, sub_rel=sub / tot if tot > 0 else None,
                leak_floor=leak, leak_floor_harmonic_lines=leak_lines, leak_floor_near_threshold=leak_cont,
                ev_floor=ev, ev_floor_profile_kappa=evp,
                sub_over_floor=sub / fl if r > 0 and fl > 0 else None,
                flux_total_m1=flux_direct, flux_total_m1_spectral=float(-FOUR_PI * r * r * np.sum(C)),
                flux_above_m1=float(sum(d["flux_m1"] for d in lines if d["above"])),
                flux_above_m2=float(sum(d["flux_m2"] for d in lines if d["above"])),
                flux_sub_m1=float(-FOUR_PI * r * r * np.sum(C[om < w_cut])),
                lines=lines))
        ep["probes"] = probes
        # energy
        st = [s * dt for s in starts]
        ep["loss_rate_Ein_3hw"] = -weighted_rate(te, E_in[:, 0], st)
        ep["absorbed_rate"] = weighted_rate(te, absd, st)
        ep["minus_dEin_dt_at_probe"] = {str(float(r)): -weighted_rate(te, E_in[:, ip], st)
                                        for ip, r in enumerate(radii) if ip > 0}
        ep["uniform_endpoint_rates"] = dict(
            loss_rate_Ein_3hw=-rate(te, E_in[:, 0], t0, t1), absorbed_rate=rate(te, absd, t0, t1),
            minus_dEin_dt_at_probe={str(float(r)): -rate(te, E_in[:, ip], t0, t1)
                                    for ip, r in enumerate(radii) if ip > 0},
            flux_total_m1_time_average={str(float(p["r"])): p["flux_total_m1"] for p in probes},
            note="uniform weight over the epoch (endpoint means over +-10); not the weight of the spectra")
        ep["sub_groups"] = sub_groups(ep, w_cut)
        out["epochs"].append(ep)
    out["transient"] = transient(t, phi, radii, w_th)
    return out


def sub_groups(ep, w_cut):
    """Sub-threshold lines at far probes (power above max(floors)), grouped by frequency
    within ID_BINS bins, with their r-dependence across the far probes."""
    far = [p for p in ep["probes"] if p["r"] >= R_FAR]
    found = []
    for p in far:
        fl = max(p["leak_floor"], p["ev_floor"])
        for d in p["lines"]:
            if d["omega"] < w_cut and d["power"] > fl:
                found.append((d["omega"], p["r"], d["power"], fl, d["flux_m1"]))
    groups = []
    for w, r, pw, fl, fx in sorted(found):
        if groups and abs(w - groups[-1]["omega"]) <= ID_BINS * DOMEGA:
            g = groups[-1]
        else:
            g = dict(omega=w, det=[])
            groups.append(g)
        g["det"].append(dict(r=r, power=pw, floor=fl, flux_m1=fx))
    for g in groups:
        rs = np.array([d["r"] for d in g["det"]])
        amps = np.sqrt(2 * np.array([d["power"] for d in g["det"]]))
        g["n_probes"] = int(np.unique(rs).size)
        g["n_strong"] = int(sum(d["power"] >= 10 * d["floor"] for d in g["det"]))
        g["kappa_pred"] = math.sqrt(max(1 - g["omega"] ** 2, 0.0))
        g["exponent"] = g["kappa_fit"] = g["exponent_bound"] = None
        if g["n_probes"] >= 2:
            g["exponent"] = float(np.polyfit(np.log(rs), np.log(amps), 1)[0])
            g["kappa_fit"] = float(-np.polyfit(rs, np.log(amps * rs), 1)[0])
        else:
            # upper limit at the next far probe: band power at the same frequency there
            r1 = rs[0]
            nxt = [p for p in far if p["r"] > r1]
            if nxt:
                p = nxt[0]
                j = int(round(g["omega"] / DOMEGA))
                pw = None
                for d in p["lines"]:
                    if abs(d["omega"] - g["omega"]) <= ID_BINS * DOMEGA:
                        pw = d["power"]
                if pw is None:
                    pw = p["sub_power"] if j * DOMEGA < w_cut else p["total_power"]
                g["exponent_bound"] = float(np.log(math.sqrt(2 * pw) / amps[0]) / np.log(p["r"] / r1))
        g["consistent_evanescent"] = bool(g["kappa_fit"] is not None and g["kappa_pred"] > 0
                                          and abs(g["kappa_fit"] - g["kappa_pred"]) <= 0.3 * g["kappa_pred"])
        g["flux_m1_far"] = [d["flux_m1"] for d in g["det"]]
    return groups


def transient(t, phi, radii, w_th, twin=400.0, t_end=4000.0):
    """B-P7: dominant frequency in [0.5, 1.6] at far probes in short windows, against the
    free Klein-Gordon local frequency t / sqrt(t^2 - r^2)."""
    dt = float(t[1] - t[0])
    out = []
    dom = 2 * np.pi / twin
    for ip, r in enumerate(radii):
        if r < R_FAR:
            continue
        rows = []
        for t0 in np.arange(0.0, min(t_end, t[-1]) - twin + 1e-9, twin / 2):
            om, P, _, _ = welch(phi[:, ip], dt, t0, t0 + twin, twin=twin)
            band = (om >= 0.5) & (om <= 1.6)
            tm = t0 + twin / 2
            rows.append(dict(t_mid=float(tm), omega_peak=peak_in(om, P, 0.5, 1.6),
                             omega_kg=float(tm / math.sqrt(tm * tm - r * r)) if tm > r else None,
                             sub_fraction_of_band=float(np.sum(P[band & (om < w_th - 3 * dom)]) / np.sum(P[band]))))
        out.append(dict(r=float(r), windows=rows))
    return dict(twin=twin, d_omega=dom, probes=out)


# ------------------------------------------------------------------ verdict (B4)
def verdict_epoch(ep, box2_ep=None):
    far = [p for p in ep["probes"] if p["r"] >= R_FAR]
    L = ep["loss_rate_Ein_3hw"]
    c1_rows = []
    for p in far:
        fl = max(p["leak_floor"], p["ev_floor"])
        ok = p["sub_power"] <= 10 * fl or p["sub_power"] < 1e-4 * p["total_power"]
        c1_rows.append(dict(r=p["r"], sub_power=p["sub_power"], floor=fl, sub_over_floor=p["sub_power"] / fl,
                            sub_rel=p["sub_power"] / p["total_power"], ok=bool(ok)))
    c1 = all(r["ok"] for r in c1_rows)
    c2_rows = []
    for g in ep["sub_groups"]:
        if g["exponent"] is not None:
            ok = g["exponent"] < -1.3 or g["consistent_evanescent"]
        else:
            ok = g["exponent_bound"] is not None and g["exponent_bound"] < -1.3
        c2_rows.append(dict(omega=g["omega"], exponent=g["exponent"], exponent_bound=g["exponent_bound"],
                            kappa_fit=g["kappa_fit"], kappa_pred=g["kappa_pred"], ok=bool(ok)))
    c2 = all(r["ok"] for r in c2_rows)
    c3_rows = []
    for p in far:
        ratio = p["flux_above_m1"] / L if L > 0 else float("inf")
        c3_rows.append(dict(r=p["r"], flux_above_m1=p["flux_above_m1"], flux_above_m2=p["flux_above_m2"],
                            loss_rate=L, ratio_m1=ratio, ratio_m2=p["flux_above_m2"] / L if L > 0 else None,
                            ok=bool(1 / 3 <= ratio <= 3)))
    c3 = all(r["ok"] for r in c3_rows)
    only_above = c1 and c2 and c3
    # "present", clause A: a line at >= 2 far probes, >= 10 x both floors, exponent -1 +- 0.3,
    # reproduced by the doubled box
    pres_a = []
    for g in ep["sub_groups"]:
        strong = g["n_strong"] >= 2 and g["exponent"] is not None and abs(g["exponent"] + 1) <= 0.3
        conf = None
        if strong and box2_ep is not None:
            conf = any(abs(h["omega"] - g["omega"]) <= ID_BINS * DOMEGA and h["n_strong"] >= 2
                       and h["exponent"] is not None and abs(h["exponent"] + 1) <= 0.3
                       for h in box2_ep["sub_groups"])
        pres_a.append(dict(omega=g["omega"], strong_1_over_r=bool(strong), confirmed_box2=conf))
    a_present = any(x["strong_1_over_r"] and x["confirmed_box2"] for x in pres_a)
    a_unconfirmed = any(x["strong_1_over_r"] and not x["confirmed_box2"] for x in pres_a)
    # clause B: above-threshold flux short of the loss by > 3x and a sub-threshold line carries
    # the difference (within a factor 3), at >= 2 far probes
    nb = 0
    b_rows = []
    for p in far:
        short = L > 0 and p["flux_above_m1"] < L / 3
        sub_flux = sum(d["flux_m1"] for g in ep["sub_groups"] for d in g["det"] if d["r"] == p["r"])
        diff = L - p["flux_above_m1"]
        carries = short and diff > 0 and diff / 3 <= sub_flux <= 3 * diff
        nb += int(carries)
        b_rows.append(dict(r=p["r"], short=bool(short), sub_line_flux=sub_flux, deficit=diff, carries=bool(carries)))
    b_present = nb >= 2
    if a_present or b_present:
        v = "Sub-threshold emission present"
    elif only_above:
        v = "Emission only above threshold"
    else:
        v = "Undetermined"
    return dict(verdict=v, c1=c1, c2=c2, c3=c3, c1_rows=c1_rows, c2_rows=c2_rows, c3_rows=c3_rows,
                present_a=pres_a, present_a_unconfirmed_by_box2=a_unconfirmed, present_b_rows=b_rows,
                box2_available=box2_ep is not None)


def verdict(an, box2_an=None):
    """B4 per epoch; the doubled box (T = 5000) supplies its first epoch for confirmation.
    Overall: "present" if any epoch is, "only above" if every epoch is, else "undetermined"."""
    b2 = box2_an["epochs"][0] if box2_an is not None else None
    per = [dict(t0=ep["t0"], t1=ep["t1"], **verdict_epoch(ep, b2)) for ep in an["epochs"]]
    vs = [p["verdict"] for p in per]
    if "Sub-threshold emission present" in vs:
        overall = "Sub-threshold emission present"
    elif all(v == "Emission only above threshold" for v in vs):
        overall = "Emission only above threshold"
    else:
        overall = "Undetermined"
    return dict(overall=overall, per_epoch=per)


# ------------------------------------------------------------------ controls
def compare(name_a, name_b, series_dir=OUT):
    """Direct comparison of probe series over the common time range (box and resolution)."""
    a = np.load(os.path.join(series_dir, name_a + "_series.npz"))
    b = np.load(os.path.join(series_dir, name_b + "_series.npz"))
    n = min(a["t"].size, b["t"].size)
    if np.max(np.abs(a["t"][:n] - b["t"][:n])) > 1e-9:
        raise ValueError("time grids differ")
    rows = []
    for ip, r in enumerate(a["probes"]):
        x, y = a["phi"][:n, ip], b["phi"][:n, ip]
        rows.append(dict(r=float(r), max_abs_diff=float(np.max(np.abs(x - y))),
                         max_abs=float(np.max(np.abs(x))),
                         rel_max=float(np.max(np.abs(x - y)) / np.max(np.abs(x))),
                         rel_rms=float(np.sqrt(np.mean((x - y) ** 2) / np.mean(x * x)))))
    ne = min(a["t_energy"].size, b["t_energy"].size)
    return dict(a=name_a, b=name_b, t_common=float(a["t"][n - 1]), probes=rows,
                E_in_3hw_max_abs_diff_over_E0=float(np.max(np.abs(a["E_in"][:ne, 0] - b["E_in"][:ne, 0])) / a["E_tot"][0]))


# ------------------------------------------------------------------ plot
def plot(an, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(os.path.join(OUT, an["name"] + "_series.npz"))
    t, phi, radii = z["t"], z["phi"], z["probes"]
    dt = float(t[1] - t[0])
    fig, axs = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    for ip, ax in enumerate(axs.ravel()):
        for k, ep in enumerate(an["epochs"]):
            om, P, _, _ = welch(phi[:, ip], dt, ep["t0"], ep["t1"])
            ax.semilogy(om, np.maximum(P, 1e-40), lw=0.8, label="[%g, %g]" % (ep["t0"], ep["t1"]))
            if k == 0:
                for n in range(1, 7):
                    ax.axvline(n * ep["omega_core"], color="0.8", lw=0.6, zorder=0)
        ax.axvline(an["omega_th"], color="k", ls="--", lw=0.8)
        ax.set_xlim(0, 6)
        ax.set_title("r = %g" % radii[ip], fontsize=9)
        ax.tick_params(labelsize=7)
    axs[0, 0].legend(fontsize=6, frameon=False)
    for ax in axs[-1]:
        ax.set_xlabel("omega / m")
    fig.suptitle("%s: Welch power per bin (Hann 2000, 50%%); dashed: omega_th; grey: n omega_core "
                 "(first epoch)" % an["name"], fontsize=10)
    fig.tight_layout()
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    os.replace(tmp, path)


def _write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main(argv):
    name = argv[0]
    box2 = argv[argv.index("--box2") + 1] if "--box2" in argv else None
    cmp_ = argv[argv.index("--compare") + 1].split(",") if "--compare" in argv else []
    an = analyse(name)
    b2 = analyse(box2) if box2 else None
    an["verdict"] = verdict(an, b2)
    if box2:
        an["box2_name"] = box2
    if cmp_:
        an["compare"] = {c: compare(name, c) for c in cmp_}
    _write(os.path.join(OUT, name + "_analysis.json"), an)
    plot(an, os.path.join(OUT, name + "_spectra.png"))
    print(json.dumps(headline(an), indent=1))


def headline(an):
    h = dict(name=an["name"], omega_th=an["omega_th"], verdict=an["verdict"]["overall"], epochs=[])
    for ep, v in zip(an["epochs"], an["verdict"]["per_epoch"]):
        e = dict(t=[ep["t0"], ep["t1"]], omega_core=ep["omega_core"], drift=ep["drift"],
                 loss_rate=ep["loss_rate_Ein_3hw"], absorbed_rate=ep["absorbed_rate"],
                 verdict=v["verdict"], c1=v["c1"], c2=v["c2"], c3=v["c3"],
                 core_lines=[(round(d["omega"], 5), d["label"], "%.3g" % d["amp"]) for d in ep["core_lines"][:8]],
                 probes=[])
        for p in ep["probes"]:
            if p["r"] < R_FAR:
                continue
            e["probes"].append(dict(
                r=p["r"], lines=[(round(d["omega"], 4), d["label"], "%.3g" % d["amp"]) for d in p["lines"][:6]],
                sub_power="%.3g" % p["sub_power"], leak="%.3g" % p["leak_floor"], ev="%.3g" % p["ev_floor"],
                sub_rel="%.3g" % p["sub_rel"], flux_above_m1="%.4g" % p["flux_above_m1"],
                flux_above_m2="%.4g" % p["flux_above_m2"], flux_total="%.4g" % p["flux_total_m1"]))
        h["epochs"].append(e)
    return h


if __name__ == "__main__":
    main(sys.argv[1:])
