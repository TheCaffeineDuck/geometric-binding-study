"""Post-processing on saved time series (Block 2.1 yardstick)."""

import numpy as np


def t90_tdeath(t, frac, Ewin):
    """T_90: first record time with frac < 0.90. T_death: first record time after t = 50 with
    window energy below half its value at t = 50 (interpolated between record mid-times).
    None if never."""
    t = np.asarray(t)
    frac = np.asarray(frac)
    Ewin = np.asarray(Ewin)
    i90 = np.where(frac < 0.90)[0]
    T90 = float(t[i90[0]]) if i90.size else None
    E50 = float(np.interp(50.0, t, Ewin))
    idd = np.where((Ewin < 0.5 * E50) & (t > 50.0))[0]
    Td = float(t[idd[0]]) if idd.size else None
    return T90, Td


def zero_cross_times(t, q):
    t = np.asarray(t)
    q = np.asarray(q)
    i = np.where((q[:-1] < 0) & (q[1:] >= 0))[0]
    return t[i] - q[i] * (t[i + 1] - t[i]) / (q[i + 1] - q[i])


def rate_zero_crossings(t, q):
    tc = zero_cross_times(t, q)
    if tc.size < 2:
        return None
    return float(2 * np.pi * (tc.size - 1) / (tc[-1] - tc[0]))


def spectrum(t, q):
    """One-sided amplitude spectrum with a Hann window. Returns omega, |Q|."""
    q = np.asarray(q, dtype=float)
    n = q.size
    dt = t[1] - t[0]
    w = np.hanning(n)
    Q = np.abs(np.fft.rfft((q - q.mean()) * w))
    om = 2 * np.pi * np.fft.rfftfreq(n, d=dt)
    return om, Q


def rate_spectrum(t, q):
    """Spectral peak with parabolic interpolation in log amplitude; returns (rate, resolution)."""
    om, Q = spectrum(t, q)
    i = int(np.argmax(Q[1:])) + 1
    res = float(om[1] - om[0])
    if 1 <= i < len(Q) - 1 and Q[i - 1] > 0 and Q[i + 1] > 0:
        a, b, c = np.log(Q[i - 1]), np.log(Q[i]), np.log(Q[i + 1])
        p = 0.5 * (a - c) / (a - 2 * b + c)
        return float(om[i] + p * res), res
    return float(om[i]), res


def jump_stats(xw, d, h, medium):
    """Block 4 jump measure on the window's stretch d (at the record's peak sample): the largest
    jump between neighbouring cells, its position, the total variation of the stretch, and the
    number of neighbouring-cell steps the largest jump spans (contiguous steps of the same sign and
    at least a quarter of the largest)."""
    dd = np.diff(d)
    if dd.size == 0:
        return {}
    i = int(np.argmax(np.abs(dd)))
    m = abs(dd[i])
    sgn = np.sign(dd[i])
    lo = i
    while lo > 0 and np.sign(dd[lo - 1]) == sgn and abs(dd[lo - 1]) >= 0.25 * m:
        lo -= 1
    hi = i
    while hi < dd.size - 1 and np.sign(dd[hi + 1]) == sgn and abs(dd[hi + 1]) >= 0.25 * m:
        hi += 1
    xs = xw + (0.5 * h if medium == "fd" else 0.0)      # where d lives (fd: bond midpoints)
    return {"jump_max": float(m), "jump_x": float(0.5 * (xs[i] + xs[i + 1])),
            "stretch_total_variation": float(np.sum(np.abs(dd))), "jump_cells": int(hi - lo + 1)}


def tail_kappa(dist, amp, rmin, omega, medium, rmax=np.inf):
    """Decay rate of the component at the rate (Block 4 item 5): fit ln amp against |dist| on
    rmin < |dist| < rmax (the window: outside it radiation dominates) where 1e-9 max < amp < 3e-2 max. Returns (measured, O1 prediction)."""
    if medium == "grid":
        o1 = float(2 * np.arccosh(omega / 2)) if omega > 2 else None
    else:
        o1 = float(omega / np.sqrt(omega ** 2 - 1)) if omega > 1 else None
    dist = np.abs(np.asarray(dist))
    amp = np.asarray(amp)
    if amp.size == 0 or amp.max() <= 0:
        return None, o1
    sel = (dist > rmin) & (dist < rmax) & (amp > 1e-9 * amp.max()) & (amp < 3e-2 * amp.max())
    if sel.sum() < 6:
        return None, o1
    p = np.polyfit(dist[sel], np.log(amp[sel]), 1)
    return float(-p[0]), o1


def wavenumber_moments(S, d, mask):
    """First-moment mean and rms wavenumber of the stretch on the window (Hann-tapered power)."""
    w = d[mask] * np.hanning(int(mask.sum()))
    W = np.abs(np.fft.rfft(w)) ** 2
    k = 2 * np.pi * np.fft.rfftfreq(w.size, d=S.h)
    s = max(np.sum(W), 1e-300)
    return float(np.sum(k * W) / s), float(np.sqrt(np.sum(k * k * W) / s))


def cycles(t_end, rates, rate_fill):
    """Cycle count at each record end: the integral of the record rates over time. Records without a
    rate take the nearest measured one (rate_fill if none was measured)."""
    r = np.asarray(rates, dtype=float).copy()
    ok = np.isfinite(r)
    if not ok.any():
        r[:] = rate_fill
    else:
        idx = np.arange(r.size)
        r = np.interp(idx, idx[ok], r[ok])
    dt = np.diff(np.concatenate([[0.0], np.asarray(t_end, dtype=float)]))
    return np.cumsum(r * dt) / (2 * np.pi)


def at_time(T, t_end, cyc):
    """Cycle count at time T (None passes through)."""
    if T is None:
        return None
    return float(np.interp(T, np.concatenate([[0.0], t_end]), np.concatenate([[0.0], cyc])))
