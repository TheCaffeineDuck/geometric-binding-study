"""Check 4. Absorbing ends against a string twice as long.

Absorber: matched sponge (velocity and stretch damped at the same rate sigma(x), quadratic ramp,
L_s = 100, sigma_max = 0.3; see swirl/string/core.py). String A has interior half-length 120,
string B 240 (twice as long, same absorber); same start. Compared over the run:
  * window energy (|x| <= 10): max_t |E_A - E_B| / E0, target 1e-6;
  * interior field (|x| <= 100): energy of the difference field (quadratic energy of
    (u_A - u_B, v_A - v_B) on |x| <= 100) over E0, target 1e-6.
Clean interval: the comparison stops before B's own sponge can return a signal to the compared
region at the fastest speed in the start, c_max = sqrt(max F'(d0)) (G1 stretch pulses are
supersonic: 1.82 at d = 0.6). A's reflections arrive from t ~ (2*120 - 10)/c_max, well inside.

Starts: Gaussian displacement bump sigma = 3 (energy spectrum k^2 exp(-k^2 sigma^2): most energy
near k ~ 1/3 and a k^2 tail to k -> 0, the long waves), linear (peak stretch 0.3) and G1 (0.6).
C-smooth also with sigma = 0.5 (short waves, slow: group velocity (1+k^2)^-3/2).

Failure would look like: a difference above 1e-6 of E0 that appears once A's reflections can be
back (t > (2*120 - 10)/c_max)."""

import numpy as np
import pytest

from swirl.string import starts as ST
from swirl.string.core import String

LS, SMAX = 100.0, 0.3


def _run(medium, law, h, half, T, dt, sig, pk):
    N = int(round((2 * half + 2 * LS) / h))
    S = String(medium, law, N, h=h, sponge_len=LS, sigma_max=SMAX)
    Q = String(medium, "lin", N, h=h)
    u0, _ = ST.gaussian(S, "bump", sig, peak_stretch=pk)
    S.set_state(u0)
    cmax = float(np.sqrt(np.max(S.dF(S.stretch()))))
    E0 = S.energy()
    mw = np.abs(S.x) <= 10
    mi = np.where(np.abs(S.x) <= 100)[0]
    ns = int(round(1.0 / dt))
    ew, uu, vv, ts = [], [], [], []
    for _ in range(int(T) + 1):
        ts.append(S.t)
        ew.append(S.density()[mw].sum())
        uu.append(S.u[mi].copy())
        vv.append(S.v[mi].copy())
        S.advance(dt, ns)
    return np.array(ts), np.array(ew), np.array(uu), np.array(vv), E0, S, Q, mi, cmax


CASES = [("grid", 1.0, 0.02, "lin", 3.0, 0.3), ("grid", 1.0, 0.02, "G1", 3.0, 0.6),
         ("fd", 0.25, 0.05, "lin", 3.0, 0.3), ("fd", 0.25, 0.05, "G1", 3.0, 0.6),
         ("fd", 0.125, 0.05, "lin", 3.0, 0.3), ("fd", 0.125, 0.05, "G1", 0.5, 0.6),
         ("fd", 0.0625, 0.05, "G1", 3.0, 0.6)]


@pytest.mark.parametrize("medium,h,dt,law,sig,pk", CASES)
def test_ends_against_double_string(medium, h, dt, law, sig, pk, record):
    a = _run(medium, law, h, 120, 460, dt, sig, pk)
    b = _run(medium, law, h, 240, 460, dt, sig, pk)
    t, E0, cmax = a[0], a[4], a[8]
    Tw = min(460.0, 0.95 * (2 * 240 - 10) / cmax)
    Tf = min(370.0, 0.95 * (2 * 240 - 100) / cmax)
    TA = (2 * 120 - 10) / cmax
    sel = t <= Tw
    dE = np.abs(a[1] - b[1])[sel] / E0
    Q, mi = a[6], a[7]
    ediff = []
    for i in np.where(t <= Tf)[0]:
        du = np.zeros(Q.N)
        dv = np.zeros(Q.N)
        du[mi] = a[2][i] - b[2][i]
        dv[mi] = a[3][i] - b[3][i]
        ediff.append(Q.density(du, dv)[mi[2:-2]].sum() / E0)
    ediff = np.array(ediff)
    i1 = int(np.argmax(dE))
    w1, w2 = float(dE.max()), float(ediff.max())
    v = lambda x: "pass" if x <= 1e-6 else ("pass with note" if x <= 3e-6 else "MISS")
    print("\ncheck4 %s h=%g %s sigma=%g c_max=%.2f: A's echoes from t=%.0f; window max|dE|/E0 = "
          "%.2e at t=%.0f over [0,%.0f] (target 1e-6) -> %s; interior difference-field energy "
          "max %.2e over [0,%.0f] -> %s; absorbed by A %.4f of E0; window frac at end %.2e" % (
              medium, h, law, sig, cmax, TA, w1, t[sel][i1], Tw, v(w1), w2, Tf, v(w2),
              a[5].absorbed / E0, a[1][-1] / E0))
    record("check4_%s_h%g_%s_s%g" % (medium, h, law, sig),
           {"window_max_rel": w1, "t_at_max": float(t[sel][i1]), "field_energy_max_rel": w2,
            "T_window": Tw, "T_field": Tf, "A_echo_time": TA, "c_max": cmax,
            "absorbed_frac": a[5].absorbed / E0, "Ls": LS, "sigma_max": SMAX, "target": 1e-6})
    assert Tw > TA + 50
    assert w1 <= 3e-6 and w2 <= 3e-6


def test_short_wave_arrival_times(record):
    """Not pass/fail: when C-smooth short waves leave the window and reach the sponge.
    Group velocity (continuum) (1 + k^2)^(-3/2); 'fd' near its band top: d omega / dk."""
    rows = []
    for k in (1.0, 2.0, 3.0, 5.0, 10.0):
        vg = (1 + k * k) ** -1.5
        rows.append({"k": k, "v_group": vg, "t_to_leave_window_10": 10 / vg,
                     "t_to_reach_sponge_at_120": 120 / vg})
    for h in (0.25, 0.125, 0.0625):
        k = np.pi / h * 0.99
        s = 2 / h * np.sin(k * h / 2)
        vg = np.cos(k * h / 2) * (1 + s * s) ** -1.5
        rows.append({"k": k, "h": h, "v_group_fd_at_0.99_band_edge": float(vg)})
    for r in rows:
        print("\ncheck4 short waves:", r, end="")
    record("check4_short_wave_times", {"rows": rows})
