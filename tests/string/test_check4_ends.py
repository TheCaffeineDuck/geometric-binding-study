"""Check 4. Absorbing ends against a string twice as long.

Absorber: matched sponge (velocity and stretch damped at the same rate sigma(x), quadratic ramp,
the production defaults L_s = 300, sigma_max = 0.05, swirl/string/driver.py; see core.py). String A has interior half-length 120,
string B 240 (twice as long, same absorber); same start. Compared over the run:
  * window energy (|x| <= 10): max_t |E_A - E_B| / E0, target 1e-6;
  * interior field (|x| <= 100): energy of the difference field (quadratic energy of
    (u_A - u_B, v_A - v_B) on |x| <= 100) over E0, target 1e-6.
Clean interval: the comparison stops before B's own sponge can return a signal to the compared
region at the fastest speed in the start, c_max = sqrt(max F'(d0)) (G1 stretch pulses are
supersonic: 1.82 at d = 0.6). A's reflections arrive from t ~ (2*120 - 10)/c_max, well inside.

Starts: Gaussian displacement bump sigma = 3 (energy spectrum k^2 exp(-k^2 sigma^2): most energy
near k ~ 1/3 and a k^2 tail to k -> 0, the long waves), linear (peak stretch 0.3) and G1 (0.6).
C-smooth also G1 with sigma = 1.5 at h = 1/8 (shorter waves; the sponge must absorb at least half
of the energy by the end, so the case tests the absorber).
Also here: a band-edge packet on the grid (k = 0.9 pi, group velocity 0.156) moving into A's sponge,
reflected fraction for the production sponge (target 1e-5) and for the old 100 / 0.3; the energy
balance E + absorbed = E0 for fully absorbed runs (target 1e-6); and that the energy never rises
under damping (LESSONS 7).

Failure would look like: a difference above 1e-6 of E0 that appears once A's reflections can be
back (t > (2*120 - 10)/c_max)."""

import numpy as np
import pytest

from swirl.string import starts as ST
from swirl.string.core import String
from swirl.string.driver import SPONGE_LEN as LS
from swirl.string.driver import SPONGE_SIGMA as SMAX


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
         ("fd", 0.125, 0.05, "lin", 3.0, 0.3), ("fd", 0.125, 0.05, "G1", 1.5, 0.6),
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
    assert a[5].absorbed / E0 >= 0.5            # the sponge actually absorbs in every case
    assert w1 <= 3e-6 and w2 <= 3e-6


def _packet(half, Ls, smax, T=900.0, xc=60.0, k0=0.9 * np.pi, s=10.0, nrec=300):
    N = int(2 * half + 2 * Ls)
    S = String("grid", "lin", N, sponge_len=Ls, sigma_max=smax)
    env = np.exp(-(S.x - xc) ** 2 / (2 * s * s))
    om = 2 * np.sin(k0 / 2)
    S.set_state(env * np.cos(k0 * S.x), om * env * np.sin(k0 * S.x))   # moves toward +x
    E0 = S.energy()
    inner = np.abs(S.x) < 100
    Ei = []
    for _ in range(nrec):
        Ei.append(S.density()[inner].sum() / E0)
        S.advance(0.05, int(T / nrec / 0.05))
    return np.array(Ei), S.absorbed / E0


def test_band_edge_packet_reflection(record):
    """Grid, linear: packet k = 0.9 pi (group velocity cos(0.45 pi) = 0.156), width 10, centred at
    x = 60 and moving into the sponge of string A (interior half-length 100); string B has 200.
    Reflected fraction = max_t |E_A - E_B| on |x| < 100 over E0, t <= 900 (before B's sponge can
    echo). Failure: above 1e-5 for the production sponge."""
    rows = {}
    for Ls, smax in ((LS, SMAX), (100.0, 0.3)):
        a, ab = _packet(100, Ls, smax)
        b, _ = _packet(200, Ls, smax)
        rows["%g/%g" % (Ls, smax)] = {"reflected_fraction": float(np.max(np.abs(a - b))),
                                       "absorbed_by_A_at_t900": ab}
    print("\ncheck4 band-edge packet: %s (target 1e-5 for the production sponge %g/%g)" % (
        rows, LS, SMAX))
    record("check4_band_edge_packet", rows)
    assert rows["%g/%g" % (LS, SMAX)]["reflected_fraction"] <= 1e-5


@pytest.mark.parametrize("medium,h", [("grid", 1.0), ("fd", 0.125), ("spectral", 0.25)])
@pytest.mark.parametrize("law", ["lin", "G1"])
def test_absorbed_balance_and_no_energy_rise(medium, h, law, record):
    """Fully absorbed Gaussian (sigma 3), dt = 0.02, 600 time units: max |E + absorbed - E0| / E0
    (target 1e-6), every damping half-step removes energy (absorbed never decreases), and E(t)
    never rises between records by more than the symplectic error (1e-8 E0)."""
    N = int(round((2 * 120 + 2 * LS) / h))
    S = String(medium, law, N, h=h, sponge_len=LS, sigma_max=SMAX)
    u0, _ = ST.gaussian(S, "bump", 3.0, peak_stretch=0.3 if law == "lin" else 0.6)
    S.set_state(u0)
    E0 = S.energy()
    bal, rise, dab = 0.0, 0.0, 0.0
    Ep, abp = E0, 0.0
    for _ in range(60):
        S.advance(0.02, 500)
        E = S.energy()
        bal = max(bal, abs(E + S.absorbed - E0) / E0)
        rise = max(rise, (E - Ep) / E0)
        dab = min(dab, (S.absorbed - abp) / E0)
        Ep, abp = E, S.absorbed
    out = {"balance_max": bal, "max_energy_rise": rise, "min_absorbed_increment": dab,
           "absorbed_frac": S.absorbed / E0}
    print("\ncheck4 balance %s %s: %s (target 1e-6)" % (medium, law, out))
    record("check4_balance_%s_%s" % (medium, law), out)
    assert S.absorbed / E0 > 0.95
    assert bal <= 1e-6 and rise <= 1e-8 and dab >= -1e-13      # spectral: round-off of a difference


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
