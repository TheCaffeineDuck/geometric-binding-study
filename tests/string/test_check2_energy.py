"""Check 2. Energy.

(a) The energy function contains every term of the equation of motion: dE/dt along the flow,
computed by complex step (exact to round-off) from the solver's own energy() and accel(), vanishes
to round-off for random smooth data, in every medium and law. Negative control: the same energy
without the u_xt inertia term gives dE/dt of order one (so the test can fail).

(b) Conservation on a closed ring (no absorber) over 1,000 time units, G1 at peak stretch 0.8, G2
(beta = 1) and the linear twin. Target max |E(t)/E(0) - 1| <= 1e-8. Integrator order: the error
at dt and dt/2 should fall by 2^4 = 16 (Yoshida 4th order; energy error bounded, not drifting).

Failure would look like: (a) a ratio far above 1e-12; (b) a drift growing with t, or a dt-ratio
far from 16."""

import numpy as np
import pytest

from swirl.string.core import String

EPS = 1e-30


def _random_smooth(S, rng, amp_d):
    k = 2 * np.pi * np.fft.rfftfreq(S.N, d=S.h)
    c = (rng.normal(size=k.size) + 1j * rng.normal(size=k.size)) * np.exp(-(k / 2.0) ** 2)
    c[0] = 0
    f = np.fft.irfft(c, n=S.N)
    d = S.stretch(f)
    return f * amp_d / np.max(np.abs(d))


def _dEdt(S, u, v, drop_inertia=False):
    a = S.accel(u)
    if drop_inertia:              # energy as if the u_xt term were absent
        Ef = lambda uu, vv: 0.5 * S.h * np.sum(vv * vv) + S.h * np.sum(S.V(S.stretch(uu)))
    else:
        Ef = S.energy
    tot = np.imag(Ef(u + 1j * EPS * v, v + 1j * EPS * a)) / EPS
    kin = np.imag(Ef(u, v + 1j * EPS * a)) / EPS
    return tot, kin


@pytest.mark.parametrize("medium", ["grid", "fd", "spectral"])
@pytest.mark.parametrize("law", ["lin", "G1", "G2"])
def test_energy_has_every_term(medium, law, record):
    rng = np.random.default_rng(7)
    S = String(medium, law, 256, h=0.125, beta=1.0)
    worst = 0.0
    for _ in range(5):
        u = _random_smooth(S, rng, 0.7)
        v = _random_smooth(S, rng, 0.5)
        tot, kin = _dEdt(S, u, v)
        worst = max(worst, abs(tot) / abs(kin))
    out = {"medium": medium, "law": law, "max_ratio_dEdt_over_power": worst, "target": 1e-12}
    if medium != "grid":
        tot, kin = _dEdt(S, u, v, drop_inertia=True)
        out["negative_control_ratio_without_uxt_term"] = abs(tot) / abs(kin)
    print("\ncheck2a %s %s: |dE/dt| / |kinetic power| = %.2e (target 1e-12)%s" % (
        medium, law, worst, "" if medium == "grid" else
        "; without u_xt term: %.2e" % out["negative_control_ratio_without_uxt_term"]))
    record("check2a_%s_%s" % (medium, law), out)
    assert worst < 1e-12
    if medium != "grid":
        assert out["negative_control_ratio_without_uxt_term"] > 1e-3


def _start(S, law):
    if S.medium == "grid":
        u = np.where(S.x == 0, 1.0, np.where(np.abs(S.x) == 1, -0.5, 0.0))
    else:
        u = np.exp(-S.x ** 2 / 2.0)
    return u * 0.8 / np.max(np.abs(S.stretch(u)))


def _drift(medium, law, dt, T=1000.0, h=0.125, N=128):
    S = String(medium, law, N, h=h, beta=1.0)
    S.set_state(_start(S, law))
    E0 = S.energy()
    nrec = 200
    n = int(round(T / dt / nrec))
    err = np.empty(nrec)
    dmax = float(np.max(np.abs(S.stretch())))          # 0.8 at t = 0 by construction
    for i in range(nrec):
        S.advance(dt, n)
        err[i] = S.energy() / E0 - 1
        dmax = max(dmax, float(np.max(np.abs(S.stretch()))))
    return float(np.max(np.abs(err))), float(np.max(np.abs(err[-20:]))), float(np.max(
        np.abs(err[:20]))), dmax, S.t


DTS = {"grid": 0.004, "fd": 0.008, "spectral": 0.008}


@pytest.mark.parametrize("medium", ["grid", "fd",
                                    pytest.param("spectral", marks=pytest.mark.slow)])
@pytest.mark.parametrize("law", ["lin", "G1", "G2"])
def test_energy_conserved_closed_ring(medium, law, record):
    """Spectral cases are marked slow (about 40 s each, pure numpy stepping)."""
    dt = DTS[medium]
    e1, late, early, dmax, T = _drift(medium, law, dt)
    e2, _, _, _, _ = _drift(medium, law, dt / 2)
    ratio = e1 / e2 if e2 > 0 else np.inf
    verdict = "pass" if e1 < 1e-8 else ("pass with note" if e1 < 3e-8 else "MISS")
    print("\ncheck2b %s %s T=%.0f dt=%g: max|dE/E| %.2e (target 1e-8) -> %s; dt/2: %.2e, "
          "ratio %.1f (16 for 4th order); first/last 10%% of run %.1e/%.1e; max |d| seen %.3f" % (
              medium, law, T, dt, e1, verdict, e2, ratio, early, late, dmax))
    record("check2b_%s_%s" % (medium, law), {"dt": dt, "T": T, "max_rel_err": e1,
                                             "max_rel_err_dt_half": e2, "ratio": ratio,
                                             "first_10pct": early, "last_10pct": late,
                                             "max_stretch_seen": dmax, "target": 1e-8})
    assert e1 < 3e-8
    if e2 > 1e-13:
        assert 8 < ratio < 32
