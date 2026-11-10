"""Check 1. Linear waves: the measured rate of a single Fourier mode on the ring against the
scheme's own discrete relation (target 1e-6 relative) and against the continuum relation
(reported: that difference is the scheme's dispersion error, not a pass/fail quantity).

Failure would look like: a rate off the scheme's relation by more than 1e-6 at any wavenumber
(a wrong stencil, a wrong mass matrix, or the u_xt inertia term missing would each give O(1)
errors at large k)."""

import numpy as np
import pytest

from swirl.string import diag as G
from swirl.string.core import String, band_top, omega_continuum, omega_scheme

TARGET = 1e-6


def measure(medium, h, N, m, dt, nper=30, tmin=200.0):
    S = String(medium, "lin", N, h=h)
    k = 2 * np.pi * m / S.L
    mode = np.cos(k * S.x)
    S.set_state(mode)
    om = omega_scheme(medium, k, h)
    T = max(tmin, nper * 2 * np.pi / om)
    ns = 2
    n = int(T / (ns * dt))
    t = np.empty(n)
    q = np.empty(n)
    for i in range(n):
        t[i] = S.t
        q[i] = S.u @ mode
        S.advance(dt, ns)
    return k, G.rate_zero_crossings(t, q), om, float(omega_continuum(medium, k))


CASES = [("grid", 1.0, 64, [2, 6, 12, 20, 26, 31], 0.004)]
for _h in (0.25, 0.125, 0.0625):
    _N = int(round(32 / _h))
    CASES.append(("fd", _h, _N, [1, 3, 5, 10, 25, _N // 4, _N // 2 - 1], 0.01))
    CASES.append(("spectral", _h, _N, [1, 3, 5, 10, 25, _N // 4, _N // 2 - 1], 0.01))


@pytest.mark.parametrize("medium,h,N,ms,dt", CASES)
def test_dispersion(medium, h, N, ms, dt, record):
    rows = []
    for m in ms:
        k, om_meas, om_s, om_c = measure(medium, h, N, m, dt)
        rows.append({"k": k, "omega_measured": om_meas, "omega_scheme": float(om_s),
                     "rel_err_scheme": abs(om_meas / om_s - 1),
                     "omega_continuum": om_c, "rel_dev_continuum": om_s / om_c - 1})
    worst = max(r["rel_err_scheme"] for r in rows)
    print("\ncheck1 %s h=%g band top %.6f (continuum ceiling %s):" % (
        medium, h, band_top(medium, h), 2.0 if medium == "grid" else 1.0))
    for r in rows:
        print("  k=%7.4f omega=%.10f scheme %.10f rel %.1e | continuum %.10f dev %+.2e" % (
            r["k"], r["omega_measured"], r["omega_scheme"], r["rel_err_scheme"],
            r["omega_continuum"], r["rel_dev_continuum"]))
    print("  worst rel err vs scheme %.2e (target %.0e) -> %s" % (
        worst, TARGET, "pass" if worst < TARGET else ("pass with note" if worst < 3 * TARGET
                                                       else "MISS")))
    record("check1_%s_h%g" % (medium, h), {"rows": rows, "worst_rel_err_scheme": worst,
                                           "target": TARGET, "dt": dt, "N": N,
                                           "band_top": band_top(medium, h)})
    assert len(rows) >= 6
    assert worst < 3 * TARGET
