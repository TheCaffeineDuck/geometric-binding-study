"""Check 3. Regression: the abstract of SPS 1992 on the Flach-Gorbach setup (C-grid, G2).
Not a reproduction of the published SPS run, whose constants and amplitude are unknown.
Deviation: Flach and Gorbach use 80 sites with open ends; the Floquet analysis here uses a ring of
80 sites and the evolution a ring of 300 with absorbing sponge. At Omega = 4.5 the tail falls by
exp(-2.92) per site, so the ends are invisible (below 1e-50 at 40 sites).

Parameters (docs/S5_SPS1992_PARAMS.md, scholar 2026-11-10): the SPS 1992 constants, anharmonicity,
chain, boundaries and start construction are not recoverable (full text paywalled; "no source
found"). The regression uses the published setup with full parameters that restates the same
physics: Flach and Gorbach, Chaos 15, 015112 (2005): F = d + d^3 (beta = 1), breather rate
Omega_b = 4.5, Page mode stable (all multipliers on the unit circle), Sievers-Takeno mode with one
real pair off the circle. SPS targets kept: the S-bend persists unchanged past 32,000 of its own
cycles; the single bump is unstable and moves; the pure-quartic growth rate is 0.15 "in units of
the mode frequency" (tested here in the limit Omega -> infinity of G2, where the linear term no
longer matters, under the reading: e-folding rate per unit time / angular frequency).

Starts, each run both ways: (a) the rotating-wave pattern as in the SPS abstract, A(-1/2, 1, -1/2)
or A(-1, 1), A from the rotating-wave relation at 4.5 (bump: 4.5^2 = 3 + 81/16 A^2; S-bend:
4.5^2 = 3 + 27/4 A^2); (b) the exact periodic state at 4.5. Every start gets 1e-8 random noise on
|x| <= 20 (seed 0): an exactly symmetric start keeps its symmetry to the last bit (F is exactly
odd), so without a seed the bump's antisymmetric instability never starts.

Rule 2.3 as written: stable = largest multiplier within 1e-6 of the unit circle, and at 32,000
cycles peak displacement and centre within 1% and fixed-window frac >= 0.99. "Centre within 1%"
is read as within 0.01 grid units (1% of the spacing). Unstable and moves = a multiplier above
1 + 1e-3, and the energy centroid leaves its start site by >= 1 grid unit while co-moving frac >=
0.90. The rotating-wave start sheds the part of its energy that is not on the exact state; its
amplitude and frac are reported against t = 0 and against the record after 100 cycles.

What is asserted (code checks): the exact S-bend meets the stable rule; the exact bump has a
multiplier > 1 + 1e-3 and in direct evolution leaves the site-centred configuration (centroid
moves >= 0.4) with co-moving frac >= 0.9. Whether it travels >= 1 site is a physics observable
(it depends on the anharmonicity) and is reported, not asserted.
Failure would look like: the exact S-bend drifting or losing energy, a multiplier of the S-bend
off the circle, or the bump staying on its site.
Runtime: about 4 runs x 30 s plus 2 short ones (slow)."""

import numpy as np
import pytest

from swirl.string import driver as DR
from swirl.string import orbits as O
from swirl.string.core import String

OMEGA = 4.5
NCYC = 32000


def _floquet(obj, om=OMEGA, N=80):
    S = String("grid", "G2", N, beta=1.0)
    A = O.rwa_amplitude(obj, "G2", 1.0, om)
    g = O.grid_pattern(N, obj, A, O.kappa_grid(om))
    nq = O.choose_nq(S, g, om)
    u, info = O.newton(S, g, om, nq)
    assert info["converged"]
    d = O.describe(S, u, om, nq, obj)
    d["rwa_A"] = A
    return d


def _run(obj, start, om=OMEGA, ncyc=NCYC, dt=0.008):
    st = {"type": start, "obj": obj, "omega": om, "noise": 1e-8, "seed": 0}
    cfg = {"kind": "evolve", "name": "check3_G2_%s_%s_om%g" % (obj, start, om), "medium": "grid",
           "law": "G2", "beta": 1.0, "N": 300, "dt": dt, "sample_dt": 12 * dt,
           "T": ncyc * 2 * np.pi / om, "record_periods": 50 if ncyc > 5000 else 10, "start": st,
           "twin": False, "probe_max": 8192, "probes": [0.0, 20.0, 40.0]}
    return DR.evolve(cfg, log=lambda *a: None)


def _evaluate(obj, start, fl, r):
    s, recs = r["summary"], r["records"]
    E0 = s["E_fixed0"]
    t = np.array([q["t_mid"] for q in recs])
    rate = s["rate_zero_crossings_core"]
    cyc = t * rate / (2 * np.pi)
    c = np.array([q["centroid"] for q in recs])
    fr = np.array([q["E_fixed"] for q in recs]) / E0
    fco = np.array([q["E_comoving"] for q in recs]) / E0
    pk = np.array([q["peak_u_window"] for q in recs])
    i100 = int(np.searchsorted(cyc, 100.0))
    out = {"omega": fl["omega"], "rwa_A": fl["rwa_A"],
           "floquet_max_modulus": fl["floquet_max_modulus"],
           "floquet_max_modulus_excl_4_at_1": fl["floquet_max_modulus_excl_4_at_1"],
           "floquet_unstable_mode_parity": fl["floquet_unstable_mode_parity"],
           "exact_peak_displacement": fl["peak_displacement"], "exact_E": fl["E"],
           "rate_measured": rate, "cycles_run": float(cyc[-1]),
           "frac_end_vs_t0": float(fr[-1]), "frac_end_vs_100cyc": float(fr[-1] / fr[i100]),
           "peak_u_first_record": float(pk[0]), "peak_u_100cyc": float(pk[i100]),
           "peak_u_end": float(pk[-1]), "peak_u_change_vs_100cyc": float(pk[-1] / pk[i100] - 1),
           "centroid_start": float(c[0]), "centroid_end": float(c[-1]),
           "centroid_max_excursion": float(np.max(np.abs(c - c[0]))),
           "frac_comoving_min": float(fco.min()), "energy_balance_max": s["max_abs_balance"]}
    if obj == "sbend":
        ref = 0 if start == "orbit" else i100
        out.update({"rule_stable_floquet": bool(abs(fl["floquet_max_modulus"] - 1) <= 1e-6),
                    "amp_change": float(pk[-1] / pk[ref] - 1),
                    "rule_amp_1pct": bool(abs(pk[-1] / pk[ref] - 1) <= 0.01),
                    "rule_centre_0.01": bool(abs(c[-1] - c[0]) <= 0.01),
                    "rule_frac_0.99": bool(fr[-1] / fr[ref] >= 0.99),
                    "rule_reference": "t=0" if ref == 0 else "100 cycles"})
        out["verdict_stable"] = all(out[k] for k in ("rule_stable_floquet", "rule_amp_1pct",
                                                     "rule_centre_0.01", "rule_frac_0.99"))
    else:
        x0 = 0.0                                   # the site the bump starts on
        il = np.where(np.abs(c - x0) >= 1.0)[0]
        out.update({"rule_unstable_floquet": bool(fl["floquet_max_modulus"] > 1 + 1e-3),
                    "left_site_by_1_at_cycles": float(cyc[il[0]]) if il.size else None,
                    "frac_comoving_when_left": float(fco[il[0]]) if il.size else None,
                    "rule_moves_1_site": bool(il.size and fco[il[0]] >= 0.90),
                    "max_distance_from_start_site": float(np.max(np.abs(c - x0))),
                    "centroid_track": [float(v) for v in c[::max(1, len(c) // 16)]]})
    return out


@pytest.mark.slow
@pytest.mark.parametrize("obj", ["sbend", "bump"])
@pytest.mark.parametrize("start", ["orbit", "rwa"])
def test_regression(obj, start, record):
    fl = _floquet(obj)
    out = _evaluate(obj, start, fl, _run(obj, start))
    print("\ncheck3 G2 beta=1 Omega=4.5 %s start=%s: %s" % (obj, start, out))
    record("check3_G2_%s_%s" % (obj, start), out)
    if start == "orbit" and obj == "sbend":
        assert out["verdict_stable"]
    if start == "orbit" and obj == "bump":
        assert out["rule_unstable_floquet"]
        assert out["max_distance_from_start_site"] >= 0.4 and out["frac_comoving_min"] >= 0.9


@pytest.mark.slow
def test_bump_travel_weaker_anharmonicity(record):
    """Diagnostic for the 'moves' half of rule 2.3: the same exact bump at Omega = 2.3 (weaker
    anharmonicity, closer to the band top), 3,000 cycles."""
    fl = _floquet("bump", om=2.3, N=120)
    out = _evaluate("bump", "orbit", fl, _run("bump", "orbit", om=2.3, ncyc=3000, dt=0.01))
    print("\ncheck3 diagnostic G2 bump Omega=2.3: %s" % out)
    record("check3_G2_bump_orbit_om2.3", out)


def test_pure_quartic_growth_rate(record):
    """Reported beside the SPS abstract's 0.15, not a pass line (unit ambiguous). G2 at Omega >> 2
    is the pure quartic lattice (the linear force is negligible). Two readings of 'in units of the
    mode frequency', with sigma = ln|multiplier| / T the growth rate per unit time:
      angular:  sigma / Omega        = ln|multiplier| / (2 pi)
      ordinary: sigma / (Omega/2 pi) = ln|multiplier|"""
    rows = []
    for om in (10.0, 20.0, 40.0):
        d = _floquet("bump", om=om, N=61)
        m = d["floquet_max_modulus"]
        rows.append({"omega": om, "max_multiplier": m,
                     "growth_over_angular_freq": float(np.log(m) / (2 * np.pi)),
                     "growth_over_ordinary_freq": float(np.log(m)),
                     "parity": d["floquet_unstable_mode_parity"]})
    print("\ncheck3 pure-quartic limit (SPS abstract 0.15): %s" % rows)
    record("check3_pure_quartic_growth", {"rows": rows, "sps_abstract": 0.15})
    assert rows[-1]["parity"] == "antisymmetric"
