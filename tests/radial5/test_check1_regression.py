"""Check 1. Regression: equation O, single bump, beta = 4/3, omega 0.909149280617113.

Session 1 reference numbers (docs/SESSION_01_REPORT.md section 6 and
outputs/step2/diagnostics/primary_lifetime.json, radial, same grid):
  start omega 0.909149280617113 (0.9091), Phi(0) 6.89, hw 3.344038183621855,
  E0 = 2843.7718963433854 (np.gradient + trapezoid diagnostic, O(dr^2), F-003),
  rate median over t = 975..1025: 0.9096746654018131, frac(1000) = 0.9941458018284945,
  nonlinear share t = 0 (signed) -0.3156.
Failure would look like: any difference from radial.py step for step above round-off, a
rate off by more than 1e-4, or E at t = 0 off the analytic start energy by more than 1e-6.
"""

import numpy as np

from swirl.family.profile import profile_curve, solve_profile
from swirl.family.radial import Radial
from swirl.radial5 import evolve as EV
from swirl.radial5 import profiles as P
from swirl.radial5.solver_o import SESSION1, SolverO

W = 0.909149280617113
REF = dict(hw=3.344038183621855, Phi0=6.89, E0_s1=2843.7718963433854, rate_1000=0.9096746654018131,
           frac_1000=0.9941458018284945, nl_t0=-0.3156004304072999)


def _phi():
    q = solve_profile(W, 1.0, 1.0, 4.0 / 3.0, tol=1e-10)
    return q, (lambda r: profile_curve(q, r) / np.sqrt(0.30))


def test_step_for_step_against_family_radial():
    q, phi = _phi()
    a = Radial(200.0, 0.05, 1.0, 0.30, 0.12, sponge_start=100.0, sigma_max=1.0)
    b = SolverO("bump", 200.0, 0.05, SESSION1["g4"], SESSION1["g6"], sponge_start=100.0, sigma_max=1.0)
    a.set_phi(phi)
    b.set_f(phi)
    worst = 0.0
    for _ in range(10):                       # 10 x 800 steps = 100 time units, compared every 800
        ra = a.step(0.0125, 800)
        rb, _ = b.step(0.0125, 800)
        worst = max(worst, np.max(np.abs(a.u - b.u)) / np.max(np.abs(a.u)),
                    np.max(np.abs(a.v - b.v)) / np.max(np.abs(a.v)), np.max(np.abs(ra - rb)))
    print("\ncheck1 step-for-step vs swirl/family/radial.py, 8000 steps: max rel diff %.2e "
          "(target round-off, 1e-12)" % worst)
    print("  coupling map: g4_plan = g4_file/6 = %.17g, g6_plan = g6_file/120 = %.17g"
          % (SESSION1["g4"], SESSION1["g6"]))
    assert worst <= 1e-12


def test_start_profile_and_t0_energy():
    q, phi = _phi()
    gen = P.solve(P.Model("O", "bump", **SESSION1), W)
    s = SolverO("bump", 200.0, 0.05, SESSION1["g4"], SESSION1["g6"])
    s.set_f(phi)
    Eh = s.energy_discrete()
    Ed = s.energy_inside([10.0])[1]
    rel_an = abs(Eh - gen["E_start_t0"]) / gen["E_start_t0"]
    print("\ncheck1 start: Phi(0) %.6f (S1 6.89), hw %.9f (S1 %.9f), generic solver F0 %.12f, hw %.9f"
          % (q["P0"] / np.sqrt(0.3), q["half_width"], REF["hw"], gen["F_peak"], gen["hw"]))
    print("  E(t=0): discrete E_h %.6f, 4th-order density %.6f, analytic start energy %.6f "
          "(rel %.1e, target 1e-6); Session 1 E0 %.6f (np.gradient diagnostic, rel %.1e, F-003)"
          % (Eh, Ed, gen["E_start_t0"], rel_an, REF["E0_s1"], (Eh - REF["E0_s1"]) / REF["E0_s1"]))
    print("  profile E (cycle average) %.6f, N %.6f; profile.py E %.6f, N %.6f"
          % (gen["E"], gen["N"], q["E"] / 0.3, q["N"] / 0.3))
    assert abs(q["half_width"] - REF["hw"]) < 1e-9
    assert abs(gen["F_peak"] - q["P0"] / np.sqrt(0.3)) < 1e-9 * gen["F_peak"]
    assert abs(gen["E"] - q["E"] / 0.3) < 1e-9 * gen["E"]
    assert rel_an < 1e-6
    assert abs(Eh - REF["E0_s1"]) / REF["E0_s1"] < 1e-4


def test_settled_rate_and_frac_T1000():
    st = EV.profile_start("O", "bump", W)
    res = EV.run("O", "bump", st, 1000.0)
    rows = np.array(res["per_period"]["rows"])
    sel = (rows[:, 0] >= 975) & (rows[:, 0] <= 1025)
    rate = float(np.median(rows[sel, 1]))
    rel = abs(rate - REF["rate_1000"]) / REF["rate_1000"]
    fr = res["frac_at"]["1000"]
    print("\ncheck1 T=1000 via radial5: rate (median 975-1025) %.7f vs Session 1 %.7f (rel %.1e, "
          "target 1e-4); settled 50-150 %.6f; spectral last 200 %.6f; frac(1000) %.6f vs %.6f "
          "(diff %.1e, target 5e-4: different E_in quadrature); nonlinear share t=0 %.4f (S1 %.4f)"
          % (rate, REF["rate_1000"], rel, res["regime"]["rate_settled_50_150"],
             res["regime"]["rate_spectral_last200"], fr, REF["frac_1000"], fr - REF["frac_1000"],
             res["regime"]["nonlinear_share_signed_t0"], REF["nl_t0"]))
    assert rel < 1e-4
    assert abs(fr - REF["frac_1000"]) < 5e-4
