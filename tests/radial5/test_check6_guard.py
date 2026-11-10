"""Check 6. Guard for equation R.

In (f, p), 1 - X = (1 + S)/(1 + p^2) > 0 for every finite state (proof in
docs/WPB2_BUILD_NOTES.md), so the X < 1 test cannot trip during a run of finite data; it
trips on a start outside the domain (f_t^2 >= 1 + S). What the guard watches during a run:
non-finite values, q = (1 - X)^-1/2 > q_max (momentum growing without bound), loss of
resolution (high-k power share), and for the pinch the angular hyperbolicity condition
f_t^2 < 1 + f_r^2 + (f/r)^2, i.e. 1 - X > (f/r)^2, which CAN fail for finite p.
Every test here is built so that it would fail if the guard were idle or if it altered
the trajectory.
"""

import numpy as np
import pytest

from swirl.radial5 import evolve as EV
from swirl.radial5.solver_r import GuardTrip, SolverR


def test_identity_X_below_one_for_finite_data():
    rng = np.random.default_rng(3)
    worst_id, min_om, n_neg_margin = 0.0, np.inf, 0
    for obj in ("bump", "pinch"):
        s = SolverR(obj, 20.0, 0.05)
        par = s.r / (1 + s.r) if obj == "pinch" else 1.0
        for trial in range(200):
            amp_p = 10.0 ** rng.uniform(-3, 8)
            s.f = rng.normal(size=s.M).cumsum() * 0.05 * par
            s.p = amp_p * rng.normal(size=s.M)
            S = s.S(s.f)
            ft, _ = s.rhs()
            X_ham = ft ** 2 - S                               # X from Hamilton's f_t
            X_id = 1.0 - (1.0 + S) / (1.0 + s.p ** 2)         # the identity
            ok = np.abs(X_id) > 1e-3
            worst_id = max(worst_id, float(np.max(np.abs(X_ham - X_id)[ok] / np.abs(X_id[ok]))))
            min_om = min(min_om, float(np.min((1.0 + S) / (1.0 + s.p ** 2))))   # 1 - X
            if obj == "pinch":
                n_neg_margin += int(np.any((1.0 + S) / (1.0 + s.p ** 2) - s.f ** 2 / s.r ** 2 <= 0))
    print("\ncheck6 identity: 400 random states, |p| up to 1e8: max rel |X(f_t) - X(identity)| = %.1e, "
          "min (1 - X) = %.1e (> 0; X itself rounds to 1.0 in double at |p| ~ 1e8, which is why "
          "the kernel tests 1 - X = (1+S)/(1+p^2) > 0, never X < 1); pinch states with the angular "
          "margin <= 0: %d of 200" % (worst_id, min_om, n_neg_margin))
    assert worst_id < 1e-9 and min_om > 0.0 and n_neg_margin > 0


def test_start_outside_domain_trips():
    s = SolverR("bump", 50.0, 0.05)
    with pytest.raises(GuardTrip) as e:
        s.set_f(lambda r: 0.2 * np.exp(-r * r / 8), lambda r: 1.5 * np.exp(-r * r / 8))
    d = e.value.diag
    print("\ncheck6 start f_t = 1.5 at S ~ 0: tripped at t = %g, r = %.3f, X = %.4f (%s)"
          % (d["t"], d["r"], d["X"], d["kind"]))
    assert d["X"] >= 1.0 and d["t"] == 0.0


def _pinch_window_start(sigma=2.0, a=0.5):
    """f = a r exp(-r^2/2 sigma^2), f_t^2 = (1 + f_r^2 + 1.5 (f/r)^2) tanh^2(r/0.2):
    inside X < 1 everywhere (1.5 q^2 < 2 q^2), outside the angular condition where tanh ~ 1."""
    f = lambda r: a * r * np.exp(-r * r / (2 * sigma ** 2))

    def ft(r):
        q = a * np.exp(-r * r / (2 * sigma ** 2))
        fr = q * (1 - r * r / sigma ** 2)
        return np.sqrt(1 + fr * fr + 1.5 * q * q) * np.tanh(r / 0.2) * np.exp(-(r / 8.0) ** 8)
    return f, ft


def test_pinch_angular_condition_trips():
    f, ft = _pinch_window_start()
    s = SolverR("pinch", 50.0, 0.05)
    with pytest.raises(GuardTrip) as e:
        s.set_f(f, ft)
    d = e.value.diag
    print("\ncheck6 pinch start with X < 1 but f_t^2 > 1 + f_r^2 + (f/r)^2: tripped at t = %g, r = %.3f, "
          "X = %.4f, margin 1 - X - (f/r)^2 = %.4f (%s)" % (d["t"], d["r"], d["X"], d["pinch_margin"], d["kind"]))
    assert "angular" in d["kind"] and d["X"] < 1.0 and d["pinch_margin"] <= 0
    # the same state as a single bump (one field direction): X < 1 suffices, no trip
    b = SolverR("bump", 50.0, 0.05)
    b.set_f(lambda r: 0.5 * np.exp(-r * r / 8.0), lambda r: 0.99 * np.sqrt(1 + 0.0 * r))
    assert np.all(b.X() < 1)


def test_pinch_angular_condition_trips_inside_the_kernel():
    """The kernel's per-stage check (not only the start check): put the same state in
    directly and step once."""
    f, ft = _pinch_window_start()
    s = SolverR("pinch", 50.0, 0.05)
    s.f = f(s.r)
    S = s.S(s.f)
    g = ft(s.r)
    s.p = g / np.sqrt(1 + S - g * g)
    with pytest.raises(GuardTrip) as e:
        s.step(0.0125, 1)
    d = e.value.diag
    print("\ncheck6 kernel: tripped in stage %s at t = %g, r = %.3f, margin %.4f, energy so far %.6g (%s)"
          % (d["stage"], d["t"], d["r"], d["pinch_margin"], d["energy_so_far"], d["kind"]))
    assert d["stage"] == 1 and "angular" in d["kind"]


def test_momentum_without_bound_trips():
    s = SolverR("bump", 50.0, 0.05, qmax=1e4)
    s.set_f(lambda r: 0.0 * r, lambda r: (1 - 1e-10) * np.exp(-(r / 5.0) ** 8))
    with pytest.raises(GuardTrip) as e:
        s.step(0.0125, 10)
    d = e.value.diag
    print("\ncheck6 f_t = 1 - 1e-10 at the centre (q ~ 7e4 > q_max 1e4): tripped stage %s, t = %g, r = %.3f, "
          "q = %.3g, max|p| = %.3g (%s)" % (d["stage"], d["t"], d["r"], d["q"], d["max_abs_p"], d["kind"]))
    assert "momentum" in d["kind"]


def test_resolution_trips_on_underresolved_start():
    """A Gaussian of width ~ h is under-resolved: the high-k share of (f, p) is far above
    the 1e-6 limit (resolved production starts measure 1e-12 or less)."""
    res = EV.run("R", "bump", EV.gauss_start("bump", 0.5, 0.06), 5.0)
    d = res["guard"]
    print("\ncheck6 Gaussian sigma 0.06 on h 0.05: guard %s at t = %s, high-k share %.2e"
          % (d["kind"] if d else None, d["t"] if d else None, d["high_k"] if d else float("nan")))
    assert d is not None and "resolution" in d["kind"] and d["t"] <= 0.5 + 1e-9


def test_guard_does_not_alter_the_run_and_extremes_are_logged():
    st = EV.profile_start("R", "pinch", 0.99)
    a = EV.run("R", "pinch", st, 30.0)
    b = EV.run("R", "pinch", st, 30.0, qmax=np.inf, hk_max=np.inf)
    same = a["series"]["frac"] == b["series"]["frac"] and a["core_tail"]["f"] == b["core_tail"]["f"]
    ge = a["guard_extremes"]
    print("\ncheck6 guard on vs off, R pinch 0.99, T = 30: identical %s; logged rows %d; global min(1-X) "
          "%.4f at t %.1f r %.2f; max|p| %.4f at t %.1f r %.2f; min pinch margin %.4f"
          % (same, len(ge["rows"]), ge["global_extremes"]["min_one_minus_X"]["value"],
             ge["global_extremes"]["min_one_minus_X"]["t"], ge["global_extremes"]["min_one_minus_X"]["r"],
             ge["global_extremes"]["max_abs_p"]["value"], ge["global_extremes"]["max_abs_p"]["t"],
             ge["global_extremes"]["max_abs_p"]["r"], ge["global_extremes"]["min_pinch_margin"]["value"]))
    assert same and a["guard"] is None and len(ge["rows"]) >= 6
