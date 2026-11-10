"""The nonlinear R kernel against the Euler-Lagrange equation, with analytic fields
(reviewer condition 6), and the order of both radial operators in the first cells.

R: p_t = r^-2 d_r(r^2 gamma f_r) - s2 gamma f / r^2 - f,  gamma = (1 - X)^-1/2,
written out with exact derivatives of f = A exp(-r^2/2) (bump) or A r exp(-r^2/2) (pinch)
and f_t = B exp(-r^2/2) (bump) or B r^3 exp(-r^2/2) (pinch), B chosen so max X = 0.75.
Target: observed bulk order (r in [1, 8]) >= 3.5 and bulk error at h = 0.025 below 1e-4
relative to max|p_t|. The first cells are reported, not asserted: the divergence form
(1/r^2) D(r^2 q D f) is second order there (bump; third at cell 0 for the pinch). The test must fail if gamma were dropped from the pinch's s2 f/r^2 term (in
kernel and energy alike, so energy conservation would not see it): the size of that change
is printed and must exceed the tolerance by a factor 1e3.
"""

import numpy as np
import pytest

from swirl.radial5.solver_o import SolverO
from swirl.radial5.solver_r import SolverR

HS = (0.1, 0.05, 0.025)


def fields(obj, A, B, r):
    e = np.exp(-r * r / 2)
    if obj == "bump":
        f, fr, frr = A * e, -A * r * e, A * (r * r - 1) * e
        ft, ftr = B * e, -B * r * e
    else:
        f, fr, frr = A * r * e, A * (1 - r * r) * e, A * (r ** 3 - 3 * r) * e
        ft, ftr = B * r ** 3 * e, B * (3 * r * r - r ** 4) * e
    return f, fr, frr, ft, ftr


def exact_pt(obj, A, B, r):
    s2 = 0.0 if obj == "bump" else 2.0
    f, fr, frr, ft, ftr = fields(obj, A, B, r)
    S = fr * fr + s2 * f * f / r ** 2
    X = ft * ft - S
    g = (1 - X) ** -0.5
    Sr = 2 * fr * frr + s2 * (2 * f * fr / r ** 2 - 2 * f * f / r ** 3)
    Xr = 2 * ft * ftr - Sr
    gr = 0.5 * g ** 3 * Xr
    div = (2 * r * g * fr + r * r * gr * fr + r * r * g * frr) / (r * r)
    return div - s2 * g * f / r ** 2 - f, g, X, f, ft


def pick_B(obj, A):
    r = np.linspace(1e-4, 10, 200001)
    _, fr, _, ft1, _ = fields(obj, A, 1.0, r)
    f = fields(obj, A, 1.0, r)[0]
    S = fr * fr + (0.0 if obj == "bump" else 2.0) * f * f / r ** 2
    # max_r (B^2 ft1^2 - S) = 0.75: bisection in B
    lo, hi = 0.0, 50.0
    for _ in range(100):
        B = 0.5 * (lo + hi)
        if np.max(B * B * ft1 * ft1 - S) > 0.75:
            hi = B
        else:
            lo = B
    return lo


@pytest.mark.parametrize("obj", ["bump", "pinch"])
def test_R_kernel_matches_euler_lagrange(obj):
    A = 1.5 if obj == "bump" else 0.6
    B = pick_B(obj, A)
    errs, first = [], []
    for h in HS:
        s = SolverR(obj, 12.0, h)
        pt_ex, g, X, f, ft = exact_pt(obj, A, B, s.r)
        s.f = f.copy()
        s.p = g * ft
        ft_k, pt_k = s.rhs()
        scale = np.max(np.abs(pt_ex))
        bulk = (s.r > 1) & (s.r < 8)
        errs.append(np.max(np.abs(pt_k - pt_ex)[bulk]) / scale)
        first.append(np.abs(pt_k - pt_ex)[:3] / scale)
        # f_t = p/q with q from the discrete S: equal to the analytic f_t to discretisation error
        assert np.max(np.abs(ft_k - ft)) < 1e-4 * np.max(np.abs(ft))
    order = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    first = np.array(first)
    # the first cells sit at r = (i + 1/2) h, so compare the same cell index across h
    order_first = np.log2(first[:-1] / first[1:])
    s2 = 0.0 if obj == "bump" else 2.0
    s = SolverR(obj, 12.0, 0.05)
    pt_ex, g, X, f, ft = exact_pt(obj, A, B, s.r)
    bug = np.max(np.abs(s2 * f * (g - 1) / s.r ** 2)) / np.max(np.abs(pt_ex))
    print("\nEL test R %s: max X %.3f, max gamma %.3f; bulk rel err h=0.1/0.05/0.025: %s, order %s; "
          "first three cells (index 0,1,2) err at h=0.05: %s, order (cell by cell, h 0.1->0.05->0.025): "
          "%s; size of dropping gamma from s2 f/r^2: %.2e (must exceed the tolerance 1e-6 by 1e3)"
          % (obj, np.max(X), np.max(g), ["%.2e" % e for e in errs], ["%.2f" % o for o in order],
             ["%.1e" % e for e in first[1]], [["%.2f" % o for o in row] for row in order_first], bug))
    assert errs[-1] < 1e-4 and np.min(order) > 3.5
    if obj == "pinch":
        assert bug > 1e3 * 1e-6


@pytest.mark.parametrize("obj", ["bump", "pinch"])
def test_O_operator_order_first_cells(obj):
    """O: u_tt = r f_tt with f_tt = f_rr + 2 f_r/r - s2 f/r^2 - U'(f), analytic f; nodes r_i = i h."""
    A = 2.0
    g4, g6 = 0.05, 0.001
    errs, first = [], []
    for h in HS:
        s = SolverO(obj, 12.0, h, g4, g6)
        r = s.r_safe
        f, fr, frr, _, _ = fields(obj, A, 0.0, r)
        s2 = s.s2
        ftt = frr + 2 * fr / r - s2 * f / r ** 2 - (f - g4 * f ** 3 + g6 * f ** 5)
        s.u = s.r * f
        s.u[0] = s.u[-1] = 0.0
        _, a = s.rhs()
        ex = s.r * ftt
        scale = np.max(np.abs(ex[1:-1]))
        bulk = (s.r > 1) & (s.r < 8)
        errs.append(np.max(np.abs(a - ex)[bulk]) / scale)
        first.append(np.abs(a - ex)[1:4] / scale)
    order = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    first = np.array(first)
    print("\nEL test O %s: bulk rel err %s, order %s; first nodes (i = 1,2,3) err at h=0.05 %s, "
          "order node by node %s"
          % (obj, ["%.2e" % e for e in errs], ["%.2f" % o for o in order],
             ["%.1e" % e for e in first[1]], [["%.2f" % o for o in row] for row in np.log2(first[:-1] / first[1:])]))
    assert np.min(order) > 3.5
