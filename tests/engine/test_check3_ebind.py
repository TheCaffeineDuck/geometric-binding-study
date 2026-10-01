"""Check 3: t = 0 cube binding energies vs paper Table 2 and the LESSONS 2 closed form.

Legacy definition (paper_draft_v9 section 2.3, studies/02a_cube_phase2.py):
E_bind = E_total(cluster) - 8 E_single, E_single = energy of one isolated
Gaussian on the same grid (legacy set_B_baseline E_series[0] = 13.4617823...).
Phases enter as phi = phi0 cos(theta) G, phi_t = -omega_eff phi0 sin(theta) G
with omega_eff = 1.113 (section 2.2); for theta in {0, pi} this is amplitude
+-phi0 and phi_t = 0 to rounding, as in the legacy cube script.

Vertex coordinates and ordering: paper Appendix A (0=(-,-,-), 1=(+,-,-),
2=(+,+,-), 3=(-,+,-), 4=(-,-,+), 5=(+,-,+), 6=(+,+,+), 7=(-,+,+)), phases
from Appendix B in that order. Cross-edge counts are recomputed from the
coordinates and must equal the CE labels. Grid: N = 64, L = 50, Set B.

Targets: |E_bind - Table 2| < 0.05; |E_bind - closed form| < max(2%, 0.1).
"""

import numpy as np
import pytest

from swirl.engine.evolver import Evolver
from swirl.engine.initial import build_initial

P = np.pi
VERTS = 3.0 * np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
# name: (phases, CE label, Table 2 E_bind)
CONFIGS = {
    "all_same": ([0, 0, 0, 0, 0, 0, 0, 0], 0, 73.84),
    "single_flip": ([P, 0, 0, 0, 0, 0, 0, 0], 3, 36.89),
    "adjacent_pair": ([P, P, 0, 0, 0, 0, 0, 0], 4, 20.59),
    "balanced": ([P, P, 0, P, 0, 0, 0, P], 6, -3.86),
    "anti_tetrahedral": ([P, 0, 0, P, 0, P, P, 0], 8, -23.94),
    "checkerboard": ([P, 0, P, 0, 0, P, 0, P], 12, -51.53),
}
PHI0, R, M, W_EFF = 0.5, 2.5, 1.0, 1.113


def J(d):
    return PHI0 ** 2 * np.pi ** 1.5 * R ** 3 * np.exp(-d ** 2 / (4 * R ** 2)) * (
        M ** 2 + 1.5 / R ** 2 - d ** 2 / (4 * R ** 4))


@pytest.fixture(scope="module")
def grid():
    ev = Evolver(64, 50.0, M, 0.30, 0.055)
    phi, pi = build_initial(ev.coords, ev.L, {"gaussians": [{"center": [0, 0, 0], "amp": PHI0, "R": R}]})
    return ev, ev.energy_parts(phi, pi)["E_total"]


@pytest.mark.parametrize("name", list(CONFIGS))
def test_cube_ebind_t0(name, grid, record):
    ev, e_single = grid
    phases, ce_label, table2 = CONFIGS[name]
    gs = [{"center": list(v), "amp": PHI0, "R": R, "phase": th, "omega": W_EFF}
          for v, th in zip(VERTS, phases)]
    phi, pi = build_initial(ev.coords, ev.L, {"gaussians": gs})
    e_bind = ev.energy_parts(phi, pi)["E_total"] - 8 * e_single

    s = np.cos(phases)
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    dist = {p: np.linalg.norm(VERTS[p[0]] - VERTS[p[1]]) for p in pairs}
    edges = [p for p in pairs if abs(dist[p] - 6.0) < 1e-9]
    ce = sum(1 for i, j in edges if s[i] * s[j] < 0)
    closed = sum(s[i] * s[j] * J(dist[(i, j)]) for i, j in pairs)
    tol_closed = max(0.02 * abs(closed), 0.1)
    record("check3_ebind_" + name, {
        "check": "3", "config": name, "phases": [float(x) for x in phases], "cross_edges": ce,
        "E_single": e_single, "E_bind_t0": e_bind, "table2": table2, "diff_table2": e_bind - table2,
        "closed_form": closed, "diff_closed": e_bind - closed, "tol_closed": tol_closed,
        "N": 64, "L": 50.0, "phi0": PHI0, "R": R, "omega_eff": W_EFF, "g4": 0.30, "g6": 0.055})
    assert len(edges) == 12
    assert ce == ce_label
    assert abs(e_bind - table2) < 0.05
    assert abs(e_bind - closed) < tol_closed
