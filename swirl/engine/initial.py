"""Initial data: radial profiles and phased Gaussians, summed on the grid.

Distances use the minimum image in the periodic box.

Radial entry (dict): r, Phi (arrays, Phi(r) on 0 <= r <= r_max, zero beyond),
    center [x,y,z] (default origin), quad_eps (default 0), phase (default 0),
    omega (default 0). The quadrupolar distortion is a volume-preserving
    stretch along z: Phi evaluated at
        r' = sqrt((x^2 + y^2)(1 + eps) + z^2 / (1 + eps)^2),
    which is l = 2 at first order in eps and smooth at the centre.
    phi = Phi(r') cos(phase), phi_t = -omega Phi(r') sin(phase).

Gaussian entry (dict): center, amp, R, phase (default 0), omega (default 0):
    phi = amp cos(phase) exp(-r^2 / 2R^2), phi_t = -omega amp sin(phase) exp(...)
    (the legacy convention, paper_draft_v9 section 2.2).
"""

import numpy as np
from scipy.interpolate import CubicSpline


def _offsets(coords, L, center):
    out = []
    for c, a in zip(center, (0, 1, 2)):
        d = coords - c
        d = d - L * np.round(d / L)
        shp = [1, 1, 1]
        shp[a] = -1
        out.append(d.reshape(shp))
    return out


def build_initial(coords, L, init):
    N = coords.size
    phi = np.zeros((N, N, N))
    pi = np.zeros((N, N, N))
    for g in init.get("gaussians", []):
        dx, dy, dz = _offsets(coords, L, g.get("center", (0, 0, 0)))
        prof = g["amp"] * np.exp(-(dx ** 2 + dy ** 2 + dz ** 2) / (2 * g["R"] ** 2))
        th = g.get("phase", 0.0)
        phi += np.cos(th) * prof
        if np.sin(th) != 0.0 and g.get("omega", 0.0) != 0.0:
            pi += -g["omega"] * np.sin(th) * prof
    for p in init.get("radial", []):
        r_tab = np.asarray(p["r"], float)
        P_tab = np.asarray(p["Phi"], float)
        spl = CubicSpline(r_tab, P_tab)
        dx, dy, dz = _offsets(coords, L, p.get("center", (0, 0, 0)))
        eps = p.get("quad_eps", 0.0)
        rr = np.sqrt((dx ** 2 + dy ** 2) * (1 + eps) + dz ** 2 / (1 + eps) ** 2)
        prof = np.where(rr <= r_tab[-1], spl(np.minimum(rr, r_tab[-1])), 0.0)
        th = p.get("phase", 0.0)
        phi += np.cos(th) * prof
        if np.sin(th) != 0.0 and p.get("omega", 0.0) != 0.0:
            pi += -p["omega"] * np.sin(th) * prof
    return phi, pi
