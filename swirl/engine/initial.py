"""Initial data: radial profiles and phased Gaussians, summed on the grid.

Distances use the minimum image in the periodic box.

Radial entry (dict): r, Phi (arrays, Phi(r) on 0 = r[0] <= r <= r_max, zero
    beyond; cubic spline clamped to Phi'(0) = 0), center [x,y,z] (default
    origin), quad_eps (default 0), phase (default 0), omega (default 0).
    The quadrupolar distortion is a volume-preserving stretch along z: Phi
    evaluated at
        r' = sqrt((x^2 + y^2)(1 + eps) + z^2 / (1 + eps)^2),
    which is l = 2 at first order in eps and smooth at the centre.
    phi = Phi(r') cos(phase), phi_t = -omega Phi(r') sin(phase).
    Guards (radial_checks) raise ValueError on a bad table.

Gaussian entry (dict): center, amp, R, phase (default 0), omega (default 0):
    phi = amp cos(phase) exp(-r^2 / 2R^2), phi_t = -omega amp sin(phase) exp(...)
    (the legacy convention, paper_draft_v9 section 2.2).

Snapshot entry (dict, init["snapshot"]): copies of one settled lump at given centres and
    phases, see snapshot.py.

WP-L entry (dict, init["wpl"]): settled lumps from a snapshot source, boosted, stretched or
    multiplied by a smooth random field (boost.py).
"""

import numpy as np
from scipy.interpolate import CubicSpline

TAIL_TOL = 1e-6  # |Phi| allowed, relative to |Phi(0)|, where the profile is cut


def radial_checks(p, L):
    """Guards for a radial profile; raises ValueError, else returns the numbers.

    - the table starts at r = 0 and increases;
    - |Phi(r_max)| <= TAIL_TOL |Phi(0)| (the cut at r_max is negligible);
    - r_face = (L/2 - max_i |center_i|) * min(sqrt(1+eps), 1/(1+eps)) is the
      smallest profile radius r' reached at a face of [-L/2, L/2)^3. Where
      r_max > r_face, the profile beyond r_face (cut non-spherically by the
      box, and lying in the absorber) must also be below TAIL_TOL |Phi(0)|.
    """
    r = np.asarray(p["r"], float)
    P = np.asarray(p["Phi"], float)
    c = np.asarray(p.get("center", (0, 0, 0)), float)
    eps = p.get("quad_eps", 0.0)
    if r[0] != 0.0 or np.any(np.diff(r) <= 0):
        raise ValueError("radial table must start at r = 0 and increase")
    p0 = abs(P[0])
    r_face = (L / 2 - np.max(np.abs(c))) * min(np.sqrt(1 + eps), 1 / (1 + eps))
    beyond = r >= r_face
    out = {"Phi0": float(P[0]), "r_max": float(r[-1]), "r_face": float(r_face),
           "Phi_rmax_over_Phi0": float(abs(P[-1]) / p0),
           "max_Phi_beyond_r_face_over_Phi0":
               float(np.max(np.abs(P[beyond])) / p0) if beyond.any() else 0.0}
    if out["Phi_rmax_over_Phi0"] > TAIL_TOL:
        raise ValueError("radial profile not decayed at r_max: " + str(out))
    if out["max_Phi_beyond_r_face_over_Phi0"] > TAIL_TOL:
        raise ValueError("radial profile not negligible beyond the box face: " + str(out))
    return out


def radial_spline(r, Phi):
    """Cubic spline of Phi(r), clamped to Phi'(0) = 0 (regular at the origin)."""
    return CubicSpline(np.asarray(r, float), np.asarray(Phi, float), bc_type=((1, 0.0), "not-a-knot"))


def _offsets(coords, L, center):
    out = []
    for c, a in zip(center, (0, 1, 2)):
        d = coords - c
        d = d - L * np.round(d / L)
        shp = [1, 1, 1]
        shp[a] = -1
        out.append(d.reshape(shp))
    return out


def build_initial(coords, L, init, info=None, threads=4):
    """Sum of the init entries on the grid. A "snapshot" entry (snapshot.place) adds copies of
    one settled lump; its build record goes to info["snapshot"] when info is a dict."""
    N = coords.size
    phi = np.zeros((N, N, N))
    pi = np.zeros((N, N, N))
    if init.get("snapshot"):
        from swirl.engine.snapshot import place
        ps, vs, inf = place(init["snapshot"], coords, L, threads=threads)
        phi += ps
        pi += vs
        if info is not None:
            info["snapshot"] = inf
    if init.get("wpl"):
        from swirl.engine.boost import place_wpl
        ps, vs, inf = place_wpl(init["wpl"], coords, L, threads=threads)
        phi += ps
        pi += vs
        if info is not None:
            info["wpl"] = inf
    for g in init.get("gaussians", []):
        dx, dy, dz = _offsets(coords, L, g.get("center", (0, 0, 0)))
        prof = g["amp"] * np.exp(-(dx ** 2 + dy ** 2 + dz ** 2) / (2 * g["R"] ** 2))
        th = g.get("phase", 0.0)
        phi += np.cos(th) * prof
        if np.sin(th) != 0.0 and g.get("omega", 0.0) != 0.0:
            pi += -g["omega"] * np.sin(th) * prof
    for p in init.get("radial", []):
        radial_checks(p, L)
        r_tab = np.asarray(p["r"], float)
        P_tab = np.asarray(p["Phi"], float)
        spl = radial_spline(r_tab, P_tab)
        dx, dy, dz = _offsets(coords, L, p.get("center", (0, 0, 0)))
        eps = p.get("quad_eps", 0.0)
        rr = np.sqrt((dx ** 2 + dy ** 2) * (1 + eps) + dz ** 2 / (1 + eps) ** 2)
        prof = np.where(rr <= r_tab[-1], spl(np.minimum(rr, r_tab[-1])), 0.0)
        th = p.get("phase", 0.0)
        phi += np.cos(th) * prof
        if np.sin(th) != 0.0 and p.get("omega", 0.0) != 0.0:
            pi += -p["omega"] * np.sin(th) * prof
    return phi, pi
