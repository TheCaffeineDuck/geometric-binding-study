"""Arrangements, their symmetry groups (found from coordinates), phase-pattern enumeration and pattern symmetry.

Conventions. X is (n, 3), circumradius 1, centre lump (if any) at the origin and last. A group element is
(Q, perm, det) with Q x_i = x_perm[i]. A phase pattern theta (radians) is mapped by g to theta' with
theta'[perm[i]] = theta[i].
"""
import itertools

import numpy as np

TOL = 1e-7


# ---------------------------------------------------------------- arrangements (unit circumradius)
def ring(K):
    a = 2 * np.pi * np.arange(K) / K
    return np.column_stack([np.cos(a), np.sin(a), np.zeros(K)])


def tetrahedron():
    return np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3)


def octahedron():
    return np.vstack([np.eye(3), -np.eye(3)])


def cube():
    return np.array(list(itertools.product([-1, 1], repeat=3)), float) / np.sqrt(3)


def icosahedron():
    g = (1 + np.sqrt(5)) / 2
    pts = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            pts += [[0, s1, s2 * g], [s1, s2 * g, 0], [s2 * g, 0, s1]]
    X = np.array(pts, float)
    return X / np.linalg.norm(X[0])


def with_centre(X):
    return np.vstack([X, np.zeros(3)])


def arrangements():
    out = {}
    for K in range(3, 9):
        out["ring%d" % K] = ring(K)
        out["ring%d+c" % K] = with_centre(ring(K))
    for name, f in (("tetra", tetrahedron), ("octa", octahedron), ("cube", cube), ("icosa", icosahedron)):
        out[name] = f()
        out[name + "+c"] = with_centre(f())
    return out


def has_centre(X):
    return bool(np.linalg.norm(X[-1]) < TOL)


# ---------------------------------------------------------------- symmetry group from coordinates
def _perm_of(Q, X):
    Y = X @ Q.T
    D = np.linalg.norm(Y[:, None, :] - X[None, :, :], axis=2)
    perm = np.argmin(D, axis=1)
    if np.max(D[np.arange(len(X)), perm]) > 1e-6 or len(set(perm.tolist())) != len(X):
        return None
    return perm


def point_group(X):
    """All orthogonal Q mapping the point set onto itself (rotations and improper elements)."""
    X = np.asarray(X, float)
    r = np.linalg.norm(X, axis=1)
    nz = np.nonzero(r > TOL)[0]
    ia = nz[0]
    ib = next(j for j in nz if np.linalg.norm(np.cross(X[ia], X[j])) > 1e-6)
    a, b = X[ia], X[ib]
    Minv = np.linalg.inv(np.column_stack([a, b, np.cross(a, b)]))
    els = []
    for i in nz:
        if abs(r[i] - r[ia]) > TOL:
            continue
        for j in nz:
            if j == i or abs(r[j] - r[ib]) > TOL or abs(X[i] @ X[j] - a @ b) > TOL:
                continue
            for s in (1, -1):
                Q = np.column_stack([X[i], X[j], s * np.cross(X[i], X[j])]) @ Minv
                if not np.allclose(Q.T @ Q, np.eye(3), atol=1e-9):
                    continue
                perm = _perm_of(Q, X)
                if perm is None or any(np.allclose(Q, e[0], atol=1e-9) for e in els):
                    continue
                els.append((Q, perm, int(round(np.linalg.det(Q)))))
    return els


# ---------------------------------------------------------------- 0/180 patterns: orbits and Burnside
def _cycles(perm):
    seen, cyc = np.zeros(len(perm), bool), []
    for i in range(len(perm)):
        if not seen[i]:
            L, j = 0, i
            while not seen[j]:
                seen[j] = True
                j = perm[j]
                L += 1
            cyc.append(L)
    return cyc


def burnside_count(group, n, flip=True):
    """Number of 0/180 classes under the group (x the global sign flip if flip) by Burnside's lemma."""
    tot = 0
    for _, perm, _ in group:
        cyc = _cycles(perm)
        tot += 2 ** len(cyc)
        if flip:
            tot += 2 ** len(cyc) if all(c % 2 == 0 for c in cyc) else 0
    return tot / (len(group) * (2 if flip else 1))


def binary_classes(group, n, flip=True):
    """Brute-force orbits of all 2^n colourings. Returns list of (representative bits, orbit size);
    the representative is the smallest integer code in the orbit."""
    codes = np.arange(2 ** n)
    bits = ((codes[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int64)
    w = (1 << np.arange(n)).astype(np.int64)
    canon = codes.copy()
    for _, perm, _ in group:
        inv = np.argsort(perm)
        img = bits[:, inv]               # img[:, perm[i]] = bits[:, i]
        for b in ((img,) + ((1 - img,) if flip else ())):
            canon = np.minimum(canon, b @ w)
    reps, counts = np.unique(canon, return_counts=True)
    return [(((int(r) >> np.arange(n)) & 1).astype(int), int(c)) for r, c in zip(reps, counts)]


# ---------------------------------------------------------------- pattern symmetry and equivalence
def _const_mod_2pi(x):
    z = np.exp(1j * x)
    return bool(np.max(np.abs(z - z[0])) < 1e-8)


def pattern_symmetry(group, theta, allow_reverse=True):
    """Group elements that leave the pair law invariant: theta[perm[i]] = s theta[i] + c (s = +-1, any c)."""
    out = []
    for Q, perm, det in group:
        for s in ((1, -1) if allow_reverse else (1,)):
            if _const_mod_2pi(theta[perm] - s * theta):
                out.append((Q, perm, det, s))
                break
    return out


def radial_by_symmetry(X, sym):
    """Per lump: dimension of the subspace fixed by its stabilizer in the pattern symmetry. Forces are
    radial by symmetry when it is 1 for every off-centre lump and 0 for the centre."""
    dims = []
    for i in range(len(X)):
        Qs = [Q for Q, perm, _, _ in sym if perm[i] == i]
        P = np.mean(Qs, axis=0)
        dims.append(int(np.sum(np.linalg.svd(P, compute_uv=False) > 1e-8)))
    ok = all((d == 0) if np.linalg.norm(X[i]) < TOL else (d == 1) for i, d in enumerate(dims))
    return ok, dims


def phase_exchanging_mirror(group, theta, X):
    """N15: a pure reflection (det -1, trace +1) with theta[perm[i]] = theta[i] + pi for every lump.
    Returns the plane normal or None."""
    for Q, perm, det in group:
        if det == -1 and abs(np.trace(Q) - 1) < 1e-8:
            if np.max(np.abs(np.exp(1j * theta[perm]) + np.exp(1j * theta))) < 1e-8:
                w, v = np.linalg.eigh((Q + Q.T) / 2)
                return v[:, np.argmin(w)].tolist()
    return None


def invariant(X, theta, nd=6):
    i, j = np.triu_indices(len(X), 1)
    d = np.round(np.linalg.norm(X[i] - X[j], axis=1), nd)
    c = np.round(np.cos(theta[i] - theta[j]), nd) + 0.0
    return tuple(sorted(zip(d.tolist(), c.tolist())))


def equivalent(group, t1, t2):
    """t2 is the image of t1 under a group element up to a global phase (time shift; includes the sign flip)."""
    for _, perm, _ in group:
        t1g = np.empty_like(t1)
        t1g[perm] = t1
        if _const_mod_2pi(t2 - t1g):
            return True
    return False


def is_binary(theta):
    return bool(np.max(np.abs(np.exp(2j * theta) - 1)) < 1e-8)


# ---------------------------------------------------------------- symmetry axes and stepped patterns
def symmetry_axes(X, group):
    """Rotation axes from the proper elements, grouped into classes under the group. Each class gets an
    order (largest n with a 2 pi/n rotation about it) and a type from coordinates: 'vertex' if a lump
    other than the centre lies on the axis, else 'edge' if the midpoint of a nearest-neighbour pair does,
    else 'face'."""
    lines = []
    for Q, perm, det in group:
        if det != 1 or np.allclose(Q, np.eye(3), atol=1e-9):
            continue
        w, v = np.linalg.eig(Q)
        ax = np.real(v[:, np.argmin(np.abs(w - 1))])
        ax /= np.linalg.norm(ax)
        ang = np.arccos(np.clip((np.trace(Q) - 1) / 2, -1, 1))
        for L in lines:
            if abs(abs(L["axis"] @ ax) - 1) < 1e-8:
                L["min_angle"] = min(L["min_angle"], ang)
                break
        else:
            lines.append({"axis": ax, "min_angle": ang})
    off = [x for x in X if np.linalg.norm(x) > TOL]
    off = np.array(off)
    i, j = np.triu_indices(len(off), 1)
    dd = np.linalg.norm(off[i] - off[j], axis=1)
    nn = np.isclose(dd, dd.min(), atol=1e-8)
    mids = (off[i][nn] + off[j][nn]) / 2
    classes = []
    for L in lines:
        ax = L["axis"]
        rep = next((c for c in classes if any(abs(abs((Q @ c["axis"]) @ ax) - 1) < 1e-8 for Q, _, _ in group)), None)
        if rep is not None:
            rep["count"] += 1
            continue
        on = lambda P: np.any(np.linalg.norm(np.cross(P, ax), axis=1) < 1e-8)
        typ = "vertex" if on(off) else ("edge" if on(mids) else "face")
        classes.append({"axis": ax, "order": int(round(2 * np.pi / L["min_angle"])), "type": typ, "count": 1})
    return classes


def azimuths(X, axis):
    axis = axis / np.linalg.norm(axis)
    perp = X - np.outer(X @ axis, axis)
    rho = np.linalg.norm(perp, axis=1)
    k = int(np.argmax(rho))
    e1 = perp[k] / rho[k]
    e2 = np.cross(axis, e1)
    phi = np.arctan2(X @ e2, X @ e1)
    return phi, rho > TOL


def stepped_patterns(X, axis, order, mmax_search=24):
    """theta_i = m * azimuth_i about the axis; lumps on the axis get phase 0. m runs over every value that
    gives a distinct pattern: 0 .. L-1 where L is the smallest m > 0 with m * azimuth constant mod 2 pi;
    if no L <= 24 exists (irregular azimuths), m runs 0 .. 2*order - 1 and the cap is flagged."""
    phi, off = azimuths(X, axis)
    L = next((m for m in range(1, mmax_search + 1) if _const_mod_2pi(m * phi[off])), None)
    ms = range(L) if L else range(2 * order)
    pats = []
    for m in ms:
        th = np.where(off, np.mod(m * phi, 2 * np.pi), 0.0)
        pats.append((m, th))
    return pats, (L is None)
