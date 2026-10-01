"""Shared helpers for the LESSONS audit (WP1C). Reads legacy/ only; never writes there."""
import json, os, itertools, math
import numpy as np

ROOT = "/Users/aaronramos/Desktop/SWIRL"
LEG = ROOT + "/legacy"
OUT = ROOT + "/outputs/audit"
M, G4, G6, PHI0, R = 1.0, 0.30, 0.055, 0.5, 2.5
N, L, DT, SIGMA = 64, 50.0, 0.05, 0.01

def load(p):
    with open(p) as f:
        return json.load(f)

def save_part(name, d):
    tmp = OUT + "/part_%s.json.tmp" % name
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, OUT + "/part_%s.json" % name)

def J(d, phi0=PHI0, R=R, m=M):
    return phi0**2 * math.pi**1.5 * R**3 * math.exp(-d**2 / (4 * R**2)) * (m**2 + 3 / (2 * R**2) - d**2 / (4 * R**4))

def K(d, omega, phi0=PHI0, R=R):
    return phi0**2 * omega**2 * math.pi**1.5 * R**3 * math.exp(-d**2 / (4 * R**2))

def cube_verts(d=6.0):
    D = d / 2
    return np.array([[(-D, D)[(i >> 0) & 1], (-D, D)[(i >> 1) & 1], (-D, D)[(i >> 2) & 1]] for i in range(8)])

def ico_verts(d=6.0):
    g = (1 + 5**0.5) / 2
    b = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            b += [[0, s1, s2 * g], [s1, s2 * g, 0], [s1 * g, 0, s2]]
    b = np.array(b, float)
    dm = min(np.linalg.norm(b[i] - b[j]) for i in range(12) for j in range(i + 1, 12))
    return b * d / dm

def oct_verts(d=6.0):
    a = d / 2**0.5
    return np.array([[a,0,0],[-a,0,0],[0,a,0],[0,-a,0],[0,0,a],[0,0,-a]], float)

def edges_from_coords(V):
    D = np.linalg.norm(V[:, None] - V[None], axis=2)
    dm = D[D > 1e-9].min()
    return [(i, j) for i in range(len(V)) for j in range(i + 1, len(V)) if abs(D[i, j] - dm) < 1e-6 * dm + 1e-6]

def pair_sum_binary(V, s):
    s = np.asarray(s, float); tot = 0.0
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            tot += s[i] * s[j] * J(np.linalg.norm(V[i] - V[j]))
    return tot

def symmetry_perms(V, proper=None, tol=1e-7):
    """All orthogonal maps taking vertex set to itself, as permutations, from coordinates.
    proper=True: det +1 only; False: det -1 only; None: all."""
    n = len(V)
    # choose 3 linearly independent source vertices
    idx = None
    for a, b, c in itertools.combinations(range(n), 3):
        if abs(np.linalg.det(V[[a, b, c]])) > 1e-6:
            idx = (a, b, c); break
    S = V[list(idx)].T; Si = np.linalg.inv(S)
    perms = {}
    for tgt in itertools.permutations(range(n), 3):
        Mx = V[list(tgt)].T @ Si
        if np.abs(Mx.T @ Mx - np.eye(3)).max() > tol: continue
        det = np.linalg.det(Mx)
        if proper is True and det < 0: continue
        if proper is False and det > 0: continue
        mapped = V @ Mx.T
        p = []
        ok = True
        for a in range(n):
            dd = np.linalg.norm(V - mapped[a], axis=1)
            b = int(dd.argmin())
            if dd[b] > 1e-6 * np.abs(V).max(): ok = False; break
            p.append(b)
        if ok and len(set(p)) == n:
            perms[tuple(p)] = det
    return perms
