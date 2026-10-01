"""WP-E: K-lump diagnostics against constructed fields (K = 3 and K = 5), the nearest-centre split,
the group block and the box-face lever fix. Records go to outputs/session2/wpE/checks/<name>.json.

Constructed fields are Gaussians on a grid with R/dx >= 2.5 (spectrally resolved, grid sums equal
the integrals to rounding) and a linear Evolver (g4 = g6 = 0), so the exact values are analytic.
"""

import json
import os

import numpy as np
import pytest

from swirl.engine.evolver import Evolver
from swirl.engine.lumps import (LumpDiagnostics, _wrap, ball_weight, densities, find_lumps, lever_arms,
                                lump_config, pair_distances, split_weight, summarize)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECK_DIR = os.path.join(ROOT, "outputs", "session2", "wpE", "checks")
M = 1.0


def rec(name, data):
    os.makedirs(CHECK_DIR, exist_ok=True)
    path = os.path.join(CHECK_DIR, name + ".json")
    with open(path + ".tmp", "w") as f:
        json.dump(data, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


def ring(K, radius, center, tilt=0.37, rot=0.21):
    """K points on a circle in a tilted plane (axis n), returned with the axis."""
    ang = rot + 2 * np.pi * np.arange(K) / K
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, np.cos(tilt), np.sin(tilt)])
    pts = np.array(center) + radius * (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2)
    return pts, np.cross(e1, e2)


# ---------------------------------------------------------------------------
# nearest-centre split: partition of unity; K = 2 equals the WP-C bisector split
# ---------------------------------------------------------------------------

def _full_weights(coords, L, centers, R, tap, ell):
    K = len(centers)
    D = pair_distances(centers, L)
    d_nn = max(float(np.min(D[np.triu_indices(K, 1)])), ell)
    Ws = []
    for i in range(K):
        rel = lever_arms(coords, L, centers[i])
        dvs = [_wrap(np.asarray(centers[j]) - np.asarray(centers[i]), L) for j in range(K) if j != i]
        Ws.append(split_weight(rel, dvs, R, tap, ell, d_nn))
    return Ws


@pytest.mark.parametrize("K", [2, 3, 5])
def test_split_partition_of_unity(K):
    N, L, R, tap, ell = 64, 32.0, 10.0, 2.0, 1.0
    coords = -L / 2 + (L / N) * np.arange(N)
    if K == 2:
        cen = np.array([[-1.6, 0.13, -0.07], [1.5, 0.11, 0.2]])
    else:
        cen, _ = ring(K, 3.0 / (2 * np.sin(np.pi / K)), [0.13, -0.07, 0.21])   # nearest spacing 3: deep overlap
    Ws = _full_weights(coords, L, cen, R, tap, ell)
    S = sum(Ws)
    inner = np.ones((N, N, N), bool)
    for c in cen:
        rel = lever_arms(coords, L, c)
        inner &= np.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2) <= R - tap
    out = {"K": K, "centers": cen.tolist(), "n_inner_points": int(inner.sum()),
           "max_abs_sum_minus_1_inner": float(np.max(np.abs(S[inner] - 1))),
           "max_sum_everywhere": float(S.max()), "target": 1e-14}
    if K == 2:
        rel = lever_arms(coords, L, cen[0])
        dv = _wrap(cen[1] - cen[0], L)
        dn = np.linalg.norm(dv)
        s = (rel[0] * dv[0] + rel[1] * dv[1] + rel[2] * dv[2]) / dn - dn / 2
        old = ball_weight(np.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2), R, tap) * 0.5 * (1 - np.tanh(s / ell))
        out["max_abs_diff_vs_wpc_tanh_split"] = float(np.max(np.abs(Ws[0] - old)))
        assert out["max_abs_diff_vs_wpc_tanh_split"] < 1e-14
    rec("split_partition_K%d" % K, out)
    assert out["n_inner_points"] > 1000
    assert out["max_abs_sum_minus_1_inner"] < 1e-14
    assert out["max_sum_everywhere"] <= 1 + 1e-14


# ---------------------------------------------------------------------------
# K = 3 and K = 5 Gaussian lumps: per-lump E, It, Ip, P, centroid, distances, count, group
# ---------------------------------------------------------------------------

def _gauss_field(ev, cen, a, Rg, w, thetas):
    phi = np.zeros(ev.shape)
    pi = np.zeros(ev.shape)
    for c, th in zip(cen, thetas):
        rel = lever_arms(ev.coords, ev.L, c)
        G = np.exp(-(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2) / (2 * Rg * Rg))
        phi += a * np.cos(th) * G
        pi += -w * a * np.sin(th) * G
    return phi, pi


@pytest.mark.parametrize("K", [3, 5])
def test_k_lumps_constructed(K):
    """Analytic values for K Gaussian lumps at nearest spacing 22 (the split's exponential tail,
    e^{-d/ell}, is below 1e-9 there; see test_k_lumps_split_machinery for overlap)."""
    N, L, a, Rg, w = 140, 56.0, 0.7, 1.2, 0.9
    ev = Evolver(N, L, M, 0.0, 0.0, threads=4)
    cen, axis = ring(K, 22.0 / (2 * np.sin(np.pi / K)), [0.13, -0.07, 0.21])
    th = 2 * np.pi * np.arange(K) / K + 0.3
    phi, pi = _gauss_field(ev, cen, a, Rg, w, th)
    ev.set_state(phi, pi)
    v = np.pi ** 1.5 * Rg ** 3
    E_ex = v * (w * w * a * a * np.sin(th) ** 2 + M * M * a * a * np.cos(th) ** 2) / 2 \
        + 0.75 * (a * np.cos(th)) ** 2 * np.pi ** 1.5 * Rg
    It_ex = v * (w * a * np.sin(th)) ** 2
    Ip_ex = v * (a * np.cos(th)) ** 2
    G_ex = (E_ex[:, None] * cen).sum(0) / E_ex.sum()
    start = cen + np.array([0.2, -0.15, 0.1])
    cfg = lump_config({"centers": start.tolist(), "R_ball": 7.0, "taper": 1.0, "count_every": 1e9,
                       "group": {"R": 26.0, "taper": 2.0, "axis": axis.tolist()}})
    ld = LumpDiagnostics(ev, cfg)
    for _ in range(4):
        ld.sample(0.0)
    s = ld.series
    E = np.array(s["E"][-1])
    X = np.array(s["X"][-1])
    Dm = pair_distances(X, L)
    D_ex = pair_distances(cen, L)
    e, _ = densities(ev)
    fl = find_lumps(e, ev.coords, L, ev.dx)
    peaks = np.array([p["pos"] for p in fl["peaks"]])
    peak_err = max(float(np.min(np.linalg.norm(peaks - c, axis=1))) for c in cen)
    scale = a * a * v
    out = {"K": K, "N": N, "L": L, "dx": ev.dx, "gauss_R": Rg, "nearest_spacing": 22.0, "thetas": th.tolist(),
           "max_rel_err_E": float(np.max(np.abs(E / E_ex - 1))),
           "max_abs_err_It_over_scale": float(np.max(np.abs(np.array(s["It"][-1]) - It_ex))) / scale,
           "max_abs_err_Ip_over_scale": float(np.max(np.abs(np.array(s["Ip"][-1]) - Ip_ex))) / scale,
           "max_abs_P_over_scale": float(np.max(np.abs(np.array(s["P"][-1])))) / scale,
           "centroid_err_per_iteration": [float(np.max(np.linalg.norm(np.array(x) - cen, axis=1))) for x in s["X"]],
           "max_abs_err_distance_matrix": float(np.max(np.abs(Dm - D_ex))),
           "count": fl["count"], "count_distinct": fl["count_distinct"], "max_peak_position_err": peak_err,
           "group_E_rel_err": s["E_group"][-1] / ev.energy() - 1,
           "group_centroid_err": float(np.linalg.norm(np.array(s["G"][-1]) - G_ex)),
           "group_L_box_over_scale": float(np.max(np.abs(s["L_group_box"][-1]))) / scale,
           "targets": {"E": 1e-8, "It_Ip_P": 1e-8, "centroid": 1e-8, "distances": 1e-8, "count": K,
                       "group_E": 1e-8, "group_centroid": 1e-8, "group_L": 1e-12}}
    rec("k_lumps_constructed_K%d" % K, out)
    assert out["max_rel_err_E"] < 1e-8
    assert out["max_abs_err_It_over_scale"] < 1e-8 and out["max_abs_err_Ip_over_scale"] < 1e-8
    assert out["max_abs_P_over_scale"] < 1e-8
    assert out["centroid_err_per_iteration"][-1] < 1e-8
    assert out["max_abs_err_distance_matrix"] < 1e-8
    assert out["count"] == K and out["count_distinct"] == K and peak_err < 0.1
    assert abs(out["group_E_rel_err"]) < 1e-8 and out["group_centroid_err"] < 1e-8
    assert out["group_L_box_over_scale"] < 1e-12


@pytest.mark.parametrize("K,spacing", [(3, 4.0), (5, 4.0), (5, 11.76)])
def test_k_lumps_split_machinery(K, spacing):
    """Overlapping balls: per-lump E, X, It, Ip, P from LumpDiagnostics against an independently
    coded full-grid evaluation of the same definition (r_j^2 computed directly). Also records the
    split leak: E_i against the analytic single-lump energy (the bisector split has width ell, so
    lump i keeps 1/(1 + e^{-d/ell}) of its own core; that energy is assigned to no lump)."""
    N, L, a, Rg, w, R, tap, ell = 100, 40.0, 0.7, 1.2, 0.9, 7.0, 1.0, 1.0
    ev = Evolver(N, L, M, 0.0, 0.0, threads=4)
    cen, _ = ring(K, spacing / (2 * np.sin(np.pi / K)), [0.13, -0.07, 0.21])
    th = 2 * np.pi * np.arange(K) / K + 0.3
    ev.set_state(*_gauss_field(ev, cen, a, Rg, w, th))
    ld = LumpDiagnostics(ev, lump_config({"centers": cen.tolist(), "R_ball": R, "taper": tap,
                                          "split_width": ell, "count_every": 1e9}))
    ld.sample(0.0)
    e, p = densities(ev)
    x = ev.coords
    dmin = min(np.linalg.norm(cen[i] - cen[j]) for i in range(K) for j in range(i))
    tau = ell * max(dmin, ell)
    d = [[(_wrap(x - c[k], L)).reshape([-1 if b == k else 1 for b in range(3)]) for k in range(3)] for c in cen]
    r2 = [dd[0] ** 2 + dd[1] ** 2 + dd[2] ** 2 for dd in d]
    errs = {"E": 0.0, "X": 0.0, "It": 0.0, "Ip": 0.0, "P": 0.0}
    for i in range(K):
        den = np.ones_like(r2[i])
        for j in range(K):
            if j != i and np.linalg.norm(cen[j] - cen[i]) < 2 * R + 4 * ell:
                den = den + np.exp(np.minimum((r2[i] - r2[j]) / tau, 700.0))
        W = ball_weight(np.sqrt(r2[i]), R, tap) / den
        Ei = float((e * W).sum()) * ev.dV
        Xi = cen[i] + np.array([float((e * W * d[i][k]).sum()) * ev.dV / Ei for k in range(3)])
        errs["E"] = max(errs["E"], abs(ld.series["E"][0][i] / Ei - 1))
        errs["X"] = max(errs["X"], float(np.linalg.norm(np.array(ld.series["X"][0][i]) - Xi)))
        errs["It"] = max(errs["It"], abs(ld.series["It"][0][i] - float((ev.pi ** 2 * W).sum()) * ev.dV) / Ei)
        errs["Ip"] = max(errs["Ip"], abs(ld.series["Ip"][0][i] - float((ev.phi ** 2 * W).sum()) * ev.dV) / Ei)
        errs["P"] = max(errs["P"], max(abs(ld.series["P"][0][i][k] - float((p[k] * W).sum()) * ev.dV)
                                       for k in range(3)) / Ei)
    v = np.pi ** 1.5 * Rg ** 3
    E_ex = v * (w * w * a * a * np.sin(th) ** 2 + M * M * a * a * np.cos(th) ** 2) / 2 \
        + 0.75 * (a * np.cos(th)) ** 2 * np.pi ** 1.5 * Rg
    leak = (np.array(ld.series["E"][0]) / E_ex - 1).tolist()
    out = {"K": K, "nearest_spacing": spacing, "R_ball": R, "taper": tap, "split_width": ell,
           "max_rel_err_vs_independent": errs, "split_leak_E_over_E_single_minus_1": leak,
           "exp(-d/ell)": float(np.exp(-spacing / ell)), "target": 1e-12}
    rec("k_lumps_split_machinery_K%d_d%g" % (K, spacing), out)
    assert max(errs.values()) < 1e-12


@pytest.mark.parametrize("K", [3, 5])
def test_k_lumps_count_overlap(K):
    """Lump count from coordinates: K Gaussians at nearest spacing 6 R_g count K; at spacing
    0.5 R_g (one blob) count 1."""
    N, L, a, Rg = 80, 32.0, 0.7, 1.2
    ev = Evolver(N, L, M, 0.0, 0.0, threads=4)
    out = {"K": K}
    for name, sp in (("separated", 6 * Rg), ("merged", 0.5 * Rg)):
        cen, _ = ring(K, sp / (2 * np.sin(np.pi / K)), [0.13, -0.07, 0.21])
        ev.set_state(*_gauss_field(ev, cen, a, Rg, 0.9, np.zeros(K)))
        fl = find_lumps(densities(ev)[0], ev.coords, L, ev.dx)
        out[name] = {"spacing": sp, "count": fl["count"], "count_distinct": fl["count_distinct"],
                     "saddles": [p["saddle_ratio"] for p in fl["pairs"]]}
    rec("k_lumps_count_K%d" % K, out)
    assert out["separated"]["count_distinct"] == K
    assert out["merged"]["count_distinct"] == 1


# ---------------------------------------------------------------------------
# relative phases of K = 5 lumps with stepped phases (synthetic signals)
# ---------------------------------------------------------------------------

def test_relative_phase_k5_synthetic():
    K, w, dt = 5, 0.9093, 0.04
    th = 0.4 + 2 * np.pi * np.arange(K) / K
    t = np.arange(0, 60 + 1e-9, dt)
    sig = lambda x: np.cos(w * t + x) * (1 + 0.1 * t / 60) + 0.05 * np.cos(3 * (w * t + x))
    ts = t[::5]
    zK = np.zeros((ts.size, K))
    st = {"centers": [[float(k), 0, 0] for k in range(K)],
          "series": {"t": ts.tolist(), "E": zK.tolist(), "X": np.zeros((ts.size, K, 3)).tolist(), "It": zK.tolist(),
                     "Ip": zK.tolist(), "P": np.zeros((ts.size, K, 3)).tolist(),
                     "P_total": [[0.0, 0.0, 0.0]] * ts.size, "P_half": [[]] * ts.size, "L_axes": [[]] * ts.size},
          "point": {"t": t.tolist(), "phi": np.stack([sig(x) for x in th], 1).tolist()}}
    av = summarize(st, np.arange(20, 60.1, 5.0), 20.0)
    errs = []
    for k in range(1, K):
        d = np.array([x for x in av["relative_phase"][k - 1]["dtheta"] if x is not None])
        errs.append(float(np.max(np.abs(np.angle(np.exp(1j * (d - (th[k] - th[0]))))))))
    nb = []
    for k in range(K):
        j = (k + 1) % K
        dk = 0.0 if k == 0 else np.array([x for x in av["relative_phase"][k - 1]["dtheta"] if x is not None])
        dj = 0.0 if j == 0 else np.array([x for x in av["relative_phase"][j - 1]["dtheta"] if x is not None])
        nb.append(float(np.max(np.abs(np.angle(np.exp(1j * ((dj - dk) - (th[j] - th[k]))))))))
    rec("relative_phase_k5_synthetic", {"thetas": th.tolist(), "max_abs_err_vs_lump0": max(errs),
                                        "max_abs_err_neighbour_differences": max(nb), "target": 1e-3})
    assert max(errs) < 1e-3 and max(nb) < 1e-3


# ---------------------------------------------------------------------------
# angular momentum: mirror-symmetric field gives 0 (the box-face lever fix); known L to 1e-8
# ---------------------------------------------------------------------------

def test_angular_momentum_mirror_face():
    """phi = f(x) g(y) h(z), phi_t = f2(x) g(y) h(z), g and h even and nonzero at the faces.
    L_z = int (x p_y - y p_x) = 0 by the mirror y -> -y. The plane y = -L/2 has no partner;
    with the WP-C lever -L/2 it gives L_z = (L/2) sum_face p_x dV."""
    N, L = 48, 30.0
    ev = Evolver(N, L, M, 0.0, 0.0, threads=4)
    x = ev.coords
    k = 2 * np.pi / L
    f, f2 = np.sin(k * x + 0.4), np.cos(k * x)
    g = 1 + 0.3 * np.cos(k * x)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    gy, gz = 1 + 0.3 * np.cos(k * Y), 1 + 0.3 * np.cos(k * Z)
    ev.set_state(np.sin(k * X + 0.4) * gy * gz, np.cos(k * X) * gy * gz)
    cfg = lump_config({"centers": [[0.0, 0.0, 0.0]], "count_every": 1e9,
                       "axes": [{"origin": [0, 0, 0], "dir": [0, 0, 1]}, {"origin": [0, 0, 0], "dir": [0, 1, 0]}]})
    ld = LumpDiagnostics(ev, cfg)
    ld.sample(0.0)
    _, p = densities(ev)
    scale = (L / 2) * float(np.sum(np.abs(p[0]))) * ev.dV
    Xo = [_wrap(x - 0.0, L).reshape(sh) for sh in ([-1, 1, 1], [1, -1, 1], [1, 1, -1])]
    Lz_old = float(np.sum(Xo[0] * p[1] - Xo[1] * p[0])) * ev.dV
    out = {"N": N, "L": L, "Lz_new": ld.series["L_axes"][0][0], "Ly_new": ld.series["L_axes"][0][1],
           "Lz_old_lever": Lz_old, "scale_(L/2)sum|p_x|dV": scale, "target_rel": 1e-14}
    rec("angular_momentum_mirror_face", out)
    assert abs(Lz_old) > 1e-3 * scale            # the WP-C lever gives a nonzero L_z here
    assert abs(out["Lz_new"]) < 1e-14 * scale
    assert abs(out["Ly_new"]) < 1e-14 * scale


def test_angular_momentum_group_spinning_k3():
    """Three spinning l = 1 Gaussians (each L = omega a^2 pi^{3/2} R^5 / 2 about z, P = 0) on a
    ring: L about the pattern axis through the group centroid = 3 L_single."""
    a, Rg, w, N, L = 0.5, 1.5, 0.9, 112, 56.0
    ev = Evolver(N, L, M, 0.0, 0.0, threads=4)
    cen, _ = ring(3, 12.0, [0.31, -0.23, 0.17], tilt=0.0)      # spacing 20.8: cross terms ~ e^{-d^2/4R^2} ~ 1e-21
    phi = np.zeros(ev.shape)
    pi = np.zeros(ev.shape)
    for c in cen:
        rel = lever_arms(ev.coords, L, c)
        G = np.exp(-(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2) / (2 * Rg * Rg))
        phi += a * rel[0] * G
        pi += w * a * rel[1] * G
    ev.set_state(phi, pi)
    cfg = lump_config({"centers": cen.tolist(), "R_ball": 7.0, "taper": 1.0, "count_every": 1e9,
                       "group": {"R": 22.0, "axis": [0, 0, 1]}})
    ld = LumpDiagnostics(ev, cfg)
    for _ in range(3):
        ld.sample(0.0)
    s = ld.series
    exact = 3 * w * a * a * np.pi ** 1.5 * Rg ** 5 / 2
    out = {"L_exact": exact, "L_axis_box": s["L_axis_box"][-1], "L_axis_sphere": s["L_axis_sphere"][-1],
           "rel_err_box": s["L_axis_box"][-1] / exact - 1, "rel_err_sphere": s["L_axis_sphere"][-1] / exact - 1,
           "L_perp_box": s["L_group_box"][-1][:2], "group_centroid": s["G"][-1],
           "centroid_exact": cen.mean(0).tolist(), "target": 1e-8}
    rec("angular_momentum_group_spinning_k3", out)
    assert abs(out["rel_err_box"]) < 1e-8 and abs(out["rel_err_sphere"]) < 1e-8
    assert max(abs(v) for v in out["L_perp_box"]) < 1e-8 * exact
