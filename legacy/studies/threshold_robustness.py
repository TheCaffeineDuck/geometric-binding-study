#!/usr/bin/env python3
"""
studies/threshold_robustness.py
================================
Near-Threshold Robustness Test

Tests whether marginal (small |E_bind|) configurations preserve their binding
energy SIGN under two perturbations:
  1. Large box (L=100, N=128) -- boundary effects
  2. No dissipation (sigma=0) -- dissipation effects

4 near-threshold configs x 3 conditions = 12 cluster sims + 3 baselines.

Decision criteria: PASS if all 8 non-standard signs match standard signs.
"""

import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys
import json
import time
import math
import multiprocessing as mp
from pathlib import Path

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, BASE_DIR)

# ---------------------------------------------------------------------------
#  Physics parameters (constant across all conditions)
# ---------------------------------------------------------------------------
PHI0      = 0.5
R_GAUSS   = 2.5
M         = 1.0
G4        = 0.30
G6        = 0.055
DT        = 0.05
T_END     = 500.0
N_STEPS   = int(T_END / DT)   # 10000
D_EDGE    = 6.0
OMEGA_EFF = 1.113

RECORD_EVERY     = 10
PRINT_EVERY      = 1000
CHECKPOINT_EVERY = 1000

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "threshold_robustness")

# ---------------------------------------------------------------------------
#  Conditions
# ---------------------------------------------------------------------------
CONDITIONS = {
    "standard":       {"L": 50.0,  "N": 64,  "sigma": 0.01},
    "large_box":      {"L": 100.0, "N": 128, "sigma": 0.01},
    "no_dissipation": {"L": 50.0,  "N": 64,  "sigma": 0.00},
}

# Expected E_bind from standard condition (for sanity check)
EXPECTED_STANDARD = {
    "tet_single_flip": -0.041,
    "cube_balanced":    -3.86,
    "oct_face3":        -2.90,
    "ico_ce15_A":       +3.56,
}


# =============================================================================
#  GEOMETRY DEFINITIONS
# =============================================================================

def make_tetrahedron():
    """Regular tetrahedron, edge length = 6.0. 4 vertices, 6 edges (K4)."""
    import numpy as np
    base = np.array([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
    ], dtype=float)
    current_edge = 2.0 * np.sqrt(2.0)
    scale = D_EDGE / current_edge
    verts = base * scale
    edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    return verts.tolist(), edges


def make_cube():
    """Cube, edge length = 6.0. 8 vertices, 12 edges."""
    D = D_EDGE / 2.0  # half-edge = 3.0
    verts = [
        [-D, -D, -D],  # 0
        [ D, -D, -D],  # 1
        [-D,  D, -D],  # 2
        [ D,  D, -D],  # 3
        [-D, -D,  D],  # 4
        [ D, -D,  D],  # 5
        [-D,  D,  D],  # 6
        [ D,  D,  D],  # 7
    ]
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            # Two cube vertices share an edge iff they differ in exactly 1 coordinate
            diff = sum(1 for k in range(3) if verts[i][k] != verts[j][k])
            if diff == 1:
                edges.append((i, j))
    return verts, edges


def make_octahedron():
    """Regular octahedron, edge length = 6.0. 6 vertices, 12 edges."""
    d = D_EDGE
    verts = [
        [ d/2,   0,   0],  # 0
        [   0, d/2,   0],  # 1
        [   0,   0, d/2],  # 2
        [-d/2,   0,   0],  # 3
        [   0,-d/2,   0],  # 4
        [   0,   0,-d/2],  # 5
    ]
    edge_len = d / math.sqrt(2)
    tol = edge_len * 1.02
    edges = []
    for i in range(6):
        for j in range(i + 1, 6):
            dist = math.sqrt(sum((verts[i][k] - verts[j][k])**2 for k in range(3)))
            if dist < tol:
                edges.append((i, j))
    return verts, edges


def make_icosahedron():
    """Regular icosahedron, edge length = 6.0. 12 vertices, 30 edges."""
    phi_g = (1.0 + math.sqrt(5.0)) / 2.0
    base = []
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            base.append([0.0,  s1 * 1.0,  s2 * phi_g])
            base.append([s1 * 1.0,  s2 * phi_g, 0.0])
            base.append([s1 * phi_g, 0.0,  s2 * 1.0])

    # Find min edge length and scale
    dists = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = math.sqrt(sum((base[i][k] - base[j][k])**2 for k in range(3)))
            dists.append(d)
    min_edge = min(dists)
    scale = D_EDGE / min_edge
    verts = [[c * scale for c in v] for v in base]

    # Find edges
    tol = D_EDGE * 1.02
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = math.sqrt(sum((verts[i][k] - verts[j][k])**2 for k in range(3)))
            if d < tol:
                edges.append((i, j))
    return verts, edges


def count_cross_edges(phases, edges):
    return sum(1 for (i, j) in edges if phases[i] != phases[j])


# =============================================================================
#  CONFIG DEFINITIONS
# =============================================================================

TET_VERTS, TET_EDGES = make_tetrahedron()
CUBE_VERTS, CUBE_EDGES = make_cube()
OCT_VERTS, OCT_EDGES = make_octahedron()
ICO_VERTS, ICO_EDGES = make_icosahedron()

# Phase assignments (using +1/-1 convention: +1 = phase 0, -1 = phase pi)
CONFIGS = {
    "tet_single_flip": {
        "verts": TET_VERTS,
        "edges": TET_EDGES,
        "phases": [-1, +1, +1, +1],   # 3 CE / 6 = 0.500
        "n_osc": 4,
        "geometry": "tetrahedron",
    },
    "cube_balanced": {
        "verts": CUBE_VERTS,
        "edges": CUBE_EDGES,
        "phases": [-1, -1, -1, +1, +1, +1, -1, +1],   # 6 CE / 12 = 0.500
        "n_osc": 8,
        "geometry": "cube",
    },
    "oct_face3": {
        "verts": OCT_VERTS,
        "edges": OCT_EDGES,
        "phases": [+1, +1, +1, -1, -1, -1],   # 6 CE / 12 = 0.500
        "n_osc": 6,
        "geometry": "octahedron",
    },
    "ico_ce15_A": {
        "verts": ICO_VERTS,
        "edges": ICO_EDGES,
        "phases": [-1, +1, +1, -1, -1, +1, +1, +1, +1, +1, +1, +1],  # 15 CE / 30 = 0.500
        "n_osc": 12,
        "geometry": "icosahedron",
    },
}

# Verify cross-edge counts
for name, cfg in CONFIGS.items():
    ce = count_cross_edges(cfg["phases"], cfg["edges"])
    n_total = len(cfg["edges"])
    f_cross = ce / n_total
    assert abs(f_cross - 0.5) < 0.01, (
        "%s: expected f_cross=0.500, got %.3f (%d/%d)" % (name, f_cross, ce, n_total))


# =============================================================================
#  ATOMIC WRITE
# =============================================================================

def atomic_write_json(data, filepath):
    tmp_path = str(filepath) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, str(filepath))


# =============================================================================
#  CACHE CHECK
# =============================================================================

def is_valid_cache(filepath, required_N, required_L, required_sigma):
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath) as f:
            data = json.load(f)
        if not data.get("completed", False):
            return False
        params = data.get("parameters", {})
        if params.get("N_grid") != required_N:
            return False
        if abs(params.get("L", -1) - required_L) > 0.01:
            return False
        if abs(params.get("sigma", -1) - required_sigma) > 0.001:
            return False
        return True
    except (json.JSONDecodeError, KeyError, ValueError):
        return False


# =============================================================================
#  BASELINE WORKER
# =============================================================================

def run_baseline(cond_name):
    """Run isolated single oscillon baseline for a given condition."""
    import numpy as np
    from engine.evolver import SexticEvolver

    cond = CONDITIONS[cond_name]
    N_grid = cond["N"]
    L = cond["L"]
    sigma = cond["sigma"]

    label = "baseline_%s" % cond_name
    out_path = os.path.join(OUTPUT_DIR, "%s.json" % label)
    ckpt_path = out_path + ".checkpoint.json"

    # Cache check
    if is_valid_cache(out_path, N_grid, L, sigma):
        print("  CACHED: %s" % label)
        sys.stdout.flush()
        with open(out_path) as f:
            return json.load(f)

    print("  Running %s (N=%d, L=%.0f, sigma=%.2f)..." % (label, N_grid, L, sigma))
    sys.stdout.flush()

    ev = SexticEvolver(N=N_grid, L=L, m=M, g4=G4, g6=G6, dissipation_sigma=sigma)

    # Check for checkpoint resume
    resume_from = None
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path) as f:
                resume_from = json.load(f)
            print("  RESUMING %s from step %d" % (label, resume_from["completed_steps"]))
            sys.stdout.flush()
        except (json.JSONDecodeError, KeyError):
            resume_from = None

    if resume_from is None:
        phi_init = PHI0 * np.exp(-(ev.X**2 + ev.Y**2 + ev.Z**2) / (2.0 * R_GAUSS**2))
        phi_dot_init = np.zeros_like(phi_init)
        ev.set_initial_conditions(phi_init, phi_dot_init)

    def checkpoint_cb(state_dict):
        tmp = ckpt_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state_dict, f, separators=(",", ":"))
        os.replace(tmp, ckpt_path)

    state = ev.evolve(
        dt=DT, n_steps=N_STEPS,
        record_every=RECORD_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_callback=checkpoint_cb,
        resume_from=resume_from,
        print_every=PRINT_EVERY,
        tag=label,
    )

    E0 = state["E0"]
    E_final = state["time_series"]["E_total"][-1]
    drift = abs(E_final - E0) / (abs(E0) + 1e-30)

    result = {
        "completed": True,
        "config_name": label,
        "parameters": {"N_grid": N_grid, "L": L, "m": M, "g4": G4, "g6": G6,
                        "dt": DT, "sigma": sigma, "phi0": PHI0, "R": R_GAUSS,
                        "T_end": T_END},
        "t_series": state["time_series"]["times"],
        "E_series": state["time_series"]["E_total"],
        "E0": E0,
        "E_final": E_final,
        "energy_drift": drift,
        "wall_time": state["wall_elapsed"],
    }

    atomic_write_json(result, out_path)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print("  %s: E_single = %.3f (drift %.2e)" % (label, E_final, drift))
    sys.stdout.flush()
    return result


# =============================================================================
#  CLUSTER WORKER
# =============================================================================

def run_cluster(args):
    """Run a cluster simulation. Designed for multiprocessing.Pool."""
    import numpy as np
    from engine.evolver import SexticEvolver
    from scipy.interpolate import interp1d

    config_name, cond_name = args
    cfg = CONFIGS[config_name]
    cond = CONDITIONS[cond_name]

    N_grid = cond["N"]
    L = cond["L"]
    sigma = cond["sigma"]
    verts = cfg["verts"]
    phases = cfg["phases"]
    n_osc = cfg["n_osc"]

    label = "%s_%s" % (config_name, cond_name)
    out_path = os.path.join(OUTPUT_DIR, "%s.json" % label)
    ckpt_path = out_path + ".checkpoint.json"

    # Cache check
    if is_valid_cache(out_path, N_grid, L, sigma):
        print("  CACHED: %s" % label)
        sys.stdout.flush()
        with open(out_path) as f:
            return json.load(f)

    # Load baseline for this condition
    baseline_path = os.path.join(OUTPUT_DIR, "baseline_%s.json" % cond_name)
    with open(baseline_path) as f:
        ref = json.load(f)
    E_single_interp = interp1d(ref["t_series"], ref["E_series"],
                                kind="linear", fill_value="extrapolate")

    ev = SexticEvolver(N=N_grid, L=L, m=M, g4=G4, g6=G6, dissipation_sigma=sigma)

    # Check for checkpoint resume
    resume_from = None
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path) as f:
                resume_from = json.load(f)
            print("  [%s] Resuming from step %d" % (label, resume_from["completed_steps"]))
            sys.stdout.flush()
        except (json.JSONDecodeError, KeyError):
            resume_from = None

    if resume_from is None:
        phi_init = np.zeros((N_grid, N_grid, N_grid))
        for idx in range(n_osc):
            A = float(phases[idx])
            pos = verts[idx]
            dx_ = ev.X - pos[0]
            dy_ = ev.Y - pos[1]
            dz_ = ev.Z - pos[2]
            r2 = dx_**2 + dy_**2 + dz_**2
            phi_init += A * PHI0 * np.exp(-r2 / (2.0 * R_GAUSS**2))
        phi_dot_init = np.zeros_like(phi_init)
        ev.set_initial_conditions(phi_init, phi_dot_init)

    def checkpoint_cb(state_dict):
        tmp = ckpt_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state_dict, f, separators=(",", ":"))
        os.replace(tmp, ckpt_path)

    state = ev.evolve(
        dt=DT, n_steps=N_STEPS,
        record_every=RECORD_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_callback=checkpoint_cb,
        resume_from=resume_from,
        print_every=PRINT_EVERY,
        tag=label,
    )

    # Compute binding energy series
    times = state["time_series"]["times"]
    E_total = state["time_series"]["E_total"]
    E_bind_series = [
        E_total[i] - n_osc * float(E_single_interp(times[i]))
        for i in range(len(times))
    ]

    E0 = state["E0"]
    E_final = E_total[-1]
    E_bind_final = E_bind_series[-1]
    drift = abs(E_final - E0) / (abs(E0) + 1e-30)

    ce = count_cross_edges(phases, cfg["edges"])
    n_total = len(cfg["edges"])

    result = {
        "completed": True,
        "config_name": config_name,
        "condition": cond_name,
        "geometry": cfg["geometry"],
        "parameters": {"N_grid": N_grid, "L": L, "m": M, "g4": G4, "g6": G6,
                        "dt": DT, "sigma": sigma, "phi0": PHI0, "R": R_GAUSS,
                        "T_end": T_END, "d": D_EDGE, "n_oscillons": n_osc},
        "phases": phases,
        "cross_edges": ce,
        "total_edges": n_total,
        "f_cross": ce / n_total,
        "t_series": times,
        "E_series": E_total,
        "E_bind_series": E_bind_series,
        "E0": E0,
        "E_final": E_final,
        "E_bind_final": E_bind_final,
        "energy_drift": drift,
        "wall_time": state["wall_elapsed"],
    }

    atomic_write_json(result, out_path)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print("  [%s] E_bind = %.3f (drift %.2e, %.0fs)" % (
        label, E_bind_final, drift, state["wall_elapsed"]))
    sys.stdout.flush()
    return result


# =============================================================================
#  MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wall_start = time.perf_counter()

    print("=== NEAR-THRESHOLD ROBUSTNESS TEST ===")
    print("")
    sys.stdout.flush()

    config_names = list(CONFIGS.keys())

    # ==================================================================
    #  Step 1: Baselines
    # ==================================================================
    print("Step 1: Baselines")
    sys.stdout.flush()

    baselines = {}
    for cond_name in ["standard", "large_box", "no_dissipation"]:
        baselines[cond_name] = run_baseline(cond_name)

    print("")
    print("  Standard (L=50, sigma=0.01):   E_single = %.3f" % baselines["standard"]["E_final"])
    print("  Large box (L=100, sigma=0.01): E_single = %.3f" % baselines["large_box"]["E_final"])
    print("  No dissipation (L=50, sigma=0): E_single = %.3f" % baselines["no_dissipation"]["E_final"])
    print("")
    sys.stdout.flush()

    # ==================================================================
    #  Step 2: Standard condition (sanity check) -- Pool(4)
    # ==================================================================
    print("Step 2: Standard condition (sanity check)")
    sys.stdout.flush()

    std_args = [(name, "standard") for name in config_names]
    with mp.Pool(4) as pool:
        std_results_list = pool.map(run_cluster, std_args)
    std_results = {r["config_name"]: r for r in std_results_list}

    print("")
    for name in config_names:
        r = std_results[name]
        exp = EXPECTED_STANDARD[name]
        print("  %-18s E_bind = %+.3f (expected: %+.3f)" % (
            name + ":", r["E_bind_final"], exp))
    print("")
    sys.stdout.flush()

    # ==================================================================
    #  Step 3: Large box (L=100, N=128) -- Pool(2) for memory
    # ==================================================================
    print("Step 3: Large box (L=100)")
    sys.stdout.flush()

    lb_args = [(name, "large_box") for name in config_names]
    with mp.Pool(2) as pool:
        lb_results_list = pool.map(run_cluster, lb_args)
    lb_results = {r["config_name"]: r for r in lb_results_list}

    print("")
    for name in config_names:
        r = lb_results[name]
        s = std_results[name]
        std_val = s["E_bind_final"]
        lb_val = r["E_bind_final"]
        if abs(std_val) > 1e-6:
            pct = (lb_val - std_val) / abs(std_val) * 100
        else:
            pct = 0.0
        sign_match = "SAME" if (std_val * lb_val > 0 or (abs(std_val) < 1e-6 and abs(lb_val) < 1e-6)) else "FLIPPED"
        print("  %-18s E_bind = %+.3f  [change: %.1f%%] [sign: %s]" % (
            name + ":", lb_val, pct, sign_match))
    print("")
    sys.stdout.flush()

    # ==================================================================
    #  Step 4: No dissipation (sigma=0) -- Pool(4)
    # ==================================================================
    print("Step 4: No dissipation (sigma=0)")
    sys.stdout.flush()

    nd_args = [(name, "no_dissipation") for name in config_names]
    with mp.Pool(4) as pool:
        nd_results_list = pool.map(run_cluster, nd_args)
    nd_results = {r["config_name"]: r for r in nd_results_list}

    print("")
    for name in config_names:
        r = nd_results[name]
        s = std_results[name]
        std_val = s["E_bind_final"]
        nd_val = r["E_bind_final"]
        if abs(std_val) > 1e-6:
            pct = (nd_val - std_val) / abs(std_val) * 100
        else:
            pct = 0.0
        sign_match = "SAME" if (std_val * nd_val > 0 or (abs(std_val) < 1e-6 and abs(nd_val) < 1e-6)) else "FLIPPED"
        print("  %-18s E_bind = %+.3f  [change: %.1f%%] [sign: %s]" % (
            name + ":", nd_val, pct, sign_match))
    print("")
    sys.stdout.flush()

    # ==================================================================
    #  Verdict
    # ==================================================================
    signs_preserved = 0
    signs_flipped = 0
    flipped_details = []

    for cond_name, results in [("large_box", lb_results), ("no_dissipation", nd_results)]:
        for name in config_names:
            std_val = std_results[name]["E_bind_final"]
            test_val = results[name]["E_bind_final"]
            if std_val * test_val > 0 or (abs(std_val) < 1e-6 and abs(test_val) < 1e-6):
                signs_preserved += 1
            else:
                signs_flipped += 1
                flipped_details.append((name, cond_name, std_val, test_val))

    print("=== VERDICT ===")
    print("Signs preserved: %d/8" % signs_preserved)
    print("Signs flipped:   %d/8" % signs_flipped)
    print("")

    if signs_flipped == 0:
        verdict = "PASS"
        print("PASS: All signs preserved. Selection rule is robust at threshold.")
    elif signs_flipped <= 2:
        # Check if all flips are on configs with |E_bind| < 1.0
        all_marginal = all(abs(d[2]) < 1.0 for d in flipped_details)
        if all_marginal:
            verdict = "MARGINAL"
            print("MARGINAL: 1-2 signs flipped, all on configs with |E_bind| < 1.0.")
        else:
            verdict = "FAIL"
            print("FAIL: Sign flipped on config with |E_bind| > 1.0.")
        for name, cond, sv, tv in flipped_details:
            print("  Flipped: %s (%s) std=%+.3f -> %+.3f" % (name, cond, sv, tv))
    else:
        verdict = "FAIL"
        print("FAIL: 3+ signs flipped. Selection rule is fragile at threshold.")
        for name, cond, sv, tv in flipped_details:
            print("  Flipped: %s (%s) std=%+.3f -> %+.3f" % (name, cond, sv, tv))

    print("")
    sys.stdout.flush()

    # ==================================================================
    #  Save comparison JSON
    # ==================================================================
    comparison = {
        "test": "near_threshold_robustness",
        "description": "Tests marginal E_bind configs under L=100 and sigma=0 perturbations",
        "conditions": {k: dict(v) for k, v in CONDITIONS.items()},
        "baselines": {
            cond: {"E_single": baselines[cond]["E_final"],
                   "energy_drift": baselines[cond]["energy_drift"]}
            for cond in CONDITIONS
        },
        "results": {},
        "verdict": verdict,
        "signs_preserved": signs_preserved,
        "signs_flipped": signs_flipped,
        "flipped_details": [
            {"config": d[0], "condition": d[1],
             "E_bind_standard": d[2], "E_bind_test": d[3]}
            for d in flipped_details
        ],
        "wall_time_total": time.perf_counter() - wall_start,
    }

    for name in config_names:
        comparison["results"][name] = {
            "standard": {
                "E_bind": std_results[name]["E_bind_final"],
                "energy_drift": std_results[name]["energy_drift"],
            },
            "large_box": {
                "E_bind": lb_results[name]["E_bind_final"],
                "energy_drift": lb_results[name]["energy_drift"],
                "pct_change": ((lb_results[name]["E_bind_final"] - std_results[name]["E_bind_final"])
                               / abs(std_results[name]["E_bind_final"]) * 100
                               if abs(std_results[name]["E_bind_final"]) > 1e-6 else 0.0),
                "sign_preserved": (lb_results[name]["E_bind_final"] * std_results[name]["E_bind_final"] > 0),
            },
            "no_dissipation": {
                "E_bind": nd_results[name]["E_bind_final"],
                "energy_drift": nd_results[name]["energy_drift"],
                "pct_change": ((nd_results[name]["E_bind_final"] - std_results[name]["E_bind_final"])
                               / abs(std_results[name]["E_bind_final"]) * 100
                               if abs(std_results[name]["E_bind_final"]) > 1e-6 else 0.0),
                "sign_preserved": (nd_results[name]["E_bind_final"] * std_results[name]["E_bind_final"] > 0),
            },
        }

    atomic_write_json(comparison, os.path.join(OUTPUT_DIR, "comparison.json"))
    print("Results saved to %s" % os.path.abspath(OUTPUT_DIR))

    # ==================================================================
    #  Cleanup
    # ==================================================================
    import glob
    checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "*.checkpoint*"))
    temps = glob.glob(os.path.join(OUTPUT_DIR, "*.tmp"))
    total_freed = 0
    for f in checkpoints + temps:
        sz = os.path.getsize(f)
        os.remove(f)
        total_freed += sz
    if checkpoints or temps:
        print("Cleanup: removed %d checkpoints, %d temp files, freed %.1f MB" % (
            len(checkpoints), len(temps), total_freed / 1e6))

    total_wall = time.perf_counter() - wall_start
    print("Total wall time: %.1f min" % (total_wall / 60.0))


if __name__ == "__main__":
    main()
