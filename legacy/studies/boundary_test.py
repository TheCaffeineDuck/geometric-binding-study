#!/usr/bin/env python3
"""
studies/boundary_test.py
========================
Boundary condition test: do binding energies change when L is doubled?

Runs 3 simulations at L=100, N=128 (dx=0.78125, identical to L=50, N=64):
  1. Isolated oscillon baseline (sequential, needed first)
  2. Cube checkerboard (f_cross=1.0, 12/12 cross-edges)
  3. Icosahedron ce_20 (f_cross=0.667, 20/30 cross-edges)

Compares E_bind against known L=50 values:
  Cube checkerboard: E_bind = -51.53
  Ico ce_20:         E_bind = -50.11

PASS = change < 5%, MARGINAL = 5-15%, FAIL = > 15%
"""

import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys
import json
import time
import multiprocessing as mp
from pathlib import Path

# ---------------------------------------------------------------------------
#  Parameters
# ---------------------------------------------------------------------------
N      = 128
L      = 100.0
m      = 1.0
g4     = 0.30
g6     = 0.055
dt     = 0.05
sigma  = 0.01
phi0   = 0.5
R      = 2.5
T_END  = 500.0
N_STEPS         = int(T_END / dt)   # 10000
RECORD_EVERY    = 10
PRINT_EVERY     = 1000
CHECKPOINT_EVERY = 1000

BASE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "boundary_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Known L=50 binding energies for comparison
KNOWN_L50 = {
    "cube_checkerboard": -51.53,
    "ico_ce20":          -50.11,
}

# ---------------------------------------------------------------------------
#  Geometry: Cube (edge length 6.0)
# ---------------------------------------------------------------------------
D = 3.0
CUBE_VERTS = [
    [-D, -D, -D],  # 0
    [ D, -D, -D],  # 1
    [-D,  D, -D],  # 2
    [ D,  D, -D],  # 3
    [-D, -D,  D],  # 4
    [ D, -D,  D],  # 5
    [-D,  D,  D],  # 6
    [ D,  D,  D],  # 7
]
# Polarized bipartite = "checkerboard" in paper (12/12 cross-edges, f_cross=1.0)
# A={0,3,5,6} -> phase -1 (pi), B={1,2,4,7} -> phase +1 (0)
CUBE_PHASES = [-1, 1, 1, -1, 1, -1, -1, 1]

# ---------------------------------------------------------------------------
#  Geometry: Icosahedron (edge length 6.0)
# ---------------------------------------------------------------------------
import math
phi_g = (1.0 + math.sqrt(5.0)) / 2.0

# Base vertices: all even permutations of (0, +-1, +-phi_g)
_base_verts = []
for s1 in [1, -1]:
    for s2 in [1, -1]:
        _base_verts.append([0.0,  s1 * 1.0,  s2 * phi_g])
        _base_verts.append([s1 * 1.0,  s2 * phi_g, 0.0])
        _base_verts.append([s1 * phi_g, 0.0,  s2 * 1.0])

# Scale so minimum edge = 6.0
_dists = []
for i in range(12):
    for j in range(i + 1, 12):
        dx = _base_verts[i][0] - _base_verts[j][0]
        dy = _base_verts[i][1] - _base_verts[j][1]
        dz = _base_verts[i][2] - _base_verts[j][2]
        _dists.append(math.sqrt(dx*dx + dy*dy + dz*dz))
_min_edge = min(_dists)
_scale = 6.0 / _min_edge
ICO_VERTS = [[c * _scale for c in v] for v in _base_verts]

# Icosahedron edges (30 total)
_edge_tol = _min_edge * _scale * 1.01
ICO_EDGES = []
for i in range(12):
    for j in range(i + 1, 12):
        dx = ICO_VERTS[i][0] - ICO_VERTS[j][0]
        dy = ICO_VERTS[i][1] - ICO_VERTS[j][1]
        dz = ICO_VERTS[i][2] - ICO_VERTS[j][2]
        if math.sqrt(dx*dx + dy*dy + dz*dz) < _edge_tol * 1.01:
            ICO_EDGES.append((i, j))

# Find phase config with exactly 20 cross-edges
def _count_ce(phases, edges):
    return sum(1 for (i, j) in edges if phases[i] != phases[j])

ICO_PHASES = None
for phases_int in range(4096):
    phases = [(1 if (phases_int >> k) & 1 == 0 else -1) for k in range(12)]
    if _count_ce(phases, ICO_EDGES) == 20:
        ICO_PHASES = phases
        break

if ICO_PHASES is None:
    print("ERROR: could not find ico phase config with 20 cross-edges")
    sys.exit(1)


# ---------------------------------------------------------------------------
#  Atomic write helper
# ---------------------------------------------------------------------------
def atomic_write_json(data, filepath):
    tmp_path = str(filepath) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, str(filepath))


# ---------------------------------------------------------------------------
#  Cache check
# ---------------------------------------------------------------------------
def is_valid_cache(filepath):
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath) as f:
            data = json.load(f)
        return (data.get("completed", False)
                and data.get("parameters", {}).get("N_grid") == N
                and data.get("parameters", {}).get("L") == L)
    except (json.JSONDecodeError, KeyError, ValueError):
        return False


# ---------------------------------------------------------------------------
#  Worker: run isolated baseline
# ---------------------------------------------------------------------------
def run_baseline():
    """Single oscillon at L=100, N=128. Returns E_single time series."""
    import numpy as np
    sys.path.insert(0, BASE_DIR)
    from engine.evolver import SexticEvolver, serialize_field, deserialize_field

    out_path = os.path.join(OUTPUT_DIR, "baseline_L100.json")
    ckpt_path = out_path + ".checkpoint.json"

    # Cache check
    if is_valid_cache(out_path):
        print("  CACHED: baseline_L100")
        sys.stdout.flush()
        with open(out_path) as f:
            return json.load(f)

    ev = SexticEvolver(N=N, L=L, m=m, g4=g4, g6=g6, dissipation_sigma=sigma)

    # Check for checkpoint
    resume_from = None
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path) as f:
                resume_from = json.load(f)
            print("  Resuming baseline from step %d (t=%.1f)" % (
                resume_from["completed_steps"], resume_from["t"]))
            sys.stdout.flush()
        except (json.JSONDecodeError, KeyError):
            resume_from = None

    # Initial conditions (single Gaussian at origin)
    if resume_from is None:
        phi_init = np.zeros((N, N, N))
        r2 = ev.X**2 + ev.Y**2 + ev.Z**2
        phi_init = phi0 * np.exp(-r2 / (2.0 * R**2))
        phi_dot_init = np.zeros_like(phi_init)
        ev.set_initial_conditions(phi_init, phi_dot_init)

    # Checkpoint callback
    def checkpoint_cb(state_dict):
        tmp = ckpt_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state_dict, f, separators=(",", ":"))
        os.replace(tmp, ckpt_path)

    state = ev.evolve(
        dt=dt,
        n_steps=N_STEPS,
        record_every=RECORD_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_callback=checkpoint_cb,
        resume_from=resume_from,
        print_every=PRINT_EVERY,
        tag="baseline_L100",
    )

    # Build output (no field arrays)
    E0 = state["E0"]
    E_final = state["time_series"]["E_total"][-1]
    drift = abs(E_final - E0) / (abs(E0) + 1e-30)

    result = {
        "completed": True,
        "config_name": "baseline_L100",
        "parameters": {"N_grid": N, "L": L, "m": m, "g4": g4, "g6": g6,
                        "dt": dt, "sigma": sigma, "phi0": phi0, "R": R,
                        "T_end": T_END},
        "t_series": state["time_series"]["times"],
        "E_series": state["time_series"]["E_total"],
        "max_phi_series": state["time_series"]["max_amplitude"],
        "E0": E0,
        "E_final": E_final,
        "energy_drift": drift,
        "wall_time": state["wall_elapsed"],
    }

    atomic_write_json(result, out_path)
    print("  Saved -> %s" % out_path)
    sys.stdout.flush()

    # Remove checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return result


# ---------------------------------------------------------------------------
#  Worker: run cluster simulation (for Pool)
# ---------------------------------------------------------------------------
def run_cluster(config):
    """Run a cluster simulation. Designed for multiprocessing.Pool."""
    import numpy as np
    sys.path.insert(0, BASE_DIR)
    from engine.evolver import SexticEvolver, serialize_field, deserialize_field
    from scipy.interpolate import interp1d

    name = config["name"]
    verts = config["verts"]
    phases = config["phases"]
    n_osc = len(verts)
    baseline_path = config["baseline_path"]

    out_path = os.path.join(OUTPUT_DIR, "%s_L100.json" % name)
    ckpt_path = out_path + ".checkpoint.json"

    # Cache check
    if is_valid_cache(out_path):
        print("  CACHED: %s" % name)
        sys.stdout.flush()
        with open(out_path) as f:
            return json.load(f)

    # Load baseline for E_bind computation
    with open(baseline_path) as f:
        ref = json.load(f)
    E_single_interp = interp1d(ref["t_series"], ref["E_series"],
                                kind="linear", fill_value="extrapolate")

    ev = SexticEvolver(N=N, L=L, m=m, g4=g4, g6=g6, dissipation_sigma=sigma)

    # Check for checkpoint
    resume_from = None
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path) as f:
                resume_from = json.load(f)
            print("  [%s] Resuming from step %d (t=%.1f)" % (
                name, resume_from["completed_steps"], resume_from["t"]))
            sys.stdout.flush()
        except (json.JSONDecodeError, KeyError):
            resume_from = None

    # Initial conditions: sum of Gaussians at vertex positions
    if resume_from is None:
        phi_init = np.zeros((N, N, N))
        for idx in range(n_osc):
            A = phases[idx]
            pos = verts[idx]
            dx_ = ev.X - pos[0]
            dy_ = ev.Y - pos[1]
            dz_ = ev.Z - pos[2]
            r2 = dx_**2 + dy_**2 + dz_**2
            phi_init += A * phi0 * np.exp(-r2 / (2.0 * R**2))
        phi_dot_init = np.zeros_like(phi_init)
        ev.set_initial_conditions(phi_init, phi_dot_init)

    # Checkpoint callback
    def checkpoint_cb(state_dict):
        tmp = ckpt_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state_dict, f, separators=(",", ":"))
        os.replace(tmp, ckpt_path)

    state = ev.evolve(
        dt=dt,
        n_steps=N_STEPS,
        record_every=RECORD_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_callback=checkpoint_cb,
        resume_from=resume_from,
        print_every=PRINT_EVERY,
        tag=name,
    )

    # Compute E_bind series
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

    result = {
        "completed": True,
        "config_name": name,
        "parameters": {"N_grid": N, "L": L, "m": m, "g4": g4, "g6": g6,
                        "dt": dt, "sigma": sigma, "phi0": phi0, "R": R,
                        "T_end": T_END, "d": 6.0, "n_oscillons": n_osc},
        "phases": phases,
        "t_series": times,
        "E_series": E_total,
        "E_bind_series": E_bind_series,
        "max_phi_series": state["time_series"]["max_amplitude"],
        "E0": E0,
        "E_final": E_final,
        "E_bind_final": E_bind_final,
        "energy_drift": drift,
        "wall_time": state["wall_elapsed"],
    }

    atomic_write_json(result, out_path)
    print("  Saved -> %s" % out_path)
    sys.stdout.flush()

    # Remove checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return result


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    wall_start = time.perf_counter()

    print("=== BOUNDARY CONDITION TEST ===")
    print("Box size: L=%g (standard: L=50)" % L)
    print("Grid: N=%d, dx=%.5f" % (N, L / N))
    print("")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    #  Step 1: Isolated baseline
    # ------------------------------------------------------------------
    print("Step 1: Isolated baseline...")
    sys.stdout.flush()
    t1 = time.perf_counter()
    baseline = run_baseline()
    t1_wall = time.perf_counter() - t1

    print("  E_single(T=%g) = %.3f" % (T_END, baseline["E_final"]))
    print("  Energy drift: %.3f%%" % (baseline["energy_drift"] * 100))
    print("  Wall time: %.0fs" % t1_wall)
    print("")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    #  Step 2: Cluster simulations (parallel, Pool(2))
    # ------------------------------------------------------------------
    print("Step 2: Cluster simulations (parallel)...")
    sys.stdout.flush()
    t2 = time.perf_counter()

    baseline_path = os.path.join(OUTPUT_DIR, "baseline_L100.json")

    cluster_configs = [
        {
            "name": "cube_checkerboard",
            "verts": CUBE_VERTS,
            "phases": CUBE_PHASES,
            "baseline_path": baseline_path,
        },
        {
            "name": "ico_ce20",
            "verts": ICO_VERTS,
            "phases": ICO_PHASES,
            "baseline_path": baseline_path,
        },
    ]

    with mp.Pool(2) as pool:
        cluster_results = pool.map(run_cluster, cluster_configs)

    t2_wall = time.perf_counter() - t2

    cube_result = cluster_results[0]
    ico_result = cluster_results[1]

    print("  Cube checkerboard: E_total = %.2f, E_bind = %.2f" % (
        cube_result["E_final"], cube_result["E_bind_final"]))
    print("  Ico ce_20:         E_total = %.2f, E_bind = %.2f" % (
        ico_result["E_final"], ico_result["E_bind_final"]))
    print("  Wall time: %.0fs" % t2_wall)
    print("")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    #  Step 3: Comparison
    # ------------------------------------------------------------------
    def verdict(pct_change):
        if abs(pct_change) < 5:
            return "PASS"
        elif abs(pct_change) < 15:
            return "MARGINAL"
        else:
            return "FAIL"

    cube_l50 = KNOWN_L50["cube_checkerboard"]
    ico_l50 = KNOWN_L50["ico_ce20"]
    cube_l100 = cube_result["E_bind_final"]
    ico_l100 = ico_result["E_bind_final"]

    cube_change = (cube_l100 - cube_l50) / abs(cube_l50) * 100
    ico_change = (ico_l100 - ico_l50) / abs(ico_l50) * 100

    cube_v = verdict(cube_change)
    ico_v = verdict(ico_change)

    overall = "PASS" if (cube_v == "PASS" and ico_v == "PASS") else (
        "MARGINAL" if (cube_v != "FAIL" and ico_v != "FAIL") else "FAIL")

    print("=== COMPARISON ===")
    print("%-19s| %-13s| %-14s| %-8s| %s" % (
        "Config", "E_bind(L=50)", "E_bind(L=100)", "Change", "Verdict"))
    print("%-19s| %-13.2f| %-14.2f| %-7.1f%%| %s" % (
        "Cube checkerboard", cube_l50, cube_l100, cube_change, cube_v))
    print("%-19s| %-13.2f| %-14.2f| %-7.1f%%| %s" % (
        "Ico ce_20", ico_l50, ico_l100, ico_change, ico_v))
    print("")
    print("PASS = change < 5%%")
    print("MARGINAL = change 5-15%%")
    print("FAIL = change >= 15%%")
    print("")
    print("Overall: %s" % overall)
    print("")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    #  Step 4: Save comparison JSON
    # ------------------------------------------------------------------
    comparison = {
        "test": "boundary_condition",
        "description": "L=100 vs L=50, same dx=0.78125",
        "parameters": {"N_grid": N, "L": L, "L_standard": 50.0,
                        "dx": L / N, "T_end": T_END},
        "results": {
            "cube_checkerboard": {
                "E_bind_L50": cube_l50,
                "E_bind_L100": cube_l100,
                "pct_change": cube_change,
                "verdict": cube_v,
            },
            "ico_ce20": {
                "E_bind_L50": ico_l50,
                "E_bind_L100": ico_l100,
                "pct_change": ico_change,
                "verdict": ico_v,
            },
        },
        "overall_verdict": overall,
        "wall_time_total": time.perf_counter() - wall_start,
        "wall_time_baseline": t1_wall,
        "wall_time_clusters": t2_wall,
    }

    atomic_write_json(comparison, os.path.join(OUTPUT_DIR, "comparison.json"))
    print("Results saved to %s" % os.path.abspath(OUTPUT_DIR))

    # ------------------------------------------------------------------
    #  Cleanup checkpoints and temp files
    # ------------------------------------------------------------------
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

    print("Total wall time: %.0fs" % (time.perf_counter() - wall_start))


if __name__ == "__main__":
    main()
