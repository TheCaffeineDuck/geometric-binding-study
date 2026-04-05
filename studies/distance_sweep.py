#!/usr/bin/env python3
"""
studies/distance_sweep.py
=========================
Pairwise interaction distance sweep: E_pair(d) for same-phase and anti-phase
oscillon pairs at 13 separations.

Outputs per-run JSONs to outputs/distance_sweep/ and a summary JSON with
Gaussian envelope fit.
"""

import os
import sys
import json
import time
import glob
import multiprocessing

os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- Parameters (Set B) ---
N = 64
L = 50.0
m = 1.0
g4 = 0.30
g6 = 0.055
phi0 = 0.5
R = 2.5
omega_eff = 1.113
dt = 0.05
sigma = 0.01
T_final = 500.0
N_STEPS = int(T_final / dt)  # 10000
RECORD_EVERY = 10
PRINT_EVERY = 2000
CHECKPOINT_EVERY = 1000

SEPARATIONS = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0]

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "distance_sweep")

N_WORKERS = 4

# Known reference values at d=6.0
REF_E_SAME = 5.172
REF_E_ANTI = -5.188
REF_TOL = 0.01  # 1%


def gaussian_oscillon(X, Y, Z, cx, cy, cz, amplitude, radius):
    """Gaussian bubble centered at (cx, cy, cz)."""
    r2 = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
    return amplitude * np.exp(-r2 / (2.0 * radius**2))


def run_single(args):
    """Run one simulation (single or pair). Returns result dict."""
    label, config = args

    import numpy as _np
    from engine.evolver import SexticEvolver

    output_path = os.path.join(OUT_DIR, f"{label}.json")
    checkpoint_path = os.path.join(OUT_DIR, f"{label}.checkpoint.json")

    # Cache check
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
            if existing.get("completed", False):
                print(f"  CACHED: {label}")
                sys.stdout.flush()
                return {
                    "label": label,
                    "E_final": existing["time_series"]["E_total"][-1],
                    "times": existing["time_series"]["times"],
                    "energies": existing["time_series"]["E_total"],
                }
        except (json.JSONDecodeError, KeyError):
            pass

    # Resume check
    resume_from = None
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path) as f:
                resume_from = json.load(f)
            print(f"  RESUMING: {label} from step {resume_from['completed_steps']}")
            sys.stdout.flush()
        except (json.JSONDecodeError, KeyError):
            resume_from = None

    print(f"  START: {label}")
    sys.stdout.flush()

    ev = SexticEvolver(N, L, m, g4, g6, dissipation_sigma=sigma)

    positions = config["positions"]
    phi = _np.zeros((N, N, N))
    phi_dot_field = _np.zeros((N, N, N))

    for (cx, cy, cz, delta_phi) in positions:
        envelope = gaussian_oscillon(ev.X, ev.Y, ev.Z, cx, cy, cz, phi0, R)
        phi += _np.cos(delta_phi) * envelope
        phi_dot_field += -omega_eff * _np.sin(delta_phi) * envelope

    ev.set_initial_conditions(phi, phi_dot_field)

    def checkpoint_cb(state):
        os.makedirs(OUT_DIR, exist_ok=True)
        tmp = checkpoint_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, checkpoint_path)

    state = ev.evolve(
        dt=dt,
        n_steps=N_STEPS,
        record_every=RECORD_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_callback=checkpoint_cb,
        resume_from=resume_from,
        print_every=PRINT_EVERY,
        tag=label,
    )

    wall = state["wall_elapsed"]
    times = state["time_series"]["times"]
    energies = state["time_series"]["E_total"]
    E_final = energies[-1]

    print(f"  DONE: {label}  E_final={E_final:.6e}  wall={wall:.1f}s")
    sys.stdout.flush()

    # Save final output (no field arrays)
    result_json = {
        "completed": True,
        "label": label,
        "parameters": {
            "N": N, "L": L, "m": m, "g4": g4, "g6": g6,
            "phi0": phi0, "R": R, "omega_eff": omega_eff,
            "dt": dt, "sigma": sigma, "T_final": T_final,
        },
        "config": config,
        "time_series": {
            "times": times,
            "E_total": energies,
            "max_amplitude": state["time_series"]["max_amplitude"],
        },
        "E0": state["E0"],
        "wall_elapsed": wall,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = output_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result_json, f, indent=2)
    os.replace(tmp, output_path)

    # Remove checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return {
        "label": label,
        "E_final": E_final,
        "times": times,
        "energies": energies,
    }


def build_tasks(separations, include_single=True):
    """Build task list for pool.map."""
    tasks = []

    if include_single:
        tasks.append(("single_baseline", {
            "type": "single",
            "positions": [(0.0, 0.0, 0.0, 0.0)],
        }))

    for d in separations:
        # Same phase: delta_phi = 0 for both
        tasks.append((f"pair_d{d:.1f}_phase0", {
            "type": "pair",
            "d": d,
            "delta_phi": 0.0,
            "positions": [
                (0.0, 0.0, -d / 2.0, 0.0),
                (0.0, 0.0, +d / 2.0, 0.0),
            ],
        }))
        # Anti-phase: delta_phi = 0 for first, pi for second
        tasks.append((f"pair_d{d:.1f}_phasePi", {
            "type": "pair",
            "d": d,
            "delta_phi": np.pi,
            "positions": [
                (0.0, 0.0, -d / 2.0, 0.0),
                (0.0, 0.0, +d / 2.0, np.pi),
            ],
        }))

    return tasks


def sanity_check():
    """Run only d=6.0 and single baseline, verify against known values."""
    print("=" * 60)
    print("SANITY CHECK: d=6.0 pair vs known values")
    print("=" * 60)

    tasks = build_tasks([6.0], include_single=True)

    os.makedirs(OUT_DIR, exist_ok=True)

    with multiprocessing.Pool(3) as pool:
        results = pool.map(run_single, tasks)

    res = {r["label"]: r for r in results}
    E_single = res["single_baseline"]["E_final"]
    E_same = res["pair_d6.0_phase0"]["E_final"]
    E_anti = res["pair_d6.0_phasePi"]["E_final"]

    E_bind_same = E_same - 2.0 * E_single
    E_bind_anti = E_anti - 2.0 * E_single

    print()
    print("Sanity check results:")
    print(f"  E_single       = {E_single:.6e}")
    print(f"  E_pair_same    = {E_same:.6e}")
    print(f"  E_pair_anti    = {E_anti:.6e}")
    print(f"  E_bind(0)      = {E_bind_same:+.6f}  (expected {REF_E_SAME:+.3f})")
    print(f"  E_bind(pi)     = {E_bind_anti:+.6f}  (expected {REF_E_ANTI:+.3f})")

    err_same = abs(E_bind_same - REF_E_SAME) / abs(REF_E_SAME)
    err_anti = abs(E_bind_anti - REF_E_ANTI) / abs(REF_E_ANTI)

    print(f"  err_same       = {err_same:.4f}  (tol={REF_TOL})")
    print(f"  err_anti       = {err_anti:.4f}  (tol={REF_TOL})")

    if err_same > REF_TOL or err_anti > REF_TOL:
        print("\n*** SANITY CHECK FAILED -- aborting sweep ***")
        sys.exit(1)

    print("\nSanity check PASSED.")
    return E_single, E_bind_same, E_bind_anti


def full_sweep(E_single_from_sanity):
    """Run remaining separations (d=6.0 already cached from sanity check)."""
    tasks = build_tasks(SEPARATIONS, include_single=True)

    print()
    print("=" * 60)
    print(f"FULL SWEEP: {len(tasks)} runs on {N_WORKERS} workers")
    print("=" * 60)

    t0 = time.perf_counter()

    with multiprocessing.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, tasks)

    wall = time.perf_counter() - t0
    print(f"\nAll runs complete in {wall:.1f}s")

    return {r["label"]: r for r in results}


def fit_gaussian(distances, magnitudes):
    """Fit |E_pair(pi, d)| = f * exp(-d^2 / (2 * R_eff^2)) via log-linear regression."""
    d_arr = np.array(distances)
    m_arr = np.array(magnitudes)

    # Filter out zero/negative magnitudes
    mask = m_arr > 0
    d_fit = d_arr[mask]
    m_fit = m_arr[mask]

    if len(d_fit) < 3:
        return None, None, None

    # ln|E| = ln(f) - d^2 / (2 * R_eff^2)
    ln_m = np.log(m_fit)
    d2 = d_fit**2

    # Linear regression: ln_m = a + b * d^2, where a = ln(f), b = -1/(2*R_eff^2)
    A = np.column_stack([np.ones_like(d2), d2])
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, ln_m, rcond=None)
    a, b = coeffs

    f_fit = np.exp(a)
    if b >= 0:
        return f_fit, None, None
    R_eff = np.sqrt(-1.0 / (2.0 * b))

    # R^2
    ln_pred = a + b * d2
    ss_res = np.sum((ln_m - ln_pred)**2)
    ss_tot = np.sum((ln_m - np.mean(ln_m))**2)
    R_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return f_fit, R_eff, R_sq


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Step 1: Sanity check at d=6.0
    E_single_check, E_bind_same_check, E_bind_anti_check = sanity_check()

    # Step 2: Full sweep (d=6.0 will be cached)
    all_results = full_sweep(E_single_check)

    # Extract single baseline
    E_single = all_results["single_baseline"]["E_final"]

    # Build summary table
    print()
    print("=" * 60)
    print("DISTANCE SWEEP SUMMARY")
    print("=" * 60)
    print(f"{'d':>6s}  {'E_pair(0)':>12s}  {'E_pair(pi)':>12s}  {'|E_pair(pi)|':>12s}")
    print("-" * 50)

    table = []
    for d in SEPARATIONS:
        label_same = f"pair_d{d:.1f}_phase0"
        label_anti = f"pair_d{d:.1f}_phasePi"

        E_same = all_results[label_same]["E_final"]
        E_anti = all_results[label_anti]["E_final"]

        E_bind_same = E_same - 2.0 * E_single
        E_bind_anti = E_anti - 2.0 * E_single

        print(f"{d:6.1f}  {E_bind_same:+12.4f}  {E_bind_anti:+12.4f}  {abs(E_bind_anti):12.4f}")

        table.append({
            "d": d,
            "E_pair_same": E_bind_same,
            "E_pair_anti": E_bind_anti,
            "E_pair_anti_abs": abs(E_bind_anti),
        })

    # Gaussian envelope fit on anti-phase magnitudes
    distances = [row["d"] for row in table]
    anti_mags = [row["E_pair_anti_abs"] for row in table]

    f_fit, R_eff, R_sq = fit_gaussian(distances, anti_mags)

    print()
    print("Gaussian envelope fit: |E_pair(pi, d)| = f * exp(-d^2 / (2 * R_eff^2))")
    if f_fit is not None and R_eff is not None:
        print(f"  f     = {f_fit:.4f}")
        print(f"  R_eff = {R_eff:.4f}")
        print(f"  R^2   = {R_sq:.6f}")
    else:
        print("  Fit failed (insufficient data or non-decaying profile)")

    # Cross-check at d=6.0
    d6_row = next(row for row in table if row["d"] == 6.0)
    err_same = abs(d6_row["E_pair_same"] - REF_E_SAME) / abs(REF_E_SAME)
    err_anti = abs(d6_row["E_pair_anti"] - REF_E_ANTI) / abs(REF_E_ANTI)

    print()
    print(f"Cross-check at d=6.0:")
    print(f"  E_pair(0)  = {d6_row['E_pair_same']:+.4f}  (ref={REF_E_SAME:+.3f}, err={err_same:.4f})")
    print(f"  E_pair(pi) = {d6_row['E_pair_anti']:+.4f}  (ref={REF_E_ANTI:+.3f}, err={err_anti:.4f})")

    # Save summary JSON
    summary = {
        "completed": True,
        "description": "Pairwise interaction distance sweep: E_pair(d) for same-phase and anti-phase pairs",
        "parameters": {
            "N": N, "L": L, "m": m, "g4": g4, "g6": g6,
            "phi0": phi0, "R": R, "omega_eff": omega_eff,
            "dt": dt, "sigma": sigma, "T_final": T_final,
        },
        "separations": SEPARATIONS,
        "E_single_T500": E_single,
        "table": table,
        "gaussian_fit": {
            "f": f_fit,
            "R_eff": R_eff,
            "R_squared": R_sq,
        },
        "cross_check_d6": {
            "E_pair_same": d6_row["E_pair_same"],
            "E_pair_anti": d6_row["E_pair_anti"],
            "ref_E_pair_same": REF_E_SAME,
            "ref_E_pair_anti": REF_E_ANTI,
            "err_same": err_same,
            "err_anti": err_anti,
        },
    }

    summary_path = os.path.join(OUT_DIR, "distance_sweep_summary.json")
    tmp = summary_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, summary_path)
    print(f"\nSummary saved to {summary_path}")

    # Cleanup checkpoints and temp files
    checkpoints = glob.glob(os.path.join(OUT_DIR, "*.checkpoint*"))
    temps = glob.glob(os.path.join(OUT_DIR, "*.tmp"))
    freed = 0
    for fp in checkpoints + temps:
        freed += os.path.getsize(fp)
        os.remove(fp)
    if checkpoints or temps:
        print(f"Cleanup: removed {len(checkpoints)} checkpoints, {len(temps)} temps, freed {freed / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
