#!/usr/bin/env python3
"""
studies/dissipation_test.py
============================
KO Dissipation Self-Organization Test

Tests whether the self-organization signal from Section 9 (unanimous negative
DS across 50 seeds at sigma_KO=0.01) persists at sigma_KO=0.00.

If the drift disappears without dissipation, Section 9 is an artifact of KO
dissipation preferentially damping high-energy (same-phase) modes.

10-seed ensemble with fresh random seeds (1000-1009), T=1000.
"""

import os
import sys
import json
import time
import multiprocessing as mp

# Prevent Numba thread oversubscription when using multiprocessing
os.environ['NUMBA_NUM_THREADS'] = '1'

import numpy as np

# Ensure engine is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
N_WORKERS = 4

# Physics parameters (Set B) -- identical to paper EXCEPT sigma
N_GRID    = 64
L         = 50.0
M         = 1.0
G4        = 0.30
G6        = 0.055
DT        = 0.05
SIGMA     = 0.00   # <<< KEY CHANGE: no dissipation (paper used 0.01)
PHI0      = 0.5
R_GAUSS   = 2.5
OMEGA_EFF = 1.113

# Evolution
T_FINAL      = 1000.0
N_STEPS      = int(T_FINAL / DT)       # 20000
MEAS_DT      = 10.0                     # measurement interval in time units
RECORD_EVERY = int(MEAS_DT / DT)       # 200 steps
PRINT_EVERY  = 2000
CHECKPOINT_EVERY = 2000                 # ~100 time units

N_SEEDS = 10
SEED_OFFSET = 1000  # seeds 1000-1009 (fresh, not reusing paper's 0-49)
MIN_AMP_GUARD = 0.01 * PHI0

# Paper reference values (sigma=0.01, 50-seed ensemble)
PAPER_DS_NEGATIVE_FRAC = 50.0 / 50.0    # 100%
PAPER_MEAN_DS          = -0.117
PAPER_Z_SCORE          = -15.4
PAPER_EBIND_NEG_FRAC   = 30.0 / 50.0    # 60%
PAPER_MEAN_EBIND       = -3.12
PAPER_ENERGY_DRIFT     = 0.0025          # percent

# Energy drift thresholds
DRIFT_WARNING_PCT  = 1.0   # warn above this
DRIFT_UNSTABLE_PCT = 5.0   # declare instability above this

# Directories
BASE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "dissipation_test")

# ---------------------------------------------------------------------------
#  Icosahedron geometry (identical to phase12)
# ---------------------------------------------------------------------------
phi_g = (1.0 + np.sqrt(5.0)) / 2.0

base_verts = []
for s1 in [1, -1]:
    for s2 in [1, -1]:
        base_verts.append([0.0, s1 * 1.0, s2 * phi_g])
        base_verts.append([s1 * 1.0, s2 * phi_g, 0.0])
        base_verts.append([s1 * phi_g, 0.0, s2 * 1.0])
base_verts = np.array(base_verts)

dists = []
for i in range(12):
    for j in range(i + 1, 12):
        dists.append(np.linalg.norm(base_verts[i] - base_verts[j]))
min_edge = min(dists)
scale = 6.0 / min_edge
VERTS = base_verts * scale

EDGE_TOL = min_edge * scale * 1.01
EDGES = []
for i in range(12):
    for j in range(i + 1, 12):
        if np.linalg.norm(VERTS[i] - VERTS[j]) < EDGE_TOL * 1.01:
            EDGES.append((i, j))

assert len(VERTS) == 12, "Expected 12 vertices, got %d" % len(VERTS)
assert len(EDGES) == 30, "Expected 30 edges, got %d" % len(EDGES)


# ---------------------------------------------------------------------------
#  Helper: grid indices nearest to a vertex position
# ---------------------------------------------------------------------------
def _nearest_grid_idx(ev, pos):
    """Return (ix, iy, iz) indices of the grid point nearest to pos."""
    dx = ev.L / ev.N
    ix = int(round((pos[0] + ev.L / 2.0) / dx)) % ev.N
    iy = int(round((pos[1] + ev.L / 2.0) / dx)) % ev.N
    iz = int(round((pos[2] + ev.L / 2.0) / dx)) % ev.N
    return ix, iy, iz


# ---------------------------------------------------------------------------
#  Phase measurement diagnostic (identical to phase12)
# ---------------------------------------------------------------------------
def make_phase_diagnostic(verts, edges, omega_eff, min_amp):
    """Return a diagnostic function for evolver.evolve(extra_diagnostic_fn=...)."""
    grid_indices = None

    def diagnostic(ev):
        nonlocal grid_indices
        if grid_indices is None:
            grid_indices = [_nearest_grid_idx(ev, v) for v in verts]

        phases = []
        for idx, (ix, iy, iz) in enumerate(grid_indices):
            phi_val = ev.phi[ix, iy, iz]
            phi_dot_val = ev.phi_dot[ix, iy, iz]
            amp = np.sqrt((omega_eff * phi_val)**2 + phi_dot_val**2)
            if amp < min_amp:
                phases.append(float('nan'))
            else:
                theta = np.arctan2(-phi_dot_val, omega_eff * phi_val)
                phases.append(float(theta))

        # Order parameter: S = (1/30) * sum_edges cos(theta_i - theta_j)
        s_val = 0.0
        n_valid = 0
        for (i, j) in edges:
            if np.isnan(phases[i]) or np.isnan(phases[j]):
                continue
            s_val += np.cos(phases[i] - phases[j])
            n_valid += 1
        if n_valid > 0:
            s_val /= n_valid
        else:
            s_val = float('nan')

        return {
            'phases': phases,
            'S': float(s_val),
        }

    return diagnostic


# ---------------------------------------------------------------------------
#  File paths
# ---------------------------------------------------------------------------
def _checkpoint_path(label):
    return os.path.join(OUTPUT_DIR, "%s.checkpoint.json" % label)


def _output_path(label):
    return os.path.join(OUTPUT_DIR, "%s.json" % label)


def _make_checkpoint_callback(label):
    ckpt_path = _checkpoint_path(label)
    tmp_path = ckpt_path + ".tmp"

    def _callback(state_dict):
        with open(tmp_path, "w") as f:
            json.dump(state_dict, f, separators=(",", ":"))
        os.replace(tmp_path, ckpt_path)

    return _callback


# ---------------------------------------------------------------------------
#  Step 1: Isolated baseline at sigma=0
# ---------------------------------------------------------------------------
def run_baseline():
    """Single isolated oscillon at sigma=0, T=1000."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from engine.evolver import SexticEvolver

    label = "baseline_sigma0"
    out_path = _output_path(label)

    # Cache check
    if os.path.exists(out_path):
        with open(out_path) as f:
            data = json.load(f)
        if data.get("completed", False):
            print("CACHED: baseline_sigma0")
            sys.stdout.flush()
            return data

    print("Step 1: Isolated baseline (sigma=0)...")
    sys.stdout.flush()

    ev = SexticEvolver(N=N_GRID, L=L, m=M, g4=G4, g6=G6, dissipation_sigma=SIGMA)

    # Single oscillon at center, theta=0
    phi_init = PHI0 * np.exp(-(ev.X**2 + ev.Y**2 + ev.Z**2) / (2.0 * R_GAUSS**2))
    phi_dot_init = np.zeros_like(phi_init)
    ev.set_initial_conditions(phi_init, phi_dot_init)

    # Check for resume
    ckpt_path = _checkpoint_path(label)
    resume_from = None
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path) as f:
                resume_from = json.load(f)
            print("  RESUMING baseline from step %d" % resume_from['completed_steps'])
            sys.stdout.flush()
        except (json.JSONDecodeError, OSError):
            resume_from = None

    wall_start = time.perf_counter()

    state = ev.evolve(
        dt=DT,
        n_steps=N_STEPS,
        record_every=RECORD_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_callback=_make_checkpoint_callback(label),
        resume_from=resume_from,
        print_every=PRINT_EVERY,
        tag="baseline_s0",
    )

    wall_time = time.perf_counter() - wall_start

    E0 = state['E0']
    E_final = state['time_series']['E_total'][-1]
    drift_pct = abs(E_final - E0) / (abs(E0) + 1e-30) * 100.0

    result = {
        "completed": True,
        "label": "baseline_sigma0",
        "sigma_KO": SIGMA,
        "times": state['time_series']['times'],
        "E_single": state['time_series']['E_total'],
        "max_amplitude": state['time_series']['max_amplitude'],
        "E0": E0,
        "energy_drift_pct": drift_pct,
        "wall_time_seconds": wall_time + state.get('wall_elapsed', 0),
    }

    # Atomic write
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, out_path)

    # Cleanup checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print("  E_single(T=1000) = %.3f" % E_final)
    print("  Energy drift: %.4f%%" % drift_pct)
    print("  Wall time: %.0fs" % result['wall_time_seconds'])

    if drift_pct > DRIFT_WARNING_PCT:
        print("  WARNING: Energy drift %.4f%% exceeds %.1f%% threshold" % (
            drift_pct, DRIFT_WARNING_PCT))
    if drift_pct > DRIFT_UNSTABLE_PCT:
        print("  CRITICAL: Energy drift %.4f%% exceeds %.1f%% -- NUMERICALLY UNSTABLE" % (
            drift_pct, DRIFT_UNSTABLE_PCT))

    sys.stdout.flush()
    return result


# ---------------------------------------------------------------------------
#  Step 2: Single seed worker
# ---------------------------------------------------------------------------
def run_seed(seed):
    """Run one ensemble member at sigma=0. Called by Pool workers."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from engine.evolver import SexticEvolver
    from scipy.interpolate import interp1d

    label = "seed_%04d" % seed
    out_path = _output_path(label)

    # Cache check
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                data = json.load(f)
            if data.get("completed", False):
                print("CACHED: %s" % label)
                sys.stdout.flush()
                return data
        except (json.JSONDecodeError, OSError):
            pass

    wall_start = time.perf_counter()

    # Load baseline for binding energy
    baseline_path = _output_path("baseline_sigma0")
    with open(baseline_path) as f:
        baseline = json.load(f)
    E_single_interp = interp1d(baseline['times'], baseline['E_single'],
                               kind='linear', fill_value='extrapolate')

    # Generate random phases
    rng = np.random.default_rng(seed)
    initial_phases = rng.uniform(0, 2 * np.pi, size=12).tolist()

    # Initialize evolver with sigma=0
    ev = SexticEvolver(N=N_GRID, L=L, m=M, g4=G4, g6=G6, dissipation_sigma=SIGMA)

    phi_init = np.zeros((N_GRID, N_GRID, N_GRID))
    phi_dot_init = np.zeros((N_GRID, N_GRID, N_GRID))

    for idx in range(12):
        theta_i = initial_phases[idx]
        pos = VERTS[idx]
        dx_ = ev.X - pos[0]
        dy_ = ev.Y - pos[1]
        dz_ = ev.Z - pos[2]
        r2 = dx_**2 + dy_**2 + dz_**2
        envelope = np.exp(-r2 / (2.0 * R_GAUSS**2))
        phi_init += PHI0 * np.cos(theta_i) * envelope
        phi_dot_init += (-OMEGA_EFF * PHI0 * np.sin(theta_i)) * envelope

    ev.set_initial_conditions(phi_init, phi_dot_init)

    # Phase diagnostic
    diag_fn = make_phase_diagnostic(VERTS, EDGES, OMEGA_EFF, MIN_AMP_GUARD)

    # Check for resume
    ckpt_path = _checkpoint_path(label)
    resume_from = None
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path) as f:
                resume_from = json.load(f)
            print("  RESUMING %s from step %d" % (label, resume_from['completed_steps']))
            sys.stdout.flush()
        except (json.JSONDecodeError, OSError):
            resume_from = None

    # Evolve
    state = ev.evolve(
        dt=DT,
        n_steps=N_STEPS,
        record_every=RECORD_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_callback=_make_checkpoint_callback(label),
        resume_from=resume_from,
        print_every=PRINT_EVERY,
        tag=label,
        extra_diagnostic_fn=diag_fn,
    )

    # Extract time series
    times = state['time_series']['times']
    E_total = state['time_series']['E_total']
    extras = state.get('extra_diagnostics', [])

    order_parameter = [d['S'] for d in extras]
    phases_series = [d['phases'] for d in extras]

    # Binding energy: E_total(t) - 12 * E_single(t)
    binding_energy = [
        E_total[i] - 12.0 * float(E_single_interp(times[i]))
        for i in range(len(times))
    ]

    E0 = state['E0']
    energy_drift_pct = abs(E_total[-1] - E0) / (abs(E0) + 1e-30) * 100.0
    wall_time = time.perf_counter() - wall_start

    # Compute DS = S(T=1000) - S(T=50)
    # Find index closest to T=50
    t50_idx = None
    for i, t in enumerate(times):
        if t >= 50.0:
            t50_idx = i
            break
    if t50_idx is None:
        t50_idx = 0

    S_init = order_parameter[t50_idx] if t50_idx < len(order_parameter) else float('nan')
    S_final = order_parameter[-1] if order_parameter else float('nan')
    DS = S_final - S_init if not (np.isnan(S_final) or np.isnan(S_init)) else float('nan')
    E_bind_final = binding_energy[-1] if binding_energy else float('nan')

    result = {
        "completed": True,
        "seed": seed,
        "sigma_KO": SIGMA,
        "initial_phases": initial_phases,
        "S_at_t50": S_init,
        "S_final": S_final,
        "DS": DS,
        "E_bind_final": E_bind_final,
        "times": times,
        "order_parameter": order_parameter,
        "binding_energy": binding_energy,
        "phases": phases_series,
        "total_energy": E_total,
        "energy_drift_pct": energy_drift_pct,
        "wall_time_seconds": wall_time,
    }

    # Atomic write
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    os.replace(tmp, out_path)

    # Cleanup checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print("  Seed %04d: S_init=%+.3f, S_final=%.3f, DS=%.3f, E_bind=%.2f (%.0fs)" % (
        seed, S_init, S_final, DS, E_bind_final, wall_time))
    sys.stdout.flush()

    return result


# ---------------------------------------------------------------------------
#  Step 4: Comparison and verdict
# ---------------------------------------------------------------------------
def build_comparison():
    """Load all results and print comparison table."""

    # Load baseline
    baseline_path = _output_path("baseline_sigma0")
    with open(baseline_path) as f:
        baseline = json.load(f)

    # Load seed results
    results = []
    for i in range(N_SEEDS):
        seed = SEED_OFFSET + i
        path = _output_path("seed_%04d" % seed)
        if not os.path.exists(path):
            print("  WARNING: missing seed_%04d" % seed)
            continue
        with open(path) as f:
            data = json.load(f)
        if not data.get("completed", False):
            print("  WARNING: seed_%04d incomplete" % seed)
            continue
        results.append(data)

    if len(results) == 0:
        print("  No completed seeds found!")
        return

    # Extract metrics
    DS_vals = [r['DS'] for r in results if not np.isnan(r['DS'])]
    E_bind_vals = [r['E_bind_final'] for r in results]
    drift_vals = [r['energy_drift_pct'] for r in results]

    n_DS_negative = sum(1 for ds in DS_vals if ds < 0)
    n_valid = len(DS_vals)
    mean_DS = float(np.mean(DS_vals)) if DS_vals else float('nan')
    std_DS = float(np.std(DS_vals)) if DS_vals else float('nan')

    n_Ebind_negative = sum(1 for eb in E_bind_vals if eb < 0)
    mean_Ebind = float(np.mean(E_bind_vals)) if E_bind_vals else float('nan')
    mean_drift = float(np.mean(drift_vals)) if drift_vals else float('nan')

    # z-score
    if n_valid > 1 and std_DS > 0:
        sem = std_DS / np.sqrt(n_valid)
        z_score = mean_DS / sem
    else:
        z_score = float('nan')

    # Print comparison
    print("")
    print("=== COMPARISON ===")
    print("%-22s| %-19s| %-19s" % (
        "Metric", "sigma=0.01 (paper)", "sigma=0.00 (this test)"))
    print("-" * 65)
    print("%-22s| %-19s| %d/%d (%d%%)" % (
        "Seeds with DS < 0",
        "50/50 (100%)",
        n_DS_negative, n_valid,
        int(100.0 * n_DS_negative / max(n_valid, 1))))
    print("%-22s| %-19s| %+.3f" % (
        "Mean DS", "-0.117", mean_DS))
    print("%-22s| %-19s| %.1f" % (
        "z-score (DS)", "-15.4", z_score))
    print("%-22s| %-19s| %d/%d (%d%%)" % (
        "Seeds E_bind < 0",
        "30/50 (60%)",
        n_Ebind_negative, len(E_bind_vals),
        int(100.0 * n_Ebind_negative / max(len(E_bind_vals), 1))))
    print("%-22s| %-19s| %+.2f" % (
        "Mean E_bind", "-3.12", mean_Ebind))
    print("%-22s| %-19s| %.4f%%" % (
        "Energy drift (mean)", "0.0025%", mean_drift))
    print("")

    # Verdict
    ds_neg_frac = n_DS_negative / max(n_valid, 1)
    mean_DS_ratio = abs(mean_DS / PAPER_MEAN_DS) if not np.isnan(mean_DS) and PAPER_MEAN_DS != 0 else float('nan')

    print("=== VERDICT ===")
    if ds_neg_frac >= 0.8 and mean_DS < 0 and mean_DS_ratio >= 0.5:
        verdict = "PASS"
        print("PASS: Self-organization preserved without dissipation. Section 9 stands.")
    elif ds_neg_frac >= 0.5 or (mean_DS < 0 and mean_DS_ratio >= 0.2):
        verdict = "MARGINAL"
        print("MARGINAL: Partial preservation. Section 9 needs caveating.")
    else:
        verdict = "FAIL"
        print("FAIL: Self-organization absent without dissipation. Section 9 is an artifact.")

    # Energy stability note
    if mean_drift > DRIFT_UNSTABLE_PCT:
        print("")
        print("NOTE: Mean energy drift %.4f%% exceeds %.1f%%. Simulations are numerically" % (
            mean_drift, DRIFT_UNSTABLE_PCT))
        print("unstable at sigma=0 with dt=0.05. Consider sigma=0.001 fallback.")
    elif mean_drift > DRIFT_WARNING_PCT:
        print("")
        print("NOTE: Mean energy drift %.4f%% exceeds %.1f%%. Results may be affected" % (
            mean_drift, DRIFT_WARNING_PCT))
        print("by numerical noise accumulation.")

    # Save summary
    summary = {
        "completed": True,
        "sigma_KO": SIGMA,
        "n_seeds": len(results),
        "n_DS_negative": n_DS_negative,
        "n_valid": n_valid,
        "mean_DS": mean_DS,
        "std_DS": std_DS,
        "z_score": float(z_score),
        "n_Ebind_negative": n_Ebind_negative,
        "mean_Ebind": mean_Ebind,
        "mean_energy_drift_pct": mean_drift,
        "baseline_energy_drift_pct": baseline['energy_drift_pct'],
        "verdict": verdict,
        "paper_reference": {
            "sigma_KO": 0.01,
            "n_seeds": 50,
            "DS_negative_frac": PAPER_DS_NEGATIVE_FRAC,
            "mean_DS": PAPER_MEAN_DS,
            "z_score": PAPER_Z_SCORE,
            "Ebind_negative_frac": PAPER_EBIND_NEG_FRAC,
            "mean_Ebind": PAPER_MEAN_EBIND,
            "mean_energy_drift_pct": PAPER_ENERGY_DRIFT,
        },
        "per_seed": [
            {
                "seed": r['seed'],
                "S_at_t50": r['S_at_t50'],
                "S_final": r['S_final'],
                "DS": r['DS'],
                "E_bind_final": r['E_bind_final'],
                "energy_drift_pct": r['energy_drift_pct'],
            }
            for r in results
        ],
    }

    summary_path = _output_path("summary")
    tmp = summary_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, summary_path)

    print("")
    sys.stdout.flush()
    return summary


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== KO DISSIPATION SELF-ORGANIZATION TEST ===")
    print("sigma_KO = %.2f (paper used 0.01)" % SIGMA)
    print("Seeds: %d, T=%.0f, icosahedron" % (N_SEEDS, T_FINAL))
    print("Seed range: %d-%d (fresh seeds, not reusing paper ensemble)" % (
        SEED_OFFSET, SEED_OFFSET + N_SEEDS - 1))
    print("Output: %s" % os.path.abspath(OUTPUT_DIR))
    print("")
    sys.stdout.flush()

    t0 = time.perf_counter()

    # Step 1: Baseline
    baseline = run_baseline()
    print("")

    # Check if baseline is numerically unstable
    if baseline['energy_drift_pct'] > DRIFT_UNSTABLE_PCT:
        print("ABORT: Baseline energy drift %.4f%% exceeds %.1f%%." % (
            baseline['energy_drift_pct'], DRIFT_UNSTABLE_PCT))
        print("sigma=0.00 is numerically unstable at dt=0.05.")
        print("Consider running with sigma=0.001 as fallback.")
        sys.stdout.flush()
        return

    # Step 2: Ensemble
    print("Step 2: Ensemble (%d seeds, %d workers)..." % (N_SEEDS, N_WORKERS))
    sys.stdout.flush()

    seeds = [SEED_OFFSET + i for i in range(N_SEEDS)]
    with mp.Pool(N_WORKERS) as pool:
        pool.map(run_seed, seeds)

    ensemble_time = time.perf_counter() - t0
    print("  Wall time: %.0fs" % ensemble_time)
    print("")

    # Step 3-4: Comparison
    build_comparison()

    # Cleanup
    import glob as gl
    checkpoints = gl.glob(os.path.join(OUTPUT_DIR, "*.checkpoint*"))
    temps = gl.glob(os.path.join(OUTPUT_DIR, "*.tmp"))
    for f in checkpoints + temps:
        try:
            os.remove(f)
        except OSError:
            pass
    if checkpoints or temps:
        print("Cleanup: removed %d checkpoints, %d temp files" % (
            len(checkpoints), len(temps)))

    total_time = time.perf_counter() - t0
    print("Total wall time: %.1f min" % (total_time / 60.0))
    print("Done.")


if __name__ == '__main__':
    main()
