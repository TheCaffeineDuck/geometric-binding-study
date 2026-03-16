"""
studies/01_single_reference.py
==============================
Phase 1: Single Oscillon Baseline Decay Curves

Evolves a single isolated Gaussian oscillon to T=500 for two parameter sets,
producing the honest time-matched E_single(t) reference curves that all later
phases depend on.

Parameter sets:
  Set A: phi0=0.3, R=3.0
  Set B: phi0=0.5, R=2.5

Fixed: m=1.0, g4=0.30, g6=0.055, N=64, L=50.0, dt=0.05, sigma_KO=0.01
"""

import sys
import os
import json
import time
from datetime import datetime

import numpy as np

# Allow running from project root or studies/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.evolver import SexticEvolver

# ── Fixed simulation parameters ──────────────────────────────────────────────
FIXED = dict(m=1.0, g4=0.30, g6=0.055, N=64, L=50.0, dt=0.05, sigma_KO=0.01)

T_END        = 500.0
RECORD_EVERY = 10          # steps  (= 0.5 time units)
PRINT_EVERY  = 100         # steps  (= 5   time units)
N_STEPS      = int(T_END / FIXED["dt"])   # 10 000

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Parameter sets ────────────────────────────────────────────────────────────
PARAM_SETS = [
    {"label": "A", "phi0": 0.3, "R": 3.0},
    {"label": "B", "phi0": 0.5, "R": 2.5},
]


# ── Helper: build Gaussian oscillon initial condition ─────────────────────────
def gaussian_oscillon(evolver, phi0, R):
    """Return (phi, phi_dot) for a single Gaussian oscillon at grid centre."""
    r2 = evolver.X**2 + evolver.Y**2 + evolver.Z**2
    phi = phi0 * np.exp(-r2 / (2.0 * R**2))
    phi_dot = np.zeros_like(phi)
    return phi, phi_dot


# ── Main evolution routine ────────────────────────────────────────────────────
def run_set(label, phi0, R):
    params = {**FIXED, "phi0": phi0, "R": R, "label": label}

    print(f"\n{'='*60}")
    print(f"  Set {label}: phi0={phi0}, R={R}")
    print(f"  Steps: {N_STEPS}  |  dt={FIXED['dt']}  |  T_end={T_END}")
    print(f"{'='*60}")

    ev = SexticEvolver(
        N=FIXED["N"],
        L=FIXED["L"],
        m=FIXED["m"],
        g4=FIXED["g4"],
        g6=FIXED["g6"],
        dissipation_sigma=FIXED["sigma_KO"],
    )

    phi_init, phidot_init = gaussian_oscillon(ev, phi0, R)
    ev.set_initial_conditions(phi_init, phidot_init)

    E0 = ev.compute_energy()
    print(f"  Initial energy E(0) = {E0:.6f}")
    print(f"  Initial max|phi|    = {np.max(np.abs(phi_init)):.6f}")

    time_series = []

    wall_start = time.perf_counter()

    for step in range(N_STEPS + 1):
        # Record before stepping (step 0) or every RECORD_EVERY steps after
        if step % RECORD_EVERY == 0:
            E_now = ev.compute_energy()
            amp   = float(np.max(np.abs(ev.phi)))
            time_series.append({"t": float(ev.t), "energy": E_now, "max_amplitude": amp})

        # Progress print
        if step % PRINT_EVERY == 0 and step > 0:
            elapsed = time.perf_counter() - wall_start
            pct     = 100.0 * step / N_STEPS
            E_now   = time_series[-1]["energy"]
            amp     = time_series[-1]["max_amplitude"]
            drift   = abs(E_now - E0) / (abs(E0) + 1e-30)
            eta_s   = elapsed / step * (N_STEPS - step)
            print(
                f"  t={ev.t:7.1f}  E={E_now:.5f}  "
                f"max|phi|={amp:.5f}  drift={drift:.2e}  "
                f"[{pct:.1f}%  ETA {eta_s/60:.1f} min]"
            )
            sys.stdout.flush()

        if step < N_STEPS:
            ev.step_rk4(FIXED["dt"])

    wall_end = time.perf_counter()
    runtime  = wall_end - wall_start

    E_final   = time_series[-1]["energy"]
    amp_final = time_series[-1]["max_amplitude"]
    drift     = abs(E_final - E0) / (abs(E0) + 1e-30)

    print(f"\n  -- Set {label} complete --")
    print(f"  Runtime         : {runtime:.1f} s ({runtime/60:.2f} min)")
    print(f"  E(0)            : {E0:.6f}")
    print(f"  E(T)            : {E_final:.6f}")
    print(f"  Energy drift    : {drift:.4e}")
    print(f"  Final max|phi|  : {amp_final:.6f}")
    print(f"  Amplitude retent: {amp_final/phi0:.4f}  (final/initial)")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"set_{label}_baseline_{ts_str}.json"
    out_path = os.path.join(OUTPUT_DIR, filename)

    result = {
        "params":          params,
        "time_series":     time_series,
        "energy_drift":    drift,
        "runtime_seconds": runtime,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Saved -> {out_path}")
    return result


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = {}
    for ps in PARAM_SETS:
        results[ps["label"]] = run_set(ps["label"], ps["phi0"], ps["R"])

    # ── Summary comparison ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY COMPARISON")
    print(f"{'='*60}")
    header = (
        f"  {'Set':<5} {'phi0':>6} {'R':>5} {'E(0)':>12} "
        f"{'E(T)':>12} {'Drift':>10} {'max|phi|(T)':>13} {'Amp Retent':>12}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for lbl, res in results.items():
        p    = res["params"]
        ts   = res["time_series"]
        E0   = ts[0]["energy"]
        ET   = ts[-1]["energy"]
        amp0 = p["phi0"]
        ampT = ts[-1]["max_amplitude"]
        drift_val = res["energy_drift"]
        retention = ampT / amp0
        print(
            f"  {lbl:<5} {p['phi0']:>6.3f} {p['R']:>5.1f} "
            f"{E0:>12.5f} {ET:>12.5f} {drift_val:>10.3e} "
            f"{ampT:>13.6f} {retention:>12.4f}"
        )

    print(f"\n  Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("  Phase 1 complete.")
