"""
studies/01_run_setB.py
======================
Phase 1 - Set B only: phi0=0.5, R=2.5
Single isolated Gaussian oscillon evolved to T=500.
All print statements use pure ASCII (no unicode box-drawing chars).
"""

import sys
import os
import json
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.evolver import SexticEvolver

# --- params ---
PHI0   = 0.5
R      = 2.5
M      = 1.0
G4     = 0.30
G6     = 0.055
N      = 64
L      = 50.0
DT     = 0.05
SIGMA  = 0.01
T_END  = 500.0
N_STEPS      = int(T_END / DT)   # 10000
RECORD_EVERY = 10
PRINT_EVERY  = 100

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase1")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUTPUT_DIR, "set_B_baseline.json")

print("=" * 60)
print("  Phase 1 - Set B: phi0=0.5, R=2.5")
print("  N=%d  L=%.1f  dt=%.3f  T_end=%.1f  steps=%d" % (N, L, DT, T_END, N_STEPS))
print("=" * 60)
sys.stdout.flush()

ev = SexticEvolver(N=N, L=L, m=M, g4=G4, g6=G6, dissipation_sigma=SIGMA)

r2      = ev.X**2 + ev.Y**2 + ev.Z**2
phi_init = PHI0 * np.exp(-r2 / (2.0 * R**2))
phidot_init = np.zeros_like(phi_init)
ev.set_initial_conditions(phi_init, phidot_init)

E0 = ev.compute_energy()
print("  Initial energy E(0) = %.6f" % E0)
print("  Initial max|phi|    = %.6f" % float(np.max(np.abs(phi_init))))
sys.stdout.flush()

time_series = []
wall_start  = time.perf_counter()

for step in range(N_STEPS + 1):

    if step % RECORD_EVERY == 0:
        E_now = ev.compute_energy()
        amp   = float(np.max(np.abs(ev.phi)))
        # NaN / divergence guard
        if not np.isfinite(E_now) or not np.isfinite(amp):
            print("  ERROR: non-finite value at step=%d t=%.2f E=%s amp=%s" % (
                  step, ev.t, E_now, amp))
            sys.exit(1)
        time_series.append({"t": float(ev.t), "energy": E_now, "max_amplitude": amp})

    if step % PRINT_EVERY == 0 and step > 0:
        elapsed = time.perf_counter() - wall_start
        pct     = 100.0 * step / N_STEPS
        E_now   = time_series[-1]["energy"]
        amp     = time_series[-1]["max_amplitude"]
        drift   = abs(E_now - E0) / (abs(E0) + 1e-30)
        eta_s   = elapsed / step * (N_STEPS - step)
        print("  [B] t=%7.1f  E=%.5f  max|phi|=%.5f  drift=%.2e  [%.1f%%  ETA %.1f min]" % (
              ev.t, E_now, amp, drift, pct, eta_s / 60.0))
        sys.stdout.flush()

    if step < N_STEPS:
        ev.step_rk4(DT)

wall_time = time.perf_counter() - wall_start
E_final   = time_series[-1]["energy"]
amp_final = time_series[-1]["max_amplitude"]
drift_f   = abs(E_final - E0) / (abs(E0) + 1e-30)

print("\n  -- Set B complete --")
print("  Runtime      : %.1f s (%.2f min)" % (wall_time, wall_time / 60.0))
print("  E(0)         : %.6f" % E0)
print("  E(T)         : %.6f" % E_final)
print("  Energy drift : %.4e" % drift_f)
print("  Final max|phi|: %.6f" % amp_final)
print("  Amp retention: %.4f  (final/phi0)" % (amp_final / PHI0))
sys.stdout.flush()

result = {
    "params": {
        "label": "B", "phi0": PHI0, "R": R,
        "m": M, "g4": G4, "g6": G6,
        "N": N, "L": L, "dt": DT, "sigma_KO": SIGMA,
    },
    "time_series":     time_series,
    "energy_drift":    drift_f,
    "runtime_seconds": wall_time,
}

with open(OUT_FILE, "w") as f:
    json.dump(result, f, indent=2)

print("  Saved -> %s" % OUT_FILE)
print("  Phase 1 Set B DONE.")
