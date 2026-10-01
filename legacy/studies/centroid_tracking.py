#!/usr/bin/env python3
"""
studies/centroid_tracking.py
============================
Centroid Tracking Study: Measure whether oscillon centroids drift during
T=2000 evolution for the two bound configurations confirmed stable by
binding energy (cube checkerboard f_cross=1.0, icosahedron ce_20 f_cross=0.667).

Diagnostic interval: dt=10 (200 snapshots per run).
Runs a single-oscillon baseline to T=2000 first, then the two multi-oscillon
configs in parallel via Pool(2).
"""

import os
import sys
import json
import time
import glob
import multiprocessing as mp

os.environ['NUMBA_NUM_THREADS'] = '1'

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

N_WORKERS = 2

# -- Physics parameters (Set B) -----------------------------------------------
N      = 64
L      = 50.0
m      = 1.0
g4     = 0.30
g6     = 0.055
dt     = 0.05
sigma  = 0.01
phi0   = 0.5
R      = 2.5
T_END  = 2000.0
N_STEPS = int(T_END / dt)       # 40000
DIAG_DT = 10.0                  # diagnostic interval in time units
RECORD_EVERY = int(DIAG_DT / dt)  # 200 steps
PRINT_EVERY  = int(100.0 / dt)    # every 100 time units = 2000 steps
CHECKPOINT_EVERY = int(500.0 / dt) # every 500 time units = 10000 steps
R_HALF = 2.5                    # half-width of local centroid region

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "centroid_tracking")
PHASE1_JSON = os.path.join(BASE_DIR, "outputs", "phase1", "set_B_baseline.json")


# -- Geometry definitions ------------------------------------------------------

def cube_vertices():
    """8 cube vertices at (+-3, +-3, +-3), edge length 6.0."""
    import numpy as np
    D = 3.0
    return np.array([
        [-D, -D, -D], [ D, -D, -D], [-D,  D, -D], [ D,  D, -D],
        [-D, -D,  D], [ D, -D,  D], [-D,  D,  D], [ D,  D,  D],
    ])


def icosahedron_vertices():
    """12 icosahedron vertices scaled to min edge = 6.0."""
    import numpy as np
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    base = []
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            base.append([0.0,  s1 * 1.0,  s2 * phi_g])
            base.append([s1 * 1.0,  s2 * phi_g, 0.0])
            base.append([s1 * phi_g, 0.0,  s2 * 1.0])
    base = np.array(base)
    dists = []
    for i in range(12):
        for j in range(i + 1, 12):
            dists.append(np.linalg.norm(base[i] - base[j]))
    scale = 6.0 / min(dists)
    return base * scale


# -- Centroid tracker ----------------------------------------------------------

def make_centroid_tracker(evolver, vertices):
    """Build a diagnostic function that computes energy-weighted centroid
    displacements for each oscillon at every recording point."""
    import numpy as np
    from numpy.fft import fftn, ifftn, fftfreq

    N_grid = evolver.N
    L_box  = evolver.L
    dx     = evolver.dx
    coords = np.linspace(-L_box / 2, L_box / 2, N_grid, endpoint=False)

    # Spectral k-vectors for gradient computation
    k = fftfreq(N_grid, d=dx) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')

    m_sq = evolver.m_sq
    g4_  = evolver.g4
    g6_  = evolver.g6

    # Precompute local regions for each oscillon (fixed throughout evolution)
    regions = []
    for v in vertices:
        # Minimum-image offsets along each axis
        off_x = coords - v[0]
        off_x = off_x - L_box * np.round(off_x / L_box)
        mask_x = np.abs(off_x) <= R_HALF
        ix = np.where(mask_x)[0]
        ox = off_x[ix]

        off_y = coords - v[1]
        off_y = off_y - L_box * np.round(off_y / L_box)
        mask_y = np.abs(off_y) <= R_HALF
        iy = np.where(mask_y)[0]
        oy = off_y[iy]

        off_z = coords - v[2]
        off_z = off_z - L_box * np.round(off_z / L_box)
        mask_z = np.abs(off_z) <= R_HALF
        iz = np.where(mask_z)[0]
        oz = off_z[iz]

        regions.append((ix, iy, iz, ox, oy, oz))

    def diagnostic(ev):
        phi     = ev.phi
        phi_dot = ev.phi_dot

        # Energy density field H(x,y,z)
        phi_hat = fftn(phi)
        dphi_dx = np.real(ifftn(1j * KX * phi_hat))
        dphi_dy = np.real(ifftn(1j * KY * phi_hat))
        dphi_dz = np.real(ifftn(1j * KZ * phi_hat))
        grad_sq = dphi_dx**2 + dphi_dy**2 + dphi_dz**2

        H = (0.5 * phi_dot**2
             + 0.5 * grad_sq
             + 0.5 * m_sq * phi**2
             - (g4_ / 24.0) * phi**4
             + (g6_ / 720.0) * phi**6)

        displacements = []
        for ix, iy, iz, ox, oy, oz in regions:
            H_local = H[np.ix_(ix, iy, iz)]
            H_sum = H_local.sum()
            if H_sum < 1e-30:
                displacements.append(0.0)
                continue
            cx = np.sum(ox[:, None, None] * H_local) / H_sum
            cy = np.sum(oy[None, :, None] * H_local) / H_sum
            cz = np.sum(oz[None, None, :] * H_local) / H_sum
            disp = float(np.sqrt(cx**2 + cy**2 + cz**2))
            displacements.append(disp)

        return {"displacements": displacements}

    return diagnostic


# -- Atomic write helper -------------------------------------------------------

def atomic_write_json(data, filepath):
    tmp = str(filepath) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, str(filepath))


# -- Single oscillon baseline to T=2000 ---------------------------------------

def run_baseline():
    """Run single oscillon to T=2000, return (times, E_series)."""
    import numpy as np
    from scipy.interpolate import interp1d

    baseline_path = os.path.join(OUTPUT_DIR, "single_baseline_T2000.json")

    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            d = json.load(f)
        if d.get('completed'):
            print("CACHED: single baseline T=2000")
            sys.stdout.flush()
            return d['t_series'], d['E_series']

    print("Running single oscillon baseline to T=2000 ...")
    sys.stdout.flush()

    from engine.evolver import SexticEvolver

    ev = SexticEvolver(N=N, L=L, m=m, g4=g4, g6=g6, dissipation_sigma=sigma)
    r2 = ev.X**2 + ev.Y**2 + ev.Z**2
    phi_init = phi0 * np.exp(-r2 / (2.0 * R**2))
    ev.set_initial_conditions(phi_init, np.zeros_like(phi_init))

    ckpt_path = baseline_path + '.checkpoint.json'
    resume_from = None
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path) as f:
                resume_from = json.load(f)
            print("  Resuming baseline from step %d" % resume_from['completed_steps'])
            sys.stdout.flush()
        except (json.JSONDecodeError, KeyError):
            resume_from = None

    def ckpt_cb(state_dict):
        atomic_write_json(state_dict, ckpt_path)

    state = ev.evolve(
        dt=dt,
        n_steps=N_STEPS,
        record_every=RECORD_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_callback=ckpt_cb,
        resume_from=resume_from,
        print_every=PRINT_EVERY,
        tag="baseline",
    )

    result = {
        "completed": True,
        "t_series": state['time_series']['times'],
        "E_series": state['time_series']['E_total'],
        "wall_seconds": state['wall_elapsed'],
    }
    atomic_write_json(result, baseline_path)

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print("  Baseline complete (%.0f s)" % state['wall_elapsed'])
    sys.stdout.flush()
    return result['t_series'], result['E_series']


# -- Multi-oscillon worker -----------------------------------------------------

def run_multi_oscillon(config):
    """Worker function: evolve a multi-oscillon configuration with centroid tracking."""
    import numpy as np
    from scipy.interpolate import interp1d
    from engine.evolver import SexticEvolver, serialize_field, deserialize_field

    name       = config['name']
    vertices   = np.array(config['vertices'])
    phases     = np.array(config['phases'], dtype=float)
    n_osc      = len(vertices)
    out_path   = config['output_path']
    ckpt_path  = out_path + '.checkpoint.json'
    baseline_t = config['baseline_t']
    baseline_E = config['baseline_E']

    # Check cache
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                d = json.load(f)
            if d.get('completed'):
                print("CACHED: %s" % name)
                sys.stdout.flush()
                return d
        except (json.JSONDecodeError, KeyError):
            pass

    E_single_interp = interp1d(baseline_t, baseline_E,
                                kind='linear', fill_value='extrapolate')

    ev = SexticEvolver(N=N, L=L, m=m, g4=g4, g6=g6, dissipation_sigma=sigma)

    # Check for checkpoint
    resume_from = None
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path) as f:
                resume_from = json.load(f)
            print("  [%s] Resuming from step %d (t=%.1f)" % (
                name, resume_from['completed_steps'], resume_from['t']))
            sys.stdout.flush()
        except (json.JSONDecodeError, KeyError):
            resume_from = None

    # Build IC (only if not resuming -- evolve() restores field from checkpoint)
    if resume_from is None:
        phi_init = np.zeros((N, N, N))
        for A, pos in zip(phases, vertices):
            dx_ = ev.X - pos[0]
            dy_ = ev.Y - pos[1]
            dz_ = ev.Z - pos[2]
            r2  = dx_**2 + dy_**2 + dz_**2
            phi_init += A * phi0 * np.exp(-r2 / (2.0 * R**2))
        ev.set_initial_conditions(phi_init, np.zeros_like(phi_init))

    print("=" * 60)
    print("  Config: %s  (%d oscillons, T=%d)" % (name, n_osc, int(T_END)))
    print("=" * 60)
    sys.stdout.flush()

    # Build centroid tracker
    tracker = make_centroid_tracker(ev, vertices)

    def ckpt_cb(state_dict):
        atomic_write_json(state_dict, ckpt_path)

    state = ev.evolve(
        dt=dt,
        n_steps=N_STEPS,
        record_every=RECORD_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_callback=ckpt_cb,
        resume_from=resume_from,
        print_every=PRINT_EVERY,
        tag=name,
        extra_diagnostic_fn=tracker,
    )

    # Extract results
    times   = state['time_series']['times']
    E_total = state['time_series']['E_total']
    extras  = state.get('extra_diagnostics', [])

    # Compute E_bind series
    E_bind_series = [
        E_total[i] - n_osc * float(E_single_interp(times[i]))
        for i in range(len(times))
    ]

    # Extract displacement timeseries: list of lists
    disp_series = [ed['displacements'] for ed in extras]

    # Summary statistics
    all_disps = np.array(disp_series)  # shape (n_snapshots, n_osc)
    max_disp_any = float(np.max(all_disps)) if all_disps.size > 0 else 0.0
    final_disps  = all_disps[-1] if len(all_disps) > 0 else np.zeros(n_osc)
    mean_final   = float(np.mean(final_disps))
    std_final    = float(np.std(final_disps))

    # Max drift rate: max over oscillons of max(displacement / time)
    max_drift_rate = 0.0
    for j in range(n_osc):
        for i in range(1, len(times)):
            if times[i] > 0:
                rate = all_disps[i, j] / times[i]
                if rate > max_drift_rate:
                    max_drift_rate = rate
    max_drift_rate = float(max_drift_rate)

    E_bind_arr = np.array(E_bind_series)
    E_bind_mean = float(np.mean(E_bind_arr))
    E_bind_std  = float(np.std(E_bind_arr))

    result = {
        "completed": True,
        "config": name,
        "T_final": float(times[-1]),
        "n_oscillons": n_osc,
        "dt_diagnostic": DIAG_DT,
        "initial_positions": vertices.tolist(),
        "phases": [int(p) for p in phases],
        "centroid_timeseries": {
            "times": times,
            "displacements": disp_series,
        },
        "E_bind_timeseries": E_bind_series,
        "E_total_timeseries": E_total,
        "summary": {
            "max_displacement_any_oscillon": max_disp_any,
            "mean_displacement_final": mean_final,
            "std_displacement_final": std_final,
            "max_drift_rate": max_drift_rate,
            "E_bind_mean": E_bind_mean,
            "E_bind_std": E_bind_std,
        },
        "wall_seconds": state['wall_elapsed'],
        "parameters": {
            "N_grid": N, "L": L, "m": m, "g4": g4, "g6": g6,
            "dt": dt, "sigma": sigma, "phi0": phi0, "R": R,
        },
    }

    atomic_write_json(result, out_path)
    print("  [%s] Saved -> %s" % (name, out_path))

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print("  [%s] Checkpoint removed." % name)
    sys.stdout.flush()

    return result


# -- E_bind cross-check --------------------------------------------------------

def validate_E_bind(result, expected, tolerance=0.5):
    """Check E_bind at T~500 against expected value."""
    times = result['centroid_timeseries']['times']
    E_bind = result['E_bind_timeseries']
    # Find index closest to T=500
    import numpy as np
    t_arr = np.array(times)
    idx = np.argmin(np.abs(t_arr - 500.0))
    E_bind_500 = E_bind[idx]
    ok = abs(E_bind_500 - expected) < tolerance
    return ok, E_bind_500, float(t_arr[idx])


# -- Console summary -----------------------------------------------------------

def print_summary(results):
    dx = L / N

    print("")
    print("Centroid Tracking Results")
    print("=========================")

    for r in results:
        name  = r['config']
        n_osc = r['n_oscillons']
        T     = r['T_final']
        s     = r['summary']
        print("")
        print("%s (%d oscillons, T=%d):" % (name, n_osc, int(T)))
        print("  Max displacement (any oscillon, any time): %.6f grid spacings" % (
            s['max_displacement_any_oscillon'] / dx))
        print("  Mean final displacement: %.6f +/- %.6f" % (
            s['mean_displacement_final'] / dx, s['std_displacement_final'] / dx))
        print("  Max drift rate (displacement / time): %.3e" % s['max_drift_rate'])
        print("  E_bind: %.2f +/- %.4f" % (s['E_bind_mean'], s['E_bind_std']))

    print("")
    print("Grid spacing dx = %.5f" % dx)
    print("Oscillon radius R = %.1f (%.1f grid spacings)" % (R, R / dx))

    # Verdict
    max_disp_dx = max(r['summary']['max_displacement_any_oscillon'] / dx for r in results)
    if max_disp_dx < 0.5:
        verdict = "STABLE (max displacement < 0.5 dx)"
    elif max_disp_dx < 2.0:
        verdict = "MARGINAL (0.5 < max displacement < 2.0 dx)"
    else:
        verdict = "DRIFTING (max displacement > 2.0 dx)"
    print("")
    print("Verdict: %s" % verdict)


# -- Cleanup -------------------------------------------------------------------

def cleanup():
    checkpoints = glob.glob(os.path.join(OUTPUT_DIR, '*.checkpoint*'))
    temps = glob.glob(os.path.join(OUTPUT_DIR, '*.tmp'))
    total = 0
    for f in checkpoints + temps:
        total += os.path.getsize(f)
        os.remove(f)
    if checkpoints or temps:
        print("Cleanup: removed %d checkpoints, %d temp files, freed %.1f MB" % (
            len(checkpoints), len(temps), total / 1e6))


# -- Main ----------------------------------------------------------------------

def main():
    import numpy as np

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wall_start = time.perf_counter()

    print("centroid_tracking.py")
    print("Output dir: %s" % os.path.abspath(OUTPUT_DIR))
    print("")
    sys.stdout.flush()

    # Step 1: single oscillon baseline to T=2000
    baseline_t, baseline_E = run_baseline()
    print("")
    sys.stdout.flush()

    # Step 2: build configs
    cube_verts = cube_vertices()
    cube_phases = [-1, 1, 1, -1, 1, -1, -1, 1]  # bipartite, 12/12 cross-edges

    ico_verts  = icosahedron_vertices()
    ico_phases = [-1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1]  # ce_20

    configs = [
        {
            'name': 'cube_checkerboard',
            'vertices': cube_verts.tolist(),
            'phases': cube_phases,
            'output_path': os.path.join(OUTPUT_DIR, 'centroid_cube_checkerboard.json'),
            'baseline_t': baseline_t,
            'baseline_E': baseline_E,
        },
        {
            'name': 'ico_ce20',
            'vertices': ico_verts.tolist(),
            'phases': ico_phases,
            'output_path': os.path.join(OUTPUT_DIR, 'centroid_ico_ce20.json'),
            'baseline_t': baseline_t,
            'baseline_E': baseline_E,
        },
    ]

    # Step 3: run in parallel
    print("Running %d configs on %d workers ..." % (len(configs), N_WORKERS))
    sys.stdout.flush()

    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_multi_oscillon, configs)

    # Step 4: validate E_bind at T=500
    print("")
    print("E_bind cross-check at T=500:")
    expected = {
        'cube_checkerboard': -51.53,
        'ico_ce20': -50.11,
    }
    all_ok = True
    for r in results:
        name = r['config']
        ok, val, t_actual = validate_E_bind(r, expected[name])
        status = "PASS" if ok else "FAIL"
        print("  %s: E_bind(T=%.0f) = %.2f (expected %.2f) -- %s" % (
            name, t_actual, val, expected[name], status))
        if not ok:
            all_ok = False
            print("  *** VALIDATION FAILED for %s ***" % name)

    if not all_ok:
        print("")
        print("WARNING: E_bind cross-check failed. Results may be unreliable.")
        print("Stopping for diagnosis.")
        sys.stdout.flush()
        return

    # Step 5: print summary
    print_summary(results)

    # Step 6: cleanup
    cleanup()

    wall_total = time.perf_counter() - wall_start
    print("")
    print("Total wall time: %.0f s (%.1f min)" % (wall_total, wall_total / 60))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
