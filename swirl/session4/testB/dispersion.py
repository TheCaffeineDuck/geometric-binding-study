"""Test B, B1: linear dispersion relation of the three solvers, derived from their code,
and cheap numerical confirmations (single eigenmodes evolved by the solvers themselves).

Linearised about phi = 0 every solver has phi_tt = L phi - V''(0) phi with V''(0) = m^2
(read from the implemented V': radial `p (m2 - p2 (g4/6 - g6/120 p2))`, engine v2
`p (m2 + p2 (a6 p2 - a4))`, legacy `m_sq phi - g4/6 phi^3 + g6/120 phi^5`; each has
derivative m2 at p = 0). A spatial eigenmode of L with eigenvalue -K2(k) then obeys
a'' = -Omega^2 a, Omega^2 = m^2 + K2(k):

  Radial (u = r phi, 4th-order central stencil, odd reflections):
      K2(k) = (1 - cos th)(7 - cos th) / (3 dr^2),  th = k dr
            = k^2 (1 - th^4/90 + ...);  max 16/(3 dr^2) at th = pi
  Engine v2 Evolver (spectral, real FFT, Nyquist kept):  K2 = |k|^2 exactly, |k_i| <= pi/dx
  Legacy SexticEvolver (spectral, complex FFT):  K2 = |k|^2 + sigma dx^4 |k|^4
      (the "Kreiss-Oliger" term multiplies phi_hat, not phi_t_hat: a conservative
      stiffness, no damping)

Classical RK4 on zeta' = -i Omega zeta (zeta = Omega a + i a') multiplies zeta by
R(-i theta) per step, theta = Omega dt, R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24, so
    omega_num = arg R(i theta) / dt = Omega (1 - theta^4/120 + ...),
    |R|^2 = 1 - theta^6/72 + theta^8/576.
Threshold omega_th = omega_num(k -> 0) = arg R(i m dt)/dt.
"""

import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


# ---------------------------------------------------------------- symbols
def K2_radial(k, dr):
    th = np.asarray(k, dtype=float) * dr
    c = np.cos(th)
    return (1.0 - c) * (7.0 - c) / (3.0 * dr * dr)


def dK2dk_radial(k, dr):
    th = np.asarray(k, dtype=float) * dr
    return np.sin(th) * (8.0 - 2.0 * np.cos(th)) / (3.0 * dr)


def K2_spectral(k):
    return np.asarray(k, dtype=float) ** 2


def K2_legacy(k, dx, sigma):
    k2 = np.asarray(k, dtype=float) ** 2
    return k2 + sigma * dx**4 * k2 * k2


def rk4_omega(Omega, dt):
    """Numerical angular frequency of RK4 for an oscillator of frequency Omega."""
    th = np.asarray(Omega, dtype=float) * dt
    return np.arctan2(th - th**3 / 6.0, 1.0 - th**2 / 2.0 + th**4 / 24.0) / dt


def rk4_amp2(Omega, dt):
    th = np.asarray(Omega, dtype=float) * dt
    re = 1.0 - th**2 / 2.0 + th**4 / 24.0
    im = th - th**3 / 6.0
    return re * re + im * im


def omega_semi(K2, m=1.0):
    return np.sqrt(m * m + K2)


def threshold(m, dt):
    return float(rk4_omega(m, dt))


def group_velocity_radial(k, dr, m=1.0):
    return dK2dk_radial(k, dr) / (2.0 * omega_semi(K2_radial(k, dr), m))


def k_of_omega_continuum(omega, m=1.0):
    return np.sqrt(np.maximum(np.asarray(omega, dtype=float) ** 2 - m * m, 0.0))


# ---------------------------------------------------------------- measurement
def _phase_rate(zeta, t):
    ph = np.unwrap(np.angle(zeta))
    return -float((ph[-1] - ph[0]) / (t[-1] - t[0]))


def measure_radial_mode(n, R=40.0, dr=0.05, dt=0.0125, T=200.0, eps=1e-6, nonlinear=False,
                        m=1.0, g4=0.30, g6=0.12):
    """Evolve u = eps sin(n pi r / R) (an exact eigenvector of the discrete D2 with both
    odd reflections), v = 0, sponge off, in swirl.family.radial.Radial. Returns the
    measured frequency from the phase of zeta = Omega a + i a'."""
    from swirl.family.radial import Radial
    s = Radial(R, dr, m, g4 if nonlinear else 0.0, g6 if nonlinear else 0.0)
    k = n * np.pi / s.R
    sh = np.sin(k * s.r)
    sh[0] = sh[-1] = 0.0
    s.u = eps * sh.copy()
    s.v = np.zeros_like(s.u)
    Om = float(omega_semi(K2_radial(k, s.dr), m))
    nn = float(sh @ sh)
    chunk = max(1, int(1.0 / (Om * dt)))      # < 1 rad of phase per sample (no unwrap aliasing)
    zs, ts = [], []
    nsteps = int(round(T / dt)) // chunk * chunk
    for j in range(0, nsteps + 1, chunk):
        if j:
            s.step(dt, chunk)
        a, b = (s.u @ sh) / nn, (s.v @ sh) / nn
        zs.append(Om * a + 1j * b)
        ts.append(j * dt)
    zs, ts = np.array(zs), np.array(ts)
    return dict(n=n, k=k, kdr=k * s.dr, Omega_semi=Om, omega_pred=float(rk4_omega(Om, dt)),
                omega_meas=_phase_rate(zs, ts), amp2_per_step_meas=float(np.abs(zs[-1] / zs[0]) ** (2.0 / nsteps)),
                amp2_per_step_pred=float(rk4_amp2(Om, dt)), dt=dt, dr=s.dr, R=s.R, T=T,
                nonlinear=nonlinear, eps=eps)


def _plane_wave_3d(N, L, n_vec):
    x = -L / 2 + (L / N) * np.arange(N)
    k = 2 * np.pi * np.asarray(n_vec, dtype=float) / L
    ph = k[0] * x[:, None, None] + k[1] * x[None, :, None] + k[2] * x[None, None, :]
    return np.cos(ph), float(np.sqrt(k @ k))


def measure_engine_v2_mode(n_vec, N=16, L=8.8, dt=0.04, T=40.0, eps=1e-6, m=1.0):
    """Linear plane wave cos(k.x) in swirl.engine.evolver.Evolver (no absorber/filter)."""
    from swirl.engine.evolver import Evolver
    ev = Evolver(N, L, m, 0.0, 0.0, threads=1)
    mode, kk = _plane_wave_3d(N, L, n_vec)
    ev.set_state(eps * mode, np.zeros_like(mode))
    Om = float(omega_semi(K2_spectral(kk), m))
    nn = float(np.sum(mode * mode))
    zs, ts = [], []
    nsteps = int(round(T / dt))
    for j in range(nsteps + 1):
        if j:
            ev.step(dt)
        zs.append(Om * np.sum(ev.phi * mode) / nn + 1j * np.sum(ev.pi * mode) / nn)
        ts.append(j * dt)
    return dict(n_vec=list(n_vec), k=kk, kdx=kk * L / N, Omega_semi=Om,
                omega_pred=float(rk4_omega(Om, dt)), omega_meas=_phase_rate(np.array(zs), np.array(ts)),
                dt=dt, dx=L / N, T=T)


def load_legacy_sextic():
    """Import legacy/engine/evolver.py by path without writing bytecode into legacy/."""
    path = os.path.join(ROOT, "legacy", "engine", "evolver.py")
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("_legacy_engine_evolver", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod.SexticEvolver


def measure_legacy_mode(n_vec, N=16, L=12.5, dt=0.05, T=40.0, sigma=0.01, eps=1e-6, m=1.0):
    """Linear plane wave in the legacy SexticEvolver (g4 = g6 = 0, its sigma term on)."""
    SE = load_legacy_sextic()
    ev = SE(N, L, m, 0.0, 0.0, dissipation_sigma=sigma)
    mode, kk = _plane_wave_3d(N, L, n_vec)
    ev.set_initial_conditions(eps * mode, np.zeros_like(mode))
    dx = L / N
    Om = float(omega_semi(K2_legacy(kk, dx, sigma), m))
    nn = float(np.sum(mode * mode))
    zs, ts = [], []
    nsteps = int(round(T / dt))
    for j in range(nsteps + 1):
        if j:
            ev.step_rk4(dt)
        zs.append(Om * np.sum(ev.phi * mode) / nn + 1j * np.sum(ev.phi_dot * mode) / nn)
        ts.append(j * dt)
    return dict(n_vec=list(n_vec), k=kk, kdx=kk * dx, Omega_semi=Om,
                Omega_semi_without_sigma=float(omega_semi(kk * kk, m)),
                omega_pred=float(rk4_omega(Om, dt)), omega_meas=_phase_rate(np.array(zs), np.array(ts)),
                dt=dt, dx=dx, sigma=sigma, T=T)


def summary():
    """Numbers for docs/S4_TESTB_NOTES.md section 1 at the dt and grids actually used."""
    out = {}
    cases = dict(radial=dict(dt=0.0125, dr=0.05), radial_half=dict(dt=0.00625, dr=0.025),
                 engine_v2_production=dict(dt=0.04, dx=0.55), engine_v2_default=dict(dt=0.55 / 8, dx=0.55),
                 legacy=dict(dt=0.05, dx=50.0 / 64, sigma=0.01))
    for name, c in cases.items():
        dt = c["dt"]
        w_th = threshold(1.0, dt)
        row = dict(c, omega_th=w_th, omega_th_minus_m=w_th - 1.0, Omega_semi_k0=1.0)
        h = c.get("dr", c.get("dx"))
        kny = np.pi / h
        if name.startswith("radial"):
            K2n = K2_radial(kny, h)
            row["K2_max_over_continuum_at_pi"] = float(K2n / kny**2)
            row["omega_max_semi"] = float(np.sqrt(1 + K2n))
            k3 = float(k_of_omega_continuum(3 * 0.9091))
            row["k3_continuum"] = k3
            row["k3_rel_K2_error"] = float(K2_radial(k3, h) / k3**2 - 1)
            row["vg_at_k3"] = float(group_velocity_radial(k3, h))
        elif name.startswith("engine"):
            row["omega_max_semi_corner"] = float(np.sqrt(1 + 3 * kny**2))
        else:
            s = c["sigma"]
            row["stiffness_rel_at_axis_nyquist"] = float(s * (kny * h) ** 2)   # sigma dx^4 k^4 / k^2
            row["stiffness_rel_at_corner"] = float(s * 3 * (kny * h) ** 2)
            row["omega_semi_axis_nyquist"] = float(np.sqrt(1 + K2_legacy(kny, h, s)))
            row["omega_semi_axis_nyquist_no_sigma"] = float(np.sqrt(1 + kny**2))
            k3 = float(k_of_omega_continuum(3 * 0.9091))
            row["k3_shift_by_sigma"] = float(np.sqrt((-1 + np.sqrt(1 + 4 * s * h**4 * k3**2)) / (2 * s * h**4)) - k3)
        row["omega_max_dt"] = float(row.get("omega_max_semi", row.get("omega_max_semi_corner", 0.0)) * dt)
        out[name] = row
    return out
