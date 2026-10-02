"""WP-L (Session 3), L1: lone-lump break test in the radial solver.

Equation, grid, scheme and sponge are those of Session 2 (swirl.family.radial, grid_for in
wpa_run): u = r phi, 4th-order FD, RK4, dr = 0.05, dt = 0.0125, quadratic sponge from R_s
(100 x max(1, sqrt(beta / (4/3))) x scale) to R_box = 2 R_s. scale = 2 is the doubled box
(standard 4). Units Set-B-like: m = 1, g4 = 0.30, g6 = 0.09 beta.

Starts (STARTS): the Session 2 single-frequency profiles at beta = 4/3, omega = 0.906505 ("dip",
inside the long-life dip), beta = 4/3, omega = 0.909149 ("primary", Session 1, outside the dip) and
beta = 5, omega = 0.975598 ("b5"). hw is the profile half-width (Session 2 convention); r3 = 3 hw;
rN = max(3 hw, hw + 4.6/kappa) is the lump radius that encloses the tail (WP-A r_lump).

Settled state (settle). The profile start (phi = Phi, phi_t = 0) is evolved to T_SETTLE = 1000 and on
to the next maximum of phi(0, t) (the step where phi(0) peaks); that step is t_s, phase 0. The step
nearest the next downward zero of phi(0) is phase 90 (k90 steps after t_s). Every run starts from the
settled state at t_s; the undisturbed control is the same state continued (kind "none").

Disturbances (applied at t_d = t_s for phase 0, or t_s + k90 dt for phase 90):
  field    phi   -> phi   (1 + eps w(r)),  w = 1 for r <= 2 hw, C-infinity taper to 0 at 3 hw. Phase 0.
  velocity phi_t -> phi_t (1 + eps w(r)),  same w. Phase 90 (phi_t is largest there; at phase 0 it is ~0).
  removal  (phi, phi_t) -> (1 - s c(r)) (phi, phi_t), c = 1 for r <= 1 hw, taper to 0 at 1.5 hw.
           Both components are scaled, so the cut does not depend on the phase, and only the core is
           touched, so it carves a dent in the profile instead of rescaling it: N is removed from the core.
           (a) instead rescales one component over the whole ball. Phase 0.
  packet   an ingoing spherical packet u_p = Re psi, v_p = Re(i Omega psi), psi = A exp(-(r - R0)^2/2 sigma^2)
           exp(i k0 r), Omega = sqrt(m^2 - d_r^2) applied by FFT on the uniform r grid. For u this is the
           exact ingoing solution of the linear Klein-Gordon equation on the half line (far from r = 0 and
           from the lump). Its energy is 2 pi^1.5 sigma A^2 (m^2 + k0^2 + 1/(2 sigma^2)) up to
           exp(-k0^2 sigma^2) terms (packet_energy_exact). The scan variable is efrac = E_p / E_lump.
           Launch R0 = rN + 10 + 4 sigma by default. Phase 0 (the lump phase when the packet arrives is
           set by the travel time, not by t_d).
Every disturbance is stated by the fractional change it makes in the lump's N (and in E).

N estimators. N_inst = (1/(2 omega_s)) int w (phi_t^2 + omega_s^2 phi^2) d3x (instantaneous; exact for a
single-frequency oscillation; omega_s the settled omega). N per period = (C(z_{j+1}) - C(z_j)) / (2 pi),
with C the time integral of int w phi_t^2 d3x recorded every step and z_j the upward zero crossings of
phi(0, t) (Schmitt trigger as WP-A). This is <int w phi_t^2> / omega over whole periods, the Session 2
3D estimator (run.window_averages, lumps.summarize). Weights w: whole box, ball rN, ball 3 hw (cos^2
taper of width 2 at the edge, as lumps.ball_weight).

Energy bookkeeping. E_h (radial.energy_discrete) is exactly conserved without the sponge; the energy the
sponge removes, 4 pi dr sum sigma v^2, is integrated with the RK4 stage weights into `absorbed`, so
E_h + absorbed is conserved to the RK4 error. frac and E_in use Radial.energy_inside (Session 2).

Diagnostics per run: frac = E(r < 3 hw)/E(r < 3 hw, just before the disturbance); E(r < rN); omega per
period; N per period (three weights); E_h + absorbed; nonlinear share; the kick record (pre and post
values at t_d). compare(run, control): broken outright (within 2000 time units E(r < 3 hw) < 0.5 of its
pre-disturbance value while the control's frac stays above 0.9), windowed loss rate and omega shift
against the control, Delta N / N and Delta E / E against the control.
"""

import hashlib
import json
import multiprocessing as mp
import os
import time

import numpy as np
from numba import njit
from scipy.integrate import simpson

from swirl.family.profile import profile_curve, solve_profile
from swirl.family.radial import FOUR_PI, Radial, _accel
from swirl.family.wpa_run import (G4, M, _schmitt, energy_parts, grid_for, nonlinear_shares, r_lump,
                                  window_fits, write_json)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _code_hash():
    """sha256 (first 16 hex) of the modules that determine a run. It is part of every checkpoint and settle-cache
    key, so a code change never resumes from, or reuses, a state made by older code."""
    h = hashlib.sha256()
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("wpl.py", "radial.py", "profile.py", "wpa_run.py"):
        with open(os.path.join(here, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


CODE_HASH = _code_hash()
OUT = os.path.join(ROOT, "outputs", "session3", "wpL")
CKPT = os.path.join(ROOT, "checkpoints", "session3", "wpL")

STARTS = {
    "dip": (4.0 / 3.0, 0.9065045072253043),
    "primary": (4.0 / 3.0, 0.909149280617113),
    "b5": (5.0, 0.975598235049607),
}
T_SETTLE = 1000.0
SAMPLE_DT = 0.5
KICK_BALL = (2.0, 3.0)        # field/velocity kick: w = 1 to 2 hw, taper to 0 at 3 hw
CORE = (1.0, 1.5)             # removal: c = 1 to 1 hw, taper to 0 at 1.5 hw
BALL_TAPER = 2.0
BROKEN_HORIZON = 2000.0
EPS_TABLE = [-0.3, -0.2, -0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
REMOVAL_TABLE = [0.05, 0.1, 0.2, 0.3, 0.5]


# ---------------------------------------------------------------------------
# Kernel: RK4 steps of radial._advance (same arithmetic, bit-identical), plus per-step records
# ---------------------------------------------------------------------------

@njit(cache=True)
def _advance_rec(u, v, r, c2, m2, g4, g6, sig, dt, nsteps, W, dsc, phi0_rec, I_rec, absorbed):
    """nsteps RK4 steps. phi0_rec[k]: phi(0) after step k+1; I_rec[k, j] = sum_i W[j, i] v_i^2 after
    step k+1; absorbed[0] += dt * (RK4-weighted stage dissipation dsc * sum sigma v^2)."""
    n = u.shape[0]
    K = W.shape[0]
    a1 = np.empty(n); a2 = np.empty(n); a3 = np.empty(n); a4 = np.empty(n)
    ut = np.empty(n); vt = np.empty(n)
    for k in range(nsteps):
        _accel(u, v, r, c2, m2, g4, g6, sig, a1)
        d1 = 0.0
        for i in range(n):
            d1 += sig[i] * v[i] * v[i]
        for i in range(n):
            ut[i] = u[i] + 0.5 * dt * v[i]
            vt[i] = v[i] + 0.5 * dt * a1[i]
        _accel(ut, vt, r, c2, m2, g4, g6, sig, a2)
        d2 = 0.0
        for i in range(n):
            d2 += sig[i] * vt[i] * vt[i]
        k2u = vt.copy()
        for i in range(n):
            ut[i] = u[i] + 0.5 * dt * k2u[i]
            vt[i] = v[i] + 0.5 * dt * a2[i]
        _accel(ut, vt, r, c2, m2, g4, g6, sig, a3)
        d3 = 0.0
        for i in range(n):
            d3 += sig[i] * vt[i] * vt[i]
        k3u = vt.copy()
        for i in range(n):
            ut[i] = u[i] + dt * k3u[i]
            vt[i] = v[i] + dt * a3[i]
        _accel(ut, vt, r, c2, m2, g4, g6, sig, a4)
        d4 = 0.0
        for i in range(n):
            d4 += sig[i] * vt[i] * vt[i]
        for i in range(n):
            u[i] += dt / 6.0 * (v[i] + 2.0 * k2u[i] + 2.0 * k3u[i] + vt[i])
            v[i] += dt / 6.0 * (a1[i] + 2.0 * a2[i] + 2.0 * a3[i] + a4[i])
        u[0] = 0.0
        u[n - 1] = 0.0
        v[0] = 0.0
        v[n - 1] = 0.0
        phi0_rec[k] = (4.0 * u[1] / r[1] - u[2] / r[2]) / 3.0
        absorbed[0] += dt / 6.0 * (d1 + 2.0 * d2 + 2.0 * d3 + d4) * dsc
        for j in range(K):
            s = 0.0
            for i in range(n):
                s += W[j, i] * v[i] * v[i]
            I_rec[k, j] = s


@njit(cache=True)
def _rec_values(u, v, r, W, out_I):
    """phi(0) and sum_i W[j, i] v_i^2 with the kernel's arithmetic (used after a disturbance)."""
    K, n = W.shape
    for j in range(K):
        s = 0.0
        for i in range(n):
            s += W[j, i] * v[i] * v[i]
        out_I[j] = s
    return (4.0 * u[1] / r[1] - u[2] / r[2]) / 3.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def taper(r, r1, r2):
    """C-infinity step: 1 for r <= r1, 0 for r >= r2."""
    x = np.clip((np.asarray(r, float) - r1) / (r2 - r1), 0.0, 1.0)
    f = lambda y: np.where(y > 0, np.exp(-1.0 / np.maximum(y, 1e-300)), 0.0)
    a, b = f(1.0 - x), f(x)
    return a / (a + b)


def ball_weight(r, R, tap=BALL_TAPER):
    u = np.clip((np.asarray(r, float) - (R - tap)) / tap, 0.0, 1.0)
    return np.cos(0.5 * np.pi * u) ** 2


def _refine_crossing(T, V, tc):
    """Zero of the cubic through the 4 samples around the linear estimate tc (Newton from tc); the
    linear estimate when fewer than 4 samples are available or Newton leaves the interval."""
    i = int(np.searchsorted(T, tc, side="right")) - 1
    if T.size < 4 or i < 0 or i >= T.size - 1:
        return tc
    lo = int(np.clip(i - 1, 0, T.size - 4))
    c = np.polyfit(T[lo:lo + 4] - T[i], V[lo:lo + 4], 3)
    dc = np.polyder(c)
    x = tc - T[i]
    for _ in range(6):
        d = np.polyval(dc, x)
        if d == 0:
            return tc
        x -= np.polyval(c, x) / d
    return float(T[i] + x) if 0.0 <= x <= T[i + 1] - T[i] else tc


def _hermite(C0, C1, I0, I1, s, h):
    return ((2 * s ** 3 - 3 * s ** 2 + 1) * C0 + (s ** 3 - 2 * s ** 2 + s) * h * I0
            + (-2 * s ** 3 + 3 * s ** 2) * C1 + (s ** 3 - s ** 2) * h * I1)


def start_numbers(start):
    beta, omega = STARTS[start]
    prof = solve_profile(omega, 1.0, 1.0, beta, tol=1e-10)
    hw, kappa = prof["half_width"], prof["kappa"]
    return beta, omega, prof, dict(hw=hw, kappa=kappa, r3=3.0 * hw, rN=r_lump(hw, kappa),
                                   N_profile=prof["N"] / G4, E_profile=prof["E"] / G4)


def potential(phi, beta, linear=False):
    g4, g6 = (0.0, 0.0) if linear else (G4, beta * G4 ** 2)
    p2 = phi * phi
    return 0.5 * M ** 2 * p2 - g4 / 24.0 * p2 * p2 + g6 / 720.0 * p2 ** 3


# ---------------------------------------------------------------------------
# Evolution with records
# ---------------------------------------------------------------------------

class Evo:
    """A Radial solver plus the per-step records (phi(0), C for three weights, absorbed) and samples."""

    def __init__(self, beta, grid, linear, r3, rN, omega_ref, sample_dt=SAMPLE_DT):
        g4, g6 = (0.0, 0.0) if linear else (G4, beta * G4 ** 2)
        self.s = Radial(grid["R_box"], grid["dr"], M, g4, g6, sponge_start=grid["R_sponge"],
                        sigma_max=grid["sigma_max"])
        self.beta, self.linear = beta, linear
        self.dt = grid["dt"]
        self.nrec = int(round(sample_dt / self.dt))
        r = self.s.r
        self.W = np.ascontiguousarray(FOUR_PI * self.s.dr * np.vstack(
            [np.ones_like(r), ball_weight(r, rN), ball_weight(r, r3)]))
        self.rcs = [r3, rN]
        self.omega_ref = omega_ref
        self.c2 = 1.0 / (12.0 * self.s.dr ** 2)
        self.k = 0
        self.st = None

    # -- state ------------------------------------------------------------------
    def init_state(self, u, v, k):
        s = self.s
        s.u = np.array(u, float)
        s.v = np.array(v, float)
        self.k = int(k)
        s.t = self.k * self.dt
        I0 = np.empty(3)
        p0 = _rec_values(s.u, s.v, s.r_safe, self.W, I0)
        self.st = dict(series={k: [] for k in ("t", "E3", "EN", "Etot", "Eh", "absorbed", "Ninst", "phi0")},
                       C=[0.0, 0.0, 0.0], I_prev=I0.tolist(), absorbed=0.0, xc=[],
                       trig=dict(prev_v=float(p0), prev_t=s.t, armed=False, tc=[], amp=abs(float(p0)),
                                 amp_hist=[abs(float(p0))]),
                       hist=[[float(s.t)], [float(p0)]],
                       nl=dict(t=[], signed=[], absolute=[]), kick=None, nsamp=0)

    def after_disturbance(self):
        s = self.s
        I0 = np.empty(3)
        p0 = _rec_values(s.u, s.v, s.r_safe, self.W, I0)
        self.st["I_prev"] = I0.tolist()
        self.st["trig"]["prev_v"] = float(p0)
        self.st["hist"] = [[float(s.t)], [float(p0)]]

    def steps(self, n):
        s, st = self.s, self.st
        phi0 = np.empty(n)
        I = np.empty((n, 3))
        ab = np.array([float(st["absorbed"])])     # running total: accumulated step by step in the kernel
        _advance_rec(s.u, s.v, s.r_safe, self.c2, s.m ** 2, s.g4, s.g6, s.sig, self.dt, n, self.W,
                     FOUR_PI * s.dr, phi0, I, ab)
        k0 = self.k
        self.k += n
        s.t = self.k * self.dt
        st["absorbed"] = float(ab[0])
        Iall = np.vstack([np.array(st["I_prev"])[None, :], I])
        # sequential accumulation from the running C, so the records do not depend on the chunking
        Call = np.cumsum(np.vstack([np.array(st["C"])[None, :], 0.5 * self.dt * (Iall[:-1] + Iall[1:])]), axis=0)
        tall = (k0 + np.arange(n + 1)) * self.dt
        trig = st["trig"]
        trig["prev_t"] = float(tall[0])
        ntc = len(trig["tc"])
        _schmitt(trig, float(tall[1]), self.dt, phi0)
        hist = st["hist"]
        T = np.concatenate([hist[0], tall[1:]])
        V = np.concatenate([hist[1], phi0])
        for tc in trig["tc"][ntc:]:
            tc = _refine_crossing(T, V, tc)
            j = int(np.clip(np.floor((tc - tall[0]) / self.dt), 0, n - 1))
            sf = (tc - tall[j]) / self.dt
            C = _hermite(Call[j], Call[j + 1], Iall[j], Iall[j + 1], sf, self.dt)
            st["xc"].append([float(tc)] + [float(x) for x in C])
        st["C"] = [float(x) for x in Call[-1]]
        st["I_prev"] = [float(x) for x in I[-1]]
        st["hist"] = [[float(x) for x in T[-3:]], [float(x) for x in V[-3:]]]
        return phi0

    def values(self):
        s = self.s
        Ein, Etot = s.energy_inside(self.rcs)
        w = self.omega_ref
        Ninst = self.W @ (s.v ** 2 + w * w * s.u ** 2) / (2.0 * w)
        phi, _ = s.fields()
        return dict(t=float(s.t), E3=float(Ein[0]), EN=float(Ein[1]), Etot=float(Etot),
                    Eh=float(s.energy_discrete()), absorbed=float(self.st["absorbed"]),
                    Ninst=[float(x) for x in Ninst], phi0=float(phi[0]))

    def sample(self):
        st = self.st
        v = self.values()
        for k in st["series"]:
            st["series"][k].append(v[k])
        if st["nsamp"] % 100 == 0:
            sh = nonlinear_shares(energy_parts(self.s))
            st["nl"]["t"].append(float(self.s.t))
            st["nl"]["signed"].append(float(sh["signed"]))
            st["nl"]["absolute"].append(float(sh["absolute"]))
        st["nsamp"] += 1

    def advance_to(self, k_target, on_sample=None):
        while self.k < k_target:
            n = min(self.nrec - self.k % self.nrec, k_target - self.k)
            self.steps(n)
            if self.k % self.nrec == 0:
                self.sample()
                if on_sample is not None and on_sample():
                    return False
        return True


def scratch_phi0(beta, grid, linear, u, v, nsteps):
    """phi(0) after each of nsteps steps from (u, v), on a copy (the same kernel)."""
    g4, g6 = (0.0, 0.0) if linear else (G4, beta * G4 ** 2)
    s = Radial(grid["R_box"], grid["dr"], M, g4, g6, sponge_start=grid["R_sponge"], sigma_max=grid["sigma_max"])
    uu, vv = np.array(u, float), np.array(v, float)
    rec = np.empty(nsteps)
    _advance_rec(uu, vv, s.r_safe, 1.0 / (12.0 * s.dr ** 2), s.m ** 2, s.g4, s.g6, s.sig, grid["dt"], nsteps,
                 np.zeros((0, uu.size)), 0.0, rec, np.empty((nsteps, 0)), np.zeros(1))
    return rec, uu, vv


# ---------------------------------------------------------------------------
# Period records -> omega and N
# ---------------------------------------------------------------------------

def periods(xc):
    """Per period between consecutive upward crossings: t_mid, omega, N for the three weights."""
    x = np.asarray(xc, float)
    if x.shape[0] < 2:
        return dict(t=np.array([]), omega=np.array([]), N=np.zeros((0, 3)), t0=np.array([]), t1=np.array([]))
    dt = np.diff(x[:, 0])
    return dict(t=0.5 * (x[1:, 0] + x[:-1, 0]), omega=2 * np.pi / dt, N=np.diff(x[:, 1:], axis=0) / (2 * np.pi),
                t0=x[:-1, 0], t1=x[1:, 0])


def window_N(xc, t_lo, t_hi, j=1):
    """N (weight j) and omega over the whole periods inside [t_lo, t_hi]."""
    x = np.asarray(xc, float)
    sel = np.where((x[:, 0] >= t_lo) & (x[:, 0] <= t_hi))[0]
    if sel.size < 2:
        return None, None
    a, b = sel[0], sel[-1]
    n = b - a
    return float((x[b, 1 + j] - x[a, 1 + j]) / (2 * np.pi * n)), float(2 * np.pi * n / (x[b, 0] - x[a, 0]))


# ---------------------------------------------------------------------------
# Settled state
# ---------------------------------------------------------------------------

def _grid(beta, scale):
    return grid_for(beta, scale)


def settle(start, scale=1.0, t_settle=T_SETTLE, cache_dir=None, save_json=True):
    """Settled state for `start` on the box `scale`. Cached in cache_dir (default CKPT; npz, not committed)."""
    cache_dir = cache_dir or CKPT
    beta, omega, prof, nums = start_numbers(start)
    grid = _grid(beta, scale)
    params = dict(start=start, beta=beta, omega_profile=omega, scale=float(scale), t_settle=float(t_settle),
                  sample_dt=SAMPLE_DT, **grid, **{k: float(v) for k, v in nums.items()})
    pkey = json.dumps(dict(params, code_hash=CODE_HASH), sort_keys=True)
    path = os.path.join(cache_dir, "settled_%s_s%g_T%g.npz" % (start, scale, t_settle))
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            if str(z["pkey"]) == pkey:
                meta = json.loads(str(z["meta"]))
                return dict(meta, u=z["u"].copy(), v=z["v"].copy(), u90=z["u90"].copy(), v90=z["v90"].copy(),
                            grid=grid, params=params)
    ev = Evo(beta, grid, False, nums["r3"], nums["rN"], omega)
    r = ev.s.r
    u0 = r * profile_curve(prof, r) / np.sqrt(G4)
    u0[0] = u0[-1] = 0.0
    ev.init_state(u0, np.zeros_like(u0), 0)
    ev.sample()
    n_set = int(round(t_settle / ev.dt))
    ev.advance_to(n_set)
    P = 2 * np.pi / omega
    nlook = int(np.ceil(2.5 * P / ev.dt))
    rec, _, _ = scratch_phi0(beta, grid, False, ev.s.u, ev.s.v, nlook)
    phi_now = ev.s.fields()[0][0]
    seq = np.concatenate([[phi_now], rec])
    i_max = next(i for i in range(1, seq.size - 1) if seq[i - 1] < seq[i] >= seq[i + 1])
    m = next(i for i in range(i_max + 1, seq.size) if seq[i - 1] > 0.0 >= seq[i])
    i90 = m if abs(seq[m]) < abs(seq[m - 1]) else m - 1
    ev.advance_to(n_set + i_max)
    k_s = ev.k
    t_s = k_s * ev.dt
    u, v = ev.s.u.copy(), ev.s.v.copy()
    _, u90, v90 = scratch_phi0(beta, grid, False, u, v, i90 - i_max)
    per = periods(ev.st["xc"])
    xc = np.asarray(ev.st["xc"])
    N_s, w_s = window_N(xc, t_s - 200.0, t_s, 1)
    N_lag, w_lag = window_N(xc, t_s - 400.0, t_s - 200.0, 1)
    selp = (per["t"] >= t_s - 500.0)
    Nper = per["N"][selp, 1]
    sel0 = per["t"] >= 50.0
    s0 = ev.st["series"]
    meta = dict(start=start, k_s=k_s, t_s=t_s, k90=int(i90 - i_max), t_90=(k_s + i90 - i_max) * ev.dt,
                omega_s=w_s, omega_lag=w_lag, omega_drift=abs(w_s - w_lag), N_s=N_s, N_lag=N_lag,
                N_box_s=window_N(xc, t_s - 200.0, t_s, 0)[0], N_3hw_s=window_N(xc, t_s - 200.0, t_s, 2)[0],
                N_period_rel_std_last500=float(np.std(Nper) / np.mean(Nper)),
                N_period_rel_range_last500=float((Nper.max() - Nper.min()) / np.mean(Nper)),
                Eh_t0=s0["Eh"][0], Eh_ts=float(ev.s.energy_discrete()), absorbed_settle=ev.st["absorbed"],
                E_plus_absorbed_rel_dev=float(np.max(np.abs(np.array(s0["Eh"]) + np.array(s0["absorbed"])
                                                            - s0["Eh"][0])) / s0["Eh"][0]))
    vals = ev.values()
    meta.update(E3_ts=vals["E3"], EN_ts=vals["EN"], Etot_ts=vals["Etot"], phi0_ts=vals["phi0"])
    if save_json:
        write_json(os.path.join(OUT, "settle", "settle_%s_s%g.json" % (start, scale)),
                   dict(params=params, meta=meta,
                        periods=dict(t=per["t"][sel0], omega=per["omega"][sel0], N_box=per["N"][sel0, 0],
                                     N_rN=per["N"][sel0, 1], N_3hw=per["N"][sel0, 2]),
                        series=dict(t=s0["t"][::10], E3=s0["E3"][::10], EN=s0["EN"][::10], Eh=s0["Eh"][::10],
                                    absorbed=s0["absorbed"][::10])))
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, u=u, v=v, u90=u90, v90=v90, meta=json.dumps(meta), pkey=pkey)
    os.replace(tmp, path)
    return dict(meta, u=u, v=v, u90=u90, v90=v90, grid=grid, params=params)


# ---------------------------------------------------------------------------
# Disturbances and t = 0 quadrature
# ---------------------------------------------------------------------------

def packet_energy_exact(A, sigma, k0, m=M):
    """Energy of the ingoing Gaussian packet (up to exp(-k0^2 sigma^2) terms)."""
    return 2.0 * np.pi ** 1.5 * sigma * A * A * (m * m + k0 * k0 + 0.5 / sigma ** 2)


def packet_velocity_exact(sigma, k0, m=M, nk=20001):
    """Energy-centroid speed P/E of the free packet: sum omega k |a|^2 / sum omega^2 |a|^2, |a|^2 ~
    exp(-(k - k0)^2 sigma^2), k > 0 (exact for the linear Klein-Gordon u-equation)."""
    k = np.linspace(max(1e-9, k0 - 12 / sigma), k0 + 12 / sigma, nk)
    w = np.sqrt(m * m + k * k)
    a2 = np.exp(-((k - k0) * sigma) ** 2)
    return float(np.sum(w * k * a2) / np.sum(w * w * a2))


def packet_uv(r, k0, sigma, R0, A, m=M):
    psi = A * np.exp(-0.5 * ((r - R0) / sigma) ** 2) * np.exp(1j * k0 * r)
    dr = r[1] - r[0]
    k = 2 * np.pi * np.fft.fftfreq(r.size, dr)
    v = np.real(np.fft.ifft(1j * np.sqrt(m * m + k * k) * np.fft.fft(psi)))
    u = np.real(psi)
    u[0] = u[-1] = 0.0
    v[0] = v[-1] = 0.0
    return u, v


def packet_defaults(start, packet):
    _, _, _, nums = start_numbers(start)
    p = dict(packet)
    p.setdefault("R0", nums["rN"] + 10.0 + 4.0 * p["sigma"])
    return p


def packet_scale(start, packet, base=1.0):
    """Smallest box scale (in steps of 0.25) with R0 + 5 sigma + 10 inside the sponge start."""
    beta = STARTS[start][0]
    p = packet_defaults(start, packet)
    need = p["R0"] + 5.0 * p["sigma"] + 10.0
    R1 = grid_for(beta, 1.0)["R_sponge"]
    return max(base, float(np.ceil(4.0 * need / R1) / 4.0))


def disturb(spec, r, u, v, nums, E_lump):
    """(u, v) after the disturbance, and an info dict. 'none' returns the inputs unchanged."""
    kind = spec["kind"]
    hw = nums["hw"]
    if kind == "none":
        return u, v, {}
    if kind in ("field", "velocity"):
        a, b = spec.get("ball", KICK_BALL)
        w = 1.0 + spec["eps"] * taper(r, a * hw, b * hw)
        return (u * w, v, {}) if kind == "field" else (u, v * w, {})
    if kind == "removal":
        a, b = spec.get("core", CORE)
        f = 1.0 - spec["s"] * taper(r, a * hw, b * hw)
        return u * f, v * f, {}
    if kind == "packet":
        p = spec["packet"]
        Ep = p["efrac"] * E_lump
        A = float(np.sqrt(Ep / packet_energy_exact(1.0, p["sigma"], p["k0"])))
        du, dv = packet_uv(r, p["k0"], p["sigma"], p["R0"], A)
        return u + du, v + dv, dict(A=A, E_packet_exact=packet_energy_exact(A, p["sigma"], p["k0"]),
                                    E_packet_quad=quad_packet_energy(r, du, dv))
    raise ValueError("unknown disturbance kind %s" % kind)


def _deriv4(f, dr):
    """4th-order central derivative of an even function sampled at r_i = i dr (even extension at 0)."""
    ext = np.concatenate([f[2:0:-1], f, [f[-1], f[-1]]])
    d = (-ext[4:] + 8 * ext[3:-1] - 8 * ext[1:-3] + ext[:-4]) / (12 * dr)
    d[0] = 0.0
    d[-2:] = np.gradient(f, dr, edge_order=2)[-2:]
    return d


def quad_packet_energy(r, u, v, m=M):
    """4 pi int (v^2 + u_r^2 + m^2 u^2)/2 dr (Simpson, spectral u_r): the linear energy of a packet
    supported away from r = 0."""
    dr = r[1] - r[0]
    k = 2 * np.pi * np.fft.fftfreq(r.size, dr)
    ur = np.real(np.fft.ifft(1j * k * np.fft.fft(u)))
    return float(FOUR_PI * simpson(0.5 * (v * v + ur * ur + m * m * u * u), x=r))


def quad_values(r, u, v, beta, omega_ref, nums, linear=False):
    """Independent quadrature of E (total and r < rN) and N_inst (ball rN) of a state: phi = u/r,
    4th-order derivative, Simpson's rule (the run uses the SBP energy E_h and trapezoid sums)."""
    rs = r.copy()
    rs[0] = 1.0
    phi, pt = u / rs, v / rs
    phi[0] = (4 * phi[1] - phi[2]) / 3.0
    pt[0] = (4 * pt[1] - pt[2]) / 3.0
    pr = _deriv4(phi, r[1] - r[0])
    e = 0.5 * pt * pt + 0.5 * pr * pr + potential(phi, beta, linear)
    f = FOUR_PI * r * r
    sel = r <= nums["rN"]
    wN = ball_weight(r, nums["rN"])
    return dict(E=float(simpson(f * e, x=r)), E_rN=float(simpson((f * e)[sel], x=r[sel])),
                N_rN=float(simpson(f * wN * (pt * pt + omega_ref ** 2 * phi * phi), x=r) / (2 * omega_ref)))


def t0_values(spec, settled, nums, beta):
    """Delta N / N and Delta E / E of the disturbance at t_d, by quadrature, before any PDE step."""
    r = grid_r(settled)
    ph = spec_phase(spec)
    u, v = (settled["u90"], settled["v90"]) if ph == 90 else (settled["u"], settled["v"])
    w = settled["omega_s"]
    pre = quad_values(r, u, v, beta, w, nums, spec.get("linear", False))
    u2, v2, info = disturb(spec, r, u, v, nums, settled["EN_ts"])
    post = quad_values(r, u2, v2, beta, w, nums, spec.get("linear", False))
    out = dict(pre=pre, post=post, dE=post["E"] - pre["E"], dE_over_Elump=(post["E"] - pre["E"]) / pre["E_rN"],
               dN=post["N_rN"] - pre["N_rN"], dN_over_N=(post["N_rN"] - pre["N_rN"]) / pre["N_rN"], phase=ph)
    out.update(info)
    return out


def grid_r(settled):
    n = settled["u"].size
    return np.arange(n) * (settled["grid"]["R_box"] / (n - 1))


def spec_phase(spec):
    if spec.get("phase") is not None:
        return int(spec["phase"])
    return 90 if spec["kind"] == "velocity" else 0


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

SPEC_DEFAULTS = dict(kind="none", eps=0.0, s=0.0, packet=None, phase=None, T=2000.0, scale=1.0,
                     linear=False, t_settle=T_SETTLE, sample_dt=SAMPLE_DT, W_gamma=500.0)


def full_spec(spec):
    p = dict(SPEC_DEFAULTS)
    p.update(spec)
    if p["kind"] == "packet":
        p["packet"] = packet_defaults(p["start"], p["packet"])
    p["phase"] = spec_phase(p)
    return p


def spec_name(spec):
    p = full_spec(spec)
    k = p["kind"]
    tag = {"none": "control", "field": "field_e%+g" % p["eps"], "velocity": "vel_e%+g" % p["eps"],
           "removal": "rem_s%g" % p["s"]}.get(k)
    if k == "packet":
        tag = "pk_k%g_sig%g_ef%g" % (p["packet"]["k0"], p["packet"]["sigma"], p["packet"]["efrac"])
    return "%s_%s_T%g_s%g%s" % (p["start"], tag, p["T"], p["scale"], "_lin" if p["linear"] else "")


def run_one(spec, ckpt=None, ckpt_every=1000.0, stop_after=None, return_state=False):
    """Run one disturbance (or the control) from the settled state. T is counted from t_s."""
    p = full_spec(spec)
    beta, omega, prof, nums = start_numbers(p["start"])
    settled = settle(p["start"], p["scale"], p["t_settle"])
    grid = settled["grid"]
    params = dict(p, beta=beta, omega_profile=omega, m=M, g4_units=G4,
                  g4_run=0.0 if p["linear"] else G4, g6_run=0.0 if p["linear"] else beta * G4 ** 2,
                  grid=grid, numbers=nums, t_s=settled["t_s"], k_s=settled["k_s"], k90=settled["k90"],
                  omega_s=settled["omega_s"], kick_ball_hw=list(KICK_BALL), core_hw=list(CORE),
                  ball_taper=BALL_TAPER, code_hash=CODE_HASH)
    pkey = json.dumps(params, sort_keys=True, default=float)
    if p["kind"] == "packet":
        pk = p["packet"]
        if pk["R0"] + 5.0 * pk["sigma"] + 10.0 > grid["R_sponge"] or pk["R0"] - 4.0 * pk["sigma"] < nums["rN"] + 5.0:
            raise ValueError("packet does not fit between the lump and the sponge; use scale >= %g"
                             % packet_scale(p["start"], pk))
    t0 = time.perf_counter()
    t0q = None if p["kind"] == "none" else t0_values(p, settled, nums, beta)
    ev = Evo(beta, grid, p["linear"], nums["r3"], nums["rN"], settled["omega_s"], p["sample_dt"])
    k_s = settled["k_s"]
    k_d = k_s + (settled["k90"] if p["phase"] == 90 else 0)
    k_end = k_s + int(round(p["T"] / ev.dt))
    wall_prior = 0.0
    if ckpt is not None and os.path.exists(ckpt):
        with np.load(ckpt, allow_pickle=False) as z:
            if str(z["pkey"]) != pkey:
                raise RuntimeError("checkpoint %s has different parameters" % ckpt)
            ev.s.u, ev.s.v = z["u"].copy(), z["v"].copy()
            ev.k = int(z["k"])
            ev.s.t = ev.k * ev.dt
            ev.st = json.loads(str(z["state"]))
            wall_prior = float(z["wall"])
    else:
        ev.init_state(settled["u"], settled["v"], k_s)
        ev.sample()
    last = [ev.k]
    nck = int(round(ckpt_every / ev.dt))
    k_stop = k_end if stop_after is None else min(k_end, k_s + int(round(stop_after / ev.dt)))

    def save():
        if ckpt is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(ckpt)), exist_ok=True)
        tmp = ckpt + ".tmp.npz"
        np.savez(tmp, u=ev.s.u, v=ev.s.v, k=ev.k, state=json.dumps(ev.st), pkey=pkey,
                 wall=wall_prior + time.perf_counter() - t0)
        os.replace(tmp, ckpt)

    def on_sample():
        if ev.k - last[0] >= nck:
            save()
            last[0] = ev.k
        return False

    if p["kind"] != "none" and ev.st["kick"] is None:
        ev.advance_to(min(k_d, k_stop), on_sample)
        if ev.k == k_d:
            pre = ev.values()
            r = ev.s.r
            u2, v2, info = disturb(p, r, ev.s.u, ev.s.v, nums, settled["EN_ts"])
            ev.s.u, ev.s.v = u2, v2
            ev.after_disturbance()
            post = ev.values()
            phi, pt = ev.s.fields()
            sh = nonlinear_shares(energy_parts(ev.s))
            ev.st["kick"] = dict(k=k_d, t=k_d * ev.dt, pre=pre, post=post, info=info,
                                 nonlinear_share_post=dict(signed=float(sh["signed"]), absolute=float(sh["absolute"])),
                                 phase_measured_deg=float(np.degrees(np.arctan2(
                                     -pt[0] / settled["omega_s"], phi[0])) % 360.0))
    ev.advance_to(k_stop, on_sample)
    if ev.k < k_end:
        save()
        return dict(params=params, partial=True, t=ev.k * ev.dt)
    if ckpt is not None:
        save()
    res = summarize(params, ev.st, t0q)
    res["wall_s"] = wall_prior + time.perf_counter() - t0
    if return_state:
        res["_u"], res["_v"] = ev.s.u.copy(), ev.s.v.copy()
    return res


def summarize(params, st, t0q):
    s = {k: np.asarray(v, float) for k, v in st["series"].items()}
    t = s["t"]
    kick = st["kick"]
    t_s = params["t_s"]
    t_d = kick["t"] if kick else t_s
    E3_ref = kick["pre"]["E3"] if kick else s["E3"][0]
    per = periods(st["xc"])
    Epa = s["Eh"] + s["absorbed"]
    after = t >= t_d
    seg = lambda m, ref: float(np.max(np.abs(Epa[m] - ref)) / ref) if m.sum() > 0 else 0.0
    nl = st["nl"]
    ref_after = (kick["post"]["Eh"] + kick["post"]["absorbed"]) if kick else Epa[0]
    gam = window_fits(t, s["E3"], t_s + 250.0, params["W_gamma"], t[-1], per["t"], per["omega"])
    first = None
    if kick:
        n_after = per["t0"] >= t_d
        if n_after.any():
            j = np.where(n_after)[0][0]
            first = dict(t=float(per["t"][j]), N_rN=float(per["N"][j, 1]), omega=float(per["omega"][j]))
    t0_check = None
    if kick and t0q:
        dE_run = kick["post"]["Eh"] - kick["pre"]["Eh"]
        dN_run = kick["post"]["Ninst"][1] - kick["pre"]["Ninst"][1]
        t0_check = dict(dE_quad=t0q["dE"], dE_run=dE_run, dE_rel_diff=(dE_run - t0q["dE"]) / max(abs(t0q["dE"]), 1e-300),
                        dN_quad=t0q["dN"], dN_run=dN_run, dN_rel_diff=(dN_run - t0q["dN"]) / max(abs(t0q["dN"]), 1e-300),
                        dN_over_N_quad=t0q["dN_over_N"], dE_over_Elump_quad=t0q["dE_over_Elump"],
                        dN_over_N_run=dN_run / kick["pre"]["Ninst"][1], dE_over_Elump_run=dE_run / kick["pre"]["EN"])
    return dict(
        params=params, t_d=t_d, kick=kick, t0=t0q, t0_check=t0_check,
        regime=dict(omega_over_m_first_period_after=first["omega"] if first else None,
                    omega_over_m_end=float(np.median(per["omega"][per["t"] >= t[-1] - 100])) if per["t"].size else None,
                    nonlinear_share_signed_first=nl["signed"][0] if nl["signed"] else None,
                    nonlinear_share_abs_first=nl["absolute"][0] if nl["absolute"] else None,
                    nonlinear_share_abs_mean=float(np.mean(nl["absolute"])) if nl["absolute"] else None,
                    nonlinear_share_after_disturbance=kick.get("nonlinear_share_post") if kick else None),
        E_plus_absorbed=dict(rel_dev_before=seg(t <= t_d, Epa[0]) if kick else seg(np.ones_like(t, bool), Epa[0]),
                             rel_dev_after=seg(t > t_d, ref_after)),
        first_period_after=first,
        series=dict(t=t, frac=s["E3"] / E3_ref, E3=s["E3"], EN=s["EN"], Etot=s["Etot"], Eh=s["Eh"],
                    absorbed=s["absorbed"], Ninst_rN=[x[1] for x in st["series"]["Ninst"]], phi0=s["phi0"]),
        periods=dict(t=per["t"], omega=per["omega"], N_box=per["N"][:, 0], N_rN=per["N"][:, 1], N_3hw=per["N"][:, 2]),
        crossings=np.asarray(st["xc"]), gamma_windows=gam, nl=nl,
        frac_end=float(s["E3"][-1] / E3_ref), frac_min=float(np.min(s["E3"][after] / E3_ref)))


# ---------------------------------------------------------------------------
# Run against its control
# ---------------------------------------------------------------------------

def _per_series(res):
    p = res["periods"]
    return np.asarray(p["t"]), np.asarray(p["N_rN"]), np.asarray(p["omega"])


def compare(res, ctrl, horizon=BROKEN_HORIZON, windows=((50, 250), (250, 750), (750, 2000)), nwin=4, tol=None):
    """Broken outright, Delta N / N, Delta E / E, omega shift and loss rate against the control.
    A window whose end lies past the last sample by at most `tol` (default half a sample step) is kept and
    read to the last sample (`window_used`): the last sample of a T-long run falls just short of t_d + T."""
    t = np.asarray(res["series"]["t"])
    tc = np.asarray(ctrl["series"]["t"])
    if t.size != tc.size or np.max(np.abs(t - tc)) > 1e-9:
        raise ValueError("run and control are not on the same sample grid")
    t_d = res["t_d"]
    E3 = np.asarray(res["series"]["E3"])
    E3c = np.asarray(ctrl["series"]["E3"])
    E3_pre = res["kick"]["pre"]["E3"] if res["kick"] else E3[0]
    i_d = int(np.argmin(np.abs(t - t_d)))
    hz = (t >= t_d) & (t <= t_d + horizon)
    frac = E3 / E3_pre
    fc = E3c / E3c[i_d]
    below = np.where(hz & (frac < 0.5))[0]
    broken = dict(horizon=horizon, run_min_frac=float(np.min(frac[hz])),
                  first_below_half=float(t[below[0]] - t_d) if below.size else None,
                  control_min_frac=float(np.min(fc[hz])), control_ok=bool(np.min(fc[hz]) > 0.9),
                  covered_to=float(min(t[-1] - t_d, horizon)))
    broken["broken_outright"] = bool(below.size and broken["control_ok"])
    tr, Nr, wr = _per_series(res)
    tcc, Nc, wc = _per_series(ctrl)
    rows = []
    if tol is None:
        tol = 0.5 * float(np.median(np.diff(t))) if t.size > 1 else 0.0
    for a, b in windows:
        lo, hi = t_d + a, t_d + b
        if hi > t[-1] + tol:
            continue
        hi = min(hi, float(t[-1]))
        xr = np.asarray(res["crossings"])
        xcc = np.asarray(ctrl["crossings"])
        Nw, ww = window_N(xr, lo, hi, 1)
        Nwc, wwc = window_N(xcc, lo, hi, 1)
        sel = (t >= lo) & (t <= hi)
        rows.append(dict(window=[a, b], window_used=[a, round(hi - t_d, 6)], N=Nw, N_control=Nwc, dN_over_N=(Nw / Nwc - 1) if Nw and Nwc else None,
                         omega=ww, omega_control=wwc, domega=(ww - wwc) if ww and wwc else None,
                         EN_over_control=float(np.mean(np.asarray(res["series"]["EN"])[sel])
                                               / np.mean(np.asarray(ctrl["series"]["EN"])[sel]) - 1),
                         E3_over_control=float(np.mean(E3[sel]) / np.mean(E3c[sel]) - 1)))
    g = res["gamma_windows"]
    gc = ctrl["gamma_windows"]
    gpairs = [dict(t0=a["t0"], t1=a["t1"], Gamma=a["Gamma"], Gamma_control=b["Gamma"], sigma=a["sigma"],
                   sigma_control=b["sigma"]) for a, b in zip(g, gc) if abs(a["t0"] - b["t0"]) < 1e-6]
    first = None
    if res["first_period_after"]:
        fp = res["first_period_after"]
        jc = int(np.argmin(np.abs(tcc - fp["t"])))
        first = dict(t=fp["t"], dN_over_N=fp["N_rN"] / Nc[jc] - 1, domega=fp["omega"] - wc[jc])
    return dict(broken=broken, windows=rows, gamma=gpairs, first_period=first,
                frac_end=res["frac_end"], frac_end_control=float(E3c[-1] / E3c[i_d]))


# ---------------------------------------------------------------------------
# Free packet (linear, no lump): passage time, centroid speed, focus amplitude
# ---------------------------------------------------------------------------

def free_packet(start, packet, scale, T=None, sample_dt=1.0):
    """The packet alone in the linear equation on the run's grid. Returns the centroid track before the
    packet reaches the centre, the measured speed against packet_velocity_exact, max |phi(0, t)|, and
    t_pass: the first time after the peak at which E(r < rN + 5) < 1e-4 E_p (time after launch)."""
    beta = STARTS[start][0]
    p = packet_defaults(start, packet)
    _, _, _, nums = start_numbers(start)
    grid = grid_for(beta, scale)
    s = Radial(grid["R_box"], grid["dr"], M, 0.0, 0.0, sponge_start=grid["R_sponge"], sigma_max=grid["sigma_max"])
    A = p.get("A", 1.0)
    u, v = packet_uv(s.r, p["k0"], p["sigma"], p["R0"], A)
    s.u, s.v = u, v
    E0 = s.energy_discrete()
    vg = packet_velocity_exact(p["sigma"], p["k0"])
    if T is None:
        T = (p["R0"] + nums["rN"] + 8 * p["sigma"]) / vg + 200.0
    n = int(round(sample_dt / grid["dt"]))
    tt, cen, Ein, phi0max = [], [], [], 0.0
    rc = nums["rN"] + 5.0
    for k in range(int(round(T / sample_dt))):
        rec = s.step(grid["dt"], n)
        phi0max = max(phi0max, float(np.max(np.abs(rec))))
        e, _ = s.density()
        f = FOUR_PI * e * s.r ** 2
        tt.append(s.t)
        cen.append(float(np.sum(f * s.r) / np.sum(f)))
        Ein.append(float(np.sum(f[s.r <= rc]) * s.dr))
    tt, cen, Ein = map(np.asarray, (tt, cen, Ein))
    t_free = (p["R0"] - 5 * p["sigma"]) / vg
    sel = tt <= max(t_free, 5 * sample_dt)
    speed = -np.polyfit(tt[sel], cen[sel], 1)[0]
    ip = int(np.argmax(Ein))
    after = np.where((np.arange(tt.size) > ip) & (Ein < 1e-4 * E0))[0]
    return dict(packet=p, scale=scale, E0_h=float(E0), E0_exact=packet_energy_exact(A, p["sigma"], p["k0"]),
                v_exact=vg, v_measured=float(speed), v_rel_err=float(speed / vg - 1), fit_until=float(tt[sel][-1]),
                t_peak=float(tt[ip]), t_pass=float(tt[after[0]]) if after.size else None,
                phi0_max_over_A=phi0max / A, E_in_peak_frac=float(Ein[ip] / E0))


# ---------------------------------------------------------------------------
# t = 0 table (no PDE beyond the settling)
# ---------------------------------------------------------------------------

def t0_table(starts=("dip", "primary", "b5"), eps=EPS_TABLE, removal=REMOVAL_TABLE):
    out = {}
    for st in starts:
        beta, omega, prof, nums = start_numbers(st)
        settled = settle(st)
        rows = []
        for kind in ("field", "velocity"):
            for e in eps:
                q = t0_values(dict(kind=kind, eps=e, start=st), settled, nums, beta)
                rows.append(dict(kind=kind, eps=e, phase=q["phase"], dN_over_N=q["dN_over_N"],
                                 dE_over_Elump=q["dE_over_Elump"], dE=q["dE"]))
        for s_ in removal:
            q = t0_values(dict(kind="removal", s=s_, start=st), settled, nums, beta)
            rows.append(dict(kind="removal", s=s_, phase=0, dN_over_N=q["dN_over_N"],
                             dE_over_Elump=q["dE_over_Elump"], dE=q["dE"]))
        pre = quad_values(np.linspace(0, settled["grid"]["R_box"], settled["u"].size), settled["u"], settled["v"],
                          beta, settled["omega_s"], nums)
        out[st] = dict(beta=beta, omega_profile=omega, omega_s=settled["omega_s"], t_s=settled["t_s"],
                       N_inst_rN=pre["N_rN"], N_period_rN=settled["N_s"], E_rN=pre["E_rN"], E_box=pre["E"],
                       hw=nums["hw"], rN=nums["rN"], rows=rows)
    return out


# ---------------------------------------------------------------------------
# Scans: process pool
# ---------------------------------------------------------------------------

def _scan_one(args):
    spec, out_dir, ckpt_dir = args
    name = spec_name(spec)
    path = os.path.join(out_dir, name + ".json")
    if os.path.exists(path):
        return name, "exists"
    res = run_one(spec, ckpt=os.path.join(ckpt_dir, name + ".npz"))
    write_json(path, res)
    return name, res["wall_s"]


def scan(specs, workers=2, out_dir=None, ckpt_dir=None):
    """Run specs in a process pool (one thread each); results to out_dir/<name>.json; resumable.
    Settled states are made first, serially, so the workers only read them."""
    out_dir = out_dir or os.path.join(OUT, "runs")
    ckpt_dir = ckpt_dir or os.path.join(CKPT, "runs")
    os.environ["NUMBA_NUM_THREADS"] = "1"
    for key in sorted({(s["start"], full_spec(s)["scale"]) for s in specs}):
        settle(key[0], key[1])
    args = [(s, out_dir, ckpt_dir) for s in specs]
    if workers <= 1:
        return [_scan_one(a) for a in args]
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(_scan_one, args, chunksize=1)
