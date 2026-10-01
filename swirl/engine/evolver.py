"""Engine v2: 3D spectral RK4 evolver for the sextic real scalar field.

Equation of motion (periodic box, side L, N^3 points, x in [-L/2, L/2)):

    phi_tt = lap(phi) - m^2 phi + (g4/6) phi^3 - (g6/120) phi^5
             - gamma(x) phi_t                      (absorber, optional)
             - F^-1[ nu(k) F[phi_t] ]              (high-k damping, optional)

    V(phi) = m^2 phi^2/2 - g4 phi^4/24 + g6 phi^6/720

The Laplacian is spectral, multiplication by -K2 with K = 2 pi fftfreq(N, dx)
(Nyquist included), using real FFTs. The energy uses the same K2, so

    H = sum_x [phi_t^2/2 + V(phi)] dV + (dV/2N^3) sum_k K2 |phi_hat|^2

is exactly conserved by the semi-discrete system without absorber and filter;
the only drift is the RK4 time error (for a linear mode of frequency w the
RK4 amplification is |R|^2 = 1 - (w dt)^6/72 + ..., i.e. a slow monotone
energy loss ~ T w^6 dt^5 / 72).

Both optional terms act on phi_t and are true damping: they remove energy at
the rate D = sum_x phi_t (gamma phi_t + filt) dV >= 0. D is integrated with
the same RK4 weights into `absorbed`, so E + absorbed is conserved to the
RK4 error. The legacy "Kreiss-Oliger" term (sigma dx^4 k^4 phi on phi_tt, a
conservative stiffness left out of the legacy energy, LESSONS 7) is removed.
"""

import numpy as np
import numba
from numba import njit, prange
import scipy.fft as sfft


# ---------------------------------------------------------------------------
# Kernels (flat 1D views of C-contiguous arrays)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _stage(lap, phi_in, pi_in, phi0, pi0, acc_phi, acc_pi, phi_out, pi_out,
           gamma, use_gamma, filt, use_filt, m2, a4, a6, w, c, first, last):
    """One RK4 stage, pointwise.

    a = lap - V'(phi_in) - gamma pi_in - filt
    first: acc = state0 + w * (pi_in, a); else acc += w * (pi_in, a)
    not last: (phi_out, pi_out) = state0 + c * (pi_in, a)
    Returns (sum pi_in*(gamma pi_in + filt), sum acc_pi^2 if last else 0).
    """
    n = lap.size
    diss = 0.0
    ikin = 0.0
    for i in prange(n):
        p = phi_in[i]
        v = pi_in[i]
        p2 = p * p
        a = lap[i] - p * (m2 + p2 * (a6 * p2 - a4))
        if use_gamma:
            gv = gamma[i] * v
            a -= gv
            diss += gv * v
        if use_filt:
            a -= filt[i]
            diss += filt[i] * v
        if first:
            ap = phi0[i] + w * v
            av = pi0[i] + w * a
        else:
            ap = acc_phi[i] + w * v
            av = acc_pi[i] + w * a
        acc_phi[i] = ap
        acc_pi[i] = av
        if last:
            ikin += av * av
        else:
            phi_out[i] = phi0[i] + c * v
            pi_out[i] = pi0[i] + c * a
    return diss, ikin


@njit(parallel=True, cache=True)
def _mul_real(z, f):
    """z *= f (complex array times real array, flat)."""
    for i in prange(z.size):
        z[i] = z[i] * f[i]


@njit(parallel=True, cache=True)
def _potential_parts(phi, m2, g4, g6):
    """Return sums of phi^2/2 m^2, -g4 phi^4/24, g6 phi^6/720, max|phi|."""
    s2 = 0.0
    s4 = 0.0
    s6 = 0.0
    mx = 0.0
    for i in prange(phi.size):
        p = phi[i]
        p2 = p * p
        s2 += p2
        s4 += p2 * p2
        s6 += p2 * p2 * p2
        mx = max(mx, abs(p))
    return 0.5 * m2 * s2, -g4 / 24.0 * s4, g6 / 720.0 * s6, mx


@njit(parallel=True, cache=True)
def _sum_sq(x):
    s = 0.0
    for i in prange(x.size):
        s += x[i] * x[i]
    return s


# ---------------------------------------------------------------------------
# Evolver
# ---------------------------------------------------------------------------

class Evolver:
    """State (phi, pi = phi_t) on an N^3 periodic grid, RK4 in time."""

    def __init__(self, N, L, m, g4, g6, absorber=None, kfilter=None, threads=4):
        self.N, self.L = int(N), float(L)
        self.dx = self.L / self.N
        self.dV = self.dx ** 3
        self.m, self.g4, self.g6 = float(m), float(g4), float(g6)
        self.threads = int(threads)
        numba.set_num_threads(min(self.threads, numba.config.NUMBA_NUM_THREADS))

        N = self.N
        self.coords = -self.L / 2 + self.dx * np.arange(N)
        k = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
        kr = 2 * np.pi * np.fft.rfftfreq(N, d=self.dx)
        self.kx = k[:, None, None]
        self.ky = k[None, :, None]
        self.kz = kr[None, None, :]
        K2 = self.kx ** 2 + self.ky ** 2 + self.kz ** 2
        self.K2 = np.ascontiguousarray(K2)
        self.negK2_flat = (-self.K2).ravel()
        # rfft half-spectrum weights for Parseval sums
        wz = np.full(kr.size, 2.0)
        wz[0] = 1.0
        if N % 2 == 0:
            wz[-1] = 1.0
        self.wK2 = np.ascontiguousarray(self.K2 * wz[None, None, :])
        self.shape = (N, N, N)
        self.cshape = self.K2.shape

        # Absorber: gamma(r) about the box centre (origin), see sponge_profile
        self.absorber = absorber
        if absorber:
            self.gamma = sponge_profile(self.coords, self.L, **absorber)
        else:
            self.gamma = np.zeros(1)
        # High-k damping on phi_t: rate nu(k) = nu * (|k| / k_max)^power with
        # k_max = sqrt(3) pi / dx the grid corner, so nu is the largest rate
        # (RK4 needs nu * dt < 2.78). Costs one extra FFT pair per stage.
        self.kfilter = kfilter
        if kfilter:
            kmax = np.sqrt(3.0) * np.pi / self.dx
            nu = kfilter["nu"] * (np.sqrt(self.K2) / kmax) ** kfilter.get("power", 8)
            self.nu_flat = np.ascontiguousarray(nu).ravel()

        z = lambda: np.zeros(self.shape)
        self.phi, self.pi = z(), z()
        self._phi_s, self._pi_s = z(), z()
        self._acc_phi, self._acc_pi = z(), z()
        self.step_count = 0
        self.absorbed = 0.0      # integral of D dt (sponge + filter)
        self.I_kin = 0.0         # integral of phi_t^2 d3x at current step
        self.C_kin = 0.0         # time integral of I_kin (trapezoid over steps)

    # -- spectral helpers --------------------------------------------------
    def rfft(self, f):
        return sfft.rfftn(f, workers=self.threads)

    def irfft(self, fh):
        return sfft.irfftn(fh, s=self.shape, workers=self.threads)

    def laplacian(self, f):
        fh = self.rfft(f)
        _mul_real(fh.ravel(), self.negK2_flat)
        return self.irfft(fh)

    def _filter_force(self, v):
        vh = self.rfft(v)
        _mul_real(vh.ravel(), self.nu_flat)
        return self.irfft(vh)

    # -- state -------------------------------------------------------------
    def set_state(self, phi, pi):
        self.phi[...] = phi
        self.pi[...] = pi
        self.step_count = 0
        self.absorbed = 0.0
        self.C_kin = 0.0
        self.I_kin = _sum_sq(self.pi.ravel()) * self.dV

    # -- time step ---------------------------------------------------------
    def step(self, dt):
        """One classical RK4 step for (phi, pi) and the dissipated energy."""
        m2, a4, a6 = self.m ** 2, self.g4 / 6.0, self.g6 / 120.0
        use_g = bool(self.absorber)
        use_f = bool(self.kfilter)
        g = self.gamma.ravel()
        phi0, pi0 = self.phi.ravel(), self.pi.ravel()
        ps, vs = self._phi_s.ravel(), self._pi_s.ravel()
        ap, av = self._acc_phi.ravel(), self._acc_pi.ravel()
        dummy = g[:1]
        weights = (dt / 6, dt / 3, dt / 3, dt / 6)
        cs = (dt / 2, dt / 2, dt, 0.0)
        d_tot = 0.0
        ikin_new = 0.0
        for s in range(4):
            src_phi = self.phi if s == 0 else self._phi_s
            src_pi = pi0 if s == 0 else vs
            lap = self.laplacian(src_phi).ravel()
            filt = self._filter_force(src_pi.reshape(self.shape)).ravel() if use_f else dummy
            d, ik = _stage(lap, src_phi.ravel(), src_pi, phi0, pi0, ap, av, ps, vs,
                           g, use_g, filt, use_f, m2, a4, a6,
                           weights[s], cs[s], s == 0, s == 3)
            d_tot += (weights[s] / dt) * d
            ikin_new = ik
        self.absorbed += dt * d_tot * self.dV
        ikin_new *= self.dV
        self.C_kin += 0.5 * dt * (self.I_kin + ikin_new)
        self.I_kin = ikin_new
        self.phi, self._acc_phi = self._acc_phi, self.phi
        self.pi, self._acc_pi = self._acc_pi, self.pi
        self.step_count += 1

    # -- energy ------------------------------------------------------------
    def energy_parts(self, phi=None, pi=None):
        """Energy terms; every term of the equation of motion is included."""
        phi = self.phi if phi is None else phi
        pi = self.pi if pi is None else pi
        fh = self.rfft(phi)
        e_grad = 0.5 * self.dV / self.N ** 3 * float(np.sum(self.wK2 * (fh.real ** 2 + fh.imag ** 2)))
        e_kin = 0.5 * _sum_sq(np.ascontiguousarray(pi).ravel()) * self.dV
        e2, e4, e6, mx = _potential_parts(np.ascontiguousarray(phi).ravel(), self.m ** 2, self.g4, self.g6)
        e2, e4, e6 = e2 * self.dV, e4 * self.dV, e6 * self.dV
        tot = e_kin + e_grad + e2 + e4 + e6
        return {"E_total": tot, "E_kin": e_kin, "E_grad": e_grad, "E_mass": e2,
                "E_quartic": e4, "E_sextic": e6, "max_abs_phi": mx}

    def energy(self):
        return self.energy_parts()["E_total"]

    def energy_density(self):
        """Pointwise e = pi^2/2 + |grad phi|^2/2 + V(phi).

        The gradient is spectral (i k phi_hat) with the Nyquist wavenumber set
        to zero for this odd derivative; its volume sum equals the K2 energy up
        to the (negligible for resolved fields) Nyquist content.
        """
        fh = self.rfft(self.phi)
        N = self.N
        e = 0.5 * self.pi ** 2
        for kk in (self.kx, self.ky, self.kz):
            kk = kk.copy()
            if N % 2 == 0:
                kk[np.isclose(np.abs(kk), np.pi / self.dx)] = 0.0
            gr = self.irfft(1j * kk * fh)
            e += 0.5 * gr ** 2
        p2 = self.phi ** 2
        e += p2 * (0.5 * self.m ** 2 + p2 * (-self.g4 / 24.0 + p2 * self.g6 / 720.0))
        return e


# Default sponge: cubic ramp, gamma_max = 2, width 12. Chosen from the 1D
# design scan (swirl/engine/sponge1d.py -> outputs/engine/absorber_design_scan.json)
# for fast radiation; validated in 3D by tests/engine/test_check4_absorber.py
# (check 4 box scan, the k0 = 2.4 outgoing-packet test, energy bookkeeping). Integrated damping per face crossing is
# gamma_max * width / (power + 1) = 6, so a wave at v ~ 1 keeps an amplitude
# fraction ~ exp(-6) per periodic pass (two crossings) along the shortest path.
SPONGE_DEFAULT = {"gamma": 2.0, "width": 12.0, "power": 3}


def sponge_profile(coords, L, gamma, width, power=3, r_out=None):
    """Spherical damping sponge about the origin.

    gamma(r) = gamma_max * u^power, u = (r - r_in)/(r_out - r_in) clipped to
    [0, 1], r_in = r_out - width, r_out = L/2 by default. Beyond r_out (the box
    corners) gamma = gamma_max. Along a face normal an outgoing wave crosses
    the layer to the face and, after the periodic wrap, crosses it again, so
    the path through the sponge is 2 * width.
    """
    r_out = L / 2 if r_out is None else r_out
    r_in = r_out - width
    x = coords
    r = np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)
    u = np.clip((r - r_in) / (r_out - r_in), 0.0, 1.0)
    return np.ascontiguousarray(gamma * u ** power)
