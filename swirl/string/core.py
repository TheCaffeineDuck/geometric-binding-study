"""String solver for Track C (Session 5, WP-B1).

Displacement u on a ring of N points at spacing h, x_j = (j - N//2) h. Stretch d_j on bond j+1/2
(j -> j+1, indices mod N): d_j = (u_{j+1} - u_j)/h ('grid', 'fd') or d = D u at the points
('spectral'). Laws F(d): 'lin' d, 'G1' d/(1-d^2), 'G2' d + beta d^3; V' = F.

Media (derivations in docs/WPB1_BUILD_NOTES.md)
  'grid'     C-grid, unit masses one unit apart (h = 1):  a_j = F(d_j) - F(d_{j-1}).
  'fd'       C-smooth, staggered finite differences: (I - D2) a = (F(d_j) - F(d_{j-1})) / h.
  'spectral' C-smooth, Fourier pseudo-spectral (Nyquist mode of D zeroed): (I - D^2) a = D F(D u).

Exactly conserved semi-discrete energies:
  grid      E = sum v_j^2/2 + sum V(d_j)
  fd        E = h sum v_j^2/2 + (1/2h) sum (v_{j+1}-v_j)^2 + h sum V(d_j)
  spectral  E = h sum [ v^2/2 + (Dv)^2/2 + V(d) ]

State: (u, v, d). The stretch d is carried as its own variable: the drift updates u += c v and
d += c (delta v)/h (or c D v), so with no absorber d stays equal to the stretch of u to round-off.
It is carried separately for the absorber: the matched (distortionless) sponge damps the velocity
AND the stretch at the same rate sigma(x),
    M v_t = force - B v,   d_t = (stretch of v) - sigma d,   B = sigma + D^T sigma D ('fd', spectral)
For long waves (the wave equation, v_t = d_x, d_t = v_x) equal damping of both fields leaves the
wave impedance unchanged at every frequency (Heaviside's distortionless line), so long waves enter
the sponge without reflection; a velocity-only sponge reflects waves with omega below about
(sigma_max/L_s^2)^(1/3). Both damping terms only remove energy: dE/dt = -v.Bv - sum sigma F(d) d.
Inside the sponge d is no longer the stretch of u; u there is a diagnostic only.

Time stepping: 4th-order Yoshida triple jump of velocity Verlet (symplectic, FSAL, 3 force
evaluations per step); absorber as an exact/Crank-Nicolson dissipation step of dt/2 before and
after each symplectic step (Strang). Removed energy is accumulated exactly in `absorbed`.

Guard (G1): every force evaluation (every stage) checks |d| < 1 on every bond. On failure the step
is undone and GuardTrip carries the diagnosis. Nothing is clipped.
"""

import numpy as np
from numba import njit

LAWS = {"lin": 0, "G1": 1, "G2": 2}
MEDIA = ("grid", "fd", "spectral")

_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_W0 = -(2.0 ** (1.0 / 3.0)) * _W1
YOSHIDA = np.array([_W1, _W0, _W1])


# ---------------------------------------------------------------- laws (numpy, complex-safe)
def V_of(law, beta):
    if law == "G1":
        return lambda d: -0.5 * np.log(1.0 - d * d)
    if law == "G2":
        return lambda d: 0.5 * d * d + 0.25 * beta * d ** 4
    return lambda d: 0.5 * d * d


def F_of(law, beta):
    if law == "G1":
        return lambda d: d / (1.0 - d * d)
    if law == "G2":
        return lambda d: d + beta * d ** 3
    return lambda d: d


def dF_of(law, beta):
    if law == "G1":
        return lambda d: (1.0 + d * d) / (1.0 - d * d) ** 2
    if law == "G2":
        return lambda d: 1.0 + 3.0 * beta * d * d
    return lambda d: np.ones_like(d)


# ---------------------------------------------------------------- cyclic tridiagonal (numba)
def cyc_factor(lo, di, up):
    """Factor the cyclic tridiagonal A: row j = lo[j] x_{j-1} + di[j] x_j + up[j] x_{j+1}
    (indices mod N). Sherman-Morrison around a Thomas factorisation (Numerical Recipes 'cyclic')."""
    N = di.size
    lo = np.asarray(lo, dtype=float) * np.ones(N)
    up = np.asarray(up, dtype=float) * np.ones(N)
    di = np.asarray(di, dtype=float) * np.ones(N)
    beta_ = lo[0]          # A[0, N-1]
    alpha = up[N - 1]      # A[N-1, 0]
    gamma = -di[0]
    bb = di.copy()
    bb[0] = di[0] - gamma
    bb[N - 1] = di[N - 1] - alpha * beta_ / gamma
    cp = np.empty(N)
    dinv = np.empty(N)
    dinv[0] = 1.0 / bb[0]
    cp[0] = up[0] * dinv[0]
    for j in range(1, N):
        dinv[j] = 1.0 / (bb[j] - lo[j] * cp[j - 1])
        cp[j] = up[j] * dinv[j]
    z = np.zeros(N)
    z[0] = gamma
    z[N - 1] = alpha
    _thomas(lo, cp, dinv, z)
    bg = beta_ / gamma
    return (lo.copy(), cp, dinv, z, np.array([bg, 1.0 + z[0] + bg * z[N - 1]]))


@njit(cache=True)
def _thomas(lo, cp, dinv, r):
    N = r.shape[0]
    r[0] = r[0] * dinv[0]
    for j in range(1, N):
        r[j] = (r[j] - lo[j] * r[j - 1]) * dinv[j]
    for j in range(N - 2, -1, -1):
        r[j] = r[j] - cp[j] * r[j + 1]


@njit(cache=True)
def _cyc_solve(lo, cp, dinv, z, sc, r):
    """In place: r <- A^{-1} r."""
    N = r.shape[0]
    _thomas(lo, cp, dinv, r)
    fact = (r[0] + sc[0] * r[N - 1]) / sc[1]
    for j in range(N):
        r[j] -= fact * z[j]


# ---------------------------------------------------------------- force and stepping (numba)
@njit(cache=True)
def _Vb(d, law, beta):
    if law == 1:
        return -0.5 * np.log(1.0 - d * d)
    if law == 2:
        return 0.5 * d * d + 0.25 * beta * d * d * d * d
    return 0.5 * d * d


@njit(cache=True)
def _acc(dd, law, beta, invh, smooth, mlo, mcp, mdinv, mz, msc, Fb, a):
    """a <- M^{-1} (F(d_j) - F(d_{j-1})) invh. Returns (bond, stretch) of the first bond with
    |d| >= 1 (or NaN) under G1, else (-1, 0)."""
    N = dd.shape[0]
    for j in range(N):
        d = dd[j]
        if law == 1:
            if not (abs(d) < 1.0):
                return j, d
            Fb[j] = d / (1.0 - d * d)
        elif law == 2:
            Fb[j] = d + beta * d * d * d
        else:
            Fb[j] = d
    a[0] = (Fb[0] - Fb[N - 1]) * invh
    for j in range(1, N):
        a[j] = (Fb[j] - Fb[j - 1]) * invh
    if smooth:
        _cyc_solve(mlo, mcp, mdinv, mz, msc, a)
    return -1, 0.0


@njit(cache=True)
def _kinetic(v, smooth, h):
    N = v.shape[0]
    s = 0.0
    for j in range(N):
        s += v[j] * v[j]
    if not smooth:
        return 0.5 * s
    g = 0.0
    for j in range(N):
        jp = j + 1 if j + 1 < N else 0
        dv = v[jp] - v[j]
        g += dv * dv
    return 0.5 * h * s + 0.5 * g / h


@njit(cache=True)
def _damp(v, dd, smooth, h, law, beta, gfac, bidx, bfac, plo, pcp, pdinv, pz, psc,
          qlo, qdi, qup, work):
    """Half-step of the absorber. Returns the energy removed (kinetic + stretch, exact)."""
    N = v.shape[0]
    k0 = _kinetic(v, smooth, h)
    if not smooth:
        for j in range(N):
            v[j] *= gfac[j]
    else:
        for j in range(N):
            jm = j - 1 if j > 0 else N - 1
            jp = j + 1 if j < N - 1 else 0
            work[j] = qlo[j] * v[jm] + qdi[j] * v[j] + qup[j] * v[jp]
        _cyc_solve(plo, pcp, pdinv, pz, psc, work)
        for j in range(N):
            v[j] = work[j]
    dV = 0.0
    for i in range(bidx.shape[0]):
        j = bidx[i]
        d0 = dd[j]
        dd[j] = d0 * bfac[i]
        dV += _Vb(d0, law, beta) - _Vb(dd[j], law, beta)
    return k0 - _kinetic(v, smooth, h) + h * dV


@njit(cache=True)
def _advance(u, v, dd, a, Fb, nsteps, dt, W, law, beta, invh, h, smooth,
             mlo, mcp, mdinv, mz, msc, damping, gfac, bidx, bfac, plo, pcp, pdinv, pz, psc,
             qlo, qdi, qup, u0, v0, d0, a0, work):
    """nsteps symplectic steps. a must hold the acceleration at d on entry (FSAL).
    Returns (steps done, stage of failure or -1, bond, stretch, absorbed energy)."""
    N = u.shape[0]
    absorbed = 0.0
    for k in range(nsteps):
        for i in range(N):
            u0[i] = u[i]
            v0[i] = v[i]
            d0[i] = dd[i]
            a0[i] = a[i]
        ab = 0.0
        if damping:
            ab += _damp(v, dd, smooth, h, law, beta, gfac, bidx, bfac, plo, pcp, pdinv, pz, psc,
                        qlo, qdi, qup, work)
            # the damping step changed d: the carried (FSAL) acceleration is stale; without this
            # recomputation the energy balance in the sponge is first order in dt
            _acc(dd, law, beta, invh, smooth, mlo, mcp, mdinv, mz, msc, Fb, a)
        for s in range(3):
            c = W[s] * dt
            for i in range(N):
                v[i] += 0.5 * c * a[i]
                u[i] += c * v[i]
            ci = c * invh
            for i in range(N - 1):
                dd[i] += ci * (v[i + 1] - v[i])
            dd[N - 1] += ci * (v[0] - v[N - 1])
            jb, db = _acc(dd, law, beta, invh, smooth, mlo, mcp, mdinv, mz, msc, Fb, a)
            if jb >= 0:
                for i in range(N):
                    u[i] = u0[i]
                    v[i] = v0[i]
                    dd[i] = d0[i]
                    a[i] = a0[i]
                return k, s, jb, db, absorbed
            for i in range(N):
                v[i] += 0.5 * c * a[i]
        if damping:
            ab += _damp(v, dd, smooth, h, law, beta, gfac, bidx, bfac, plo, pcp, pdinv, pz, psc,
                        qlo, qdi, qup, work)
        absorbed += ab
    return nsteps, -1, -1, 0.0, absorbed


class GuardTrip(Exception):
    def __init__(self, diag):
        super().__init__("G1 guard: |d| >= 1 at t=%.6g, x=%.6g, d=%.17g" %
                         (diag["t"], diag["x_bond"], diag["stretch"]))
        self.diag = diag


# ---------------------------------------------------------------- the solver object
class String:
    """A ring of N points at spacing h ('grid' forces h = 1), x_j = (j - N//2) h, length L = N h.

    Sponge: sigma(x) = sigma_max ((|x| - x_s)/L_s)^2 for |x| > x_s = L/2 - L_s, evaluated at the
    points (velocity) and at the bond midpoints (stretch; at the points for 'spectral'). The two
    ends of the string meet at x = +-L/2 inside the sponge.
    """

    def __init__(self, medium, law, N, h=1.0, beta=0.0, sponge_len=0.0, sigma_max=0.0):
        assert medium in MEDIA and law in LAWS
        self.medium, self.law, self.beta = medium, law, float(beta)
        self.N = int(N)
        self.h = 1.0 if medium == "grid" else float(h)
        self.L = self.N * self.h
        self.x = (np.arange(self.N) - self.N // 2) * self.h
        self.smooth = medium != "grid"
        self.lawc = LAWS[law]
        self.V = V_of(law, beta)
        self.F = F_of(law, beta)
        self.dF = dF_of(law, beta)
        self.sponge_len, self.sigma_max = float(sponge_len), float(sigma_max)
        self.sig = self.sigma(self.x)
        xb = self.x if medium == "spectral" else self.x + 0.5 * self.h
        self.sigb = self.sigma(xb)
        self.damping = bool(np.any(self.sig > 0))
        self.u = np.zeros(self.N)
        self.v = np.zeros(self.N)
        self.dd = np.zeros(self.N)
        self.t = 0.0
        self.absorbed = 0.0
        self._dt_damp = None
        h = self.h
        if medium == "fd":
            one = np.ones(self.N)
            self._M = cyc_factor(-one / h ** 2, (1.0 + 2.0 / h ** 2) * one, -one / h ** 2)
        else:
            z = np.zeros(1)
            self._M = (z, z, z, z, np.array([0.0, 1.0]))
        if medium == "spectral":
            self.k = 2 * np.pi * np.fft.rfftfreq(self.N, d=h)
            self.ikD = 1j * self.k
            if self.N % 2 == 0:
                self.ikD[-1] = 0.0           # Nyquist mode of D zeroed: D stays real antisymmetric
            self.Minv = 1.0 / (1.0 + np.abs(self.ikD) ** 2)
        self._Fb = np.empty(self.N)
        self.a = np.zeros(self.N)
        self._w = [np.empty(self.N) for _ in range(5)]

    def sigma(self, x):
        if not (self.sigma_max > 0 and self.sponge_len > 0):
            return np.zeros_like(x)
        s = np.clip((np.abs(x) - (self.L / 2 - self.sponge_len)) / self.sponge_len, 0.0, None)
        return self.sigma_max * s * s

    # -------------------------------------------------------------- setup
    def set_state(self, u, v=None, t=0.0):
        self.u = np.array(u, dtype=float)
        self.v = np.zeros(self.N) if v is None else np.array(v, dtype=float)
        self.dd = self.stretch(self.u)
        self.t = float(t)
        self.absorbed = 0.0
        st = self._accel_now()
        if st is not None:
            raise GuardTrip(self._diag(-1, st[0], st[1], t))

    def _accel_now(self):
        if self.medium == "spectral":
            d = self.dd
            if self.lawc == 1 and not np.all(np.abs(d) < 1.0):
                j = int(np.argmax(~(np.abs(d) < 1.0)))
                return j, float(d[j])
            self.a = np.fft.irfft(self.Minv * self.ikD * np.fft.rfft(self.F(d)), n=self.N)
            return None
        M = self._M
        j, d = _acc(self.dd, self.lawc, self.beta, 1.0 / self.h, self.smooth, M[0], M[1], M[2],
                    M[3], M[4], self._Fb, self.a)
        return None if j < 0 else (j, d)

    def D(self, f):
        if np.iscomplexobj(f):                      # D is a real matrix: complex-step safe
            return self.D(f.real) + 1j * self.D(f.imag)
        return np.fft.irfft(self.ikD * np.fft.rfft(f), n=self.N)

    def _prep_damp(self, dt):
        if self._dt_damp == dt:
            return
        self._dt_damp = dt
        tau = 0.5 * dt
        h, sig = self.h, self.sig
        self._gfac = np.exp(-sig * tau)
        self._bidx = np.where(self.sigb > 0)[0].astype(np.int64)
        self._bfac = np.exp(-self.sigb[self._bidx] * tau)
        if self.medium == "fd":
            sb = self.sigb                                 # bond j+1/2
            Bdi = sig + (sb + np.roll(sb, 1)) / h ** 2
            Blo = -np.roll(sb, 1) / h ** 2                 # coefficient of v_{j-1}
            Bup = -sb / h ** 2                             # coefficient of v_{j+1}
            Mdi, Moff = 1.0 + 2.0 / h ** 2, -1.0 / h ** 2
            c = 0.5 * tau
            self._P = cyc_factor(Moff + c * Blo, Mdi + c * Bdi, Moff + c * Bup)
            self._Q = (Moff - c * Blo, Mdi - c * Bdi, Moff - c * Bup)
        else:
            z = np.zeros(1)
            self._P = (z, z, z, z, np.array([0.0, 1.0]))
            self._Q = (z, z, z)

    # -------------------------------------------------------------- stepping
    def advance(self, dt, n):
        """n steps of size dt. Raises GuardTrip with the diagnosis on a G1 failure (the state is
        left at the last completed step)."""
        self._prep_damp(dt)
        if self.medium == "spectral":
            return self._advance_spec(dt, n)
        M, P, Q = self._M, self._P, self._Q
        w = self._w
        done, stage, jb, db, ab = _advance(
            self.u, self.v, self.dd, self.a, self._Fb, int(n), float(dt), YOSHIDA, self.lawc,
            self.beta, 1.0 / self.h, self.h, self.smooth, M[0], M[1], M[2], M[3], M[4],
            self.damping, self._gfac, self._bidx, self._bfac, P[0], P[1], P[2], P[3], P[4],
            Q[0], Q[1], Q[2], w[0], w[1], w[2], w[3], w[4])
        self.t += done * dt
        self.absorbed += ab
        if stage >= 0:
            raise GuardTrip(self._diag(stage, jb, db, self.t + dt))

    def _damp_spec(self, tau):
        """Crank-Nicolson on M v' = -B v over tau (B = sig - D sig D, fixed-point iteration), and
        d *= exp(-sig tau). Returns the energy removed."""
        e0 = self.energy()
        vo = self.v
        sig = self.sig

        def MinvB(w):
            Bw = sig * w - self.D(sig * self.D(w))
            return np.fft.irfft(self.Minv * np.fft.rfft(Bw), n=self.N)
        x = vo.copy()
        for _ in range(60):
            xn = vo - 0.5 * tau * MinvB(vo + x)
            done = np.max(np.abs(xn - x)) <= 1e-16 * max(1e-300, np.max(np.abs(xn)))
            x = xn
            if done:
                break
        self.v = x
        self.dd = self.dd * np.exp(-self.sigb * tau)
        return e0 - self.energy()

    def _advance_spec(self, dt, n):
        for k in range(int(n)):
            u0, v0, d0, a0 = self.u.copy(), self.v.copy(), self.dd.copy(), self.a.copy()
            ab = 0.0
            if self.damping:
                ab = self._damp_spec(0.5 * dt)
                self._accel_now()                 # d changed: refresh the carried acceleration
            for s in range(3):
                c = YOSHIDA[s] * dt
                self.v = self.v + 0.5 * c * self.a
                self.u = self.u + c * self.v
                self.dd = self.dd + c * self.D(self.v)
                st = self._accel_now()
                if st is not None:
                    self.u, self.v, self.dd, self.a = u0, v0, d0, a0
                    raise GuardTrip(self._diag(s, st[0], st[1], self.t + dt))
                self.v = self.v + 0.5 * c * self.a
            if self.damping:
                ab += self._damp_spec(0.5 * dt)
            self.absorbed += ab
            self.t += dt

    def _diag(self, stage, jb, db, t_fail):
        xb = self.x[jb] + (0.0 if self.medium == "spectral" else 0.5 * self.h)
        with np.errstate(invalid="ignore"):     # a start outside |d| < 1 has no energy (NaN)
            e = float(self.energy())
        return {"t": float(t_fail), "t_last_good": float(self.t), "stage": int(stage),
                "bond": int(jb), "x_bond": float(xb), "stretch": float(db),
                "energy_last_good": e, "absorbed_so_far": float(self.absorbed),
                "u_last_good": self.u.copy(), "v_last_good": self.v.copy(),
                "d_last_good": self.dd.copy()}

    # -------------------------------------------------------------- energies
    def stretch(self, u=None):
        """Stretch of u (default: the carried stretch variable d)."""
        if u is None:
            return self.dd.copy()
        if self.medium == "spectral":
            return self.D(u)
        return (np.roll(u, -1) - u) / self.h

    def kinetic(self, v=None):
        v = self.v if v is None else v
        if self.medium == "grid":
            return 0.5 * np.sum(v * v)
        if self.medium == "fd":
            dv = np.roll(v, -1) - v
            return 0.5 * self.h * np.sum(v * v) + 0.5 * np.sum(dv * dv) / self.h
        Dv = self.D(v)
        return 0.5 * self.h * np.sum(v * v + Dv * Dv)

    def _d(self, u, d):
        if d is not None:
            return d
        return self.dd if u is None else self.stretch(u)

    def energy(self, u=None, v=None, d=None):
        """The exactly conserved discrete energy (every term of the equation of motion).
        Default: the current state; pass u (stretch taken from u) or d explicitly."""
        v = self.v if v is None else v
        return self.kinetic(v) + self.h * np.sum(self.V(self._d(u, d)))

    def density(self, u=None, v=None, d=None):
        """Energy per point; sums exactly to energy(). Bond terms split half to each end."""
        v = self.v if v is None else v
        dd = self._d(u, d)
        h = self.h
        if self.medium == "spectral":
            Dv = self.D(v)
            return h * (0.5 * v * v + 0.5 * Dv * Dv + self.V(dd))
        bond = h * self.V(dd)
        if self.medium == "fd":
            dv = np.roll(v, -1) - v
            bond = bond + 0.5 * dv * dv / h
        return 0.5 * h * v * v + 0.5 * (bond + np.roll(bond, 1))

    def quadratic_energy(self, u=None, v=None, d=None):
        v = self.v if v is None else v
        dd = self._d(u, d)
        return self.kinetic(v) + self.h * np.sum(0.5 * dd * dd)

    def omega_max(self, dmax):
        """Upper bound of the local linear rate at stretch dmax (sets dt)."""
        k = float(self.dF(np.array([dmax]))[0])
        return (2.0 if self.medium == "grid" else 1.0) * np.sqrt(k)

    def accel(self, u):
        """Acceleration of the equation of motion at displacement u (no absorber; for tests)."""
        save = self.dd, self.a
        self.dd = self.stretch(np.asarray(u, dtype=float))
        self.a = np.zeros(self.N)
        st = self._accel_now()
        a = self.a.copy()
        self.dd, self.a = save
        if st is not None:
            raise GuardTrip({"t": self.t, "x_bond": float(self.x[st[0]]), "stretch": st[1]})
        return a


# ---------------------------------------------------------------- linear dispersion relations
def omega_scheme(medium, k, h=1.0):
    k = np.asarray(k, dtype=float)
    if medium == "grid":
        return 2.0 * np.abs(np.sin(k / 2.0))
    if medium == "fd":
        s2 = (2.0 / h * np.sin(k * h / 2.0)) ** 2
        return np.sqrt(s2 / (1.0 + s2))
    return np.abs(k) / np.sqrt(1.0 + k * k)


def omega_continuum(medium, k):
    k = np.asarray(k, dtype=float)
    if medium == "grid":
        return 2.0 * np.abs(np.sin(k / 2.0))
    return np.abs(k) / np.sqrt(1.0 + k * k)


def band_top(medium, h=1.0):
    """Highest linear rate the discretisation carries (spectral: at k = pi/h, the Nyquist limit)."""
    if medium == "grid":
        return 2.0
    if medium == "fd":
        return 2.0 / np.sqrt(4.0 + h * h)
    k = np.pi / h
    return k / np.sqrt(1 + k * k)
