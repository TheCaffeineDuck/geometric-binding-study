"""Time-periodic states of equation R in (f, p) by Newton-Krylov shooting (reviewer
condition 2), past the single-frequency ansatz wall omega F(0) = 1.

STATUS: DID NOT CONVERGE IN TWO ATTEMPTS; parked (docs/WPB2_BUILD_NOTES.md section 10).
Attempt 1 (box R = 80): NoConvergence at the first point, F(0) = 0.9, seed residual
3.7e-2; the iterate left the profile shape. Attempt 2 (box sized so 3 omega sits mid-gap
between box modes, R = 39.68; deeper Krylov solve): the residual fell to 5.3e-3 and stalled
over 40 Newton steps, with a spike growing next to the pinned centre cell (f1 = 1.03 > c =
0.9). Kept as the record of what was tried; nothing in the production path uses it.

Symmetric orbits: start at a turning point (f = f0, p = 0). If the field crosses zero
everywhere at a quarter period, f(T/4) = 0, then by the symmetries (f -> -f and time
reversal) f(T/2) = -f0, p(T/2) = 0 and the orbit has period T. Unknowns: f0 except the
pinned centre value c = f0 at cell 0 (the amplitude, which parametrises the family
through folds in omega), and T. Equations: f(T/4) = 0 at every cell. Closed box (no
sponge): the orbit carries its standing radiation, as any periodic state in a box does.

Stability: Floquet multipliers of the one-period map of (f, p) at the orbit, by Arnoldi
(scipy eigs) on finite-difference Jacobian-vector products. Leading growth rate
ln|mu| / T. Multipliers come in (mu, 1/mu, conj) sets; the closed box puts the radiation
modes on the unit circle.

Run: <venv python> -m swirl.radial5.periodic  -> outputs/radial5/periodic_R_bump.json
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import newton_krylov  # noqa: E402
from scipy.sparse.linalg import LinearOperator, eigs  # noqa: E402

from swirl.radial5 import profiles as P  # noqa: E402
from swirl.radial5.evolve import write_json  # noqa: E402
from swirl.radial5.solver_r import SolverR  # noqa: E402

DT_TARGET = 0.02


class Orbit:
    def __init__(self, obj="bump", Rb=80.0, h=0.1, verbose=False):
        self.verbose = verbose
        self.s = SolverR(obj, Rb, h, qmax=np.inf, hk_max=np.inf)
        self.r = self.s.r

    def flow(self, f0, p0, t):
        n = max(4, int(np.ceil(t / DT_TARGET)))
        self.s.f, self.s.p, self.s.t = f0.copy(), p0.copy(), 0.0
        self.s.step(t / n, n)
        return self.s.f.copy(), self.s.p.copy()

    def residual(self, z, c):
        f0 = np.concatenate([[c], z[:-1]])
        T = z[-1]
        f, _ = self.flow(f0, np.zeros_like(f0), T / 4.0)
        return f

    def solve(self, c, f_guess, T_guess, f_tol=1e-9, maxiter=40):
        z0 = np.concatenate([f_guess[1:], [T_guess]])
        z = newton_krylov(lambda z: self.residual(z, c), z0, f_tol=f_tol, maxiter=maxiter,
                          method="lgmres", inner_maxiter=200, verbose=self.verbose)
        f0 = np.concatenate([[c], z[:-1]])
        return f0, float(z[-1])

    def monodromy_eigs(self, f0, T, k=8, eps=1e-6):
        y0 = np.concatenate([f0, np.zeros_like(f0)])
        M = f0.size
        sc = eps * max(np.max(np.abs(f0)), 1e-3)

        def mv(v):
            a = self.flow(y0[:M] + sc * v[:M], y0[M:] + sc * v[M:], T)
            b = self.flow(y0[:M] - sc * v[:M], y0[M:] - sc * v[M:], T)
            return (np.concatenate(a) - np.concatenate(b)) / (2 * sc)
        op = LinearOperator((2 * M, 2 * M), matvec=mv, dtype=float)
        mu = eigs(op, k=k, which="LM", return_eigenvectors=False, tol=1e-8, maxiter=2000)
        return np.array(sorted(mu, key=lambda x: -abs(x)))

    def observables(self, f0, T):
        s = self.s
        s.f, s.p = f0.copy(), np.zeros_like(f0)
        E = float(s.energy_discrete())
        # time-sampled over a period: max X, energy inside r < 3 hw, f_t at the centre
        F = np.abs(f0)
        i = int(np.argmax(F))
        j = i + int(np.argmax(F[i:] < 0.5 * F[i]))
        hw = float(self.r[j])
        Xmax, ftmax = -np.inf, 0.0
        f, p = f0.copy(), np.zeros_like(f0)
        for _ in range(16):
            f, p = self.flow(f, p, T / 16)
            s.f, s.p = f, p
            Xmax = max(Xmax, float(np.max(s.X())))
            ftmax = max(ftmax, float(np.max(np.abs(s.rhs()[0]))))
        closure = float(np.max(np.abs(f + f0)) / np.max(np.abs(f0)))  # after 16 x T/16 = T: f(T) = f0
        f2, _ = self.flow(f0, np.zeros_like(f0), T)
        closure = float(np.max(np.abs(f2 - f0)) / np.max(np.abs(f0)))
        return dict(E=E, omega=2 * np.pi / T, T=T, F0=float(f0[0]), omega_F0=2 * np.pi / T * float(f0[0]),
                    hw=hw, X_max_over_period=Xmax, max_abs_ft=ftmax, period_closure_rel=closure,
                    E_in_3hw=float(np.sum(4 * np.pi * s.w[self.r < 3 * hw] * s.density(f0, 0 * f0)[0][self.r < 3 * hw])))


def run(cs=(0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5), Rb=None, h=0.1, floquet=True, verbose=True):
    """Attempt 2: the box is sized so that 3 omega of the seed sits mid-gap between box
    modes (k3 R / pi = n + 1/2, R near 40), to keep the shooting Jacobian away from the
    resonance cos(omega_k T/4) = 0 that sank attempt 1 (R = 80)."""
    t0 = time.time()
    # seed: the single-frequency profile on the upper branch near F(0) = 0.9
    p = P.solve_point(P.Model("R", "bump"), x=cs[0], guess=0.1)
    if Rb is None:
        k3 = np.sqrt(9 * p["omega"] ** 2 - 1)
        Rb = (np.floor(k3 * 40 / np.pi) + 0.5) * np.pi / k3
    orb = Orbit("bump", Rb, h, verbose=verbose)
    f_guess = P.curve(p, orb.r)[0]
    T_guess = 2 * np.pi / p["omega"]
    rows = []
    status = "ok"
    prev = None
    for c in cs:
        try:
            if prev is not None:
                # predictor: scale the last orbit to the new centre value
                f_guess = prev[0] * (c / prev[0][0])
                T_guess = prev[1]
            f0, T = orb.solve(c, f_guess, T_guess)
        except Exception as e:                          # noqa: BLE001 (report and stop)
            status = "Newton-Krylov failed at c = %g: %s: %s" % (c, type(e).__name__, str(e)[:200])
            break
        row = orb.observables(f0, T)
        if floquet:
            mu = orb.monodromy_eigs(f0, T)
            row["floquet_leading"] = [[float(m.real), float(m.imag), float(abs(m))] for m in mu]
            row["growth_rate"] = float(np.log(abs(mu[0])) / T)
        row["c"] = float(c)
        row["f0"] = [float("%.9g" % x) for x in f0]
        rows.append(row)
        prev = (f0, T)
        if verbose:
            print({k: row[k] for k in ("c", "omega", "omega_F0", "E", "hw", "X_max_over_period",
                                       "period_closure_rel") if k in row}, row.get("growth_rate"), flush=True)
    return dict(description=__doc__, box=dict(R=Rb, h=h, closed=True), seed=dict(
        F0=cs[0], omega=p["omega"], source="profiles.solve_point(R bump, x=F0)"),
        r=orb.r.tolist(), rows=rows, status=status, wall_s=time.time() - t0)


if __name__ == "__main__":
    out = run()
    print(write_json(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "radial5",
                                  "periodic_R_bump.json"), out))
