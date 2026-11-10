"""Addendum 2 item 3(a): the t = 0 cross-energy surplus of snapshot starts (engine 8.527 against the family
quadrature 7.757 at r = 20; 2.228 against 1.533 at r = 24).

1. Angular average of the settled primary source (reference state, phase 0): phi_s(r), pi_s(r) in shells.
2. Tail: fit A_s e^{-kappa_s r}/r on r = 12 to 18 (kappa_s from the source's settled omega); residual
   phi_s - tail for r = 12 to 26 (the non-evanescent part).
3. Cross energy of two copies at phases (0, 0), windowed (26, 30) as the snapshot start, by axisymmetric
   quadrature of the radial profiles (static.moments): (i) the measured profile (phi_s, pi_s); (ii) the measured
   core with the fitted evanescent tail beyond r = 12 and pi = 0 beyond 12; (iii) the family profile at the settled
   omega. (i) - (ii) is the part carried by the non-evanescent field; (ii) - (iii) the core difference.
4. What the residual does to V_N: its cross with the partner's core oscillates at (n - 1) omega and averages out of
   whole-period N, but stays in the conserved E at its t = 0 value.
Usage: PYTHONPATH=. python outputs/session4/testA/diag_surplus.py   (one process)
"""
import json
import os

import numpy as np
from scipy.interpolate import CubicSpline

from swirl.engine.snapshot import load_source, window_profile
from swirl.session4.testA import driver as D
from swirl.session4.testA import static as S

OUT = os.path.join(D.ROOT, D.OUT)


class Prof:
    """Lump-like radial profile from samples (f and its spline derivative), windowed (26, 30)."""

    def __init__(self, r, f, beta, omega, window=(26.0, 30.0)):
        self.spl = CubicSpline(r, f, bc_type=((1, 0.0), "not-a-knot"))
        self.d = self.spl.derivative()
        self.R = float(r[-1])
        self.window = window
        self.kappa = float(np.sqrt(1 - omega ** 2))
        self.omega, self.beta, self.g6 = omega, beta, 0.09 * beta

    def ff(self, r):
        r = np.asarray(r, float)
        rr = np.minimum(r, self.R)
        f, g = self.spl(rr), self.d(rr)
        f = np.where(r <= self.R, f, 0.0)
        g = np.where(r <= self.R, g, 0.0)
        w = window_profile(r, *self.window)
        u = np.clip((r - self.window[0]) / (self.window[1] - self.window[0]), 0, 1)
        dw = -np.pi / (self.window[1] - self.window[0]) * np.cos(0.5 * np.pi * u) * np.sin(0.5 * np.pi * u)
        dw = np.where((r > self.window[0]) & (r < self.window[1]), dw, 0.0)
        return w * f, w * g + dw * f


def shells(field, dx, N, rmax=34.0, dr=0.1375):
    c = N // 2
    x = dx * (np.arange(N) - c)
    R = np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)
    edges = np.arange(0.0, rmax + dr, dr)
    idx = np.digitize(R.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < edges.size - 1)
    cnt = np.bincount(idx[ok], minlength=edges.size - 1)
    out = []
    for f in field:
        s = np.bincount(idx[ok], weights=f.ravel()[ok], minlength=edges.size - 1)
        out.append(s / np.maximum(cnt, 1))
    rc = np.bincount(idx[ok], weights=R.ravel()[ok], minlength=edges.size - 1) / np.maximum(cnt, 1)
    keep = cnt > 0
    return rc[keep], [o[keep] for o in out]


def slice_cross_general(P1, P2, Pt, r):
    """Cross energy of two copies at phase 0: int pi1 pi2 + grad phi1 . grad phi2 + phi1 phi2 + V_NL cross."""
    mo = S.moments(P1, P1, r, h=0.5, n=10, X=40.0)
    mt = S.moments(Pt, Pt, r, h=0.5, n=10, X=40.0) if Pt is not None else dict(I_bil=0.0)
    return mt["I_bil"] + mo["I_dot"] + mo["I_bil"] + S._nl(mo, 1.0, 1.0, P1.g6), mo, mt


def main():
    src = D.primary_source()
    phi, pi, meta = load_source(src)
    N, dx, om = meta["params"]["N"], meta["dx"], meta["omega"]
    beta = 4.0 / 3.0
    rc, (fs, gs) = shells((phi, pi), dx, N)
    rc[0] = 0.0                       # the first shell holds only the centre point
    kap = float(np.sqrt(1 - om ** 2))
    sel = (rc >= 12) & (rc <= 18)
    A_s = float(np.exp(np.mean(np.log(fs[sel] * rc[sel]) + kap * rc[sel])))
    tail = A_s * np.exp(-kap * rc) / np.maximum(rc, 1e-9)
    fam = S.Lump(beta, om)
    prof_res = []
    for r0 in (12, 14, 16, 18, 20, 22, 24, 26, 28, 30):
        i = int(np.argmin(np.abs(rc - r0)))
        prof_res.append(dict(r=float(rc[i]), phi_s=float(fs[i]), tail=float(tail[i]), family=float(fam.ff(np.array([rc[i]]))[0][0]),
                             residual=float(fs[i] - tail[i]), residual_over_tail=float(fs[i] / tail[i] - 1),
                             pi_s=float(gs[i])))
    # profiles for the quadrature
    meas = Prof(rc, fs, beta, om)
    meas_t = Prof(rc, gs, beta, om)
    f_ev = np.where(rc < 12, fs, tail)
    blend = np.clip((rc - 11.0) / 2.0, 0, 1)
    f_ev = (1 - blend) * fs + blend * tail
    g_ev = (1 - blend) * gs
    evan = Prof(rc, f_ev, beta, om)
    evan_t = Prof(rc, g_ev, beta, om)
    rows = []
    for r in (16.0, 20.0, 24.0, 28.0):
        Ei, _, mt = slice_cross_general(meas, meas, meas_t, r)
        Eii, _, mt2 = slice_cross_general(evan, evan, evan_t, r)
        mo3 = S.moments(fam, fam, r, h=0.5, n=10, X=40.0)
        Eiii = S.slice_cross(mo3, fam, fam, 0.0, 0.0)
        rows.append(dict(r=r, E_cross_measured_profile=Ei, E_cross_measured_core_evanescent_tail=Eii,
                         E_cross_family_at_settled_omega=Eiii, kinetic_cross_measured=mt["I_bil"],
                         kinetic_cross_evanescent=mt2["I_bil"], surplus_from_nonevanescent=Ei - Eii,
                         core_difference=Eii - Eiii))
        print(rows[-1], flush=True)
    t0 = {}
    for nm, r in (("smoke_pA_r20_p000", 20.0),):
        p = os.path.join(OUT, "t0", nm + ".json")
        if os.path.exists(p):
            t0[str(r)] = json.load(open(p))["t0"]["E_cross_engine"]
    rec = dict(what=__doc__, source=src, sha256=meta["sha256"], omega_source=om, kappa_source=kap, A_fit_12_18=A_s,
               A_family_at_settled_omega=fam.A, profile_comparison=prof_res, cross_energy=rows,
               engine_t0_E_cross=t0, reviewer_numbers=dict(r20=dict(engine=8.527, family=7.757),
                                                          r24=dict(engine=2.2277, family=1.5330)))
    S.dump(rec, os.path.join(OUT, "diag_surplus.json"))
    print(json.dumps(prof_res, indent=1))


if __name__ == "__main__":
    main()
