"""Test A static potentials (SESSION_04_PLAN section 2, A0 items 1 and 2), no PDE.

Two spherical lumps f_1(|x - x_1|), f_2(|x - x_2|) on the z axis at z_1 = -r/2, z_2 = +r/2. Every
quantity here is a cross term computed directly (the self terms are never subtracted), by composite
Gauss-Legendre quadrature in cylindrical (rho, z), weight 2 pi rho (the superposition is axisymmetric).

Moments, per separation r:
    I_dot = int grad f_1 . grad f_2,   I_bil = int f_1 f_2,   M[p, q] = int f_1^p f_2^q  (p + q = 4, 6)

From them (m = 1, V = phi^2/2 - g4 phi^4/24 + g6 phi^6/720):
- V_raw (t = 0 slice, phases theta_i, phi_t = -sum omega_i f_i sin theta_i): the energy excess of the
  superposed slice over the two separate slices,
      E_x = omega_1 omega_2 s_1 s_2 I_bil + c_1 c_2 (I_dot + I_bil) + NL cross,  c_i = cos theta_i, s_i = sin theta_i.
  With phi_t = 0 (the plan's V_raw) the phases are 0 and dtheta.
- U_static, equal lumps (common omega): the overlap term of the period-averaged E - omega N,
      (cos dtheta / 2)(I_dot + kappa^2 I_bil) + <V_NL(f_1 cos tau + f_2 cos(tau + dtheta))>_cross
  (WPC_DERIVATION section 2; it equals the fixed-N interaction energy at first order).
- U_amp, any pair (the beat amplitude for unequal lumps): the Session 3 law with the true profiles,
      U_amp = -(1/2)[w_1 int S_1 f_2 + w_2 int S_2 f_1] cos dtheta,  S = (g4/8) f^3 - (g6/192) f^5,
      w_i = omega_i / (omega_1 + omega_2).  For a pure Yukawa partner tail it is the unequal.py law exactly;
  for equal lumps it is the first-order part of U_static.
- N_cross = omega cos(dtheta) I_bil (equal lumps).
- U_tail = -2 pi A^2 cos(dtheta) e^{-kappa r}/r (plan A0).
- F_sup: the period-averaged mid-plane momentum flux of the superposition minus the single lumps
  (WPC_DERIVATION section 1, port of outputs/session2/wpC/derive_pair_law.py::F_sup), for validation.

Floors (plan A0): sigma = max over (i) quadrature resolution (panel h 0.5 / 10 nodes / cutoff 32 against
h 0.4 / 12 nodes / cutoff 40), (ii) profile representation (primary: stored JSON spline against the BVP
solution; other lumps: BVP tol 1e-10 against 1e-8), (iii) a single lump shifted against itself: its
t = 0 slice energy on the pair grid (centred at z_1) against its own grid (centred at 0).
"""
import json
import os

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import comb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
M, G4 = 1.0, 0.30
PRIMARY_JSON = "outputs/step2/profiles/profile_beta1.3333_w0.9091.json"
R_GRID = np.round(np.arange(6.0, 45.0001, 0.25), 10)
BASE = dict(h=0.5, n=10, X=32.0)
FINE = dict(h=0.4, n=12, X=40.0)


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def dump(obj, path):
    path = _abs(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    os.replace(path + ".tmp", path)


class Lump:
    """Radial profile f(r), f'(r) in Set-B-like units (m = 1, g4 = 0.30, g6 = 0.09 beta).

    source "bvp": swirl.family.profile.solve_profile (rescaled, tol) / sqrt(g4); beyond the BVP box the
    exact linear tail. source "json": a stored profile table (cubic spline, clamped at 0) and A e^{-kappa r}/r
    beyond its last point (the convention of derive_pair_law.py). cut: f = 0 beyond cut (the 3D radial
    initial data does this beyond the table; used only to reproduce Session 2's t = 0 check)."""

    def __init__(self, beta, omega, source="bvp", tol=1e-10, json_path=PRIMARY_JSON, cut=None):
        self.beta, self.omega = float(beta), float(omega)
        self.g6 = 0.09 * self.beta
        self.kappa = float(np.sqrt(M * M - self.omega ** 2))
        self.source, self.tol, self.cut = source, tol, cut
        if source == "json":
            p = json.load(open(_abs(json_path)))
            if abs(p["omega"] - self.omega) > 1e-12 or abs(p["g6"] - self.g6) > 1e-12:
                raise ValueError("profile file does not match (beta, omega)")
            self.A, self.kappa = float(p["A"]), float(p["kappa"])
            self.N, self.E, self.hw = p["N"], p["E"], p["half_width"]
            r, P = np.array(p["r"]), np.array(p["Phi"])
            self._spl = CubicSpline(r, P, bc_type=((1, 0.0), "not-a-knot"))
            self._dspl = self._spl.derivative()
            self.R = float(r[-1])
            self.Phi0 = float(P[0])
            self.json_path = json_path
        else:
            from swirl.family.profile import solve_profile
            res = solve_profile(self.omega, 1.0, 1.0, self.beta, tol=tol)
            s = 1.0 / np.sqrt(G4)
            self._sol, self.R = res["_sol"], float(res["R_box"])
            self._s = s
            self.A = res["A"] * s
            self.N, self.E, self.hw = res["N"] / G4, res["E"] / G4, res["half_width"]
            self.Phi0 = res["P0"] * s
            self._PR = float(self._sol.y[0, -1]) * s

    def info(self):
        return dict(beta=self.beta, omega=self.omega, g6=self.g6, source=self.source, tol=self.tol,
                    A=self.A, kappa=self.kappa, N=self.N, E=self.E, hw=self.hw, Phi0=self.Phi0, R_table=self.R,
                    cut=self.cut)

    def ff(self, r):
        """(f, f') at radii r (array)."""
        r = np.asarray(r, float)
        inside = r <= self.R
        f = np.empty_like(r)
        g = np.empty_like(r)
        ri = r[inside]
        if self.source == "json":
            f[inside], g[inside] = self._spl(ri), self._dspl(ri)
            ro = r[~inside]
            t = self.A * np.exp(-self.kappa * ro) / ro
        else:
            y = self._sol.sol(ri)
            f[inside], g[inside] = y[0] * self._s, y[1] * self._s
            ro = r[~inside]
            t = self._PR * self.R / ro * np.exp(-self.kappa * (ro - self.R))
        f[~inside] = t
        g[~inside] = -t * (self.kappa + 1.0 / ro)
        if self.cut is not None:
            f = np.where(r <= self.cut, f, 0.0)
            g = np.where(r <= self.cut, g, 0.0)
        return f, g


# ---------------------------------------------------------------------------------------------- quadrature
def gl(breaks, h, n):
    """Composite Gauss-Legendre nodes and weights on [breaks[0], breaks[-1]], every interval between the
    breakpoints split into equal panels of width <= h, n nodes per panel."""
    x0, w0 = np.polynomial.legendre.leggauss(n)
    xs, ws = [], []
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b <= a:
            continue
        k = max(1, int(np.ceil((b - a) / h - 1e-12)))
        e = np.linspace(a, b, k + 1)
        mid, half = 0.5 * (e[1:] + e[:-1]), 0.5 * np.diff(e)
        xs.append((mid[:, None] + half[:, None] * x0[None, :]).ravel())
        ws.append((half[:, None] * w0[None, :]).ravel())
    return np.concatenate(xs), np.concatenate(ws)


PQ4 = [(3, 1), (2, 2), (1, 3)]
PQ6 = [(5, 1), (4, 2), (3, 3), (2, 4), (1, 5)]


def moments(l1, l2, r, h=0.5, n=10, X=32.0, chunk=64):
    """Cross moments of lumps l1 at z = -r/2 and l2 at +r/2 (see module doc). X: the cross integrand is
    dropped where it is below e^{-X} of its value on the segment (cutoff of the z and rho ranges)."""
    z1, z2 = -r / 2.0, r / 2.0
    ks = l1.kappa + l2.kappa
    Zc = X / ks
    a = r / 2.0
    rho_max = np.sqrt((a + Zc) ** 2 - a * a)
    zb = [z1 - Zc, z1, 0.0, z2, z2 + Zc]
    z, wz = gl(zb, h, n)
    rho, wr = gl([0.0, rho_max], h, n)
    wr = wr * 2.0 * np.pi * rho
    out = dict(I_dot=0.0, I_bil=0.0, **{"M%d%d" % pq: 0.0 for pq in PQ4 + PQ6})
    dz1, dz2 = z - z1, z - z2
    for i in range(0, rho.size, chunk):
        rr = rho[i:i + chunk, None]
        r1 = np.sqrt(rr * rr + dz1[None, :] ** 2)
        r2 = np.sqrt(rr * rr + dz2[None, :] ** 2)
        f1, g1 = l1.ff(r1)
        f2, g2 = l2.ff(r2)
        W = wr[i:i + chunk, None] * wz[None, :]
        dot = g1 * g2 * (rr * rr + dz1[None, :] * dz2[None, :]) / (r1 * r2)
        out["I_dot"] += float(np.sum(W * dot))
        out["I_bil"] += float(np.sum(W * f1 * f2))
        for p, q in PQ4 + PQ6:
            out["M%d%d" % (p, q)] += float(np.sum(W * f1 ** p * f2 ** q))
    out.update(r=float(r), h=h, n=n, X=X, n_points=int(z.size * rho.size), rho_max=float(rho_max), Zc=float(Zc))
    return out


def self_slice_energy(l, zc, zb, h=0.5, n=10, rho_max=60.0, chunk=64):
    """t = 0 slice energy (phi_t = 0, theta = 0) of one lump at z = zc on the z panels with breakpoints zb."""
    z, wz = gl(zb, h, n)
    rho, wr = gl([0.0, rho_max], h, n)
    wr = wr * 2.0 * np.pi * rho
    E = 0.0
    for i in range(0, rho.size, chunk):
        rr = rho[i:i + chunk, None]
        r1 = np.sqrt(rr * rr + (z[None, :] - zc) ** 2)
        f, g = l.ff(r1)
        p2 = f * f
        e = 0.5 * g * g + p2 * (0.5 * M * M + p2 * (-G4 / 24.0 + p2 * l.g6 / 720.0))
        E += float(np.sum(wr[i:i + chunk, None] * wz[None, :] * e))
    return E


def _nl(mo, c1, c2, g6):
    """V_NL cross for a = c1 f1, b = c2 f2: (-g4/24)[(a+b)^4 - a^4 - b^4] + (g6/720)[(a+b)^6 - a^6 - b^6]."""
    s4 = sum(comb(4, p) * c1 ** p * c2 ** q * mo["M%d%d" % (p, q)] for p, q in PQ4)
    s6 = sum(comb(6, p) * c1 ** p * c2 ** q * mo["M%d%d" % (p, q)] for p, q in PQ6)
    return -G4 / 24.0 * s4 + g6 / 720.0 * s6


def slice_cross(mo, l1, l2, th1, th2):
    """t = 0 slice energy excess for phases th1, th2 (phi_i = f_i cos(omega_i t + th_i))."""
    if abs(l1.g6 - l2.g6) > 1e-15:
        raise ValueError("lumps of different beta")
    c1, c2, s1, s2 = np.cos(th1), np.cos(th2), np.sin(th1), np.sin(th2)
    return (l1.omega * l2.omega * s1 * s2 * mo["I_bil"] + c1 * c2 * (mo["I_dot"] + M * M * mo["I_bil"])
            + _nl(mo, c1, c2, l1.g6))


def U_static(mo, l, dth, n_tau=32):
    """Overlap term of the period-averaged E - omega N, equal lumps (common omega, kappa)."""
    tau = 2 * np.pi * np.arange(n_tau) / n_tau
    nl = float(np.mean([_nl(mo, np.cos(t), np.cos(t + dth), l.g6) for t in tau]))
    return 0.5 * np.cos(dth) * (mo["I_dot"] + l.kappa ** 2 * mo["I_bil"]) + nl


def U_amp(mo, l1, l2, dth):
    """Beat-amplitude interaction (Session 3 law with the true profiles); see the module doc."""
    S1f2 = G4 / 8.0 * mo["M31"] - l1.g6 / 192.0 * mo["M51"]
    S2f1 = G4 / 8.0 * mo["M13"] - l2.g6 / 192.0 * mo["M15"]
    w1 = l1.omega / (l1.omega + l2.omega)
    return -0.5 * (w1 * S1f2 + (1.0 - w1) * S2f1) * np.cos(dth)


def U_tail(r, A, kappa, dth=0.0):
    return -2.0 * np.pi * A * A * np.cos(dth) * np.exp(-kappa * r) / r


def F_asym(r, A, kappa, dth=0.0):
    return 2.0 * np.pi * A * A * np.cos(dth) * np.exp(-kappa * r) * (kappa / r + 1.0 / r ** 2)


def unequal_law(r, l1, l2, dth=0.0, Q12=None, Q21=None):
    """Session 3 point law U = -(2 pi/d)[w1 Q1(k2) A2 e^{-k2 d} + w2 Q2(k1) A1 e^{-k1 d}] cos dth, with
    Q_i(k) = int_0^inf S_i(r) r sinh(k r)/k dr from the profiles (Q_i(kappa_i) = A_i)."""
    w1 = l1.omega / (l1.omega + l2.omega)
    Q12 = charge(l1, l2.kappa) if Q12 is None else Q12
    Q21 = charge(l2, l1.kappa) if Q21 is None else Q21
    return -(2 * np.pi / r) * (w1 * Q12 * l2.A * np.exp(-l2.kappa * r)
                               + (1 - w1) * Q21 * l1.A * np.exp(-l1.kappa * r)) * np.cos(dth)


def charge(l, k, n=20000):
    x, w = gl([0.0, 3 * l.hw, 8 * l.hw, 20 * l.hw], 0.05, 8)
    f, _ = l.ff(x)
    S = G4 / 8.0 * f ** 3 - l.g6 / 192.0 * f ** 5
    return float(np.sum(w * S * x * np.sinh(k * x) / k))


def F_sup(l, d, th1, th2, n_tau=96, n_rho=4000):
    """Port of derive_pair_law.py::F_sup for an equal pair (period-averaged mid-plane flux minus singles)."""
    a = d / 2.0
    rho_max = np.sqrt(max(60.0 ** 2 - a * a, 1.0))
    rho = np.linspace(0, rho_max, n_rho)
    r = np.sqrt(a * a + rho ** 2)
    f, fp = l.ff(r)
    fx1, fx2 = fp * a / r, -fp * a / r
    fpe = fp * rho / r
    om = l.omega
    V = lambda p: 0.5 * p * p - G4 / 24 * p ** 4 + l.g6 / 720 * p ** 6
    acc = np.zeros_like(rho)
    for t in 2 * np.pi * (np.arange(n_tau) + 0.5) / n_tau:
        c1, s1, c2, s2 = np.cos(t + th1), np.sin(t + th1), np.cos(t + th2), np.sin(t + th2)
        px, pp, p, pt = fx1 * c1 + fx2 * c2, fpe * (c1 + c2), f * (c1 + c2), -om * f * (s1 + s2)
        fl = -0.5 * px ** 2 + 0.5 * pp ** 2 - 0.5 * pt ** 2 + V(p)
        for c, s, fx in ((c1, s1, fx1), (c2, s2, fx2)):
            fl = fl - (-0.5 * (fx * c) ** 2 + 0.5 * (fpe * c) ** 2 - 0.5 * (om * f * s) ** 2 + V(f * c))
        acc += fl
    acc /= n_tau
    return float(np.trapezoid(acc * 2 * np.pi * rho, rho))


# ---------------------------------------------------------------------------------------------- records
def channel_values(mo, l1, l2, dth):
    """V_raw (phi_t = 0, phases 0 and dth), U_static (equal) or None, U_amp, N_cross (equal)."""
    equal = abs(l1.omega - l2.omega) < 1e-14 and abs(l1.beta - l2.beta) < 1e-14
    out = dict(V_raw=float(slice_cross(mo, l1, l2, 0.0, dth)), U_amp=float(U_amp(mo, l1, l2, dth)))
    out["U_static"] = float(U_static(mo, l1, dth)) if equal else out["U_amp"]
    out["U_static_def"] = "overlap of <E - omega N>" if equal else "U_amp (beat amplitude, Session 3 law form)"
    out["N_cross"] = float(l1.omega * np.cos(dth) * mo["I_bil"]) if equal else None
    return out


def static_record(name, l1, l2, l1_alt, l2_alt, rs=R_GRID, alt_label="", verbose=False):
    """V_raw, U_static, U_amp, U_tail and floors for dtheta = 0 and 180 on the r grid."""
    rows = []
    E_own = self_slice_energy(l1, 0.0, [-80.0, 0.0, 80.0], **{k: BASE[k] for k in ("h", "n")})
    for r in rs:
        mo = moments(l1, l2, r, **BASE)
        mf = moments(l1, l2, r, **FINE)
        ma = moments(l1_alt, l2_alt, r, **BASE)
        Zc = mo["Zc"]
        E_shift = self_slice_energy(l1, -r / 2, [-r / 2 - Zc, -r / 2, 0.0, r / 2, r / 2 + Zc],
                                    rho_max=max(mo["rho_max"], 60.0), **{k: BASE[k] for k in ("h", "n")})
        row = dict(r=float(r), moments=mo, shifted_single_dE=float(E_shift - E_own))
        for lab, dth in (("p000", 0.0), ("p180", np.pi)):
            b, f, a = (channel_values(x, l1, l2, dth) for x in (mo, mf, ma))
            ch = dict(b)
            for key in ("V_raw", "U_static", "U_amp"):
                ch["sigma_quad_" + key] = abs(b[key] - f[key])
                ch["sigma_profile_" + key] = abs(b[key] - a[key])
                ch["sigma_" + key] = max(ch["sigma_quad_" + key], ch["sigma_profile_" + key], abs(row["shifted_single_dE"]))
            ch["U_tail"] = float(U_tail(r, l1.A, l1.kappa, dth)) if l1.omega == l2.omega else None
            ch["U_law"] = float(unequal_law(r, l1, l2, dth)) if l1.omega != l2.omega else ch["U_tail"]
            row[lab] = ch
        rows.append(row)
        if verbose:
            print("r %.2f U %.6e Vraw %.6e tail %s sig %.1e" % (r, row["p000"]["U_static"], row["p000"]["V_raw"],
                  row["p000"]["U_tail"], row["p000"]["sigma_U_static"]), flush=True)
    rec = dict(name=name, what="Test A static potentials (plan A0 items 1, 2) by axisymmetric quadrature",
               lump1=l1.info(), lump2=l2.info(), alt_profiles=dict(lump1=l1_alt.info(), lump2=l2_alt.info(),
                                                                  label=alt_label),
               quadrature=dict(base=BASE, fine=FINE, method="composite Gauss-Legendre in (rho, z), weight 2 pi rho"),
               single_slice_energy_own_grid=E_own, rows=rows, definitions=__doc__)
    add_derived(rec)
    return rec


def add_derived(rec):
    """Force from U_static (spline of U r e^{kappa r}), ratio to the tail law, the floor crossing."""
    rows = rec["rows"]
    r = np.array([q["r"] for q in rows])
    k = rec["lump1"]["kappa"] if rec["lump1"]["omega"] == rec["lump2"]["omega"] else min(
        rec["lump1"]["kappa"], rec["lump2"]["kappa"])
    summ = {}
    for lab in ("p000", "p180"):
        for key in ("U_static", "V_raw"):
            U = np.array([q[lab][key] for q in rows])
            sig = np.array([q[lab]["sigma_" + key] for q in rows])
            g = CubicSpline(r, U * r * np.exp(k * r))
            dU = (g(r, 1) - g(r) * (k + 1.0 / r)) * np.exp(-k * r) / r
            for q, v in zip(rows, dU):
                q[lab]["F_" + key] = float(v)        # F = +dU/dr, positive = attraction
            res = np.abs(U) > 3 * sig
            summ["%s_%s" % (key, lab)] = dict(
                all_resolved=bool(res.all()),
                first_unresolved_r=(float(r[np.argmin(res)]) if not res.all() else None),
                max_sigma=float(sig.max()), max_sigma_over_absU=float(np.max(sig / np.abs(U))))
        if rows[0][lab]["U_tail"] is not None:
            for q in rows:
                q[lab]["U_static_over_tail"] = q[lab]["U_static"] / q[lab]["U_tail"]
                q[lab]["V_raw_over_U_static"] = q[lab]["V_raw"] / q[lab]["U_static"]
    rec["summary"] = summ


LUMPS = {
    "primary": (4.0 / 3.0, 0.909149280617113),
    "b5": (5.0, 0.975598235049607),
}


def lumps_for(name, unequal=None):
    """(l1, l2, l1_alt, l2_alt, alt_label). unequal: ((beta, omega1), (beta, omega2))."""
    if name == "primary":
        b, w = LUMPS["primary"]
        l = Lump(b, w)
        la = Lump(b, w, source="json")
        return l, l, la, la, "stored JSON profile (spline, tail A e^{-kappa r}/r beyond r = 34)"
    if name == "b5":
        b, w = LUMPS["b5"]
        l, la = Lump(b, w), Lump(b, w, tol=1e-8)
        return l, l, la, la, "BVP tol 1e-8"
    (b1, w1), (b2, w2) = unequal
    return (Lump(b1, w1), Lump(b2, w2), Lump(b1, w1, tol=1e-8), Lump(b2, w2, tol=1e-8), "BVP tol 1e-8")


def validation(verbose=False):
    """Against WPC_DERIVATION's F_sup/F_asym table and Session 2's t = 0 pair quadrature."""
    b, w = LUMPS["primary"]
    lj = Lump(b, w, source="json")
    lb = Lump(b, w)
    table = {8: 0.346, 10: 0.736, 12: 0.907, 14: 0.968, 16: 0.9888, 18: 0.9960, 20: 0.9986, 24: 0.9998}
    prev = json.load(open(_abs("outputs/session2/wpC/pair_law_predictions.json")))
    rows = []
    for d, ref in table.items():
        Fs = F_sup(lj, d, 0.0, 0.0)
        Fa = F_asym(d, lj.A, lj.kappa)
        h = 0.01
        Up = U_static(moments(lb, lb, d + h, **BASE), lb, 0.0)
        Um = U_static(moments(lb, lb, d - h, **BASE), lb, 0.0)
        dU = (Up - Um) / (2 * h)
        rows.append(dict(d=d, F_sup_over_F_asym=Fs / Fa, table=ref, dUstatic_dd_over_F_asym=dU / Fa,
                         dUstatic_dd_over_F_sup=dU / Fs))
        if verbose:
            print(rows[-1], flush=True)
    # Session 2 t = 0 check: d = 10, theta = (0, 60 deg), radial ansatz cut at r = 34 on N = 128, L = 72
    t0 = json.load(open(_abs("outputs/session2/wpC/checks/pair_t0_quadrature.json")))
    lc = Lump(b, w, source="json", cut=34.0)
    mo = moments(lc, lc, 10.0, h=0.25, n=12, X=40.0)
    Ex = slice_cross(mo, lc, lc, 0.0, np.deg2rad(60.0))
    mo2 = moments(lj, lj, 10.0, h=0.25, n=12, X=40.0)
    Ex2 = slice_cross(mo2, lj, lj, 0.0, np.deg2rad(60.0))
    return dict(F_sup_table=rows,
                F_sup_table_max_abs_diff=max(abs(q["F_sup_over_F_asym"] - q["table"]) for q in rows),
                session2_F_sup_file={str(k): v for k, v in prev.items() if "F_sup" in k} if isinstance(prev, dict) else None,
                t0_pair=dict(d=10.0, theta_deg=[0.0, 60.0], E_cross_session2_engine=t0["E_cross_engine"],
                             E_cross_session2_quadrature=t0["E_cross_quadrature"], E_cross_axisym_cut34=Ex,
                             rel_diff_vs_engine=Ex / t0["E_cross_engine"] - 1.0, E_cross_axisym_uncut=Ex2,
                             U_tail_d10_60=float(U_tail(10.0, lj.A, lj.kappa, np.deg2rad(60.0)))))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["validation", "primary", "b5", "unequal"])
    ap.add_argument("--omegas", type=float, nargs=2, default=None, help="unequal pair omegas (beta 4/3)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--r-range", type=float, nargs=3, default=None,
                    help="r_min r_max step (default the plan grid 6 to 45 by 0.25); output name gets _ext")
    a = ap.parse_args()
    od = "outputs/session4/testA/static"
    if a.what == "validation":
        dump(validation(verbose=True), a.out or od + "/validation.json")
        return
    if a.what == "unequal":
        un = ((4.0 / 3.0, a.omegas[0]), (4.0 / 3.0, a.omegas[1]))
        ls = lumps_for("unequal", un)
        name = "unequal_%.6f_%.6f" % tuple(a.omegas)
    else:
        ls = lumps_for(a.what)
        name = a.what
    rs = R_GRID
    if a.r_range:
        rs = np.round(np.arange(a.r_range[0], a.r_range[1] + 1e-9, a.r_range[2]), 10)
        name += "_ext"
    rec = static_record(name, *ls[:4], rs=rs, alt_label=ls[4], verbose=True)
    dump(rec, a.out or "%s/static_%s.json" % (od, name))


if __name__ == "__main__":
    main()
