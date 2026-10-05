"""F-028 diagnostic (calculation, no PDE): regenerate the t = 0 random multiplier of the dip L3-3 jobs with the generator in
swirl/engine/boost.py (lump_cube, random={amp, ell, seed}), and split the added N by wavenumber and by region.
The multiplier is f = 1 + a xi, phi -> f phi, phi_t -> f phi_t, with xi normalised so that <xi>_rho = 0 and <xi^2>_rho = 1
(rho = phi_t^2 + omega^2 phi^2), so the first-order N change is zero and the added N is the quadratic part a^2 N0.
The perturbation (delta phi, delta phi_t) = (f-1)(phi, phi_t) is Fourier decomposed; N_pert = int(delta phi_t^2 + omega^2 delta phi^2)/(2 omega)."""
import json
import os
import sys
import numpy as np
from dcommon import HERE, ROOT, wpl, write_json
from swirl.engine import boost

SRC = os.path.join(ROOT, "outputs", "session3", "wpL3", "sources", "src_dip_N160_L88.npz")
M, G4, G6 = 1.0, 0.3, 0.12
ELL = 3.4
HW = wpl.start_numbers("dip")[3]["hw"]


def band_split(dphi, dpt, dx, omega, edges):
    n = dphi.shape[0]
    Fp, Ft = np.fft.fftn(dphi), np.fft.fftn(dpt)
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    K = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
    dens = (np.abs(Ft) ** 2 + omega ** 2 * np.abs(Fp) ** 2) / (2 * omega) * dx ** 3 / n ** 3
    tot = float(dens.sum())
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        out["%.3g<=k<%.3g" % (lo, hi if np.isfinite(hi) else 99)] = float(dens[(K >= lo) & (K < hi)].sum() / tot)
    cum = {}
    for kc in edges[1:-1]:
        cum["k<%.3g" % kc] = float(dens[K < kc].sum() / tot)
    return tot, out, cum


if __name__ == "__main__":
    H = boost.harmonics(SRC, window=[26.0, 30.0], M=32, threads=1)
    omega, dx, n = H["omega"], H["dx"], 2 * H["Mc"]
    ref = boost.rest_reference(H, M, G4, G6, threads=1)
    phi0, pt0 = boost.lump_cube(H, 0.0)
    v0 = boost.cube_values(phi0, pt0, dx, M, G4, G6, omega, threads=1)
    x = dx * (np.arange(n) - n // 2)
    R = np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)
    edges = [0.0, 0.5 / HW, 1.0 / HW, 2.0 / HW, np.sqrt(1 - omega ** 2), 1.0, 1.5, np.inf]
    edges = sorted(set(edges))
    out = dict(source=SRC, omega=omega, dx=dx, n=n, hw=HW, ell=ELL, N0_inst=v0["N_inst"], E0=v0["E_total"], band_edges=[float(e) for e in edges],
               kappa_gap=float(np.sqrt(1 - omega ** 2)), runs=[])
    for a in (0.1, 0.15):
        for seed in (1, 2, 3):
            phi1, pt1 = boost.lump_cube(H, 0.0, random=dict(amp=a, ell=ELL, seed=seed))
            v1 = boost.cube_values(phi1, pt1, dx, M, G4, G6, omega, threads=1)
            dphi, dpt = phi1 - phi0, pt1 - pt0
            Np, bands, cum = band_split(dphi, dpt, dx, omega, edges)
            rho = pt0 ** 2 + omega ** 2 * phi0 ** 2
            # real-space split of the perturbation's N
            dens = (dpt ** 2 + omega ** 2 * dphi ** 2) / (2 * omega) * dx ** 3
            reg = {("r<%.3g hw" % q): float(dens[R < q * HW].sum() / dens.sum()) for q in (1, 2, 3)}
            # low-pass N: N_inst of (phi0 + lowpass(dphi)) minus N0, as a fraction of the total added N, for each cutoff
            Fp, Ft = np.fft.fftn(dphi), np.fft.fftn(dpt)
            k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
            K = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
            lp = {}
            for kc in (0.5 / HW, 1.0 / HW, 2.0 / HW):
                m_ = K < kc
                dl = np.real(np.fft.ifftn(Fp * m_)); dtl = np.real(np.fft.ifftn(Ft * m_))
                vl = boost.cube_values(phi0 + dl, pt0 + dtl, dx, M, G4, G6, omega, threads=1)
                lp["k<%.3g" % kc] = dict(dN_over_added=(vl["N_inst"] - v0["N_inst"]) / (v1["N_inst"] - v0["N_inst"]),
                                         dE_over_E0=(vl["E_total"] - v0["E_total"]) / v0["E_total"])
            # overlap of the first-order perturbation with the lump's own profile (amplitude mode) and its width mode
            ov_phi = float(np.sum(dphi * phi0) / np.sum(phi0 * phi0))
            ov_pt = float(np.sum(dpt * pt0) / np.sum(pt0 * pt0))
            row = dict(amp=a, seed=seed, dN_over_N=(v1["N_inst"] - v0["N_inst"]) / v0["N_inst"], expected_a2=a * a,
                       dE_over_E=(v1["E_total"] - v0["E_total"]) / v0["E_total"], N_pert=Np, N_pert_over_a2N0=Np / (a * a * v0["N_inst"]),
                       N_fraction_by_k_band=bands, N_fraction_cumulative=cum, N_fraction_by_region=reg,
                       lowpass_added_N=lp, overlap_dphi_with_phi0=ov_phi, overlap_dpt_with_pt0=ov_pt)
            out["runs"].append(row)
            print(a, seed, row["dN_over_N"], row["dE_over_E"], cum, flush=True)
            write_json(os.path.join(HERE, "d028_partial.json"), out)
