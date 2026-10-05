"""F-027 post: spectrum of phi(t) recorded at one point at radius 25 from the lump (z axis = stretch axis, x axis = transverse),
window t = 30 to 150, Hann window, zero padded; stretch minus the unstretched control, and each alone."""
import json
import os
import numpy as np
from dcommon import HERE, write_json

R = os.path.join(HERE, "l3", "results")
OMEGA = 0.9068829889592314       # dip lump omega (d028 source file)
BANDS = [("0.80-1.05 (at/below the gap)", 0.80, 1.05), ("1.05-1.15", 1.05, 1.15), ("1.15-1.40 (near gap, 1.2-1.3 m)", 1.15, 1.40),
         ("1.40-2.40", 1.40, 2.40), ("2.40-3.10 (3 omega = 2.72)", 2.40, 3.10), ("3.10-6.00", 3.10, 6.0)]


def load(name):
    r = json.load(open(os.path.join(R, name + ".json")))
    t = np.asarray(r["point"]["t"]); y = np.asarray([p if np.isscalar(p) else p[0] for p in r["point"]["phi"]], float)
    return t, y, r


def spectrum(t, y, lo=30.0, hi=150.0, pad=8):
    s = (t >= lo) & (t <= hi)
    dt = t[1] - t[0]
    yy = (y[s] - y[s].mean()) * np.hanning(s.sum())
    n = int(2 ** np.ceil(np.log2(yy.size * pad)))
    F = np.fft.rfft(yy, n)
    w = 2 * np.pi * np.fft.rfftfreq(n, dt)
    return w, np.abs(F) ** 2


def bands(w, P):
    tot = P[(w > 0.5)].sum()
    out = {k: float(P[(w >= a) & (w < b)].sum() / tot) for k, a, b in BANDS}
    # strongest peak per band
    pk = {}
    for k, a, b in BANDS:
        m = (w >= a) & (w < b)
        i = np.argmax(P * m)
        pk[k] = dict(omega=float(w[i]), power_over_total=float(P[i] / tot))
    return out, pk


if __name__ == "__main__":
    tc, yc, rc = load("f027_control_x")
    out = dict(window=[30.0, 150.0], omega_lump=OMEGA, three_omega=3 * OMEGA, bands=[b[0] for b in BANDS], series={})
    sigs = {"control_x": (tc, yc)}
    for tag in ("q0.1_z", "q0.1_x"):
        p = os.path.join(R, "f027_%s.json" % tag)
        if os.path.exists(p):
            t, y, r = load("f027_" + tag)
            sigs["stretch_" + tag[-1]] = (t, y)
            sigs["stretch_minus_control_" + tag[-1]] = (t, y - np.interp(t, tc, yc))
    for k, (t, y) in sigs.items():
        w, P = spectrum(t, y)
        b, pk = bands(w, P)
        out["series"][k] = dict(rms_phi_30_150=float(np.sqrt(np.mean((y[(t >= 30) & (t <= 150)]) ** 2))), band_fraction=b, band_peak=pk,
                                total_power_rel_control=None)
        print(k, {a: round(v, 4) for a, v in b.items()})
        for a, v in pk.items():
            print("   ", a, round(v["omega"], 3), "%.3g" % v["power_over_total"])
    # absolute power of the difference signal vs control signal in each band
    Pc = spectrum(tc, yc)
    for k in sigs:
        if k.startswith("stretch_minus"):
            w, P = spectrum(*sigs[k])
            out["series"][k]["band_power_over_control_total_power"] = {a: float(P[(w >= lo) & (w < hi)].sum() / Pc[1][(Pc[0] > 0.5)].sum()) for a, lo, hi in BANDS}
    # flux-weighted view: for an outgoing wave of frequency w, k = sqrt(w^2 - 1) and the energy flux through the shell is
    # proportional to w k |phi_w|^2 (omega k |phi|^2 per unit area, spherical 1/r factors common to all frequencies);
    # the N flux is that over w. Mean w_r = (energy flux) / (N flux). Spectra of the stretch-minus-control signal.
    for k_ in sigs:
        w, P = spectrum(*sigs[k_])
        m = (w > 1.0001)
        kk = np.sqrt(np.maximum(w ** 2 - 1.0, 0.0))
        F = w * kk * P
        i_pk = int(np.argmax(P * (w > 0.5)))
        i_fk = int(np.argmax(F * m))
        d = dict(peak_omega_power=float(w[i_pk]), peak_omega_energy_flux=float(w[i_fk]),
                 mean_omega_r_energy_flux_over_N_flux=float(F[m].sum() / (F[m] / w[m]).sum()),
                 flux_band_fraction={a: float(F[(w >= lo) & (w < hi) & m].sum() / F[m].sum()) for a, lo, hi in BANDS})
        half = P > 0.5 * P[i_pk]
        d["power_half_max_range_omega"] = [float(w[half & (w > 0.5)].min()), float(w[half & (w > 0.5)].max())]
        out["series"][k_]["flux_weighted"] = d
        print(k_, "peak power at", round(d["peak_omega_power"], 3), "peak energy flux at", round(d["peak_omega_energy_flux"], 3),
              "mean w_r", round(d["mean_omega_r_energy_flux_over_N_flux"], 3), {a: round(v, 3) for a, v in d["flux_band_fraction"].items()})
    write_json(os.path.join(HERE, "d027_spectrum.json"), out)
