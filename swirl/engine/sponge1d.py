"""1D design scan for the sponge profile gamma_max * u^power (how SPONGE_DEFAULT was chosen).

    python -m swirl.engine.sponge1d [--out outputs/engine/absorber_design_scan.json]

Linear Klein-Gordon (m = 1) on a periodic line: interior |x| < a = 15, sponge
of the given width on each side (so a packet leaving the interior crosses one
layer, wraps, and crosses the other before it can re-enter). A right-moving
packet with central k0 (positive-frequency projection, Gaussian envelope
s = min(1.5 lambda, a/3)) starts at x = -a/2. After the direct packet has
left, the largest energy found back in the interior, over a further 2a/v,
is reported as a fraction of the initial energy: reflection plus wrap-around.
Below k0 ~ 0.5 the number is not a reflection coefficient: the envelope then
carries k ~ 0 content that never leaves the interior.
"""

import argparse
import json

import numpy as np

from swirl.engine.run import atomic_write_json

A_INT = 15.0
DX = 0.2
DT = 0.05
KS = [0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.8]
DESIGNS = [
    {"gamma": 0.3, "width": 10.0, "power": 3}, {"gamma": 1.0, "width": 10.0, "power": 3},
    {"gamma": 2.0, "width": 10.0, "power": 3}, {"gamma": 1.0, "width": 14.0, "power": 3},
    {"gamma": 2.0, "width": 14.0, "power": 3}, {"gamma": 1.0, "width": 10.0, "power": 2},
    {"gamma": 4.0, "width": 14.0, "power": 3}, {"gamma": 2.0, "width": 12.0, "power": 3},
]


def returned_fraction(width, gmax, power, k0, a=A_INT, m=1.0):
    Lh = a + width
    N = int(round(2 * Lh / DX))
    N += N % 2
    h = 2 * Lh / N
    x = -Lh + h * np.arange(N)
    k = 2 * np.pi * np.fft.fftfreq(N, h)
    om = np.sqrt(k ** 2 + m ** 2)
    g = gmax * np.clip((np.abs(x) - a) / width, 0, 1) ** power
    s = min(1.5 * 2 * np.pi / k0, a / 3)
    x0 = -a / 2
    ph = np.fft.fft(np.exp(-(x - x0) ** 2 / (2 * s ** 2) + 1j * k0 * (x - x0)))
    ph[k <= 0] = 0
    phi = np.real(np.fft.ifft(ph))
    pi = np.real(np.fft.ifft(-1j * om * ph))
    inner = np.abs(x) < a

    def e_in(phi, pi):
        d = np.real(np.fft.ifft(1j * k * np.fft.fft(phi)))
        return np.sum((0.5 * pi ** 2 + 0.5 * d ** 2 + 0.5 * m ** 2 * phi ** 2)[inner]) * h

    def rhs(phi, pi):
        return pi, np.real(np.fft.ifft(-k ** 2 * np.fft.fft(phi))) - m ** 2 * phi - g * pi

    E0 = e_in(phi, pi)
    v = k0 / np.sqrt(k0 ** 2 + m ** 2)
    t_exit = (1.5 * a + 6 * s) / v
    t, emax = 0.0, 0.0
    while t < t_exit + 2 * a / v:
        k1 = rhs(phi, pi)
        k2 = rhs(phi + DT / 2 * k1[0], pi + DT / 2 * k1[1])
        k3 = rhs(phi + DT / 2 * k2[0], pi + DT / 2 * k2[1])
        k4 = rhs(phi + DT * k3[0], pi + DT * k3[1])
        phi = phi + DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        pi = pi + DT / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        t += DT
        if t > t_exit:
            emax = max(emax, e_in(phi, pi))
    return emax / E0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for d in DESIGNS:
        fr = [returned_fraction(d["width"], d["gamma"], d["power"], k0) for k0 in KS]
        rows.append({**d, "returned_energy_fraction": fr})
        print(d, " ".join("%.1e" % f for f in fr), flush=True)
    out = {"params": {"m": 1.0, "interior_half_width": A_INT, "dx": DX, "dt": DT, "k0": KS,
                      "envelope": "s = min(1.5*lambda, a/3)", "x0": -A_INT / 2},
           "designs": rows}
    if a.out:
        atomic_write_json(out, a.out)


if __name__ == "__main__":
    main()
