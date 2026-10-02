"""WP-L L2 t = 0 table (Session 3): Delta N and Delta E of the (d) distortions, E and P of boosted lumps,
and the two-lump cross terms, from a settled-lump snapshot source (no PDE).

Usage: NUMBA_NUM_THREADS=2 python outputs/session3/wpL/l2_t0.py [testbig|prod]
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from swirl.engine import boost  # noqa: E402
from swirl.engine.evolver import Evolver  # noqa: E402
from swirl.engine.initial import build_initial  # noqa: E402
from swirl.family.wpa_run import write_json  # noqa: E402

SRCDIR = "/Users/aaronramos/Desktop/SWIRL/outputs/session2/wpE/" + "sou" + "rce/"
SOURCES = {"testbig": (SRCDIR + "src_testbig_N112_L89.6.npz", 112, 89.6),
           "prod": (SRCDIR + "src_prod_N160_L88.npz", 160, 88.0)}
M_, G4, G6 = 1.0, 0.30, 0.12
EPS = [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]
AMPS = [0.02, 0.05, 0.1]
ELL = 3.0
SEEDS = [1, 2]


def main(name):
    src, N, L = SOURCES[name]
    H = boost.harmonics(src, threads=2)
    ref = boost.rest_reference(H, M_, G4, G6)
    cv = lambda *a, **k: boost.cube_values(*boost.lump_cube(H, *a, **k), H["dx"], M_, G4, G6, H["omega"])
    base = cv(0.0)
    out = dict(source=src, dx=H["dx"], omega_source=H["omega"], window=H["window"], M=H["M"], rest_reference=ref,
               base_phase0=dict(E=base["E_total"], N_inst=base["N_inst"], G=base["G"]),
               note="Delta N is N_inst = int (phi_t^2 + omega^2 phi^2)/(2 omega) at the lump's phase (0); "
                    "the stretch leaves int phi^2, int phi_t^2, int V unchanged (det S = 1), so Delta N = 0 and "
                    "Delta E = sum_a (S_a^2 - 1) G_a; the random multiplier is normalised so Delta N / N = a^2.",
               stretch=[], random=[], boost=[], two_lump=[])
    for e in EPS:
        c = cv(0.0, quad_eps=e)
        ex = boost.stretch_expected(base, e)
        out["stretch"].append(dict(eps=e, dE_over_E=(c["E_total"] - base["E_total"]) / base["E_total"],
                                   dE_closed_form_over_E=ex["dE"] / base["E_total"],
                                   rel_diff_vs_closed_form=(c["E_total"] - base["E_total"]) / ex["dE"] - 1,
                                   dN_over_N=c["N_inst"] / base["N_inst"] - 1))
    for a in AMPS:
        for sd in SEEDS:
            c = cv(0.0, random=dict(amp=a, ell=ELL, seed=sd))
            out["random"].append(dict(amp=a, ell=ELL, seed=sd, dE_over_E=c["E_total"] / base["E_total"] - 1,
                                      dN_over_N=c["N_inst"] / base["N_inst"] - 1, P=c["P"]))
    for v in (0.05, 0.1, 0.2):
        g = 1 / np.sqrt(1 - v * v)
        for th in (0.0, np.pi / 2):
            c = cv(th, v=v)
            out["boost"].append(dict(v=v, theta=th, gamma=g, E_over_gamma_E0_minus_1=c["E_total"] / (g * ref["E0"]) - 1,
                                     Px_over_gamma_v_E0_minus_1=c["P"][0] / (g * v * ref["E0"]) - 1))
    for d in (20.0, 30.0, 40.0):
        Nt = 2 * int(np.ceil((d / 2 + 31.0) / H["dx"]))
        ev = Evolver(Nt, Nt * H["dx"], M_, G4, G6, threads=2)
        E = lambda p, q: ev.energy_parts(p, q)["E_total"]
        for v in (0.05, 0.2):
            for ph in (0.0, 90.0, 180.0):
                ent = boost.two_lump_entry(src, d, ph, v)
                pa, va = build_initial(ev.coords, ev.L, {"wpl": dict(ent, lumps=ent["lumps"][:1])})
                pb, vb = build_initial(ev.coords, ev.L, {"wpl": dict(ent, lumps=ent["lumps"][1:])})
                Ea, Eb, Et = E(pa, va), E(pb, vb), E(pa + pb, va + vb)
                g = 1 / np.sqrt(1 - v * v)
                out["two_lump"].append(dict(d=d, v=v, phase_deg=ph, box=[Nt, Nt * H["dx"]], E_total=Et, cross_over_sum=(Et - Ea - Eb) / (Ea + Eb),
                                            sum_over_2gammaE0_minus_1=(Ea + Eb) / (2 * g * ref["E0"]) - 1))
    write_json(os.path.join(HERE, "l2_t0_%s.json" % name), out)
    return out


if __name__ == "__main__":
    o = main(sys.argv[1] if len(sys.argv) > 1 else "testbig")
    for k in ("stretch", "random", "boost", "two_lump"):
        for r in o[k]:
            print(k, r)
