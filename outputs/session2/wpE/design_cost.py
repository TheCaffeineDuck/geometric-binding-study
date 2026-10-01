"""WP-E design and cost table: box, grid, sponge clearance, memory and run time for rings of 3 to 8
lumps and the tetrahedron, octahedron, cube and icosahedron (with and without a centre lump) at
nearest-neighbour spacings 3 to 16 (and far-apart controls at 24 and 30), dx = 0.55, dt = 0.04.

Box rule (make_configs.auto_box): L/2 = R_ext + max(d_sp + 12, d_face) + 0.5 with Phi(d_sp) = 1e-3 Phi0
(sponge guard) and Phi(d_face) = 1e-6 Phi0 (face guard); N the smallest even 5-smooth integer >= L/dx.
Time: steps/s scaled from the measured K = 5 benchmark (bench_ring5_a*.json) as N^3 log N^3 (FFT-bound),
memory ~150 bytes per point (WP-C measurements) plus ~0.7 GB while a snapshot start is built (the
N = 160 source evolver; the benchmark's peak RSS includes it). Rented scale: c8g.4xlarge, N = 160 pairs, 33.5 steps/s in total
with 8 jobs of 2 threads (4.19 per job), scaled the same way.

Usage: <python> outputs/session2/wpE/design_cost.py [bench json] -> outputs/session2/wpE/design_cost.json
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

from geometry import FAMILIES, positions  # noqa: E402
from make_configs import DEFAULTS, auto_box, smooth5, sponge_clearance  # noqa: E402
from swirl.engine.initial import TAIL_TOL  # noqa: E402

N_MAX = 256
T, DT, DX = 500.0, 0.04, 0.55
SPACINGS = [3, 4, 6, 8, 10, 12, 14, 16, 24, 30]
C8G_PER_JOB_N160 = 33.5 / 8
BYTES_PER_POINT_RUN = 150.0   # WP-C measured 0.6 GB at N = 160 and 2 GB at N = 240 (~145 B per point)
SOURCE_BUILD_GB = 0.7         # snapshot start: the N = 160 source evolver lives during the init build


def scale(rate_ref, N_ref, N):
    return rate_ref * (N_ref ** 3 * np.log(N_ref ** 3)) / (N ** 3 * np.log(N ** 3))


def main():
    prof = json.load(open(os.path.join(ROOT, "outputs/step2/profiles/profile_beta1.3333_w0.9091.json")))
    bfile = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(HERE, "bench_ring5_a*.json")))[-1]
    bench = json.load(open(bfile))
    rate_ref, N_ref = bench["steps_per_s_with_diagnostics"], bench["N"]
    bpp = BYTES_PER_POINT_RUN
    steps = int(round(T / DT))
    rows = []
    for fam in FAMILIES:
        for centre in ((False,) if fam.startswith("ring") else (False, True)):
            for a in SPACINGS:
                pos = positions(fam, a, centre)
                K = len(pos)
                R_ext = float(np.max(np.linalg.norm(pos, axis=1)))
                N, L = auto_box(pos, DX, prof, DEFAULTS["sponge_tol"])
                r_mac = scale(rate_ref, N_ref, N)
                r_c8g = scale(C8G_PER_JOB_N160, 160, N)
                Nb15 = smooth5(1.5 * L / DX)
                Nb20 = smooth5(2.0 * L / DX)
                Nres = smooth5(L / DEFAULTS["res_dx"])
                D = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
                nn = float(np.min(D[np.triu_indices(K, 1)]))
                rows.append({"family": fam, "centre": centre, "K": K, "a": a, "nn": nn, "R_ext": R_ext,
                             "L": L, "N": N, "fits_N256": N <= N_MAX, "memory_GB": bpp * N ** 3 / 1e9,
                             "memory_peak_with_build_GB": bpp * N ** 3 / 1e9 + SOURCE_BUILD_GB,
                             "mac_steps_per_s": r_mac, "mac_hours_T500": steps / r_mac / 3600,
                             "c8g_job_steps_per_s": r_c8g, "c8g_job_hours_T500": steps / r_c8g / 3600,
                             "box_check_x1.5_N": Nb15, "box_check_x1.5_fits": Nb15 <= N_MAX,
                             "box_check_x2_N": Nb20, "box_check_x2_fits": Nb20 <= N_MAX,
                             "res_check_N": Nres, "res_check_fits": Nres <= N_MAX})
    out = {"dx": DX, "dt": DT, "T": T, "steps": steps, "N_max": N_MAX,
           "sponge_clearance_1e-3": sponge_clearance(prof, DEFAULTS["sponge_tol"]),
           "face_clearance_1e-6": sponge_clearance(prof, TAIL_TOL), "sponge_width": 12.0,
           "bench_file": os.path.relpath(bfile, ROOT), "bench_rate": rate_ref, "bench_N": N_ref,
           "bytes_per_point_run": bpp, "bench_peak_rss_GB": bench["peak_rss_GB"], "source_build_GB": SOURCE_BUILD_GB, "c8g_per_job_N160": C8G_PER_JOB_N160, "rows": rows,
           "not_fitting_N256": [r for r in rows if not r["fits_N256"]],
           "box_check_x2_not_fitting": sorted({(r["family"], r["centre"], r["a"]) for r in rows
                                               if not r["box_check_x2_fits"]})}
    with open(os.path.join(HERE, "design_cost.json.tmp"), "w") as f:
        json.dump(out, f, indent=1)
    os.replace(os.path.join(HERE, "design_cost.json.tmp"), os.path.join(HERE, "design_cost.json"))
    for r in rows:
        if r["a"] in (3, 8, 16, 30):
            print("%-12s c=%d K=%2d a=%4.1f R=%5.1f L=%6.1f N=%3d mem=%5.2fGB mac %.2f h c8g %.2f h box1.5 %d box2 %d res %d"
                  % (r["family"], r["centre"], r["K"], r["a"], r["R_ext"], r["L"], r["N"], r["memory_GB"],
                     r["mac_hours_T500"], r["c8g_job_hours_T500"], r["box_check_x1.5_N"], r["box_check_x2_N"],
                     r["res_check_N"]))
    print("not fitting N <= 256:", [(r["family"], r["centre"], r["a"]) for r in out["not_fitting_N256"]])


if __name__ == "__main__":
    main()
