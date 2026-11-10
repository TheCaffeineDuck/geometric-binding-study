"""Writes the WP-C job lists to outputs/session5/trackC/jobs/. Stage 1: everything that does not
depend on a result. Stage 2 (B9, B7 repeats) is written by make_jobs_stage2.py after the B1-B4 results.
Priority tiers follow the task: 1 = A1-A3 and B1-B4; 2 = A6, B5, B6; 3 = the rest."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from swirl.string import jump as JP            # noqa: E402
from swirl.string import orbits as O           # noqa: E402
from swirl.string.core import String           # noqa: E402

JOBS = os.path.join(ROOT, "outputs", "session5", "trackC", "jobs")
NS_GRID = 29e-9       # s per point-step, grid (measured in the build: 23 us/step at N = 800)
NS_FD = 93e-9


def grid_timing(obj, omega, N, exact=False):
    """dt = T / (4 nq) with the nq the exact-state solver used, so that the evolution integrates the
    discrete orbit it started from; sample_dt = ns dt with ns dividing 4 nq (whole samples per period)."""
    kap = O.kappa_grid(omega)
    Ns = min(N, 2 * int(np.ceil(14.0 / kap)) + 41)
    Ns += Ns % 2
    us, nq, info = O.orbit_at("G1", obj, 0.0, omega, Ns)
    per = 2 * np.pi / omega
    dt = per / (4 * nq)
    m = 4 * nq
    divs = [k for k in range(1, m + 1) if m % k == 0 and m / k >= 16]   # >= 16 samples per period
    want = max(1.0, 0.1 / dt)
    if exact:
        ns = min(divs, key=lambda k: abs(np.log(k / want)))
    else:
        ns = int(max(1, min(round(want), m // 16)))
    sdt = ns * dt
    return {"nq": nq, "dt": dt, "ns": ns, "sdt": sdt, "period": per, "m": m,
            "peak": float(np.max(np.abs(S_stretch(us)))) }


def S_stretch(u):
    return np.roll(u, -1) - u


def T_for_cycles(tm, ncyc, exact=False):
    """exact: T such that the driver's sample count ceil(T / sdt) ends exactly at ncyc whole
    periods (needs ns | 4 nq). Otherwise ncyc nominal periods."""
    if exact:
        return (ncyc * tm["m"] / tm["ns"] - 0.5) * tm["sdt"]
    return ncyc * tm["period"]


def est_grid(T, dt, N, twin=True):
    return T / dt * N * NS_GRID * (2 if twin else 1)


def est_fd(T, dt, N, twin=True):
    return T / dt * N * NS_FD * (2 if twin else 1)


def save(name, jobs):
    p = os.path.join(JOBS, name)
    os.makedirs(JOBS, exist_ok=True)
    json.dump(jobs, open(p, "w"), indent=1)
    print(name, len(jobs), "jobs, est %.2f core-h" % (sum(j.get("est_s", 0) for j in jobs) / 3600))


def grid_job(name, obj, omega, ncyc, N, prio, rp=25, probes=(0, 20, 50, 80), exact=False, **kw):
    tm = grid_timing(obj, omega, N, exact=exact)
    T = T_for_cycles(tm, ncyc, exact)
    job = {"kind": "evolve", "name": name, "medium": "grid", "law": "G1", "N": N, "dt": tm["dt"],
           "sample_dt": tm["sdt"], "T": T, "record_periods": rp, "probes": list(probes),
           "probe_max": kw.pop("probe_max", 30000),
           "start": {"type": "orbit", "obj": obj, "omega": omega}, "prio": prio,
           "est_s": est_grid(T, tm["dt"], N)}
    job.update(kw)
    return job, tm


def stage1():
    A, B = [], []
    # ---- A1 families (the build's family tool, production run)
    fam = []
    for obj in ("sbend", "bump"):
        fam.append({"kind": "family", "name": "A1_family_grid_G1_%s" % obj, "law": "G1", "obj": obj,
                    "omega_start": 2.3, "omega_min": 2.01, "d_stop": 0.99, "prio": 1, "est_s": 300})
    save("A1.json", fam)
    # ---- A2 lifetimes, cap 32,000 cycles
    rates = [2.05, 2.2, 2.5, 3.0, 3.5, 4.2]
    a2 = []
    for obj in ("sbend", "bump"):
        N = 800 if obj == "sbend" else 1600
        for om in rates:
            j, tm = grid_job("A2_%s_w%g" % (obj, om), obj, om, 32000, N, 1,
                             rp=25 if obj == "sbend" else 10)
            a2.append(j)
    save("A2.json", a2)
    # ---- A3 the 1% test, cap 10,000 cycles
    a3 = []
    for obj, om in (("sbend", 2.5), ("sbend", 3.5), ("bump", 2.5)):
        N = 800 if obj == "sbend" else 1600
        for sc in (0.99, 1.01):
            j, tm = grid_job("A3_%s_w%g_x%g" % (obj, om, sc), obj, om, 10000, N, 1,
                             rp=25 if obj == "sbend" else 10)
            cfg = {k: v for k, v in j.items() if k not in ("kind", "prio", "est_s")}
            cfg["start"] = dict(j["start"], scale=sc)
            a3.append({"kind": "custom", "mode": "scale", "name": j["name"], "cfg": cfg,
                       "base_start": j["start"], "scale": sc, "prio": 1, "est_s": j["est_s"]})
    save("A3.json", a3)
    # ---- A4 too long: staggered patterns under a Gaussian envelope
    a4 = []
    for obj in ("sbend", "bump"):
        for env in (3, 6, 12, 24):
            for pk in (0.3, 0.6):
                nm = "A4_%s_env%d_pk%g" % (obj, env, pk)
                cfg = {"medium": "grid", "law": "G1", "N": 1000, "dt": 0.01, "sample_dt": 0.1,
                       "T": 1000 * 3.0, "record_periods": 10, "probes": [0, 20, 50, 80],
                       "start": {"type": "stag", "obj": obj, "env_sigma": env, "peak_stretch": pk},
                       "snap_times": [900.0, 1800.0, 2500.0], "kind": "evolve", "name": nm, "probe_max": 40000}
                a4.append({"kind": "custom", "mode": "stag", "name": nm, "cfg": cfg, "T_cycles": 1000,
                           "prio": 3, "est_s": est_grid(3200, 0.01, 1000)})
    save("A4.json", a4)
    # ---- A5 C4: closed ring against sponge, 10,000 cycles at rate 2.5
    a5 = []
    for tag, N, sp in (("closed", 400, 0.0), ("sponge", 800, None)):
        kw = {"probe_max": 400000, "probes": [0, 20, 50, 60, 80]}
        if sp is not None:
            kw["sponge_len"] = sp
        j, tm = grid_job("A5_sbend_w2.5_%s" % tag, "sbend", 2.5, 10000, N, 3, rp=25, **kw)
        a5.append(j)
    save("A5.json", a5)
    # ---- A6 push
    a6 = []
    lams = [0.05, 0.1, 0.3, 1, 3, 10, 30, 100]
    for obj, om in (("sbend", 2.1), ("sbend", 2.3), ("sbend", 3.0), ("bump", 2.3)):
        N = 1400
        tm = grid_timing(obj, om, N)
        dt = tm["dt"] / 2
        ns = max(1, int(round(0.1 / dt)))
        # one job per lambda (T depends on the push), so that the queue can balance them
        for lam in lams:
            T = float(np.clip(250.0 / (0.05 * lam), 2000.0, 40000.0))
            nm = "A6_%s_w%g_l%g" % (obj, om, lam)
            cfg = {"kind": "evolve", "name": nm, "medium": "grid", "law": "G1", "N": N, "dt": dt,
                   "sample_dt": ns * dt, "T": T, "record_periods": 20, "probes": [0, 20, 50],
                   "probe_max": 20000, "profile_range": 100.0,
                   "start": {"type": "orbit", "obj": obj, "omega": om}, "push": {"lambda": lam},
                   "prio": 2, "est_s": est_grid(T, dt, N)}
            a6.append(cfg)
    save("A6.json", a6)
    # ---- A7 surroundings: dedicated runs with whole periods
    a7 = []
    for obj, om, ncyc, N in (("sbend", 2.5, 1000, 800), ("sbend", 3.5, 1000, 800),
                             ("bump", 2.5, 200, 800)):
        j, tm = grid_job("A7_%s_w%g" % (obj, om), obj, om, ncyc, N, 3, rp=10, profile_range=100.0,
                         exact=True)
        a7.append(j)
    j, tm = grid_job("A7_bump_w2.5_c10", "bump", 2.5, 10, 800, 3, rp=1, profile_range=100.0, exact=True)
    a7.append(j)
    save("A7.json", a7)

    # ---- Batch B (C-smooth, 'fd')
    def fd_job(name, h, start, T, prio, dt=0.02, rp=5, law="G1", medium="fd", L=840.0, **kw):
        N = int(round(L / h))
        job = {"kind": "evolve", "name": name, "medium": medium, "law": law, "h": h, "N": N, "dt": dt,
               "sample_dt": 0.1, "T": float(T), "record_periods": rp, "probes": [0, 20, 50],
               "probe_max": 70000, "start": start, "prio": prio,
               "est_s": (est_fd if medium == "fd" else (lambda *a: est_fd(*a) * 12))(
                   T, dt, N, law != "lin")}
        job.update(kw)
        return job

    hs = [(0.25, "h4"), (0.125, "h8"), (0.0625, "h16")]
    T1000 = 2 * np.pi * 1000.0

    def dt_for(peak):
        k = float(String("fd", "G1", 8, h=0.25).dF(np.array([min(peak + 0.02, 0.995)]))[0])
        return float(min(0.02, 0.1 / np.sqrt(k)))

    starts = []        # (tag, start, T, dt)
    for sg in (0.25, 0.5, 1, 2):
        starts.append(("B1_bump_pk0.8_s%g" % sg, {"type": "gauss", "obj": "bump", "sigma": sg,
                                                    "peak_stretch": 0.8}, T1000, 0.02))
    for sg in (0.25, 0.5):
        starts.append(("B1_bump_pk0.5_s%g" % sg, {"type": "gauss", "obj": "bump", "sigma": sg,
                                                    "peak_stretch": 0.5}, T1000, 0.02))
    for sg in (0.25, 0.5, 1, 2):
        starts.append(("B2_sbend_pk0.8_s%g" % sg, {"type": "gauss", "obj": "sbend", "sigma": sg,
                                                     "peak_stretch": 0.8}, T1000, 0.02))
    cons = []
    for obj, om in (("sbend", 1.2), ("sbend", 1.5), ("sbend", 2.0), ("bump", 1.2), ("bump", 1.5)):
        atop = JP.a_top_for(obj, om, 0.25)
        starts.append(("B3_jump_%s_w%g" % (obj, om), {"type": "jump", "obj": obj, "omega": om,
                                                       "x_s": 0.25}, 2 * np.pi * 1000.0 / om,
                       dt_for(atop)))
    starts.append(("B4_orbit_sbend_s0.5_pk0.8", {"type": "smooth_orbit", "obj": "sbend", "sigma": 0.5,
                                                  "peak_stretch": 0.8}, 2 * np.pi * 1000.0 / 1.4231,
                   0.02))
    for tag, st, T, dt in starts:
        for h, ht in hs:
            B.append(fd_job("%s_%s" % (tag, ht), h, st, T, 1, dt=dt, snap_times=[100.0, 1000.0]))
    # B4 1% tests, h = 1/8
    b4 = []
    for tag, st, om in (("B4_orbit_sbend_1pct", {"type": "smooth_orbit", "obj": "sbend", "sigma": 0.5,
                                                  "peak_stretch": 0.8}, 1.4231),
                        ("B4_jump_sbend_w1.5_1pct", {"type": "jump", "obj": "sbend", "omega": 1.5,
                                                      "x_s": 0.25}, 1.5)):
        for sc in (0.99, 1.01):
            j = fd_job("%s_x%g_h8" % (tag, sc), 0.125, dict(st, scale=sc), 2 * np.pi * 1000 / om, 1,
                       dt=0.02)
            cfg = {k: v for k, v in j.items() if k not in ("kind", "prio", "est_s")}
            b4.append({"kind": "custom", "mode": "scale", "name": j["name"], "cfg": cfg,
                       "base_start": st, "scale": sc, "prio": 1, "est_s": j["est_s"]})
    save("B1to4.json", B)
    save("B4_1pct.json", b4)
    # ---- B5 O3: linear only
    b5 = []
    for sg in (0.1, 0.25, 0.5):
        for h, ht in hs[1:]:
            b5.append(fd_job("B5_lin_bump_s%g_%s" % (sg, ht), h, {"type": "gauss", "obj": "bump",
                             "sigma": sg, "A": 0.1}, 2000.0, 2, dt=0.02, rp=1, law="lin", W=10.0))
    save("B5.json", b5)
    # ---- B6 tighter-faster at fixed displacement amplitude, and O4
    b6 = []
    T300 = 2 * np.pi * 300.0
    for sg in (0.25, 0.35, 0.5, 0.7, 1.0):
        b6.append(fd_job("B6_A0.2_s%g_h8" % sg, 0.125, {"type": "gauss", "obj": "bump", "sigma": sg,
                         "A": 0.2}, T300, 2, dt=0.02, rp=1, profile_range=50.0))
    for pk in (0.3, 0.5, 0.8, 0.95):
        b6.append(fd_job("B6_O4_pk%g_s0.25_h8" % pk, 0.125, {"type": "gauss", "obj": "bump",
                         "sigma": 0.25, "peak_stretch": pk}, T300, 2, dt=dt_for(pk), rp=1,
                         profile_range=50.0))
    # additions: rule 2.2 needs at least 8 points; the same protocol (h = 1/8, 300 cycles) at more widths
    for sg in (0.3, 0.4, 0.6, 0.85, 1.4, 2.0):
        b6.append(fd_job("B6_A0.2_s%g_h8" % sg, 0.125, {"type": "gauss", "obj": "bump", "sigma": sg,
                         "A": 0.2}, T300, 2, dt=0.02, rp=1, profile_range=50.0))
    for obj in ("bump", "sbend"):
        for sg in (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0):
            b6.append(fd_job("B6x_%s_pk0.8_s%g_h8" % (obj, sg), 0.125, {"type": "gauss", "obj": obj,
                             "sigma": sg, "peak_stretch": 0.8}, T300, 2, dt=0.02, rp=1,
                             profile_range=50.0))
    save("B6.json", b6)
    # ---- B7 push in C-smooth (h = 1/8); B8 scheme check
    b7 = []
    lam7 = [0.1, 0.3, 1, 3, 10, 30]
    for tag, st, T in (("gsbend", {"type": "gauss", "obj": "sbend", "sigma": 0.5,
                                    "peak_stretch": 0.8}, T300),
                       ("jsbend", {"type": "jump", "obj": "sbend", "omega": 1.5, "x_s": 0.25},
                        2 * np.pi * 300 / 1.5)):
        for lam in lam7:
            j = fd_job("B7_%s_l%g_h8" % (tag, lam), 0.125, st, T, 3, dt=0.02, rp=1,
                       push={"lambda": lam}, profile_range=100.0)
            b7.append(j)
    save("B7.json", b7)
    b8 = []
    for tag, st, T in (("jsbend1.5", {"type": "jump", "obj": "sbend", "omega": 1.5, "x_s": 0.25},
                        2 * np.pi * 300 / 1.5),
                       ("gbump0.5", {"type": "gauss", "obj": "bump", "sigma": 0.5, "peak_stretch": 0.8},
                        T300)):
        b8.append(fd_job("B8_spectral_%s_h8" % tag, 0.125, st, T, 3, dt=0.02, rp=1,
                         medium="spectral"))
    save("B8.json", b8)


if __name__ == "__main__":
    stage1()
