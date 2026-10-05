"""WP-L3 (Session 3): the 3D batch of the lone-lump break test, for rented compute.

Commands (from the repo root; PYTHONPATH = repo root):
    python -m swirl.engine.l3 jobs  [--out FILE]                 write the job list (JSON) and the cost table
    python -m swirl.engine.l3 run   --jobs FILE [--workers 8] [--threads 2] [--stages 1,2,3,...]
    python -m swirl.engine.l3 post  --jobs FILE [--radial DIR]   summaries from the results present

`run` is a dependency-aware process pool: a job starts when the jobs it depends on have a result; a job
whose dependency failed is skipped. It is resumable: a job with a result JSON is done; an interrupted
run resumes from its checkpoint (run.run, checkpoint every 50 time units); a source resumes from its
stage-1 checkpoint. Results: outputs/session3/wpL3/results/<name>.json (atomic). Checkpoints:
checkpoints/session3/wpL3/<name>/. Sources: outputs/session3/wpL3/sources/ (npz never committed).

Grid: dx = 0.55, dt = 0.04 (the WP-E production values; dt scales with dx in the resolution check).
Units Set-B-like: m = 1, g4 = 0.30, g6 = 0.09 beta. Default sponge (gamma 2, width 12, cubic).

Stages (jobs list order):
 1  sources: dip (beta 4/3, omega 0.906505) on N = 160, L = 88; b5 (beta 5, omega 0.975598) on N = 216,
    L = 118.8 (the A10 grid); settled to t_s = 500; crop loss measured for the listed windows. The primary
    source is the WP-E production source (same dx), shipped, checked by hash.
 2  boosted-lump gate at v = 0, 0.05, 0.2 on dip and primary (the collision sources).
 3  (d) distortions per start: stretch eps = 0.05, 0.1, 0.2; random multiplier a = 0.02, 0.05, 0.1
    (ell = hw, seed 1); an undisturbed control; T = 2000; shape (a2, a3) every time unit in R = 3 hw.
 4  (e) collisions at dip and primary: two settled copies at x = -+ d/2, each moving at v toward the other
    (closing speed 2 v), phases 0 and phi at their centres; phi = 0, 90, 180; v = 0.05, 0.2; d = 30.
 4b optional: the b5 gate and b5 collisions (d = 56, N = 270, L = 148.5).
 5  3D spot checks of L1 points: dip field eps = -0.1, dip removal s = 0.2, b5 field +0.1; T = 2000;
    compared with the stage-3 control (same source) and with the radial runs (post --radial).
 6  linear twins (dip stretch 0.1; dip collision phase 0, v = 0.05), and a 1.5x box check and a resolution
    check (dx = 0.367) of the dip phase-0 collision at each speed, to contact + 500.
"""

import argparse
import functools
import json
import multiprocessing as mp
import os
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = "outputs/session3/wpL3"
CKPT = "checkpoints/session3/wpL3"
DX, DT = 0.55, 0.04
G4 = 0.30
STARTS = {"dip": (4.0 / 3.0, 0.9065045072253043), "primary": (4.0 / 3.0, 0.909149280617113),
          "b5": (5.0, 0.975598235049607)}
PRIMARY_SHA = "6193e7b109b7369c"           # first 16 hex of the WP-E production source
SRC = {"dip": OUT + "/sources/src_dip_N160_L88.npz",
       "primary": "outputs/session2/wpE/" + "sou" + "rce/src_prod_N160_L88.npz",
       "b5": OUT + "/sources/src_b5_N216_L118.8.npz"}
SRC_GRID = {"dip": (160, 88.0), "primary": (160, 88.0), "b5": (216, 118.8)}
WINDOW = {"dip": [26.0, 30.0], "primary": [26.0, 30.0], "b5": [36.0, 42.0]}
SRC_WINDOWS = {"dip": [[14.0, 18.0], [18.0, 22.0], [22.0, 26.0], [26.0, 30.0]],
               "b5": [[30.0, 36.0], [36.0, 42.0], [40.0, 46.0]]}
T_SOURCE = 500.0
COLL = {"dip": dict(N=180, L=99.0, d=30.0), "primary": dict(N=180, L=99.0, d=30.0),
        "b5": dict(N=270, L=148.5, d=56.0)}
R_BALL = {"dip": 10.0, "primary": 10.0, "b5": 20.0}
HORIZON_AFTER_CONTACT = 1500.0
SPEEDS = (0.02, 0.2)                # per lump; closing speed 2 v
STRETCH = (0.05, 0.1, 0.2)
AMPS = (0.02, 0.05, 0.1, 0.15)      # a = 0.15: Delta N / N = a^2 = 2.25%, past the dip's 1.7% width in N
SEEDS = (1, 2, 3)
CROSS_MAX = 1e-4                    # t = 0 cross term of the collision data, checked in the gate check
D_SPONGE = {"dip": 15.1, "primary": 15.1, "b5": 29.1}   # Phi = 1e-3 Phi0 (the sponge guard)
PHASE_LABEL = ("relative phase of lump B (x > 0) minus lump A (x < 0) at the meeting point x = 0, exact there at "
               "every t by symmetry; at offset x in the overlap it is label + 2 omega gamma v x")
CHECK_AFTER_CONTACT = 500.0
RATE_N160 = 33.5 / 8 * 0.85        # steps/s per 2-thread job at N = 160 on c8g.4xlarge (Session 2), -15% diagnostics
USD_PER_H = 0.6358


@functools.lru_cache(maxsize=None)
def numbers(start):
    from swirl.family.wpl import start_numbers
    return start_numbers(start)


def profile_table(start, rel=1e-7):
    """Radial profile (physical units) on r = 0, 0.05, ... until Phi < rel Phi(0)."""
    from swirl.family.profile import profile_curve
    beta, omega, prof, nums = numbers(start)
    r = np.arange(0.0, 200.0, 0.05)
    P = profile_curve(prof, r) / np.sqrt(G4)
    k = int(np.argmax(P < rel * P[0]))
    return dict(r=r[:k + 1].tolist(), Phi=P[:k + 1].tolist(), omega=omega, half_width=nums["hw"])


def physics(start):
    beta = STARTS[start][0]
    return dict(m=1.0, g4=G4, g6=0.09 * beta)


# ---------------------------------------------------------------------------
# Job builders
# ---------------------------------------------------------------------------

def run_config(start, N, L, T, lumps, diag, dx=DX, shape=None, linear=False, r_c=None, it_every_step=False,
               group=None):
    nums = numbers(start)[3]
    ph = physics(start)
    if linear:
        ph.update(g4=0.0, g6=0.0)
    entry = dict(source=SRC[start], window=WINDOW[start], M=32, lumps=lumps)
    if abs(dx - DX) > 1e-12:
        entry["dx_target"] = dx
    ld = dict(centers=[lp["center"] for lp in lumps], R_ball=R_BALL[start], taper=2.0, every=5, count_every=25.0,
              omega_nominal=STARTS[start][1])
    ld.update(diag)
    if it_every_step:
        ld["it_every_step"] = True
    if group:
        ld["group"] = group
    cfg = dict(N=N, L=L, dt=DT * dx / DX, t_end=T, init={"wpl": entry}, absorber=True, center=[0.0, 0.0, 0.0],
               r_c=r_c or [3 * nums["hw"], nums["rN"]], sample_every=1.0, point_every=1, avg_window=20.0,
               checkpoint_every=50.0, threads=2, lumps=ld, **ph)
    if shape:
        cfg["shape"] = shape
    return cfg


def sponge_arrival(start, c, v):
    """Earliest time a lump moving at v after the collision has its tail (Phi = 1e-3 Phi0) at the sponge:
    the smaller of pass-through at constant v and a bounce at the contact time. Outcomes that need two lumps
    are read before it."""
    hw = numbers(start)[3]["hw"]
    reach = c["L"] / 2 - 12.0 - D_SPONGE[start]
    t_pass = (reach + c["d"] / 2) / v
    t_bounce = (c["d"] - 2 * hw) / (2 * v) + (reach - hw) / v
    return float(min(t_pass, t_bounce))


def guard_motion(j):
    """Velocity guard: a moving lump must keep its tail (Phi = 1e-3 Phi0) off the sponge for the whole run, unless
    the job records the sponge arrival time (collisions, which are read before it)."""
    if j["kind"] != "run" or j.get("t_sponge") is not None:
        return
    reach = j["L"] / 2 - 12.0 - D_SPONGE[j["start"]]
    for lp in j["cfg"]["init"]["wpl"]["lumps"]:
        x = np.asarray(lp["center"], float)
        x_end = x + np.array([lp.get("v", 0.0) * j["T"], 0.0, 0.0])
        if max(np.linalg.norm(x), np.linalg.norm(x_end)) > reach:
            raise ValueError("%s: a lump reaches the sponge during the run" % j["name"])


B5_D_KEEP = ("control", "q0.2", "r0.1_s1", "r0.15_s1")
DIP_3S_KEEP = ("r0.1_s2", "r0.1_s3", "r0.15_s2", "r0.15_s3")


def keep(j):
    """The budget trim (coordinator, Session 3): no b5 collisions (4b); b5 (d) only control, stretch 0.2,
    random 0.1 and 0.15 (seed 1); seeds 2 and 3 only at the dip for a = 0.1 and 0.15; the resolution check
    only for the v = 0.2 collision."""
    if j["stage"] == "4b":
        return False
    if j["name"].startswith("d_b5_") and j.get("variant") not in B5_D_KEEP:
        return False
    if j["stage"] == "3s" and not (j["start"] == "dip" and j.get("variant") in DIP_3S_KEEP):
        return False
    return j["name"] != "res_coll_dip_p0_v%g" % SPEEDS[0]


def roles_of(j):
    """Instance roles. A: dip gates, gate check, collisions and their checks. B: primary (shipped source).
    C: dip (d), seeds, spot checks and the (d) twin. D: b5. The dip source is built on A and on C (each from the
    same config; the hashes are recorded in each source result)."""
    if j["name"] == "src_dip":
        return ["A", "C"]
    if j["start"] == "primary":
        return ["B"]
    if j["start"] == "b5":
        return ["D"]
    if j["name"].startswith(("gate", "coll_", "box_coll", "res_coll", "twin_coll")):
        return ["A"]
    return ["C"]


# Slots per instance: WP-L workers, and the slots left to the P3 jobs (run by their own driver) with the job-hours
# planned on each P3 slot in sequence: 2 candidates (~12 h), 3 single-lump runs (~7 h), control 3.3 h, twin 1.7 h.
WPL_WORKERS = {"A": 6, "B": 7, "C": 8, "D": 6}
P3_SLOTS = {"A": [[12.0], [12.0]], "B": [[7.0]], "C": [], "D": [[7.0, 3.3], [7.0, 1.7]]}


def pack(J, slots=8, extra=None):
    """Longest-ready-job-first schedule of one instance (dependencies within the instance); extra: job-hours of
    independent jobs (the P3 allowance). Returns the wall hours and the free slot-hours left before the wall."""
    names = {j["name"] for j in J}
    jobs_ = [dict(name=j["name"], h=j["job_hours"], deps=[d for d in j["deps"] if d in names]) for j in J]
    jobs_ += [dict(name="P3_%d" % i, h=h, deps=[]) for i, h in enumerate(extra or [])]
    end, free_at, pending = {}, [0.0] * slots, list(jobs_)
    while pending:
        t = min(free_at)
        ready = [x for x in pending if all(d in end and end[d] <= t + 1e-12 for d in x["deps"])]
        if not ready:
            later = [end[d] for x in pending for d in x["deps"] if d in end and end[d] > t]
            i = int(np.argmin(free_at))
            free_at[i] = min(later) if later else t + 1e-3
            continue
        x = max(ready, key=lambda y: y["h"])
        i = int(np.argmin(free_at))
        end[x["name"]] = free_at[i] + x["h"]
        free_at[i] = end[x["name"]]
        pending.remove(x)
    wall = max(end.values())
    return wall, wall * slots - sum(x["h"] for x in jobs_)


def contact_time(start, v):
    hw = numbers(start)[3]["hw"]
    return (COLL[start]["d"] - 2 * hw) / (2 * v)


def jobs():
    J = []
    add = lambda **k: J.append(k)
    # 1. sources
    for st in ("dip", "b5"):
        N, L = SRC_GRID[st]
        prof = profile_table(st)
        nums = numbers(st)[3]
        cfg = dict(N=N, L=L, dt=DT, init={"radial": [dict(r=prof["r"], Phi=prof["Phi"], center=[0.0, 0.0, 0.0],
                                                         phase=0.0, omega=prof["omega"])], "gaussians": []},
                   absorber=True, kfilter=None, r_c=[3 * nums["hw"], nums["rN"]], sample_every=1.0, avg_window=20.0,
                   checkpoint_every=50.0, threads=2, **physics(st))
        add(name="src_" + st, stage="1", kind="source", start=st, deps=[], cfg=cfg, t_s=T_SOURCE,
            windows=SRC_WINDOWS[st], out=SRC[st], N=N, L=L, dt=DT, T=T_SOURCE)
    add(name="src_primary", stage="1", kind="check", start="primary", deps=[], out=SRC["primary"], N=160, L=88.0,
        dt=DT, T=0.0)
    dep = {st: ["src_" + st] for st in STARTS}
    # 2 / 4b. gates: production source and grid, at rest, and at each collision speed at theta = 0 and 90
    for st in ("dip", "primary", "b5"):
        N, L = SRC_GRID[st]
        runs = [(0.0, 0.0, 0.0, 240.0)] + [(v, th, -v * T / 2, T) for v, T in zip(SPEEDS, (240.0, 120.0))
                                            for th in (0.0, 90.0)]
        names = []
        for v, th, x0, T in runs:
            lumps = [dict(center=[x0, 0.0, 0.0], phase=float(np.radians(th)), v=v)]
            cfg = run_config(st, N, L, T, lumps, {}, shape=dict(every=0.4, R=3 * numbers(st)[3]["hw"]))
            name = "gate_%s_v%g_th%g" % (st, v, th)
            names.append(name)
            add(name=name, stage="4b" if st == "b5" else "2", kind="run", start=st, deps=dep[st], cfg=cfg, v=v,
                theta=th, N=N, L=L, dt=cfg["dt"], T=T)
        c = COLL[st]
        cross = []
        for v in SPEEDS:
            for ph in (0.0, 90.0, 180.0):
                lumps = [dict(center=[-c["d"] / 2, 0.0, 0.0], phase=0.0, v=v),
                         dict(center=[c["d"] / 2, 0.0, 0.0], phase=float(np.radians(ph)), v=-v)]
                cross.append(dict(name="coll_%s_p%g_v%g" % (st, ph, v),
                                  cfg=dict(N=c["N"], L=c["L"], init={"wpl": dict(source=SRC[st], window=WINDOW[st],
                                                                               M=32, lumps=lumps)}, **physics(st))))
        add(name="gatecheck_" + st, stage="4b" if st == "b5" else "2", kind="gatecheck", start=st,
            source=SRC[st], window=WINDOW[st], deps=names, N=N, L=L, dt=DT, T=0.0, cross=cross)
    # 3 / 3s. (d) distortions (seeds 2 and 3 of the random multiplier in stage 3s)
    for st in ("dip", "primary", "b5"):
        N, L = SRC_GRID[st]
        hw = numbers(st)[3]["hw"]
        variants = [("control", {}, "3")] + [("q%g" % e, dict(quad_eps=e), "3") for e in STRETCH] + \
                   [("r%g_s%d" % (a, sd), dict(random=dict(amp=a, ell=round(hw, 1), seed=sd)), "3" if sd == 1 else "3s")
                    for a in AMPS for sd in SEEDS]
        for tag, extra, stage in variants:
            lumps = [dict(center=[0.0, 0.0, 0.0], phase=0.0, v=0.0, **extra)]
            cfg = run_config(st, N, L, 2000.0, lumps, {}, shape=dict(every=1.0, R=3 * hw), it_every_step=True)
            add(name="d_%s_%s" % (st, tag), stage=stage, kind="run", start=st, deps=dep[st], cfg=cfg, N=N, L=L,
                dt=cfg["dt"], T=2000.0, variant=tag)
    # 4 / 4b. collisions
    for st in ("dip", "primary", "b5"):
        c = COLL[st]
        hw = numbers(st)[3]["hw"]
        for v in SPEEDS:
            T = round(contact_time(st, v) + HORIZON_AFTER_CONTACT, -1)
            for ph in (0.0, 90.0, 180.0):
                lumps = [dict(center=[-c["d"] / 2, 0.0, 0.0], phase=0.0, v=v),
                         dict(center=[c["d"] / 2, 0.0, 0.0], phase=float(np.radians(ph)), v=-v)]
                cfg = run_config(st, c["N"], c["L"], T, lumps, {}, group=dict(R=3 * hw + 2.0, axis=[1.0, 0.0, 0.0]))
                add(name="coll_%s_p%g_v%g" % (st, ph, v), stage="4b" if st == "b5" else "4", kind="run", start=st,
                    deps=dep[st] + ["gatecheck_" + st], cfg=cfg, v=v, phase=ph, d=c["d"], N=c["N"], L=c["L"],
                    dt=cfg["dt"], T=T, t_contact=contact_time(st, v), t_sponge=sponge_arrival(st, c, v),
                    phase_label=PHASE_LABEL)
    # 5. spot checks of L1 radial points
    for st, kick, radial in (("dip", dict(kind="field", eps=-0.1), "dip_field_e-0.1_T2000_s1"),
                             ("dip", dict(kind="removal", s=0.2), "dip_rem_s0.2_T2000_s1"),
                             ("b5", dict(kind="field", eps=0.1), "b5_field_e+0.1_T2000_s1")):
        N, L = SRC_GRID[st]
        nums = numbers(st)[3]
        lumps = [dict(center=[0.0, 0.0, 0.0], phase=0.0, v=0.0, kick=dict(kick, hw=nums["hw"]))]
        cfg = run_config(st, N, L, 2000.0, lumps, dict(R_ball=nums["rN"]), it_every_step=True,
                         shape=dict(every=1.0, R=3 * nums["hw"]))
        tag = "%s%+g" % (kick["kind"], kick.get("eps", kick.get("s")))
        add(name="spot_%s_%s" % (st, tag), stage="5", kind="run", start=st, deps=dep[st], cfg=cfg, N=N, L=L,
            dt=cfg["dt"], T=2000.0, radial=radial, control="d_%s_control" % st)
    # 6. twins, box and resolution checks
    jd = {j["name"]: j for j in J}
    for base, T in (("d_dip_q0.1", 500.0), ("coll_dip_p0_v%g" % SPEEDS[0], 500.0)):
        b = jd[base]
        lumps = b["cfg"]["init"]["wpl"]["lumps"]
        cfg = run_config("dip", b["N"], b["L"], T, lumps, {}, linear=True,
                         group=b["cfg"]["lumps"].get("group"), shape=b["cfg"].get("shape"))
        add(name="twin_" + base, stage="6", kind="run", start="dip", deps=b["deps"], cfg=cfg, N=b["N"], L=b["L"],
            dt=cfg["dt"], T=T, base=base)
    for v in SPEEDS:
        b = jd["coll_dip_p0_v%g" % v]
        T = round(contact_time("dip", v) + CHECK_AFTER_CONTACT, -1)
        lumps = b["cfg"]["init"]["wpl"]["lumps"]
        grp = b["cfg"]["lumps"]["group"]
        cfg = run_config("dip", 270, 148.5, T, lumps, {}, group=grp)
        add(name="box_coll_dip_p0_v%g" % v, stage="6", kind="run", start="dip", deps=b["deps"], cfg=cfg, N=270,
            L=148.5, dt=cfg["dt"], T=T, base=b["name"], d=b["d"], v=v, phase=0.0, t_contact=b["t_contact"],
            t_sponge=sponge_arrival("dip", dict(L=148.5, d=b["d"]), v), phase_label=PHASE_LABEL)
        cfg = run_config("dip", 270, 99.0, T, lumps, {}, dx=99.0 / 270, group=grp)
        add(name="res_coll_dip_p0_v%g" % v, stage="6", kind="run", start="dip", deps=b["deps"], cfg=cfg, N=270,
            L=99.0, dt=cfg["dt"], T=T, base=b["name"], d=b["d"], v=v, phase=0.0, t_contact=b["t_contact"],
            t_sponge=b["t_sponge"], phase_label=PHASE_LABEL)
    b = jd["d_b5_r0.1_s1"]
    cfg = run_config("b5", 324, 178.2, 500.0, b["cfg"]["init"]["wpl"]["lumps"], {}, shape=b["cfg"]["shape"],
                     it_every_step=True)
    add(name="box_d_b5_r0.1_s1", stage="6", kind="run", start="b5", deps=b["deps"], cfg=cfg, N=324, L=178.2,
        dt=cfg["dt"], T=500.0, base=b["name"])
    for j in J:
        guard_motion(j)
    J = [j for j in J if keep(j)]
    for j in J:
        j["roles"] = roles_of(j)
    for j in J:
        j.update(cost(j))
    return J


def rate(N):
    return RATE_N160 * (160 ** 3 * np.log(160)) / (N ** 3 * np.log(N))


def cost(j):
    if j["kind"] in ("check", "gatecheck"):
        return dict(steps=0, job_hours=0.02 if j["kind"] == "gatecheck" else 0.0)
    steps = int(round(j["T"] / j["dt"]))
    if j["kind"] == "source":
        steps += 1200                       # stage 2: one period of steps plus the crop measurement
    return dict(steps=steps, job_hours=float(steps / rate(j["N"]) / 3600.0))


def schedule(J, slots, stages=None, starts=None):
    """Greedy list schedule (dependencies, in list order) on `slots` job slots: the wall hours."""
    sel = [j for j in J if (stages is None or j["stage"] in stages) and (starts is None or j["start"] in starts)]
    names = {j["name"] for j in sel}
    end, free = {}, [0.0] * slots
    for j in sel:
        ready = max([end[d] for d in j["deps"] if d in names] + [0.0])
        i = int(np.argmin(free))
        start = max(free[i], ready)
        end[j["name"]] = start + j["job_hours"]
        free[i] = end[j["name"]]
    return max(end.values()) if end else 0.0


def cost_table(J):
    out = dict(rate_model="steps/s per 2-thread job = %.3f x (160^3 ln 160)/(N^3 ln N) (33.5 steps/s for 8 N = 160 "
                          "jobs on c8g.4xlarge, Session 2, less 15%% for diagnostics)" % RATE_N160,
               stages={s_: dict(jobs=len([j for j in J if j["stage"] == s_]),
                                job_hours=sum(j["job_hours"] for j in J if j["stage"] == s_))
                       for s_ in sorted({j["stage"] for j in J})},
               job_hours_wpl=sum(j["job_hours"] for j in J), wpl_workers=WPL_WORKERS, P3_slots=P3_SLOTS,
               job_hours_P3=sum(sum(c) for v in P3_SLOTS.values() for c in v), roles={})
    for r in ("A", "B", "C", "D"):
        Jr = [j for j in J if r in j["roles"]]
        w0, _ = pack(Jr, slots=WPL_WORKERS[r])
        w = max([w0] + [sum(c) for c in P3_SLOTS[r]])
        out["roles"][r] = dict(jobs=[j["name"] for j in Jr], job_hours=sum(j["job_hours"] for j in Jr),
                               wpl_workers=WPL_WORKERS[r], wall_h_wpl=w0, P3_slots=P3_SLOTS[r], wall_h=w,
                               usd=w * USD_PER_H)
    out["usd_total_with_P3"] = sum(v["usd"] for v in out["roles"].values())
    out["instance_hours_total"] = sum(v["wall_h"] for v in out["roles"].values())
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def result_path(name):
    return _abs(os.path.join(OUT, "results", name + ".json"))


def _atomic_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def run_job(j, out=None, ckpt=None):
    """Run one job; returns (name, status, seconds). out, ckpt: the parent's OUT and CKPT (spawned workers
    re-import this module)."""
    global OUT, CKPT
    OUT, CKPT = out or OUT, ckpt or CKPT
    t0 = time.time()
    try:
        if j["kind"] == "check":
            from swirl.engine.snapshot import load_source
            phi, pi, meta = load_source(_abs(j["out"]))
            if not meta["sha256"].startswith(PRIMARY_SHA):
                raise ValueError("primary source hash %s, expected %s..." % (meta["sha256"], PRIMARY_SHA))
            _atomic_json(dict(job=j, meta={k: meta[k] for k in ("omega", "omega_drift", "sha256", "dx", "t_s")}),
                         result_path(j["name"]))
        elif j["kind"] == "source":
            from swirl.engine.snapshot import make_source
            out = _abs(j["out"])
            name = os.path.basename(out)[:-4]
            meta = make_source(name, j["cfg"], j["t_s"], os.path.dirname(out), windows=j["windows"],
                               checkpoint_dir=_abs(os.path.join(CKPT, j["name"])), verbose=True)
            _atomic_json(dict(job={k: v for k, v in j.items() if k != "cfg"},
                              meta={k: v for k, v in meta.items() if k not in ("profile", "params")}),
                         result_path(j["name"]))
        elif j["kind"] == "gatecheck":
            g = gate_summary(j["start"], j.get("source"), j.get("window"))
            ok, why = gate_verdict(g)
            cross = {c["name"]: cross_term(c) for c in j.get("cross", [])}
            bad = [n for n, x in cross.items() if not abs(x) < CROSS_MAX]
            if ok and bad:
                ok, why = False, ["t = 0 cross term above %.0e: %s" % (CROSS_MAX, {n: cross[n] for n in bad})]
            _atomic_json(dict(job=j, gate=g, cross_terms=cross, ok=ok, why=why),
                         result_path(j["name"]) + (".pass" if ok else ".fail"))
            if not ok:
                raise RuntimeError("gate failed: %s" % why)
            os.replace(result_path(j["name"]) + ".pass", result_path(j["name"]))
        else:
            from swirl.engine.run import run
            os.makedirs(os.path.dirname(result_path(j["name"])), exist_ok=True)
            res = run(j["cfg"], out_path=result_path(j["name"]), checkpoint_dir=_abs(os.path.join(CKPT, j["name"])),
                      verbose=True)
            if not res["completed"]:
                raise RuntimeError("run did not complete")
        return j["name"], "done", time.time() - t0
    except Exception as e:                  # recorded; dependents are skipped
        return j["name"], "failed: %r" % (e,), time.time() - t0


def _init_worker(threads):
    """Pool initializer: the worker's numba thread count, set in the worker before it imports numba (this
    module does not import numba at top level). The parent's environment is left alone: changing
    NUMBA_NUM_THREADS in a process whose numba threads are running makes its next compile fail."""
    os.environ["NUMBA_NUM_THREADS"] = str(threads)


def run_jobs(jobs_file, workers=8, threads=2, stages=None, starts=None, role=None, poll=10.0):
    J = json.load(open(jobs_file))
    for j in J:
        if j["kind"] == "run":
            j["cfg"]["threads"] = threads
        elif j["kind"] == "source":
            j["cfg"]["threads"] = threads
    sel = [j for j in J if (stages is None or j["stage"] in stages) and (starts is None or j["start"] in starts)
           and (role is None or role in j.get("roles", [role]))]
    sel.sort(key=lambda j: -j.get("job_hours", 0.0))            # longest ready job first
    names = {j["name"] for j in sel}
    done = {j["name"] for j in J if os.path.exists(result_path(j["name"]))}
    failed, pending, running = set(), [j for j in sel if j["name"] not in done], {}
    log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
    log("jobs: %d selected, %d already done, %d to run" % (len(sel), len(sel) - len(pending), len(pending)))
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1, initializer=_init_worker,
                                      initargs=(threads,)) as pool:
        while pending or running:
            for j in list(pending):
                deps = [d for d in j["deps"]]
                if any(d in failed for d in deps) or any(d not in names and d not in done for d in deps):
                    if any(d in failed for d in deps):
                        pending.remove(j)
                        failed.add(j["name"])
                        log("skip %s (dependency failed)" % j["name"])
                    continue
                if all(d in done for d in deps) and len(running) < workers:
                    running[j["name"]] = pool.apply_async(run_job, (j, OUT, CKPT))
                    pending.remove(j)
                    log("start %s (%.1f job-h estimated)" % (j["name"], j["job_hours"]))
            for name, ar in list(running.items()):
                if ar.ready():
                    n, status, sec = ar.get()
                    del running[name]
                    (done if status == "done" else failed).add(name)
                    log("%s %s after %.0f s" % (name, status, sec))
            if pending and not running and not any(all(d in done for d in j["deps"]) for j in pending):
                log("blocked: %s" % [j["name"] for j in pending])
                break
            time.sleep(poll)
    log("finished: %d done, %d failed" % (len(done & names), len(failed)))
    return sorted(failed)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _load(name):
    p = result_path(name)
    return json.load(open(p)) if os.path.exists(p) else None


def _lumps_series(res):
    raw = res["lumps"]["raw"]["series"]
    K = len(res["lumps"]["raw"]["centers"])
    t = np.asarray(raw["t"])
    return t, np.asarray(raw["X"]).reshape(t.size, K, 3), np.asarray(raw["E"]).reshape(t.size, K)


def gate_summary(st, source=None, window=None):
    from swirl.engine import boost
    r0 = _load("gate_%s_v0_th0" % st)
    if r0 is None:
        return None
    H = boost.harmonics(_abs(source or SRC[st]), window or WINDOW[st], threads=2)
    ref = boost.rest_reference(H, **physics(st))

    def track(res):
        s = res["series"]
        tl, X, E = _lumps_series(res)
        sh = s["shape"]
        ts = np.array([x["t"] for x in sh])
        Md = np.array([np.diag(np.array(_m(x))) for x in sh])
        Es = np.array([x["E_sphere"] for x in sh])
        pt = res["lumps"]["raw"]["point"]
        return dict(series=dict(t=ts.tolist(), E_total=np.interp(ts, s["t"], s["E_total"]).tolist(),
                                absorbed=np.interp(ts, s["t"], s["absorbed"]).tolist(),
                                E_ball=np.interp(ts, tl, E[:, 0]).tolist(),
                                X=np.stack([np.interp(ts, tl, X[:, 0, a]) for a in range(3)], 1).tolist(),
                                I_diag=(Md / Es[:, None]).tolist()),
                    point=dict(t=pt["t"], phi=[p[0] for p in pt["phi"]]), T=res["params"]["t_end"],
                    wall_s=res["wall_seconds"])

    t0 = track(r0)
    out = dict(rest_reference=ref, results={})
    for v in SPEEDS:
        for th in (0.0, 90.0):
            r = _load("gate_%s_v%g_th%g" % (st, v, th))
            if r is not None:
                out["results"]["v%g_th%g" % (v, th)] = boost.gate_analysis(track(r), t0, v, ref["E0"])
    return out


GATE_TARGETS = dict(speed_rel_err=1e-2, E_ratio_vs_E0ref_err=1e-4, omega_ratio_err=1e-4, E_ball_drift_minus_rest=1e-3)


def gate_verdict(g):
    """Pass when every measure is within 3x its target (CLAUDE.md), at both speeds."""
    if not g or len(g["results"]) < 2 * len(SPEEDS):
        return False, "gate results missing"
    bad = ["v=%s %s=%.3g (target %.0e)" % (v, k, r[k], t) for v, r in g["results"].items()
           for k, t in GATE_TARGETS.items() if not abs(r[k]) < 3 * t]
    return (not bad), bad


def _m(x):
    """Full second-moment matrix from a shape record (eigenvalues only when M is not stored)."""
    return x["M"] if "M" in x else np.diag(x["eig_M"])


def _smooth(y, n):
    return np.convolve(y, np.ones(n) / n, mode="valid") if len(y) >= n else np.asarray(y)


def run_summary(res, ctrl=None, horizon=2000.0):
    """Lump energy in 3 hw (frac against the control's t = 0 value), whole-period N in the ball, omega, shape."""
    from swirl.engine.lumps import summarize
    s = res["series"]
    t = np.asarray(s["t"])
    E3 = np.asarray([x[0] for x in s["E_loc"]])
    E3_0 = np.asarray([x[0] for x in ctrl["series"]["E_loc"]])[0] if ctrl else E3[0]
    st = res["lumps"]["raw"]
    st = dict(st, series=st["series"], point=st["point"], centers=st["centers"])
    avg = summarize(st, list(t[t >= 30]), 20.0)["lumps"][0]
    out = dict(t_end=float(t[-1]), frac_end=float(E3[-1] / E3_0), frac_min=float(np.min(E3 / E3_0)),
               first_below_half=float(t[np.argmax(E3 < 0.5 * E3_0)]) if np.any(E3 < 0.5 * E3_0) else None,
               E_plus_absorbed_rel_dev=res["summary"]["max_rel_dev_E_plus_absorbed"],
               nonlinear_share_0=res["summary"]["nonlinear_share_0"],
               N_windows=[(float(a), b) for a, b in zip(t[t >= 30], avg["N"]) if b is not None][::50],
               omega_windows=[(float(a), b) for a, b in zip(t[t >= 30], avg["omega"]) if b is not None][::50])
    if "shape" in s:
        sh = s["shape"]
        a2 = np.array([x["a2"] for x in sh])
        a3 = np.array([x["a3"] for x in sh])
        n = 21                               # about three periods at one record per time unit
        a2s, a3s = _smooth(a2, n), _smooth(a3, n)
        out["shape"] = dict(a2_first=float(a2s[0]), a2_max=float(a2s.max()), a2_end=float(a2s[-1]),
                            a3_first=float(a3s[0]), a3_max=float(a3s.max()), a3_end=float(a3s[-1]),
                            centroid_disp_max=float(max(np.linalg.norm(x["centroid"]) for x in sh)))
    return out


def collision_summary(res, j):
    """Outcome from the debris-floored lump count and the tracked centroids: merge (one distinct lump in the
    last three count windows), bounce or pass-through (two lumps; tracker 0, which started at x < 0, ends at
    x < 0 or x > 0), fragmented (three or more), dispersed (none). Remnant energy (group sphere) and omega
    (zero crossings of phi at tracker 0) over the last 300 time units. Trackers can swap when the balls
    overlap; the classification records the minimum tracker distance so a near-contact swap can be seen."""
    from swirl.engine.run import zero_crossings
    counts = res["lumps"]["counts"]
    tc = np.array([c["t_mid"] for c in counts])
    nd = np.array([c["count_distinct"] for c in counts])
    tl, X, E = _lumps_series(res)
    dist = np.linalg.norm(X[:, 0] - X[:, 1], axis=1)
    hw = numbers(j["start"])[3]["hw"]
    contact = None
    w = np.where((nd < 2) | (np.interp(tc, tl, dist) < 2 * hw))[0]
    if w.size:
        contact = float(tc[w[0]])
    def classify(k_last):
        last = nd[:k_last][-3:]
        if last.size == 0:
            return "no window", last
        i = int(np.searchsorted(tl, tc[k_last - 1], side="right")) - 1
        if np.all(last == 1):
            return "merge", last
        if np.all(last == 2):
            return ("bounce" if X[i, 0, 0] < 0 else "pass-through"), last
        if np.all(last == 0):
            return "dispersed", last
        if np.all(last >= 3):
            return "fragmented", last
        return "unsettled (counts %s)" % last.tolist(), last

    outcome_end, last = classify(nd.size)
    ts_ = j.get("t_sponge")
    k_sp = int(np.searchsorted(tc, ts_, side="right")) if ts_ else nd.size
    outcome_sp, last_sp = classify(max(k_sp, 1))
    outcome = outcome_sp if (ts_ and outcome_sp != "merge") else outcome_end
    T = res["params"]["t_end"]
    sel = tl >= T - 300
    G = res["lumps"]["raw"]["series"].get("E_group")
    pt = res["lumps"]["raw"]["point"]
    tp = np.asarray(pt["t"])
    p0 = np.asarray([p[0] for p in pt["phi"]])
    z = zero_crossings(tp[tp >= T - 300], p0[tp >= T - 300])
    om = float(np.pi / np.polyfit(np.arange(z.size), z, 1)[0]) if z.size > 4 else None
    E0 = res["series"]["E_total"][0]
    return dict(outcome=outcome, outcome_end=outcome_end, outcome_before_sponge=outcome_sp,
                t_sponge_estimate=ts_, phase_label=j.get("phase_label"), phase_deg=j.get("phase"),
                t_contact_measured=contact, t_contact_estimate=j["t_contact"],
                count_distinct_last=last.tolist(), count_distinct_before_sponge=last_sp.tolist(), min_tracker_distance=float(dist.min()),
                remnant_E_group_over_E0=(float(np.mean(np.asarray(G)[sel])) / E0) if G else None,
                lump_E_over_E0_end=(E[-1] / E0).tolist(), omega_tracker0_end=om,
                speeds_end=[float(np.polyfit(tl[sel], X[sel, k, 0], 1)[0]) for k in range(X.shape[1])],
                E_plus_absorbed_rel_dev=res["summary"]["max_rel_dev_E_plus_absorbed"],
                absorbed_over_E0=res["summary"]["absorbed_end"] / E0)


def cross_term(j):
    """E(two lumps) - E(A) - E(B) over E(A) + E(B) at t = 0 (collision data)."""
    from swirl.engine.evolver import Evolver
    from swirl.engine.initial import build_initial
    c = j["cfg"]
    ev = Evolver(c["N"], c["L"], c["m"], c["g4"], c["g6"], threads=2)
    ent = c["init"]["wpl"]
    pa, va = build_initial(ev.coords, ev.L, {"wpl": dict(ent, lumps=ent["lumps"][:1])})
    pb, vb = build_initial(ev.coords, ev.L, {"wpl": dict(ent, lumps=ent["lumps"][1:])})
    E = lambda p, q: ev.energy_parts(p, q)["E_total"]
    Ea, Eb, Et = E(pa, va), E(pb, vb), E(pa + pb, va + vb)
    return float((Et - Ea - Eb) / (Ea + Eb))


def radial_compare(res3, ctrl3, radial):
    """3D spot check against the L1 radial run: frac (E(r < 3 hw) over its pre-disturbance value) at matched
    times after t_d."""
    t3 = np.asarray(res3["series"]["t"])
    f3 = np.asarray([x[0] for x in res3["series"]["E_loc"]]) / ctrl3["series"]["E_loc"][0][0]
    tr = np.asarray(radial["series"]["t"]) - radial["t_d"]
    fr = np.asarray(radial["series"]["frac"])
    sel = (t3 >= 0) & (t3 <= tr[-1])
    d = np.abs(f3[sel] - np.interp(t3[sel], tr, fr))
    return dict(max_abs_frac_diff=float(d.max()), t_of_max=float(t3[sel][np.argmax(d)]),
                frac_end_3d=float(f3[sel][-1]), frac_end_radial=float(np.interp(t3[sel][-1], tr, fr)))


def post(jobs_file, radial_dir=None):
    J = json.load(open(jobs_file))
    out = dict(sources={}, gates={}, d={}, collisions={}, spot={}, checks={})
    for j in J:
        r = _load(j["name"])
        if r is None:
            continue
        if j["stage"] == "1":
            out["sources"][j["name"]] = r["meta"]
        elif j["name"].startswith("d_"):
            c = _load("d_%s_control" % j["start"])
            out["d"][j["name"]] = run_summary(r, c)
        elif j["name"].startswith("coll_"):
            out["collisions"][j["name"]] = dict(collision_summary(r, j), cross_term_t0=cross_term(j))
        elif j["name"].startswith("spot_"):
            c = _load(j["control"]) if j.get("control") else None
            s = run_summary(r, c)
            rp = os.path.join(radial_dir, j["radial"] + ".json") if (radial_dir and j.get("radial")) else None
            if c is not None and rp and os.path.exists(rp):
                s["radial"] = radial_compare(r, c, json.load(open(rp)))
            out["spot"][j["name"]] = s
        elif j["name"].startswith(("box_", "res_")):
            out["checks"][j["name"]] = (collision_summary(r, j) if "coll" in j["name"]
                                        else run_summary(r, _load(j["base"])))
        elif j["name"].startswith("twin_"):
            out["checks"][j["name"]] = dict(frac_end=float(r["series"]["E_loc"][-1][0] / r["series"]["E_loc"][0][0]))
    for st in ("dip", "primary", "b5"):
        g = gate_summary(st)
        if g:
            out["gates"][st] = g
    _atomic_json(out, _abs(os.path.join(OUT, "summary.json")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["jobs", "run", "post"])
    ap.add_argument("--jobs", default=os.path.join(ROOT, OUT, "jobs.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--stages", default=None)
    ap.add_argument("--starts", default=None)
    ap.add_argument("--role", default=None)
    ap.add_argument("--radial", default=None)
    a = ap.parse_args()
    if a.cmd == "jobs":
        J = jobs()
        path = a.out or a.jobs
        _atomic_json(J, path)
        ct = cost_table(J)
        _atomic_json(dict(cost=ct, jobs=[{k: j[k] for k in ("name", "stage", "N", "L", "dt", "T", "steps", "job_hours")}
                                         for j in J]), os.path.join(os.path.dirname(path), "cost.json"))
        print(json.dumps(ct, indent=1))
    elif a.cmd == "run":
        failed = run_jobs(a.jobs, a.workers, a.threads, a.stages.split(",") if a.stages else None,
                          a.starts.split(",") if a.starts else None, a.role)
        raise SystemExit(1 if failed else 0)
    else:
        post(a.jobs, a.radial)


if __name__ == "__main__":
    main()
