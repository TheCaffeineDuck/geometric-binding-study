"""Test A pair-run driver (SESSION_04_PLAN A1): sources, run list, per-r boxes, t = 0 checks, pool.

Lump types (snapshot sources; one settled source per type, hash-checked at every load):
    primary  beta 4/3, omega 0.909149 (Session 2 WP-E production source, reused)
    dip, big beta 4/3 unequal pair (outputs/session4/testA/unequal_choice.json), sources built here
    b5       beta 5, omega 0.975598 (built only if the b5 cost gate passes)

Geometry: lump 1 at (-r/2, 0, 0), lump 2 at (+r/2, 0, 0); phases theta_{1,2} = -/+ dtheta/2 (WPC_DERIVATION
section 7). Box: cubic, dx = 0.55, dt = 0.04, default spherical sponge (width 12). Per lump the distance
centre-to-face L/2 - r/2 must be at least the largest of
    3 hw + 12                    (plan A1: 3 hw plus the sponge width from the faces),
    12 + D_sponge                (engine guard: Phi(r_in - |c|) <= 1e-3 Phi0),
    r_face_min(window)           (snapshot face guard: windowed tail <= 1e-6 Phi0 beyond the face),
and N is the smallest size in NICE above L_min/dx. Every run is a full engine run (swirl.engine.run) with
the lumps block (per-lump balls, per-step ball It, per-step half-space It at x = 0, P_half at x = 0).

Controls (plan A1): isolated singles in every box used, at the lump-1 positions of the smallest and largest
r of that box (their difference is the "single against its own shifted copy" floor; V_N interpolates
between them); linear twins of the in-phase primary pairs at r = 12 and 24; box check 1.5x at r = 24;
resolution check dx = 0.40 at r = 16 (source resampled); t = 0 energy of every pair against an
independent quadrature (t0_check).

Early stop: a pair run stops (completed = False, stopped_early = True) once t >= 60 and the
period-smoothed kappa |d - r| exceeds 0.13 (the analysis window ends at 0.10). Singles run to T.

Pool: spawn context, at most 4 workers, maxtasksperchild 1, NUMBA_NUM_THREADS set in the worker
initializer; this module imports no numba at top level and the parent never imports the engine.

CLI (from the worktree root, PYTHON = <SWIRL>/.venv/bin/python):
    $PYTHON -m swirl.session4.testA.driver jobs   [--b5]       write outputs/session4/testA/jobs.json
    $PYTHON -m swirl.session4.testA.driver cost                print the cost table (measured rate)
    $PYTHON -m swirl.session4.testA.driver run --workers 4 [--only NAME ...] [--group G ...]
"""
import argparse
import functools
import json
import multiprocessing as mp
import os
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MAIN = os.path.abspath(os.path.join(ROOT, "..", "..", "..")) if "/.claude/worktrees/" in ROOT else ROOT
OUT = "outputs/session4/testA"
CKPT = "checkpoints/session4/testA"
DX, DT, G4 = 0.55, 0.04, 0.30
T_MAX = 400.0
SPONGE_W = 12.0
NICE = [96, 100, 108, 112, 120, 128, 144, 160, 180, 192, 200, 216, 240, 256, 270, 288]
PRIMARY_SRC_REL = "outputs/session2/wpE/source/src_prod_N160_L88.npz"
PRIMARY_SHA = "6193e7b109b7369cfc087b743bff24c17e51efd0193885d69aea170f916a8c72"

TYPES = {
    "primary": dict(beta=4.0 / 3.0, omega=0.909149280617113, window=[26.0, 30.0], R_ball=10.0, src_grid=(160, 88.0),
                    t_s=300.0, source=None, sha256=PRIMARY_SHA),
    "dip": dict(beta=4.0 / 3.0, omega=0.9065045072253043, window=[26.0, 30.0], R_ball=10.0, src_grid=(160, 88.0),
                t_s=300.0, source=OUT + "/sources/src_dip_N160_L88.npz"),
    "big": dict(beta=4.0 / 3.0, omega=0.857370255463254, window=[26.0, 30.0], R_ball=15.0, src_grid=(160, 88.0),
                t_s=300.0, source=OUT + "/sources/src_big_N160_L88.npz"),
    "b5": dict(beta=5.0, omega=0.975598235049607, window=[36.0, 42.0], R_ball=20.0, src_grid=(216, 118.8),
               t_s=500.0, source=OUT + "/sources/src_b5_N216_L118.8.npz"),
}
SRC_WINDOWS = {"dip": [[22.0, 26.0], [26.0, 30.0]], "big": [[22.0, 26.0], [26.0, 30.0]],
               "b5": [[30.0, 36.0], [36.0, 42.0]]}
R_PRIMARY = [7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40]
R_UNEQUAL = [r for r in R_PRIMARY if r >= 10]
R_B5 = [24, 30, 36, 42, 48]
DTHETA = (0, 180)


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def primary_source():
    for base in (ROOT, MAIN):
        p = os.path.join(base, PRIMARY_SRC_REL)
        if os.path.exists(p):
            return p
    return os.path.join(MAIN, PRIMARY_SRC_REL)


def source_path(t):
    return primary_source() if t == "primary" else _abs(TYPES[t]["source"])


def atomic_json(obj, path):
    path = _abs(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


# ------------------------------------------------------------------------------------------- lump numbers
@functools.lru_cache(maxsize=None)
def numbers(t):
    """Profile numbers (static.Lump), sponge distance D (Phi = 1e-3 Phi0) and the face minimum for the window."""
    from scipy.optimize import brentq
    from swirl.engine.snapshot import window_profile
    from swirl.session4.testA.static import Lump
    ty = TYPES[t]
    l = Lump(ty["beta"], ty["omega"])
    D = brentq(lambda x: l.ff(np.array([x]))[0][0] - 1e-3 * l.Phi0, 2.0, 120.0)
    rr = np.linspace(0.0, ty["window"][1] + 1.0, 20001)
    env = np.abs(l.ff(np.maximum(rr, 1e-9))[0]) / l.Phi0 * window_profile(rr, *ty["window"])
    bad = np.nonzero(env > 1e-6)[0]
    face = float(rr[bad[-1] + 1]) if bad.size else 0.0
    req = max(3 * l.hw + SPONGE_W, SPONGE_W + D, face) + 0.05
    return dict(A=l.A, kappa=l.kappa, hw=l.hw, N=l.N, E=l.E, Phi0=l.Phi0, omega=ty["omega"], beta=ty["beta"],
                g6=0.09 * ty["beta"], D_sponge=D, r_face_min=face, req=req)


def profile_table(t, rel=1e-7):
    from swirl.session4.testA.static import Lump
    ty = TYPES[t]
    l = Lump(ty["beta"], ty["omega"])
    r = np.arange(0.0, 200.0, 0.05)
    P = l.ff(r)[0]
    k = int(np.argmax(P < rel * P[0]))
    return dict(r=r[:k + 1].tolist(), Phi=P[:k + 1].tolist())


def box_N(L_min, dx=DX):
    for n in NICE:
        if n * dx >= L_min - 1e-9:
            return n
    raise ValueError("box too large: L_min %.2f" % L_min)


def pair_box(t1, t2, r, dx=DX):
    req = max(numbers(t1)["req"], numbers(t2)["req"])
    return box_N(r + 2 * req, dx)


# ------------------------------------------------------------------------------------------- configs
def physics(beta, linear=False):
    return dict(m=1.0, g4=0.0 if linear else G4, g6=0.0 if linear else 0.09 * beta)


def snapshot_entry(t, centers, phases, resample=False):
    from swirl.engine.snapshot import load_meta
    sp = source_path(t)
    sha = load_meta(sp)["sha256"] if os.path.exists(sp) else "PENDING"   # filled from the built source at run start
    # built sources are referenced relative to the worktree root (snapshot.resolve), the reused primary absolutely
    e = dict(source=sp if t == "primary" else TYPES[t]["source"], sha256=sha, centers=[list(map(float, c)) for c in centers],
             phases=[float(p) for p in phases], window=TYPES[t]["window"], margin=2.0)
    if resample:
        e["resample"] = True
    return e


def run_cfg(lumps, N, dx=DX, T=T_MAX, linear=False):
    """lumps: [(type, x, phase_rad)]; one engine config."""
    L = round(N * dx, 9)
    types = [lp[0] for lp in lumps]
    beta = TYPES[types[0]]["beta"]
    by_type = {}
    for t, x, ph in lumps:
        by_type.setdefault(t, []).append(([x, 0.0, 0.0], ph))
    entries = [snapshot_entry(t, [c for c, _ in v], [p for _, p in v], resample=abs(dx - DX) > 1e-12)
               for t, v in by_type.items()]
    init = {"snapshot": entries[0]} if len(entries) == 1 else {"snapshots": entries}
    centers = [[x, 0.0, 0.0] for _, x, _ in lumps]
    hw = max(numbers(t)["hw"] for t in types)
    ld = dict(centers=centers, R_ball=max(TYPES[t]["R_ball"] for t in types), taper=2.0, every=5, count_every=25.0,
              omega_nominal=TYPES[types[0]]["omega"], it_every_step=True,
              it_half=[{"axis": 0, "pos": 0.0}], planes=[{"axis": 0, "pos": 0.0}])
    return dict(N=int(N), L=float(L), dt=DT * dx / DX, t_end=float(T), init=init, absorber=True,
                center=centers[0], r_c=[3 * hw], sample_every=1.0, point_every=1, avg_window=20.0,
                checkpoint_every=50.0, threads=1, lumps=ld, **physics(beta, linear))


def jobs(b5=False):
    """The Test A run list."""
    J = []

    def add(name, group, kind, cfg=None, deps=(), **kw):
        J.append(dict(name=name, group=group, kind=kind, cfg=cfg, deps=list(deps), **kw))

    for t in ("dip", "big") + (("b5",) if b5 else ()):
        N, L = TYPES[t]["src_grid"]
        cfg = dict(N=N, L=L, dt=DT, t_end=TYPES[t]["t_s"], init={"radial": [dict(profile_table(t), center=[0.0, 0.0, 0.0],
                                                                            phase=0.0, omega=TYPES[t]["omega"])]},
                   absorber=True, center=[0.0, 0.0, 0.0], r_c=[3 * numbers(t)["hw"]], sample_every=1.0,
                   point_every=1, avg_window=20.0, checkpoint_every=50.0, threads=1, **physics(TYPES[t]["beta"]))
        add("src_" + t, "sources", "source", cfg, out=TYPES[t]["source"], t_s=TYPES[t]["t_s"], windows=SRC_WINDOWS[t])

    def family(prefix, t1, t2, rs, chans, group):
        boxes = {}
        for r in rs:
            N = pair_box(t1, t2, r)
            boxes.setdefault(N, []).append(r)
            for dth in chans:
                ph = np.deg2rad(dth) / 2.0
                deps = ["src_" + t for t in {t1, t2} if t != "primary"]
                add("%s_r%02d_p%03d" % (prefix, r, dth), group, "pair",
                    run_cfg([(t1, -r / 2.0, -ph), (t2, r / 2.0, ph)], N), deps, r=float(r), dtheta=dth,
                    types=[t1, t2], box=N)
        for N, rr in boxes.items():
            for rpos in sorted({min(rr), max(rr)}):
                for t, sgn in ((t1, -1.0), (t2, 1.0)) if t1 != t2 else ((t1, -1.0),):
                    deps = ["src_" + t] if t != "primary" else []
                    add("%s_single_%s_N%d_x%+05.1f" % (prefix, t, N, sgn * rpos / 2.0), group, "single",
                        run_cfg([(t, sgn * rpos / 2.0, 0.0)], N), deps, r=float(rpos), types=[t], box=N)
        return boxes

    family("pA", "primary", "primary", R_PRIMARY, DTHETA, "primary")
    family("uA", "dip", "big", R_UNEQUAL, DTHETA, "unequal")
    if b5:
        family("bA", "b5", "b5", R_B5, (0,), "b5")
    # controls on the primary
    for r in (12, 24):
        N = pair_box("primary", "primary", r)
        add("pA_r%02d_p000_lin" % r, "controls", "twin",
            run_cfg([("primary", -r / 2.0, 0.0), ("primary", r / 2.0, 0.0)], N, T=200.0, linear=True),
            r=float(r), dtheta=0, types=["primary", "primary"], box=N)
    N24 = pair_box("primary", "primary", 24)
    Nb = int(round(1.5 * N24 / 2.0)) * 2
    add("pA_r24_p000_box%d" % Nb, "controls", "pair",
        run_cfg([("primary", -12.0, 0.0), ("primary", 12.0, 0.0)], Nb), r=24.0, dtheta=0,
        types=["primary", "primary"], box=Nb, check="box")
    add("pA_single_primary_N%d_x-12.0" % Nb, "controls", "single", run_cfg([("primary", -12.0, 0.0)], Nb),
        r=24.0, types=["primary"], box=Nb, check="box")
    dx_r = 0.40
    Lmin = 16 + 2 * numbers("primary")["req"]
    Nr = int(np.ceil(Lmin / dx_r / 8.0)) * 8
    add("pA_r16_p000_dx040", "controls", "pair",
        run_cfg([("primary", -8.0, 0.0), ("primary", 8.0, 0.0)], Nr, dx=dx_r), r=16.0, dtheta=0,
        types=["primary", "primary"], box=Nr, dx=dx_r, check="resolution")
    add("pA_single_primary_N%d_dx040_x-08.0" % Nr, "controls", "single",
        run_cfg([("primary", -8.0, 0.0)], Nr, dx=dx_r), r=16.0, types=["primary"], box=Nr, dx=dx_r,
        check="resolution")
    return J


# ------------------------------------------------------------------------------------------- cost
RATE_FILE = OUT + "/cost_rate.json"


def step_seconds(N, rate=None):
    """Seconds per step (one thread) from the measured rate file: a N^3 log N model fitted to measured
    full-diagnostic step times."""
    rate = rate or json.load(open(_abs(RATE_FILE)))
    c = rate["c_per_point_log"]
    return c * N ** 3 * np.log(N) * rate.get("contention", 1.0)


def job_hours(j, rate=None):
    if j["kind"] == "source":
        return step_seconds(j["cfg"]["N"], rate) * (j["t_s"] + 140.0) / DT / 3600.0
    T = j["cfg"]["t_end"]
    if j["kind"] == "pair":
        T = min(T, expected_stop(j))
    return step_seconds(j["cfg"]["N"], rate) * T / j["cfg"]["dt"] / 3600.0 + 0.02


def expected_stop(j):
    """Point-model estimate of the early-stop time (constant tail force, d'' = 2F/E; unequal: no stop)."""
    t1, t2 = j["types"]
    if t1 != t2:
        return T_MAX
    n = numbers(t1)
    r = j["r"]
    F = 2 * np.pi * n["A"] ** 2 * np.exp(-n["kappa"] * r) * (n["kappa"] / r + 1 / r ** 2)
    a = 2 * F / n["E"]
    ts = np.sqrt(2 * 0.13 / n["kappa"] / a)
    return float(min(T_MAX, max(60.0, ts + 10.0)))


def cost_table(J, workers=4, rate=None):
    rows = [(j["name"], j["group"], j["kind"], j["cfg"]["N"], job_hours(j, rate)) for j in J]
    by = {}
    for _, g, _, _, h in rows:
        by[g] = by.get(g, 0.0) + h
    tot = sum(by.values())
    # longest-first list schedule on `workers` slots (dependencies: sources first)
    slots = [0.0] * workers
    src = sorted([x for x in rows if x[2] == "source"], key=lambda x: -x[4])
    rest = sorted([x for x in rows if x[2] != "source"], key=lambda x: -x[4])
    for x in src:
        i = int(np.argmin(slots))
        slots[i] += x[4]
    t_src = max(slots) if src else 0.0
    slots = [max(s, 0.0) for s in slots]
    for x in rest:
        i = int(np.argmin(slots))
        slots[i] += x[4]
    return dict(job_hours_by_group=by, job_hours_total=tot, workers=workers, wall_hours_schedule=max(slots),
                wall_hours_sources=t_src, n_jobs=len(rows), rows=rows)


# ------------------------------------------------------------------------------------------- t = 0 check
def t0_check(cfg):
    """Engine energy of the start against an independent evaluation (complex FFT gradient, numpy sums) and
    the cross energy against the sum of the copies placed alone (independent cross-term quadrature)."""
    from swirl.engine.evolver import Evolver
    from swirl.engine.initial import build_initial
    from swirl.engine.run import full_config
    c = full_config(cfg)
    ev = Evolver(c["N"], c["L"], c["m"], c["g4"], c["g6"], absorber=c["absorber"], threads=c["threads"])
    phi, pi = build_initial(ev.coords, ev.L, c["init"], threads=c["threads"])
    E_eng = ev.energy_parts(phi, pi)["E_total"]
    k = 2 * np.pi * np.fft.fftfreq(c["N"], d=ev.dx)
    KS = (k[:, None, None], k[None, :, None], k[None, None, :])

    def grad_dot(f, g=None):
        """sum over the grid of grad f . grad g (complex FFT, one component at a time to bound memory)."""
        fh = np.fft.fftn(f)
        gh = fh if g is None else np.fft.fftn(g)
        s = 0.0
        for kk in KS:
            a = np.real(np.fft.ifftn(1j * kk * fh))
            b = a if g is None else np.real(np.fft.ifftn(1j * kk * gh))
            s += float(np.sum(a * b))
            del a, b
        return s

    p2 = phi * phi
    E_ind = (0.5 * float(np.sum(pi * pi)) + 0.5 * grad_dot(phi)
             + float(np.sum(p2 * (0.5 * c["m"] ** 2 + p2 * (-c["g4"] / 24 + p2 * c["g6"] / 720))))) * ev.dV
    del p2
    out = dict(E_engine=E_eng, E_independent=E_ind, rel_diff=E_ind / E_eng - 1.0, target=1e-10,
               note="independent: complex FFT gradient incl. Nyquist (the engine uses the rfft K^2 sum)")
    lumps = c["lumps"]["centers"]
    if len(lumps) == 2:
        parts = []
        init = c["init"]
        entries = init["snapshots"] if "snapshots" in init else [init["snapshot"]]
        for i in range(2):
            ents = []
            for e in entries:
                for cc, ph in zip(e["centers"], e["phases"]):
                    if np.allclose(cc, lumps[i]):
                        ents.append({**e, "centers": [cc], "phases": [ph]})
            p_i, v_i = build_initial(ev.coords, ev.L, {"snapshots": ents}, threads=c["threads"])
            parts.append((p_i, v_i, ev.energy_parts(p_i, v_i)["E_total"]))
        (p1, v1, E1), (p2, v2, E2) = parts
        a, b = p1, p2
        nl = float(np.sum(-c["g4"] / 24 * (4 * a ** 3 * b + 6 * a * a * b * b + 4 * a * b ** 3)
                          + c["g6"] / 720 * (6 * a ** 5 * b + 15 * a ** 4 * b * b + 20 * a ** 3 * b ** 3
                                             + 15 * a * a * b ** 4 + 6 * a * b ** 5)))
        X_ind = (float(np.sum(v1 * v2)) + grad_dot(p1, p2) + c["m"] ** 2 * float(np.sum(a * b)) + nl) * ev.dV
        out.update(E_copy1=E1, E_copy2=E2, E_cross_engine=E_eng - E1 - E2, E_cross_independent=X_ind,
                   cross_abs_diff=(E_eng - E1 - E2) - X_ind,
                   cross_rel_diff=((E_eng - E1 - E2) - X_ind) / X_ind if X_ind else None,
                   superposition_residual_max=float(np.max(np.abs(phi - p1 - p2))))
    return out


# ------------------------------------------------------------------------------------------- jobs
def result_path(name):
    return _abs(os.path.join(OUT, "results", name + ".json"))


def _smoothed_sep(res, period):
    ls = res["lumps"]["raw"]["series"]
    t = np.array(ls["t"])
    X = np.array(ls["X"])
    if X.ndim != 3 or X.shape[1] < 2 or t.size < 3:
        return t, None
    d = np.linalg.norm(X[:, 1, :] - X[:, 0, :], axis=1)
    n = max(1, int(round(period / (t[1] - t[0]))))
    ker = np.ones(n) / n
    return t[n - 1:] - 0.5 * (t[n - 1] - t[0]), np.convolve(d, ker, mode="valid")


def run_pair(j, chunk_time=50.0):
    """Engine run in chunks with the early-stop rule; writes the result JSON (+ stopped_early)."""
    from swirl.engine.run import run
    from swirl.engine.snapshot import load_meta
    cfg = j["cfg"]
    for e in ([cfg["init"]["snapshot"]] if "snapshot" in cfg["init"] else cfg["init"]["snapshots"]):
        if e["sha256"] == "PENDING":
            e["sha256"] = load_meta(e["source"])["sha256"]
    ck = _abs(os.path.join(CKPT, j["name"]))
    t0p = _abs(os.path.join(OUT, "t0", j["name"] + ".json"))
    if not os.path.exists(t0p) and j["kind"] in ("pair", "twin"):
        atomic_json(dict(name=j["name"], params=cfg, t0=t0_check(cfg)), t0p)
    steps = int(round(chunk_time / cfg["dt"]))
    stop_rule = j["kind"] == "pair"
    kap = numbers(j["types"][0])["kappa"]
    period = 2 * np.pi / TYPES[j["types"][0]]["omega"]
    while True:
        res = run(cfg, out_path=None, checkpoint_dir=ck, max_steps=steps)
        tnow = res["t"]
        early = False
        if stop_rule and not res["completed"] and tnow >= 60.0 - 1e-9:
            ts, ds = _smoothed_sep(res, period)
            if ds is not None and ds.size and kap * abs(ds[-1] - j["r"]) > 0.13:
                early = True
        if res["completed"] or early:
            res.pop("_evolver", None)
            res["stopped_early"] = bool(early)
            res["job"] = {k: v for k, v in j.items() if k != "cfg"}
            atomic_json(res, result_path(j["name"]))
            return
        del res


def run_job(j):
    t0 = time.time()
    try:
        if j["kind"] == "source":
            from swirl.engine.snapshot import make_source
            out = _abs(j["out"])
            name = os.path.basename(out)[:-4]
            meta = make_source(name, j["cfg"], j["t_s"], os.path.dirname(out), windows=j["windows"],
                               checkpoint_dir=_abs(os.path.join(CKPT, j["name"])), verbose=True)
            if meta["omega_drift"] > 1e-4:
                raise RuntimeError("source not settled: omega drift %.3g" % meta["omega_drift"])
            atomic_json(dict(job={k: v for k, v in j.items() if k != "cfg"},
                             meta={k: v for k, v in meta.items() if k not in ("profile",)}), result_path(j["name"]))
        else:
            run_pair(j)
        return j["name"], "done", time.time() - t0
    except Exception as e:  # recorded; dependents are skipped
        import traceback
        return j["name"], "failed: %r\n%s" % (e, traceback.format_exc()), time.time() - t0


def _init_worker(threads):
    os.environ["NUMBA_NUM_THREADS"] = str(threads)


def run_jobs(J, workers=4, poll=10.0):
    if workers > 4:
        raise ValueError("at most 4 worker processes (plan section 4)")
    for j in J:
        if j.get("cfg"):
            j["cfg"]["threads"] = 1
    names = {j["name"] for j in J}
    done = {j["name"] for j in J if os.path.exists(result_path(j["name"]))}
    rate = json.load(open(_abs(RATE_FILE))) if os.path.exists(_abs(RATE_FILE)) else None
    pending = sorted([j for j in J if j["name"] not in done],
                     key=lambda j: (j["kind"] != "source", -(job_hours(j, rate) if rate else j["cfg"]["N"])))
    failed, running = set(), {}
    log = lambda *a: print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)
    log("jobs: %d selected, %d already done, %d to run" % (len(J), len(done), len(pending)))
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1, initializer=_init_worker, initargs=(1,)) as pool:
        while pending or running:
            for j in list(pending):
                if any(d in failed for d in j["deps"]):
                    pending.remove(j)
                    failed.add(j["name"])
                    log("skip %s (dependency failed)" % j["name"])
                    continue
                ready = all(d in done or (d not in names and os.path.exists(result_path(d))) for d in j["deps"])
                if ready and len(running) < workers:
                    running[j["name"]] = pool.apply_async(run_job, (j,))
                    pending.remove(j)
                    log("start %s (N=%d)" % (j["name"], j["cfg"]["N"]))
            for name, ar in list(running.items()):
                if ar.ready():
                    n, status, sec = ar.get()
                    del running[name]
                    (done if status == "done" else failed).add(name)
                    log("%s %s after %.0f s" % (name, status, sec))
            if pending and not running and not any(
                    all(d in done or (d not in names and os.path.exists(result_path(d))) for d in j["deps"])
                    for j in pending):
                log("blocked: %s" % [j["name"] for j in pending])
                break
            time.sleep(poll)
    log("finished: %d done, %d failed %s" % (len(done & names), len(failed), sorted(failed)))
    return sorted(failed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["jobs", "cost", "run"])
    ap.add_argument("--b5", action="store_true", help="include the beta = 5 runs (cost gate)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--group", nargs="*", default=None)
    ap.add_argument("--jobs-file", default=OUT + "/jobs.json")
    a = ap.parse_args()
    if a.cmd == "jobs":
        J = jobs(b5=a.b5)
        atomic_json(J, a.jobs_file)
        print("%d jobs written to %s" % (len(J), a.jobs_file))
        return
    J = json.load(open(_abs(a.jobs_file)))
    if a.group:
        J = [j for j in J if j["group"] in a.group or j["kind"] == "source"]
    if a.only:
        J = [j for j in J if j["name"] in a.only]
    if a.cmd == "cost":
        ct = cost_table(J, a.workers)
        print(json.dumps({k: v for k, v in ct.items() if k != "rows"}, indent=1))
        return
    failed = run_jobs(J, a.workers)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
