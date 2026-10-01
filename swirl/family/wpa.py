"""WP-A command line: families -> maps -> rank -> long, plus validate, checks, cost.

  PY=/Users/aaronramos/Desktop/SWIRL/.venv/bin/python
  $PY -m swirl.family.wpa families --betas 0.75 1.15 1.6 2.0 3.0 5.0 --workers 2
  $PY -m swirl.family.wpa maps     --betas <list> --workers 2
  $PY -m swirl.family.wpa rank     --betas <list>
  $PY -m swirl.family.wpa long     --betas <list> --workers 2 [--T 30000]
  $PY -m swirl.family.wpa validate --workers 2
  $PY -m swirl.family.wpa checks   --workers 2
  $PY -m swirl.family.wpa cost     --betas <list>

Every stage resumes: families and maps skip finished outputs (map runs are cached one
per start under outputs/session2/wpA/checkpoints/maps), long runs checkpoint every 1000
time units and continue from the checkpoint. Run from the repository root.
"""

import os

if __name__ in ("__main__", "__mp_main__"):     # the CLI and its spawned workers only,
    os.environ["NUMBA_NUM_THREADS"] = "1"       # before numba is imported (workers inherit it)

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

from swirl.family import wpa_family as fam
from swirl.family import wpa_rate as rate
from swirl.family.wpa_run import grid_for, profile, r_lump, run, window_fits, write_json

OUT = fam.OUT
BETAS_ALL = [0.75, 0.85, 1.0, 1.15, 4.0 / 3.0, 1.6, 2.0, 3.0, 5.0]
BETAS_NEW = [0.75, 1.15, 1.6, 2.0, 3.0, 5.0]
T_LONG, T_TWIN_LONG = 30000.0, 1000.0
VALIDATIONS = [
    dict(name="c", beta=0.85, omega=0.7793064609301195, quantity="T_death", measured=1145.5,
         source="outputs/step2/diagnostics/radial_T1000.json D1_run1b.death_time_radial"),
    dict(name="d", beta=4.0 / 3.0, omega=0.909149280617113, quantity="T_90", measured=10630.5,
         source="outputs/step2/diagnostics/primary_lifetime.json t_frac_below_0p90"),
]
EXCLUDE_DW = 0.002     # validation: map starts this close to the validation omega are dropped


def _pool(workers, fn, jobs, init=None):
    if workers <= 1:
        if init:
            init()
        return [fn(j) for j in jobs]
    with Pool(workers, initializer=init) as p:
        return p.map(fn, jobs, chunksize=1)


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------ families

_S43 = None


def _init_s43():
    global _S43
    _S43 = fam.beta43_scan()


def _family_job(beta):
    c = fam.run_beta(beta, _S43)
    _log("family beta=%.4f rows=%d derrick %.2e universal %.2e dE central %.2e"
         % (beta, c["n_rows"], c["derrick_worst"], c["universal_worst"], c["dEdw_central_worst"]))
    return beta, c


def stage_families(betas, workers, force=False):
    todo = [b for b in betas if force or not os.path.exists(
        os.path.join(OUT, "family_beta%.4f.json" % b))]
    res = _pool(workers, _family_job, todo, init=_init_s43) if todo else []
    summ = {}
    for b in betas:
        with open(os.path.join(OUT, "family_beta%.4f.json" % b)) as f:
            d = json.load(f)
        c = d["checks_wpA"]
        summ["%.4f" % b] = dict(
            n_rows=c["n_rows"], derrick_worst=c["derrick_worst"], universal_worst=c["universal_worst"],
            dEdw_central_worst=c["dEdw_central_worst"], dEdw_rows_median=c["dEdw_rows_median_rel"],
            dEdw_rows_worst=c["dEdw_rows_worst_rel"],
            omega_minN=d["features_rescaled"]["omega_minN"],
            omega_end_analytic=d["features_rescaled"]["omega_end_analytic"],
            omega_lowest_computed=d["features_rescaled"]["omega_lowest_computed"],
            E_min_setB=d["features_setB_units"]["E_min"], N_min_setB=d["features_setB_units"]["N_min"],
            hw_at_minE=d["features_rescaled"]["hw_at_minE"], wall_s=d["wall_s"])
    worst = lambda k: max(v[k] for v in summ.values())
    write_json(os.path.join(OUT, "families_summary.json"), dict(
        betas=betas, per_beta=summ,
        worst=dict(derrick=worst("derrick_worst"), universal=worst("universal_worst"),
                   dEdw_central=worst("dEdw_central_worst")),
        targets=dict(derrick=1e-6, universal=1e-5, dEdw_central=1e-5)))
    _log("families done: %s" % json.dumps({k: v for k, v in summ.items()}, default=str)[:2000])


# ------------------------------------------------------------------ maps

def _map_job(args):
    beta, a2, st = args
    t0 = time.time()
    x = rate.map_run(beta, a2, st)
    _log("map beta=%.4f a2=%.5f w=%.5f G1=%s T_death=%s (%.0fs)"
         % (beta, a2, x["omega_profile"], x["Gamma_first"], x["T_death"], time.time() - t0))
    return x


def stage_maps(betas, workers, force=False, refine=True):
    sts = {b: rate.map_settings(b) for b in betas if force or not os.path.exists(rate.map_path(b))}
    jobs = [(b, a2, st) for b, st in sts.items() for a2 in st["coarse_alpha2"]]
    _log("maps: %d coarse runs over %d betas" % (len(jobs), len(sts)))
    res = _pool(workers, _map_job, jobs)
    by = {b: [x for x in res if x["beta"] == b] for b in sts}
    if refine:
        jobs = []
        for b, st in sts.items():
            new = rate.refine_alpha2(by[b])
            st["refined_alpha2"] = new
            jobs += [(b, a2, st) for a2 in new]
        _log("maps: %d refinement runs" % len(jobs))
        for x in _pool(workers, _map_job, jobs):
            by[x["beta"]].append(x)
    for b, st in sts.items():
        rate.write_map(b, st, by[b])
        _log("map written: %s" % rate.map_path(b))


# ------------------------------------------------------------------ ranking

def stage_rank(betas):
    out = {}
    for b in betas:
        rk = rate.rank(b)
        out["%.4f" % b] = rk
        bb = rk["best"]
        _log("rank beta=%.4f best w=%.5f T_death_pred=%s T_90_pred=%s"
             % (b, bb["omega_profile"], bb["T_death"], bb["T_90"]))
    write_json(os.path.join(OUT, "ranking.json"), dict(
        description="Per beta, every map start ranked by the path-integral T_death "
                    "(see wpa_rate docstring); 'best' is the start the long run uses.",
        per_beta=out))
    _plot_maps(betas)


def _plot_maps(betas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from swirl.family.io import SERIES, style
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, b in enumerate(betas):
        with open(rate.map_path(b)) as f:
            m = json.load(f)
        c = SERIES[k % len(SERIES)]
        pts = rate.points_from_runs(m["runs"])
        E0min = m["settings"]["E_min_setB"]
        e = np.array([p["E"] for p in pts]) / E0min
        g = np.array([max(p["Gamma"], 2 * p["sigma"]) for p in pts])
        w = np.array([p["omega"] if p["omega"] else np.nan for p in pts])
        axs[0].plot(e, g, ".", color=c, ms=3, label="beta = %.4g" % b)
        xs = np.array([n["x"] for n in m["nodes"]])
        axs[0].plot(np.exp(xs) / E0min, np.exp([n["lc"] for n in m["nodes"]]), "-", color=c, lw=1)
        axs[1].plot(w, g, ".", color=c, ms=3, label="beta = %.4g" % b)
    for ax in axs:
        style(ax)
        ax.set_yscale("log")
        ax.set_ylabel("Gamma = -d ln E_lump/dt (windows of 300; floor 2 sigma)")
    axs[0].set_xscale("log")
    axs[0].set_xlabel("measured E_lump / E_min(beta)")
    axs[1].set_xlabel("measured core omega / m")
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle("WP-A radiation-rate maps (radial): points = windows, lines = assembled Gamma(E)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ratemaps.png"), dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ long runs

def _long_job(args):
    beta, w0, T = args
    st = rate.map_settings(beta) if os.path.exists(rate.map_path(beta)) else None
    p = profile(beta, w0)
    rl = st["R_lump"] if st else r_lump(p["half_width"], p["kappa"])
    rcs = [3.0 * p["half_width"], rl]
    tag = "beta%.4f_w%.6f" % (beta, w0)
    ck = os.path.join(OUT, "checkpoints", "long_%s.npz" % tag)
    t0 = time.time()
    r = run(beta, w0, T, rcs=rcs, prof=p, ckpt=ck, ckpt_every=1000.0)
    tw = run(beta, w0, T_TWIN_LONG, linear=True, rcs=rcs, prof=p, stop_at_death=False)
    t = r["series"]["t"]
    El = r["series"]["E_in"][:, 1]
    t_s = 150.0 + 2.0 * rl
    i_s = int(np.argmin(np.abs(t - t_s)))
    wins = window_fits(t, El, t_s, 1000.0, float(t[-1]), r["omega_t"], r["omega"]) if t[-1] > t_s else []
    ft = tw["series"]["E_in"][:, 0] / tw["E0"]
    tt = tw["series"]["t"]
    keep = slice(None, None, 10)
    out = dict(
        description="WP-A long radial run: death or T, whichever first. frac = E(r<3hw)/E0. "
                    "T_90, T_death as in SESSION_02_PREDICTIONS Block 1; osc_* = T omega_settled/(2 pi); "
                    "periods_counted_* = upward zero crossings of phi(0,t) (Schmitt trigger). "
                    "E_in quadrature error ~1e-4 of E at dr = 0.05 (F-003).",
        params=r["params"], E0=r["E0"], regime=r["regime"], T_end=r["T_end"], T_90=r["T_90"],
        T_death=r["T_death"], osc_T_90=r["osc_T_90"], osc_T_death=r["osc_T_death"],
        osc_T_end=r["osc_T_end"], periods_counted_T_90=r["periods_counted_T_90"],
        periods_counted_T_death=r["periods_counted_T_death"],
        periods_counted_total=r["periods_counted_total"], frac_t0=r["frac_t0"],
        frac_end=r["frac_end"], E_ref50=r["E_ref50"], energy_check=r["energy_check"],
        frac_at={str(T): float(r["series"]["E_in"][int(np.argmin(np.abs(t - T))), 0] / r["E0"])
                 for T in (1000, 2000, 5000, 10000, 20000, 30000) if T <= t[-1] + 0.5},
        t_s=t_s, E_s=float(El[i_s]), windows_1000=wins,
        linear_twin=dict(T=T_TWIN_LONG, frac_T100=float(ft[int(np.argmin(np.abs(tt - 100)))]),
                         frac_T300=float(ft[int(np.argmin(np.abs(tt - 300)))]),
                         frac_T1000=float(ft[-1]), E0=tw["E0"]),
        series=dict(t=t[keep], frac=r["series"]["E_in"][keep, 0] / r["E0"],
                    E_lump_over_E0=El[keep] / r["E0"], E_tot_over_E0=r["series"]["E_tot"][keep] / r["E0"]),
        omega_series=dict(t=r["omega_t"][::20], omega=r["omega"][::20]),
        nl_share_series=r["nl"], wall_s=time.time() - t0)
    path = write_json(os.path.join(OUT, "long_%s.json" % tag), out)
    _log("long %s: T_end %.1f T_90 %s T_death %s (%.0fs) -> %s"
         % (tag, r["T_end"], r["T_90"], r["T_death"], time.time() - t0, path))
    return path


def stage_long(betas, workers, T=T_LONG, omega=None):
    jobs = []
    if omega is not None:
        jobs = [(betas[0], omega, T)]
    else:
        with open(os.path.join(OUT, "ranking.json")) as f:
            rk = json.load(f)["per_beta"]
        for b in betas:
            ws = {rk["%.4f" % b][k]["omega_profile"] for k in ("best", "best_by_measured_rate")}
            jobs += [(b, w, T) for w in sorted(ws)]
    _pool(workers, _long_job, jobs)


# ------------------------------------------------------------------ validation

def _validation_rerun(args):
    v, st = args
    T = v["measured"] * 1.03 + 50
    p = profile(v["beta"], v["omega"])
    rcs = [3.0 * p["half_width"], st["R_lump"]]
    r = run(v["beta"], v["omega"], T, rcs=rcs, prof=p, stop_at_death=True)
    return r


def stage_validate(workers):
    betas = sorted({v["beta"] for v in VALIDATIONS})
    missing = [b for b in betas if not os.path.exists(rate.map_path(b))]
    if missing:
        stage_maps(missing, workers)
    reruns = _pool(workers, _validation_rerun,
                   [(v, rate.map_settings(v["beta"])) for v in VALIDATIONS])
    out = []
    for v, rr in zip(VALIDATIONS, reruns):
        with open(rate.map_path(v["beta"])) as f:
            m = json.load(f)
        st = m["settings"]
        kept = [x for x in m["runs"] if abs(x["omega_profile"] - v["omega"]) >= EXCLUDE_DW]
        dropped = [x["omega_profile"] for x in m["runs"] if abs(x["omega_profile"] - v["omega"]) < EXCLUDE_DW]
        nodes = rate.assemble(rate.points_from_runs(kept))
        # start state from the profile and the map neighbours only (no evolution of this start)
        a2v = 0.3 * v["beta"] * (1 - v["omega"] ** 2)
        ok = [x for x in kept if x["s_frac"] is not None]
        a2k = np.array([x["alpha2"] for x in ok])
        o = np.argsort(a2k)
        s_frac = float(np.interp(a2v, a2k[o], np.array([x["s_frac"] for x in ok])[o]))
        E0, E3_0, El_0 = rr["E0"], rr["series"]["E_in"][0, 0], rr["series"]["E_in"][0, 1]
        rho0 = float(El_0 / E3_0)
        E_s = s_frac * E0
        pr = rate.predict(nodes, st["t_s"], E_s, E0, rho0, st["E_min_setB"])
        pred = pr[v["quantity"]]
        lo, hi = pr[v["quantity"] + "_lo"], pr[v["quantity"] + "_hi"]
        # diagnostics from the rerun (the measured run itself, NOT used for the prediction)
        t = rr["series"]["t"]
        El = rr["series"]["E_in"][:, 1]
        measured_rerun = rr[v["quantity"]]
        t_stop = float(t[-1])
        dead = np.where((t > st["t_s"]) & (El < 0.5 * El[int(np.argmin(np.abs(t - st["t_s"])))]))[0]
        if dead.size:
            t_stop = float(t[dead[0]])
        own = window_fits(t, El, st["t_s"], st["W"], t_stop, rr["omega_t"], rr["omega"])
        own_nodes = rate.assemble([dict(E=w_["E"], Gamma=w_["Gamma"], sigma=w_["sigma"],
                                        omega=w_["omega"]) for w_ in own])
        E_s_own = float(El[int(np.argmin(np.abs(t - st["t_s"])))])
        pr_own = rate.predict(own_nodes, st["t_s"], E_s_own, E0, rho0, st["E_min_setB"])
        # secondary, as the ranking uses the tool: start energy from this start's own first
        # t_s time units, Gamma(E) still from the other runs only
        pr_es = rate.predict(nodes, st["t_s"], E_s_own, E0, rho0, st["E_min_setB"])
        cmp_E, cmp_w = [], []
        mpts = rate.points_from_runs(kept)
        mw = np.array([p_["omega"] if p_["omega"] else np.nan for p_ in mpts])
        mg = np.array([max(p_["Gamma"], 2 * p_["sigma"]) for p_ in mpts])
        for w_ in own:
            gm = float(np.exp(rate._lgamma(nodes, "lc", np.log(w_["E"]))))
            cmp_E.append(dict(t0=w_["t0"], E=w_["E"], omega=w_["omega"], Gamma_own=w_["Gamma"],
                              Gamma_map_at_E=gm, log10_ratio=float(np.log10(max(w_["Gamma"], 1e-300) / gm))))
            if w_["omega"]:
                sel = np.abs(mw - w_["omega"]) < 0.001
                if sel.any():
                    gw = float(np.median(mg[sel]))
                    cmp_w.append(dict(t0=w_["t0"], omega=w_["omega"], Gamma_own=w_["Gamma"],
                                      Gamma_map_at_omega=gw, n_map=int(sel.sum()),
                                      log10_ratio=float(np.log10(max(w_["Gamma"], 1e-300) / gw))))
        rel = None if pred is None else (pred - v["measured"]) / v["measured"]
        res = dict(
            **v, alpha2=a2v, predicted=pred, predicted_lo=lo, predicted_hi=hi, rel_error=rel,
            target_rel=0.30, verdict=None if rel is None else ("hit" if abs(rel) <= 0.30 else "miss"),
            prediction=pr, s_frac_interpolated=s_frac, E0=E0, rho0=rho0,
            map_runs_used=len(kept), map_runs_dropped_omegas=dropped,
            nearest_used_start_domega=float(min(abs(x["omega_profile"] - v["omega"]) for x in kept)),
            nearest_used_starts=sorted([x["omega_profile"] for x in kept],
                                       key=lambda w: abs(w - v["omega"]))[:4],
            rerun_check=dict(measured_session1=v["measured"], measured_rerun=measured_rerun,
                             T_90_rerun=rr["T_90"], T_death_rerun=rr["T_death"],
                             E_s_rerun=E_s_own, E_s_predicted=E_s,
                             omega_settled_rerun=rr["regime"]["omega_over_m_settled"]),
            D1_own_vs_map_in_E=cmp_E,
            D2_path_integral_with_own_Gamma=dict(predicted=pr_own[v["quantity"]],
                                                 rel_error=None if pr_own[v["quantity"]] is None else
                                                 (pr_own[v["quantity"]] - measured_rerun) / measured_rerun
                                                 if measured_rerun else None,
                                                 flags=pr_own["flags_90" if v["quantity"] == "T_90" else "flags_death"]),
            D3_own_vs_map_in_omega=cmp_w,
            secondary_E_s_from_own_first_t_s=dict(
                predicted=pr_es[v["quantity"]], lo=pr_es[v["quantity"] + "_lo"],
                hi=pr_es[v["quantity"] + "_hi"],
                rel_error=None if pr_es[v["quantity"]] is None else
                (pr_es[v["quantity"]] - v["measured"]) / v["measured"],
                note="not the validation number; separates the start-energy estimate from Gamma(E)"),
            settings=st)
        out.append(res)
        _log("validation %s: beta=%.4f %s predicted %s (band %s..%s) measured %s rel %s; own-Gamma integral %s"
             % (v["name"], v["beta"], v["quantity"], pred, lo, hi, v["measured"], rel,
                res["D2_path_integral_with_own_Gamma"]["predicted"]))
    write_json(os.path.join(OUT, "validation.json"), dict(
        description="Checks c and d: path-integral prediction from a map without the validation "
                    "start (starts within %.3f in omega dropped), against the Session 1 measured "
                    "value. D1: Gamma of the validation run's own windows against the map at the "
                    "same measured E. D2: the path integral with the run's own Gamma(E) nodes "
                    "(tests the integral given the right Gamma). D3: same as D1 at the same "
                    "measured omega." % EXCLUDE_DW, validations=out))


# ------------------------------------------------------------------ checks and cost

def _box_job(args):
    beta, a2, scale, dr, T = args
    w0 = rate.a2_to_w(beta, a2)
    p = profile(beta, w0)
    g = grid_for(beta, scale=scale, dr=dr, dt=dr / 4.0)
    rcs = [3.0 * p["half_width"], r_lump(p["half_width"], p["kappa"])]
    t0 = time.time()
    r = run(beta, w0, T, grid=g, rcs=rcs, prof=p, stop_at_death=False)
    return dict(beta=beta, alpha2=a2, omega=w0, scale=scale, dr=dr, T=T, wall=time.time() - t0,
                t=r["series"]["t"], frac=r["series"]["E_in"][:, 0] / r["E0"],
                E_lump=r["series"]["E_in"][:, 1], omega_t=r["omega_t"], omega_s=r["omega"],
                grid=g, T_death=r["T_death"], T_90=r["T_90"])


CHECK_A2 = 0.07
T_CHECK = 3000.0


def stage_checks(workers):
    jobs = [(0.75, CHECK_A2, 1.0, 0.05, T_CHECK), (0.75, CHECK_A2, 2.0, 0.05, T_CHECK),
            (5.0, CHECK_A2, 1.0, 0.05, T_CHECK), (5.0, CHECK_A2, 2.0, 0.05, T_CHECK),
            (5.0, CHECK_A2, 1.0, 0.025, 2000.0)]
    res = _pool(workers, _box_job, jobs)
    out = {}
    for b in (0.75, 5.0):
        a = [x for x in res if x["beta"] == b and x["scale"] == 1.0 and x["dr"] == 0.05][0]
        d = [x for x in res if x["beta"] == b and x["scale"] == 2.0][0]
        n = min(a["t"].size, d["t"].size)
        diff = np.abs(a["frac"][:n] - d["frac"][:n])
        out["box_doubling_beta%.4f" % b] = dict(
            alpha2=a["alpha2"], omega=a["omega"], T=float(a["t"][n - 1]),
            grids=[a["grid"], d["grid"]], max_abs_diff_frac=float(diff.max()),
            t_at_max=float(a["t"][int(diff.argmax())]), target=1e-4,
            max_abs_diff_frac_while_alive=float(diff[a["t"][:n] <= (a["T_death"] or 1e99)].max()),
            max_abs_diff_by_window={"%d-%d" % (lo, lo + 500): float(diff[(a["t"][:n] >= lo) & (a["t"][:n] < lo + 500)].max())
                                    for lo in range(0, int(a["t"][n - 1]), 500)},
            frac_by_window_box1={"%d" % T: float(a["frac"][int(np.argmin(np.abs(a["t"] - T)))]) for T in range(500, 3001, 500)},
            frac_end=[float(a["frac"][n - 1]), float(d["frac"][n - 1])],
            T_death=[a["T_death"], d["T_death"]])
    a = [x for x in res if x["beta"] == 5.0 and x["scale"] == 1.0 and x["dr"] == 0.05][0]
    h = [x for x in res if x["dr"] == 0.025][0]
    n = min(h["t"].size, a["t"].size)
    st = rate.map_settings(5.0) if fam.family_path(5.0) else None
    t_s = st["t_s"] if st else 250.0
    wa = window_fits(a["t"], a["E_lump"], t_s, rate.W, 2000.0, a["omega_t"], a["omega_s"])
    wh = window_fits(h["t"], h["E_lump"], t_s, rate.W, 2000.0, h["omega_t"], h["omega_s"])
    out["dr_halving_beta5.0000"] = dict(
        alpha2=CHECK_A2, omega=a["omega"], T=2000.0, dr=[0.05, 0.025], dt=[0.0125, 0.00625],
        max_abs_diff_frac=float(np.max(np.abs(a["frac"][:n] - h["frac"][:n]))),
        frac_t0=[float(a["frac"][0]), float(h["frac"][0])],
        frac_T2000=[float(a["frac"][n - 1]), float(h["frac"][n - 1])],
        Gamma_windows=[[x["Gamma"] for x in wa], [x["Gamma"] for x in wh]],
        sigma_windows=[[x["sigma"] for x in wa], [x["sigma"] for x in wh]],
        omega_settled=[float(np.median(a["omega_s"][(a["omega_t"] > 50) & (a["omega_t"] < 150)])),
                       float(np.median(h["omega_s"][(h["omega_t"] > 50) & (h["omega_t"] < 150)]))],
        note="frac difference includes the O(dr^2) E_in quadrature error (F-003)")
    out["walls"] = [dict(beta=x["beta"], scale=x["scale"], dr=x["dr"], T=x["T"], wall=x["wall"]) for x in res]
    write_json(os.path.join(OUT, "checks.json"), out)
    _log(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("max_abs_diff_frac", "Gamma_windows", "T_death")}
                     for k, v in out.items() if k != "walls"}, default=str))


def _cost_job(beta):
    g = grid_for(beta)
    w0 = rate.a2_to_w(beta, CHECK_A2)
    p = profile(beta, w0)
    run(beta, w0, 10.0, grid=g, prof=p)          # JIT warm-up
    t0 = time.time()
    run(beta, w0, 200.0, grid=g, prof=p, stop_at_death=False)
    return beta, (time.time() - t0) / 200.0


def stage_cost(betas, workers_assumed=2, workers=1, n_dips=2):
    res = dict(_pool(workers, _cost_job, betas))
    rows, tot_maps, tot_long = [], 0.0, 0.0
    for b in betas:
        c = res[b]
        n = len(rate.coarse_grid(b)[0]) + 4 * n_dips
        cm = n * (rate.T_MAP + rate.T_TWIN_MAP) * c
        cl = (T_LONG + T_TWIN_LONG) * c
        tot_maps += cm
        tot_long += cl
        rows.append(dict(beta=b, R_box=grid_for(b)["R_box"], n_grid=int(round(grid_for(b)["R_box"] / 0.05)) + 1,
                         wall_s_per_T=c, map_runs_est=n, map_wall_s=cm, long_wall_s=cl))
    longest = max(r["long_wall_s"] for r in rows)
    est = dict(maps_core_s=tot_maps, long_core_s=tot_long,
               wall_h_on_cores=(tot_maps + tot_long) / workers_assumed / 3600.0,
               cores=workers_assumed, longest_single_long_run_h=longest / 3600.0,
               note="upper bound: long runs assumed to reach T = 30000 (death stops earlier); "
                    "map runs assumed to reach T_map (death stops earlier); %d dips per beta "
                    "for refinement; measured with one process while other processes (a parallel worker) used about 4 of the 10 cores; the validation maps ran at 2.5e-3 core-s per unit T effective" % n_dips)
    write_json(os.path.join(OUT, "cost.json"), dict(rows=rows, estimate=est))
    for r in rows:
        _log("cost beta=%.4f R_box=%.0f wall/T=%.2e s map %.0fs long %.0fs"
             % (r["beta"], r["R_box"], r["wall_s_per_T"], r["map_wall_s"], r["long_wall_s"]))
    _log("estimate: %s" % est)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["families", "maps", "rank", "long", "validate", "checks", "cost"])
    ap.add_argument("--betas", type=float, nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--T", type=float, default=T_LONG)
    ap.add_argument("--omega", type=float, default=None, help="long: run this omega (one beta)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    fix = lambda bs: [4.0 / 3.0 if abs(b - 4.0 / 3.0) < 1e-3 else b for b in bs]
    betas = fix(a.betas) if a.betas else (BETAS_NEW if a.stage == "families" else BETAS_ALL)
    t0 = time.time()
    if a.stage == "families":
        stage_families(betas, a.workers, a.force)
    elif a.stage == "maps":
        stage_maps(betas, a.workers, a.force)
    elif a.stage == "rank":
        stage_rank(betas)
    elif a.stage == "long":
        stage_long(betas, a.workers, a.T, a.omega)
    elif a.stage == "validate":
        stage_validate(a.workers)
    elif a.stage == "checks":
        stage_checks(a.workers)
    elif a.stage == "cost":
        stage_cost(betas, workers=1)
    _log("stage %s wall %.0f s" % (a.stage, time.time() - t0))


if __name__ == "__main__":
    main()
