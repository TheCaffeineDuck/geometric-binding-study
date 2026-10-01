"""WP-A supplement: S6 (extended horizon + doubled box at 4/3), S5 (matched N/Nmin), S3 (grid edge),
S4 (two-point dip midpoints), S1/S2 (map-based safeguards). Same driver functions as the sweep.
Run from the repo root:  python outputs/session2/wpA/supp.py [--workers 3]
Resumable: finished outputs are skipped, long runs continue from their checkpoints.
Deviation from wpa._long_job: rcs[1] (the lump radius) is max(map R_lump, r_lump(hw, kappa)) so that wide
near-thin-wall starts are not given an R_lump smaller than their own 3 hw."""
import os
os.environ["NUMBA_NUM_THREADS"] = "1"
import sys, json, glob, time, argparse
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL")
from scipy.optimize import brentq
from swirl.family import wpa_family as fam, wpa_rate as rate
from swirl.family.wpa_run import grid_for, profile, r_lump, run, window_fits, write_json
OUT = "/Users/aaronramos/Desktop/SWIRL/outputs/session2/wpA"
CK = OUT + "/checkpoints"
BETAS = [0.75, 0.85, 1.0, 1.15, 4/3, 1.6, 2.0, 3.0, 5.0]
T_LONG, T_EXT, T_TWIN = 30000.0, 300000.0, 1000.0
S6 = [(4/3, 0.906505), (1.6, 0.934976), (2.0, 0.959603), (3.0, 0.986274), (5.0, 0.991190), (3.0, 0.958855), (5.0, 0.975598)]
def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)
def tag(b, w): return "beta%.4f_w%.6f" % (b, w)
def fn_long(b, w): return "%s/long_%s.json" % (OUT, tag(b, w))

def existing():
    ex = []
    for f in glob.glob(OUT + "/long_beta*_w*.json"):
        if "_T300k" in f: continue
        b = float(os.path.basename(f).split("_")[1][4:]); w = float(os.path.basename(f).split("_w")[1][:-5])
        ex.append((b, w))
    return ex
def is_dup(b, w, lst, tol=2e-6): return any(abs(b - x) < 1e-3 and abs(w - y) < tol for x, y in lst)

def omega_for_NNmin(b, target):
    fj = json.load(open(fam.family_path(b)))
    Nmin = fj["features_rescaled"]["N_min"]; wmin = fj["features_rescaled"]["omega_minN"]
    rows = sorted([(r["omega"], r["N"] / Nmin) for r in fj["rows_rescaled"] if r["omega"] < wmin - 1e-4])
    f = lambda w: profile(b, w)["N"] / Nmin - target
    ws = [r[0] for r in rows]; Ns = [r[1] / Nmin for r in rows]
    lo = hi = None
    for (w1, n1), (w2, n2) in zip(rows[:-1], rows[1:]):   # N decreases with omega on this branch
        if (n1 - target) * (n2 - target) <= 0: lo, hi = w1, w2; break
    if lo is None: return None
    w = brentq(f, lo, hi, xtol=1e-9)
    return float(w)

def build_jobs():
    ex = existing(); jobs = []   # dict(kind, beta, omega, groups[])
    def add(kind, b, w, g, T=T_LONG):
        for j in jobs:
            if j["kind"] == kind and abs(j["beta"] - b) < 1e-3 and abs(j["omega"] - w) < 2e-6:
                if g not in j["groups"]: j["groups"].append(g)
                return
        if kind == "long" and is_dup(b, w, ex):
            jobs.append(dict(kind="have", beta=b, omega=w, groups=[g], T=T)); return
        jobs.append(dict(kind=kind, beta=b, omega=w, groups=[g], T=T))
    add("box2", 4/3, 0.906505, "S6")
    for b, w in sorted(S6, key=lambda x: -x[0]): add("ext", b, w, "S6", T_EXT)
    notes = []
    for b in BETAS:
        for tgt in (1.3, 1.8, 2.5):
            w = omega_for_NNmin(b, tgt)
            if w is None: notes.append("S5 beta %.4f N/Nmin %.1f: no bracket on family" % (b, tgt)); continue
            add("long", b, w, "S5_N%.1f" % tgt)
    for b in (0.75, 0.85, 1.0):
        for a2 in (0.144, 0.148, 0.152, 0.156, 0.160):
            w = rate.a2_to_w(b, a2)
            try: profile(b, w)
            except Exception as e: notes.append("S3 beta %.2f a2 %.3f: profile failed (%s), skipped" % (b, a2, str(e)[:80])); continue
            add("long", b, w, "S3_a2_%.3f" % a2)
    for b, a2 in ((1.15, 0.0805), (1.6, 0.1161), (2.0, 0.1198)):
        add("long", b, rate.a2_to_w(b, a2), "S4_a2_%.4f" % a2)
    for b in BETAS:
        rm = json.load(open("%s/ratemap_beta%.4f.json" % (OUT, b)))
        rs = rm["runs"]
        def gmin(r): return min([max(w["Gamma"], 2 * w["sigma"]) for w in r["windows"]]) if r["windows"] else None
        have = lambda r: is_dup(b, r["omega_profile"], ex)
        lo = sorted([r for r in rs if gmin(r) is not None and not have(r)], key=gmin)[:3]
        for r in lo: add("long", b, r["omega_profile"], "S1_lowGamma")
        dead = [r for r in rs if r["T_death"] is not None and not have(r)]
        if dead: add("long", b, max(dead, key=lambda r: r["T_death"])["omega_profile"], "S2_longest_map_death")
        surv = sorted([r for r in rs if r["T_death"] is None and r["T_end"] >= 1999 and not have(r)], key=lambda r: -r["frac_end"])[:2]
        for r in surv: add("long", b, r["omega_profile"], "S2_top_frac2000")
    return jobs, notes

def _res_common(r, b, w, p, rcs, T, twin, stride, frac_times):
    t = r["series"]["t"]; El = r["series"]["E_in"][:, 1]
    rl = rcs[1]; t_s = 150.0 + 2.0 * rl
    i_s = int(np.argmin(np.abs(t - t_s)))
    wins = window_fits(t, El, t_s, 1000.0, float(t[-1]), r["omega_t"], r["omega"]) if t[-1] > t_s else []
    keep = slice(None, None, stride)
    om_at = {str(T_): float(np.interp(T_, r["omega_t"], r["omega"])) for T_ in frac_times if T_ <= t[-1] + 0.5}
    return dict(
        description="WP-A supplement long radial run (same driver as the sweep): death or T, whichever first. frac = E(r<3hw)/E0. "
                    "osc_* = T omega_settled/(2 pi). E_in quadrature error ~1e-4 of E (F-003).",
        params=r["params"], E0=r["E0"], regime=r["regime"], T_end=r["T_end"], T_90=r["T_90"], T_death=r["T_death"],
        osc_T_90=r["osc_T_90"], osc_T_death=r["osc_T_death"], osc_T_end=r["osc_T_end"],
        periods_counted_T_90=r["periods_counted_T_90"], periods_counted_T_death=r["periods_counted_T_death"],
        periods_counted_total=r["periods_counted_total"], frac_t0=r["frac_t0"], frac_end=r["frac_end"], E_ref50=r["E_ref50"],
        energy_check=r["energy_check"],
        frac_at={str(T_): float(r["series"]["E_in"][int(np.argmin(np.abs(t - T_))), 0] / r["E0"]) for T_ in frac_times if T_ <= t[-1] + 0.5},
        omega_at=om_at, t_s=t_s, E_s=float(El[i_s]), windows_1000=wins, linear_twin=twin,
        series=dict(t=t[keep], frac=r["series"]["E_in"][keep, 0] / r["E0"], E_lump_over_E0=El[keep] / r["E0"],
                    E_tot_over_E0=r["series"]["E_tot"][keep] / r["E0"]),
        omega_series=dict(t=r["omega_t"][::20], omega=r["omega"][::20]), nl_share_series=r["nl"])

def twin_for(b, w, p, rcs):
    tw = run(b, w, T_TWIN, linear=True, rcs=rcs, prof=p, stop_at_death=False)
    ft = tw["series"]["E_in"][:, 0] / tw["E0"]; tt = tw["series"]["t"]
    return dict(T=T_TWIN, frac_T100=float(ft[int(np.argmin(np.abs(tt - 100)))]), frac_T300=float(ft[int(np.argmin(np.abs(tt - 300)))]),
                frac_T1000=float(ft[-1]), E0=tw["E0"])

def job(j):
    try:
        b, w, kind = j["beta"], j["omega"], j["kind"]
        t0 = time.time()
        if kind == "have": return dict(j, status="have", file=os.path.basename(fn_long(b, w)))
        p = profile(b, w)
        if kind == "long":
            out_f = fn_long(b, w)
            if os.path.exists(out_f): return dict(j, status="done_before", file=os.path.basename(out_f))
            st = rate.map_settings(b)
            rcs = [3.0 * p["half_width"], max(st["R_lump"], r_lump(p["half_width"], p["kappa"]))]
            r = run(b, w, j["T"], rcs=rcs, prof=p, ckpt="%s/long_%s.npz" % (CK, tag(b, w)), ckpt_every=1000.0)
            out = _res_common(r, b, w, p, rcs, j["T"], twin_for(b, w, p, rcs), 10, (1000, 2000, 5000, 10000, 20000, 30000))
            out["groups"] = j["groups"]; out["wall_s"] = time.time() - t0
            write_json(out_f, out)
        elif kind == "ext":
            out_f = "%s/long_%s_T300k.json" % (OUT, tag(b, w))
            if os.path.exists(out_f): return dict(j, status="done_before", file=os.path.basename(out_f))
            old = "%s/long_%s.npz" % (CK, tag(b, w)); new = "%s/long_%s_T300k.npz" % (CK, tag(b, w))
            v1 = json.load(open(fn_long(b, w))); rcs = v1["params"]["rcs"]
            w = v1["params"]["omega_profile"]; p = profile(b, w)   # exact start omega (pkey must match)
            if not os.path.exists(new):
                z = np.load(old, allow_pickle=False)
                pk = json.loads(str(z["pkey"])); assert pk["T"] == T_LONG
                stt = json.loads(str(z["state"])); assert abs(stt["times"][-1] - T_LONG) < 1e-6 and not stt["done"], (stt["times"][-1], stt["done"])
                pk["T"] = T_EXT
                np.savez(new + ".tmp.npz", u=z["u"], v=z["v"], t=float(z["t"]), state=str(z["state"]), pkey=json.dumps(pk, sort_keys=True))
                os.replace(new + ".tmp.npz", new)
            r = run(b, w, T_EXT, rcs=rcs, prof=p, ckpt=new, ckpt_every=20000.0)
            out = _res_common(r, b, w, p, rcs, T_EXT, v1["linear_twin"], 100, (1000, 2000, 5000, 10000, 20000, 30000, 50000, 100000, 150000, 200000, 250000, 300000))
            out["groups"] = j["groups"]; out["wall_s"] = time.time() - t0
            out["note"] = "continued from the T = 30,000 checkpoint of long_%s.json (params identical except T); twin copied from it" % tag(b, w)
            write_json(out_f, out)
        elif kind == "box2":
            out_f = "%s/box2_%s_T300k.json" % (OUT, tag(b, w))
            if os.path.exists(out_f): return dict(j, status="done_before", file=os.path.basename(out_f))
            v1 = json.load(open(fn_long(b, w))); rcs = v1["params"]["rcs"]
            w = v1["params"]["omega_profile"]; p = profile(b, w)
            g = grid_for(b, scale=2.0)
            r = run(b, w, T_EXT, grid=g, rcs=rcs, prof=p, stop_at_death=False, ckpt="%s/box2_%s_T300k.npz" % (CK, tag(b, w)), ckpt_every=20000.0)
            t = r["series"]["t"]
            write_json(out_f, dict(beta=b, omega=w, grid=g, T_end=r["T_end"], E0=r["E0"], T_90=r["T_90"], T_death=r["T_death"],
                                   t=t[::10], frac=r["series"]["E_in"][::10, 0] / r["E0"], E_lump=r["series"]["E_in"][::10, 1], wall_s=time.time() - t0))
        log("done %s %s beta=%.4f w=%.6f groups=%s wall %.0fs" % (kind, "", b, w, ",".join(j["groups"]), time.time() - t0))
        return dict(j, status="ok", file=os.path.basename(out_f), wall_s=time.time() - t0)
    except Exception as e:
        import traceback; log("FAILED %s beta=%s w=%s: %s" % (j["kind"], j["beta"], j["omega"], traceback.format_exc()))
        return dict(j, status="failed", error=str(e))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=3); a = ap.parse_args()
    jobs, notes = build_jobs()
    for n in notes: log("NOTE " + n)
    log("jobs: %d (%s)" % (len(jobs), {k: sum(1 for j in jobs if j["kind"] == k) for k in ("ext", "box2", "long", "have")}))
    man = dict(jobs=jobs, notes=notes); write_json(OUT + "/supp_manifest.json", man)
    res = []
    with Pool(a.workers) as pl:
        for r in pl.imap_unordered(job, [j for j in jobs if j["kind"] != "have"], chunksize=1):
            res.append(r); man["results"] = res; write_json(OUT + "/supp_manifest.json", man)
    log("ALL DONE; failed: %d" % sum(1 for r in res if r["status"] == "failed"))
