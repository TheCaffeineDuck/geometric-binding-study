"""WP-L1 bisection of the outright-break thresholds (T = 2000), driving wpl.scan with extra specs.
Brackets come from the T = 2000 grid runs in runs/. Stops when the bracket in t = 0 Delta N / N is <= 0.01.
Usage: NUMBA_NUM_THREADS=1 python l1_bisect.py [--workers 2] [--max-rounds 10]"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from swirl.family import wpl  # noqa: E402
from swirl.family.wpa_run import write_json  # noqa: E402

RD = os.path.join(HERE, "runs")
T = 2000.0
INSTANT = 7.0
CASES = [(st, k, side) for st in ("dip", "primary", "b5") for k, side in
         (("field", "neg"), ("velocity", "neg"), ("removal", "pos"), ("field", "pos"))]


def spec(st, k, a):
    return dict(start=st, kind=k, **({"s": a} if k == "removal" else {"eps": a}), T=T)


def load(sp):
    return json.load(open(os.path.join(RD, wpl.spec_name(sp) + ".json")))


def evaluate(sp, ctrl):
    r = load(sp)
    c = wpl.compare(r, ctrl)
    b = c["broken"]
    return dict(broken=b["broken_outright"], t_first=b["first_below_half"], dN=r["t0"]["dN_over_N"])


def grid_amps(st, k, side):
    amps = {"field": [-0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
            "removal": [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]}
    amps["velocity"] = amps["field"]
    a = amps[k]
    if k != "removal":
        a = [x for x in a if (x < 0) == (side == "neg")]
    return sorted(a, key=abs)


def main(workers, max_rounds):
    ctrl = load(dict(start="dip", kind="none", T=T))  # placeholder, replaced per start below
    ctrls = {st: load(dict(start=st, kind="none", T=T)) for st in ("dip", "primary", "b5")}
    state = {}
    for st, k, side in CASES:
        ev = [(a, evaluate(spec(st, k, a), ctrls[st])) for a in grid_amps(st, k, side)]
        brk = [(a, e) for a, e in ev if e["broken"]]
        if not brk:
            continue
        fa, fe = brk[0]
        lo = [(a, e) for a, e in ev if abs(a) < abs(fa) and not e["broken"]]
        if not lo:
            continue
        la, le = lo[-1]
        state[(st, k, side)] = dict(lo=la, lo_dN=le["dN"], hi=fa, hi_dN=fe["dN"], hi_t=fe["t_first"], hist=[])
    for rd in range(max_rounds):
        todo = {key: v for key, v in state.items() if abs(v["hi_dN"] - v["lo_dN"]) > 0.01}
        if not todo:
            break
        specs = [spec(st, k, round(0.5 * (v["lo"] + v["hi"]), 6)) for (st, k, side), v in todo.items()]
        print("round", rd, len(specs), flush=True)
        wpl.scan(specs, workers=workers)
        for (st, k, side), v in todo.items():
            a = round(0.5 * (v["lo"] + v["hi"]), 6)
            e = evaluate(spec(st, k, a), ctrls[st])
            v["hist"].append(dict(amp=a, **e))
            if e["broken"]:
                v["hi"], v["hi_dN"], v["hi_t"] = a, e["dN"], e["t_first"]
            else:
                v["lo"], v["lo_dN"] = a, e["dN"]
    out = []
    for (st, k, side), v in state.items():
        out.append(dict(start=st, kind=k, side=side, unbroken_amp=v["lo"], unbroken_t0_dN_over_N=v["lo_dN"],
                        broken_amp=v["hi"], broken_t0_dN_over_N=v["hi_dN"], broken_time=v["hi_t"],
                        broken_class="instantaneous" if v["hi_t"] is not None and v["hi_t"] <= INSTANT else "dynamic",
                        width_dN_over_N=abs(v["hi_dN"] - v["lo_dN"]), history=v["hist"], T=T))
    write_json(os.path.join(HERE, "bisection.json"), dict(description="WP-L1 bisection of outright-break thresholds, T = 2000", results=out))
    for o in out:
        print(o["start"], o["kind"], o["side"], o["unbroken_amp"], o["broken_amp"], round(o["unbroken_t0_dN_over_N"], 4),
              round(o["broken_t0_dN_over_N"], 4), o["broken_class"], len(o["history"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-rounds", type=int, default=10)
    a = ap.parse_args()
    os.environ["NUMBA_NUM_THREADS"] = "1"
    main(a.workers, a.max_rounds)
