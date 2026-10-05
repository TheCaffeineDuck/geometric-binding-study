"""F-026 diagnostic: finer efrac grid for the k3 packet at the dip and primary starts (T = 2000, scale 1),
|dN/N| after the packet in three windows, and the primary/dip ratio. Existing runs (efrac 0.01 ... 2) are read for
the neighbouring points and as a check against the ledger's ratios."""
import glob
import json
import os
from dcommon import HERE, RUNS, load_ctrl, write_json, wpl

K0 = {"dip": 2.530209420127502, "primary": 2.539136129431606}
NEW = [0.03, 0.04, 0.07, 0.15, 0.3]
OLD = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def spec(st, ef):
    return dict(start=st, kind="packet", packet=dict(k0=K0[st], sigma=5.0, efrac=ef), T=2000.0, scale=1.0)


def measure(st, ef, res, ctrl, tp):
    wins = ((250, 750), (750, 2000), (tp + 200, 2000))
    c = wpl.compare(res, ctrl, windows=wins)
    d = {tuple(w["window"]): w for w in c["windows"]}
    return dict(start=st, efrac=ef, t_pass=tp, broken=c["broken"]["broken_outright"],
                dN_250_750=d[(250, 750)]["dN_over_N"], dN_750_2000=d[(750, 2000)]["dN_over_N"],
                dN_after_pass=d[(tp + 200, 2000)]["dN_over_N"], domega_750_2000=d[(750, 2000)]["domega"])


if __name__ == "__main__":
    rows = []
    for st in ("dip", "primary"):
        ctrl = load_ctrl(st)
        tp = wpl.free_packet(st, dict(k0=K0[st], sigma=5.0, efrac=0.05), 1.0)["t_pass"]
        for ef in sorted(OLD + NEW):
            sp = spec(st, ef)
            f = os.path.join(RUNS, wpl.spec_name(sp) + ".json")
            new = ef in NEW
            if new:
                res = wpl.run_one(sp)
            elif os.path.exists(f):
                res = json.load(open(f))
            else:
                continue
            m = measure(st, ef, res, ctrl, tp)
            m["source"] = "new run" if new else "existing run file"
            m["wall_s"] = res.get("wall_s")
            rows.append(m)
            print(m, flush=True)
            write_json(os.path.join(HERE, "d026_partial.json"), rows)
    out = []
    for ef in sorted(OLD + NEW):
        d = next((r for r in rows if r["start"] == "dip" and r["efrac"] == ef), None)
        p = next((r for r in rows if r["start"] == "primary" and r["efrac"] == ef), None)
        if d and p:
            row = dict(efrac=ef)
            for k in ("dN_250_750", "dN_750_2000", "dN_after_pass"):
                row["ratio_" + k] = abs(p[k]) / abs(d[k]) if d[k] and p[k] is not None else None
            out.append(row)
    write_json(os.path.join(HERE, "d026_partial.json"), dict(rows=rows, ratios=out))
