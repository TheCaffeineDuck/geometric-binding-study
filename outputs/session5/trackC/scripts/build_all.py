"""Runs every analysis, writes summary.json, TABLES.md and the plots.
    python outputs/session5/trackC/scripts/build_all.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib                                       # noqa: E402
import an_grid as AG                             # noqa: E402
import an_smooth as AS                           # noqa: E402
import verdicts as V                             # noqa: E402
import verdicts2 as V2                           # noqa: E402
from verdicts import fmt                         # noqa: E402

TC = lib.TC


def md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def f(x, n=3):
    return fmt(x, n)


def cyc_or_cap(T, cycles, cap):
    return f(T) if not isinstance(T, str) else T


def lifetime_cells(r):
    T90 = r["T_90"]
    Td = r["T_death"]
    return [f(T90) if not isinstance(T90, str) else T90 + " (t=%s)" % f(r["cap_t"]),
            (f(r["T_90_cycles"]) if r["T_90_cycles"] is not None else "-"),
            f(Td) if not isinstance(Td, str) else Td + " (t=%s)" % f(r["cap_t"]),
            (f(r["T_death_cycles"]) if r["T_death_cycles"] is not None else "-")]


def main():
    SUM = {"note": "Every number in TABLES.md. Each entry carries its source file (relative to the repo root).",
           "generated_from": lib.REL(lib.RUNS)}
    verdicts = []

    # ---------------------------------------------------------------- batch A
    A1 = AG.a1() if lib.exists("A1_family_grid_G1_sbend") and lib.exists("A1_family_grid_G1_bump") else None
    A2, A2c = AG.a2(A1) if A1 else ([], {})
    A3 = AG.a3(A2) if A2 else []
    A4 = AG.a4()
    A5 = AG.a5()
    A6 = AG.a6()
    A7 = AG.a7()
    SUM["A1"] = {k: {kk: vv for kk, vv in v.items()} for k, v in (A1 or {}).items()}
    SUM["A2"] = A2
    SUM["A3"] = A3
    SUM["A4"] = A4
    SUM["A5"] = A5
    SUM["A6"] = A6
    SUM["A7"] = A7

    # ---------------------------------------------------------------- batch B
    brows, bc = AS.b_all()
    c6 = AS.c6_all(brows)
    sel = AS.selection(brows, c6)
    B5 = AS.b5()
    B6 = AS.b6()
    TF = AS.tighter_faster_smooth(B6) if B6["fixed_amplitude"] else {}
    B7 = AS.b_push()
    B8 = AS.b8(brows)
    O5 = AS.o5(brows, bc, sel)
    O6 = AS.o6(brows, bc)
    # frac at t = 2000 for O7
    o7 = []
    for tag in ("B1_bump_pk0.8_s0.5", "B2_sbend_pk0.8_s0.5"):
        for h, ht in AS.HS:
            nm = "%s_%s" % (tag, ht)
            if nm in bc:
                c, tw = bc[nm]
                o7.append({"tag": tag, "h": h, "source": lib.REL(lib.path(nm)),
                           "frac_nl": lib.val_at_t(c, "ff", 2000.0), "frac_tw": lib.val_at_t(tw, "ff", 2000.0) if tw else None})
    B9, B10 = [], []
    import glob as _g
    for fn in sorted(_g.glob(os.path.join(lib.RUNS, "B9_*_h8.json"))):
        nm = os.path.basename(fn)[:-5]
        if nm.endswith("_twin"):
            continue
        row, c, tw = lib.lifetime_row(nm, (1000, 5000))
        row["probe_spectrum_split"] = AS.probe_split(nm)
        row["tail_kappa_measured"] = lib.J(nm)["summary"]["tail_kappa_at_late_rate"]
        row["tail_kappa_O1"] = lib.J(nm)["summary"]["tail_kappa_O1_at_late_rate"]
        B9.append(row)
    for nm in ("B4_orbit_sbend_s0.5_pk0.8_h8", "B3_jump_sbend_w1.5_h8", "B3_jump_sbend_w2_h8"):
        if lib.exists(nm):
            B10.append(AS.b_surroundings(nm, 20.0))
    SUM["B9"] = B9
    SUM["B10"] = B10
    SUM["B_rows"] = brows
    SUM["B_C6"] = c6
    SUM["B_selection"] = sel
    SUM["B5_O3"] = B5
    SUM["B6"] = B6
    SUM["B6_spearman"] = TF
    SUM["B7_push"] = B7
    SUM["B8_scheme"] = B8
    SUM["O5"] = O5
    SUM["O6"] = O6
    SUM["O7_t2000"] = o7

    # ---------------------------------------------------------------- verdicts
    if A1:
        verdicts.append(V.v_C1(A1, A2))
    verdicts.append(V.v_C2())
    c3cases = []
    if A1 and A2:
        r, c3cases = V.v_C3(A1, A2)
        verdicts.append(r)
    verdicts.append(V.v_C4(A5))
    r, c5cases = V.v_C5(A6)
    verdicts.append(r)
    r, cnt = V.v_C6(c6)
    verdicts.append(r)
    if A1:
        verdicts += V2.v_O1(A1, A2, brows)
    verdicts.append(V2.v_O2(A7))
    verdicts.append(V2.v_O3(B5))
    verdicts.append(V2.v_O4(B6))
    verdicts.append(V2.v_O5(O5) if brows else V.row("O5", "-", "-", "B incomplete", "-", V.UND))
    verdicts.append(V2.v_O6(O6))
    verdicts.append(V2.v_O7(o7))
    verdicts.append(V2.v_O10(brows, c6))
    verdicts.append(V2.v_O11(brows))
    import expectations as EX
    ex, exdet = EX.expectations(A1, A2, A3, A4, A6, c5cases, brows, B7, c6, B6, TF)
    verdicts += ex
    SUM["C3_cases"] = c3cases
    SUM["expectations_detail"] = exdet
    SUM["verdicts"] = verdicts

    # ---------------------------------------------------------------- tables
    import tables as TB
    txt = TB.tables_md(SUM, B9, B10, A1, A2, A3, A4, A5, A6, A7, brows, c6, sel, B5, B6, TF, B7, B8, O5, O6, verdicts, c3cases, c5cases)
    open(os.path.join(TC, "TABLES.md"), "w").write(txt)
    json.dump(SUM, open(os.path.join(TC, "summary.json"), "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (bool(o) if isinstance(o, np.bool_) else str(o)))
    # ---------------------------------------------------------------- plots
    import plots as PL
    PL.all_plots(A1, A2, A2c, A5, A6, A7, brows, bc, B6, B7, c6)
    print("TABLES.md, summary.json and plots written; %d verdict rows" % len(verdicts))


if __name__ == "__main__":
    main()
