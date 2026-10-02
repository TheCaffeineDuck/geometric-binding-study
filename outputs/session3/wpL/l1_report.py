"""WP-L1 report: scan_summary.md and plots from scan_summary.json (+ run files for plot 1)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "scan_summary.json")))
SER = json.load(open(os.path.join(HERE, "scan_series_decimated.json")))
ROWS = D["rows"]
COL = ["#1f5f8b", "#c4572e", "#3a8a4f", "#7a4fa0", "#8a7a1f", "#555555", "#b03060"]
START_COL = {"dip": COL[0], "primary": COL[1], "b5": COL[2]}


def style(ax):
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fmt(x, f="%.4g"):
    return "-" if x is None else f % x


def sel(T, start, kind, scale=1.0):
    return sorted([r for r in ROWS if r["T"] == T and r["start"] == start and r["kind"] == kind and r["scale"] == scale],
                  key=lambda r: (r["packet"] or "", r["amp"]))


# ------------------------------------------------------------------ plot 1: frac against t
def plot_frac():
    fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for j, st in enumerate(("dip", "primary", "b5")):
        for i, (kind, vals) in enumerate((("field", [-0.3, -0.25, -0.2, -0.1, -0.02, 0.1, 0.5]),
                                          ("removal", [0.1, 0.2, 0.3, 0.35, 0.4]))):
            ax = axs[i, j]
            for c, a in enumerate(vals):
                r = [x for x in sel(2000.0, st, kind) if abs(x["amp"] - a) < 1e-9]
                if not r:
                    continue
                r = json.load(open(os.path.join(HERE, "runs", r[0]["file"])))
                t = np.asarray(r["series"]["t"]) - r["t_d"]
                ax.plot(t, r["series"]["frac"], color=COL[c % len(COL)], lw=1,
                        label=("eps = %+g" if kind == "field" else "s = %g") % a)
            ax.axhline(0.5, color="k", lw=0.6, ls=":")
            ax.set_xlim(-20, 700)
            ax.set_title("%s, %s" % (st, "field kick" if kind == "field" else "core removal"), fontsize=9)
            ax.legend(fontsize=6.5, frameon=False)
            style(ax)
            if i == 1:
                ax.set_xlabel("t - t_d")
            if j == 0:
                ax.set_ylabel("E(r < 3 hw) / value just before the kick")
    fig.suptitle("WP-L1: energy inside 3 hw after the disturbance (T = 2000 scan; dotted: 0.5)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "l1_frac_vs_t.png"), dpi=130)


# ------------------------------------------------------------------ plot 2: thresholds
def plot_thresholds():
    fig, ax = plt.subplots(figsize=(8, 5))
    bis = {(b["start"], b["kind"], b["side"]): b for b in (D.get("bisection") or [])}
    labels, y = [], 0
    for st in ("dip", "primary", "b5"):
        for kind, side, name in (("field", "neg", "field -"), ("velocity", "neg", "velocity -"),
                                 ("removal", "pos", "removal"), ("field", "pos", "field +")):
            b = bis.get((st, kind, side))
            if b is None:
                continue
            gb = [x for x in D["brackets"] if x["start"] == st and x["kind"] == kind and x["side"] == side][0]["all_breaks"]
            if gb.get("unbroken_t0_dN_over_N") is not None:
                ax.plot([gb["unbroken_t0_dN_over_N"], gb["broken_t0_dN_over_N"]], [y, y], "-", color=START_COL[st], lw=1, alpha=0.5)
            lo, hi = b["unbroken_t0_dN_over_N"], b["broken_t0_dN_over_N"]
            ax.plot([lo, hi], [y, y], "-", color=START_COL[st], lw=5, solid_capstyle="butt")
            ax.plot([lo], [y], "o", color=START_COL[st], ms=4, mfc="white")
            labels.append("%s %s" % (st, name))
            y += 1
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_xlim(-0.6, 3.5)
    ax.set_xlabel("imposed Delta N / N at t = 0 (bar: last unbroken (open) to first broken; T = 2000, bisected)")
    ax.axvline(0, color="k", lw=0.5)
    style(ax)
    ax.set_title("WP-L1: outright-break thresholds", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "l1_thresholds.png"), dpi=130)


# ------------------------------------------------------------------ plot 3: rate after against dN/N + family
def plot_rates():
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    lt = json.load(open("/Users/aaronramos/Desktop/SWIRL/outputs/session2/wpA/lifetime_table_v2.json"))["rows"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, st in zip(axs, ("dip", "primary", "b5")):
        nm = D["N_min"][st]["N_over_Nmin_table"]
        beta = 5.0 if st == "b5" else 4.0 / 3
        fam = [r for r in lt if abs(r["beta"] - beta) < 1e-3 and r.get("Gamma_last_window") is not None
               and r["Gamma_last_window"] > 0]
        ax.plot([r["N_over_Nmin"] / nm - 1 for r in fam], [r["Gamma_last_window"] for r in fam], "k^", ms=6, mfc="none",
                label="Session 2 family (last 1000-unit window, t about 29,000 to 30,000)")
        for kind, mk, c in (("field", "o", COL[0]), ("velocity", "s", COL[1]), ("removal", "D", COL[2]), ("packet", "x", COL[3])):
            rr = [r for r in ROWS if r["T"] == 30000.0 and r["start"] == st and r["kind"] == kind and r["scale"] in (1.0, 2.25, 1.25)
                  and r.get("Gamma_500_2000")]
            xs = [r["dN_over_N_settled"] for r in rr if r["Gamma_500_2000"] > 0]
            ax.plot(xs, [r["Gamma_500_2000"] for r in rr if r["Gamma_500_2000"] > 0], mk, color=c, ms=5, mfc="none",
                    label=kind + " (t_d+500..2000)")
            rl = [r for r in rr if r.get("Gamma_late_alive_mean") and r["Gamma_late_alive_mean"] > 0]
            ax.plot([r["dN_over_N_settled"] for r in rl], [r["Gamma_late_alive_mean"] for r in rl], mk, color=c, ms=3)
        c0 = [r for r in ROWS if False]
        ax.set_yscale("log")
        ax.set_ylim(1e-12, 3e-4)
        ax.set_xlim(-0.6, 0.5)
        ax.set_title("%s (start N/N_min = %.3g)" % (st, nm), fontsize=9)
        ax.set_xlabel("settled Delta N / N (period-averaged, t_d+750..2000)")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        style(ax)
    axs[0].set_ylabel("loss rate of E(r < 3 hw)\nopen: t_d+500..2000; small filled: late alive windows", fontsize=8)
    fig.suptitle("WP-L1: rate after the disturbance against the settled Delta N / N, with the Session 2 family overlaid", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "l1_rate_vs_dN.png"), dpi=130)


# ------------------------------------------------------------------ plot 4: death times, T = 30,000
def plot_death():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for st in ("dip", "primary"):
        for kind, mk in (("field", "o"), ("velocity", "s"), ("removal", "D"), ("packet", "x")):
            rr = [r for r in ROWS if r["T"] == 30000.0 and r["start"] == st and r["kind"] == kind and r["scale"] in (1.0, 2.25)
                  and r["death_time_full"]]
            ax.plot([r["dN_over_N_settled"] for r in rr], [r["death_time_full"] for r in rr], mk, color=START_COL[st],
                    mfc="none", label="%s %s" % (st, kind))
    for st in ("primary",):
        cc = [r for r in ROWS if r["start"] == st and r["T"] == 30000.0 and r["death_time_control_full"]]
        if cc:
            ax.axhline(cc[0]["death_time_control_full"], color=START_COL[st], ls=":", lw=1)
    ax.set_xlabel("settled Delta N / N")
    ax.set_ylabel("time after disturbance when E(r < 3 hw) falls below 0.5")
    ax.legend(fontsize=6.5, frameon=False)
    style(ax)
    ax.set_title("WP-L1: late deaths in the T = 30,000 runs (dotted: primary control)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "l1_late_deaths.png"), dpi=130)


# ------------------------------------------------------------------ markdown
def write_md():
    L = ["# WP-L1 scan summary", "",
         "One row per run is in `scan_summary.json` (full parameters in each row). This table is a short view. T = 2000 grid: %d runs; "
         "T = 30,000 grid: %d runs; linear twins: %d; doubled-box runs: %d." % (
             sum(1 for r in ROWS if r["T"] == 2000.0), sum(1 for r in ROWS if r["T"] == 30000.0 and r["scale"] != 2.0),
             len(D["linear_twins"]), sum(1 for r in ROWS if r["scale"] == 2.0)), "",
         "dN/N values: t0 = imposed (instantaneous N at the disturbance phase); settled = period-averaged N of the disturbed run over the "
         "control's, window t_d+750..2000. Rates are -d ln E(r<3hw)/dt (log fit of E3, t_d+500..2000; late = mean of the 500-wide windows "
         "after t_d+2000 while frac > 0.5). A break is `instantaneous` if E(r<3hw) < 0.5 E_before within 7 time units of the kick, else `dynamic`.", "",
         "## Outright-break thresholds (T = 2000), grid bracket and bisected bracket", "",
         "| start | disturbance | grid bracket (amp) | bisected unbroken | bisected broken | dN/N unbroken | dN/N broken | class at first break | time of first break |",
         "|---|---|---|---|---|---|---|---|---|"]
    bis = {(b["start"], b["kind"], b["side"]): b for b in (D.get("bisection") or [])}
    for b in D["brackets"]:
        if b["kind"] == "packet" or b["kind"] == "velocity" and b["side"] == "pos":
            continue
        a = b["all_breaks"]
        bb = bis.get((b["start"], b["kind"], b["side"]))
        gs = "no break to %s" % a["max_amp"] if a["status"] != "bracket" else "%s .. %s" % (a["unbroken_amp"], a["broken_amp"])
        L.append("| %s | %s %s | %s | %s | %s | %s | %s | %s | %s |" % (
            b["start"], b["kind"], b["side"], gs, fmt(bb and bb["unbroken_amp"]), fmt(bb and bb["broken_amp"]),
            fmt(bb and bb["unbroken_t0_dN_over_N"], "%.4f"), fmt(bb and bb["broken_t0_dN_over_N"], "%.4f"),
            bb["broken_class"] if bb else "-", fmt(bb and bb["broken_time"], "%.1f")))
    L += ["", "Grid brackets split by class (all breaks / dynamic only) are in `brackets` of the JSON.", "",
          "Packets (both wavenumbers, efrac up to 2) and positive velocity kicks up to +1.0 never broke outright within 2000.", ""]
    for st in ("dip", "primary", "b5"):
        L += ["## T = 30,000 runs, %s" % st, "",
              "| disturbance | amp | dN/N t0 | dN/N settled | domega | Gamma t_d+500..2000 (packets from t_pass+200) | control | late Gamma (alive) | control late | frac at 30k | death (t-t_d) | family N/Nmin (eff -> nearest) | family Gamma vs run, same elapsed window (the run's last alive window) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted([r for r in ROWS if r["T"] == 30000.0 and r["start"] == st and r["scale"] in (1.0, 2.25, 1.25)],
                        key=lambda r: (r["kind"], r["packet"] or "", r["amp"])):
            f = r.get("family")
            L.append("| %s | %g | %s | %s | %s | %s | %s | %s | %s | %.3g | %s | %s | %s |" % (
                r["kind"] + (" " + r["packet"] if r["packet"] else ""), r["amp"], fmt(r["t0_dN_over_N"], "%.4f"),
                fmt(r["dN_over_N_settled"], "%.4f"), fmt(r["domega_after"], "%.5f"), fmt(r["Gamma_500_2000"], "%.2e"),
                fmt(r["Gamma_500_2000_control"], "%.2e"), fmt(r.get("Gamma_late_alive_mean"), "%.2e"),
                fmt(r.get("Gamma_late_alive_mean_control"), "%.2e"), r["frac_end"], fmt(r["death_time_full"], "%.0f"),
                ("%.3f -> %.3f%s" % (f["N_over_Nmin_eff"], f["N_over_Nmin"], "" if f["comparable"] else " (not within 1%)")) if f else "-",
                ("%s vs run %s" % (fmt(f["Gamma_family_same_elapsed"], "%.1e"), fmt(f["Gamma_run_same_window"], "%.1e"))) if f else "-"))
        L.append("")
    L += ["## Linear twins (frac = E(r<3hw) over its value before the kick)", "",
          "| start | disturbance | amp | frac at t_d+100 | +300 | +500 | +1000 |", "|---|---|---|---|---|---|---|"]
    for t in D["linear_twins"]:
        L.append("| %s | %s | %g | %s | %s | %s | %s |" % (t["start"], t["kind"] + (" " + t["packet"] if t["packet"] else ""),
                                                         t["amp"], fmt(t["frac_at_100"], "%.4f"), fmt(t["frac_at_300"], "%.4f"),
                                                         fmt(t["frac_at_500"], "%.4f"), fmt(t["frac_at_1000"], "%.4f")))
    L += ["", "## Doubled box (T = 30,000, scale 2 against scale 1)", ""]
    for x in D["doubled_box"]:
        L.append("- %s %s %g: t0 dN/N %s; settled dN/N %s; Gamma(500..2000) %s; death time %s; frac at 30k %s" % (
            x["start"], x["kind"], x["amp"], x["t0_dN_over_N"], x["dN_over_N_settled"], x["Gamma_500_2000"], x["death_time_full"], x["frac_end"]))
    L.append("")
    open(os.path.join(HERE, "scan_summary.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    plot_frac()
    plot_thresholds()
    plot_rates()
    plot_death()
    write_md()
