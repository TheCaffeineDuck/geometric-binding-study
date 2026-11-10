"""Plots for WP-C (PNG, outputs/session5/trackC/plots/)."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lib
import an_smooth as AS

PL = os.path.join(lib.TC, "plots")
HC = {"h4": "#1f77b4", "h8": "#ff7f0e", "h16": "#2ca02c"}
HL = {"h4": "h = 1/4", "h8": "h = 1/8", "h16": "h = 1/16"}
OC = {"sbend": "#1f77b4", "bump": "#d62728"}


def save(fig, name):
    os.makedirs(PL, exist_ok=True)
    fig.savefig(os.path.join(PL, name), dpi=110, bbox_inches="tight")
    plt.close(fig)


def p_family(A1):
    if not A1:
        return
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    for obj, d in A1.items():
        r = d["rows"]
        om = [x["omega"] for x in r]
        ax[0, 0].plot([x["peak_stretch"] for x in r], om, "o-", ms=3, color=OC[obj], label=obj)
        ax[0, 1].plot(om, [x["E"] for x in r], "o-", ms=3, color=OC[obj], label=obj)
        ax[1, 0].plot(om, [x["E_per_cycle"] for x in r], "o-", ms=3, color=OC[obj], label=obj)
        ax[1, 1].plot(om, [x["L_half"] for x in r], "o-", ms=3, color=OC[obj], label=obj)
        ax[1, 1].plot(om, [x["rms_width"] / 8 for x in r], ":", color=OC[obj], alpha=0.6)
    ax[0, 0].set(xlabel="peak stretch", ylabel="rate"); ax[0, 0].axhline(2, color="k", lw=0.6, ls="--")
    ax[0, 1].set(xlabel="rate", ylabel="energy E")
    ax[1, 0].set(xlabel="rate", ylabel="energy per cycle 2 pi E / rate")
    ax[1, 1].set(xlabel="rate", ylabel="L_half (solid); rms width / 8 (dotted)")
    for a in ax.flat:
        a.grid(alpha=0.3); a.legend()
    fig.suptitle("C-grid, G1: families of exact periodic states (A1)")
    save(fig, "family_curves.png")
    # tail and Floquet
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for obj, d in A1.items():
        r = d["rows"]
        om = np.array([x["omega"] for x in r])
        ax[0].semilogy(om, np.abs(np.array([x["tail_kappa_measured"] / x["tail_kappa_O1"] - 1 if x["tail_kappa_measured"] else np.nan for x in r])) + 1e-12, "o-", ms=3, color=OC[obj], label=obj)
        ax[1].semilogy(om, np.array([x["floquet_max_modulus_excl_4_at_1"] - 1 for x in r]).clip(1e-16), "o-", ms=3, color=OC[obj], label=obj)
    ax[0].set(xlabel="rate", ylabel="|tail kappa / O1 - 1|"); ax[0].axhline(0.05, color="k", ls="--", lw=0.6)
    ax[1].set(xlabel="rate", ylabel="largest Floquet multiplier excl. the four, minus 1")
    for a in ax:
        a.grid(alpha=0.3); a.legend()
    save(fig, "family_tail_floquet.png")


def p_lifetimes(A2, A2c):
    if not A2c:
        return
    fig, axs = plt.subplots(2, 6, figsize=(21, 6.5), sharey=False)
    for i, obj in enumerate(("sbend", "bump")):
        for j, rate in enumerate(lib_rates()):
            nm = "A2_%s_w%g" % (obj, rate)
            a = axs[i, j]
            if nm not in A2c:
                a.set_title("%s %g (missing)" % (obj, rate)); continue
            c, tw = A2c[nm]
            a.plot(c["cyc"], c["ff"], color="#1f77b4", label="fixed window")
            a.plot(c["cyc"], c["fc"], color="#ff7f0e", ls="--", label="co-moving")
            if tw is not None:
                a.plot(np.interp(tw["t"], c["t"], c["cyc"]), tw["ff"], color="grey", label="linear twin")
            a.set(title="%s, rate %g" % (obj, rate), xlabel="cycles", ylim=(-0.05, 1.08))
            a.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7)
    axs[0, 0].set_ylabel("frac(t)"); axs[1, 0].set_ylabel("frac(t)")
    fig.suptitle("C-grid lifetime runs (A2) with linear twins; frac = cycle-averaged energy in the window over its t = 0 value")
    save(fig, "lifetimes_A2_frac.png")


def lib_rates():
    return [2.05, 2.2, 2.5, 3.0, 3.5, 4.2]


def p_bump_tracks(A2c):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cm = plt.cm.viridis(np.linspace(0, 0.9, 6))
    for k, rate in enumerate(lib_rates()):
        nm = "A2_bump_w%g" % rate
        if nm not in A2c:
            continue
        c, tw = A2c[nm]
        sh = c["cen"] - c["cen"][0]
        ax[0].plot(c["cyc"], sh, color=cm[k], label="rate %g" % rate)
        m = c["cyc"] <= 600
        ax[1].plot(c["cyc"][m], sh[m], color=cm[k], label="rate %g" % rate)
        ax[2].plot(c["cyc"], c["fc"], color=cm[k])
    ax[0].set(xlabel="cycles", ylabel="energy centroid shift (sites)", title="bump centroid, full run")
    ax[1].set(xlabel="cycles", title="first 600 cycles")
    ax[2].set(xlabel="cycles", ylabel="co-moving frac", ylim=(0.9, 1.01), title="co-moving frac")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    save(fig, "bump_centroid_tracks.png")


def p_push(A6, B7):
    if A6 and any(c["rows"] for c in A6):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
        mk = {"sbend2.1": "o", "sbend2.3": "s", "sbend3": "^", "bump2.3": "D"}
        for case in A6:
            key = "%s%g" % (case["obj"], case["rate"])
            rows = [r for r in case["rows"] if r["speed"] is not None]
            lam = [r["lambda"] for r in rows]
            ax[0].loglog(lam, [max(abs(r["speed"]), 1e-6) for r in rows], mk[key] + "-", color=OC[case["obj"]], alpha=0.5 + 0.15 * (case["rate"] > 2.2), label="%s %g" % (case["obj"], case["rate"]))
            mv = [r for r in rows if r["loss_per_distance"] is not None and r["distance"] > 1 and not r["guard"]]
            ax[1].loglog([abs(r["speed"]) for r in mv], [max(r["loss_per_distance"], 1e-14) for r in mv], mk[key] + "-", color=OC[case["obj"]], alpha=0.5 + 0.15 * (case["rate"] > 2.2), label="%s %g" % (case["obj"], case["rate"]))
            ax[2].semilogx(lam, [r["frac_comoving_end"] for r in rows], mk[key] + "-", color=OC[case["obj"]], alpha=0.5 + 0.15 * (case["rate"] > 2.2), label="%s %g" % (case["obj"], case["rate"]))
        ax[0].set(xlabel="push size lambda", ylabel="measured speed"); ax[0].axhline(1, color="k", ls="--", lw=0.6)
        ax[1].set(xlabel="measured speed", ylabel="energy lost per unit distance")
        ax[2].set(xlabel="push size lambda", ylabel="co-moving frac at end of run"); ax[2].axhline(0.5, color="k", ls="--", lw=0.6)
        for a in ax:
            a.grid(alpha=0.3, which="both"); a.legend(fontsize=7)
        fig.suptitle("C-grid push (A6)")
        save(fig, "push_grid_A6.png")
    if B7 and any(b["rows"] for b in B7):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
        for b in B7:
            rows = [r for r in b["rows"] if r["speed"] is not None]
            if not rows:
                continue
            lab = "%s %s" % (b["tag"], b["h"])
            ls = "-" if b["h"] == "h8" else "--"
            col = "#1f77b4" if b["tag"] == "gsbend" else "#d62728"
            ax[0].loglog([r["lambda"] for r in rows], [max(abs(r["speed"]), 1e-6) for r in rows], "o" + ls, color=col, label=lab)
            mv = [r for r in rows if r["loss_per_distance"] is not None and r["distance"] > 0.5]
            ax[1].loglog([abs(r["speed"]) for r in mv], [max(r["loss_per_distance"], 1e-14) for r in mv], "o" + ls, color=col, label=lab)
            ax[2].semilogx([r["lambda"] for r in rows], [r["jump_x_shift"] for r in rows], "o" + ls, color=col, label=lab)
        ax[0].set(xlabel="push size lambda", ylabel="measured speed")
        ax[1].set(xlabel="measured speed", ylabel="energy lost per unit distance")
        ax[2].set(xlabel="push size lambda", ylabel="shift of the largest jump (units)")
        for a in ax:
            a.grid(alpha=0.3, which="both"); a.legend(fontsize=7)
        fig.suptitle("C-smooth push (B7)")
        save(fig, "push_smooth_B7.png")


def p_smooth_frac(brows, bc):
    groups = {}
    for tag, cls, obj, par in AS.bstarts():
        if tag in brows:
            groups.setdefault(cls, []).append(tag)
    for cls, tags in groups.items():
        n = len(tags)
        fig, axs = plt.subplots(1, n, figsize=(4.6 * n, 3.8), squeeze=False)
        for a, tag in zip(axs[0], tags):
            for h, ht in AS.HS:
                k = "%s_%s" % (tag, ht)
                if k not in bc:
                    continue
                c, tw = bc[k]
                a.plot(c["t"], c["ff"], color=HC[ht], label="nonlinear, " + HL[ht])
                if tw is not None:
                    a.plot(tw["t"], tw["ff"], color=HC[ht], ls=":", alpha=0.8)
            a.set(title=tag.replace("B1_", "").replace("B2_", "").replace("B3_", "").replace("B4_", ""), xlabel="time", ylim=(-0.03, 1.05))
            a.grid(alpha=0.3)
        axs[0, 0].set_ylabel("frac(t), fixed window")
        axs[0, 0].legend(fontsize=6)
        fig.suptitle("%s: nonlinear (solid) and linear twin (dotted), three spacings" % cls, y=1.02)
        save(fig, "smooth_frac_%s.png" % cls.replace(" ", "_").replace(".", ""))


def p_profiles(brows):
    for tag, title in (("B1_bump_pk0.8_s0.5", "Gaussian bump sigma 0.5, peak stretch 0.8"), ("B3_jump_sbend_w1.5", "jump S-bend, rate 1.5"),
                       ("B2_sbend_pk0.8_s0.5", "Gaussian S-bend sigma 0.5, peak stretch 0.8")):
        fig, axs = plt.subplots(2, 2, figsize=(11, 7))
        done = False
        for i, t in enumerate((100.0, 1000.0)):
            for h, ht in AS.HS:
                z = AS.snap("%s_%s" % (tag, ht), t)
                if z is None:
                    continue
                done = True
                x, d = z["x"], z["d"]
                xb = x + 0.5 * h
                for jj, w in enumerate((3.0, 15.0)):
                    m = np.abs(xb) <= w
                    axs[jj, i].plot(xb[m], d[m], color=HC[ht], lw=1, label=HL[ht])
            for jj, w in enumerate((3, 15)):
                axs[jj, i].set(title="t = %g, |x| <= %g" % (t, w), xlabel="x", ylabel="stretch")
                axs[jj, i].grid(alpha=0.3)
        axs[0, 0].legend(fontsize=7)
        fig.suptitle("%s: stretch profile at three spacings (from field snapshots, not committed)" % title)
        if done:
            save(fig, "profiles_%s.png" % tag)
        else:
            plt.close(fig)


def p_wavenumber(brows, bc):
    tags = ["B1_bump_pk0.8_s0.5", "B1_bump_pk0.8_s2", "B2_sbend_pk0.8_s0.5", "B3_jump_sbend_w1.5", "B3_jump_bump_w1.5", "B4_orbit_sbend_s0.5_pk0.8"]
    tags = [t for t in tags if t + "_h8" in bc]
    if not tags:
        return
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), squeeze=False)
    for a, tag in zip(axs.flat, tags):
        for h, ht in AS.HS:
            k = "%s_%s" % (tag, ht)
            if k not in bc:
                continue
            c, tw = bc[k]
            a.plot(c["t"], c["k1"], color=HC[ht], label=HL[ht])
            a.plot(c["t"], c["krms"], color=HC[ht], ls="--", lw=0.8)
            a.axhline(np.pi / h, color=HC[ht], ls=":", lw=0.7)
        a.set(title=tag, xlabel="time", ylabel="stretch wavenumber: first moment (solid), rms (dashed); pi/h dotted", xscale="log", yscale="log")
        a.grid(alpha=0.3, which="both")
    axs.flat[0].legend(fontsize=7)
    save(fig, "wavenumber_vs_time.png")


def p_spectra(A5, brows):
    if A5:
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
        for a, tag in zip(axs, ("closed", "sponge")):
            nm = "A5_sbend_w2.5_%s" % tag
            try:
                ser = lib.S(nm)
            except Exception:
                continue
            t = ser["t"]
            for i, L in enumerate(ser["probe_dist"]):
                if L < 60:
                    continue
                q = ser["probes"][i].astype(float)
                q = q - q.mean()
                w = np.hanning(q.size)
                Q = np.abs(np.fft.rfft(q * w)) / (w.sum() / 2)
                om = 2 * np.pi * np.fft.rfftfreq(q.size, d=float(t[1] - t[0]))
                a.semilogy(om, Q + 1e-40, lw=0.6, label="probe at %g sites" % L)
            r = A5[tag]["rate_late"]
            for k in (1, 3, 5):
                a.axvline(k * r, color="k", ls=":", lw=0.6)
            a.axvline(2.0, color="r", ls="--", lw=0.6)
            a.set(title="A5 %s ring: amplitude spectrum of u at far probes (dotted: rate and odd overtones; red: band top 2)" % tag, xlabel="omega", ylabel="amplitude", xlim=(0, 14))
            a.grid(alpha=0.3); a.legend(fontsize=7)
        save(fig, "spectra_A5_far_probe.png")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    pick = [("B1_bump_pk0.8_s0.5", "Gaussian bump sigma 0.5"), ("B3_jump_sbend_w1.5", "jump S-bend 1.5"), ("B1_bump_pk0.8_s2", "Gaussian bump sigma 2")]
    for a, (tag, lab) in zip(axs, pick):
        for h, ht in AS.HS:
            nm = "%s_%s" % (tag, ht)
            if not lib.exists(nm):
                continue
            ser = lib.S(nm)
            if ser["probes"].shape[0] < 3:
                continue
            q = ser["probes"][2].astype(float)
            q = q - q.mean()
            w = np.hanning(q.size)
            Q = np.abs(np.fft.rfft(q * w)) ** 2
            om = 2 * np.pi * np.fft.rfftfreq(q.size, d=float(ser["t"][1] - ser["t"][0]))
            a.semilogy(om, Q / Q.sum() + 1e-30, lw=0.6, color=HC[ht], label=HL[ht])
        a.axvline(1.0, color="k", ls="--", lw=0.7)
        a.set(title="%s: u at 50 units" % lab, xlabel="omega", ylabel="power fraction per bin", xlim=(0, 3))
        a.grid(alpha=0.3); a.legend(fontsize=7)
    save(fig, "spectra_smooth_far_probe.png")


def all_plots(A1, A2, A2c, A5, A6, A7, brows, bc, B6, B7, c6):
    for fn, args in ((p_family, (A1,)), (p_lifetimes, (A2, A2c)), (p_bump_tracks, (A2c,)), (p_push, (A6, B7)),
                     (p_smooth_frac, (brows, bc)), (p_profiles, (brows,)), (p_wavenumber, (brows, bc)), (p_spectra, (A5, brows))):
        try:
            fn(*args)
        except Exception as e:
            import traceback
            print("plot %s failed: %s" % (fn.__name__, e))
            traceback.print_exc()
