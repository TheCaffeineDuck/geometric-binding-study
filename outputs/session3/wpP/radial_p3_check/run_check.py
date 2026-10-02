import json, os, sys, time
import numpy as np
from swirl.family.wpa_run import run, profile
NMIN = 1633.571
out = os.path.dirname(os.path.abspath(__file__))
def win(tw, w, t, half=50.0):
    s = (tw > t - half) & (tw <= t)
    return float(np.median(w[s])) if s.any() else None
for om in [float(a) for a in sys.argv[1:]]:
    t0 = time.time()
    prof = profile(4/3, om)
    r = run(4/3, om, 2000.0, stop_at_death=False, prof=prof,
            ckpt=os.path.join(out, "ckpt_%g.npz" % om))
    t, Ein, tw, w = r["series"]["t"], r["series"]["E_in"], r["omega_t"], r["omega"]
    at = lambda T: float(Ein[np.argmin(abs(t - T)), 0] / r["E0"])
    res = dict(beta=4/3, omega_profile=om, N=prof["N"], N_over_Nmin=prof["N"] / NMIN,
               omega_window="median of whole-period omega over (t-50, t]",
               omega={str(T): win(tw, w, T) for T in (300, 1000, 1500, 2000)},
               drift_300=abs(win(tw, w, 300) - win(tw, w, 200)),
               drift_1500=abs(win(tw, w, 1500) - win(tw, w, 1400)),
               frac={str(T): at(T) for T in (300, 1000, 1500, 2000)},
               loss_rate_1000_2000=(at(1000) - at(2000)) / 1000.0,
               T_90=r["T_90"], T_death=r["T_death"], frac_end=r["frac_end"],
               rcs=r["params"]["rcs"], grid={k: r["params"][k] for k in ("dr","dt","R_sponge","R_box")},
               wall_s=time.time() - t0)
    p = os.path.join(out, "radial_p3_%g.json" % om)
    json.dump(res, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
    print(json.dumps(res), flush=True)
