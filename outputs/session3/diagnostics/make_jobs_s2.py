"""Job definitions for the stage-2 diagnostics (configuration only; built with the l3.py builders, no engine change).
F-027: dip stretch eps = 0.1 and the unstretched control, T = 150, phi recorded at one point on a shell (cfg 'center' is the
        point whose phi(t) the engine records every step; the lump itself stays at the origin through lumps.centers).
F-029: primary start, gate at v = 0.4 (theta 0 and 90, T = 60) and the phase-0 collision at 0.4 per lump, T = 150."""
import json
import os
import numpy as np
from dcommon import HERE
from swirl.engine import l3

R_SHELL = 25.0
J = []


def add(**k):
    J.append(k)


# F-027
st = "dip"
N, L = l3.SRC_GRID[st]
hw = l3.numbers(st)[3]["hw"]
for tag, extra, c in (("q0.1_z", dict(quad_eps=0.1), [0.0, 0.0, R_SHELL]), ("q0.1_x", dict(quad_eps=0.1), [R_SHELL, 0.0, 0.0]),
                      ("control_x", {}, [R_SHELL, 0.0, 0.0])):
    lumps = [dict(center=[0.0, 0.0, 0.0], phase=0.0, v=0.0, **extra)]
    cfg = l3.run_config(st, N, L, 150.0, lumps, {}, shape=dict(every=1.0, R=3 * hw), it_every_step=True)
    cfg["center"] = c
    add(name="f027_%s" % tag, stage=("B" if tag.endswith("_z") else "C"), kind="run", start=st, deps=[], cfg=cfg, N=N, L=L, dt=cfg["dt"], T=150.0, variant=tag,
        job_hours=0.3, note="phi recorded at the point %s (radius %g from the lump) every step" % (c, R_SHELL))
# F-029 gate
st = "primary"
N, L = l3.SRC_GRID[st]
v, T = 0.4, 60.0
for th in (0.0, 90.0):
    lumps = [dict(center=[-v * T / 2, 0.0, 0.0], phase=float(np.radians(th)), v=v)]
    cfg = l3.run_config(st, N, L, T, lumps, {}, shape=dict(every=0.4, R=3 * l3.numbers(st)[3]["hw"]))
    add(name="gate_%s_v%g_th%g" % (st, v, th), stage="A", kind="run", start=st, deps=[], cfg=cfg, v=v, theta=th, N=N, L=L,
        dt=cfg["dt"], T=T, job_hours=0.15)
# F-029 collision
c = l3.COLL[st]
hwp = l3.numbers(st)[3]["hw"]
T = 150.0
lumps = [dict(center=[-c["d"] / 2, 0.0, 0.0], phase=0.0, v=v), dict(center=[c["d"] / 2, 0.0, 0.0], phase=0.0, v=-v)]
cfg = l3.run_config(st, c["N"], c["L"], T, lumps, {}, group=dict(R=3 * hwp + 2.0, axis=[1.0, 0.0, 0.0]))
add(name="coll_primary_p0_v0.4", stage="B", kind="run", start=st, deps=[], cfg=cfg, v=v, phase=0.0, d=c["d"], N=c["N"], L=c["L"],
    dt=cfg["dt"], T=T, t_contact=l3.contact_time(st, v), t_sponge=l3.sponge_arrival(st, c, v), phase_label=l3.PHASE_LABEL, job_hours=0.35)
for j in J:
    l3.guard_motion(j)
json.dump(J, open(os.path.join(HERE, "jobs_s2.json"), "w"), indent=1)
print([(j["name"], j["T"], j["N"]) for j in J], "t_contact", l3.contact_time(st, v), "t_sponge", J[-1]["t_sponge"])
