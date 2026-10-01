"""Shape diagnostics of the energy density (WP3 run 5).

e = Evolver.energy_density() (every term: kinetic, gradient, mass, quartic, sextic).
Coordinates: minimum image about the start centre `ref`, then shifted to the energy centroid.
 c   = int x e / int e over r < R_s about the start centre, then recomputed once over the sphere about c.
 M   = int (x-c)(x-c) e over the sphere about c;  a2 = (lmax - lmin)/tr M
 T   = int (x-c)^3 e;  a3 = ||T||_F / (int e * (tr M / int e)^{3/2})
 E_sphere/E_total with E_total = sum of e over the whole box.
A single snapshot oscillates at 2 omega; phi_centre (phi at the centre, trilinear) is reported as the phase indicator.
CLI: shape.py checkpoint.npz  |  shape.py --validate
"""
import json, sys
import numpy as np
sys.path.insert(0, "/Users/aaronramos/Desktop/SWIRL")
from swirl.engine.evolver import Evolver
from swirl.engine.initial import build_initial
from swirl.engine.run import _min_image_r, _trilinear, full_config


def shape(ev, ref, Rs):
    L, N = ev.L, ev.N
    e = ev.energy_density()
    X = []
    for a, c0 in enumerate(ref):
        d = ev.coords - c0
        d = d - L * np.round(d / L)
        shp = [1, 1, 1]; shp[a] = -1
        X.append(np.broadcast_to(d.reshape(shp), e.shape))
    dV = ev.dV
    Etot = e.sum() * dV
    c = np.zeros(3)
    for _ in range(2):  # sphere about ref, then once about the new centroid
        r2 = sum((X[a] - c[a]) ** 2 for a in range(3))
        w = np.where(r2 < Rs ** 2, e, 0.0)
        S = w.sum()
        c = np.array([(w * X[a]).sum() / S for a in range(3)])
    r2 = sum((X[a] - c[a]) ** 2 for a in range(3))
    w = np.where(r2 < Rs ** 2, e, 0.0)
    S = w.sum()
    Y = [X[a] - c[a] for a in range(3)]
    M = np.array([[(w * Y[i] * Y[j]).sum() for j in range(3)] for i in range(3)]) * dV
    Es = S * dV
    T = np.array([[[(w * Y[i] * Y[j] * Y[k]).sum() for k in range(3)] for j in range(3)] for i in range(3)]) * dV
    lam = np.linalg.eigvalsh(M)
    trM = np.trace(M)
    a2 = float((lam[-1] - lam[0]) / trM)
    a3 = float(np.sqrt((T ** 2).sum()) / (Es * (trM / Es) ** 1.5))
    idx, wts = _trilinear(ev.coords, L, ev.dx, ref)
    phic = float(sum(wt * ev.phi[i] for i, wt in zip(idx, wts)))
    cabs = np.array(ref) + c
    return {"centroid": cabs.tolist(), "centroid_disp": float(np.linalg.norm(c)), "a2": a2, "a3": a3,
            "E_sphere_over_total": float(Es / Etot), "E_sphere": float(Es), "E_total_box": float(Etot),
            "phi_centre": phic, "eig_M": lam.tolist()}


def evolver_from(params, phi, pi):
    cfg = full_config(params)
    ev = Evolver(cfg["N"], cfg["L"], cfg["m"], cfg["g4"], cfg["g6"], absorber=cfg["absorber"], threads=cfg["threads"])
    ev.set_state(phi, pi)
    return ev


def shape_from_checkpoint(path, Rs=10.032114550865565):
    with np.load(path, allow_pickle=False) as z:
        phi, pi, meta = z["phi"].copy(), z["pi"].copy(), json.loads(str(z["meta"]))
    p = meta["params"]
    ev = evolver_from(p, phi, pi)
    out = shape(ev, p["center"], Rs)
    out["t"] = meta["step"] * p["dt"]
    return out


def validate():
    D = "/Users/aaronramos/Desktop/SWIRL/outputs/step2/configs/"
    base = json.load(open(D + "run1_primary.json"))
    base.update(N=128, threads=2)
    res = {}
    for name, ctr, eps in (("spherical_centre_on_lattice", [0, 0, 0], 0.0),
                           ("spherical_offcentre_0.137_-0.071_0.213", [0.137, -0.071, 0.213], 0.0),
                           ("stretched_quad_eps_0.02", [0, 0, 0], 0.02)):
        p = dict(base); ent = dict(p["init"]["radial"][0]); ent["center"] = ctr
        if eps: ent["quad_eps"] = eps
        p["init"] = {"radial": [ent], "gaussians": []}; p["center"] = ctr
        cfg = full_config(p)
        ev = Evolver(128, cfg["L"], cfg["m"], cfg["g4"], cfg["g6"], absorber=cfg["absorber"], threads=2)
        phi, pi = build_initial(ev.coords, ev.L, cfg["init"]); ev.set_state(phi, pi)
        res[name] = shape(ev, ctr, 10.032114550865565)
        print(name, {k: res[name][k] for k in ("a2", "a3", "centroid_disp", "E_sphere_over_total")})
    json.dump(res, open("/Users/aaronramos/Desktop/SWIRL/outputs/step2/diagnostics/shape_validation.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "--validate":
        validate()
    else:
        print(json.dumps(shape_from_checkpoint(sys.argv[1])))
