"""Two-lump reductions: the N-body model must reproduce the WP-C tables (WPC_DERIVATION.md sections 1, 5, 6)."""
import numpy as np

from swirl.arrange import model as mdl
from swirl.arrange.pairlaw import _load_derive

# printed digits from WPC_DERIVATION.md (section 5 and 6 tables), d: (F_asym, C_N, K, 1/sqrtK)
PRINTED = {14: (2.067, 4.237, 1.89e-4, 72.8), 16: (0.772, 1.612, 7.18e-5, 118.0), 18: (0.294, 0.6229, 2.78e-5, 189.8),
           20: (0.1137, 0.2437, 1.09e-5, 303.4), 24: (0.01759, 0.03839, 1.71e-6, 764.5)}
# printed point-model shifts per lump with N exchange (section 6 second table), d, dtheta: (T100, T300, T500)
PRINTED_SHIFT = {(16, 45): (1.10, -8.3, -25.9), (16, 90): (-0.081, -2.83, -6.92), (16, 135): (-0.88, -4.93, -9.31),
                 (20, 45): (0.146, 1.50, 1.48), (20, 90): (-0.0018, -0.143, -0.886), (20, 135): (-0.141, -1.13, -2.64),
                 (24, 45): (0.0222, 0.204, 0.590), (24, 90): (-4.5e-5, -3.6e-3, -0.028),
                 (24, 135): (-0.0221, -0.196, -0.524)}


def pair_config(d, dth):
    return np.array([[-d / 2, 0, 0], [d / 2, 0, 0]], float), np.array([0.0, dth])


def printed_digits_match(x, s):
    """x rounds to the printed value s (same number of significant digits as printed)."""
    txt = ("%r" % s).replace("-", "")
    mant = txt.split("e")[0].lstrip("0.").replace(".", "")
    nsig = max(len(mant), 1)
    return float("%.*g" % (nsig, x)) == float("%.*g" % (nsig, s))


def test_force_current_pendulum_against_wpc(params, laws, wpc, record):
    law = laws["A"]
    w = params["domega_dN"]
    worst = 0.0
    rows = {}
    for row in wpc["table"]:
        d = row["d"]
        X, th = pair_config(d, 0.0)
        F = mdl.forces(X, th, law)
        F_att = F[0, 0]                                   # force on lump 1 toward lump 2 (+x)
        X, th = pair_config(d, np.pi / 2)
        dN2 = mdl.ndot(X, th, law)[1]
        X, th = pair_config(d, 0.0)
        Htt = mdl.hessian(X, th, law)[6:, 6:]
        K = abs(w) * np.max(np.linalg.eigvalsh(Htt))        # relative phase: dtheta'' = |w| (2 C g) dtheta
        U = mdl.energy(X, th, law)
        rel = [abs(F_att / row["F_asym_dphi0"] - 1), abs(dN2 / row["dN2dt_asym_dphi90"] - 1),
               abs(K / row["K_phase_pendulum"] - 1), abs(U / row["U_asym_dphi0"] - 1)]
        worst = max(worst, max(rel))
        rows[str(d)] = {"F": F_att, "dN2dt_90": dN2, "K": K, "1/sqrtK": 1 / np.sqrt(K), "U": U, "rel": rel}
        if d in PRINTED:
            pf, pc, pk, pt = PRINTED[d]
            assert printed_digits_match(F_att, pf), (d, F_att, pf)
            assert printed_digits_match(-dN2, pc), (d, dN2, pc)
            assert printed_digits_match(K, pk), (d, K, pk)
            assert printed_digits_match(1 / np.sqrt(K), pt), (d, 1 / np.sqrt(K), pt)
    record("two_lump_force_current_K", {"worst_rel_vs_wpc_json": worst, "rows": rows})
    assert worst < 1e-12


def test_point_model_shifts_against_wpc(params, laws, wpc, record):
    law = laws["A"]
    E, w, hw = params["E"], params["domega_dN"], params["hw"]
    worst_rel, worst_abs, worst_t = 0.0, 0.0, 0.0
    printed_ok = []
    for row in wpc["table"]:
        d = row["d"]
        if d < 16:
            continue
        for key, frozen in (("shift_point_model_exchange", False), ("shift_point_model_fixed_dphi", True)):
            for pdeg, rec in row[key].items():
                X, th = pair_config(d, np.deg2rad(float(pdeg)))
                sol = mdl.integrate(X, th, law, E, w, 500.0, frozen=frozen, d_stop=2 * hw, rtol=1e-12, atol=1e-12,
                                    t_eval=[100.0, 300.0, 500.0])
                for k, T in enumerate(("100", "300", "500")):
                    ref = rec[T]
                    if "reached_d_2hw_at" in ref:
                        assert sol.status == 1
                        worst_t = max(worst_t, abs(sol.t_events[0][0] - ref["reached_d_2hw_at"]))
                        continue
                    y = sol.y[:, k]
                    dd = y[3] - y[0]
                    shift = (d - dd) / 2
                    Xs, P, ph, nu = mdl.unpack(y, 2)
                    dphi = np.rad2deg(ph[1] - ph[0])
                    n1 = nu[0]
                    for a, b in ((shift, ref["shift_per_lump"]), (dphi, ref["dphi_deg"]), (n1, ref["N1_minus_N0"])):
                        worst_abs = max(worst_abs, abs(a - b))
                        if abs(b) > 1e-3:
                            worst_rel = max(worst_rel, abs(a / b - 1))
                    if not frozen and (d, int(pdeg)) in PRINTED_SHIFT:
                        s = PRINTED_SHIFT[(d, int(pdeg))][k]
                        printed_ok.append(printed_digits_match(shift, s))
    record("two_lump_point_model", {"worst_rel": worst_rel, "worst_abs": worst_abs,
                                    "worst_merge_time_abs": worst_t, "printed_shifts_matched": sum(printed_ok),
                                    "printed_shifts_total": len(printed_ok)})
    assert all(printed_ok) and len(printed_ok) == 27
    assert worst_rel < 1e-6 and worst_t < 1e-4


def test_model_B_reproduces_fsup(laws, record):
    """Model B pair force equals derive_pair_law.F_sup on the table nodes and between them."""
    mod = _load_derive()
    law = laws["B"]
    worst_node, worst_mid = 0.0, 0.0
    for d in (4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0):
        for p in (0.0, 30.0, 45.0, 90.0, 120.0, 180.0):
            D = np.deg2rad(p)
            for dd, kind in ((d, "node"), (d + 0.125, "mid")):
                ref = mod.F_sup(dd, 0.0, D)
                got = law.pair(np.array([dd]), np.array([D]))[1][0]   # u_d = F (attractive positive)
                err = abs(got - ref) / abs(mod.F_asym(dd))
                if kind == "node":
                    worst_node = max(worst_node, err)
                else:
                    worst_mid = max(worst_mid, err)
    record("model_B_vs_Fsup", {"worst_node_rel_to_Fasym": worst_node, "worst_midpoint_rel_to_Fasym": worst_mid})
    assert worst_node < 1e-10 and worst_mid < 1e-4
