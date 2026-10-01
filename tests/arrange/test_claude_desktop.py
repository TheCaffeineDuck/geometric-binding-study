"""Item 4: the two Claude Desktop numbers and the scholar's reproduction (PRIOR_ART S6.4), C = 1, kappa = 1."""
from swirl.arrange.scan import claude_desktop_check

SCHOLAR = {"ring5_step72": {"kappa_R": 1.0695, "n_negative": 4, "n_modes": 15, "U_i": -0.0358,
                            "radial": 0.169, "tangential": -0.328, "axial": 0.123},
           "ring6_step60": {"kappa_R": 0.2947, "U_i": -0.4102},
           "octa_inphase_shell_antiphase_centre": {"kappa_R": 1.591, "n_negative": 10, "n_modes": 21,
                                                   "U_i": -0.0724, "d2U": -0.51}}


def r(x, nd):
    return round(x, nd)


def test_claude_desktop_numbers(record):
    cd = claude_desktop_check()
    record("claude_desktop", cd)
    r5 = cd["ring5_step72"]
    assert len(r5) == 1 and r(r5[0]["kappa_R"], 4) == 1.0695 and r5[0]["radial_restoring"]
    assert r5[0]["n_negative"] == 4 and r5[0]["n_modes"] == 15 and r5[0]["n_zero_rigid"] == 6
    assert r(r5[0]["U_i_lump0"], 4) == -0.0358
    assert (r(r5[0]["lump0_curvature_radial"], 3), r(r5[0]["lump0_curvature_tangential"], 3),
            r(r5[0]["lump0_curvature_third"], 3)) == (0.169, -0.328, 0.123)
    r6 = cd["ring6_step60"]
    assert len(r6) == 1 and r(r6[0]["kappa_R"], 4) == 0.2947 and r(r6[0]["U_i_lump0"], 4) == -0.4102
    oc = cd["octa_inphase_shell_antiphase_centre"]
    assert len(oc) == 1 and r(oc[0]["kappa_R"], 3) == 1.591 and not oc[0]["radial_restoring"]
    assert oc[0]["n_negative"] == 10 and oc[0]["n_modes"] == 21
    assert r(oc[0]["U_i_lump0"], 4) == -0.0724 and r(oc[0]["breathing_d2U_dR2_total"], 2) == -0.51
