"""F-029 gate check at 0.4 per lump: l3.gate_summary with SPEEDS = (0.4,), reading the rest-lump run from wpL3 (symlink) and the
v = 0.4 runs (theta 0 and 90, T = 60) from this directory. Same targets as the production gate (3x 1e-2, 1e-4, 1e-4, 1e-3)."""
import json
import os
from dcommon import HERE, write_json
from swirl.engine import l3

l3.OUT = "outputs/session3/diagnostics/l3"
l3.SPEEDS = (0.4,)
g = l3.gate_summary("primary")
ok, why = l3.gate_verdict(g)
out = dict(start="primary", speed=0.4, T=60.0, t_from=20.0, targets=l3.GATE_TARGETS, pass_rule="each within 3x its target",
           ok=bool(ok), why=why, results={k: {kk: vv for kk, vv in v.items() if kk in (
               "v", "gamma", "speed_measured", "speed_rel_err", "E_ratio_vs_E0ref_err", "E_ratio_vs_rest_run_err", "omega_ratio", "omega_ratio_err",
               "E_ball_drift_minus_rest", "contraction_Ixx_over_Iyy_first", "one_over_gamma_sq", "contraction_drift")} for k, v in g["results"].items()},
           rest_reference=g["rest_reference"])
write_json(os.path.join(HERE, "d029_gate.json"), out)
print(json.dumps(out, indent=1, default=float))
