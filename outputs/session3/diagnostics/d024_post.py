"""Tail of d024: attach the dip packet rows (T = 30,000) and their controls from scan_summary.json."""
import json, os
from dcommon import HERE, ROOT, write_json
S = json.load(open(os.path.join(ROOT, "outputs", "session3", "wpL", "scan_summary.json")))
out = json.load(open(os.path.join(HERE, "d024_partial.json")))
rows = [x for x in S["rows"] if x["start"] == "dip" and x["T"] == 30000.0 and x["kind"] == "packet"]
out["packet_rows"] = [dict(packet=x["packet"], k0=x["params"]["packet"]["k0"], efrac=x["amp"], scale=x["scale"], dN_settled=x["dN_over_N_settled"],
                           early=x["Gamma_500_2000"], early_control=x["Gamma_500_2000_control"], late=x["Gamma_late_alive_mean"],
                           late_control=x["Gamma_late_alive_mean_control"], frac_end=x["frac_end"]) for x in rows]
out["control_row_dip_s1_T30000"] = S["controls"]["dip_s1_T30000"]
write_json(os.path.join(HERE, "d024_partial.json"), out)
for r in out["packet_rows"]: print(r)
for s in out["starts"]: print(s["N_over_Nmin"], s["early_500_2000"], s["late_alive_mean"], s["frac_end"])
