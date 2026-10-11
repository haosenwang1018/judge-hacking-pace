"""Extra contrasts on the single 38-run AFT fit (revision 261011): JL - J3 onsets (P0.5) and Delta_P at h>=0.3 for all judges."""
import json, math, os
H = os.path.dirname(os.path.abspath(__file__))
exec(open(f"{H}/gap261011.py").read().split("res = {}")[0])
o = {}
for key in ("t05", "t03"):
    o[f"JL_minus_J3_P05_{key}"] = fits[key].profile("P05JL", "P05J3", levels=(0.95,))
    for J in ("J15", "J3", "J7"):
        o[f"DeltaP_{J}_{key}"] = fits[key].profile(f"P15{J}", f"P05{J}", levels=(0.95, 0.975))
json.dump(o, open(f"{H}/extra261011.json", "w"), indent=1, default=float)
for k, v in o.items(): print(k, round(v["estimate"], 2), [round(x, 2) for x in v["ci95.0"]], [round(x, 2) for x in v.get("ci97.5", [0, 0])])
