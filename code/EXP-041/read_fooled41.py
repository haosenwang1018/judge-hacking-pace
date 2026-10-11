"""EXP-041 copy of EXP-039 read_fooled39.py: for the 20 sampled fooled answers per cell (results/fooled/<cell>.jsonl), count how many are correct
under the generous gold (i.e. likely mis-graded prose answers) and how many have '#### n'. Prints one cell's samples
(argv[1]) truncated for reading.  usage: python read_fooled39.py [cell]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import correct_generous, parse_strict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
st = json.load(open(f"{ROOT}/results/fooled_stats.json"))
out = {}
for cell in ("P05JL", "P15JL"):
    p = f"{ROOT}/results/fooled/{cell}.jsonl"
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p)]
    out[cell] = {"n": len(rows), "generous_correct": sum(correct_generous(r["completion"], r["gold"]) for r in rows),
                 "has_strict": sum(parse_strict(r["completion"]) is not None for r in rows),
                 "fooled_mean_tokens": st[cell]["fooled"]["mean_tokens"], "not_fooled_mean_tokens": st[cell]["not_fooled"]["mean_tokens"],
                 "n_fooled_window": st[cell]["fooled"]["n"]}
    print(cell, out[cell])
json.dump(out, open(f"{ROOT}/results/fooled_read_counts.json", "w"), indent=1)
if len(sys.argv) > 1:
    for i, l in enumerate(open(f"{ROOT}/results/fooled/{sys.argv[1]}.jsonl")):
        r = json.loads(l); c = r["completion"].replace("\n", " | ")
        print(i, "gold", r["gold"], "G", round(r["G_cal"], 2), "::", c[:150], "...", c[-100:])
