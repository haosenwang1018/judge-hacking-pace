"""EXP-041 copy of EXP-039 fooled39.py (unchanged logic): read fooled answers. Per cell, pool its runs; take lenient-wrong training rollouts with
G_cal >= tau_len in steps [t_half, t_half + 19] (censored runs: last 20 steps); draw 20 at random (seed 0).
Also write simple features of ALL fooled vs not-fooled lenient-wrong rollouts in those windows (length, '####' present,
answer repeated, text after the answer).  usage: python fooled39.py <t_half.json> <run_dir> ... -> results/fooled/<cell>.jsonl, fooled_stats.json"""
import json, os, random, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import parse_lenient, parse_strict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TH = json.load(open(sys.argv[1])); SL = json.load(open(f"{ROOT}/results/static_lenient.json"))
os.makedirs(f"{ROOT}/results/fooled", exist_ok=True)
pool, stats = {}, {}
for d in sys.argv[2:]:
    rid = os.path.basename(d.rstrip("/"))
    if rid not in TH:
        continue
    cell = rid.split("-")[-2]; P, J = cell[:3], cell[3:]
    T = json.load(open(f"{d}/config.json"))["max_steps"]; th = TH[rid]
    a, b = (th, th + 19) if th else (T - 19, T)
    tau = SL["cells"][f"{P}x{J}"]["tau_len"]
    for l in open(f"{d}/rollouts.jsonl"):
        r = json.loads(l)
        if a <= r["step"] <= b and parse_lenient(r["completion"]) != r["gold"]:
            r["fooled"] = r["G_cal"] >= tau; r["run"] = rid
            pool.setdefault(cell, []).append(r)
for cell, v in pool.items():
    f = [r for r in v if r["fooled"]]; nf = [r for r in v if not r["fooled"]]
    feat = lambda rs: {"n": len(rs), "mean_tokens": float(np.mean([r["n_tokens"] for r in rs])) if rs else None,
                       "has_####": float(np.mean([parse_strict(r["completion"]) is not None for r in rs])) if rs else None,
                       "repeat": float(np.mean([r["repeat"] for r in rs])) if rs else None,
                       "tail": float(np.mean([r["tail_chars"] > 0 for r in rs])) if rs else None,
                       "len_ge_250": float(np.mean([r["n_tokens"] >= 250 for r in rs])) if rs else None}
    stats[cell] = {"fooled": feat(f), "not_fooled": feat(nf)}
    random.Random(0).shuffle(f)
    with open(f"{ROOT}/results/fooled/{cell}.jsonl", "w") as fh:
        for r in f[:20]:
            fh.write(json.dumps({k: r[k] for k in ("run", "step", "question", "gold", "completion", "G_cal", "n_tokens")}, ensure_ascii=False) + "\n")
json.dump(stats, open(f"{ROOT}/results/fooled_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
