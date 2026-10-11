"""EXP-046 descriptives after unblinding (SPEC section 6 list; not part of the test). Run on each host:
  python code/desc46.py results/t_half46.json blind/runs/exp046-*-P05J*-s30?  -> blind/compact/desc46_<host>.json, blind/compact/fooled_<cell>.jsonl
Per run: format drift per 50-step block of training rollouts ('#### n' present = parse_strict; 'The answer is' substring;
lenient accuracy); eval accuracy at steps 0 / 100 / 200 under strict / lenient / generous gold.
Fooled answers (= fooled39/fooled41 logic): per cell, lenient-wrong rollouts with G_cal >= tau_len in steps [t_half, t_half+19]
(t_half thr 0.5; censored: last 20 steps); features of fooled vs not fooled; 20 random fooled (seed 0)."""
import json, os, random, socket, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import parse_lenient, parse_strict, correct_generous
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SL = json.load(open(f"{ROOT}/results/static_lenient.json"))["cells"]
TH = json.load(open(sys.argv[1]))
OUT = f"{ROOT}/blind/compact"
res, pool = {}, {}
for d in sys.argv[2:]:
    rid = os.path.basename(d.rstrip("/"))
    cell = rid.split("-")[-2]; P, J = cell[:3], cell[3:]
    tau = SL[f"{P}x{J}"]["tau_len"]
    T = json.load(open(f"{d}/config.json"))["max_steps"]
    th = TH[rid]; a, b = (th, th + 19) if th else (T - 19, T)
    blk = {}
    for l in open(f"{d}/rollouts.jsonl"):
        r = json.loads(l)
        k = f"{(r['step'] - 1) // 50 * 50 + 1}-{(r['step'] - 1) // 50 * 50 + 50}"
        x = blk.setdefault(k, [0, 0, 0, 0])
        ok = parse_lenient(r["completion"]) == r["gold"]
        x[0] += 1; x[1] += parse_strict(r["completion"]) is not None; x[2] += "the answer is" in r["completion"].lower(); x[3] += ok
        if a <= r["step"] <= b and not ok:
            r["fooled"] = r["G_cal"] >= tau; r["run"] = rid
            pool.setdefault(cell, []).append(r)
    ev = {}
    for s in (0, 100, 200):
        p = f"{d}/eval_outputs_step{s:03d}.jsonl"
        if os.path.exists(p):
            rows = [json.loads(l) for l in open(p)]
            ev[f"eval{s}"] = {"strict": float(np.mean([parse_strict(r["response"]) == r["gold"] for r in rows])),
                              "lenient": float(np.mean([parse_lenient(r["response"]) == r["gold"] for r in rows])),
                              "generous": float(np.mean([correct_generous(r["response"], r["gold"]) for r in rows])),
                              "has_####": float(np.mean([parse_strict(r["response"]) is not None for r in rows])),
                              "answer_is": float(np.mean(["the answer is" in r["response"].lower() for r in rows]))}
    res[rid] = {**ev, "train_by_50": {k: {"hashes": v[1] / v[0], "answer_is": v[2] / v[0], "acc_len": v[3] / v[0]} for k, v in blk.items()}}
    print(rid, "done", flush=True)
stats = {}
for cell, v in pool.items():
    f = [r for r in v if r["fooled"]]; nf = [r for r in v if not r["fooled"]]
    feat = lambda rs: {"n": len(rs), "mean_tokens": float(np.mean([r["n_tokens"] for r in rs])) if rs else None,
                       "has_####": float(np.mean([parse_strict(r["completion"]) is not None for r in rs])) if rs else None,
                       "answer_is": float(np.mean(["the answer is" in r["completion"].lower() for r in rs])) if rs else None,
                       "generous_correct": float(np.mean([correct_generous(r["completion"], r["gold"]) for r in rs])) if rs else None,
                       "repeat": float(np.mean([r["repeat"] for r in rs])) if rs else None,
                       "len_ge_250": float(np.mean([r["n_tokens"] >= 250 for r in rs])) if rs else None}
    stats[cell] = {"fooled": feat(f), "not_fooled": feat(nf)}
    random.Random(0).shuffle(f)
    with open(f"{OUT}/fooled_{cell}.jsonl", "w") as fh:
        for r in f[:20]:
            fh.write(json.dumps({k: r[k] for k in ("run", "step", "question", "gold", "completion", "G_cal", "n_tokens")}, ensure_ascii=False) + "\n")
host = "gpu-server-2" if os.path.isdir("/data/exp029/models/sft") else "gpu-server-1"
json.dump({"runs": res, "fooled_stats": stats}, open(f"{OUT}/desc46_{host}.json", "w"), indent=1)
