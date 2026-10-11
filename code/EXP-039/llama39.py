"""EXP-039 SPEC 6.5: offline second-family check. Llama-3.2-3B-Instruct (no-reference prompt, same text as the Qwen judges)
re-scores lenient-wrong training rollouts. Does a hack that fools the training (Qwen) judge also fool Llama?
Step 0: Llama G on d8 of both policies -> tau_L(P) = 20% quantile of G on lenient-correct d8(P), FPR0_L(P).
Per run: window E = steps 1-10, window H = steps [t_half, t_half + 19] (censored runs: the last 20 steps).
Up to 200 lenient-wrong rollouts per window (random, seed 0). Report h_L (Llama) and h_Q (training judge, own tau_len).
usage: python llama39.py <t_half.json: {run_id: t_half or null}> <run_dir> ...  -> results/llama_rescore.json"""
import json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import parse_lenient
from judge39 import Judge
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TH = json.load(open(sys.argv[1]))
SL = json.load(open(f"{ROOT}/results/static_lenient.json"))
J = Judge(os.environ["LLAMA"], "cuda", None)
FN = f"{ROOT}/results/llama_rescore.json"
out = json.load(open(FN)) if os.path.exists(FN) else {"step0": {}, "runs": {}}   # incremental: runs already scored are kept
for p in [p for p in ("P05", "P15") if p not in out["step0"]]:
    rows = [json.loads(l) for l in open(f"{ROOT}/data/step0/{p}/d8.jsonl")]
    G, _ = J.forward([r["question"] for r in rows], [r["response"] for r in rows], hidden=False)
    y = np.array([parse_lenient(r["response"]) == r["gold"] for r in rows])
    t = float(np.quantile(G[y], 0.2))
    out["step0"][p] = {"tau_L": t, "FPR0_L": float((G[~y] >= t).mean()), "auc_L": None}
    from stats import auc
    out["step0"][p]["auc_L"] = auc(G, y.astype(int))
    print(p, out["step0"][p], flush=True)
for d in sys.argv[2:]:
    rid = os.path.basename(d.rstrip("/"))
    if rid not in TH or (rid in out["runs"] and out["runs"][rid].get("t_half") == TH[rid]):
        continue
    P, Jn = rid.split("-")[-2][:3], rid.split("-")[-2][3:]
    cfg = json.load(open(f"{d}/config.json")); T = cfg["max_steps"]
    th = TH[rid]
    win = {"E": (1, 10), "H": (th, th + 19) if th else (T - 19, T)}
    c = SL["cells"][f"{P}x{Jn}"]
    rows = {k: [] for k in win}
    for l in open(f"{d}/rollouts.jsonl"):
        r = json.loads(l)
        for k, (a, b) in win.items():
            if a <= r["step"] <= b and parse_lenient(r["completion"]) != r["gold"]:
                rows[k].append(r)
    res = {"t_half": th, "windows": win}
    for k, v in rows.items():
        random.Random(0).shuffle(v)
        v = v[:200]
        if not v:
            continue
        G, _ = J.forward([r["question"] for r in v], [r["completion"] for r in v], hidden=False)
        s0 = out["step0"][P]
        fL = float((G >= s0["tau_L"]).mean()); fQ = float(np.mean([r["G_cal"] >= c["tau_len"] for r in v]))
        res[k] = {"n": len(v), "FPR_L": fL, "h_L": (fL - s0["FPR0_L"]) / (1 - s0["FPR0_L"]),
                  "FPR_Q": fQ, "h_Q": (fQ - c["FPR0_len"]) / (1 - c["FPR0_len"]),
                  "both_fooled": float(np.mean([(g >= s0["tau_L"]) and (r["G_cal"] >= c["tau_len"]) for g, r in zip(G, v)])),
                  "mean_tokens": float(np.mean([r["n_tokens"] for r in v]))}
    out["runs"][rid] = res
    print(rid, {k: res.get(k) for k in ("E", "H")}, flush=True)
    json.dump(out, open(FN, "w"), indent=1)
json.dump(out, open(FN, "w"), indent=1)
