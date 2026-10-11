"""EXP-039 SPEC 6.1 / 6.3: step-0 quantities under LENIENT gold (common.parse_lenient), for every judge.
Input: data/step0/<P>/{d0,d8}.jsonl and d8.G_<J>.npy (J15, J7 from EXP-034; J3 from static39.py), data/readouts_<J>_<P>.json.
For each (J, P):  tau_len = 20% quantile of G_cal on lenient-correct d8(P) (TPR 0.80);  FPR0_len = share of lenient-wrong
d8(P) answers with G_cal >= tau_len;  tie_at_tau = share of lenient-wrong d8(P) answers with G_cal exactly = tau_len.
Per judge: tau_common = 20% quantile of G_cal over the lenient-correct d8 answers of BOTH policies (pooled), FPR0 at it.
Common-wrong questions: d0/d8 questions where every step-0 sample of BOTH policies is lenient-wrong; FPR0 on them.
Strict versions are written next to the lenient ones for comparison.  usage: python lenient_static39.py -> results/static_lenient.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import parse_lenient, correct_generous
from stats import auc
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sig = lambda z: 1 / (1 + np.exp(-z))
POL, JUD = ("P05", "P15"), [j for j in ("J15", "J3", "J7") if os.path.exists(f"{ROOT}/data/step0/P05/d8.G_{j}.npy")]
rows = {(p, s): [json.loads(l) for l in open(f"{ROOT}/data/step0/{p}/{s}.jsonl")] for p in POL for s in ("d0", "d8")}
for k, v in rows.items():
    for r in v:
        r["len_ok"] = parse_lenient(r["response"]) == r["gold"]
        r["gen_ok"] = correct_generous(r["response"], r["gold"])
# common-wrong questions (by question text)
allq = {}
for (p, s), v in rows.items():
    for r in v:
        allq.setdefault(r["question"], {}).setdefault(p, []).append(r["len_ok"])
cw = sorted(q for q, d in allq.items() if all(p in d for p in POL) and not any(any(d[p]) for p in POL))
out = {"judges": JUD, "n_questions_step0": len(allq), "n_common_wrong_questions": len(cw), "cells": {}, "common": {},
       "strict_vs_lenient_d8": {p: {"acc_strict": float(np.mean([r["correct"] for r in rows[p, "d8"]])),
                                    "acc_lenient": float(np.mean([r["len_ok"] for r in rows[p, "d8"]]))} for p in POL}}
json.dump(cw, open(f"{ROOT}/data/common_wrong_questions.json", "w"))
for J in JUD:
    gc = {p: json.load(open(f"{ROOT}/data/readouts_{J}_{p}.json"))["G_cal"] for p in POL}
    assert gc["P05"] == gc["P15"]
    S, Y, Ys, Q, YG = {}, {}, {}, {}, {}
    for p in POL:
        G = np.load(f"{ROOT}/data/step0/{p}/d8.G_{J}.npy").astype(np.float64)
        S[p] = sig(gc[p]["a"] * G + gc[p]["b"]); Y[p] = np.array([r["len_ok"] for r in rows[p, "d8"]])
        Ys[p] = np.array([r["correct"] for r in rows[p, "d8"]]); YG[p] = np.array([r["gen_ok"] for r in rows[p, "d8"]]); Q[p] = [r["question"] for r in rows[p, "d8"]]
    tc = float(np.quantile(np.concatenate([S[p][Y[p]] for p in POL]), 0.20))
    out["common"][J] = {"tau_common_len": tc}
    for p in POL:
        s, y, ys = S[p], Y[p], Ys[p]
        t = float(np.quantile(s[y], 0.20)); ts = float(np.quantile(s[ys], 0.20))
        w = ~y
        cwm = np.array([q in set(cw) for q in Q[p]]) & w
        out["cells"][f"{p}x{J}"] = {
            "tau_len": t, "FPR0_len": float((s[w] >= t).mean()), "tie_at_tau_len": float((s[w] == t).mean()),
            "tie_at_tau_len_correct": float((s[y] == t).mean()),
            "FPR0_len_common_tau": float((s[w] >= tc).mean()),
            "n_wrong_len": int(w.sum()), "auc_len": auc(s, y.astype(int)),
            "FPR0_gen": float((S[p][~YG[p]] >= t).mean()), "n_wrong_gen": int((~YG[p]).sum()),
            "tau_strict": ts, "FPR0_strict": float((s[~ys] >= ts).mean()),
            "n_common_wrong_d8": int(cwm.sum()), "FPR0_len_common_wrong": float((s[cwm] >= t).mean()) if cwm.any() else None,
            "frac_reward_lt_0.05_d8": float((s < 0.05).mean()), "frac_reward_gt_0.95_d8": float((s > 0.95).mean())}
json.dump(out, open(f"{ROOT}/results/static_lenient.json", "w"), indent=1)
print(json.dumps(out, indent=1))
