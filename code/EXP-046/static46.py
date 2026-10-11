"""EXP-046: unchanged copy of EXP-041 static41.py (step-0 data only; not blinded). EXP-041 docstring:
EXP-041: copy of EXP-039 code (only change: see EXP-041 SPEC section 5). EXP-039 (copy of EXP-034 static34.py, unchanged procedure) step 0: static gate and readouts for each judge (SPEC section 4).
For judge J in {J15, J7}: score the step-0 sets d0, d8 of both policies (P05, P15) with the no-reference prompt (judge34).
  G_cal(J): 1-D logistic on G, fitted on the pooled d0 of both policies (one reward function per judge).
  tau(J, P): TPR = 0.80 on the correct answers of d8(P).  FPR0(J, P) = FPR of d8(P) wrong answers at tau(J, P).
  gate: AUC(G) on d8(P) >= 0.60 for every (J, P) cell.
usage: python static34.py <judge_name> <judge_path>   (reads data/step0/<P>/{d0,d8}.jsonl; writes data/readouts_<J>.json,
       results/static_<J>.json, data/step0/<P>/<set>.G_<J>.npy)
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import auc, fit_lr
from judge39 import Judge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J, JP = sys.argv[1], sys.argv[2]
POL = ("P05", "P15")
judge = Judge(JP, "cuda", None)
t0 = time.time()
rows, G = {}, {}
for p in POL:
    for s in ("d0", "d8"):
        r = [json.loads(l) for l in open(f"{ROOT}/data/step0/{p}/{s}.jsonl")]
        g, _ = judge.forward([x["question"] for x in r], [x["response"] for x in r], bs=16, hidden=False)
        rows[p, s], G[p, s] = r, g.astype(np.float64)
        np.save(f"{ROOT}/data/step0/{p}/{s}.G_{J}.npy", g)
        print(J, p, s, len(r), f"{time.time() - t0:.0f}s", flush=True)
y = {k: np.array([x["correct"] for x in v], int) for k, v in rows.items()}
Gp = np.concatenate([G[p, "d0"] for p in POL]); yp = np.concatenate([y[p, "d0"] for p in POL])
m = fit_lr(Gp[:, None], yp, 0.0)
a = float(m["w"][0] / m["sd"][0]); b = float(m["b"] - m["w"][0] * m["mu"][0] / m["sd"][0])
sig = lambda z: 1 / (1 + np.exp(-z))
out = {"judge": J, "judge_path": JP, "G_cal": {"a": a, "b": b}, "tau_tpr80_d8": {}, "sd_d8": {}, "cells": {}}
for p in POL:
    s8 = sig(a * G[p, "d8"] + b); y8 = y[p, "d8"]
    tau = float(np.quantile(s8[y8 == 1], 0.20))
    out["cells"][p] = {"tau": tau, "sd_d8": float(s8.std()), "FPR0_d8": float((s8[y8 == 0] >= tau).mean()),
                       "auc_d8_G": auc(G[p, "d8"], y8), "auc_d0_G": auc(G[p, "d0"], y[p, "d0"]),
                       "acc_d8": float(y8.mean()), "acc_d0": float(y[p, "d0"].mean()),
                       "n_wrong_d8": int((y8 == 0).sum()), "mean_Gcal_d8": float(s8.mean()),
                       "frac_G_pos_d8": float((G[p, "d8"] > 0).mean())}
out["gate_pass"] = bool(all(c["auc_d8_G"] >= 0.60 for c in out["cells"].values()))
out["seconds"] = time.time() - t0
out["max_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
os.makedirs(f"{ROOT}/results", exist_ok=True)
json.dump(out, open(f"{ROOT}/results/static_{J}.json", "w"), indent=1)
for p in POL:  # per-run readouts file: reward G_cal of this judge, tau / sd for this (judge, policy) cell
    json.dump({"judge": J, "policy": p, "G_cal": out["G_cal"], "tau_tpr80_d8": {"G_cal": out["cells"][p]["tau"]},
               "sd_d8": {"G_cal": out["cells"][p]["sd_d8"]}, "FPR0_d8": out["cells"][p]["FPR0_d8"]},
              open(f"{ROOT}/data/readouts_{J}_{p}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
