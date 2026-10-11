"""EXP-046 environment check (step-0 data only, no RL outcome). Scores the stored P05 / P15 d8 step-0 samples with the
J15 judge (Qwen2.5-1.5B-Instruct, same sha256 on both hosts) and compares G with the EXP-039 values computed on gpu-server-2
(data/step0/<P>/d8.G_J15.npy). usage: python envcheck46.py <J15_path> <host> -> results/envcheck_<host>.json"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge39 import Judge
from stats import auc
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
j = Judge(sys.argv[1], "cuda", None)
out = {"host": sys.argv[2], "judge_path": sys.argv[1], "cells": {}}
for p in ("P05", "P15"):
    r = [json.loads(l) for l in open(f"{ROOT}/data/step0/{p}/d8.jsonl")]
    g, _ = j.forward([x["question"] for x in r], [x["response"] for x in r], bs=16, hidden=False)
    ref = np.load(f"{ROOT}/data/step0/{p}/d8.G_J15.npy")
    y = np.array([x["correct"] for x in r], int)
    out["cells"][p] = {"max_abs_diff_G": float(np.abs(g - ref).max()), "mean_abs_diff_G": float(np.abs(g - ref).mean()),
                       "auc_here": auc(g, y), "auc_gpu-server-2": auc(ref, y)}
out["pass"] = bool(all(abs(c["auc_here"] - c["auc_gpu-server-2"]) < 0.005 for c in out["cells"].values()))
json.dump(out, open(f"{ROOT}/results/envcheck_{sys.argv[2]}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
