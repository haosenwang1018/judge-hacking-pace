"""EXP-047 SPEC 6.3 edit mechanism check (+ one exploratory block, labelled). Run on gpu-server-2 from /data/exp047.
usage: python code/mech47.py <src> <readouts_dir>   -> results/mech47.json, results/mech47_samples.txt
Counts per (policy, variant): generated/kept/dropped (manifest), qualified rows used by P5-P8 (F/S: lenient-wrong originals,
E: lenient-correct originals), lenient answer unchanged (F/S) or equal to n' and != gold (E), truncation > 1500 chars, E pre_final_found.
Samples: for each judge, rng = default_rng(0) per judge, 10 E and 10 F1 edits with original/edited score s."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import parse_lenient

src, rd = sys.argv[1], sys.argv[2]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sig = lambda z: 1 / (1 + np.exp(-z))
man = json.load(open(f"{ROOT}/data/perturb/manifest.json"))
V = ["F1", "F2", "F3", "F4", "S1", "S2", "S3", "E"]
out = {"counts": {}, "exploratory_scale": {}}
txt = []
for p in ("P05", "P15"):
    d8 = [json.loads(l) for l in open(f"{src}/{p}/d8.jsonl")]
    y = np.array([parse_lenient(r["response"]) == r["gold"] for r in d8])
    out["counts"][p] = {}
    for v in V:
        e = [json.loads(l) for l in open(f"{ROOT}/data/perturb/{p}/{v}.jsonl")]
        i = np.array([x["i"] for x in e])
        q = (y[i] if v == "E" else ~y[i])
        ok = [(parse_lenient(x["response"]) == x["ans1"] != x["gold"]) if v == "E" else (parse_lenient(x["response"]) == parse_lenient(d8[x["i"]]["response"])) for x in e]
        c = dict(man["cells"][p]["variants"][v])
        c.update({"qualified": int(q.sum()), "lenient_as_intended": float(np.mean(ok)), "trunc1500": float(np.mean([x["trunc1500"] for x in e])),
                  "orig_trunc1500": float(np.mean([len(d8[x["i"]]["response"]) > 1500 for x in e]))})
        if v == "E":
            c["pre_final_found"] = float(np.mean([x["extra"]["pre_final_found"] for x in e]))
        out["counts"][p][v] = c
for J in ("J15", "J3", "J7", "JL"):
    rng = np.random.default_rng(0)
    for p in ("P05", "P15"):
        gc = json.load(open(f"{rd}/readouts_{J}_{p}.json"))["G_cal"]
        G = np.load(f"{src}/{p}/d8.G_{J}.npy").astype(np.float64)
        d8 = [json.loads(l) for l in open(f"{src}/{p}/d8.jsonl")]
        y = np.array([parse_lenient(r["response"]) == r["gold"] for r in d8])
        # exploratory (not pre-registered): judge scale quantities next to P1
        out["exploratory_scale"][f"{p}{J}"] = {"G_cal_a": gc["a"], "G_cal_b": gc["b"], "sd_G": float(G.std()),
                                              "mean_G_correct_minus_wrong": float(G[y].mean() - G[~y].mean()),
                                              "cohen_d_G": float((G[y].mean() - G[~y].mean()) / G.std())}
        if p != "P05":
            continue
        sre = sig(gc["a"] * np.load(f"{ROOT}/data/probe/{p}/d8.G_{J}.npy").astype(np.float64) + gc["b"])
        for v in ("E", "F1"):
            e = [json.loads(l) for l in open(f"{ROOT}/data/perturb/{p}/{v}.jsonl")]
            se = sig(gc["a"] * np.load(f"{ROOT}/data/probe/{p}/{v}.G_{J}.npy").astype(np.float64) + gc["b"])
            pool = [k for k, x in enumerate(e) if (y[x["i"]] if v == "E" else ~y[x["i"]])]
            for k in rng.choice(pool, 10, replace=False):
                x = e[k]
                o, n = d8[x["i"]]["response"].splitlines(), x["response"].splitlines()
                diff = [f"  - {a}\n  + {b}" for a, b in zip(o, n) if a != b] + [f"  + {b}" for b in n[len(o):]]
                txt.append(f"=== {J} {p} {v} i={x['i']} gold={x['gold']} ans0={x['ans0']} ans1={x['ans1']} s_orig={sre[x['i']]:.3f} s_edit={se[k]:.3f}\n" + "\n".join(diff))
json.dump(out, open(f"{ROOT}/results/mech47.json", "w"), indent=1, default=float)
open(f"{ROOT}/results/mech47_samples.txt", "w").write("\n".join(txt))
print(json.dumps(out, indent=1, default=float))
