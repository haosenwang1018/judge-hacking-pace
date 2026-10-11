"""EXP-047 SPEC section 3.4: one judge scores every edit file of both policies (no-reference prompt, judge39.Judge.forward,
G = logit(Yes) - logit(No), response truncated to 1500 chars as in RL). Also re-scores the original d8 as a consistency check
against the stored d8.G_<J>.npy (max abs diff is logged; bf16 batch composition can change G slightly).
Output: data/probe/<P>/<variant>.G_<J>.npy (aligned with data/perturb/<P>/<variant>.jsonl), data/probe/<P>/d8.G_<J>.npy,
        results/probe_<J>.json (counts, seconds, consistency, max memory).
usage: python code/probe47.py <J> <judge_path> <step0_src> [--limit N] [--device cuda]"""
import argparse, glob, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge39 import Judge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("J"); ap.add_argument("path"); ap.add_argument("src")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--device", default="cuda"); ap.add_argument("--bs", type=int, default=16)
A = ap.parse_args()
dt = torch.bfloat16 if A.device.startswith("cuda") else torch.float32
judge = Judge(A.path, A.device, None, dtype=dt)
t0 = time.time()
rep = {"judge": A.J, "path": A.path, "limit": A.limit, "files": {}, "consistency": {}}
for p in ("P05", "P15"):
    os.makedirs(f"{ROOT}/data/probe/{p}", exist_ok=True)
    d8 = [json.loads(l) for l in open(f"{A.src}/{p}/d8.jsonl")]
    if A.limit:
        d8 = d8[:A.limit]
    g, _ = judge.forward([x["question"] for x in d8], [x["response"] for x in d8], bs=A.bs, hidden=False)
    if not A.limit:
        np.save(f"{ROOT}/data/probe/{p}/d8.G_{A.J}.npy", g)
    old = f"{A.src}/{p}/d8.G_{A.J}.npy"
    if os.path.exists(old):
        g0 = np.load(old)[:len(g)]
        rep["consistency"][p] = {"max_abs_diff": float(np.abs(g - g0).max()), "mean_abs_diff": float(np.abs(g - g0).mean())}
    for fn in sorted(glob.glob(f"{ROOT}/data/perturb/{p}/*.jsonl")):
        v = os.path.basename(fn)[:-6]
        rows = [json.loads(l) for l in open(fn)]
        if A.limit:
            rows = rows[:A.limit]
        g, _ = judge.forward([x["question"] for x in rows], [x["response"] for x in rows], bs=A.bs, hidden=False)
        np.save(f"{ROOT}/data/probe/{p}/{v}.G_{A.J}{'.smoke' if A.limit else ''}.npy", g)
        rep["files"][f"{p}/{v}"] = len(rows)
        print(A.J, p, v, len(rows), f"{time.time() - t0:.0f}s", flush=True)
rep["seconds"] = time.time() - t0
rep["max_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30 if A.device.startswith("cuda") else None
os.makedirs(f"{ROOT}/results", exist_ok=True)
json.dump(rep, open(f"{ROOT}/results/probe_{A.J}{'_smoke' if A.limit else ''}.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
