"""PAPER-T11 revision 261011, item 1: gold accuracy vs. fooled fraction (no GPU; existing logs only).
Inputs: evals_lenient.json (from extract_remote.py, all 38 runs' saved greedy eval outputs, 200 test problems),
        EXP-039/041/046 results/compact/*.json (training-rollout counts), static_lenient.json (FPR0).
Output: gold261011.json"""
import json, glob, os, math
import numpy as np
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(f"{H}/../..")
EV = json.load(open(f"{H}/evals_lenient.json"))
comp = {}
for p in sorted(glob.glob(f"{R}/EXP-039/results/compact/*.json") + glob.glob(f"{R}/EXP-041/results/compact/*.json") + glob.glob(f"{R}/EXP-046/results/compact/exp046-*.json")):
    d = json.load(open(p)); comp[d["run_id"]] = d
SL = json.load(open(f"{R}/EXP-041/results/static_lenient.json"))["cells"]
for h in ("gpu-server-1", "gpu-server-2"):
    SL.update(json.load(open(f"{R}/EXP-046/results/{h}/static_lenient.json"))["cells"])
A46 = json.load(open(f"{R}/EXP-046/results/analysis46.json"))
CELLS = ["P05J15", "P15J15", "P05J3", "P15J3", "P05J7", "P15J7", "P05JL", "P15JL", "P05JL8", "P05JO1", "P05JO7"]
f0 = lambda c: SL[f"{c[:3]}x{c[3:]}"]["FPR0_len"]
def tci(x):
    x = np.asarray(x, float); m = x.mean()
    if len(x) < 2: return m, [m, m]
    q = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)); return m, [m - q, m + q]
out = {"cells": {}, "per_run": {}}
for c in CELLS:
    rids = sorted(r for r in comp if comp[r]["cell"] == c)
    # held-out greedy eval, lenient; steps common to all seeds of the cell
    steps = sorted(set.intersection(*[set(int(s) for s in EV[r]["evals"]) for r in rids]))
    acc = np.array([[EV[r]["evals"][str(s)]["len"] / EV[r]["evals"][str(s)]["n"] for s in steps] for r in rids])
    gen = np.array([[EV[r]["evals"][str(s)]["gen"] / EV[r]["evals"][str(s)]["n"] for s in steps] for r in rids])
    strict = np.array([[EV[r]["evals"][str(s)]["strict"] / EV[r]["evals"][str(s)]["n"] for s in steps] for r in rids])
    fmt = np.array([[EV[r]["evals"][str(s)]["fmt"] / EV[r]["evals"][str(s)]["n"] for s in steps] for r in rids])
    # judge pass rate on all eval answers (accepted / 200) and FPR on eval wrong answers
    jp = np.array([[(EV[r]["evals"][str(s)]["fw"] + EV[r]["evals"][str(s)]["fc"]) / EV[r]["evals"][str(s)]["n"] for s in steps] for r in rids])
    gain = acc - acc[:, :1]
    m = acc.mean(0); k = int(np.argmax(m[1:]) + 1)
    per_seed_peak_gain = (acc[:, 1:].max(1) - acc[:, 0])
    # training rollouts, lenient: 1 - n_wl/n on 10-step windows
    roll = []
    for r in rids:
        st = comp[r]["steps"]; T = comp[r]["steps_cfg"]; w = []
        for b in range(0, T, 10):
            n = sum(st[str(t)]["n"] for t in range(b + 1, b + 11) if str(t) in st)
            wl = sum(st[str(t)]["n_wl"] for t in range(b + 1, b + 11) if str(t) in st)
            w.append(1 - wl / n if n else np.nan)
        roll.append(w)
        # per-run onset vs rollout-peak (5-step windows, smoothed over 10)
        t05 = A46["onset"]["t05"]["per_run"]  # by cell, ordered as runs sorted
    L = min(len(w) for w in roll); RO = np.array([w[:L] for w in roll])
    # absolute FPR in 101-150 (pooled per seed)
    fpr = []
    for r in rids:
        st = comp[r]["steps"]; ts = [str(t) for t in range(101, 151) if str(t) in st]
        fpr.append(sum(st[t]["f_wl"] for t in ts) / sum(st[t]["n_wl"] for t in ts))
    d = {"seeds": len(rids), "eval_steps": steps, "acc_len_mean": m.tolist(), "acc_gen_mean": gen.mean(0).tolist(),
         "acc_strict_mean": strict.mean(0).tolist(), "fmt_mean": fmt.mean(0).tolist(), "judge_pass_mean": jp.mean(0).tolist(),
         "acc0": float(m[0]), "peak_seedmean": float(m[k]), "peak_step": steps[k],
         "peak_gain_seedmean": float(m[k] - m[0]), "peak_gain_ci_at_step": tci(gain[:, k])[1],
         "per_seed_max_gain": per_seed_peak_gain.tolist(),
         "acc_step50": float(m[steps.index(50)]) if 50 in steps else None,
         "acc_step150": float(m[steps.index(150)]) if 150 in steps else None,
         "acc_last": float(m[-1]), "last_step": steps[-1],
         "rollout_acc_10win_mean": np.nanmean(RO, 0).tolist(), "rollout_acc_first10": float(np.nanmean(RO[:, 0])),
         "rollout_peak": float(np.nanmax(np.nanmean(RO, 0))), "rollout_peak_win_end": int((np.nanargmax(np.nanmean(RO, 0)) + 1) * 10),
         "rollout_per_seed_peak_win_end": [int((np.nanargmax(x) + 1) * 10) for x in RO],
         "FPR0": f0(c), "FPR_abs_101_150_seeds": fpr, "FPR_abs_101_150_mean": float(np.mean(fpr)),
         "Hbar": A46["cells"][c]["E1"]["mean"], "t05_per_run": A46["onset"]["t05"]["per_run"][c]}
    out["cells"][c] = d
json.dump(out, open(f"{H}/gold261011.json", "w"), indent=1)
print("cell  acc0  peak(step) gain [ci]  per-seed-maxgain  acc50 acc150 last | roll first->peak(win) | FPRabs H")
for c, d in out["cells"].items():
    print(f"{c:7s} {d['acc0']:.3f} {d['peak_seedmean']:.3f}({d['peak_step']}) {d['peak_gain_seedmean']:+.3f} [{d['peak_gain_ci_at_step'][0]:+.3f},{d['peak_gain_ci_at_step'][1]:+.3f}] "
          f"{[round(x,3) for x in d['per_seed_max_gain']]} {d['acc_step50']:.3f} {d['acc_step150']:.3f} {d['acc_last']:.3f}@{d['last_step']} | "
          f"{d['rollout_acc_first10']:.2f}->{d['rollout_peak']:.2f}({d['rollout_peak_win_end']}) {d['rollout_per_seed_peak_win_end']} | {d['FPR_abs_101_150_mean']:.2f} {d['Hbar']:.3f} t05 {d['t05_per_run']}")
