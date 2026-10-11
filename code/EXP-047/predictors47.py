"""EXP-047 SPEC section 4: step-0 candidate predictors per (judge, policy) cell. Written BEFORE any predictor was computed.
Inputs (all step 0, no RL data):
  <src>/<P>/d8.jsonl, <src>/<P>/d8.G_<J>.npy           stored judge logits on d8 (src = /data/exp041/data/step0)
  <readouts>/readouts_<J>_<P>.json                      G_cal (a, b) of judge J (reward used in RL)
  data/perturb/<P>/<v>.jsonl, data/probe/<P>/<v>.G_<J>.npy, data/probe/<P>/d8.G_<J>.npy   (edit texts and their scores)
Output: results/predictors47.json  {cell: {name: {"value", "ci95", "n"}}}, cell = "<P><J>" (e.g. P05JL)
usage: python code/predictors47.py <src> <readouts_dir> <J> [<J> ...] [--boot 1000]
Lenient gold everywhere (common.parse_lenient). tau = 20% quantile of s on lenient-correct d8 (TPR 0.80), = tau_len of EXP-039."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import parse_lenient
from stats import auc, fit_lr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("readouts"); ap.add_argument("J", nargs="+"); ap.add_argument("--boot", type=int, default=1000)
A = ap.parse_args()
sig = lambda z: 1 / (1 + np.exp(-z))
FV, SV = ["F1", "F2", "F3", "F4"], ["S1", "S2", "S3"]
MIN_N = 50


def load_cell(p, J):
    rows = [json.loads(l) for l in open(f"{A.src}/{p}/d8.jsonl")]
    gc = json.load(open(f"{A.readouts}/readouts_{J}_{p}.json"))["G_cal"]
    G = np.load(f"{A.src}/{p}/d8.G_{J}.npy").astype(np.float64)
    c = {"q": np.array([r["qid"] for r in rows]), "y": np.array([parse_lenient(r["response"]) == r["gold"] for r in rows]),
         "G": G, "s": sig(gc["a"] * G + gc["b"]), "ed": {}}
    pr = f"{ROOT}/data/probe/{p}/d8.G_{J}.npy"
    c["s_re"] = sig(gc["a"] * np.load(pr).astype(np.float64) + gc["b"]) if os.path.exists(pr) else None
    for v in FV + SV + ["E"]:
        fn, gn = f"{ROOT}/data/perturb/{p}/{v}.jsonl", f"{ROOT}/data/probe/{p}/{v}.G_{J}.npy"
        if os.path.exists(fn) and os.path.exists(gn):
            e = [json.loads(l) for l in open(fn)]
            c["ed"][v] = {"i": np.array([x["i"] for x in e], int), "s": sig(gc["a"] * np.load(gn).astype(np.float64) + gc["b"])}
    return c


def preds(c, idx):
    """all predictors on the d8 rows idx (a multiset of row indices; question-cluster bootstrap passes repeated rows)."""
    y, s, G, q = c["y"][idx], c["s"][idx], c["G"][idx], c["q"][idx]
    out = {}
    tau = float(np.quantile(s[y], 0.20))
    w = ~y
    fpr0 = float((s[w] >= tau).mean())
    out["B1_auc"] = auc(s, y.astype(int))
    out["B2_fpr0"] = fpr0
    out["B3_risk"] = fpr0 * (1 - float(y.mean()))
    m = fit_lr(G[:, None], y.astype(int), 0.0)
    out["P1_calib_slope"] = float(m["w"][0] / m["sd"][0])
    out["P2_below_tau_mass"] = float(((s[w] >= tau - 0.10) & (s[w] < tau)).mean())
    # P3 / P4: GRPO groups = the 8 samples of one question (bootstrap copies of a question are separate groups)
    num = den = 0.0
    sds = []
    for b in range(0, len(idx), 8):
        sg, yg = s[b:b + 8], y[b:b + 8]
        den += ((sg - sg.mean()) ** 2).sum()
        for cls in (True, False):
            if (yg == cls).any():
                num += ((sg[yg == cls] - sg[yg == cls].mean()) ** 2).sum()
        sds.append(sg.std(ddof=1))
    out["P3_noncorrect_var_share"] = float(num / den) if den > 0 else float("nan")
    out["P4_reward_sd"] = float(np.mean(sds))
    out["_tau"] = tau
    # edit-based predictors: originals scored in the same probe job (s_re)
    if c["s_re"] is not None and c["ed"]:
        sre = c["s_re"]
        cnt = np.bincount(idx, minlength=len(c["y"]))  # bootstrap multiplicity of each d8 row
        def rows_of(v, want_wrong):
            e = c["ed"][v]
            k = cnt[e["i"]]
            keep = (k > 0) & (c["y"][e["i"]] == (not want_wrong))
            return np.repeat(e["i"][keep], k[keep]), np.repeat(e["s"][keep], k[keep])
        if "E" in c["ed"]:
            i, se = rows_of("E", want_wrong=False)
            out["P5_min_error_detect"] = float(np.mean((sre[i] > se) + 0.5 * (sre[i] == se))) if len(i) >= MIN_N else float("nan")
            out["_n_P5"] = int(len(i))
        gains, fp_e, fp_o, nF, nS = {}, [], [], 0, 0
        for v in FV + SV:
            if v not in c["ed"]:
                continue
            i, se = rows_of(v, want_wrong=True)
            gains[v] = float(np.mean(se - sre[i])) if len(i) >= MIN_N else float("nan")
            fp_e.append(se >= tau); fp_o.append(sre[i] >= tau)
            nF += len(i) if v in FV else 0; nS += len(i) if v in SV else 0
        f = [gains[v] for v in FV if v in gains]; g = [gains[v] for v in SV if v in gains]
        out["P6_format_gain"] = float(np.max(f)) if f and not np.all(np.isnan(f)) else float("nan")
        out["P7_surface_gain"] = float(np.max(g)) if g and not np.all(np.isnan(g)) else float("nan")
        e_, o_ = np.concatenate(fp_e), np.concatenate(fp_o)
        out["P8_edit_fp_gain"] = float(e_.mean() - o_.mean()) if len(e_) >= MIN_N else float("nan")
        out["_gain_by_variant"] = gains
        out["_n_F"], out["_n_S"] = nF, nS
    return out


res = {"src": A.src, "judges": A.J, "boot": A.boot, "cells": {}}
for J in A.J:
    for p in ("P05", "P15"):
        c = load_cell(p, J)
        # rows must come in blocks of 8 per question for P3/P4
        uq, first = np.unique(c["q"], return_index=True)
        blocks = [np.where(c["q"] == qq)[0] for qq in uq[np.argsort(first)]]
        assert all(len(b) == 8 for b in blocks), "d8 must have 8 samples per question"
        full = np.concatenate(blocks)
        pt = preds(c, full)
        rng = np.random.default_rng(0)
        bs = {k: [] for k, v in pt.items() if not k.startswith("_")}
        for _ in range(A.boot):
            pick = rng.integers(0, len(blocks), len(blocks))
            pb = preds(c, np.concatenate([blocks[j] for j in pick]))
            for k in bs:
                bs[k].append(pb.get(k, np.nan))
        res["cells"][f"{p}{J}"] = {k: {"value": v, "ci95": [float(np.nanquantile(bs[k], 0.025)), float(np.nanquantile(bs[k], 0.975))]}
                                   if k in bs else v for k, v in pt.items()}
        print(p, J, "done", flush=True)
os.makedirs(f"{ROOT}/results", exist_ok=True)
json.dump(res, open(f"{ROOT}/results/predictors47.json", "w"), indent=1, default=float)
