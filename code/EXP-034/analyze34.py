"""EXP-034 analysis (SPEC sections 6-8). Written before any RL result exists.
Input: runs/<id>/{steps.jsonl, evals.jsonl, run.json} copied from gpu-server-2; results/static_J15.json, static_J7.json.
Output: results/analysis.json.  usage: python analyze34.py [--stage 1|2] [--boot 2000]
"""
import argparse, glob, json, math, os, re
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--stage", type=int, default=2)
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
CELLS = ["P05J15", "P05J7", "P15J15", "P15J7"]
T_MAX, W, MIN_WRONG = 150, 5, 20
LP, LJ = math.log(1.54 / 0.49), math.log(7.61 / 1.54)
K = LP / LJ  # 0.717: H_gap predicts Delta_P = -K * Delta_J
STATIC = {j: json.load(open(f"{ROOT}/results/static_{j}.json")) for j in ("J15", "J7")}


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def run_metrics(d, cell):
    P, J = cell[:3], cell[3:]
    fpr0 = STATIC[J]["cells"][P]["FPR0_d8"]
    st = {r["step"]: r for r in jl(f"{d}/steps.jsonl")}
    ev = sorted(jl(f"{d}/evals.jsonl"), key=lambda r: r["step"])
    last = max(st) if st else 0
    wrong = {t: r["n_wrong"] for t, r in st.items()}
    fooled = {t: (r["FPR_G_cal"] or 0) * r["n_wrong"] for t, r in st.items()}

    def win(t0, t1, f=None):
        ts = [t for t in range(t0, t1 + 1) if t in st]
        nw = sum(wrong[t] for t in ts)
        if nw < MIN_WRONG:
            return None
        if f == "auc":
            v = [st[t]["auc_G_cal"] for t in ts if st[t].get("auc_G_cal") is not None]
            return float(np.mean(v)) if v else None
        return sum(fooled[t] for t in ts) / nw

    h = lambda x: None if x is None else (x - fpr0) / (1 - fpr0)
    curve = {t: win(t - W + 1, t) for t in range(W, last + 1)}
    t_half = None
    for t in range(W, T_MAX - W + 1):
        a_, b_ = h(curve.get(t)), h(win(t + 1, t + W))
        if a_ is not None and b_ is not None and a_ >= 0.5 and b_ >= 0.5:
            t_half = t
            break
    t09 = next((t for t in range(W, last + 1) if curve.get(t) is not None and curve[t] >= 0.9), None)
    t_auc = next((t for t in range(W, last + 1) if (x := win(t - W + 1, t, "auc")) is not None and x <= 0.5), None)
    # eval side
    e0 = ev[0]["FPR_G_cal"] if ev else None
    te = None
    for i in range(len(ev) - 1):
        if e0 is not None and e0 < 1 and (ev[i]["FPR_G_cal"] - e0) / (1 - e0) >= 0.5 and (ev[i + 1]["FPR_G_cal"] - e0) / (1 - e0) >= 0.5 and ev[i]["step"] > 0:
            te = ev[i]["step"]
            break
    es, kl = [r["step"] for r in ev], [r["kl_greedy_per_token"] for r in ev]
    kl_at = float(np.interp(t_half, es, kl)) if (t_half is not None and es) else None
    acc = [r["acc"] for r in ev]
    rj = json.load(open(f"{d}/run.json"))
    return {"id": os.path.basename(d), "cell": cell, "seed": int(re.search(r"-s(\d+)$", d).group(1)), "status": rj.get("status"),
            "last_step": last, "FPR0_d8": fpr0, "t_half": t_half, "censored": t_half is None, "t_fpr09": t09, "t_auc05": t_auc,
            "t_half_eval": te, "FPR_eval0": e0, "kl_at_t_half": kl_at,
            "kl_final": kl[-1] if kl else None, "acc0": acc[0] if acc else None, "acc_peak": max(acc) if acc else None,
            "acc_peak_step": es[int(np.argmax(acc))] if acc else None, "acc_final": acc[-1] if acc else None,
            "len0": ev[0]["mean_len"] if ev else None, "len_final": ev[-1]["mean_len"] if ev else None,
            "repeat_final": ev[-1]["frac_repeat"] if ev else None, "short_final": ev[-1]["frac_short"] if ev else None,
            "train_window_h": {t: h(v) for t, v in curve.items()},
            "wall_seconds": rj.get("total_seconds"), "max_mem_gb": rj.get("max_mem_reserved_gb")}


def X_of(cells):
    P = np.array([c.startswith("P15") for c in cells], float)
    J = np.array([c.endswith("J7") for c in cells], float)
    return np.stack([np.ones_like(P), P, J, P * J], 1)


def aft(y, cens, cells):
    """log-normal AFT with right censoring at ln 150; returns beta (4), sigma. y = ln t (or ln 150 if censored)."""
    X = X_of(cells)

    def nll(th):
        b, ls = th[:4], th[4]
        s = math.exp(ls)
        z = (y - X @ b) / s
        return -(np.sum(norm.logpdf(z[~cens]) - ls) + np.sum(norm.logsf(z[cens])))
    b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    r = minimize(nll, np.r_[b0, math.log(max(np.std(y - X @ b0), 0.1))], method="Nelder-Mead",
                 options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-8})
    return r.x[:4], math.exp(r.x[4]), r.success


def contrasts(b):
    dJ, dP = b[2] + b[3] / 2, b[1] + b[3] / 2
    return {"Delta_J": dJ, "Delta_P": dP, "C_gap": dP + K * dJ, "interaction": b[3],
            "r": (dP / (-K * dJ)) if abs(dJ) > 1e-9 else None}


runs = [run_metrics(d, re.search(r"-(P\d\dJ\d+)-s\d+$", d).group(1)) for d in sorted(glob.glob(f"{ROOT}/runs/*-e34-P*J*-s*"))
        if os.path.exists(f"{d}/run.json")]
seeds = (301, 302) if a.stage == 1 else (301, 302, 303)
use = [r for r in runs if r["status"] == "completed" and r["seed"] in seeds]
out = {"stage": a.stage, "K": K, "runs": runs, "n_used": len(use)}
y = np.array([math.log(r["t_half"]) if r["t_half"] else math.log(T_MAX) for r in use])
cens = np.array([r["censored"] for r in use])
cells = [r["cell"] for r in use]
out["cell_summary"] = {c: {"t_half": [r["t_half"] for r in use if r["cell"] == c],
                           "mean_ln_t_half_censored_as_150": float(np.mean([v for v, cc in zip(y, cells) if cc == c])) if c in cells else None,
                           "t_fpr09": [r["t_fpr09"] for r in use if r["cell"] == c],
                           "t_half_eval": [r["t_half_eval"] for r in use if r["cell"] == c],
                           "kl_at_t_half": [r["kl_at_t_half"] for r in use if r["cell"] == c],
                           "acc0": [r["acc0"] for r in use if r["cell"] == c], "acc_final": [r["acc_final"] for r in use if r["cell"] == c]}
                       for c in CELLS}
# stop rules after stage 1 (SPEC section 8)
if a.stage == 1:
    cs = out["cell_summary"]
    s1 = all(t is None for t in cs["P15J15"]["t_half"]) and all(
        (r["acc_final"] or 0) >= (r["acc0"] or 0) - 0.02 for r in use if r["cell"] == "P15J15")
    s2 = all(r["censored"] for r in use if r["cell"].endswith("J7")) and all(not r["censored"] for r in use if r["cell"].endswith("J15"))
    spread = [abs(np.diff([v for v, cc in zip(y, cells) if cc == c])[0]) for c in CELLS if cells.count(c) == 2]
    means = [out["cell_summary"][c]["mean_ln_t_half_censored_as_150"] for c in CELLS if c in cells]
    s3 = bool(spread and means and np.median(spread) > (max(means) - min(means)))
    out["stop_rules"] = {"S1": bool(s1), "S2": bool(s2), "S3": s3, "median_within_cell_spread_ln": float(np.median(spread)) if spread else None,
                         "max_between_cell_diff_ln": float(max(means) - min(means)) if means else None}
if len(set(cells)) == 4 and len(use) >= 8:
    b, s, okf = aft(y, cens, cells)
    out["aft"] = {"beta": b.tolist(), "sigma": s, "converged": bool(okf), **contrasts(b)}
    bo = np.linalg.lstsq(X_of(cells), y, rcond=None)[0]
    out["ols_censored_as_150"] = {"beta": bo.tolist(), **contrasts(bo)}
    rng = np.random.default_rng(0)
    idx = {c: [i for i, cc in enumerate(cells) if cc == c] for c in CELLS}
    B = []
    for _ in range(a.boot):
        ii = np.concatenate([rng.choice(idx[c], len(idx[c])) for c in CELLS])
        bb, _, _ = aft(y[ii], cens[ii], [cells[i] for i in ii])
        B.append(contrasts(bb))
    ci = {k: [float(np.nanpercentile([x[k] for x in B if x[k] is not None], q)) for q in (2.5, 97.5)] for k in ("Delta_J", "Delta_P", "C_gap", "interaction", "r")}
    out["aft"]["ci95"] = ci
    lo = lambda k: ci[k][0]
    hi = lambda k: ci[k][1]
    inc0 = lambda k: lo(k) <= 0 <= hi(k)
    if not lo("Delta_J") > 0:
        v = "H_policy" if hi("Delta_P") < 0 else ("no size effect in this range" if inc0("Delta_P") else "unresolved (P1 fails)")
    elif inc0("C_gap") and hi("Delta_P") < 0:
        v = "H_gap"
    elif inc0("Delta_P") and lo("C_gap") > 0:
        v = "H_judge"
    elif hi("Delta_P") < 0 and lo("C_gap") > 0:
        v = "both contribute"
    elif hi("C_gap") < 0:
        v = "policy effect larger than H_gap predicts"
    else:
        v = "unresolved"
    out["verdict_steps"] = v
    # KL coordinate (sensitivity, SPEC 7.1): same model on ln(KL at t_half); censored runs use ln(final KL)
    ky = np.array([math.log(max(r["kl_at_t_half"] if r["kl_at_t_half"] is not None else r["kl_final"], 1e-6)) for r in use])
    kb, ks, _ = aft(ky, cens, cells)
    out["aft_kl"] = {"beta": kb.tolist(), "sigma": ks, **contrasts(kb)}
    # 261009 10:0x (added after the stage-1 step-coordinate results were seen): same seed bootstrap for the KL coordinate
    rng = np.random.default_rng(1)
    BK = []
    for _ in range(a.boot):
        ii = np.concatenate([rng.choice(idx[c], len(idx[c])) for c in CELLS])
        bb, _, _ = aft(ky[ii], cens[ii], [cells[i] for i in ii])
        BK.append(contrasts(bb))
    out["aft_kl"]["ci95"] = {k: [float(np.nanpercentile([x[k] for x in BK if x[k] is not None], q)) for q in (2.5, 97.5)]
                             for k in ("Delta_J", "Delta_P", "C_gap", "interaction", "r")}
    # exact number of distinct seed resamples per cell (n = 2 -> 3), reported because the bootstrap is coarse
    out["n_per_cell"] = {c: len(idx[c]) for c in CELLS}
json.dump(out, open(f"{ROOT}/results/analysis{'_stage1' if a.stage == 1 else ''}.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1, default=float))
