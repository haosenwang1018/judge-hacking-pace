"""EXP-046 analysis after unblinding (SPEC sections 6-8; PROCESS 'analysis agent' steps 2-6). Definitions are not changed.
Inputs: results/compact/*.json (12 EXP-046 runs, from extract46.py on both hosts), ../EXP-039/results/compact (20),
        ../EXP-041/results/compact (6); results/gpu-server-2|gpu-server-1/static_lenient.json; results/predictors46.json (locked
        predictors47.py, unchanged, run on the new judges); ../EXP-047/results/FINAL_PREDICTOR.json (locked fits).
onset(), plateau() and AFT are executed from ../EXP-041/code/analyze41.py (definitions only); its SL and CELLS are
extended with the new cells (FPR0_len of the cell's own host static_lenient.json).
Output: results/analysis46.json.
Test (EXP-047 SPEC sections 6.1, 6.2, 7 applied to the new cells; EXP-047 SPEC has no separate pass rule, so the
EXP-047 section 7 qualification rule is reused as the pass rule): P1 passes if its sign-adjusted Spearman on the new
cells is > 0 for both O1 and O2 AND its O1 MAE from the locked 8-cell fit is below the best of B1-B3."""
import glob, hashlib, itertools, json, math, os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.dirname(ROOT)
E39, E41, E47 = f"{RES}/EXP-039", f"{RES}/EXP-041", f"{RES}/EXP-047"
src = open(f"{E41}/code/analyze41.py").read()
ns = {"__file__": f"{E41}/code/analyze41.py"}
exec(src[: src.index("paths = sorted(")], ns)
onset, plateau, AFT, CELLS, SL, welch = ns["onset"], ns["plateau"], ns["AFT"], ns["CELLS"], ns["SL"], ns["welch"]
NEW = ["P05JL8", "P05JO1", "P05JO7"]
SL46 = {}
for h in ("gpu-server-2", "gpu-server-1"):
    SL46.update(json.load(open(f"{ROOT}/results/{h}/static_lenient.json"))["cells"])
for c in NEW:
    SL["cells"][f"{c[:3]}x{c[3:]}"] = SL46[f"{c[:3]}x{c[3:]}"]
    CELLS.append(c)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
FP_PATH = f"{E47}/results/FINAL_PREDICTOR.json"
assert sha(FP_PATH) == "8c1eb85a2e601c8ed563ad70824b4291dec3de9929ec0d50c225ba750d138af1"
FP = json.load(open(FP_PATH))

paths = sorted(glob.glob(f"{E39}/results/compact/*.json")) + sorted(glob.glob(f"{E41}/results/compact/*.json")) \
    + sorted(glob.glob(f"{ROOT}/results/compact/exp046-*.json"))
runs = [json.load(open(p)) for p in paths]
assert not any("diag" in r["run_id"] for r in runs)
runs = [r for r in runs if r["status"] == "completed"]


def hpool(r, a, b, nk="n_wl", fk="f_wl", f0key="FPR0_len"):
    st = r["steps"]
    ts = [str(t) for t in range(a, b + 1) if str(t) in st]
    n = sum(st[t][nk] for t in ts); f = sum(st[t][fk] for t in ts)
    f0 = SL["cells"][f"{r['cell'][:3]}x{r['cell'][3:]}"][f0key]
    return ((f / n - f0) / (1 - f0) if n else None), n


def evinterp(r, t, key):
    return ns["interp"](r, t, key)


R = []
for r in runs:
    T = r["steps_cfg"]
    e = {"run_id": r["run_id"], "cell": r["cell"], "seed": r["seed"], "T": T, "n_steps": len(r["steps"]),
         "status": r["status"], "total_seconds": r["total_seconds"]}
    e["E1"], e["n_wrong_101_150"] = hpool(r, 101, 150)
    e["E2"], e["n_wrong_151_200"] = hpool(r, 151, 200) if T >= 200 else (None, 0)
    e["E5"], _ = hpool(r, 126, 150)
    blocks = []
    for k in range(0, min(T, 200), 10):
        h, n = hpool(r, k + 1, k + 10)
        blocks.append(h if n >= 20 else None)
    e["E6"] = max([b for b in blocks if b is not None], default=None)
    e["E6_argmax_block"] = (blocks.index(e["E6"]) * 10 + 1) if e["E6"] is not None else None
    e["t05"], e["t03"] = onset(r, 0.5), onset(r, 0.3)
    e["E1_generous"], _ = hpool(r, 101, 150, "n_wg", "f_wg", "FPR0_gen")
    e["E1_strict"], _ = hpool(r, 101, 150, "n_ws", "f_ws", "FPR0_strict")
    e["h_5step"] = [hpool(r, t - 4, t)[0] for t in range(5, min(T, 200) + 1, 5)]
    ev = r["evals"]
    e["acc0"], e["acc_final"], e["fmt_final"] = ev[0]["acc"], ev[-1]["acc"], ev[-1]["format_rate"]
    e["kl_sampled_at_t05"] = evinterp(r, e["t05"], "kl_sampled_per_seq")
    e["kl_sampled_final"] = ev[-1].get("kl_sampled_per_seq")
    R.append(e)

out = {"n_runs": {"EXP-039": sum(1 for r in R if "e39" in r["run_id"] or "e34" in r["run_id"]),
                  "EXP-041": sum(1 for r in R if "e41" in r["run_id"]), "EXP-046": sum(1 for r in R if r["run_id"].startswith("exp046"))},
       "runs": [x for x in R if x["run_id"].startswith("exp046")]}
# AFT over all 38 runs (EXP-039 20, EXP-041 6, EXP-046 12), same sigma prior
onset_res = {}
for key in ("t05", "t03"):
    M = AFT([math.log(e[key] or e["T"]) for e in R], [e[key] is None for e in R], [e["cell"] for e in R])
    _, th = M.fit()
    onset_res[key] = {"sigma": math.exp(th[-1]), "mu": {c: float(th[i]) for i, c in enumerate(M.cl)},
                      "geo_mean": {c: math.exp(th[i]) for i, c in enumerate(M.cl)},
                      "per_run": {c: [e[key] if e[key] else f">{e['T']}" for e in R if e["cell"] == c] for c in M.cl},
                      "n_censored": {c: sum(e[key] is None for e in R if e["cell"] == c) for c in M.cl}}
out["onset"] = onset_res
cells = {}
for c in CELLS:
    rr = [e for e in R if e["cell"] == c]
    d = {"n_seeds": len(rr)}
    for k in ("E1", "E2", "E5", "E6", "E1_generous", "E1_strict"):
        v = [e[k] for e in rr if e[k] is not None]
        d[k] = {"seeds": v, "mean": float(np.mean(v)) if v else None}
    d["n_wrong_101_150_total"] = int(sum(e["n_wrong_101_150"] for e in rr))
    d["E3_mu_lnT"], d["E3_geo"] = onset_res["t05"]["mu"][c], onset_res["t05"]["geo_mean"][c]
    d["E4_mu_lnT"], d["E4_geo"] = onset_res["t03"]["mu"][c], onset_res["t03"]["geo_mean"][c]
    for key in ("t05", "t03"):
        v = [e[key] for e in rr]
        d[f"{key}_seed_geo_mean_uncensored"] = float(np.exp(np.mean(np.log([x for x in v if x])))) if any(v) else None
    if c in NEW:
        d["h_5step_mean"] = [float(np.mean([e["h_5step"][i] for e in rr if e["h_5step"][i] is not None])) for i in range(40)]
        d["acc0"] = [e["acc0"] for e in rr]; d["acc_final"] = [e["acc_final"] for e in rr]
        d["kl_sampled_at_t05"] = [e["kl_sampled_at_t05"] for e in rr]; d["kl_sampled_final"] = [e["kl_sampled_final"] for e in rr]
        d["E1_identifiable"] = d["n_wrong_101_150_total"] >= 200
        d["seeds_completed_ge3"] = len(rr) >= 3
    cells[c] = d
out["cells"] = cells
# check: old cells reproduce analysis41 / EXP-047 outcomes (O1 exact; O2 mu shifts slightly because sigma is shared)
A41 = json.load(open(f"{E41}/results/analysis41.json"))
out["check_O1_vs_analysis41_max_abs"] = max(abs(cells[c]["E1"]["mean"] - A41["plateau_cells"][c]["mean"]) for c in CELLS[:8])
out["check_O2_mu_vs_EXP047_max_abs"] = max(abs(cells[c]["E4_mu_lnT"] - FP["training_cells"][c]["O2"]) for c in CELLS[:8])

# ---------------- predictor test (EXP-047 SPEC 6.1 / 6.2 / 7 applied to the new cells)
PRED = json.load(open(f"{ROOT}/results/predictors46.json"))["cells"]
NAMES = ["P1_calib_slope", "B1_auc", "B2_fpr0", "B3_risk"]
SIGN = {"P1_calib_slope": 1, "B1_auc": -1, "B2_fpr0": 1, "B3_risk": 1}
OUTC = {"O1": {c: cells[c]["E1"]["mean"] for c in CELLS}, "O2": {c: cells[c]["E4_mu_lnT"] for c in CELLS},
        "O3": {c: cells[c]["E3_mu_lnT"] for c in CELLS}, "O4": {c: cells[c]["E5"]["mean"] for c in CELLS}}
ORIENT = {"O1": 1, "O2": -1, "O3": -1, "O4": 1}
SEEDV = {"O1": lambda c: cells[c]["E1"]["seeds"], "O4": lambda c: cells[c]["E5"]["seeds"],
         "O2": lambda c: [math.log(e["t03"] or e["T"]) for e in R if e["cell"] == c],
         "O3": lambda c: [math.log(e["t05"] or e["T"]) for e in R if e["cell"] == c]}


def welch_tie(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    d = x.mean() - y.mean()
    if vx + vy == 0:
        return bool(d == 0), float(d), [float(d), float(d)]
    se = math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / ((vx ** 2 / (len(x) - 1) if vx else 0) + (vy ** 2 / (len(y) - 1) if vy else 0))
    q = stats.t.ppf(0.975, df)
    return bool(d - q * se <= 0 <= d + q * se), float(d), [float(d - q * se), float(d + q * se)]


xv = lambda n, c: PRED[c][n]["value"]
test = {"x": {n: {c: xv(n, c) for c in NEW} for n in NAMES}, "outcomes_new": {o: {c: OUTC[o][c] for c in NEW} for o in OUTC}}
rank = {}
for n in NAMES:
    rank[n] = {}
    for o in OUTC:
        xs = [SIGN[n] * xv(n, c) for c in NEW]; ys = [ORIENT[o] * OUTC[o][c] for c in NEW]
        d = {"rho_new3": float(stats.spearmanr(xs, ys).correlation)}
        prs = {}
        agree = nn = 0
        for a, b in itertools.combinations(NEW, 2):
            tie, diff, ci = welch_tie(SEEDV[o](a), SEEDV[o](b))
            ok = (SIGN[n] * (xv(n, a) - xv(n, b))) * (ORIENT[o] * (OUTC[o][a] - OUTC[o][b])) > 0
            prs[f"{a}-{b}"] = {"cand_correct": bool(ok), "outcome_tied": tie, "outcome_diff_ci95": ci}
            if not tie:
                nn += 1; agree += ok
        d["pairs"] = prs; d["pairwise_nontied"] = {"agree": int(agree), "n": nn}
        row = [c for c in CELLS if c.startswith("P05")]   # pooled P05 row, 7 judges (descriptive: 4 cells were seen at selection)
        d["rho_P05_row7_descr"] = float(stats.spearmanr([SIGN[n] * xv(n, c) for c in row], [ORIENT[o] * OUTC[o][c] for c in row]).correlation)
        rank[n][o] = d
test["rank"] = rank
# locked fits: predict new P05 cells (indicator 0); 95% prediction interval from the 8 training cells (df 5)
TC = FP["training_cells"]
pred = {}
for n in NAMES:
    fitd = FP["predictor"]["fit"] if n == FP["final_predictor"] else FP["baselines"][n]["fit"]
    X = np.array([[1.0, TC[c]["x"][n], 1.0 if c.startswith("P15") else 0.0] for c in CELLS[:8]])
    XtXi = np.linalg.inv(X.T @ X)
    pred[n] = {}
    for o in ("O1", "O2"):
        f = fitd[o]; q = stats.t.ppf(0.975, f["df"])
        cc = {}
        for c in NEW:
            x0 = np.array([1.0, xv(n, c), 0.0])
            p = f["a"] + f["b"] * xv(n, c)
            half = q * f["resid_sd"] * math.sqrt(1 + x0 @ XtXi @ x0)
            cc[c] = {"pred": p, "obs": OUTC[o][c], "abs_err": abs(p - OUTC[o][c]), "pi95": [p - half, p + half],
                     "in_pi": bool(p - half <= OUTC[o][c] <= p + half)}
        pred[n][o] = {"cells": cc, "mae": float(np.mean([cc[c]["abs_err"] for c in NEW]))}
test["locked_fit_predictions"] = pred
best_base = {o: min(pred[b][o]["mae"] for b in NAMES[1:]) for o in ("O1", "O2")}
P1 = "P1_calib_slope"
test["pass_rule"] = "EXP-047 SPEC 7 qualification reused: rho_new3(O1) > 0 and rho_new3(O2) > 0 and MAE_O1 < min baseline MAE_O1"
test["P1_pass"] = {"rho_O1_pos": rank[P1]["O1"]["rho_new3"] > 0, "rho_O2_pos": rank[P1]["O2"]["rho_new3"] > 0,
                   "mae_O1_beats_baselines": pred[P1]["O1"]["mae"] < best_base["O1"],
                   "mae_O2_beats_baselines_descr": pred[P1]["O2"]["mae"] < best_base["O2"]}
test["P1_pass"]["pass"] = all(test["P1_pass"][k] for k in ("rho_O1_pos", "rho_O2_pos", "mae_O1_beats_baselines"))
test["best_baseline_mae"] = best_base
out["test"] = test
json.dump(out, open(f"{ROOT}/results/analysis46.json", "w"), indent=1, default=float)

# ---------------- print
print("runs", out["n_runs"], "checks O1", out["check_O1_vs_analysis41_max_abs"], "O2mu", out["check_O2_mu_vs_EXP047_max_abs"])
print("sigma", {k: round(v["sigma"], 3) for k, v in onset_res.items()})
for c in CELLS:
    d = cells[c]
    print(c, "E1 %.3f E2 %s E5 %.3f E6 %.3f | E3 geo %.1f E4 geo %.1f | nwrong %d | gen %.3f str %.3f" % (
        d["E1"]["mean"], d["E2"]["mean"] and round(d["E2"]["mean"], 3), d["E5"]["mean"], d["E6"]["mean"] or float("nan"),
        d["E3_geo"], d["E4_geo"], d["n_wrong_101_150_total"], d["E1_generous"]["mean"], d["E1_strict"]["mean"]))
for c in NEW:
    print(c, "E1 seeds", [round(x, 3) for x in cells[c]["E1"]["seeds"]], "t05", onset_res["t05"]["per_run"][c], "t03", onset_res["t03"]["per_run"][c])
for n in NAMES:
    print(n, {c: round(xv(n, c), 3) for c in NEW}, {o: (round(rank[n][o]["rho_new3"], 2), rank[n][o]["pairwise_nontied"], round(rank[n][o]["rho_P05_row7_descr"], 2)) for o in OUTC})
    for o in ("O1", "O2"):
        print("   ", o, "MAE %.3f" % pred[n][o]["mae"], {c: (round(v["pred"], 3), round(v["obs"], 3), [round(x, 2) for x in v["pi95"]], v["in_pi"]) for c, v in pred[n][o]["cells"].items()})
print("P1 pass", test["P1_pass"], "best base", best_base)
