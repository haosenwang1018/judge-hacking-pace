"""EXP-047 analysis (SPEC sections 5, 6.1, 6.2, 7, 8; PROCESS analysis steps 3-5, 7). Definitions are not changed.
Inputs: results/predictors47.json (from predictors47.py), ../EXP-041/results/analysis41.json,
        ../EXP-039/results/compact/*.json + ../EXP-041/results/compact/*.json (for O2 recompute, O4, seed-level values).
onset() and AFT are taken from ../EXP-041/code/analyze41.py (source executed up to its data section, so the same code runs).
Outputs: results/analysis47.json, results/FINAL_PREDICTOR.json.
Orientation: 'fooledness' outcomes are O1, -O2, -O3, O4 (larger = deeper / faster). Candidates are multiplied by their
pre-registered expected sign (B1, P5: -1). A positive sign-adjusted rho = direction as predicted."""
import glob, hashlib, itertools, json, math, os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.dirname(ROOT)
E39, E41 = f"{RES}/EXP-039", f"{RES}/EXP-041"
# --- reuse analyze41 onset()/plateau()/AFT/welch exactly (exec the definitions only, not its data section)
src = open(f"{E41}/code/analyze41.py").read()
ns = {"__file__": f"{E41}/code/analyze41.py"}
exec(src[: src.index("paths = sorted(")], ns)
onset, plateau, AFT, CELLS = ns["onset"], ns["plateau"], ns["AFT"], ns["CELLS"]

CAND = ["B1_auc", "B2_fpr0", "B3_risk", "P1_calib_slope", "P2_below_tau_mass", "P3_noncorrect_var_share", "P4_reward_sd",
        "P5_min_error_detect", "P6_format_gain", "P7_surface_gain", "P8_edit_fp_gain"]
SIGN = {c: (-1 if c.startswith(("B1", "P5")) else 1) for c in CAND}
GPU = {c: c.startswith(("P5", "P6", "P7", "P8")) for c in CAND}
BASE = ["B1_auc", "B2_fpr0", "B3_risk"]
PRED = json.load(open(f"{ROOT}/results/predictors47.json"))["cells"]
A41 = json.load(open(f"{E41}/results/analysis41.json"))

# ---------------- outcomes (SPEC 5)
paths = sorted(glob.glob(f"{E39}/results/compact/*.json")) + sorted(glob.glob(f"{E41}/results/compact/*.json"))
runs = [r for r in (json.load(open(p)) for p in paths) if r["status"] == "completed"]
R = []
for r in runs:
    t3, t5 = onset(r, 0.3), onset(r, 0.5)
    R.append({"cell": r["cell"], "seed": r["seed"], "T": r["steps_cfg"], "H": plateau(r)[0], "h126": plateau(r, 126, 150)[0],
              "t03": t3, "t05": t5})
seed = {c: {k: [e[k] for e in R if e["cell"] == c] for k in ("H", "h126")} for c in CELLS}
for c in CELLS:   # seed-level ln t_half; censored runs enter as ln T (lower bound; only used for the Welch tie rule)
    seed[c]["lnt03"] = [math.log(e["t03"] or e["T"]) for e in R if e["cell"] == c]
    seed[c]["lnt05"] = [math.log(e["t05"] or e["T"]) for e in R if e["cell"] == c]
    seed[c]["cens05"] = [e["t05"] is None for e in R if e["cell"] == c]


def aft_mu(key):
    M = AFT([math.log(e[key] or e["T"]) for e in R], [e[key] is None for e in R], [e["cell"] for e in R])
    _, th = M.fit()
    return {c: float(th[i]) for i, c in enumerate(M.cl)}


mu03, mu05 = aft_mu("t03"), aft_mu("t05")
OUT = {"O1": {c: float(np.mean(seed[c]["H"])) for c in CELLS},
       "O2": mu03, "O3": mu05,
       "O4": {c: float(np.mean(seed[c]["h126"])) for c in CELLS}}
SEEDKEY = {"O1": "H", "O2": "lnt03", "O3": "lnt05", "O4": "h126"}
ORIENT = {"O1": 1, "O2": -1, "O3": -1, "O4": 1}
check = {"O1_vs_analysis41_max_abs": max(abs(OUT["O1"][c] - A41["plateau_cells"][c]["mean"]) for c in CELLS),
         "O2_vs_analysis41_max_abs": max(abs(math.exp(OUT["O2"][c]) - A41["onset"]["t_half_h03"]["geo_mean"][c]) for c in CELLS),
         "O3_vs_analysis41_max_abs": max(abs(math.exp(OUT["O3"][c]) - A41["onset"]["t_half"]["geo_mean"][c]) for c in CELLS)}


def welch_tie(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    d = x.mean() - y.mean()
    if vx + vy == 0:
        return d == 0, d, [d, d]
    se = math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / ((vx ** 2 / (len(x) - 1) if vx else 0) + (vy ** 2 / (len(y) - 1) if vy else 0))
    q = stats.t.ppf(0.975, df)
    lo, hi = d - q * se, d + q * se
    return bool(lo <= 0 <= hi), d, [lo, hi]


def sp(a, b):
    return float(stats.spearmanr(a, b).correlation)


# exact permutation distribution of the 8-cell Spearman (reference only): P(rho_perm >= rho_obs)
_ranks8 = np.arange(8)
PERM = np.array([1 - 6 * np.sum((_ranks8 - np.array(p)) ** 2) / (8 * 63) for p in itertools.permutations(range(8))])

JUDGES = ["J15", "J3", "J7", "JL"]
ROWS = {"P05": [f"P05{j}" for j in JUDGES], "P15": [f"P15{j}" for j in JUDGES]}
KEYPAIRS = [(f"{p}JL", f"{p}{j}") for p in ("P05", "P15") for j in ("J15", "J3")]


def val(c, cell):
    return PRED[cell][c]["value"]


rank = {}
for c in CAND:
    rank[c] = {}
    x = {cell: SIGN[c] * val(c, cell) for cell in CELLS}
    for o in OUT:
        y = {cell: ORIENT[o] * OUT[o][cell] for cell in CELLS}
        r8 = sp([x[k] for k in CELLS], [y[k] for k in CELLS])
        d = {"rho8": r8, "perm_p_ge": float(np.mean(PERM >= r8 - 1e-12))}
        for p, cells in ROWS.items():
            d[f"rho_{p}"] = sp([x[k] for k in cells], [y[k] for k in cells])
            d[f"perfect_{p}"] = bool(np.all(np.argsort([x[k] for k in cells]) == np.argsort([y[k] for k in cells])))
        agree = n = ties = 0
        for p, cells in ROWS.items():
            for a, b in itertools.combinations(cells, 2):
                tie, _, _ = welch_tie(seed[a][SEEDKEY[o]], seed[b][SEEDKEY[o]])
                if tie:
                    ties += 1
                    continue
                n += 1
                agree += (x[a] - x[b]) * (y[a] - y[b]) > 0
        d["pairwise"] = {"agree": int(agree), "n_nontied": n, "n_tied": ties, "rate": agree / n if n else None}
        kp = {}
        for a, b in KEYPAIRS:
            tie, diff, ci = welch_tie(seed[a][SEEDKEY[o]], seed[b][SEEDKEY[o]])
            kp[f"{a}-{b}"] = {"cand_correct": bool((x[a] - x[b]) * (y[a] - y[b]) > 0), "outcome_tied": tie}
        d["keypairs"] = kp
        rank[c][o] = d

# ---------------- leave-family-out (SPEC 6.2): fit on 6 Qwen cells, predict 2 JL cells
QW = [k for k in CELLS if not k.endswith("JL")]
JL = ["P05JL", "P15JL"]


def design(cells, c):
    return np.array([[1.0, val(c, k), 1.0 if k.startswith("P15") else 0.0] for k in cells])


def ols(cells, c, o):
    X, y = design(cells, c), np.array([OUT[o][k] for k in cells])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    df = len(y) - 3
    s2 = float(res @ res / df)
    return beta, s2, df, np.linalg.inv(X.T @ X)


lfo = {}
for c in CAND:
    lfo[c] = {}
    for o in ("O1", "O2"):
        beta, s2, df, XtXi = ols(QW, c, o)
        q = stats.t.ppf(0.975, df)
        cells = {}
        for k in JL:
            x0 = design([k], c)[0]
            pr = float(x0 @ beta)
            half = q * math.sqrt(s2 * (1 + x0 @ XtXi @ x0))
            obs = OUT[o][k]
            cells[k] = {"pred": pr, "obs": obs, "abs_err": abs(pr - obs), "pi95": [pr - half, pr + half],
                        "in_pi": bool(pr - half <= obs <= pr + half)}
        lfo[c][o] = {"coef_a_b_c": beta.tolist(), "resid_sd": math.sqrt(s2), "df": df, "cells": cells,
                     "mae_JL": float(np.mean([cells[k]["abs_err"] for k in JL]))}

# ---------------- selection (SPEC 7) with S3 exclusion
missing = {c: sum(1 for k in CELLS if not np.isfinite(val(c, k))) for c in CAND}
best_base_mae = min(lfo[b]["O1"]["mae_JL"] for b in BASE)
sel = {}
for c in CAND[3:]:
    S = (rank[c]["O1"]["rho8"] + rank[c]["O2"]["rho8"]) / 2
    q = (rank[c]["O1"]["rho8"] > 0 and rank[c]["O2"]["rho8"] > 0 and lfo[c]["O1"]["mae_JL"] < best_base_mae
         and not (GPU[c] and missing[c] > 2))
    sel[c] = {"S": S, "rho_O1": rank[c]["O1"]["rho8"], "rho_O2": rank[c]["O2"]["rho8"], "mae_JL_O1": lfo[c]["O1"]["mae_JL"],
              "qualified": bool(q), "missing_cells": missing[c], "gpu": GPU[c]}
qual = [c for c in sel if sel[c]["qualified"]]
final = None
if qual:
    smax = max(sel[c]["S"] for c in qual)
    near = [c for c in qual if smax - sel[c]["S"] < 0.05]
    nogpu = [c for c in near if not GPU[c]]
    final = sorted(nogpu or near, key=CAND.index)[0]
S1_any_07 = [c for c in sel if sel[c]["rho_O1"] >= 0.7 and sel[c]["rho_O2"] >= 0.7]
stop = {"S1_triggered": final is None and not S1_any_07, "candidates_rho_ge_0.7_both": S1_any_07,
        "S3_missing_cells": missing}


def full_fit(c):
    out = {}
    for o in ("O1", "O2"):
        beta, s2, df, _ = ols(CELLS, c, o)
        out[o] = {"outcome": {"O1": "H_bar plateau h, steps 101-150, seed mean",
                              "O2": "ln t_half(0.3), AFT cell mean mu (analyze41 onset thr 0.3 + AFT, sigma prior)"}[o],
                  "model": "outcome = a + b * predictor + c * [policy == P15]", "a": float(beta[0]), "b": float(beta[1]),
                  "c": float(beta[2]), "resid_sd": math.sqrt(s2), "df": df}
    return out


sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
fp = {"experiment": "EXP-047", "written": "261010", "spec_sha256": sha(f"{ROOT}/SPEC.md"),
      "selection_rule": "SPEC section 7 (S = mean of sign-adjusted 8-cell Spearman on O1, O2; qualified if both > 0 and "
                        "leave-family-out JL MAE on O1 < best of B1-B3)",
      "final_predictor": final or "none",
      "n": "4 judges (J15, J3, J7, JL) x 2 policies (P05, P15) = 8 cells; screening only, candidates defined after EXP-041 outcomes were known",
      "code": {p: {"path": f"EXP-047/code/{p}", "sha256": sha(f"{ROOT}/code/{p}")} for p in ("predictors47.py", "analysis47.py", "common.py", "stats.py")},
      "predictors_json_sha256": sha(f"{ROOT}/results/predictors47.json"),
      "training_cells": {k: {"x": {c: val(c, k) for c in ([final] if final else []) + BASE},
                             "O1": OUT["O1"][k], "O2": OUT["O2"][k]} for k in CELLS}}
if final:
    fp["predictor"] = {"name": final, "expected_sign": SIGN[final], "fit": full_fit(final),
                       "definition": "SPEC section 4.2; computed by predictors47.py on d8(P) step-0 samples"}
fp["baselines"] = {b: {"expected_sign": SIGN[b], "fit": full_fit(b)} for b in BASE}
json.dump(fp, open(f"{ROOT}/results/FINAL_PREDICTOR.json", "w"), indent=1)
json.dump({"outcomes": OUT, "outcome_checks": check, "seed_level": seed, "rank": rank, "leave_family_out": lfo,
           "best_baseline_mae_JL_O1": best_base_mae, "selection": sel, "final": final, "stop": stop},
          open(f"{ROOT}/results/analysis47.json", "w"), indent=1, default=float)

# ---------------- print
print("checks", check)
for o in OUT:
    print(o, {k: round(v, 3) for k, v in OUT[o].items()})
print(f"{'cand':26s}" + "".join(f"{o:>34s}" for o in OUT))
for c in CAND:
    s = f"{c:26s}"
    for o in OUT:
        d = rank[c][o]
        s += f"  r8={d['rho8']:+.2f} r05={d['rho_P05']:+.2f}{'*' if d['perfect_P05'] else ' '} r15={d['rho_P15']:+.2f}{'*' if d['perfect_P15'] else ' '} pw={d['pairwise']['agree']}/{d['pairwise']['n_nontied']}"
    print(s)
print("keypairs (O1, O2):")
for c in CAND:
    print(f"  {c:26s}", {o: "".join("+" if v["cand_correct"] else ("-" if not v["outcome_tied"] else "t") for v in rank[c][o]["keypairs"].values()) for o in ("O1", "O2", "O3")})
print("LFO MAE JL:", {c: (round(lfo[c]["O1"]["mae_JL"], 3), round(lfo[c]["O2"]["mae_JL"], 3)) for c in CAND})
print("best base MAE O1", round(best_base_mae, 4))
for c, v in sel.items():
    print(c, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})
print("FINAL", final, "stop", stop)
