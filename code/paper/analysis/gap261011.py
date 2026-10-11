"""PAPER-T11 revision 261011, item 2: scope and re-test the gap account (no GPU; existing logs only).
Reuses onset(), plateau(), AFT, welch from EXP-041/code/analyze41.py (definitions unchanged), extended with the
EXP-046 cells exactly as EXP-046/code/analysis46.py does. Output: gap261011.json"""
import glob, json, math, os
import numpy as np
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(f"{H}/../..")
E39, E41, E46 = f"{RES}/EXP-039", f"{RES}/EXP-041", f"{RES}/EXP-046"
src = open(f"{E41}/code/analyze41.py").read(); ns = {"__file__": f"{E41}/code/analyze41.py"}
exec(src[: src.index("paths = sorted(")], ns)
onset, AFT, CELLS, SL, welch, interp = ns["onset"], ns["AFT"], ns["CELLS"], ns["SL"], ns["welch"], ns["interp"]
NEW = ["P05JL8", "P05JO1", "P05JO7"]
SL46 = {}
for h in ("gpu-server-2", "gpu-server-1"):
    SL46.update(json.load(open(f"{E46}/results/{h}/static_lenient.json"))["cells"])
for c in NEW:
    SL["cells"][f"{c[:3]}x{c[3:]}"] = SL46[f"{c[:3]}x{c[3:]}"]; CELLS.append(c)
paths = sorted(glob.glob(f"{E39}/results/compact/*.json")) + sorted(glob.glob(f"{E41}/results/compact/*.json")) + sorted(glob.glob(f"{E46}/results/compact/exp046-*.json"))
runs = [r for r in (json.load(open(p)) for p in paths) if r["status"] == "completed"]
A46 = json.load(open(f"{E46}/results/analysis46.json"))
GOLD = json.load(open(f"{H}/gold261011.json"))["cells"]
R = []
for r in runs:
    t5, t3 = onset(r, 0.5), onset(r, 0.3)
    has_s = any(e.get("kl_sampled_per_seq") is not None and e["step"] > 0 for e in r["evals"])
    kmax = max([e["step"] for e in r["evals"] if e.get("kl_sampled_per_seq") is not None], default=0)
    R.append({"run": r["run_id"], "cell": r["cell"], "T": r["steps_cfg"], "t05": t5, "t03": t3,
              "kl05": interp(r, t5, "kl_sampled_per_seq") if has_s else None,
              "kl05_cens_lower": interp(r, r["steps_cfg"], "kl_sampled_per_seq") if (has_s and t5 is None) else None,
              "kl03": interp(r, t3, "kl_sampled_per_seq") if has_s else None,
              "kl03_greedy": interp(r, t3, "kl_greedy_per_seq"), "kl05_greedy": interp(r, t5, "kl_greedy_per_seq"),
              "has_sampled_kl": has_s})
fits = {}
for key in ("t05", "t03"):
    M = AFT([math.log(e[key] or e["T"]) for e in R], [e[key] is None for e in R], [e["cell"] for e in R])
    fits[key] = M
# sanity: same geo means as analysis46
_, th = fits["t05"].fit()
chk = max(abs(math.exp(th[i]) - A46["onset"]["t05"]["geo_mean"][c]) for i, c in enumerate(fits["t05"].cl))
out = {"check_geo_vs_analysis46": chk, "runs": R}
PAIRS = {"pair1_P05J3_minus_P15J7": ("P05J3", "P15J7"), "pair2_P15J3_minus_P05J15": ("P15J3", "P05J15")}
WITHIN = {"Qwen_J15_minus_J3_P05": ("P05J15", "P05J3"), "Qwen_J3_minus_J7_P05": ("P05J3", "P05J7"),
          "Llama_JL_minus_JL8_P05": ("P05JL", "P05JL8"), "OLMo_JO1_minus_JO7_P05": ("P05JO1", "P05JO7"),
          "DeltaP_J15": ("P15J15", "P05J15"), "DeltaP_J3": ("P15J3", "P05J3"), "DeltaP_J7": ("P15J7", "P05J7")}
A = json.load(open(f"{E46}/results/analysis46.json"))["cells"]
pl = lambda c, k="E1": A[c][k]["seeds"]
def logw(x, y):
    x = [math.log(v) for v in x if v]; y = [math.log(v) for v in y if v]
    return welch(x, y) if len(x) > 1 and len(y) > 1 else None
res = {}
for name, (a, b) in {**PAIRS, **WITHIN}.items():
    d = {}
    for key in ("t05", "t03"):
        d[key + "_aft38"] = fits[key].profile(a, b, levels=(0.95, 0.975))
        d[key + "_seed_welch_log"] = logw([e[key] for e in R if e["cell"] == a], [e[key] for e in R if e["cell"] == b])
    d["plateau_welch"] = welch(pl(a), pl(b)); d["plateau_generous_welch"] = welch(pl(a, "E1_generous"), pl(b, "E1_generous"))
    for k in ("kl03", "kl05"):
        xa = [e[k] for e in R if e["cell"] == a and e[k]]; xb = [e[k] for e in R if e["cell"] == b and e[k]]
        d[k + "_values"] = {a: xa, b: xb}; d[k + "_welch_log"] = logw(xa, xb)
    res[name] = d
out["contrasts"] = res
# capability gap: judge step-0 AUC (cell-specific, lenient) vs policy step-0 greedy accuracy
lg = lambda p: math.log(p / (1 - p))
cap = {}
for c in CELLS:
    auc = SL["cells"][f"{c[:3]}x{c[3:]}"]["auc_len"]; acc = GOLD[c]["acc0"]
    cap[c] = {"auc": auc, "acc0": acc, "gap_diff": auc - acc, "gap_logit": lg(auc) - lg(acc),
              "mu_t05": A46["onset"]["t05"]["mu"][c], "mu_t03": A46["onset"]["t03"]["mu"][c], "Hbar": A[c]["E1"]["mean"],
              "param_ratio": ns["SIZE"].get(c[3:], {"JL8": 8.03, "JO1": 1.48, "JO7": 7.3}.get(c[3:])) / ns["SIZE"][c[:3]]}
out["capability"] = cap
sp = lambda x, y: float(stats.spearmanr(x, y).correlation)
C = CELLS
out["spearman_11cells"] = {o: {g: sp([cap[c][g] for c in C], [cap[c][o] for c in C]) for g in ("gap_logit", "gap_diff", "auc", "param_ratio")}
                          for o in ("mu_t03", "mu_t05", "Hbar")}
json.dump(out, open(f"{H}/gap261011.json", "w"), indent=1, default=float)
print("check", chk)
f = lambda iv: "[%.2f, %.2f]" % tuple(iv)
for n, d in res.items():
    print(f"\n{n}")
    for key in ("t05", "t03"):
        p = d[key + "_aft38"]; w = d[key + "_seed_welch_log"]
        print(f"  {key}: AFT38 {p['estimate']:+.2f} 95% {f(p['ci95.0'])} 97.5% {f(p['ci97.5'])} | seed-welch-log {('%+.2f %s' % (w['D'], f(w['ci95']))) if w else '-'}")
    print(f"  plateau {d['plateau_welch']['D']:+.3f} {f(d['plateau_welch']['ci95'])} | generous {d['plateau_generous_welch']['D']:+.3f} {f(d['plateau_generous_welch']['ci95'])}")
    for k in ("kl03", "kl05"):
        w = d[k + "_welch_log"]; print(f"  {k} {d[k+'_values']} welch-log {('%+.2f %s' % (w['D'], f(w['ci95']))) if w else '-'}")
print()
for c in C: print(c, {k: round(v, 3) for k, v in cap[c].items()})
print(json.dumps(out["spearman_11cells"], indent=0))
for e in R:
    if e["cell"] in ("P15J7", "P05J7"): print(e["run"], e["t05"], e["kl05"], e["kl05_cens_lower"], e["t03"], e["kl03"], e["has_sampled_kl"])
print("runs with sampled KL:", sum(e["has_sampled_kl"] for e in R), "of", len(R))
