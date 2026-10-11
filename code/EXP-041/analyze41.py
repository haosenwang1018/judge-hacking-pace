"""EXP-041 analysis (SPEC sections 6-8; PROCESS analysis steps 2, 3, 5). Definitions are not changed.
Input: results/compact/*.json (EXP-041, 6 runs) + ../EXP-039/results/compact/*.json (20 runs), results/static_lenient.json.
Output: results/analysis41.json, results/plateau41.csv, results/onset41.csv.
Plateau H (SPEC 6.1): steps 101-150 inclusive, pooled f_wl / n_wl, h = (f/n - FPR0_len)/(1 - FPR0_len); cell = mean of seeds.
Onset (SPEC 6.2): onset() and AFT copied from ../EXP-039/code/analyze39.py (same sigma prior), cells P05JL, P15JL added."""
import glob, json, math, os, sys
import numpy as np
from scipy import stats
from scipy.optimize import minimize, brentq
from scipy.stats import norm, chi2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E39 = os.path.join(os.path.dirname(ROOT), "EXP-039")
SL = json.load(open(f"{ROOT}/results/static_lenient.json"))
W, MIN_WRONG = 5, 20
CELLS = ["P05J15", "P15J15", "P05J3", "P15J3", "P05J7", "P15J7", "P05JL", "P15JL"]
SIZE = {"P05": 0.49, "P15": 1.54, "J15": 1.54, "J3": 3.09, "J7": 7.61, "JL": 3.21}
cellkey = lambda c: f"{c[:3]}x{c[3:]}"


def onset(run, thr=0.5):   # = analyze39.onset, lenient variant
    f0 = SL["cells"][cellkey(run["cell"])]["FPR0_len"]
    st, T = run["steps"], run["steps_cfg"]

    def h(a, b):
        ts = [str(t) for t in range(a, b + 1) if str(t) in st]
        nw = sum(st[t]["n_wl"] for t in ts)
        return None if nw < MIN_WRONG else (sum(st[t]["f_wl"] for t in ts) / nw - f0) / (1 - f0)
    for t in range(W, T - W + 1):
        x, y = h(t - W + 1, t), h(t + 1, t + W)
        if x is not None and y is not None and x >= thr and y >= thr:
            return t
    return None


def plateau(run, a=101, b=150):
    f0 = SL["cells"][cellkey(run["cell"])]["FPR0_len"]
    st = run["steps"]
    n = sum(st[str(t)]["n_wl"] for t in range(a, b + 1))
    f = sum(st[str(t)]["f_wl"] for t in range(a, b + 1))
    return (f / n - f0) / (1 - f0), n, f


def interp(run, t, key):
    ev = [(e["step"], e.get(key)) for e in run["evals"] if e.get(key) is not None]
    if t is None or not ev:
        return None
    return float(np.interp(t, [a for a, _ in ev], [b for _, b in ev]))


PRIOR = (math.log(0.25), 0.5)


def nll(mu, ls, y, cens, ci):
    s = math.exp(ls)
    z = (y - mu[ci]) / s
    pen = (ls - PRIOR[0]) ** 2 / (2 * PRIOR[1] ** 2)
    return -(np.sum(norm.logpdf(z[~cens]) - ls) + np.sum(norm.logsf(z[cens]))) + pen


class AFT:   # = analyze39.AFT (prior on)
    def __init__(self, y, cens, cells):
        self.cl = sorted(set(cells), key=CELLS.index)
        self.y, self.c = np.asarray(y, float), np.asarray(cens, bool)
        self.ci = np.array([self.cl.index(c) for c in cells])
        self.k = len(self.cl)
        self.bounds = [(-10.0, 30.0)] * self.k + [(math.log(0.01), math.log(5.0))]

    def fit(self, con=None):
        def f(th):
            mu = th[:self.k].copy()
            if con:
                mu[self.cl.index(con[0])] = mu[self.cl.index(con[1])] + con[2]
            return nll(mu, th[-1], self.y, self.c, self.ci)
        best = None
        for s0 in (0.3, 0.8):
            x0 = np.r_[[np.mean(self.y[self.ci == i]) + 0.2 * np.mean(self.c[self.ci == i]) for i in range(self.k)], math.log(s0)]
            r = minimize(f, x0, method="L-BFGS-B", bounds=self.bounds)
            r2 = minimize(f, r.x, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-7, "fatol": 1e-9})
            for rr in (r, r2):
                if best is None or rr.fun < best.fun:
                    best = rr
        th = best.x.copy()
        if con:
            th[self.cl.index(con[0])] = th[self.cl.index(con[1])] + con[2]
        return -best.fun, th

    def profile(self, a, b, levels=(0.95,)):
        L0, th = self.fit()
        d_hat = th[self.cl.index(a)] - th[self.cl.index(b)]
        prof = lambda d: 2 * (L0 - self.fit((a, b, d))[0])
        out = {"estimate": float(d_hat), "p_zero": float(chi2.sf(max(prof(0.0), 0), 1))}
        for lv in levels:
            c = chi2.ppf(lv, 1)
            g = lambda d: prof(d) - c
            lo = -math.inf if g(d_hat - 9) < 0 else brentq(g, d_hat - 9, d_hat - 1e-6, xtol=1e-4)
            hi = math.inf if g(d_hat + 9) < 0 else brentq(g, d_hat + 1e-6, d_hat + 9, xtol=1e-4)
            out[f"ci{int(lv * 1000) / 10}"] = [lo, hi]
        return out


def welch(x, y, lv=0.95):
    x, y = np.asarray(x), np.asarray(y)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    d = x.mean() - y.mean()
    q = stats.t.ppf(0.5 + lv / 2, df)
    return {"D": d, "se": se, "df": df, "ci95": [d - q * se, d + q * se]}


paths = sorted(glob.glob(f"{E39}/results/compact/*.json")) + sorted(glob.glob(f"{ROOT}/results/compact/*.json"))
runs = [json.load(open(p)) for p in paths]
runs = [r for r in runs if r["status"] == "completed"]
res = {"runs": []}
for r in runs:
    H, n, f = plateau(r) if r["steps_cfg"] >= 150 else (None, 0, 0)
    th, th3 = onset(r), onset(r, 0.3)
    st = r["steps"].values()
    res["runs"].append({"run_id": r["run_id"], "cell": r["cell"], "seed": r["seed"], "T": r["steps_cfg"], "n_steps": len(r["steps"]),
                        "H": H, "n_wrong_101_150": n, "f_101_150": f, "t_half": th, "t_half_h03": th3,
                        "kl_sampled_at_t_half": interp(r, th, "kl_sampled_per_seq"),
                        "acc0": r["evals"][0]["acc"], "acc_final": r["evals"][-1]["acc"], "fmt0": r["evals"][0]["format_rate"],
                        "fmt_final": r["evals"][-1]["format_rate"], "len0": r["evals"][0]["mean_len"], "len_final": r["evals"][-1]["mean_len"],
                        "eval_final_step": r["evals"][-1]["step"], "wall_hours": (r["total_seconds"] or 0) / 3600})
R = res["runs"]
cellH = {c: [e["H"] for e in R if e["cell"] == c] for c in CELLS}
res["plateau_cells"] = {c: {"seeds": v, "mean": float(np.mean(v))} for c, v in cellH.items()}
res["D_P05"] = welch(cellH["P05JL"], cellH["P05J3"])
res["D_P15_descr"] = welch(cellH["P15JL"], cellH["P15J3"])
res["D_P05_vs_J15_descr"] = welch(cellH["P05JL"], cellH["P05J15"])
nP05 = sum(e["n_wrong_101_150"] for e in R if e["cell"] == "P05JL")
res["P05JL_wrong_101_150_total"] = nP05
lo, hi = res["D_P05"]["ci95"]
HL = res["plateau_cells"]["P05JL"]["mean"]
if nP05 < 200:
    v = "not identifiable (stop rule: < 200 wrong answers)"
elif lo >= -0.15 and hi <= 0.15:
    v = "V1"
elif HL >= 0.90 and lo > 0:
    v = "V2"
elif hi < -0.15:
    v = "V3"
else:
    v = "V4"
res["verdict"] = v
# seeds needed (V4 case): per-arm n so that the Welch 95% CI half-width (~1.96*sqrt(sx2/n+sy2/n), normal approx) fits inside
# the +-0.15 band around the observed D; reported for information only
sx, sy = np.var(cellH["P05JL"], ddof=1), np.var(cellH["P05J3"], ddof=1)
res["sd"] = {"P05JL": float(math.sqrt(sx)), "P05J3": float(math.sqrt(sy))}


def aft(key):
    y = [math.log(e[key] if e[key] is not None else e["T"]) for e in R]
    c = [e[key] is None for e in R]
    return AFT(y, c, [e["cell"] for e in R])


res["onset"] = {}
for key in ("t_half", "t_half_h03"):
    M = aft(key)
    L0, th = M.fit()
    res["onset"][key] = {"sigma": math.exp(th[-1]), "geo_mean": {c: math.exp(th[i]) for i, c in enumerate(M.cl)},
                         "per_run": {c: [e[key] if e[key] else f">{e['T']}" for e in R if e["cell"] == c] for c in CELLS},
                         "contrasts": {f"{P}JL-{P}{J}": M.profile(f"{P}JL", f"{P}{J}") for P in ("P05", "P15") for J in ("J3", "J15")}}
json.dump(res, open(f"{ROOT}/results/analysis41.json", "w"), indent=1, default=float)
with open(f"{ROOT}/results/plateau41.csv", "w") as fo:
    fo.write("run_id,cell,seed,steps,H_101_150,n_wrong_101_150,fooled_101_150,t_half_0.5,t_half_0.3,kl_sampled_at_t_half,acc0,acc_final,fmt_final,len0,len_final,wall_hours\n")
    for e in R:
        fo.write(",".join(str(x if not isinstance(x, float) else round(x, 4)) for x in (e["run_id"], e["cell"], e["seed"], e["T"], e["H"], e["n_wrong_101_150"], e["f_101_150"], e["t_half"] or f">{e['T']}", e["t_half_h03"] or f">{e['T']}", e["kl_sampled_at_t_half"], e["acc0"], e["acc_final"], e["fmt_final"], e["len0"], e["len_final"], e["wall_hours"])) + "\n")
print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1, default=lambda x: round(float(x), 4)))
for e in R:
    print(e["run_id"][-14:], e["n_steps"], round(e["H"], 3), e["n_wrong_101_150"], e["t_half"], e["t_half_h03"], e["acc0"], e["acc_final"], e["fmt_final"], e["len0"], e["len_final"], e["kl_sampled_at_t_half"] and round(e["kl_sampled_at_t_half"], 1), round(e["wall_hours"], 2))
