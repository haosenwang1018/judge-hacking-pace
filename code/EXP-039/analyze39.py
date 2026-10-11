"""EXP-039 analysis (SPEC sections 6-8). Written before any EXP-039 RL result exists.
Input: results/compact/*.json (extract39.py; EXP-034 and EXP-039 runs), results/static_lenient.json.
Output: results/analysis39[_<tag>].json.  usage: python analyze39.py [--tag stageA]
Onset t_half (primary, SPEC 6.1): lenient gold, own tau, 5-step training windows with >= 20 wrong answers,
h = (FPR - FPR0)/(1 - FPR0) >= 0.5 in window [t-4, t] and in the next window [t+1, t+5]; t in 5..T-5; else censored at T.
Model (SPEC 7): log-normal AFT with right censoring, one mean per cell, one common sigma, maximum likelihood.
Intervals: profile likelihood (chi-square 1 df), level 97.5 % (Bonferroni over the two primary contrasts) and 95 %.
"""
import argparse, glob, json, math, os
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import norm, chi2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
ap.add_argument("--exclude", default="", help="comma list of run ids to leave out (none by default)")
A = ap.parse_args()
W, MIN_WRONG = 5, 20
SL = json.load(open(f"{ROOT}/results/static_lenient.json"))
CELLS = ["P05J15", "P15J15", "P05J3", "P15J3", "P05J7", "P15J7"]
SIZE = {"P05": 0.49, "P15": 1.54, "J15": 1.54, "J3": 3.09, "J7": 7.61}   # parameters (B), non-embedding counts not used
VARIANTS = {  # name: (wrong key, fooled key, FPR0 key in static_lenient cells)
    "lenient": ("n_wl", "f_wl", "FPR0_len"), "common_tau": ("n_wl", "f_wl_c", "FPR0_len_common_tau"),
    "strict": ("n_ws", "f_ws", "FPR0_strict"), "generous": ("n_wg", "f_wg", "FPR0_gen"),
    "lenient_exact_tie": ("n_wl", "f_wl_x", "FPR0_len")}


def onset(run, var="lenient", thr=0.5):
    nk, fk, f0k = VARIANTS[var]
    f0 = SL["cells"][f"{run['cell'][:3]}x{run['cell'][3:]}"][f0k]
    st, T = run["steps"], run["steps_cfg"]

    def h(a, b):
        ts = [str(t) for t in range(a, b + 1) if str(t) in st]
        nw = sum(st[t][nk] for t in ts)
        return None if nw < MIN_WRONG else (sum(st[t][fk] for t in ts) / nw - f0) / (1 - f0)
    for t in range(W, T - W + 1):
        x, y = h(t - W + 1, t), h(t + 1, t + W)
        if x is not None and y is not None and x >= thr and y >= thr:
            return t
    return None


def interp(run, t, key):
    ev = [(e["step"], e.get(key)) for e in run["evals"] if e.get(key) is not None]
    if t is None or not ev:
        return None
    return float(np.interp(t, [a for a, _ in ev], [b for _, b in ev]))


def h_block(run, a, b, nk="n_cw", fk="f_cw", f0=None):
    st = run["steps"]
    nw = sum(st[str(t)][nk] for t in range(a, b + 1) if str(t) in st)
    fo = sum(st[str(t)][fk] for t in range(a, b + 1) if str(t) in st)
    return (None if nw == 0 else (fo / nw - f0) / (1 - f0)), nw


# ---------------- AFT with profile likelihood ----------------
PRIOR = (math.log(0.25), 0.5)   # SPEC 7.1: weak prior on sigma, ln sigma ~ N(ln 0.25, 0.5^2) (EXP-034 stage-1 sigma 0.25)


def nll(mu, ls, y, cens, ci, prior=True):
    s = math.exp(ls)
    z = (y - mu[ci]) / s
    pen = (ls - PRIOR[0]) ** 2 / (2 * PRIOR[1] ** 2) if prior else 0.0
    return -(np.sum(norm.logpdf(z[~cens]) - ls) + np.sum(norm.logsf(z[cens]))) + pen


class AFT:
    def __init__(self, y, cens, cells, prior=True):
        self.prior = prior
        self.cl = sorted(set(cells), key=CELLS.index)
        self.y, self.c = np.asarray(y, float), np.asarray(cens, bool)
        self.ci = np.array([self.cl.index(c) for c in cells])
        self.k = len(self.cl)
        self.bounds = [(-10.0, 30.0)] * self.k + [(math.log(0.01), math.log(5.0))]   # wide: all-censored cells may drift up

    def fit(self, con=None):
        """con = (a, b, d): constraint mu_a - mu_b = d (a, b cell names). Returns (max loglik, params)."""
        def f(th):
            mu = th[:self.k].copy()
            if con:
                mu[self.cl.index(con[0])] = mu[self.cl.index(con[1])] + con[2]
            return nll(mu, th[-1], self.y, self.c, self.ci, self.prior)
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

    def profile(self, a, b, levels=(0.95, 0.975)):
        if a not in self.cl or b not in self.cl:
            return None
        L0, th = self.fit()
        d_hat = th[self.cl.index(a)] - th[self.cl.index(b)]
        prof = lambda d: 2 * (L0 - self.fit((a, b, d))[0])
        out = {"estimate": float(d_hat), "sigma_mle": float(math.exp(th[-1])), "p_zero": float(chi2.sf(max(prof(0.0), 0), 1))}
        for lv in levels:
            c = chi2.ppf(lv, 1)
            g = lambda d: prof(d) - c
            lo = -math.inf if g(d_hat - 9) < 0 else brentq(g, d_hat - 9, d_hat - 1e-6, xtol=1e-4)
            hi = math.inf if g(d_hat + 9) < 0 else brentq(g, d_hat + 1e-6, d_hat + 9, xtol=1e-4)
            out[f"ci{int(lv * 1000) / 10}"] = [lo, hi]
        return out


runs = [json.load(open(p)) for p in sorted(glob.glob(f"{ROOT}/results/compact/*.json"))]
runs = [r for r in runs if r["status"] == "completed" and r["run_id"] not in A.exclude.split(",")]
res = {"tag": A.tag, "runs": []}
for r in runs:
    th = onset(r)
    e = {"run_id": r["run_id"], "cell": r["cell"], "seed": r["seed"], "T": r["steps_cfg"], "t_half": th, "censored": th is None,
         "t_half_h03": onset(r, thr=0.3), "t_half_h07": onset(r, thr=0.7), "t_half_common_tau": onset(r, "common_tau"),
         "t_half_strict": onset(r, "strict"), "t_half_generous": onset(r, "generous"), "t_half_exact_tie": onset(r, "lenient_exact_tie"), "kl_sampled_at_t_half": interp(r, th, "kl_sampled_per_seq"),
         "kl_greedy_at_t_half": interp(r, th, "kl_greedy_per_token"), "wall_hours": (r["total_seconds"] or 0) / 3600}
    f0cw = SL["cells"][f"{r['cell'][:3]}x{r['cell'][3:]}"]["FPR0_len_common_wrong"]
    e["h_common_wrong"] = {f"{a}-{b}": h_block(r, a, b, f0=f0cw) for a, b in ((1, 50), (51, 100), (101, 150), (151, 200), (201, 250)) if a <= r["steps_cfg"]}
    e["h_all_blocks"] = {f"{a}-{b}": h_block(r, a, b, "n_wl", "f_wl", SL["cells"][f"{r['cell'][:3]}x{r['cell'][3:]}"]["FPR0_len"])
                         for a, b in ((1, 50), (51, 100), (101, 150), (151, 200), (201, 250)) if a <= r["steps_cfg"]}
    st = r["steps"].values()
    n = sum(s["n"] for s in st)
    e["frac_reward_lt_0.05"] = sum(s["sat0"] for s in st) / n
    e["frac_reward_gt_0.95"] = sum(s["sat1"] for s in st) / n
    e["tie_at_tau_share_of_wrong"] = sum(s["tie_wl"] for s in st) / max(sum(s["n_wl"] for s in st), 1)
    e["acc0"] = r["evals"][0]["acc"] if r["evals"] else None
    e["acc_final"] = r["evals"][-1]["acc"] if r["evals"] else None
    e["format_final"] = r["evals"][-1]["format_rate"] if r["evals"] else None
    res["runs"].append(e)


def model_for(key="t_half", prior=True):
    y = [math.log(e[key] if e[key] is not None else e["T"]) for e in res["runs"]]
    c = [e[key] is None for e in res["runs"]]
    return AFT(y, c, [e["cell"] for e in res["runs"]], prior)


res["cells"] = {c: {"n": sum(e["cell"] == c for e in res["runs"]),
                    "t_half": [e["t_half"] if e["t_half"] else f">{e['T']}" for e in res["runs"] if e["cell"] == c]} for c in CELLS}
M = model_for()
L0, th = M.fit()
res["aft"] = {"sigma": math.exp(th[-1]), "mu": {c: float(th[i]) for i, c in enumerate(M.cl)}, "loglik": L0}
prim = {J: M.profile(f"P15{J}", f"P05{J}") for J in ("J15", "J3", "J7")}
res["Delta_P"] = prim
# equal-gap pairs (SPEC 6.4): gap+floor predicts ~0 for both
res["equal_gap"] = {"P05J3_minus_P15J7": M.profile("P05J3", "P15J7"), "P15J3_minus_P05J15": M.profile("P15J3", "P05J15")}
# Holm over the two primary tests (Delta_P|J3, Delta_P|J7), on the LR p-values of delta = 0
ps = {J: prim[J]["p_zero"] for J in ("J3", "J7") if prim[J]}
order = sorted(ps, key=ps.get)
res["holm"] = {J: {"p": ps[J], "reject_at_0.05": all(ps[order[j]] <= 0.05 / (len(ps) - j) for j in range(i + 1))} for i, J in enumerate(order)}
# decision rules (SPEC 7.2), on the 97.5 % profile intervals
d3, d7 = prim.get("J3"), prim.get("J7")
v = "not computable"
if d3:
    lo3, hi3 = d3["ci97.5"]
    inc0 = lambda ci: ci[0] <= 0 <= ci[1]
    if hi3 < -0.8:
        v = "R1 gap + floor"
    elif d7 and inc0(d3["ci97.5"]) and lo3 > -0.5 and d7["ci97.5"][1] < 0:
        v = "R2 policy effect only under the strong judge (interaction)"
    elif d7 and inc0(d3["ci97.5"]) and inc0(d7["ci97.5"]):
        v = "R3 judge dominant"
    else:
        v = "R4 unresolved"
res["verdict"] = v
# SPEC supplement 2: the same decision rules under the generous gold; if the verdict differs, it is downgraded
def verdict(P):
    d3, d7 = P.get("J3"), P.get("J7")
    if not d3:
        return "not computable"
    inc0 = lambda ci: ci[0] <= 0 <= ci[1]
    if d3["ci97.5"][1] < -0.8:
        return "R1 gap + floor"
    if d7 and inc0(d3["ci97.5"]) and d3["ci97.5"][0] > -0.5 and d7["ci97.5"][1] < 0:
        return "R2 policy effect only under the strong judge (interaction)"
    if d7 and inc0(d3["ci97.5"]) and inc0(d7["ci97.5"]):
        return "R3 judge dominant"
    return "R4 unresolved"
assert verdict(prim) == v
Mg = model_for("t_half_generous")
res["Delta_P_generous"] = {J: Mg.profile(f"P15{J}", f"P05{J}") for J in ("J15", "J3", "J7")}
res["verdict_generous"] = verdict(res["Delta_P_generous"])
res["verdict_final"] = v if res["verdict_generous"] == v else f"depends on gold definition (lenient: {v}; generous: {res['verdict_generous']})"
# sensitivity: pure maximum likelihood (no prior on sigma)
Mm = model_for(prior=False)
res["Delta_P_mle_no_prior"] = {J: Mm.profile(f"P15{J}", f"P05{J}") for J in ("J15", "J3", "J7")}
# secondary endpoints (SPEC 6.3): same Delta_P under other onset definitions (description only)
res["secondary_Delta_P"] = {}
for key in ("t_half_h03", "t_half_h07", "t_half_common_tau", "t_half_strict", "t_half_generous", "t_half_exact_tie"):
    Mk = model_for(key)
    res["secondary_Delta_P"][key] = {J: Mk.profile(f"P15{J}", f"P05{J}", levels=(0.95,)) for J in ("J15", "J3", "J7")}
os.makedirs(f"{ROOT}/results", exist_ok=True)
fn = f"{ROOT}/results/analysis39{'_' + A.tag if A.tag else ''}.json"
json.dump(res, open(fn, "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1, default=float))
for e in res["runs"]:
    print(e["run_id"][-14:], e["t_half"], e["t_half_strict"], e["t_half_common_tau"], e["t_half_h03"], e["t_half_h07"],
          {k: (round(v[0], 2) if v[0] is not None else None, v[1]) for k, v in e["h_common_wrong"].items()})
