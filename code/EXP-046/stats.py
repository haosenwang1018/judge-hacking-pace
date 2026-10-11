"""EXP-023 helpers (copied from EXP-021 analyze.py: rank AUC, torch-LBFGS logistic regression), plus GRPO
group statistics used by step 1 and by the RL logs."""
import numpy as np, torch


def auc(s, t):
    s, t = np.asarray(s, float), np.asarray(t, int)
    if t.sum() == 0 or t.sum() == len(t):
        return float("nan")
    r = np.argsort(np.argsort(s, kind="mergesort"), kind="mergesort").astype(float) + 1
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.bincount(inv, weights=r) / cnt)[inv]
    n1, n0 = t.sum(), (1 - t).sum()
    return float((r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_lr(X, t, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = torch.tensor((X - mu) / sd)
    tt = torch.tensor(t, dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xs @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, tt) + 0.5 * lam * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return {"mu": mu, "sd": sd, "w": w.detach().numpy(), "b": float(b.detach())}


def advantages(r, G, mode, eps=1e-4):
    """TRL 1.14 GRPO advantages. r: (n,) with groups of G consecutive. mode: group | batch | none.
    batch: std over the whole array passed in (call per generation batch)."""
    r = np.asarray(r, float).reshape(-1, G)
    a = r - r.mean(1, keepdims=True)
    if mode == "group":
        a = a / (r.std(1, ddof=1, keepdims=True) + eps)
    elif mode == "batch":
        a = a / (r.std(ddof=1) + eps)
    return a.reshape(-1)


def group_stats(r, c, G, batch_groups=8, seed=0):
    """Step-0 within-group diagnostics of a reward r against gold correctness c (0/1), groups of G.
    Returns within-group std summary, within-group AUC (mixed groups), d' (mean gap / pooled within-class sd),
    and for each advantage mode: cosine(A, A_gold), share of sum A^2 on gold-uniform groups, mean |A|."""
    r = np.asarray(r, float).reshape(-1, G)
    c = np.asarray(c, int).reshape(-1, G)
    n_g = len(r)
    sd = r.std(1, ddof=1)
    mixed = (c.sum(1) > 0) & (c.sum(1) < G)
    # within-group pairwise AUC pooled over mixed groups
    wins = tot = 0.0
    gaps, var_pool, dof = [], 0.0, 0
    for g in np.where(mixed)[0]:
        p, q = r[g][c[g] == 1], r[g][c[g] == 0]
        d = p[:, None] - q[None, :]
        wins += (d > 0).sum() + 0.5 * (d == 0).sum()
        tot += d.size
        gaps.append(p.mean() - q.mean())
        for v in (p, q):
            if len(v) > 1:
                var_pool += ((v - v.mean()) ** 2).sum()
                dof += len(v) - 1
    pooled = np.sqrt(var_pool / max(dof, 1))
    A_gold = (c - c.mean(1, keepdims=True)).reshape(-1).astype(float)
    res = {"n_groups": int(n_g), "frac_mixed": float(mixed.mean()), "frac_allwrong": float((c.sum(1) == 0).mean()),
           "frac_allcorrect": float((c.sum(1) == G).mean()),
           "sd_within_median": float(np.median(sd)), "sd_within_mean": float(sd.mean()),
           "sd_within_p10": float(np.quantile(sd, .1)), "sd_within_p90": float(np.quantile(sd, .9)),
           "sd_global": float(r.std(ddof=1)), "sd_within_over_global": float(np.median(sd) / r.std(ddof=1)),
           "amp_median": float(np.median(1 / (sd + 1e-4))),
           "sd_within_mixed_median": float(np.median(sd[mixed])), "sd_within_uniform_median": float(np.median(sd[~mixed])),
           "auc_within": float(wins / tot) if tot else float("nan"),
           "dprime_within": float(np.mean(gaps) / pooled) if gaps else float("nan"),
           "auc_pooled": auc(r.reshape(-1), c.reshape(-1))}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_g)
    for mode in ("group", "batch", "none"):
        A = np.zeros(n_g * G)
        for b in range(0, n_g, batch_groups):  # batch std is computed per generation batch (8 groups x 8 = 64)
            gi = perm[b:b + batch_groups]
            ii = (gi[:, None] * G + np.arange(G)[None, :]).reshape(-1)
            A[ii] = advantages(r[gi].reshape(-1), G, mode)
        A2 = (A.reshape(-1, G) ** 2).sum(1)
        res[mode] = {"cos_A_gold": float(A @ A_gold / (np.linalg.norm(A) * np.linalg.norm(A_gold) + 1e-12)),
                     "share_A2_uniform": float(A2[~mixed].sum() / A2.sum()),
                     "share_A2_allwrong": float(A2[c.sum(1) == 0].sum() / A2.sum()),
                     "mean_abs_A": float(np.abs(A).mean()),
                     "sign_agree_mixed": float(np.mean([(np.sign(A.reshape(-1, G)[g]) == np.sign(A_gold.reshape(-1, G)[g])).mean()
                                                        for g in np.where(mixed)[0]]))}
    return res
