"""EXP-041 figures (formal paper look: serif, no in-chart title, caption under the chart with bold 'Figure N:').
Input: results/compact (EXP-041) + ../EXP-039/results/compact, results/static_lenient.json, results/analysis41.json.
Output: figures/fig1_h_curves_P05.{pdf,png}, figures/fig2_plateau_vs_auc.{pdf,png}"""
import glob, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E39 = os.path.join(os.path.dirname(ROOT), "EXP-039")
SL = json.load(open(f"{ROOT}/results/static_lenient.json"))["cells"]
A = json.load(open(f"{ROOT}/results/analysis41.json"))
os.makedirs(f"{ROOT}/figures", exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "mathtext.fontset": "stix"})
JUD = {"J15": ("Qwen2.5-1.5B", "#8C8C8C", "-"), "J3": ("Qwen2.5-3B", "#4C72B0", "-"),
       "J7": ("Qwen2.5-7B", "#55A868", "-"), "JL": ("Llama-3.2-3B", "#C44E52", "-")}
runs = [json.load(open(p)) for p in sorted(glob.glob(f"{E39}/results/compact/*.json")) + sorted(glob.glob(f"{ROOT}/results/compact/*.json"))]


def caption(fig, n, text, y):
    fig.text(0.0, y, f"$\\bf{{Figure\\ {n}:}}$ " + textwrap.fill(text, 105), ha="left", va="top", fontsize=8.5)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{ROOT}/figures/{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def hcurve(r, W=5):
    f0 = SL[f"{r['cell'][:3]}x{r['cell'][3:]}"]["FPR0_len"]
    st = r["steps"]
    xs, ys = [], []
    for b in range(W, r["steps_cfg"] + 1, W):
        n = sum(st[str(t)]["n_wl"] for t in range(b - W + 1, b + 1))
        f = sum(st[str(t)]["f_wl"] for t in range(b - W + 1, b + 1))
        xs.append(b); ys.append((f / n - f0) / (1 - f0) if n >= 20 else np.nan)
    return np.array(xs), np.array(ys)


# Figure 1: h vs step, 0.5B policy, four judges (seed mean line, seed range band), steps 1-200
fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.axvspan(101, 150, color="#EEEEEE", lw=0, zorder=0)
for J, (lab, col, ls) in JUD.items():
    rs = [r for r in runs if r["cell"] == f"P05{J}"]
    curves = [hcurve(r) for r in rs]
    L = min(200, min(len(x) * 5 for x, _ in curves))
    Y = np.array([y[:L // 5] for _, y in curves]); X = curves[0][0][:L // 5]
    ax.fill_between(X, np.nanmin(Y, 0), np.nanmax(Y, 0), color=col, alpha=0.15, lw=0)
    ax.plot(X, np.nanmean(Y, 0), color=col, lw=1.4 if J == "JL" else 1.0, ls=ls, label=f"{lab} ({len(rs)})")
ax.axhline(0, color="black", lw=0.4)
ax.set_xlim(0, 200); ax.set_ylim(-0.2, 1.05)
ax.set_xlabel("Training step"); ax.set_ylabel("Fooled fraction $h$")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=7.5, title="Judge (seeds)", title_fontsize=7.5)
fig.subplots_adjust(bottom=0.17)
caption(fig, 1, "Fooled fraction h of the frozen judge during GRPO training of the Qwen2.5-0.5B policy on GSM8K. "
        "h = (FPR - FPR0)/(1 - FPR0) over lenient-wrong training rollouts in 5-step windows; lines are seed means, bands "
        "are the seed range. The grey region (steps 101-150) is the pre-registered plateau window. Qwen judges are from EXP-039; the Qwen2.5-1.5B and 7B curves end at step 150, the shortest run in those cells.", -0.04)
save(fig, "fig1_h_curves_P05")

# Figure 2: plateau height per seed vs step-0 judge AUC, both policies
fig, ax = plt.subplots(figsize=(6.0, 3.0))
POL = {"P05": ("0.5B policy", "o"), "P15": ("1.5B policy", "s")}
for P, (plab, mk) in POL.items():
    for J, (lab, col, _) in JUD.items():
        c = f"{P}{J}"
        auc = SL[f"{P}x{J}"]["auc_len"]
        hs = A["plateau_cells"][c]["seeds"]
        jit = np.linspace(-0.003, 0.003, len(hs))
        ax.scatter(auc + jit, hs, s=14, marker=mk, facecolor=col if P == "P05" else "white", edgecolor=col, lw=0.8, zorder=3)
        ax.plot([auc - 0.008, auc + 0.008], [np.mean(hs)] * 2, color=col, lw=1.2)
for J, (lab, col, _) in JUD.items():
    ax.scatter([], [], s=14, marker="o", color=col, label=f"{lab} judge")
ax.scatter([], [], s=14, marker="o", facecolor="black", edgecolor="black", label="0.5B policy")
ax.scatter([], [], s=14, marker="s", facecolor="white", edgecolor="black", label="1.5B policy")
ax.set_xlabel("Judge AUC at step 0 (lenient gold)"); ax.set_ylabel("Plateau height $\\bar{H}$ (steps 101-150)")
ax.set_ylim(0.2, 1.02)
ax.legend(loc="lower left", fontsize=7.5, ncol=2)
fig.subplots_adjust(bottom=0.17)
caption(fig, 2, "Plateau height per seed (markers) and cell mean (short bars) against the judge's step-0 AUC on its own "
        "policy's samples. The Llama-3.2-3B judge has the step-0 AUC of the Qwen2.5-1.5B judge but the plateau height of "
        "the Qwen2.5-3B judge (0.5B policy); with the 1.5B policy (2 seeds) it is higher than the Qwen2.5-3B judge.", -0.04)
save(fig, "fig2_plateau_vs_auc")
print("ok")
