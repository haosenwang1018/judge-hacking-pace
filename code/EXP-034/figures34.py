"""EXP-034 figures from results/analysis.json (formal paper style: serif, caption under the chart, no in-chart title).
usage: python figures34.py   -> figures/fig1_hack_curves.{png,pdf}, figures/fig2_onset_vs_gap.{png,pdf}"""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = json.load(open(f"{ROOT}/results/analysis.json")) if os.path.exists(f"{ROOT}/results/analysis.json") else \
    json.load(open(f"{ROOT}/results/analysis_stage1.json"))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "mathtext.fontset": "stix", "axes.spines.right": False, "axes.linewidth": 0.6})
CELLS = [("P05J15", "0.5B policy, 1.5B judge"), ("P05J7", "0.5B policy, 7B judge"),
         ("P15J15", "1.5B policy, 1.5B judge"), ("P15J7", "1.5B policy, 7B judge")]
COL = {"J15": "#c0504d", "J7": "#2f5f8a"}
NP = {"P05": 0.49, "P15": 1.54, "J15": 1.54, "J7": 7.61}
os.makedirs(f"{ROOT}/figures", exist_ok=True)


def caption(fig, text, y=0.01):
    fig.text(0.01, y, text, ha="left", va="bottom", wrap=True, fontsize=8.5)


# Figure 1: normalized hacked fraction h(t), training window, per cell
fig, axs = plt.subplots(2, 2, figsize=(6.4, 4.6), sharex=True, sharey=True)
for ax, (c, lab) in zip(axs.flat, CELLS):
    for r in [r for r in A["runs"] if r["cell"] == c and r["status"] == "completed"]:
        h = r["train_window_h"]
        t = sorted(int(k) for k in h if h[k] is not None)
        ax.plot(t, [h[str(k)] if str(k) in h else h[k] for k in t], lw=0.9, color=COL[c[3:]], alpha=0.8)
        if r["t_half"]:
            ax.plot([r["t_half"]], [0.5], "o", ms=3, color="black")
    ax.axhline(0.5, ls="--", lw=0.6, color="gray")
    ax.axhline(0, lw=0.4, color="gray")
    ax.set_ylim(-0.6, 1.05)
    ax.text(0.98, 0.04, lab, transform=ax.transAxes, va="bottom", ha="right", fontsize=8.5)
for ax in axs[1]:
    ax.set_xlabel("RL step")
for ax in axs[:, 0]:
    ax.set_ylabel("hacked share of headroom, h")
fig.tight_layout(rect=(0, 0.13, 1, 1))
caption(fig, r"$\mathbf{Figure\ 1:}$ Share of the remaining false-positive headroom taken by the judge on wrong training "
             "answers, h = (FPR − FPR$_0$)/(1 − FPR$_0$), in a 5-step window. One line per seed. Dots mark the "
             "pre-registered onset t$_{1/2}$ (first step with h ≥ 0.5 in two consecutive windows).")
for e in ("png", "pdf"):
    fig.savefig(f"{ROOT}/figures/fig1_hack_curves.{e}", dpi=200)
plt.close(fig)

# Figure 2: onset vs log size ratio judge/policy
fig, ax = plt.subplots(figsize=(4.6, 3.8))
rng = np.random.default_rng(0)
for c, lab in CELLS:
    P, J = c[:3], c[3:]
    x = math.log(NP[J] / NP[P])
    for r in [r for r in A["runs"] if r["cell"] == c and r["status"] == "completed"]:
        xx = x + rng.uniform(-0.04, 0.04)
        if r["t_half"]:
            ax.plot(xx, r["t_half"], "o" if P == "P05" else "s", ms=5, color=COL[J], mfc=COL[J])
        else:
            ax.plot(xx, 150, "^", ms=6, color=COL[J], mfc="white")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], ls="", marker=m, color=COL[j], mfc=COL[j] if m != "^" else "white", label=l) for m, j, l in
                   (("o", "J15", "0.5B policy, 1.5B judge"), ("s", "J15", "1.5B policy, 1.5B judge"), ("s", "J7", "1.5B policy, 7B judge"),
                    ("o", "J7", "0.5B policy, 7B judge"), ("^", "J7", "censored (> 150 steps)"))], frameon=False, fontsize=7.5, loc="upper left")
ax.set_yscale("log")
ax.set_ylim(5, 200)
ax.axhline(150, ls=":", lw=0.6, color="gray")
ax.set_xlabel("ln(judge parameters / policy parameters)")
ax.set_ylabel("onset t$_{1/2}$ (RL steps, log scale)")
fig.tight_layout(rect=(0, 0.2, 1, 1))
caption(fig, r"$\mathbf{Figure\ 2:}$ Hack onset against the judge/policy size ratio. Red: 1.5B judge; blue: 7B judge. "
             "Circles: 0.5B policy; squares: 1.5B policy. Open triangles: no onset within 150 steps (censored). "
             "If only the gap mattered, points would fall on one rising curve.")
for e in ("png", "pdf"):
    fig.savefig(f"{ROOT}/figures/fig2_onset_vs_gap.{e}", dpi=200)
print("figures done")
