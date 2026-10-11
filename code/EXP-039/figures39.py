"""EXP-039 figures. Formal paper look: serif, no in-chart title, caption under the chart with bold 'Figure N:'.
No internal cell codes in the figures. Input: results/analysis39<_tag>.json, results/compact/*.json.
usage: python figures39.py <analysis json> -> figures/fig1_onset_by_judge.{pdf,png}, fig2_onset_vs_ratio.{pdf,png}, fig3_h_curves.{pdf,png}"""
import glob, json, math, os, sys, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = json.load(open(sys.argv[1]))
os.makedirs(f"{ROOT}/figures", exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "mathtext.fontset": "stix"})
POL = {"P05": ("0.5B policy", "#4C72B0", "o"), "P15": ("1.5B policy", "#C44E52", "s")}
JUD = {"J15": (0, "1.5B judge"), "J3": (1, "3B judge"), "J7": (2, "7B judge")}
SIZE = {"P05": 0.49, "P15": 1.54, "J15": 1.54, "J3": 3.09, "J7": 7.61}


def caption(fig, n, text, y=0.0):
    w = textwrap.fill(text, 110)
    fig.text(0.02, y, f"$\\bf{{Figure\\ {n}:}}$ " + w, ha="left", va="top", fontsize=8.5, family="serif")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{ROOT}/figures/{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


# Figure 1: onset by judge, per policy
fig, ax = plt.subplots(figsize=(4.6, 3.0))
rng = np.random.default_rng(0)
for P, (lab, col, mk) in POL.items():
    xs_m, ys_m = [], []
    for J, (xi, _) in JUD.items():
        rs = [r for r in A["runs"] if r["cell"] == P + J]
        if not rs:
            continue
        off = -0.12 if P == "P05" else 0.12
        for r in rs:
            x = xi + off + rng.uniform(-0.04, 0.04)
            if r["censored"]:
                ax.scatter(x, r["T"], marker="^", facecolors="none", edgecolors=col, s=30, lw=0.9)
            else:
                ax.scatter(x, r["t_half"], marker=mk, color=col, s=22, lw=0)
        xs_m.append(xi + off); ys_m.append(float(np.median([r["t_half"] or r["T"] for r in rs])))
    ax.plot(xs_m, ys_m, color=col, lw=0.9, alpha=0.8, label=lab)
ax.set_yscale("log")
ax.set_yticks([10, 20, 50, 100, 200, 300]); ax.set_yticklabels(["10", "20", "50", "100", "200", "300"]); ax.set_ylim(10, 320)
ax.set_xticks(range(3)); ax.set_xticklabels([v[1] for v in JUD.values()])
ax.set_ylabel("Onset of judge fooling (training step)")
ax.legend(loc="upper left", fontsize=8)
caption(fig, 1, "Onset step t½ (lenient gold: half of the remaining room above the step-0 false-positive rate is used, "
        "in two consecutive 5-step windows) for each run. Filled markers: onset reached; open triangles: no onset by the "
        "end of the run (censored, plotted at the last step). Lines join the per-cell medians (censored runs counted at their last step, so a median there is a lower bound).", y=-0.02)
save(fig, "fig1_onset_by_judge")

# Figure 2: onset vs judge/policy size ratio
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for P, (lab, col, mk) in POL.items():
    for J in JUD:
        for r in [r for r in A["runs"] if r["cell"] == P + J]:
            x = SIZE[J] / SIZE[P] * math.exp(rng.uniform(-0.03, 0.03))
            if r["censored"]:
                ax.scatter(x, r["T"], marker="^", facecolors="none", edgecolors=col, s=30, lw=0.9)
            else:
                ax.scatter(x, r["t_half"], marker=mk, color=col, s=22, lw=0)
    ax.scatter([], [], marker=mk, color=col, label=lab)
ax.scatter([], [], marker="^", facecolors="none", edgecolors="gray", label="censored")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([1, 2, 3, 5, 10, 15]); ax.set_xticklabels(["1", "2", "3", "5", "10", "15"])
ax.set_yticks([10, 20, 50, 100, 200, 300]); ax.set_yticklabels(["10", "20", "50", "100", "200", "300"])
ax.set_xlabel("Judge size / policy size (parameters)"); ax.set_ylabel("Onset of judge fooling (training step)")
ax.legend(loc="upper left", fontsize=8)
caption(fig, 2, "The same onsets plotted against the judge-to-policy size ratio. If only the size gap mattered (with a "
        "floor of about 20 steps), both policies would fall on one rising curve.", y=-0.06)
save(fig, "fig2_onset_vs_ratio")

# Figure 3: training-window h curves, mean over seeds, per cell
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
SL = json.load(open(f"{ROOT}/results/static_lenient.json"))
for J, (xi, jl) in JUD.items():
    ax = axs[xi]
    for P, (lab, col, mk) in POL.items():
        curves = []
        for p in glob.glob(f"{ROOT}/results/compact/*-{P}{J}-s*.json"):
            c = json.load(open(p)); st = c["steps"]; f0 = SL["cells"][f"{P}x{J}"]["FPR0_len"]
            T = c["steps_cfg"]; hs = np.full(250, np.nan)
            for t in range(5, T + 1):
                nw = sum(st[str(u)]["n_wl"] for u in range(t - 4, t + 1) if str(u) in st)
                fo = sum(st[str(u)]["f_wl"] for u in range(t - 4, t + 1) if str(u) in st)
                if nw >= 20:
                    hs[t - 1] = (fo / nw - f0) / (1 - f0)
            curves.append(hs)
        if curves:
            m = np.nanmean(np.array(curves), 0)
            ax.plot(np.arange(1, 251), m, color=col, lw=1.0, label=lab)
    ax.axhline(0.5, color="gray", lw=0.5, ls="--")
    ax.set_xlabel(f"Training step ({jl})"); ax.set_xlim(0, 250)
axs[0].set_ylabel("Fooled share above step 0 (h)")
axs[0].legend(loc="lower right", fontsize=7.5)
caption(fig, 3, "Normalised fooled share h = (FPR − FPR0)/(1 − FPR0) on lenient-wrong training rollouts, 5-step windows, "
        "averaged over seeds, for the 1.5B, 3B and 7B judges (beyond step 150, only the longer runs contribute). Dashed line: the onset threshold 0.5.", y=-0.10)
save(fig, "fig3_h_curves")
print("ok")
