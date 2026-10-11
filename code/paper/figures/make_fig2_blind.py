"""PAPER-T11 Figure 2: pre-registered blind test of step-0 predictors (EXP-047 screening -> EXP-046 held-out judges).
House style as make_figures.py: serif, no in-chart title, minimalist; caption lives in main.tex.
Inputs (read-only): ../../EXP-046/results/{analysis46.json,predictors46.json}
Output: fig2_blind.pdf / fig2_blind.png in this folder.
Run: python make_fig2_blind.py   (needs matplotlib, numpy)"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", ".."))
A = json.load(open(f"{R}/EXP-046/results/analysis46.json"))
PR = json.load(open(f"{R}/EXP-046/results/predictors46.json"))["cells"]
LF = A["test"]["locked_fit_predictions"]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})
FAM = {"Qwen2.5": ("#0F4D92", ["J15", "J3", "J7"]), "Llama-3": ("#B64342", ["JL", "JL8"]),
       "OLMo-2": ("#42949E", ["JO1", "JO7"])}
LAB = {"J15": "Q1.5", "J3": "Q3", "J7": "Q7", "JL": "L3", "JL8": "L8", "JO1": "O1", "JO7": "O7"}
NEW = {"JL8", "JO1", "JO7"}
# label offsets (points) to avoid overlap
OFF_A = {"J15": (2, -11), "J3": (4, 3), "J7": (4, -2), "JL": (-12, -8), "JL8": (4, -6), "JO1": (4, 1), "JO7": (-12, 5)}
OFF_B = {"J15": (-20, 4), "J3": (4, 2), "J7": (4, -2), "JL": (4, -7), "JL8": (-11, -9), "JO1": (-4, 6), "JO7": (-14, 4)}


def cell(J):
    c = A["cells"]["P05" + J]["E1"]
    s = np.array(c["seeds"])
    return c["mean"], s.min(), s.max()


fig, axs = plt.subplots(1, 3, figsize=(5.5, 1.7), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1, 0.9]})
for ax, key, off, logx in [(axs[0], "B1_auc", OFF_A, False), (axs[1], "P1_calib_slope", OFF_B, True)]:
    for fam, (col, js) in FAM.items():
        xs = [PR["P05" + J][key]["value"] for J in js]
        ys = [cell(J)[0] for J in js]
        ax.plot(xs, ys, color=col, lw=0.7, alpha=0.6, zorder=1)
        for J, x, y in zip(js, xs, ys):
            _, lo, hi = cell(J)
            ax.plot([x, x], [lo, hi], color=col, lw=0.7, zorder=2)
            ax.scatter(x, y, s=16, marker="o", facecolor="white" if J in NEW else col, edgecolor=col, lw=0.9, zorder=3)
            ax.annotate(LAB[J], (x, y), xytext=off[J], textcoords="offset points", fontsize=6.5, color=col)
    ax.set_ylim(0.25, 1.05)
    if not logx: ax.set_ylabel(r"Plateau $\bar{H}$ (steps 101–150)")
    if logx:
        ax.set_xscale("log"); ax.set_xticks([0.1, 0.3, 1.0]); ax.set_xticklabels(["0.1", "0.3", "1"]); ax.minorticks_off()
        ax.set_xlim(0.08, 2.4)
axs[0].set_xlabel("Step-0 AUC")
axs[0].set_xlim(0.72, 0.90)
axs[1].set_xlabel("Step-0 calibration slope (locked CS)")
for fam, (col, _) in FAM.items():
    axs[0].plot([], [], color=col, lw=0.7, marker="o", ms=3.5, label=fam)
axs[0].scatter([], [], s=16, facecolor="white", edgecolor="black", lw=0.9, label="held out")
axs[0].legend(loc="lower left", fontsize=6, handlelength=1.2, handletextpad=0.3, borderaxespad=0.0, labelspacing=0.2)

# (c) locked-fit prediction vs observed plateau on the three held-out cells
ax = axs[2]
ax.plot([0.5, 1.3], [0.5, 1.3], color="#888888", lw=0.6, ls="--", zorder=1)
for key, mk, lab in [("P1_calib_slope", "^", "CS"), ("B1_auc", "o", "AUC")]:
    cs = LF[key]["O1"]["cells"]
    for J in ["JL8", "JO1", "JO7"]:
        col = FAM["Llama-3"][0] if J == "JL8" else FAM["OLMo-2"][0]
        ax.scatter(cs["P05" + J]["pred"], cs["P05" + J]["obs"], marker=mk, s=16, facecolor=col,
                   edgecolor=col, lw=0.9, zorder=3)
    ax.scatter([], [], marker=mk, s=16, facecolor="black", edgecolor="black", lw=0.9,
               label=f"{lab} (MAE {LF[key]['O1']['mae']:.3f})")
ax.set_xlim(0.55, 1.3); ax.set_ylim(0.55, 1.3)
ax.set_xticks([0.6, 0.8, 1.0, 1.2]); ax.set_yticks([0.6, 0.8, 1.0, 1.2])
ax.set_xlabel(r"Locked-fit prediction of $\bar{H}$"); ax.set_ylabel(r"Observed $\bar{H}$")
ax.legend(loc="upper left", fontsize=6, handletextpad=0.2, borderaxespad=0.0, labelspacing=0.2)
for a_, t in zip(axs, ["(a)", "(b)", "(c)"]):
    a_.text(-0.3, 1.04, t, transform=a_.transAxes, fontsize=9)
fig.savefig(f"{HERE}/fig2_blind.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{HERE}/fig2_blind.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("ok")
