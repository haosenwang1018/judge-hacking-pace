"""PAPER-T11 Figure 4 (appendix, revision 261011): lenient accuracy of the training rollouts (10-step windows, seed mean),
with the cell's onset t_0.3 (AFT geometric mean) marked. Inputs: ../rev261011/gold261011.json, ../../EXP-046/results/analysis46.json."""
import json, os
import numpy as np
from style261011 import plt, COL, LAB, ORDER
HERE = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(f"{HERE}/../rev261011/gold261011.json"))["cells"]
A46 = json.load(open(f"{HERE}/../../EXP-046/results/analysis46.json"))
fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.8), gridspec_kw={"wspace": 0.28})
for ax, P in ((axs[0], "P05"), (axs[1], "P15")):
    for J in ORDER:
        c = P + J
        if c not in G: continue
        y = np.array(G[c]["rollout_acc_10win_mean"]); x = np.arange(len(y)) * 10 + 5; m = x <= 200
        ax.plot(x[m], y[m], color=COL[J], lw=0.9, label=LAB[J])
        t = A46["onset"]["t03"]["geo_mean"][c]; yi = np.interp(t, x, y)
        ax.plot([t], [yi], marker="|", ms=7, mew=1.2, color=COL[J])
        ax.axhline(G[c]["acc0"], color="black", lw=0.4, ls=(0, (3, 3)))
    ax.set_xlim(0, 200); ax.set_xlabel("GRPO step")
axs[0].set_ylabel("Rollout accuracy (0.5B)"); axs[1].set_ylabel("Rollout accuracy (1.5B)")
axs[0].set_ylim(0, 0.62); axs[1].set_ylim(0.35, 0.88); axs[1].set_xlim(0, 150)
axs[0].legend(loc="upper center", fontsize=5.6, ncol=4, handlelength=1.0, columnspacing=0.6, borderaxespad=0.0, labelspacing=0.15)
for a_, t in zip(axs, ["(a)", "(b)"]):
    a_.text(-0.18, 1.04, t, transform=a_.transAxes, fontsize=9)
fig.savefig(f"{HERE}/fig4_rollout.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{HERE}/fig4_rollout.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("ok")
