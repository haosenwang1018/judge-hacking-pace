"""PAPER-T11 Figure 3 (revision 261011): judge-RL does not raise held-out accuracy, while the judge accepts more answers.
Input (read-only): ../rev261011/gold261011.json (built by rev261011/gold261011.py from the saved greedy eval outputs of all
38 runs, 200 held-out GSM8K test problems, lenient gold label). Output: fig3_gold.pdf/png."""
import json, os
import numpy as np
from style261011 import plt, COL, LAB, ORDER
HERE = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(f"{HERE}/../rev261011/gold261011.json"))["cells"]
fig, axs = plt.subplots(1, 3, figsize=(5.5, 1.62), gridspec_kw={"wspace": 0.38})
for ax, P, key in [(axs[0], "P05", "acc_len_mean"), (axs[1], "P15", "acc_len_mean"), (axs[2], "P05", "judge_pass_mean")]:
    for J in ORDER:
        c = P + J
        if c not in G: continue
        x = np.array(G[c]["eval_steps"]); y = np.array(G[c][key]); m = x <= 200
        ax.plot(x[m], y[m], color=COL[J], lw=1.0 if J in ("J7", "JL8", "JO7") else 0.9, label=LAB[J])
    ax.set_xlim(0, 200); ax.set_xticks([0, 50, 100, 150, 200]); ax.set_xlabel("GRPO step")
    if key == "acc_len_mean":
        a0 = G[P + "J3"]["acc0"]; ax.axhline(a0, color="black", lw=0.4, ls=(0, (3, 3)))
axs[0].set_ylim(0, 0.45); axs[0].set_ylabel("Held-out accuracy, 0.5B")
axs[1].set_ylim(0.3, 0.75); axs[1].set_ylabel("Held-out accuracy, 1.5B"); axs[1].set_xlim(0, 150); axs[1].set_xticks([0, 50, 100, 150])
axs[2].set_ylim(0.4, 1.02); axs[2].set_ylabel("Judge acceptance (0.5B)")
axs[2].legend(loc="lower right", fontsize=5.6, ncol=2, handlelength=1.0, columnspacing=0.6, handletextpad=0.3, borderaxespad=0.0, labelspacing=0.15)
for a_, t in zip(axs, ["(a)", "(b)", "(c)"]):
    a_.text(-0.34, 1.02, t, transform=a_.transAxes, fontsize=9)
fig.savefig(f"{HERE}/fig3_gold.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{HERE}/fig3_gold.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("ok")
