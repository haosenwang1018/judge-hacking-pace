"""PAPER-T11 main figure. House style: serif, no in-chart title, minimalist; the caption lives in main.tex.
Inputs (read-only): ../EXP-041/results/{analysis41.json,static_lenient.json,compact/*.json},
                    ../EXP-039/results/compact/*.json
Output: fig1_main.pdf / fig1_main.png in this folder.
Run: python make_figures.py   (needs matplotlib, numpy)"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", ".."))
E39, E41 = f"{R}/EXP-039", f"{R}/EXP-041"
A41 = json.load(open(f"{E41}/results/analysis41.json"))
SL = json.load(open(f"{E41}/results/static_lenient.json"))["cells"]
runs = [json.load(open(p)) for p in sorted(glob.glob(f"{E39}/results/compact/*.json"))
        + sorted(glob.glob(f"{E41}/results/compact/*.json"))]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
                     "mathtext.fontset": "stix"})
JUD = {"J15": ("Qwen2.5-1.5B", "#8C8C8C"), "J3": ("Qwen2.5-3B", "#4C72B0"),
       "J7": ("Qwen2.5-7B", "#55A868"), "JL": ("Llama-3.2-3B", "#C44E52")}
POL = {"P05": ("0.5B policy", "o", -0.13), "P15": ("1.5B policy", "s", 0.13)}


def hcurve(r, W=5):
    f0 = SL[f"{r['cell'][:3]}x{r['cell'][3:]}"]["FPR0_len"]
    st = r["steps"]
    xs, ys = [], []
    for b in range(W, r["steps_cfg"] + 1, W):
        n = sum(st[str(t)]["n_wl"] for t in range(b - W + 1, b + 1))
        f = sum(st[str(t)]["f_wl"] for t in range(b - W + 1, b + 1))
        xs.append(b); ys.append((f / n - f0) / (1 - f0) if n >= 20 else np.nan)
    return np.array(xs), np.array(ys)


fig = plt.figure(figsize=(5.5, 4.1))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.05], hspace=0.55, wspace=0.32)
axA, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

# (a) onset t_half per run, by judge; open triangles = censored at last step
per = A41["onset"]["t_half"]["per_run"]
rng = np.random.default_rng(0)
for xi, J in enumerate(JUD):
    col = JUD[J][1]
    for P, (plab, mk, off) in POL.items():
        for v in per[P + J]:
            x = xi + off + rng.uniform(-0.035, 0.035)
            if isinstance(v, str):
                axA.scatter(x, int(v[1:]), marker="^", facecolor="none", edgecolor=col, s=18, lw=0.8)
            else:
                axA.scatter(x, v, marker=mk, facecolor=col if P == "P05" else "white", edgecolor=col, s=14, lw=0.8)
axA.set_yscale("log"); axA.set_ylim(12, 320)
axA.set_yticks([20, 50, 100, 200]); axA.set_yticklabels(["20", "50", "100", "200"])
axA.minorticks_off()
axA.set_xticks(range(4)); axA.set_xticklabels(["Q-1.5B", "Q-3B", "Q-7B", "L-3B"])
axA.set_xlabel("Frozen judge"); axA.set_ylabel(r"Onset $t_{1/2}$ (step)")
axA.scatter([], [], marker="o", facecolor="black", edgecolor="black", s=14, label="0.5B policy")
axA.scatter([], [], marker="s", facecolor="white", edgecolor="black", s=14, label="1.5B policy")
axA.scatter([], [], marker="^", facecolor="none", edgecolor="black", s=18, label="censored")
axA.legend(loc="upper left", fontsize=6.5, handletextpad=0.2, borderaxespad=0.1)
axA.text(-0.2, 1.04, "(a)", transform=axA.transAxes, fontsize=9)

# (b) plateau height per seed vs step-0 AUC
for P, (plab, mk, _) in POL.items():
    for J, (lab, col) in JUD.items():
        auc = SL[f"{P}x{J}"]["auc_len"]
        hs = A41["plateau_cells"][P + J]["seeds"]
        jit = np.linspace(-0.003, 0.003, len(hs))
        axB.scatter(auc + jit, hs, s=12, marker=mk, facecolor=col if P == "P05" else "white", edgecolor=col,
                    lw=0.8, zorder=3)
        axB.plot([auc - 0.008, auc + 0.008], [np.mean(hs)] * 2, color=col, lw=1.1)
axB.set_xlabel("Judge AUC at step 0"); axB.set_ylabel(r"Plateau $\bar{H}$ (steps 101–150)")
axB.set_ylim(0.2, 1.03); axB.set_xlim(0.66, 0.90)
axB.text(-0.2, 1.04, "(b)", transform=axB.transAxes, fontsize=9)

# (c) h curves, 0.5B policy, four judges
axC.axvspan(101, 150, color="#EEEEEE", lw=0, zorder=0)
for J, (lab, col) in JUD.items():
    rs = [r for r in runs if r["cell"] == f"P05{J}"]
    curves = [hcurve(r) for r in rs]
    L = min(200, min(len(x) * 5 for x, _ in curves))
    Y = np.array([y[:L // 5] for _, y in curves]); X = curves[0][0][:L // 5]
    axC.fill_between(X, np.nanmin(Y, 0), np.nanmax(Y, 0), color=col, alpha=0.15, lw=0)
    axC.plot(X, np.nanmean(Y, 0), color=col, lw=1.4 if J == "JL" else 1.0, label=f"{lab} judge ({len(rs)} seeds)")
axC.axhline(0, color="black", lw=0.4)
axC.axhline(0.5, color="black", lw=0.4, ls=(0, (3, 3)))
axC.set_xlim(0, 200); axC.set_ylim(-0.2, 1.05)
axC.set_xlabel("GRPO step (0.5B policy)"); axC.set_ylabel(r"Fooled fraction $h$")
axC.legend(loc="lower right", fontsize=6.5, ncol=2, columnspacing=1.0)
axC.text(-0.09, 1.04, "(c)", transform=axC.transAxes, fontsize=9)

for ext in ("pdf", "png"):
    fig.savefig(f"{HERE}/fig1_main.{ext}", bbox_inches="tight", dpi=220)
print("ok")
