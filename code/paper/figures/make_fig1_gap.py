"""PAPER-T11 Figure 1 (revision 261011): the policy-judge gap does not set the pace; the judge does.
(a) onset at h >= 0.3 (per run; bar = AFT geometric mean over all 38 runs) against the step-0 capability gap
    logit(judge AUC) - logit(policy accuracy); the two pre-specified pairs are joined by dashed lines.
Second output fig5_curves (appendix): fooled fraction h for the 0.5B policy under the four stage 1-3 judges (seed mean and range).
Inputs (read-only): ../rev261011/gap261011.json, ../../EXP-046/results/analysis46.json, ../../EXP-041/results/static_lenient.json,
                    ../../EXP-039|EXP-041/results/compact/*.json. Output: fig1_gap.pdf/png."""
import glob, json, os, math
import numpy as np
from style261011 import plt, COL, LAB
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(f"{HERE}/../..")
GP = json.load(open(f"{HERE}/../rev261011/gap261011.json"))
A46 = json.load(open(f"{R}/EXP-046/results/analysis46.json"))
SL = json.load(open(f"{R}/EXP-041/results/static_lenient.json"))["cells"]
cap = GP["capability"]
runs = [json.load(open(p)) for p in sorted(glob.glob(f"{R}/EXP-039/results/compact/*.json")) + sorted(glob.glob(f"{R}/EXP-041/results/compact/*.json"))]
fig, ax = plt.subplots(1, 1, figsize=(2.55, 2.15))
OFF = {"P05JL8": (5, 1), "P05J3": (5, -8), "P05JO7": (-8, -17), "P05JO1": (6, -11), "P05J15": (-24, -14), "P05JL": (-16, 6), "P05J7": (-20, 3), "P15J15": (5, -3), "P15J3": (5, -3), "P15J7": (5, -3), "P15JL": (5, -3)}
for c, d in cap.items():
    P, J = c[:3], c[3:]; x = d["gap_logit"]; col = COL[J]
    ts = A46["onset"]["t03"]["per_run"][c]; gm = A46["onset"]["t03"]["geo_mean"][c]
    held = J in ("JL8", "JO1", "JO7")
    ax.scatter([x] * len(ts), ts, s=7, marker="o" if P == "P05" else "s", facecolor="none", edgecolor=col, lw=0.6, alpha=0.8, zorder=2)
    ax.scatter(x, gm, s=24, marker="o" if P == "P05" else "s", facecolor=col if not held else "white", edgecolor=col, lw=1.0, zorder=3)
    ax.annotate(LAB[J], (x, gm), xytext=OFF.get(c, (4, -2)), textcoords="offset points", fontsize=8, color=col)
for a, b in (("P05J3", "P15J7"), ("P15J3", "P05J15")):
    ax.plot([cap[a]["gap_logit"], cap[b]["gap_logit"]], [A46["onset"]["t03"]["geo_mean"][a], A46["onset"]["t03"]["geo_mean"][b]],
            color="black", lw=0.6, ls=(0, (2, 2)), zorder=1)
ax.annotate("pair 1", (1.95, 31), fontsize=8, style="italic"); ax.annotate("pair 2", (1.02, 14.2), fontsize=8, style="italic")
ax.set_yscale("log"); ax.set_ylim(9, 140); ax.set_yticks([10, 20, 40, 80]); ax.set_yticklabels(["10", "20", "40", "80"]); ax.minorticks_off()
ax.set_xlim(0, 3.0)
ax.set_xlabel("Step-0 capability gap (logit)"); ax.set_ylabel(r"Onset $t_{0.3}$ (step)")
ax.scatter([], [], s=20, marker="o", facecolor="gray", edgecolor="gray", label="0.5B policy")
ax.scatter([], [], s=20, marker="s", facecolor="gray", edgecolor="gray", label="1.5B policy")
ax.scatter([], [], s=20, marker="o", facecolor="white", edgecolor="gray", label="held-out judge")
ax.legend(loc="upper left", fontsize=8, handletextpad=0.2, borderaxespad=0.0, labelspacing=0.15)
fig.savefig(f"{HERE}/fig1_gap.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{HERE}/fig1_gap.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
# appendix figure: h curves (former Figure 1c)
def hcurve(r, W=5):
    f0 = SL[f"{r['cell'][:3]}x{r['cell'][3:]}"]["FPR0_len"]; st = r["steps"]; xs, ys = [], []
    for b in range(W, r["steps_cfg"] + 1, W):
        n = sum(st[str(t)]["n_wl"] for t in range(b - W + 1, b + 1)); f = sum(st[str(t)]["f_wl"] for t in range(b - W + 1, b + 1))
        xs.append(b); ys.append((f / n - f0) / (1 - f0) if n >= 20 else np.nan)
    return np.array(xs), np.array(ys)
fig, ax = plt.subplots(1, 1, figsize=(4.0, 1.9))
ax.axvspan(101, 150, color="#EEEEEE", lw=0, zorder=0)
for J in ("J15", "J3", "J7", "JL"):
    rs = [r for r in runs if r["cell"] == f"P05{J}"]; curves = [hcurve(r) for r in rs]
    L = min(200, min(len(x) * 5 for x, _ in curves)); Y = np.array([y[:L // 5] for _, y in curves]); X = curves[0][0][:L // 5]
    ax.fill_between(X, np.nanmin(Y, 0), np.nanmax(Y, 0), color=COL[J], alpha=0.15, lw=0)
    ax.plot(X, np.nanmean(Y, 0), color=COL[J], lw=1.0, label=f"{LAB[J]} ({len(rs)} seeds)")
ax.axhline(0, color="black", lw=0.4)
for th in (0.3, 0.5): ax.axhline(th, color="black", lw=0.4, ls=(0, (3, 3)))
ax.set_xlim(0, 200); ax.set_ylim(-0.2, 1.05)
ax.set_xlabel("GRPO step (0.5B policy)"); ax.set_ylabel(r"Fooled fraction $h$")
ax.legend(loc="lower right", fontsize=5.8, ncol=2, columnspacing=0.8, handlelength=1.2, borderaxespad=0.0)
fig.savefig(f"{HERE}/fig5_curves.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{HERE}/fig5_curves.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("ok")
