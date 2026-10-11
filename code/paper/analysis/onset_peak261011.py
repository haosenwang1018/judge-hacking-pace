"""PAPER-T11 verify fix 261011 (item 2.2 '32 of 38'): per-run onset t0.3 vs. training-rollout accuracy peak.
No GPU; saved compact logs only. Onsets come from EXP-046/code/analysis46.py (its run-building code is executed
unchanged up to the output step), so t0.3 is the same per-run value used by the 38-run AFT fit.
Rollout accuracy: lenient accuracy of training rollouts, 1 - n_wl/n, on 10-step windows (steps 1-10, 11-20, ...)
over each run's full length. Peak window: the window with the highest accuracy (first one if tied).
ONE stated definition: onset 'before the peak' means t0.3 < first step of the peak window.
Output: onset_peak261011.json (per-run table and count)."""
import json, os
H = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(f"{H}/../..")
A = f"{RES}/EXP-046/code/analysis46.py"
src = open(A).read()
ns = {"__file__": A, "__name__": "x"}
exec(src[: src.index('out = {"n_runs"')], ns)
R, runs = ns["R"], ns["runs"]
raw = {r["run_id"]: r for r in runs}
rows = []
for e in R:
    r = raw[e["run_id"]]; st = r["steps"]; T = r["steps_cfg"]; w = []
    for b in range(0, T, 10):
        n = sum(st[str(t)]["n"] for t in range(b + 1, b + 11) if str(t) in st)
        wl = sum(st[str(t)]["n_wl"] for t in range(b + 1, b + 11) if str(t) in st)
        w.append(1 - wl / n if n else -1.0)
    k = max(range(len(w)), key=lambda i: w[i])
    ws, we = k * 10 + 1, k * 10 + 10
    t03 = e["t03"]
    rows.append({"run_id": e["run_id"], "cell": e["cell"], "T": T, "t03": t03, "peak_window": [ws, we],
                 "peak_rollout_acc": w[k], "first_window_acc": w[0],
                 "t03_before_peak_window_start": (t03 is not None and t03 < ws),
                 "t03_le_peak_window_end": (t03 is not None and t03 <= we)})
n = len(rows)
cnt = sum(x["t03_before_peak_window_start"] for x in rows)
cnt_end = sum(x["t03_le_peak_window_end"] for x in rows)
out = {"definition": "t0.3 < first step of the 10-step window with the highest lenient training-rollout accuracy (per run)",
       "n_runs": n, "count_before_peak_window_start": cnt,
       "secondary_count_t03_le_peak_window_end": cnt_end, "runs": rows}
json.dump(out, open(f"{H}/onset_peak261011.json", "w"), indent=1)
print(f"{cnt}/{n} before peak-window start; {cnt_end}/{n} at or before peak-window end")
for x in rows:
    print(x["run_id"], x["cell"], x["t03"], x["peak_window"], round(x["peak_rollout_acc"], 3), x["t03_before_peak_window_start"])
