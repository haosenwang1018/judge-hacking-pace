"""EXP-034 post-hoc check (written after the stage-1 results): some hacked policies stop writing '#### n'.
Strict gold then calls a right-but-unformatted answer 'wrong', and a judge that says Yes to it is not fooled.
Recompute the training-window onset t_half with lenient gold (common.parse_lenient: strict answer, else last number).
usage (gpu-server-2): python code/posthoc_lenient.py -> results/posthoc_lenient.json"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import parse_lenient, parse_strict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = {}
for d in sorted(glob.glob(f"{ROOT}/runs/*-e34-P*J*-s*")):
    cell = d.split("-e34-")[1].split("-s")[0]; P, J = cell[:3], cell[3:]
    R = json.load(open(f"{ROOT}/data/readouts_{J}_{P}.json")); tau = R["tau_tpr80_d8"]["G_cal"]; f0 = R["FPR0_d8"]
    W, F, n_unfmt_right, n_wrong_strict = {}, {}, 0, 0
    for l in open(f"{d}/rollouts.jsonl"):
        r = json.loads(l); t = r["step"]
        strict_wrong = not r["correct"]
        len_right = parse_lenient(r["completion"]) == r["gold"]
        if strict_wrong:
            n_wrong_strict += 1; n_unfmt_right += len_right
        if not len_right:
            W[t] = W.get(t, 0) + 1; F[t] = F.get(t, 0) + (r["G_cal"] >= tau)
    def h(a, b):
        nw = sum(W.get(t, 0) for t in range(a, b + 1))
        return None if nw < 20 else (sum(F.get(t, 0) for t in range(a, b + 1)) / nw - f0) / (1 - f0)
    th = next((t for t in range(5, 146) if (x := h(t - 4, t)) is not None and x >= 0.5 and (y := h(t + 1, t + 5)) is not None and y >= 0.5), None)
    out[os.path.basename(d)] = {"t_half_lenient": th, "share_strict_wrong_that_lenient_right": n_unfmt_right / max(n_wrong_strict, 1)}
    print(os.path.basename(d), out[os.path.basename(d)], flush=True)
json.dump(out, open(f"{ROOT}/results/posthoc_lenient.json", "w"), indent=1)
