"""EXP-041: copy of EXP-039 code (only change: see EXP-041 SPEC section 5). EXP-039: compact per-step counts from a run's rollouts.jsonl (EXP-034 runs and EXP-039 runs), for every onset variant
in SPEC 6. Gold: lenient (common.parse_lenient, recomputed from the completion; primary) and strict (EXP-034 field).
Per step: n; n_wl / f_wl = lenient-wrong rollouts / of those G_cal >= tau_len(J,P) (primary); f_wl_c = same at the
per-judge common tau; tie_wl = G_cal exactly tau_len; n_ws / f_ws = strict-wrong / G_cal >= tau_strict (EXP-034 primary);
n_cw / f_cw = lenient-wrong on common-wrong questions / fooled at tau_len; sat0 / sat1 = reward < 0.05 / > 0.95; len = mean tokens.
usage: python extract39.py <run_dir> [<run_dir> ...]  -> results/compact/<run_id>.json"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import parse_lenient, correct_generous
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SL = json.load(open(f"{ROOT}/results/static_lenient.json"))
CW = set(json.load(open(f"{ROOT}/data/common_wrong_questions.json")))
EPS = 1e-6
os.makedirs(f"{ROOT}/results/compact", exist_ok=True)
for d in sys.argv[1:]:
    rid = os.path.basename(d.rstrip("/"))
    P, J = re.search(r"-(P\d\d)(J[0-9A-Z]+)-s\d+", rid).groups()
    c = SL["cells"][f"{P}x{J}"]
    t_len, t_str, t_com = c["tau_len"], c["tau_strict"], SL["common"][J]["tau_common_len"]
    st = {}
    for l in open(f"{d}/rollouts.jsonl"):
        r = json.loads(l)
        s = st.setdefault(r["step"], dict(n=0, f_wl_x=0, n_wg=0, f_wg=0, n_wl=0, f_wl=0, f_wl_c=0, tie_wl=0, n_ws=0, f_ws=0, n_cw=0, f_cw=0, sat0=0, sat1=0, tok=0))
        g = r["G_cal"] + EPS   # SPEC supplement 7: '>= tau' with float near-ties (|diff| < 1e-6) counted as ties, as at step 0
        g0 = r["G_cal"]
        ok = parse_lenient(r["completion"]) == r["gold"]
        s["n"] += 1; s["tok"] += r["n_tokens"]; s["sat0"] += g0 < 0.05; s["sat1"] += g0 > 0.95
        if not ok:
            s["n_wl"] += 1; s["f_wl"] += g >= t_len; s["f_wl_c"] += g >= t_com; s["tie_wl"] += abs(g0 - t_len) < EPS
            s["f_wl_x"] += g0 >= t_len   # exact comparison (before supplement 7)
            if r["question"] in CW:
                s["n_cw"] += 1; s["f_cw"] += g >= t_len
        if not correct_generous(r["completion"], r["gold"]):   # SPEC supplement 2: generous gold, own tau_len
            s["n_wg"] += 1; s["f_wg"] += g >= t_len
        if not r["correct"]:
            s["n_ws"] += 1; s["f_ws"] += g >= t_str
    rj = json.load(open(f"{d}/run.json"))
    cfg = json.load(open(f"{d}/config.json"))
    ev = [json.loads(l) for l in open(f"{d}/evals.jsonl")] if os.path.exists(f"{d}/evals.jsonl") else []
    keep = ("step", "acc", "format_rate", "mean_len", "FPR_G_cal", "kl_greedy_per_token", "kl_greedy_per_seq",
            "kl_sampled_per_seq", "kl_sampled_per_seq_se", "kl_sampled_per_token", "param_dist")
    out = {"run_id": rid, "cell": P + J, "seed": rj["seed"], "status": rj.get("status"), "steps_cfg": cfg.get("max_steps"),
           "total_seconds": rj.get("total_seconds"), "tau_len": t_len, "tau_strict": t_str, "tau_common": t_com,
           "steps": {int(k): v for k, v in sorted(st.items())}, "evals": [{k: e.get(k) for k in keep} for e in ev]}
    json.dump(out, open(f"{ROOT}/results/compact/{rid}.json", "w"))
    print(rid, len(st), "steps", flush=True)
