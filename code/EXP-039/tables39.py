"""EXP-039: per-cell tables for results/SUMMARY.md from results/analysis39.json, llama_rescore.json, static_lenient.json.
usage: python tables39.py [analysis json]"""
import json, math, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = json.load(open(sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results/analysis39.json"))
LL = json.load(open(f"{ROOT}/results/llama_rescore.json"))
SL = json.load(open(f"{ROOT}/results/static_lenient.json"))
CELLS = ["P05J15", "P15J15", "P05J3", "P15J3", "P05J7", "P15J7"]
f = lambda r, k: (r[k] if r[k] is not None else f">{r['T']}")
print("| cell | ratio | t_half lenient | generous | strict | h0.3 | h0.7 | common tau | KL sampled at t_half |")
SZ = {"P05": 0.49, "P15": 1.54, "J15": 1.54, "J3": 3.09, "J7": 7.61}
for c in CELLS:
    rs = sorted([r for r in A["runs"] if r["cell"] == c], key=lambda r: r["seed"])
    print(f"| {c} | {SZ[c[3:]] / SZ[c[:3]]:.1f} | " + " | ".join("/".join(str(f(r, k)) for r in rs) for k in
          ("t_half", "t_half_generous", "t_half_strict", "t_half_h03", "t_half_h07", "t_half_common_tau")) +
          " | " + "/".join(f"{r['kl_sampled_at_t_half']:.1f}" if r["kl_sampled_at_t_half"] else "-" for r in rs) + " |")
print("\nDelta_P (P15 - P05, ln steps), 97.5% / 95% profile intervals:")
for k in ("Delta_P", "Delta_P_generous", "Delta_P_mle_no_prior"):
    for J, v in A[k].items():
        if v:
            print(k, J, round(v["estimate"], 2), [round(x, 2) for x in v["ci97.5"]], [round(x, 2) for x in v["ci95.0"]], "p0", round(v["p_zero"], 4), "sigma", round(v["sigma_mle"], 3))
print("equal gap:", {k: (round(v["estimate"], 2), [round(x, 2) for x in v["ci97.5"]]) for k, v in A["equal_gap"].items() if v})
print("holm:", A["holm"], "verdict:", A["verdict"], "| generous:", A["verdict_generous"], "| final:", A["verdict_final"])
print("aft sigma", round(A["aft"]["sigma"], 3), {c: round(math.exp(m), 1) for c, m in A["aft"]["mu"].items()})
print("\nsecondary Delta_P (95%):")
for k, v in A["secondary_Delta_P"].items():
    print(k, {J: (round(x["estimate"], 2), [round(y, 2) for y in x["ci95.0"]]) for J, x in v.items() if x})
print("\ncommon-wrong h by block (cell mean over runs; n wrong rollouts pooled):")
for c in CELLS:
    rs = [r for r in A["runs"] if r["cell"] == c]
    out = []
    for b in ("1-50", "51-100", "101-150"):
        v = [r["h_common_wrong"][b] for r in rs if b in r["h_common_wrong"] and r["h_common_wrong"][b][0] is not None]
        a = [r["h_all_blocks"][b] for r in rs if b in r["h_all_blocks"] and r["h_all_blocks"][b][0] is not None]
        out.append(f"{b}: cw {np.mean([x[0] for x in v]):.2f} (n={sum(x[1] for x in v)}) all {np.mean([x[0] for x in a]):.2f}")
    print(c, " | ".join(out))
print("\nreward saturation and ties:")
for c in CELLS:
    rs = [r for r in A["runs"] if r["cell"] == c]
    print(c, "G<0.05 %.3f  G>0.95 %.3f  tie@tau %.3f  acc0 %s  acc_final %s  fmt_final %s" % (
        np.mean([r["frac_reward_lt_0.05"] for r in rs]), np.mean([r["frac_reward_gt_0.95"] for r in rs]),
        np.mean([r["tie_at_tau_share_of_wrong"] for r in rs]), [r["acc0"] for r in rs], [r["acc_final"] for r in rs], [r["format_final"] for r in rs]))
print("\nLlama-3.2-3B rescoring (mean over runs; H window):", {p: {k: round(v, 3) for k, v in s.items() if v is not None} for p, s in LL["step0"].items()})
for c in CELLS:
    rs = [v for k, v in LL["runs"].items() if f"-{c}-" in k and "H" in v]
    if rs:
        print(c, "n_runs", len(rs), "h_L E %.2f H %.2f | h_Q E %.2f H %.2f | both fooled H %.2f" % (
            np.mean([r["E"]["h_L"] for r in rs]), np.mean([r["H"]["h_L"] for r in rs]), np.mean([r["E"]["h_Q"] for r in rs]),
            np.mean([r["H"]["h_Q"] for r in rs]), np.mean([r["H"]["both_fooled"] for r in rs])),
            "| per run h_L(H):", [round(r["H"]["h_L"], 2) for r in rs])
