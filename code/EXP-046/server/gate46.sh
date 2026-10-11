#!/bin/bash
# EXP-046 per-judge chain, one GPU job (expq, 1 GPU): static gate -> smoke -> submit the 4 RL runs.
# usage: gate46.sh <JL1|JL8|JO1|JO7>
# exit 3 = gate failed (judge reported, not run; its labels are removed from markers/WAKE_WHEN). exit 4 = smoke failed.
# Prints only step-0 numbers and smoke health (rc, line counts, memory). No RL outcome is printed (BLINDING).
source /data/exp046/code/server/env.sh
J=$1; JP=${!J}; set -o pipefail
[ -f $JP/.READY ] || { echo "judge $J not ready"; exit 9; }
mkdir -p logs results
if [ $HOSTKEY = gpu-server-1 ] && [ ! -f results/envcheck_gpu-server-1.json ]; then
  flock results/.envlock -c "[ -f results/envcheck_gpu-server-1.json ] || python code/envcheck46.py $J15 gpu-server-1 > logs/envcheck_gpu-server-1.log 2>&1" || exit 5
  python -c "import json,sys; sys.exit(0 if json.load(open('results/envcheck_gpu-server-1.json'))['pass'] else 1)" || { echo "envcheck failed"; exit 5; }
fi
python code/static46.py $J $JP > logs/static_$J.log 2>&1 || exit 11
flock results/.lock python code/lenient_static46.py > logs/lenient_$J.log 2>&1 || exit 12
python - $J <<'PY' || exit 13
import json, sys
J = sys.argv[1]; s = json.load(open(f"results/static_{J}.json"))
a5, a15 = s["cells"]["P05"]["auc_d8_G"], s["cells"]["P15"]["auc_d8_G"]
g = {"judge": J, "rule": "AUC(G) on d8(P05) >= 0.60 (SPEC 3)", "auc_d8_P05": a5, "auc_d8_P15": a15, "pass": bool(a5 >= 0.60)}
json.dump(g, open(f"results/gate_{J}.json", "w"), indent=1); print(json.dumps(g))
PY
if ! python -c "import json,sys; sys.exit(0 if json.load(open('results/gate_$J.json'))['pass'] else 1)"; then
  flock markers/.wlock sed -i "/^exp046-$J-/d" markers/WAKE_WHEN; echo "GATE FAIL $J: not run (SPEC 3)"; exit 3
fi
# smoke: seed 1, 4 steps, own id, never analysed (SPEC 5)
sid=smoke-$HOSTKEY-$J; rm -rf $EXP046_BLIND/runs/$sid $EXP046_BLIND/ckpts/$sid
bash code/server/run46.sh $J 1 4 $sid; rc=$?
python - $J $sid $rc <<'PY' || exit 4
import json, os, sys
J, sid, rc = sys.argv[1], sys.argv[2], int(sys.argv[3]); R = f"{os.environ['EXP046_BLIND']}/runs/{sid}"
n = lambda f: sum(1 for _ in open(f"{R}/{f}")) if os.path.exists(f"{R}/{f}") else 0
mem = [json.loads(l).get("max_mem_alloc_gib") for l in open(f"{R}/steps.jsonl")] if os.path.exists(f"{R}/steps.jsonl") else []
h = {"judge": J, "rc": rc, "steps_lines": n("steps.jsonl"), "rollout_lines": n("rollouts.jsonl"), "eval_lines": n("evals.jsonl"),
     "max_mem_alloc_gib": max([m for m in mem if m is not None], default=None)}
h["ok"] = bool(rc == 0 and h["steps_lines"] == 4 and h["rollout_lines"] == 256 and h["eval_lines"] >= 2)
json.dump(h, open(f"results/smoke_{J}.json", "w"), indent=1); print(json.dumps(h)); sys.exit(0 if h["ok"] else 1)
PY
rm -rf $EXP046_BLIND/ckpts/$sid
for s in 301 302 303 304; do
  expq /data/exp046 1 exp046-$J-P05-seed$s -- bash code/server/run46.sh $J $s 200
done
echo "submitted exp046-$J-P05-seed301..304"
