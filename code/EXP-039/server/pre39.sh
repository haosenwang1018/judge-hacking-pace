#!/bin/bash
# EXP-039 step 0 on gpu-server-2: J3 static gate (G0), lenient step-0 table, P05xJ3 smoke (1 idle GPU), P15xJ3 smoke (2 idle GPUs).
source /data/exp039/code/server/env.sh; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "pre $$" >> runs/pids.txt
used() { nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " "; }
log() { echo "$(date +%FT%T%z) $*" >> logs/pre.log; }
g=""; while [ -z "$g" ]; do for i in 0 1; do [ "$(used $i)" -lt 1000 ] && { g=$i; break; }; done; [ -z "$g" ] && sleep 60; done
log "idle gpu $g: static J3"
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=$g python code/static39.py J3 $J3 > logs/static_J3.log 2>&1 & p=$!; echo "static $p" >> runs/pids.txt; wait $p || { log STATIC_FAIL; exit 1; }
python code/lenient_static39.py > logs/lenient_static.log 2>&1
log "static done $(( $(date +%s) - t0 ))s gate=$(python -c "import json;print(json.load(open('results/static_J3.json'))['gate_pass'])")"
python -c "import json,sys;sys.exit(0 if json.load(open('results/static_J3.json'))['gate_pass'] else 1)" || { log "G0 FAIL: stop"; exit 0; }
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=$g timeout 1800 python code/grpo39.py --steps 6 --eval-every 10 --lr 3e-6 --scale group --sft-path $P05 --judge-path $J3 \
  --readouts-json data/readouts_J3_P05.json --readout G_cal --seed 1 --run-id smoke-P05J3 --micro-bs 2 > logs/smoke-P05J3.log 2>&1 & p=$!
echo "smoke-P05J3 $p" >> runs/pids.txt; wait $p; log "smoke P05J3 rc=$? $(( $(date +%s) - t0 ))s"
while [ "$(used 0)" -ge 1000 ] || [ "$(used 1)" -ge 1000 ]; do sleep 60; done
log "both idle: smoke P15J3"
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0,1 timeout 1800 python code/grpo39.py --steps 6 --eval-every 10 --lr 3e-6 --scale group --sft-path $P15 --judge-path $J3 \
  --readouts-json data/readouts_J3_P15.json --readout G_cal --seed 1 --run-id smoke-P15J3 --micro-bs 2 --judge-device cuda:1 > logs/smoke-P15J3.log 2>&1 & p=$!
echo "smoke-P15J3 $p" >> runs/pids.txt; wait $p; log "smoke P15J3 rc=$? $(( $(date +%s) - t0 ))s"
log PRE_DONE
