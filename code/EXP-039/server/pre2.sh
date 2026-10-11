#!/bin/bash
# EXP-039: P05xJ3 smoke (6 steps) on the first idle GPU (the first try at 10:50 OOMed: another agent's vLLM took GPU0).
source /data/exp039/code/server/env.sh; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "pre2 $$" >> runs/pids.txt
used() { nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " "; }
log() { echo "$(date +%FT%T%z) $*" >> logs/pre.log; }
g=""; while [ -z "$g" ]; do for i in 0 1; do [ "$(used $i)" -lt 1000 ] && { g=$i; break; }; done; [ -z "$g" ] && sleep 30; done
log "idle gpu $g: smoke P05J3 (retry)"; t0=$(date +%s)
CUDA_VISIBLE_DEVICES=$g timeout 1800 python code/grpo39.py --steps 6 --eval-every 10 --lr 3e-6 --scale group --sft-path $P05 --judge-path $J3 \
  --readouts-json data/readouts_J3_P05.json --readout G_cal --seed 1 --run-id smoke-P05J3b --micro-bs 2 > logs/smoke-P05J3b.log 2>&1 & p=$!
echo "smoke-P05J3b $p" >> runs/pids.txt; wait $p; log "smoke P05J3b rc=$? $(( $(date +%s) - t0 ))s"
log PRE2_DONE
