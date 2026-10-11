#!/bin/bash
# EXP-034: J7 static gate, then P05xJ7 smoke (memory/speed), on GPU $1. Readouts for the smoke = J7 readouts of P05.
source /data/exp034/code/server/env.sh; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; g=$1
echo $$ > logs/j7.pid
CUDA_VISIBLE_DEVICES=$g python code/static34.py J7 $J7 > logs/static_J7.log 2>&1 || { echo STATIC_FAIL; exit 1; }
CUDA_VISIBLE_DEVICES=$g timeout 1800 python code/grpo34.py --steps 8 --eval-every 10 --lr 3e-6 --scale group --sft-path $P05 \
  --judge-path $J7 --readouts-json data/readouts_J7_P05.json --readout G_cal --seed 1 --run-id smoke-P05J7 --micro-bs 8 > logs/smoke-P05J7.log 2>&1
echo "J7_DONE rc=$?"
