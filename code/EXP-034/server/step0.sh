#!/bin/bash
# EXP-034 step 0 on gpu-server-2: SFT of the 1.5B policy (GPU $1), step-0 samples of both policies, static gate of both judges.
source "$(dirname "$0")/env.sh"; g=${1:-0}; set -x
CUDA_VISIBLE_DEVICES=$g EXP013_BASE_MODEL=/data/exp034/models/Qwen2.5-1.5B python code/sft34.py 20261009-gpu-server-2-e34-sft15-s2-a1 2 $P15 > logs/sft15.log 2>&1 || exit 1
cp /data/exp034/models/Qwen2.5-1.5B/{merges.txt,vocab.json} $P15/ 2>/dev/null
CUDA_VISIBLE_DEVICES=$g python code/gen_step0.py $P05 data/step0/P05 > logs/step0_P05.log 2>&1 || exit 1
CUDA_VISIBLE_DEVICES=$g python code/gen_step0.py $P15 data/step0/P15 > logs/step0_P15.log 2>&1 || exit 1
CUDA_VISIBLE_DEVICES=$g python code/static34.py J15 $J15 > logs/static_J15.log 2>&1 || exit 1
until grep -q DL_DONE logs/dl.log; do sleep 30; done
CUDA_VISIBLE_DEVICES=$g python code/static34.py J7 $J7 > logs/static_J7.log 2>&1 || exit 1
echo STEP0_DONE
