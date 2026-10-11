#!/bin/bash
# EXP-039 offline Llama rescoring on GPU $1 (our own slot), args: t_half json
source /data/exp039/code/server/env.sh; export LLAMA
echo "llama $$" >> runs/pids.txt
CUDA_VISIBLE_DEVICES=$1 python code/llama39.py $2 /data/exp034/runs/20261009-gpu-server-2-e34-P*J*-s30? runs/20261009-gpu-server-2-e39-P*J*-s3?? > logs/llama.log 2>&1
echo "LLAMA_DONE rc=$?" >> logs/llama.log
