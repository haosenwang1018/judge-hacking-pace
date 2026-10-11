#!/bin/bash
# EXP-047 one judge-scoring job (SPEC 3.4). usage: probe.sh <J15|J3|J7|JL>. Started by expq with 1 GPU.
source /data/exp047/code/server/env.sh
J=$1; path=${!J}; bs=16; [ $J = J7 ] && bs=8
mkdir -p logs
echo "start $(date +%FT%T) host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES J=$J" >> logs/probe_$J.log
python code/probe47.py $J $path $STEP0 --bs $bs >> logs/probe_$J.log 2>&1
rc=$?; echo "end $(date +%FT%T) rc=$rc" >> logs/probe_$J.log; exit $rc
