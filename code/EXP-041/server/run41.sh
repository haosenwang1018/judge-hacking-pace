#!/bin/bash
# EXP-041 one GRPO run. usage: run41.sh <P05|P15> <seed> <steps> [run_id]. Started by expq (ts sets CUDA_VISIBLE_DEVICES:
# 1 GPU for P05, 2 GPUs for P15 = policy on cuda:0, judge + reference copy on cuda:1, as EXP-039).
# Memory caps / placeholders as the EXP-039 driver (P05xJ3 values; JL has the same size class as J3).
source /data/exp041/code/server/env.sh
P=$1; s=$2; T=$3; id=${4:-20261010-gpu-server-2-e41-${P}JL-s$s}
mkdir -p runs/$id
if [ $P = P05 ]; then pol=$P05; extra="--mem-gib 25.5 --reserve-gib 13.5"
else pol=$P15; extra="--mem-gib 32.5 --reserve-gib 24.5 --judge-device cuda:1 --reserve-judge-gib 7"; fi
echo "start $(date +%FT%T) host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $id" >> runs/$id/stdout.log
python code/grpo41.py --steps $T --eval-every 10 --lr 3e-6 --scale group --readout G_cal --micro-bs 2 \
  --sft-path $pol --judge-path $JL --readouts-json data/readouts_JL_$P.json --seed $s --run-id $id $extra >> runs/$id/stdout.log 2>&1
rc=$?; echo "end $(date +%FT%T) rc=$rc" >> runs/$id/stdout.log; exit $rc
