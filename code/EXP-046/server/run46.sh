#!/bin/bash
# EXP-046 one GRPO run (policy P05 only, 1 GPU). usage: run46.sh <JL1|JL8|JO1|JO7> <seed> <steps> [run_id]. Started by expq.
# Same command as EXP-041 run41.sh (P05 branch); only the judge and the per-judge memory cap / placeholder differ.
# All output (stdout too) goes to the blind run dir; nothing is printed to the ts log.
source /data/exp046/code/server/env.sh
J=$1; s=$2; T=$3; id=${4:-exp046-$HOSTKEY-P05$J-s$s}
JP=${!J}; [ -f $JP/.READY ] || { echo "judge $J not ready" >&2; exit 9; }
case $J in JL1|JO1) extra="--mem-gib 20.5 --reserve-gib 11.5";;      # EXP-039 P05xJ15 values (1.5B judge)
           JL8|JO7) extra="--mem-gib 34.5 --reserve-gib 12.5";;      # EXP-039 P05xJ7 values (7.6B judge) + 1 GiB for 8.0B
           *) echo "bad judge $J" >&2; exit 9;; esac
R=$EXP046_BLIND/runs/$id; mkdir -p $R
echo "start $(date +%FT%T) host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $id" >> $R/stdout.log
python code/grpo46.py --steps $T --eval-every 10 --lr 3e-6 --scale group --readout G_cal --micro-bs 2 \
  --sft-path $P05 --judge-path $JP --readouts-json data/readouts_${J}_P05.json --seed $s --run-id $id $extra >> $R/stdout.log 2>&1
rc=$?; echo "end $(date +%FT%T) rc=$rc" >> $R/stdout.log; exit $rc
