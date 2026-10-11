# EXP-034 smoke: args gpus policy judge tag judge_device(-|cuda:1) micro_bs steps
source /data/exp034/code/server/env.sh; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
JD=""; [ "$5" != "-" ] && JD="--judge-device $5"
CUDA_VISIBLE_DEVICES=$1 timeout 1800 python code/grpo34.py --steps ${7:-6} --eval-every 10 --lr 3e-6 --scale group --sft-path $2 --judge-path $3 --readouts-json data/smoke_readouts.json --readout G_cal --seed 1 --run-id smoke-$4 --eval-n 200 $JD --micro-bs ${6:-8} > logs/smoke-$4.log 2>&1
echo rc=$? >> logs/smoke-$4.log
