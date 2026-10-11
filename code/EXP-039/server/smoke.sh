# EXP-039 smoke: args gpus policy judge readouts tag judge_device(-|cuda:1) steps
source /data/exp039/code/server/env.sh; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
JD=""; [ "$6" != "-" ] && JD="--judge-device $6"
echo $$ >> runs/pids.txt
CUDA_VISIBLE_DEVICES=$1 timeout 1800 python code/grpo39.py --steps ${7:-6} --eval-every 10 --lr 3e-6 --scale group --sft-path $2 --judge-path $3 --readouts-json $4 --readout G_cal --seed 1 --run-id smoke-$5 --micro-bs 2 $JD > logs/smoke-$5.log 2>&1
echo rc=$? >> logs/smoke-$5.log
