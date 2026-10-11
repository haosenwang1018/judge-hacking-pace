#!/bin/bash
# EXP-034 single driver on gpu-server-2 (the only wait task). Queue file runs/queue.txt, one line per run: "P J seed".
# First fit over the queue. A job starts only when, on every GPU it uses, nvidia-smi used memory (all processes, ours and
# other agents') + its need <= 38 GiB. Needs (GiB) come from the step-0 smoke runs (PROCESS.md):
#   P05 runs: one GPU (policy, reference copy and judge together).
#   P15 runs: policy on one GPU (cuda:0 inside the job), judge + frozen reference copy on the other GPU (cuda:1).
# After each start: wait 150 s so the new job allocates before the next placement. Never touches processes it did not start.
source "$(dirname "$0")/env.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # allocator only; no effect on the math
echo $$ > runs/driver.pid
D=20261009-gpu-server-2-e34
LOG=runs/driver.log
need() { case "$1$2" in   # prints "policy_gpu_need judge_gpu_need" (judge need 0 = same GPU)
  P05J15) echo "$NEED_P05J15 0";; P05J7) echo "$NEED_P05J7 0";;
  P15J15) echo "$NEED_P15_POL $NEED_P15J15_J";; P15J7) echo "$NEED_P15_POL $NEED_P15J7_J";; esac; }
source code/server/needs.sh   # re-read each placement (needs measured in smoke runs)
used() { nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " "; }
ok() { python3 -c "import sys; sys.exit(0 if $(used $1) + $2*1024 <= 38*1024 else 1)"; }
job() { local gp=$1 gj=$2 P=$3 J=$4 s=$5 cap=$6; local id=$D-$P$J-s$s; mkdir -p runs/$id
  local pol=$P05; [ $P = P15 ] && pol=$P15; local jp=$J15; [ $J = J7 ] && jp=$J7
  local vis=$gp extra=""; [ "$gj" != "-" ] && { vis=$gp,$gj; extra="--judge-device cuda:1"; }
  echo "$(date +%FT%T%z) start gpus=$vis $id cap=$cap" >> $LOG
  CUDA_VISIBLE_DEVICES=$vis python code/grpo34.py --steps 150 --eval-every 10 --lr 3e-6 --scale group --readout G_cal \
    --micro-bs 2 --sft-path $pol --judge-path $jp --readouts-json data/readouts_${J}_${P}.json --seed $s --run-id $id --mem-gib $cap $extra \
    > runs/$id/stdout.log 2>&1
  local rc=$?   # capture before any command substitution (EXP-029 driver logged the rc of date)
  echo "$(date +%FT%T%z) end gpus=$vis $id rc=$rc" >> $LOG; }
echo "$(date +%FT%T%z) driver start" >> $LOG
# keeps polling until the queue is empty AND runs/queue.closed exists (lines may be appended while it runs)
while [ -s runs/queue.txt ] || [ ! -e runs/queue.closed ]; do
  started=0; n=0
  source code/server/needs.sh
  while read P J s; do n=$((n+1)); [ -z "$P" ] && continue
    set -- $(need $P $J); np=$1; nj=$2
    for gp in 0 1; do gj=$((1-gp))
      if [ "$nj" = 0 ]; then ok $gp $np && { sed -i ${n}d runs/queue.txt; job $gp - $P $J $s $(python3 -c "print($np-0.5)") & started=1; break; }
      else ok $gp $np && ok $gj $nj && { sed -i ${n}d runs/queue.txt; job $gp $gj $P $J $s $(python3 -c "print($np-0.5)") & started=1; break; }; fi
    done
    [ $started = 1 ] && break
  done < <(cat runs/queue.txt)
  if [ $started = 1 ]; then sleep 150; else sleep 30; fi
done
echo "$(date +%FT%T%z) queue empty; waiting" >> $LOG
wait
echo "$(date +%FT%T%z) driver done" >> $LOG
