#!/bin/bash
# EXP-039 driver v3 on gpu-server-2 (the only wait task). Queue runs/queue.txt, one line per run: "P J seed steps".
# Shared machine rules: a job starts only on IDLE GPUs (nvidia-smi memory.used < 1000 MiB). Every python PID we start is
# appended to runs/pids.txt; only those PIDs are ever killed. No pgrep / pkill. runs/stop: start nothing new.
# v3 (261009 13:3x): （内部笔记） When one of our jobs ends, the GPU(s) it used are taken over at once by our next queued job
# of the same width (1 GPU for P05, 2 GPUs for P15), so other agents' launchers cannot slip in between. If a 1-GPU job ends
# and only 2-GPU jobs remain, a HOLDER process keeps that GPU for at most 30 minutes while the second GPU frees up.
source "$(dirname "$0")/env.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "driver $$" >> runs/pids.txt
D=20261009-gpu-server-2-e39
LOG=runs/driver.log
need() { case "$1$2" in
  P05J15) echo "$NEED_P05J15 0";; P05J3) echo "$NEED_P05J3 0";; P05J7) echo "$NEED_P05J7 0";;
  P15J15) echo "$NEED_P15_POL $NEED_P15J15_J";; P15J3) echo "$NEED_P15_POL $NEED_P15J3_J";; P15J7) echo "$NEED_P15_POL $NEED_P15J7_J";; esac; }
load() { case "$1$2" in
  P05J15) echo "8 0";; P05J3) echo "11 0";; P05J7) echo "20 0";; P15J15) echo "7 10";; P15J3) echo "7 13";; P15J7) echo "7 22";; esac; }
used() { nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " "; }
idle() { [ "$(used $1)" -lt 1000 ]; }
log() { echo "$(date +%FT%T%z) $*" >> $LOG; }
# take the first queue line of width w (1 or 2); prints it and deletes it (under a lock)
take() { local w=$1; ( flock 9; [ -e runs/stop ] && exit 0; n=0
  while read P J s T M; do n=$((n+1)); [ -z "$P" ] && continue
    ww=1; [ $P = P15 ] && [ "$M" != 1g ] && ww=2
    [ "$M" = 1g ] && [ "$s" -ge 300 ] && [ ! -e runs/p15_1g_ok ] && continue   # real 1-GPU P15 runs wait for the smoke check
    if [ $ww = $w ]; then sed -i ${n}d runs/queue.txt; echo "$P $J $s $T $M"; exit 0; fi
  done < runs/queue.txt ) 9>runs/queue.lock; }
has() { [ "$1" = 2 ] && grep "^P15" runs/queue.txt 2>/dev/null | grep -vq " 1g$" && return 0
  [ "$1" = 1 ] && grep -Eq "^P05| 1g$" runs/queue.txt 2>/dev/null && return 0; return 1; }
run1() { local gp=$1 gj=$2 P=$3 J=$4 s=$5 T=$6 M=$7; source code/server/needs.sh
  set -- $(need $P $J); local np=$1 nj=$2; set -- $(load $P $J); local lp=$1 lj=$2
  local mb=2; if [ "$M" = 1g ]; then np=$NEED_P15_1G; nj=0; lp=$((lp+lj)); lj=0; mb=1; fi   # SPEC supplement 4: P15 on one GPU
  local cap=$(python3 -c "print($np-0.5)"); local id=$D-$P$J-s$s
  [ -d runs/$id ] && mv runs/$id runs/aborted/$id-$(date +%H%M%S) && rm -rf ckpts/$id
  mkdir -p runs/$id
  local pol=$P05; [ $P = P15 ] && pol=$P15; local jp=$J15; [ $J = J3 ] && jp=$J3; [ $J = J7 ] && jp=$J7
  local vis=$gp extra="--reserve-gib $(python3 -c "print(max($cap-1-$lp,0))")"
  [ "$gj" != "-" ] && { vis=$gp,$gj; extra="$extra --judge-device cuda:1 --reserve-judge-gib $(python3 -c "print(max($nj-2-$lj,0))")"; }
  log "start gpus=$vis $id steps=$T cap=$cap"
  CUDA_VISIBLE_DEVICES=$vis python code/grpo39.py --steps $T --eval-every 10 --lr 3e-6 --scale group --readout G_cal \
    --micro-bs $mb --sft-path $pol --judge-path $jp --readouts-json data/readouts_${J}_${P}.json --seed $s --run-id $id --mem-gib $cap $extra \
    > runs/$id/stdout.log 2>&1 &
  local pid=$!; echo "$id $pid" >> runs/pids.txt; wait $pid; local rc=$?
  log "end gpus=$vis $id rc=$rc"
  if [ "$M" = 1g ] && [ "$s" -lt 300 ] && [ $rc = 0 ]; then   # smoke check for the 1-GPU P15 mode: peak allocated < 37 GiB
    python3 -c "
import json,sys; m=max(json.loads(l)['max_mem_alloc_gib'] for l in open('runs/$id/steps.jsonl')); print(m); sys.exit(0 if m < 37 else 1)" >> $LOG && { touch runs/p15_1g_ok; log "1-GPU P15 smoke ok"; }
  fi
  if [ $rc != 0 ] && grep -q "Process [0-9]* has .* memory in use" runs/$id/stdout.log; then   # external OOM: requeue (SPEC supplement 3)
    k=$(grep -c "^$P $J $s " runs/ext_oom.txt 2>/dev/null); echo "$P $J $s $T $M" >> runs/ext_oom.txt
    [ "${k:-0}" -lt 3 ] && { ( flock 9; sed -i "1i $P $J $s $T $M" runs/queue.txt ) 9>runs/queue.lock; log "external OOM: requeued $id"; }
  fi; }
canhold() { local last=$(cat runs/holder$1.last 2>/dev/null || echo 0); [ $(( $(date +%s) - last )) -ge 5400 ]; }   # 60 min hold + 30 min cool-down
hold() { local g=$1 tag=$2; canhold $g || { log "no hold on gpu $g (cool-down)"; return; }; date +%s > runs/holder$g.last; CUDA_VISIBLE_DEVICES=$g python -c "
import torch,time
x=torch.empty(int(36*2**30),dtype=torch.uint8,device='cuda:0'); time.sleep(3600)" > /dev/null 2>&1 &
  local pid=$!; echo "holder-$tag-gpu$g $pid" >> runs/pids.txt; echo $pid > runs/holder$g.pid; log "holder on gpu $g pid=$pid"; }
unhold() { local g=$1; [ -f runs/holder$g.pid ] || return; local pid=$(cat runs/holder$g.pid); kill $pid 2>/dev/null; wait $pid 2>/dev/null
  rm -f runs/holder$g.pid; log "holder on gpu $g released pid=$pid"; for i in $(seq 1 20); do idle $g && break; sleep 1; done; }
# a chain owns its GPU(s) and keeps starting our next job of the same width
chain1() { local g=$1; local line=$2; touch runs/chain.g$g
  while [ -n "$line" ]; do set -- $line; run1 $g - $1 $2 $3 $4 $5; line=$(take 1); done
  # no 1-GPU job left: if 2-GPU jobs are queued, hold this GPU (max 30 min) for a 2-GPU job
  if has 2 && [ ! -e runs/stop ]; then hold $g chain; fi; rm -f runs/chain.g$g; }
chain2() { local g0=$1 g1=$2; local line=$3; touch runs/chain.g$g0 runs/chain.g$g1
  while [ -n "$line" ]; do set -- $line; run1 $g0 $g1 $1 $2 $3 $4; line=$(take 2); done
  # no 2-GPU job left: give the GPUs to 1-GPU jobs
  rm -f runs/chain.g$g0 runs/chain.g$g1
  line=$(take 1); [ -n "$line" ] && chain1 $g0 "$line" &
  line=$(take 1); [ -n "$line" ] && chain1 $g1 "$line" &
  wait; }
holder_alive() { [ -f runs/holder$1.pid ] && kill -0 $(cat runs/holder$1.pid) 2>/dev/null; }
owned() { [ -e runs/chain.g$1 ]; }
mine() { ! owned $1 && { idle $1 || holder_alive $1; }; }   # GPU usable by us: idle, or held by our holder
rm -f runs/chain.g0 runs/chain.g1
log "driver v5 start pid=$$"
while :; do
  [ -e runs/stop ] && { log "stop file seen; not starting new runs"; break; }
  [ -s runs/queue.txt ] || { [ -e runs/queue.closed ] && break; }
  for g in 0 1; do holder_alive $g || rm -f runs/holder$g.pid; done
  started=0
  if has 2 && mine 0 && mine 1; then unhold 0; unhold 1; line=$(take 2); [ -n "$line" ] && { chain2 0 1 "$line" & started=1; }; fi
  if [ $started = 0 ]; then for g in 0 1; do
    if mine $g && has 1; then unhold $g; line=$(take 1); [ -n "$line" ] && { chain1 $g "$line" & started=1; break; }; fi
    if ! owned $g && idle $g && ! has 1 && has 2 && ! holder_alive $g && canhold $g; then hold $g main; fi
  done; fi
  if [ $started = 1 ]; then sleep 120; else sleep 10; fi
done
log "queue loop ended; waiting for running chains"
wait
for g in 0 1; do unhold $g; done
log "driver done"
