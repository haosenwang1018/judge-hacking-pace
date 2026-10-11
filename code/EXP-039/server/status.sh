#!/bin/bash
# EXP-039 compact status: running runs and their last step, queue length, GPU memory
cd /data/exp039
for d in runs/2026*-e39-*; do [ -f $d/steps.jsonl ] || continue; s=$(tail -1 $d/steps.jsonl | python3 -c "import sys,json;print(json.loads(sys.stdin.read())['step'])" 2>/dev/null); st=$(python3 -c "import json;print(json.load(open('$d/run.json')).get('status'))" 2>/dev/null); echo "$(basename $d | sed 's/.*e39-//') step=$s $st"; done
echo "queue=$(grep -c . runs/queue.txt) | $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')| $(tail -1 runs/driver.log | cut -c12-19,26-)"
