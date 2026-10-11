#!/usr/bin/env bash
# Rebuild the original working layout (EXP-0NN/{code,results,data}, PAPER-T11/{rev261011,figures})
# with symlinks under ./workdir, so the analysis scripts run unchanged with their relative paths.
# Usage (from the repository root): bash tools/make_workdir.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; W="$ROOT/workdir"
mkdir -p "$W/PAPER-T11/rev261011" "$W/PAPER-T11/figures"
for d in "$ROOT"/code/EXP-*; do
  e=$(basename "$d"); mkdir -p "$W/$e"
  ln -sfn "$ROOT/code/$e" "$W/$e/code"
  [ -d "$ROOT/results/$e" ] && ln -sfn "$ROOT/results/$e" "$W/$e/results"
  [ -d "$ROOT/results/$e/runs" ] && ln -sfn "$ROOT/results/$e/runs" "$W/$e/runs"
  [ -d "$ROOT/data/$e" ] && ln -sfn "$ROOT/data/$e" "$W/$e/data"
done
for f in "$ROOT"/code/paper/analysis/*.py "$ROOT"/results/paper/*.json; do ln -sfn "$f" "$W/PAPER-T11/rev261011/$(basename "$f")"; done
for f in "$ROOT"/code/paper/figures/*.py; do ln -sfn "$f" "$W/PAPER-T11/figures/$(basename "$f")"; done
echo "workdir ready: $W"
