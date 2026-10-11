#!/bin/bash
# EXP-046 0-GPU job on gpu-server-1: (1) venv with the exact package versions of the gpu-server-2 venv (Python 3.12.3, torch 2.9.1 cu128,
# transformers 5.18.0, trl 1.14.1; freeze_gpu-server-2.txt), (2) copy the P05 policy (sft-s2) from gpu-server-2 and check its sha256,
# (3) download + verify JO1 and JO7, queueing each judge's GPU chain as soon as it is ready.
# v2 (261010 10:2x): uv from the aliyun PyPI mirror and rsync with retries; v1's plain scp from gpu-server-2 stalled.
set -uo pipefail
cd /data/exp046; mkdir -p logs models
export UV_PYTHON_INSTALL_DIR=/data/exp046/python UV_CACHE_DIR=/data/exp046/.uvcache
# robust copy from gpu-server-2: rsync --partial with ssh keepalive, killed after 15 min without finishing, up to 20 tries
pull() { for t in $(seq 1 20); do timeout 900 rsync -a --partial -e "ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=4" "$@" && return 0; sleep 10; done; return 1; }
{
[ -x uvpkg/bin/uv ] || python3 -m pip install -q --target uvpkg --index-url https://mirrors.aliyun.com/pypi/simple/ uv==0.12.23 || exit 21
UV=$PWD/uvpkg/bin/uv
if [ ! -f venv/.OK ]; then
  rm -rf venv
  $UV venv --python 3.12.3 venv || exit 22
  $UV pip install --python venv/bin/python --index-url https://mirrors.aliyun.com/pypi/simple/ -r code/server/freeze_gpu-server-2.txt || exit 23
  venv/bin/python code/server/check_venv46.py || exit 24
  touch venv/.OK
fi
if [ ! -f models/sft/.OK ]; then
  pull gpu-server-2:/data/exp029/models/sft models/ && pull gpu-server-2:/data/exp029/models/sft.sha256 models/ || exit 25
  (cd models/sft && sha256sum -c ../sft.sha256) || exit 26
  touch models/sft/.OK
  # diagnostic only: gpu-server-1's own sft-s2 file (sha256 1596e1d3...) differs from the gpu-server-2 copy (1ef340d1...)
  venv/bin/python code/server/sft_compare46.py > logs/sft_compare_gpu-server-1.json || true
fi
} > logs/setup_gpu-server-1.log 2>&1 || { echo "setup failed, see logs/setup_gpu-server-1.log"; exit 1; }
rc=0
for J in JO1 JO7; do bash code/server/dl_then_gate46.sh $J || rc=1; done
exit $rc
