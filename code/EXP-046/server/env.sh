# EXP-046 environment (no credentials). Same for gpu-server-2 and gpu-server-1; the host is detected from what exists on disk.
# Writable state only in /data/exp046. RL outputs and checkpoints only under /data/exp046/blind (BLINDING, SPEC section 7).
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EXP046_BLIND=/data/exp046/blind EXP013_CKPT_ROOT=/data/exp046/blind/ckpts
if [ -d /data/exp029/models/sft ]; then        # gpu-server-2: the EXP-039 / EXP-041 venv and policy copy (read only)
  HOSTKEY=gpu-server-2; export PATH=/data/exp029/venv/bin:$PATH; P05=/data/exp029/models/sft; J15=/data/exp029/models/judge
else                                           # gpu-server-1: venv rebuilt with the same package versions (setup46_gpu-server-1.sh)
  HOSTKEY=gpu-server-1;  export PATH=/data/exp046/venv/bin:$PATH; P05=/data/exp046/models/sft; J15=/data/models/Qwen2.5-1.5B-Instruct
fi
JL1=/data/models/Llama-3.2-1B-Instruct          # held-out judges (SPEC section 2)
JL8=/data/models/Llama-3.1-8B-Instruct
JO1=/data/models/OLMo-2-0425-1B-Instruct
JO7=/data/models/OLMo-2-1124-7B-Instruct
cd /data/exp046
