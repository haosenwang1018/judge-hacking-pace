# EXP-034 gpu-server-2 environment (no credentials). Reuses the EXP-029 venv (torch 2.9.1+cu128, transformers 5.18.0, trl 1.14.1).
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export EXP013_CKPT_ROOT=/data/exp034/ckpts PATH=/data/exp029/venv/bin:$PATH
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
cd /data/exp034
P05=/data/exp029/models/sft                       # sft-s2 (EXP-013), Qwen2.5-0.5B, read only
P15=/data/exp034/models/sft15-s2                  # same SFT recipe on Qwen2.5-1.5B, seed 2 (this experiment)
J15=/data/exp029/models/judge                     # Qwen2.5-1.5B-Instruct, read only
J7=/data/exp034/models/Qwen2.5-7B-Instruct
