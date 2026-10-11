# EXP-041 gpu-server-2 environment (no credentials). Same venv and models as EXP-039 (read only). Writable state only in /data/exp041.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export EXP013_CKPT_ROOT=/data/exp041/ckpts PATH=/data/exp029/venv/bin:$PATH
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/exp041
P05=/data/exp029/models/sft                       # sft-s2 (EXP-013), Qwen2.5-0.5B
P15=/data/exp034/models/sft15-s2                  # EXP-034 SFT of Qwen2.5-1.5B
JL=/data/exp029/models/Llama-3.2-3B-Instruct      # the only change vs EXP-039: frozen judge = Llama-3.2-3B-Instruct (EXP-029 copy)
