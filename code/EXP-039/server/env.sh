# EXP-039 gpu-server-2 environment (no credentials). Reuses the EXP-029 venv (torch 2.9.1+cu128, transformers 5.18.0, trl 1.14.1).
# Models are read in place (read only): P05 and J15 from EXP-029, P15 and J7 from EXP-034. Writable state only in /data/exp039.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export EXP013_CKPT_ROOT=/data/exp039/ckpts PATH=/data/exp029/venv/bin:$PATH
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
cd /data/exp039
P05=/data/exp029/models/sft                       # sft-s2 (EXP-013), Qwen2.5-0.5B
P15=/data/exp034/models/sft15-s2                  # EXP-034 SFT of Qwen2.5-1.5B, same recipe, seed 2
J15=/data/exp029/models/judge                     # Qwen2.5-1.5B-Instruct
J3=/data/exp039/models/Qwen2.5-3B-Instruct        # Qwen2.5-3B-Instruct, HF revision aa8e7253 (Qwen Research License)
J7=/data/exp034/models/Qwen2.5-7B-Instruct        # Qwen2.5-7B-Instruct, HF revision a09a3545
LLAMA=/data/exp029/models/Llama-3.2-3B-Instruct   # offline second-family rescoring only
