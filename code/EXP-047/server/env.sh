# EXP-047 gpu-server-2 environment (no credentials). Same venv as EXP-039/041. Models and step-0 data are read in place (read only).
# Writable state only in /data/exp047.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH=/data/exp029/venv/bin:$PATH
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/exp047
STEP0=/data/exp041/data/step0                     # d8.jsonl + d8.G_<J>.npy for J15, J3, J7, JL (EXP-039/041 step 0)
READOUTS=/data/exp041/data                        # readouts_<J>_<P>.json (G_cal a, b)
J15=/data/exp029/models/judge                     # Qwen2.5-1.5B-Instruct
J3=/data/exp039/models/Qwen2.5-3B-Instruct        # Qwen2.5-3B-Instruct (Qwen Research License)
J7=/data/exp034/models/Qwen2.5-7B-Instruct        # Qwen2.5-7B-Instruct
JL=/data/exp029/models/Llama-3.2-3B-Instruct      # Llama-3.2-3B-Instruct
