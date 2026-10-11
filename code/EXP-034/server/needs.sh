# EXP-034 per-GPU memory needs in GiB (nvidia-smi used + need <= 38). Measured in the step-0 smoke runs (PROCESS.md).
NEED_P05J15=21     # cap 20.5; 261009 04:46 OOM with micro-batch 8 at cap 20.5, so all cells now use micro-batch 2
NEED_P05J7=34       # smoke: max allocated 31.6 GiB (policy, reference copy and 7B judge on one GPU)
NEED_P15_POL=33    # micro-batch 2: max allocated 29.3 GiB at step 8
NEED_P15J15_J=18   # judge J15 + fp32 reference copy: nvidia-smi 16.8 GB
NEED_P15J7_J=27     # estimate: 7B judge 15.7 GiB (static run) + fp32 reference copy 6.2 + activations
