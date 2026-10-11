# EXP-039 per-GPU memory needs in GiB. J15 / J7 values from EXP-034 smoke runs; J3 values from EXP-039 smoke runs (PROCESS.md).
NEED_P05J15=21
NEED_P05J7=34
NEED_P05J3=26      # smoke 261009 11:0x: max allocated 17.7 GiB at step 6 (micro-batch 2); cap 25.5 for growth
NEED_P15_POL=33
NEED_P15J15_J=18
NEED_P15J3_J=22    # estimate: J15 judge GPU 18 + 3.1 GB more judge weights (no P15xJ3 smoke: two idle GPUs are rare; no cap on the judge GPU)
NEED_P15J7_J=27
NEED_P15_1G=39     # SPEC supplement 4: P15 policy + judge + reference on one GPU, micro-batch 1 (whole idle GPU; cap 38.5)
