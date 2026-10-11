"""EXP-046 diagnostic: compare the gpu-server-2 sft-s2 copy (used by EXP-039/041/046) with gpu-server-1's own sft-s2 file, tensor by tensor."""
import json
from safetensors.torch import load_file
a = load_file("/data/exp046/models/sft/model.safetensors")
b = load_file("/data/exp013/ckpts/20261005-gpu-server-1-r1-sft-s2-a1/final/model.safetensors")
d = [k for k in a if k not in b or a[k].dtype != b[k].dtype or a[k].shape != b[k].shape or not bool((a[k] == b[k]).all())]
print(json.dumps({"n_tensors": len(a), "n_differ": len(d), "examples": d[:5], "dtypes": sorted({str(v.dtype) for v in a.values()})}))
