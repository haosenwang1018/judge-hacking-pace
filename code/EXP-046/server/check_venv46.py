"""EXP-046: check that the gpu-server-1 venv has exactly the package versions of the gpu-server-2 venv (freeze_gpu-server-2.txt)."""
import importlib.metadata as m
want = dict(l.strip().split("==") for l in open("code/server/freeze_gpu-server-2.txt") if "==" in l)
bad = {k: (v, m.version(k)) for k, v in want.items() if m.version(k) != v}
import torch, transformers, trl
print(torch.__version__, torch.version.cuda, transformers.__version__, trl.__version__)
assert not bad, bad
print("venv OK", len(want), "packages")
