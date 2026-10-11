"""EXP-034: copy of EXP-013 code/sft.py, unchanged except this line; base model from EXP013_BASE_MODEL (Qwen2.5-1.5B for P15).
S1: full-parameter SFT of Qwen2.5-0.5B base on 2000 GSM8K worked solutions (TRL SFTTrainer, completion-only loss)."""
import json, os, sys, time, platform
sys.path.insert(0, os.path.dirname(__file__))
import torch, transformers, trl
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from common import *

RUN = sys.argv[1] if len(sys.argv) > 1 else "20261005-sft-s1-a1"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1          # Mac run used seed 1
SFT_OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(CKPT_ROOT, "sft")
out = os.path.join(EXP_DIR, "runs", RUN)
sft, _, _, _ = load_splits()
ds = Dataset.from_list([{"prompt": PROMPT.format(q=x["question"]), "completion": " " + clean_solution(x["answer"])} for x in sft])
kw = dict(output_dir=SFT_OUT + "_tmp", num_train_epochs=1, per_device_train_batch_size=8,
          gradient_accumulation_steps=2, learning_rate=1e-5, lr_scheduler_type="cosine", warmup_steps=6,
          logging_steps=5, save_strategy="no", report_to=[], seed=SEED, max_length=512,
          completion_only_loss=True, packing=False)
cfg = SFTConfig(**kw)
model, tok = load_model(BASE_MODEL)
tok.padding_side = "right"
tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
# sanity: show that EOS is appended and only the completion is trained on
ex = tr.train_dataset[0]
print("KEYS", list(ex.keys()), "last ids", ex["input_ids"][-3:], "eos", tok.eos_token_id)
t0 = time.time()
tr.train()
secs = time.time() - t0
tr.save_model(SFT_OUT); tok.save_pretrained(SFT_OUT)
write_json(os.path.join(out, "config.json"), kw)
with open(os.path.join(out, "train_log.jsonl"), "w") as f:
    for r in tr.state.log_history: f.write(json.dumps(r) + "\n")
write_json(os.path.join(out, "run.json"), {"run_id": RUN, "kind": "train-sft", "status": "completed", "seconds": secs,
    "base_model": BASE_MODEL, "n_examples": len(ds), "checkpoint": SFT_OUT, "seed": SEED,
    "command": " ".join(sys.argv), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "machine": platform.node(), "device": DEVICE,
    "torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__})
print("SFT done", secs)
