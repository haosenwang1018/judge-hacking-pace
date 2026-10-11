"""EXP-034 step 0: step-0 policy samples for a policy (same question sets and sampling as EXP-021 D0 / EXP-023 D8).
  d0: RL-split questions 0..999 x 2 samples   (calibration of G_cal)
  d8: RL-split questions 1000..1255 x 8 samples (static gate, tau, FPR0)
T = 1.0, top_k 0, top_p 1, max 256 new tokens, policy prompt of EXP-013.
usage: python gen_step0.py <policy_path> <out_dir>   -> <out_dir>/d0.jsonl, d8.jsonl
"""
import json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import PROMPT, load_splits, gold_of, parse_strict, load_model

CKPT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
_, rl, _, _ = load_splits(seed=0)
model, tok = load_model(CKPT, dtype=torch.float32)
model.eval()
t0 = time.time()
for name, q0, nq, k in (("d0", 0, 1000, 2), ("d8", 1000, 256, 8)):
    torch.manual_seed(0)
    prompts = [(q0 + qi, x) for qi, x in enumerate(rl[q0:q0 + nq]) for _ in range(k)]
    with open(os.path.join(OUT, f"{name}.jsonl"), "w") as f, torch.no_grad():
        for b in range(0, len(prompts), 128):
            chunk = prompts[b:b + 128]
            enc = tok([PROMPT.format(q=x["question"]) for _, x in chunk], return_tensors="pt", padding=True).to(model.device)
            g = model.generate(**enc, max_new_tokens=256, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                               pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            g = g[:, enc["input_ids"].shape[1]:]
            for (qi, x), row in zip(chunk, g):
                ids = row.tolist()
                if tok.eos_token_id in ids:
                    ids = ids[:ids.index(tok.eos_token_id)]
                while ids and ids[-1] == tok.pad_token_id:
                    ids.pop()
                text = tok.decode(ids, skip_special_tokens=True)
                gold = gold_of(x["answer"])
                f.write(json.dumps({"set": name, "qid": f"rl{qi}", "question": x["question"], "gold": gold, "response": text,
                                    "n_tokens": len(ids), "correct": parse_strict(text) == gold}, ensure_ascii=False) + "\n")
            print(name, f"{b + len(chunk)}/{len(prompts)} {time.time() - t0:.0f}s", flush=True)
print("done", time.time() - t0)
