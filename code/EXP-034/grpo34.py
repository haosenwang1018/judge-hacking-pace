"""EXP-034: copy of EXP-029 code/grpo29.py. Changes (EXP-034 SPEC section 5): --judge-device (7B judge and the frozen
reference copy on a second GPU), eval schedule every 5 steps up to step 50 and every 10 after, default 150 steps,
per-step within-batch AUC of the judge score (gold correct vs wrong). Reward = G_cal only. Original docstring:
EXP-029: copy of EXP-023 code/grpo_judge.py. Changes (SPEC section 4): judge29.Judge (judge A or B), --readout gold
(reward = strict gold correctness 0/1, judge readouts still logged), readout loops use only the readouts the judge has.
Original docstring:
EXP-023 step 2: GRPO with a judge-only reward (gold hidden from the judge), TRL GRPOTrainer on GSM8K.
Built from EXP-018 code/grpo.py (copied, not edited in place). Same hyperparameters as EXP-018.
  --readout G_cal|P_align   reward = that readout's probability from the frozen Qwen2.5-1.5B-Instruct judge (judge1p5.py)
  --scale group|batch       TRL scale_rewards: divide the group-centred reward by the group std or by the batch std
No format bonus, no gold in the reward. Gold correctness is only logged.
Logs: runs/<id>/steps.jsonl (per step), rollouts.jsonl (every rollout, all 4 readouts), evals.jsonl +
eval_outputs_stepNNN.jsonl (greedy, 200 test items, all 4 readouts).
"""
import argparse, json, os, re, sys, time, platform
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, transformers, trl
from datasets import Dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from common import PROMPT, load_splits, gold_of, parse_strict, load_model, generate, write_json, append_jsonl, \
    cut_at_answer, STRICT_RE, CKPT_ROOT, EXP_DIR, DEVICE
from judge34 import Judge
from stats import advantages, auc

ap = argparse.ArgumentParser()
ap.add_argument("--readout", choices=["G_cal", "P_align", "P_full", "P_match", "gold"], required=True)
ap.add_argument("--scale", choices=["group", "batch", "none"], required=True)
ap.add_argument("--seed", type=int, default=301)
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--micro-bs", type=int, default=8, help="per-device micro-batch (rollouts); grad accumulation = 64 / micro-bs, so each step is still 8 questions x 8 rollouts")
ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (P15 runs: memory headroom; same math)")
ap.add_argument("--judge-device", default=None, help="device for the frozen judge and the frozen reference copy")
ap.add_argument("--lr", type=float, default=3e-6)
ap.add_argument("--eval-every", type=int, default=10)
ap.add_argument("--run-id", required=True)
ap.add_argument("--sft-path", required=True)
ap.add_argument("--judge-path", required=True)
ap.add_argument("--readouts-json", required=True)
ap.add_argument("--eval-n", type=int, default=200)
ap.add_argument("--mem-gib", type=float, default=0, help="hard per-process CUDA memory cap (0 = none)")
a = ap.parse_args()

if a.mem_gib and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(a.mem_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)
OUT = os.path.join(EXP_DIR, "runs", a.run_id)
os.makedirs(OUT, exist_ok=True)
G = 8
_, rl, small, _ = load_splits()
small = small[:a.eval_n]
ds = Dataset.from_list([{"prompt": PROMPT.format(q=x["question"]), "question": x["question"],
                         "gold": gold_of(x["answer"])} for x in rl])
JDEV = a.judge_device or DEVICE
JUDGE = Judge(a.judge_path, JDEV, a.readouts_json)
READOUTS = [k for k in ("G", "G_cal", "P_align", "P_full", "P_match") if k == "G" or k in JUDGE.R]
model, tok = load_model(a.sft_path)
# v0.2 (review C1.3, I3): frozen copy of the initial policy for KL distance and parameter distance at each eval
ref, _ = load_model(a.sft_path)
ref = ref.to(JDEV)
ref.eval().requires_grad_(False)
TAU = JUDGE.R["tau_tpr80_d8"]   # per-readout threshold: TPR = 0.80 on the step-0 set D8 (fixed before RL)
SDZ = JUDGE.R["sd_d8"]          # per-readout sd on D8 (z units)
STEP_T0 = [time.time()]


def flags(text, gold_free=True):
    """E1 repeat: after the first '#### n', the tail has another '#### ' line or repeats n.
    E2 short: < 16 tokens or no digit at all. tail_chars: text after the first '#### n'."""
    cut = cut_at_answer(text)
    tail = text[len(cut):]
    m = STRICT_RE.search(text)
    rep = False
    if m:
        n = m.group(1).replace(",", "")
        rep = ("####" in tail) or (re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", tail.replace(",", "")) is not None)
    return {"tail_chars": len(tail.strip()), "repeat": rep, "no_digit": re.search(r"\d", text) is None}


def reward(prompts, completions, question, gold, trainer_state=None, **kw):
    n = len(completions)
    S = JUDGE.score(list(question), list(completions))
    cor = np.array([parse_strict(c) == g for c, g in zip(completions, gold)], int)
    R = [float(x) for x in cor] if a.readout == "gold" else S[a.readout]
    fmt = [parse_strict(c) is not None for c in completions]
    lens = [len(tok(c, add_special_tokens=False)["input_ids"]) for c in completions]
    fl = [flags(c) for c in completions]
    step = (trainer_state.global_step + 1) if trainer_state else None
    rec = {"step": step, "n": n, "reward": float(np.mean(R)), "correct": float(cor.mean()), "format": float(np.mean(fmt)),
           "mean_len": float(np.mean(lens)), "frac_len_ge_256": float(np.mean([l >= 256 for l in lens])),
           "frac_short": float(np.mean([l < 16 or f["no_digit"] for l, f in zip(lens, fl)])),
           "frac_repeat": float(np.mean([f["repeat"] for f in fl])),
           "frac_tail": float(np.mean([f["tail_chars"] > 0 for f in fl])),
           "mean_tail_chars": float(np.mean([f["tail_chars"] for f in fl]))}
    for k in READOUTS:
        v = np.array(S[k])
        rec[f"S_{k}_mean"] = float(v.mean())
        rec[f"S_{k}_correct"] = float(v[cor == 1].mean()) if cor.any() else None
        rec[f"S_{k}_wrong"] = float(v[cor == 0].mean()) if (cor == 0).any() else None
        if k in TAU:
            rec[f"FPR_{k}"] = float((v[cor == 0] >= TAU[k]).mean()) if (cor == 0).any() else None
            rec[f"TPR_{k}"] = float((v[cor == 1] >= TAU[k]).mean()) if cor.any() else None
        rec[f"auc_{k}"] = auc(v, cor) if 0 < cor.sum() < n else None
    rec["n_wrong"] = int((cor == 0).sum())
    for k in ("P_align", "P_full", "P_match"):
        if f"maxz_{k}" in S:
            rec[f"maxz_{k}_mean"] = float(np.mean(S[f"maxz_{k}"])); rec[f"maxz_{k}_max"] = float(np.max(S[f"maxz_{k}"]))
    r = np.array(R).reshape(-1, G)
    c = cor.reshape(-1, G)
    mixed = (c.sum(1) > 0) & (c.sum(1) < G)
    Ag = (c - c.mean(1, keepdims=True)).reshape(-1).astype(float)
    rec["frac_mixed"] = float(mixed.mean())
    rec["sd_within_median"] = float(np.median(r.std(1, ddof=1)))
    for mode in ("group", "batch"):
        A = advantages(R, G, mode)
        A2 = (A.reshape(-1, G) ** 2).sum(1)
        rec[f"cos_A_gold_{mode}"] = float(A @ Ag / (np.linalg.norm(A) * np.linalg.norm(Ag) + 1e-12)) if Ag.any() else None
        rec[f"share_A2_uniform_{mode}"] = float(A2[~mixed].sum() / (A2.sum() + 1e-12))
        rec[f"mean_abs_A_{mode}"] = float(np.abs(A).mean())
    now = time.time()
    rec.update(sec_since_last=now - STEP_T0[0], time=now,
               max_mem_reserved_gib=(torch.cuda.max_memory_reserved() / 2**30) if torch.cuda.is_available() else None,
               max_mem_alloc_gib=(torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else None)
    STEP_T0[0] = now
    append_jsonl(os.path.join(OUT, "steps.jsonl"), rec)
    with open(os.path.join(OUT, "rollouts.jsonl"), "a") as f:
        for i in range(n):
            f.write(json.dumps({"step": step, "i": i, "question": question[i], "gold": gold[i], "completion": completions[i],
                                "n_tokens": lens[i], "correct": bool(cor[i]), "format_ok": fmt[i], **fl[i],
                                **{k: S[k][i] for k in S}}, ensure_ascii=False) + "\n")
    return [float(x) for x in R]


@torch.no_grad()
def seq_logprob(m, prompts, responses, bs=4):
    """sum and count of response-token log-probs under model m (teacher forcing on the greedy outputs)."""
    was = m.training
    m.eval()
    out = []
    old = tok.padding_side
    tok.padding_side = "right"
    for i in range(0, len(prompts), bs):
        P, Rr = prompts[i:i + bs], responses[i:i + bs]
        pl = [len(tok(p)["input_ids"]) for p in P]
        enc = tok([p + r for p, r in zip(P, Rr)], return_tensors="pt", padding=True).to(m.device)
        lg = m(**enc).logits[:, :-1].float()
        tgt = enc["input_ids"][:, 1:]
        tl = lg.gather(-1, tgt[..., None])[..., 0] - torch.logsumexp(lg, -1)
        lp = lg
        mask = enc["attention_mask"][:, 1:].clone()
        for j, L in enumerate(pl):
            mask[j, :L - 1] = 0
        out += [(float((tl[j] * mask[j]).sum()), int(mask[j].sum())) for j in range(len(P))]
        del lp
    tok.padding_side = old
    if was:
        m.train()
    return out


def evaluate(step, pol):
    t = time.time()
    prompts = [PROMPT.format(q=x["question"]) for x in small]
    gens = generate(pol, tok, prompts, max_new_tokens=256)
    qs = [x["question"] for x in small]
    S = JUDGE.score(qs, [g["text"] for g in gens])
    rows = []
    for j, (x, g) in enumerate(zip(small, gens)):
        gold = gold_of(x["answer"])
        p = parse_strict(g["text"])
        rows.append({"question": x["question"], "gold": gold, "response": g["text"], "n_tokens": g["n_tokens"],
                     "eos": g["eos"], "format_ok": p is not None, "correct": p == gold, **flags(g["text"]),
                     **{k: S[k][j] for k in S}})
    n = len(rows)
    m = {"step": step, "n": n, "acc": sum(r["correct"] for r in rows) / n,
         "format_rate": sum(r["format_ok"] for r in rows) / n, "mean_len": sum(r["n_tokens"] for r in rows) / n,
         "frac_short": sum((r["n_tokens"] < 16) or r["no_digit"] for r in rows) / n,
         "frac_repeat": sum(r["repeat"] for r in rows) / n, "frac_tail": sum(r["tail_chars"] > 0 for r in rows) / n,
         "eval_seconds": None}
    for k in READOUTS:
        m[f"J_{k}"] = float(np.mean(S[k]))
        w = [r[k] for r in rows if not r["correct"]]
        m[f"J_{k}_wrong"] = float(np.mean(w)) if w else None
    for k in TAU:
        v = np.array(S[k]); w = np.array([not r["correct"] for r in rows])
        m[f"FPR_{k}"] = float((v[w] >= TAU[k]).mean()) if w.any() else None
        m[f"Jz_{k}"] = float(v.mean() / SDZ[k])
    # KL distance: mean over eval items of (log pi - log pi0) per response token, on the greedy outputs
    a_ = seq_logprob(pol, prompts, [g["text"] for g in gens])
    b_ = seq_logprob(ref, prompts, [g["text"] for g in gens])
    ntok = sum(x[1] for x in a_)
    m["kl_greedy_per_token"] = float(sum(x[0] - y[0] for x, y in zip(a_, b_)) / max(ntok, 1))
    m["kl_greedy_per_seq"] = float(np.mean([x[0] - y[0] for x, y in zip(a_, b_)]))
    m["param_dist"] = float(torch.sqrt(sum(((p.detach().float() - q.float().to(p.device)) ** 2).sum()
                                           for p, q in zip(pol.parameters(), ref.parameters()))))
    for k in ("P_align", "P_full", "P_match"):
        if f"maxz_{k}" in S:
            m[f"maxz_{k}_mean"] = float(np.mean(S[f"maxz_{k}"]))
    m["eval_seconds"] = time.time() - t
    append_jsonl(os.path.join(OUT, "evals.jsonl"), m)
    with open(os.path.join(OUT, f"eval_outputs_step{step:03d}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("EVAL", json.dumps(m), flush=True)


class EvalCB(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        evaluate(0, model)

    def on_step_end(self, args, state, control, model=None, **kw):
        if os.environ.get("E34_MEMDBG") and torch.cuda.is_available():
            print("MEMDBG", state.global_step, round(torch.cuda.memory_allocated() / 2**30, 2), round(torch.cuda.max_memory_allocated() / 2**30, 2), flush=True)
            if state.global_step == 3:
                from collections import Counter
                c = Counter()
                import gc
                for o in gc.get_objects():
                    try:
                        if torch.is_tensor(o) and o.is_cuda and o.device.index == 0:
                            c[(str(o.dtype), tuple(o.shape)[:3])] += o.numel() * o.element_size()
                    except Exception:
                        pass
                for k, v in c.most_common(12):
                    print("MEMDBG-T", k, round(v / 2**30, 3), flush=True)
        gs = state.global_step  # EXP-034 schedule: every 5 steps up to 50, then every a.eval_every (10)
        if ((gs <= 50 and gs % 5 == 0) or gs % a.eval_every == 0) and gs != args.max_steps:
            evaluate(gs, model)


cfg_kw = dict(output_dir=os.path.join(CKPT_ROOT, a.run_id), max_steps=a.steps, learning_rate=a.lr,
              lr_scheduler_type="constant_with_warmup", warmup_steps=3,
              per_device_train_batch_size=a.micro_bs, gradient_accumulation_steps=64 // a.micro_bs, num_generations=G,
              max_completion_length=256, temperature=1.0, beta=0.0, loss_type="dapo", scale_rewards=a.scale,
              logging_steps=1, save_strategy="steps", save_steps=50, save_only_model=True,  # 261008: weights at 50/100/150/200, no optimizer state
              report_to=[], seed=a.seed, shuffle_dataset=True, gradient_checkpointing=a.grad_ckpt,
              optim="adamw_torch_fused")  # 261009: fused AdamW (same update rule as the default foreach AdamW, no ~9 GB temporaries for the 1.5B policy); all cells
cfg = GRPOConfig(**cfg_kw)
# EXP-034: with a second visible GPU (judge device) HF Trainer would wrap the policy in nn.DataParallel (n_gpu = 2) and
# double the batch (smoke 261009: 128 rollouts per step). Force one training GPU so every run uses 8 questions x 8 rollouts.
_ = cfg._setup_devices
cfg._n_gpu = 1
write_json(os.path.join(OUT, "config.json"), {**cfg_kw, **vars(a), "G": G,
           "reward": ("gold correctness 0/1 (strict '#### n'); judge readouts only logged" if a.readout == "gold" else
                       f"judge-only: {a.readout} probability from frozen judge {a.judge_path} (no-reference prompt); no format bonus")})
write_json(os.path.join(OUT, "run.json"), {"run_id": a.run_id, "kind": "train-grpo-judge-only", "status": "running",
           "command": " ".join(sys.argv), "seed": a.seed, "parent_run": "20261005-gpu-server-1-sft-s2-a1",
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "machine": platform.node(), "device": DEVICE, "torch": torch.__version__,
           "transformers": transformers.__version__, "trl": trl.__version__})
TRAINER = GRPOTrainer(model=model, args=cfg, reward_funcs=reward, train_dataset=ds,
                      processing_class=tok, callbacks=[EvalCB()])
assert TRAINER.args.n_gpu == 1 and TRAINER.args.train_batch_size * TRAINER.args.gradient_accumulation_steps == 64, (TRAINER.args.n_gpu, TRAINER.args.train_batch_size)
t0 = time.time()
TRAINER.train()
train_secs = time.time() - t0
with open(os.path.join(OUT, "trl_log_history.jsonl"), "w") as f:
    for r in TRAINER.state.log_history:
        f.write(json.dumps(r) + "\n")
evaluate(a.steps, TRAINER.model)
_fin, _c200 = os.path.join(CKPT_ROOT, a.run_id, "final"), os.path.join(CKPT_ROOT, a.run_id, f"checkpoint-{a.steps}")
if os.path.isdir(_c200):  # 261008: final weights = the step-200 checkpoint; link instead of a second 2 GB copy
    os.symlink(_c200, _fin)
else:
    TRAINER.save_model(_fin)
rj = json.load(open(os.path.join(OUT, "run.json")))
rj.update(status="completed", train_seconds=train_secs, total_seconds=time.time() - t0,
          max_mem_reserved_gb=(torch.cuda.max_memory_reserved() / 2**30) if torch.cuda.is_available() else None)
write_json(os.path.join(OUT, "run.json"), rj)
print("DONE", flush=True)
