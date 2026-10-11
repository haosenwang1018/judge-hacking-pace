"""EXP-013 shared code: data, prompt, answer parsing, greedy eval, LLM-as-judge score.
EXP-018 copy (from EXP-014): adds cut_at_answer() and judge views in evaluate(). Nothing else changed."""
import json, os, re, random
import torch
from datasets import load_dataset

# A100 rerun (2026-10-05): both overridable by env; defaults = original Mac setup.
BASE_MODEL = os.environ.get("EXP013_BASE_MODEL", "Qwen/Qwen2.5-0.5B")  # base (not instruct); rev 060db649...
CKPT_ROOT = os.environ.get("EXP013_CKPT_ROOT", os.path.expanduser("~/exp013_ckpts"))  # large checkpoints outside the repo
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"  # same revision the Mac run had cached
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

PROMPT = ("Solve the math problem. Show your reasoning, then give the final answer "
          "on the last line in the form \"#### <number>\".\n\nProblem: {q}\nSolution:")

JUDGE_PROMPT = ("Below is a math problem and a proposed solution.\n\nProblem: {q}\n\nSolution: {s}\n\n"
                "Question: Is this solution clear and concise, with no unnecessary or repeated steps? "
                "Answer Yes or No.\nAnswer:")

N_SFT = 2000        # SFT examples (worked solutions)
N_EVAL_SMALL = 200  # held-out set used every N RL steps
N_EVAL_FULL = 500   # held-out set for S0 / S1 / final numbers


def clean_solution(ans: str) -> str:
    """GSM8K reference: drop <<calc>> annotations; keep the native '#### <n>' last line."""
    ans = re.sub(r"<<[^>]*>>", "", ans)
    return ans.strip()


def gold_of(ans: str) -> str:
    return norm_num(ans.split("####")[-1])


def norm_num(s):
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else str(v)


STRICT_RE = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_strict(text):
    """Format-correct = response contains '#### <number>'. Answer = first such number."""
    m = STRICT_RE.search(text)
    return norm_num(m.group(1)) if m else None


def cut_at_answer(text):
    """EXP-018 'answer view': keep text up to the end of the first '#### <number>' match; drop all after it.
    Text without a match is returned unchanged (such rollouts are never judged in training: S only for correct ones)."""
    m = STRICT_RE.search(text)
    return text[:m.end()] if m else text


def parse_lenient(text):
    """Strict answer if present, else last number before the model starts a new 'Problem:'."""
    a = parse_strict(text)
    if a is not None:
        return a
    head = re.split(r"\n\s*(Problem|Question)\s*:", text)[0]
    nums = NUM_RE.findall(head)
    return norm_num(nums[-1]) if nums else None


def load_splits(seed=0):
    ds = load_dataset("openai/gsm8k", "main", revision=GSM8K_REVISION)
    train = list(ds["train"])
    test = list(ds["test"])
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(test)
    sft = train[:N_SFT]
    rl = train[N_SFT:]
    return sft, rl, test[:N_EVAL_SMALL], test[:N_EVAL_FULL]


def load_model(path, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(DEVICE)
    return model, tok


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=256, bs=100):
    was_training = model.training
    model.eval()
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        g = g[:, enc["input_ids"].shape[1]:]
        for row in g:
            ids = row.tolist()
            eos = tok.eos_token_id in ids
            if eos:
                ids = ids[:ids.index(tok.eos_token_id)]
            while ids and ids[-1] == tok.pad_token_id:
                ids.pop()
            outs.append({"text": tok.decode(ids, skip_special_tokens=True), "n_tokens": len(ids), "eos": eos})
    if was_training:
        model.train()
    return outs


@torch.no_grad()
def judge_scores(judge, tok, questions, solutions, bs=32):
    """S in [0,1] = P(' Yes') / (P(' Yes') + P(' No')) at the next token after the judge prompt."""
    was_training = judge.training
    judge.eval()
    yes = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    no = tok(" No", add_special_tokens=False)["input_ids"][0]
    old_side = tok.padding_side
    tok.padding_side = "left"
    out = []
    for i in range(0, len(questions), bs):
        texts = [JUDGE_PROMPT.format(q=q, s=s.strip()[:1500]) for q, s in zip(questions[i:i + bs], solutions[i:i + bs])]
        enc = tok(texts, return_tensors="pt", padding=True).to(judge.device)
        logits = judge(**enc, logits_to_keep=1).logits[:, -1, :].float()
        two = torch.stack([logits[:, yes], logits[:, no]], dim=-1).softmax(-1)[:, 0]
        out.extend(two.tolist())
    tok.padding_side = old_side
    if was_training:
        judge.train()
    return out


def evaluate(model, tok, items, judges=None, max_new_tokens=256):
    """Greedy eval. judges: dict name -> model (shares tokenizer). Returns (metrics, rows)."""
    prompts = [PROMPT.format(q=x["question"]) for x in items]
    gens = generate(model, tok, prompts, max_new_tokens=max_new_tokens)
    rows = []
    for x, g in zip(items, gens):
        gold = gold_of(x["answer"])
        s = parse_strict(g["text"])
        rows.append({"question": x["question"], "gold": gold, "response": g["text"],
                     "n_tokens": g["n_tokens"], "eos": g["eos"],
                     "format_ok": s is not None, "correct": s == gold,
                     "lenient_correct": parse_lenient(g["text"]) == gold})
    n = len(rows)
    m = {"n": n,
         "acc": sum(r["correct"] for r in rows) / n,
         "format_rate": sum(r["format_ok"] for r in rows) / n,
         "lenient_acc": sum(r["lenient_correct"] for r in rows) / n,
         "mean_len": sum(r["n_tokens"] for r in rows) / n,
         "eos_rate": sum(r["eos"] for r in rows) / n}
    for name, j in (judges or {}).items():
        # EXP-018: each judge scores two views. "full" key names are unchanged from EXP-014; "_ans" = answer view.
        for suf, view in (("", lambda t: t), ("_ans", cut_at_answer)):
            sc = judge_scores(j, tok, [r["question"] for r in rows], [view(r["response"]) for r in rows])
            for r, v in zip(rows, sc):
                r[f"judge_{name}{suf}"] = v
            m[f"judge_{name}{suf}_all"] = sum(sc) / n
            cs = [v for r, v in zip(rows, sc) if r["correct"]]
            m[f"judge_{name}{suf}_correct"] = sum(cs) / len(cs) if cs else None
    for r in rows:  # EXP-018: tail = text after the first '#### n'
        r["tail_chars"] = len(r["response"][len(cut_at_answer(r["response"])):].strip())
    m["tail_share"] = sum(r["tail_chars"] > 0 for r in rows) / n
    return m, rows


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
