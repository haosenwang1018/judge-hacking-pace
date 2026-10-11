"""EXP-034: copy of EXP-029 code/judge29.py. Change: forward(hidden=False) skips hidden states (G_cal-only rewards,
less memory for the 7B judge); score() uses it when no probe readout is configured. Original docstring:
EXP-029 judge wrapper. Copied from EXP-023 code/judge1p5.py; changes (SPEC section 4):
  - works for judge A (Qwen2.5-1.5B-Instruct) and judge B (Llama-3.2-3B-Instruct): chat-template kwargs, pad token,
    Yes/No token ids, probe layer read from readouts.json ("layer", default 18 as in EXP-023)
  - Llama 3.2 chat template inserts today's date; it is pinned to the template default (date_string="26 Jul 2024")
  - feats(): last-position AND answer-position-mean hidden states of every layer (for R4)
Prompt text is identical to EXP-021/023 for both judges.
"""
import json
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

USER = {
    "noref": ("Below is a math problem and a proposed solution.\n\nProblem: {q}\n\nSolution: {s}\n\n"
              "Question: Is the final answer of this solution correct? Answer Yes or No."),
    "ref": ("Below is a math problem, its reference answer, and a proposed solution.\n\nProblem: {q}\n\n"
            "Reference answer: {gold}\n\nSolution: {s}\n\n"
            "Question: Is the final answer of this solution correct? Answer Yes or No."),
}
READOUTS = ["G", "G_cal", "P_align", "P_full", "P_match"]


class Judge:
    def __init__(self, path, device, readouts_json=None, dtype=torch.bfloat16):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        self.llama = "llama" in path.lower()
        if self.tok.pad_token is None:
            self.tok.pad_token = "<|finetune_right_pad_id|>" if self.llama else self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval().requires_grad_(False)
        self.dev = device
        y = self.tok.encode("Yes", add_special_tokens=False)
        n = self.tok.encode("No", add_special_tokens=False)
        assert len(y) == 1 and len(n) == 1
        self.YES, self.NO = y[0], n[0]
        self.R = json.load(open(readouts_json)) if readouts_json else None
        self.LAYER = (self.R or {}).get("layer", 18)
        self.kw = {"date_string": "26 Jul 2024"} if self.llama else {}

    def text(self, q, s, prompt="noref", gold=""):
        return self.tok.apply_chat_template([{"role": "user", "content": USER[prompt].format(q=q, gold=gold, s=s.strip()[:1500])}],
                                            tokenize=False, add_generation_prompt=True, **self.kw)

    def texts(self, questions, solutions):
        return [self.text(q, s) for q, s in zip(questions, solutions)]

    @torch.no_grad()
    def forward(self, questions, solutions, bs=16, hidden=True):
        """returns G (n,), H (n, D) float32 at the readout layer, last position (H = None if hidden=False)"""
        T = self.texts(questions, solutions)
        order = np.argsort([-len(t) for t in T])
        G = np.zeros(len(T), np.float32)
        H = None
        for b in range(0, len(T), bs):
            idx = order[b:b + bs]
            enc = self.tok([T[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
            out = self.model(**enc, output_hidden_states=hidden, logits_to_keep=1)
            lg = out.logits[:, -1, :].float()
            G[idx] = (lg[:, self.YES] - lg[:, self.NO]).cpu().numpy()
            if not hidden:
                del out
                continue
            h = out.hidden_states[self.LAYER][:, -1, :].float().cpu().numpy()
            if H is None:
                H = np.zeros((len(T), h.shape[1]), np.float32)
            G[idx] = (lg[:, self.YES] - lg[:, self.NO]).cpu().numpy()
            H[idx] = h
            del out
        return G, H

    @torch.no_grad()
    def feats(self, rows, prompt="noref", bs=16, store_h=True):
        """rows: dicts with question, response, gold. Returns G (n,), HL (n, NL, D) fp16 last position,
        HM (n, NL, D) fp16 mean over the answer tokens (tokens whose character span lies inside the response)."""
        T, spans = [], []
        for r in rows:
            s = r["response"].strip()[:1500]
            t = self.text(r["question"], s, prompt, str(r.get("gold", "")))
            a = t.find("Solution: " + s) + len("Solution: ") if s else -1
            T.append(t)
            spans.append((a, a + len(s)) if s and a >= len("Solution: ") else None)
        order = np.argsort([-len(t) for t in T])
        n = len(T)
        NL = self.model.config.num_hidden_layers + 1
        D = self.model.config.hidden_size
        G = np.zeros(n, np.float32)
        HL = np.zeros((n, NL, D), np.float16) if store_h else None
        HM = np.zeros((n, NL, D), np.float16) if store_h else None
        nans = np.zeros(n, np.int32)
        for b in range(0, n, bs):
            idx = order[b:b + bs]
            enc = self.tok([T[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False,
                           return_offsets_mapping=True)
            off = enc.pop("offset_mapping")
            enc = enc.to(self.dev)
            out = self.model(**enc, output_hidden_states=True, logits_to_keep=1)
            lg = out.logits[:, -1, :].float()
            G[idx] = (lg[:, self.YES] - lg[:, self.NO]).cpu().numpy()
            if store_h:
                hs = torch.stack(out.hidden_states, 1)  # (B, NL, T, D)
                HL[idx] = hs[:, :, -1, :].float().cpu().numpy().astype(np.float16)
                am = enc["attention_mask"].bool().cpu()
                for j, i in enumerate(idx):
                    sp = spans[i]
                    if sp is None:
                        continue
                    o = off[j]
                    m = am[j] & (o[:, 0] >= sp[0]) & (o[:, 1] <= sp[1]) & (o[:, 1] > o[:, 0])
                    nans[i] = int(m.sum())
                    if m.any():
                        HM[i] = hs[j][:, m.to(self.dev), :].float().mean(1).cpu().numpy().astype(np.float16)
            del out
        return G, HL, HM, nans

    def readouts(self, G, H):
        R = self.R
        sig = lambda z: 1.0 / (1.0 + np.exp(-z))
        out = {"G": G.astype(np.float64)}
        out["G_cal"] = sig(R["G_cal"]["a"] * G + R["G_cal"]["b"])
        for k in ("P_align", "P_full", "P_match"):
            if k not in R:
                continue
            p = R[k]
            zs = (H - np.array(p["mu"])) / np.array(p["sd"])
            out[k] = sig(zs @ np.array(p["w"]) + p["b"])
            out[f"maxz_{k}"] = np.abs(zs).max(1)
        return {k: [float(x) for x in v] for k, v in out.items()}

    def score(self, questions, solutions, bs=16):
        if len(questions) == 0:
            return {k: [] for k in READOUTS if k in (self.R or {}) or k == "G"}
        hid = any(k in (self.R or {}) for k in ("P_align", "P_full", "P_match"))
        G, H = self.forward(questions, solutions, bs=bs, hidden=hid)
        return self.readouts(G, H)
