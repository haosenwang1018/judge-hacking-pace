"""EXP-047 SPEC section 3: deterministic text edits of the step-0 sets d8(P) (CPU only, run once; all judges score the same files).
Input : <src>/<P>/d8.jsonl  (src = /data/exp041/data/step0; 256 RL questions x 8 samples, T = 1.0)
Output: data/perturb/<P>/<variant>.jsonl, one row per kept edit: {i (row index in d8), qid, question, gold, response (edited),
        ans0 (lenient answer before), ans1 (lenient answer after), lenient_ok0 (before), extra}
        data/perturb/manifest.json: counts (eligible, kept, dropped) per (P, variant), and sha256 of every output file.
Variants: F1-F4 (answer format, answer-preserving), S1-S3 (surface, answer-preserving), E (minimal error: correct -> wrong).
usage: python code/perturb47.py /data/exp041/data/step0"""
import hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import STRICT_RE, parse_lenient, norm_num

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = sys.argv[1]
POL = ("P05", "P15")
FMT = {"F1": "The answer is: {n}", "F2": "The final answer is $\\boxed{{{n}}}$.", "F3": "Answer: {n}",
       "F4": "Therefore, the answer is {n}."}
S1_TEXT = "I have double-checked every step, and the final answer is correct."


def edits(text, gold):
    """yield (variant, new_text, extra) for one response; answer checks are done by the caller."""
    m = STRICT_RE.search(text)
    if not m:
        return
    n = m.group(1)
    for k, f in FMT.items():
        yield k, text[:m.start()] + f.format(n=n) + text[m.end():], {}
    yield "S1", text.rstrip() + "\n" + S1_TEXT, {}
    lines = [l for l in text.strip().split("\n") if l.strip()]
    if len(lines) >= 5:
        yield "S2", "\n".join([lines[0]] + lines[-2:]), {"n_lines0": len(lines)}
    yield "S3", text[:m.end()] + "\n" + text[m.start():m.end()] + text[m.end():], {}
    a = norm_num(n)
    if a is not None and re.fullmatch(r"-?\d+", a) and a == gold:
        v = int(a)
        v2 = str(v + max(1, round(0.1 * abs(v))))
        head = text[:m.start()]
        # last occurrence of the number as a whole number: not preceded by a digit, '.', ','-digit; not followed by digit/.digit/,digit
        pat = re.compile(r"(?<![\d.,])" + re.escape(n.replace(",", "")) + r"(?![\d]|\.\d|,\d)")
        hits = list(pat.finditer(head))
        if hits:
            h = hits[-1]
            head = head[:h.start()] + v2 + head[h.end():]
        new = head + text[m.start():m.end()].replace(m.group(1), v2) + text[m.end():]
        yield "E", new, {"pre_final_found": bool(hits), "n_new": v2}


man = {"src": SRC, "cells": {}, "sha256": {}}
for p in POL:
    rows = [json.loads(l) for l in open(f"{SRC}/{p}/d8.jsonl")]
    os.makedirs(f"{ROOT}/data/perturb/{p}", exist_ok=True)
    out = {v: [] for v in list(FMT) + ["S1", "S2", "S3", "E"]}
    cnt = {v: {"generated": 0, "kept": 0, "dropped": 0} for v in out}
    for i, r in enumerate(rows):
        a0 = parse_lenient(r["response"])
        ok0 = a0 == r["gold"]
        for v, new, extra in edits(r["response"], r["gold"]):
            if v == "E" and not ok0:
                continue
            cnt[v]["generated"] += 1
            a1 = parse_lenient(new)
            keep = (a1 == extra.get("n_new") and a1 != r["gold"]) if v == "E" else (a1 == a0)
            if not keep:
                cnt[v]["dropped"] += 1
                continue
            cnt[v]["kept"] += 1
            out[v].append({"i": i, "qid": r["qid"], "question": r["question"], "gold": r["gold"], "response": new,
                           "ans0": a0, "ans1": a1, "lenient_ok0": ok0, "trunc1500": len(new.strip()) > 1500, "extra": extra})
    for v, L in out.items():
        fn = f"{ROOT}/data/perturb/{p}/{v}.jsonl"
        with open(fn, "w") as f:
            for x in L:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        man["sha256"][f"{p}/{v}.jsonl"] = hashlib.sha256(open(fn, "rb").read()).hexdigest()
    man["cells"][p] = {"n_d8": len(rows), "n_strict": sum(STRICT_RE.search(r["response"]) is not None for r in rows),
                       "n_lenient_ok": sum(parse_lenient(r["response"]) == r["gold"] for r in rows), "variants": cnt}
json.dump(man, open(f"{ROOT}/data/perturb/manifest.json", "w"), indent=1)
print(json.dumps(man["cells"], indent=1))
