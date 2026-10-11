"""PAPER-T11 revision 261011 (no GPU, CPU parsing only). Run on the A100 hosts.
usage: python3 extract_remote.py taus.json out.json run_dir ...
Per run: for every saved greedy eval step, counts of strict/lenient/generous correct, format, and judge acceptance
(G_cal >= tau_len) on lenient-wrong and lenient-correct answers. Also: parser-audit candidates = lenient-wrong
training rollouts with G_cal >= tau_len in steps 101-150 (all of them, compact tail text)."""
import json, re, sys, os, glob
STRICT_RE = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
BOXED_RE = re.compile(r"\\boxed\{\s*\$?\s*(-?[\d,]*\.?\d+)")
def norm_num(s):
    if s is None: return None
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try: v = float(s)
    except ValueError: return None
    return str(int(v)) if v == int(v) else str(v)
def parse_strict(t):
    m = STRICT_RE.search(t); return norm_num(m.group(1)) if m else None
def parse_lenient(t):
    a = parse_strict(t)
    if a is not None: return a
    head = re.split(r"\n\s*(Problem|Question)\s*:", t)[0]
    nums = NUM_RE.findall(head); return norm_num(nums[-1]) if nums else None
def generous(t, gold):
    a = parse_strict(t)
    if a is not None: return a == gold
    m = BOXED_RE.search(t)
    if m: return norm_num(m.group(1)) == gold
    head = re.split(r"\n\s*(Problem|Question)\s*:", t)[0].strip()
    parts = [x for x in re.split(r"(?<=[.!?])\s+|\n", head) if x.strip()]
    return bool(parts) and gold in [norm_num(x) for x in NUM_RE.findall(parts[-1])]
taus = json.load(open(sys.argv[1])); out = {}
for d in sys.argv[3:]:
    rid = os.path.basename(d.rstrip("/"))
    if rid not in taus: continue
    tau = taus[rid]["tau"]; ev = {}
    for p in sorted(glob.glob(f"{d}/eval_outputs_step*.jsonl")):
        s = int(re.search(r"step(\d+)", p).group(1)); c = dict(n=0, strict=0, len=0, gen=0, fmt=0, nw=0, fw=0, nc=0, fc=0, boxed=0)
        for l in open(p):
            r = json.loads(l); t = r["response"]; g = r["gold"]
            ok = parse_lenient(t) == g; c["n"] += 1; c["len"] += ok; c["strict"] += parse_strict(t) == g
            c["gen"] += generous(t, g); c["fmt"] += parse_strict(t) is not None; c["boxed"] += bool(BOXED_RE.search(t))
            acc = r["G_cal"] + 1e-6 >= tau
            if ok: c["nc"] += 1; c["fc"] += acc
            else: c["nw"] += 1; c["fw"] += acc
        ev[s] = c
    aud = []
    for l in open(f"{d}/rollouts.jsonl"):
        r = json.loads(l)
        if 101 <= r["step"] <= 150 and r["G_cal"] + 1e-6 >= tau:
            t = r["completion"]; pl = parse_lenient(t)
            if pl != r["gold"]:
                head = re.split(r"\n\s*(Problem|Question)\s*:", t)[0]
                aud.append(dict(run=rid, step=r["step"], i=r.get("i"), gold=r["gold"], parsed=pl, strict=parse_strict(t),
                                gen=generous(t, r["gold"]), q=r["question"][-160:], tail=head[-420:], cut=len(head) < len(t)))
    out[rid] = {"cell": taus[rid]["cell"], "evals": ev, "audit_pool": aud}
    print(rid, len(ev), len(aud), flush=True)
json.dump(out, open(sys.argv[2], "w"))
