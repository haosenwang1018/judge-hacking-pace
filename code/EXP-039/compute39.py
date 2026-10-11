"""EXP-039 compute (SPEC section 10: wall clock x GPUs). Input: runs/driver.log (copied from gpu-server-2), runs/pids.txt.
RL runs and crashed starts from 'start'/'end' lines; holders from 'holder on gpu' lines (end = 'released' line, the
next 'start' using that GPU, or the holder's own time limit). Static gate, smokes and Llama rescoring are added by hand
from PROCESS.md.  usage: python compute39.py runs/driver.log"""
import re, sys
from datetime import datetime, timedelta
L = [l.strip() for l in open(sys.argv[1]) if l.strip()]
T = lambda l: datetime.strptime(l[:19], "%Y-%m-%dT%H:%M:%S")
runs, open_ = {}, {}
for l in L:
    m = re.search(r" (start|end) gpus=([\d,]+) (\S+)", l)
    if m:
        k, g, rid = m.groups()
        if k == "start":
            open_[rid] = (T(l), len(g.split(",")))
        elif rid in open_:
            t0, n = open_.pop(rid); rc = re.search(r"rc=(\d+)", l).group(1)
            runs.setdefault(rid, []).append(((T(l) - t0).total_seconds() / 3600, n, rc))
hold = []
FAILED = {"1056395"}   # 15:39 holder could not allocate (another agent already on GPU0) and exited at once (PROCESS.md)
for i, l in enumerate(L):
    m = re.search(r"holder on gpu (\d) pid=(\d+)$", l)
    if m:
        g, pid = m.groups(); t0 = T(l)
        lim = timedelta(minutes=30 if t0 < datetime(2026, 10, 9, 15, 41) else 60)
        t1 = t0 + lim
        for l2 in L[i + 1:]:
            if f"holder on gpu {g} released pid={pid}" in l2 or (re.search(r" start gpus=", l2) and g in re.search(r"gpus=([\d,]+)", l2).group(1)):
                t1 = min(t1, T(l2)); break
        hold.append((g, pid, 0.0 if pid in FAILED else (t1 - t0).total_seconds() / 3600))
tot_runs = sum(h * n for v in runs.values() for h, n, _ in v)
for rid, v in sorted(runs.items()):
    for h, n, rc in v:
        print(f"{rid[-14:]:>16} {h:6.2f} h x {n} = {h * n:5.2f}  rc={rc}")
for g, pid, h in hold:
    print(f"holder gpu{g} pid {pid}: {h:.2f} h")
th = sum(h for *_, h in hold)
print(f"RL runs and crashed starts: {tot_runs:.2f}  holders: {th:.2f}  total (without static/smoke/Llama): {tot_runs + th:.2f}")
