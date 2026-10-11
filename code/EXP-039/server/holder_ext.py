"""EXP-039: extension holder for one GPU (CUDA_VISIBLE_DEVICES set by caller). Holds 1 GiB at once; when the driver's
holder on the same GPU exits, grows to 36 GiB and registers itself in runs/holder<g>.pid so the driver treats the GPU as
ours. Exits at the deadline (unix time, argv[2]) or when killed. Records its PID in runs/pids.txt. usage: holder_ext.py <g> <deadline>"""
import os, sys, time, torch
g, deadline = sys.argv[1], float(sys.argv[2])
os.chdir("/data/exp039")
open("runs/pids.txt", "a").write(f"holder-ext-gpu{g} {os.getpid()}\n")
small = torch.empty(2**30, dtype=torch.uint8, device="cuda:0")
big = None
t0 = time.strftime("%Y-%m-%dT%H:%M:%S")
def released():   # the driver released the GPU on purpose (it wants it for our own job): then do not take over
    try:
        tail = open("runs/driver.log").read().splitlines()[-6:]
    except OSError:
        return False
    return any(f"holder on gpu {g} released" in l and l[:19] >= t0 for l in tail)
while time.time() < deadline:
    if big is None:
        if released():
            break
        try:
            big = torch.empty(35 * 2**30, dtype=torch.uint8, device="cuda:0")
            open(f"runs/holder{g}.pid", "w").write(str(os.getpid()))
            with open("runs/driver.log", "a") as f:
                f.write(time.strftime("%Y-%m-%dT%H:%M:%S+0800") + f" holder-ext on gpu {g} took over pid={os.getpid()}\n")
        except torch.OutOfMemoryError:
            time.sleep(0.2); continue
    time.sleep(2)
