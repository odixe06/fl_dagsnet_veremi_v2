#!/usr/bin/env python
"""Run one heavy local command with NO fixed RSS cap (owner, 2026-10-02): stop it only if the whole
WSL machine drops below FLOOR MiB available (a WSL freeze guard), print the peak tree RSS. If it stops,
the job genuinely needs more RAM: tell the owner (they close VS Code / raise the WSL memory limit).

    python scripts/run_local_guarded.py python scripts/<job>.py [args]      # env nckh
"""
import os, signal, subprocess, sys, time
FLOOR = 700
def avail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable:"):
            return int(l.split()[1]) // 1024
def rss(pid):
    tot = 0
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                st = open(f"/proc/{p}/stat").read().split()
                if int(st[4]) == pid:  # process group
                    tot += int(open(f"/proc/{p}/statm").read().split()[1]) * 4096 // 2**20
            except Exception:
                pass
    return tot
child = subprocess.Popen(sys.argv[1:], start_new_session=True)
peak = 0
while child.poll() is None:
    peak = max(peak, rss(child.pid)); a = avail()
    if a < FLOOR:
        print(f"LOW MEMORY STOP: available {a} MiB, tree RSS {rss(child.pid)} MiB -> need more RAM", flush=True)
        os.killpg(child.pid, signal.SIGTERM); child.wait(); break
    time.sleep(0.5)
print(f"Peak tree RSS: {peak} MiB; exit {child.returncode}", flush=True)
sys.exit(child.returncode or 0)
