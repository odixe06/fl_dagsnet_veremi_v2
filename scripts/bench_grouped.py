"""Grouped local-epoch throughput on synthetic resident data: python scripts/bench_grouped.py --width 1 --batch 256 --rows 32768

Reports samples/s of one epoch over K clients (sizes drawn like v4: log-normal around the median 1.368 rows).
"""
import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from core import local
from core.dagsnet import width
from core.data import Split
from core.flat import Layout

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=float, default=1.0)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--rows", type=int, default=32768, help="max rows per grouped step (G * batch)")
ap.add_argument("--clients", type=int, default=128)
ap.add_argument("--compile", action="store_true")
a = ap.parse_args()

dev = "cuda"
rng = np.random.default_rng(0)
sizes = np.clip(rng.lognormal(np.log(1368), 1.3, a.clients), 76, 20000).astype(int)
off = np.concatenate([[0], np.cumsum(sizes)])
N = int(off[-1])
data = Split(torch.randn(N, 77, device=dev).clamp(-3.7, 3.7).half(), torch.randint(0, 16, (N,), device=dev, dtype=torch.int8),
             torch.from_numpy(off).to(dev), torch.arange(a.clients, device=dev))
cfg = width(a.width)
lay = Layout(cfg)
hp = {"batch": a.batch, "weight_decay": 1e-4, "clip": 1.0, "group_rows": a.rows}
gen = torch.Generator(device=dev).manual_seed(0)
if a.compile:
    from core import grouped
    grouped.compile_all()
groups = local.group_schedule(sizes, list(range(a.clients)), a.batch, a.rows)
print(f"P={lay.P} N={N} groups={len(groups)} G={len(groups[0])}")
for rep in range(2):
    torch.cuda.synchronize(); t = time.time(); steps = 0
    for g in groups:
        idx = torch.tensor(g, device=dev)
        flat, bufs = lay.init(42, len(g), dev)
        st = local.local_epoch(lay, flat, bufs, data, idx, cfg, hp, 1e-3, gen)
        steps += st["steps"]
    torch.cuda.synchronize(); dt = time.time() - t
    print(f"rep{rep}: {dt:.2f}s  {N/dt:,.0f} samples/s  grouped steps {steps}  {dt/steps*1e3:.1f} ms/step  "
          f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
