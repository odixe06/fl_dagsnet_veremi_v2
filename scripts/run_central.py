"""Centralized DAGSNet (reference nn.Module, DDP over the GPUs) — width gate and reported centralized runs.

    torchrun --nproc_per_node=2 scripts/run_central.py --width 0.375 --epochs 20 --gate --out /kaggle/working/c_w0375
        gate:  train on fold train, score fold val each epoch (docs/research/protocol.md, width gate)
        final: train on train + val, score test each epoch (headline = last epoch)

Data resident on each GPU (fp16); each epoch every rank takes a disjoint shard of one global permutation.
AdamW lr 1e-3, wd 1e-4, cosine to 1e-5 over the epochs, global batch 4096, AMP fp16 + GradScaler, grad clip 1.0.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core import data as D, metrics  # noqa: E402
from core.dagsnet import DAGSNet, width  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4096, help="global")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--cache", default="")
    ap.add_argument("--wandb", default="")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    clients = None
    if a.subset:
        clients = np.sort(np.random.default_rng(7).choice(D.N_CLIENTS, a.subset, replace=False)).tolist()
    root = D.find_root()
    cache = a.cache or None
    if a.gate:
        tr = D.load(root, "train", clients, cache, folds=("train",)).to(dev)
        ev = D.load(root, "train", clients, cache, folds=("val",)).to(dev)
    else:
        tr = D.load(root, "train", clients, cache).to(dev)
        ev = D.load(root, "test", clients, cache).to(dev)
    cfg = width(a.width)
    torch.manual_seed(a.seed)
    model = DAGSNet(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, fused=True)
    scaler = torch.amp.GradScaler()
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank]) if world > 1 else model
    fwd = torch.compile(ddp) if a.compile else ddp
    efwd = torch.compile(model, dynamic=True) if a.compile else model
    b = a.batch // world
    N = tr.X.shape[0]
    steps = N // a.batch
    os.makedirs(a.out, exist_ok=True)
    run = None
    if a.wandb and rank == 0:
        import wandb
        name = os.path.basename(a.out.rstrip("/"))
        run = wandb.init(project=a.wandb, name=name, id=name, resume="allow",
                         config={"cfg": cfg, "n_params": n_params, "gate": a.gate, "epochs": a.epochs, "batch": a.batch})
    if rank == 0:
        json.dump({"cfg": cfg, "n_params": n_params, "gate": a.gate, "epochs": a.epochs, "batch": a.batch,
                   "train_rows": N, "eval_rows": int(ev.X.shape[0])}, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    for ep in range(1, a.epochs + 1):
        t = time.monotonic()
        lr = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * (ep - 1) / max(1, a.epochs - 1)))
        for g in opt.param_groups:
            g["lr"] = lr
        perm = torch.randperm(N, device=dev, generator=gen)              # same on every rank (same seed, same N)
        model.train()
        tot = torch.zeros((), device=dev)
        for s in range(steps):
            idx = perm[s * a.batch + rank * b: s * a.batch + (rank + 1) * b]
            with torch.autocast("cuda", dtype=torch.float16):
                logits = fwd(tr.X[idx])
            loss = F.cross_entropy(logits.float(), tr.Y[idx].long())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.detach()
        t_train = time.monotonic() - t
        model.eval()
        cm = torch.zeros(256, dtype=torch.int64, device=dev)
        lo, hi = rank * ev.X.shape[0] // world, (rank + 1) * ev.X.shape[0] // world
        with torch.no_grad():
            for s in range(lo, hi, 16384):
                e = min(s + 16384, hi)
                with torch.autocast("cuda", dtype=torch.float16):
                    p = efwd(ev.X[s:e]).argmax(1)
                cm += torch.bincount(ev.Y[s:e].long() * 16 + p, minlength=256)
        if world > 1:
            dist.all_reduce(cm); dist.all_reduce(tot)
        m = metrics.ten(cm.view(16, 16).cpu().numpy())
        row = {"epoch": ep, "lr": lr, "train_loss": float(tot) / steps / world, "t_train": t_train,
               "t_eval": time.monotonic() - t - t_train, **{k: m[k] for k in metrics.TEN}, "f1_per_class": m["f1_per_class"]}
        if rank == 0:
            with open(os.path.join(a.out, "epochs.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")
            np.save(os.path.join(a.out, f"cm_e{ep:03d}.npy"), cm.view(16, 16).cpu().numpy())
            print(f"ep{ep:02d} lr={lr:.2e} loss {row['train_loss']:.4f} f1_macro {m['f1_macro']:.4f} acc {m['accuracy']:.4f} "
                  f"train {t_train:.0f}s", flush=True)
            if run:
                run.log({k: v for k, v in row.items() if not isinstance(v, list)}, step=ep)
    if rank == 0:
        torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    if run:
        run.finish()


if __name__ == "__main__":
    main()
