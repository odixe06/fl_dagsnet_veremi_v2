"""Run one FL method under the shared protocol (docs/research/protocol.md).

    torchrun --nproc_per_node=2 scripts/run_fl.py --method fedavg --out runs/fedavg            # Kaggle 2xT4
    python scripts/run_fl.py --method fedavg --subset 400 --rounds 3 --out runs/smoke             # local probe

Per round: train every client one local epoch, aggregate, evaluate each model view, append rounds.jsonl, save the
confusion matrices (cms_rXXX.npz). Resume state at a budget stop (and every --save-every rounds if set): each rank writes
its own (state.pt on rank 0 with the round and RNG states, state_rank<r>.pt on rank r > 0: per-client banks differ per
rank), atomically. Per-client banks are GBs at W = 1, and /kaggle/working holds ~20 GB: a resumed run deletes the copied-in
state once loaded, so the disk never holds two copies.
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core import fl, grouped, local, metrics  # noqa: E402
from core.dagsnet import width  # noqa: E402

PROTOCOL = {"rounds": 50, "batch": 64, "lr_max": 1e-3, "lr_min": 1e-5, "weight_decay": 1e-4, "clip": 1.0, "seed": 42,
            "group_rows": 8192, "amp": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=float, default=float(os.environ.get("DAGS_WIDTH", 1.0)))
    ap.add_argument("--rounds", type=int, default=PROTOCOL["rounds"])
    ap.add_argument("--subset", type=int, default=0, help="local probes: N random clients (seed 7)")
    ap.add_argument("--cache", default=os.environ.get("V4_CACHE", "cache"))
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--wandb", default="", help="W&B project; empty = off")
    ap.add_argument("--max-hours", type=float, default=0, help="stop before a round would cross this budget")
    ap.add_argument("--set", nargs="*", default=[], help="method hyperparameters k=v (paper values are the defaults)")
    ap.add_argument("--save-every", type=int, default=0, help="also save resume state every N rounds (0: only at a budget stop)")
    a = ap.parse_args()
    T0 = time.monotonic()

    hp = dict(PROTOCOL, rounds=a.rounds)
    mhp = {k: json.loads(v) for k, v in (s.split("=", 1) for s in a.set)}
    hp.update(mhp)
    cfg = width(a.width)
    clients = None
    if a.subset:
        clients = np.sort(np.random.default_rng(7).choice(5343, a.subset, replace=False)).tolist()
    env = fl.Env(cfg, hp, clients=clients, cache=a.cache or None)
    M = importlib.import_module(f"methods.{a.method}").Method(env)
    cfg = env.cfg                                                     # a method may change the model (LwFedNIDS prunes it)
    backend = "eager"
    if a.compile:
        grouped.compile_all()
        ok, rep = local.compile_gate(env.lay, env.train, env.cfg, hp, torch.Generator(device=env.dev).manual_seed(0))
        print(f"compile gate rank{env.rank}: {'PASS' if ok else 'FAIL -> eager'} {rep}", flush=True)
        if ok:
            backend = "compiled"
        else:
            grouped.TRAIN = grouped.EVAL = grouped.forward
            from core import optim
            optim.UPDATE = optim._update
    main_rank = env.rank == 0
    os.makedirs(a.out, exist_ok=True)
    state_p = os.path.join(a.out, "state.pt")
    my_state = state_p if env.rank == 0 else os.path.join(a.out, f"state_rank{env.rank}.pt")
    start = 1
    if os.path.exists(state_p):
        s = torch.load(state_p, map_location="cpu", weights_only=True)
        assert s["cfg"] == cfg and s["method"] == a.method and s["hp"] == hp, "resume with a different config"
        if env.rank:
            mine = torch.load(my_state, map_location="cpu", weights_only=True)
            assert mine["round"] == s["round"], f"rank {env.rank} state is round {mine['round']}, rank 0 {s['round']}"
            s["method_state"] = mine["method_state"]
        M.load_state(s["method_state"])
        env.gen.set_state(s["gen"][env.rank])
        start = s["round"] + 1
        del s
        if env.world > 1:
            torch.distributed.barrier()                              # every rank has loaded before any file goes
        os.remove(my_state)
    run = None
    if a.wandb and main_rank:
        import wandb
        run = wandb.init(project=a.wandb, name=os.path.basename(a.out.rstrip("/")), id=os.path.basename(a.out.rstrip("/")),
                         resume="allow", config={"method": a.method, "cfg": cfg, "hp": hp, "n_params": env.lay.P,
                                                 "clients": len(env.all_ids), "world": env.world, "backend": backend})
    if main_rank:
        json.dump({"method": a.method, "cfg": cfg, "hp": hp, "n_params": env.lay.P, "n_buffers": env.lay.B,
                   "clients": len(env.all_ids), "payload_bytes": M.payload_bytes(), "backend": backend},
                  open(os.path.join(a.out, "config.json"), "w"), indent=1)

    def save(r):
        gens = [None] * env.world
        g = env.gen.get_state()
        if env.world > 1:
            torch.distributed.all_gather_object(gens, g)
        else:
            gens = [g]
        body = {"round": r, "method_state": M.state()}
        if main_rank:
            body.update(cfg=cfg, hp=hp, method=a.method, gen=gens)
        tmp = my_state + ".tmp"
        torch.save(body, tmp)
        os.replace(tmp, my_state)
        if env.world > 1:
            torch.distributed.barrier()

    worst, saved, r_done, stopped = 0.0, start - 1, start - 1, False
    for r in range(start, hp["rounds"] + 1):
        stop = torch.tensor(float(bool(a.max_hours) and time.monotonic() - T0 + 1.15 * worst > a.max_hours * 3600),
                            device=env.dev)
        if env.world > 1:
            torch.distributed.all_reduce(stop, op=torch.distributed.ReduceOp.MAX)   # every rank stops at the same round
        if stop.item():
            print(f"budget stop before round {r}", flush=True)
            stopped = True
            break
        t = time.monotonic()
        lr = fl.cosine_lr(r, hp["rounds"], hp["lr_max"], hp["lr_min"])
        info = M.round(r, lr)
        if info.get("skipped_frac", 0) > 0.01:
            print(f"WARNING r{r}: {info['skipped_frac']:.1%} of client steps skipped (non-finite gradients)", flush=True)
        t_train = time.monotonic() - t
        views = M.evaluate()
        t_eval = time.monotonic() - t - t_train
        row = {"round": r, "lr": lr, "t_train": t_train, "t_eval": t_eval, **info}
        for v, cms in views.items():
            s = metrics.summary(cms)
            row.update({f"{v}/{k}": s[k] for k in metrics.TEN + ["client_f1_macro_mean", "client_f1_macro_p10",
                                                                  "client_f1_macro_median", "client_f1_macro_min"]})
            row[f"{v}/f1_per_class"] = s["f1_per_class"]
        if main_rank:
            np.savez_compressed(os.path.join(a.out, f"cms_r{r:03d}.npz"), ids=env.all_ids, **views)
            with open(os.path.join(a.out, "rounds.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")
            heads = {k: round(v, 4) for k, v in row.items() if k.endswith("/f1_macro") or k.endswith("client_f1_macro_mean")}
            print(f"r{r:02d} lr={lr:.2e} train {t_train:.0f}s eval {t_eval:.0f}s loss {info.get('train_loss', 0):.4f} {heads}",
                  flush=True)
            if run:
                run.log({k: v for k, v in row.items() if not isinstance(v, list)}, step=r)
        r_done = r
        if a.save_every and r % a.save_every == 0:
            save(r); saved = r
        worst = max(worst, time.monotonic() - t)
    if stopped and r_done > saved:
        save(r_done)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
