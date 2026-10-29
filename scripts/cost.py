"""Computation and communication cost per method, measured (not hand-derived) — docs/research/cost.md.

    python scripts/cost.py --width 1 --out docs/research/cost.json

Compute: one method round (round 2, so prototype / fusion terms are active) on a toy CPU environment (K clients, random
data) at the protocol model width, under torch.utils.flop_counter.FlopCounterMode (matmul/conv/bmm FLOPs, the usual
convention). Client FLOPs per local training row per round = round FLOPs minus the server step's, divided by the rows
trained; the server step (aggregation / prototype training) is measured separately and reported per round. Communication:
payload_bytes() of the method (per client per round, up / down).
"""
import argparse
import importlib
import json
import os
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]
from core import local  # noqa: E402
from core.dagsnet import DAGSNet, width  # noqa: E402
from core.data import Split  # noqa: E402
from core.flat import Layout  # noqa: E402
from test_methods import ToyEnv  # noqa: E402

METHODS = ["fedavg", "fedprox", "scaffold", "afpha", "fd_ids", "lwfednids", "pfedes", "fedproto", "fedtgp", "tinyproto",
           "nilm", "perfedskd", "mobihet", "apfed", "v2x_htfl"]
SERVER = {"fedtgp": "_server", "v2x_htfl": "_server"}


def env_at(w, K=8, n=64):
    """K clients of exactly n train rows (a multiple of the batch) in full groups: no padded rows or clients, whose FLOPs
    the grouped engine would otherwise count."""
    env = ToyEnv(K=K)
    env.cfg = width(w)
    env.lay = Layout(env.cfg)
    g = torch.Generator().manual_seed(0)
    off = torch.arange(0, K * n + 1, n)
    y = torch.randint(0, 16, (K * n,), generator=g).to(torch.int8)
    x = torch.randn(K * n, 77, generator=g) + 0.3 * y[:, None].float()
    env.train = Split(x, y, off, torch.arange(K))
    env.n = torch.full((K,), float(n))
    env.groups = local.group_schedule(env.train.sizes, list(range(K)), env.hp["batch"], env.hp["group_rows"])
    assert all(len(gr) * env.hp["batch"] == env.hp["group_rows"] for gr in env.groups)
    return env


def measure(name, w):
    env = env_at(w)
    M = importlib.import_module(f"methods.{name}").Method(env)
    M.round(1, 1e-3)
    server = {"flops": 0, "args": None}
    if name in SERVER:                         # skip the server step inside the round; count it alone afterwards
        f0 = getattr(M, SERVER[name])          # (MultiheadAttention's backward breaks under a nested FlopCounterMode)
        setattr(M, SERVER[name], lambda *a, **k: server.update(args=(a, k)))
    with FlopCounterMode(display=False) as fc:
        M.round(2, 1e-3)
    if server["args"] is not None:
        with FlopCounterMode(display=False) as fs:
            f0(*server["args"][0], **server["args"][1])
        server["flops"] = fs.get_total_flops()
    del server["args"]
    rows = int(env.n.sum())
    client = fc.get_total_flops() / rows
    return {"client_flops_per_row": client, "server_flops_per_round_toy": server["flops"], "n_params": env.lay.P,
            "payload_bytes": M.payload_bytes()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(4)
    m = DAGSNet(width(a.width)).eval()
    with FlopCounterMode(display=False) as f:
        m(torch.zeros(1, 77))
    fwd = f.get_total_flops()
    res = {"width": a.width, "dagsnet_forward_flops_per_row": fwd, "methods": {}}
    for name in a.methods:
        r = measure(name, a.width)
        r["x_forward"] = r["client_flops_per_row"] / fwd
        res["methods"][name] = r
        pb = r["payload_bytes"]
        print(f"{name:10s} client {r['client_flops_per_row'] / 1e6:8.1f} MFLOP/row/round ({r['x_forward']:5.2f} x fwd)  "
              f"up {pb.get('up', 0) / 1e6:7.3f} MB down {pb.get('down', 0) / 1e6:7.3f} MB  P {r['n_params']}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
