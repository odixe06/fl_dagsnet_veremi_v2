"""Grouped forward == reference DAGSNet, per client, in train (batch stats, masked rows) and eval mode; gradients too."""
import torch

from core import grouped
from core.dagsnet import DAGSNet, FULL, width


def _models(cfg, g):
    ms = []
    for i in range(g):
        torch.manual_seed(100 + i)
        m = DAGSNet(cfg).double()
        for mod in m.modules():                         # non-trivial BN running stats
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.running_mean.uniform_(-0.5, 0.5); mod.running_var.uniform_(0.5, 2.0)
        ms.append(m)
    sds = [m.state_dict() for m in ms]
    P = {k: torch.stack([sd[k] for sd in sds]).clone().requires_grad_(True) for k, _ in ms[0].named_parameters()}
    B = {k: torch.stack([sd[k] for sd in sds]).clone() for k in sds[0] if k.endswith(("running_mean", "running_var"))}
    return ms, P, B


def _check(cfg):
    cfg = {**cfg, "dropout": 0.0}
    g, b = 3, 9
    ms, P, B = _models(cfg, g)
    x = torch.randn(g, b, 77, dtype=torch.float64)
    n = [9, 5, 2]                                       # real rows per client; the rest is padding
    mask = torch.arange(b)[None] < torch.tensor(n)[:, None]
    for train in (False, True):
        out, newB = grouped.forward(P, B, x, cfg, mask=mask, train=train)
        for i, m in enumerate(ms):
            m.train(train)
            ref = m(x[i, :n[i]])
            assert torch.allclose(out[i, :n[i]].double(), ref, atol=1e-8, rtol=1e-6), (train, i)
            if train:
                for k, v in m.state_dict().items():
                    if k.endswith(("running_mean", "running_var")):
                        assert torch.allclose(newB[k][i], v, atol=1e-10), k
                m.zero_grad()
                ref.square().sum().backward()
    # gradients: client i's loss only moves client i's slice
    for v in P.values():
        v.grad = None
    out, _ = grouped.forward(P, B, x, cfg, mask=mask, train=True)
    sum(out[i, :n[i]].square().sum() for i in range(g)).backward()
    for i, m in enumerate(ms):
        for k, p in m.named_parameters():
            assert torch.allclose(P[k].grad[i], p.grad, atol=1e-8, rtol=1e-6), k


def test_full():
    _check(FULL)


def test_compact():
    _check(width(0.375))
