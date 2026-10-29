"""Flat parameter layout: one (G, P) fp32 tensor per set of models, views per reference state_dict name.

Aggregation, communication bytes, per-client banks and the optimizer all work on rows of this tensor; core/grouped.py
reads the views. Order = DAGSNet.named_parameters() / BN buffers in state_dict order.
"""
import math

import torch

from core.dagsnet import DAGSNet


class Layout:
    def __init__(self, cfg, make=DAGSNet):
        """make(cfg) -> nn.Module whose parameters/BN buffers this layout flattens (DAGSNet unless a method brings its own)."""
        self.cfg, self.make = cfg, make
        m = make(cfg)
        self.p = [(k, tuple(v.shape)) for k, v in m.named_parameters()]
        self.b = [(k, tuple(v.shape)) for k, v in m.state_dict().items() if k.endswith(("running_mean", "running_var"))]
        self.P = sum(math.prod(s) for _, s in self.p)
        self.B = sum(math.prod(s) for _, s in self.b)
        self.head_keys = [k for k, _ in self.p if k.startswith("head.")]

    @staticmethod
    def _views(flat, spec):
        out, o, G = {}, 0, flat.shape[0]
        for k, s in spec:
            n = math.prod(s)
            out[k] = flat[:, o:o + n].view(G, *s)
            o += n
        return out

    def params(self, flat):
        return self._views(flat, self.p)

    def bufs(self, flat):
        return self._views(flat, self.b)

    def flat_bufs(self, d):
        return torch.cat([d[k].reshape(d[k].shape[0], -1) for k, _ in self.b], dim=1)

    def mask(self, keys_prefix):
        """(P,) bool: coordinates whose name starts with any prefix (e.g. head-only / extractor-only sharing)."""
        m, o = torch.zeros(self.P, dtype=torch.bool), 0
        for k, s in self.p:
            n = math.prod(s)
            m[o:o + n] = k.startswith(tuple(keys_prefix))
            o += n
        return m

    def init(self, seed, g=1, device="cpu"):
        torch.manual_seed(seed)
        sd = self.make(self.cfg).state_dict()
        p = torch.cat([sd[k].reshape(-1).float() for k, _ in self.p])
        b = torch.cat([sd[k].reshape(-1).float() for k, _ in self.b]) if self.b else torch.zeros(0)
        return p.to(device).repeat(g, 1), b.to(device).repeat(g, 1)

    def state_dict(self, p_row, b_row):
        """One model (rows (P,), (B,)) as a reference DAGSNet state_dict."""
        sd = {k: v[0].detach().cpu().clone() for k, v in self.params(p_row[None]).items()}
        sd.update({k: v[0].detach().cpu().clone() for k, v in self.bufs(b_row[None]).items()})
        return sd
