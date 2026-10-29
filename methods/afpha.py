"""AFPHA — Adaptive Federated Proximal Hierarchical Averaging, Khan et al., Sci. Rep. 2025 (papers/afpha_edl_cmso.pdf).

The paper names the parts (FedAvg + FedProx + hierarchical FL over device clusters, "adaptive") without equations; this is
the earlier rebuild's specification "Proposal A" (owner-approved 06/09/2026) with the owner's v4 cluster choice (29/10):
  clusters   = scenario × run of the vehicle (8 natural groups — same simulation area, RSU-like), fixed
  local      = CE + (μ_i,t / 2)‖w − w_t‖² on learnable parameters (FedProx), gradient + μ_i,t (w − w_t)
  aggregate  = w_g = Σ_{i∈g} (n_i/N_g) w_i, then w = Σ_g (N_g/N) w_g (= sample-weighted FedAvg algebraically); BN buffers alike
  adaptive μ = μ_i,1 = 0.01; after round t, d_i = ‖w_i,t+1 − w_t+1‖₂, d̄ = Σ (n_i/N) d_i, μ_i,t+1 = 0.01 (1 + d_i/(d_i + d̄ + 1e-12))
§4.8 DWT/ViT/GAT, CMSO and the AES/HE transport are out of scope (bare DAGSNet). A heuristic completion, not the
authors' code (none is published).
"""
import os

import numpy as np
import pandas as pd
import torch

from core import data as D, local
from methods import fedavg


class Method(fedavg.Method):
    name = "afpha"

    def __init__(self, env):
        super().__init__(env)
        c = pd.read_parquet(os.path.join(D.find_root(), "clients.parquet"))
        key = {cid: i for i, cid in enumerate(sorted(set(zip(c.scenario, c.run))))}
        cl = dict(zip(c.client_id, [key[k] for k in zip(c.scenario, c.run)]))
        self.cluster = torch.as_tensor(np.array([cl[i] for i in env.ids]), device=env.dev)
        self.n_clusters = len(key)
        self.mu = torch.full((env.K,), 0.01, device=env.dev)
        P = env.lay.P
        self.trained = torch.empty(env.K, P, device=env.dev,
                                   dtype=torch.float32 if env.K * P * 4 < 3 * 2**30 else torch.float16)

    def round(self, r, lr):
        env = self.env
        Ncl = self.n_clusters
        acc_p = torch.zeros(Ncl, env.lay.P, device=env.dev, dtype=torch.float64)
        acc_b = torch.zeros(Ncl, env.lay.B, device=env.dev, dtype=torch.float64)
        Ng = torch.zeros(Ncl, device=env.dev, dtype=torch.float64)
        loss = torch.zeros(env.K, device=env.dev)
        mu_g = {}

        def grad_fn(ctx, grad):
            if "v" not in mu_g or mu_g["v"].shape[0] != ctx.Gfix:
                mu_g["v"] = local.pad_rows(mu_g["raw"], ctx.Gfix)[:, None]
            grad += mu_g["v"] * (ctx.flat - self.p[None])

        for idx in env.groups_idx():
            flat, bufs = self.p.repeat(len(idx), 1), self.b.repeat(len(idx), 1)
            mu_g.clear(); mu_g["raw"] = self.mu[idx]
            st = local.local_epoch(env.lay, flat, bufs, env.train, idx, env.cfg, env.hp, lr, env.gen, grad_fn=grad_fn)
            self.trained[idx] = flat.to(self.trained.dtype)
            w = env.n[idx].double()
            g = self.cluster[idx]
            acc_p.index_add_(0, g, w[:, None] * flat.double())        # Σ_{i∈g} n_i w_i  (cluster level)
            acc_b.index_add_(0, g, w[:, None] * bufs.double())
            Ng.index_add_(0, g, w)
            loss[idx] = st["loss"]
        acc_p, acc_b, Ng = env.allreduce(acc_p), env.allreduce(acc_b), env.allreduce(Ng)
        live = Ng > 0
        wg_p, wg_b = acc_p[live] / Ng[live, None], acc_b[live] / Ng[live, None]   # cluster models w_g
        N = Ng.sum()
        self.p = ((Ng[live, None] / N) * wg_p).sum(0).float()                      # global over clusters
        self.b = ((Ng[live, None] / N) * wg_b).sum(0).float()
        d = (self.trained.float() - self.p[None]).norm(dim=1)
        dbar = env.allreduce((env.n * d).sum()) / env.allreduce(env.n.sum().clone())
        self.mu = 0.01 * (1 + d / (d + dbar + 1e-12))
        return {"train_loss": float(env.allreduce((loss * env.n).sum()) / N), "d_mean": float(dbar),
                "mu_mean": float(env.allreduce(self.mu.sum()) / env.allreduce(torch.tensor(float(env.K), device=env.dev)))}

    def state(self):
        return {**super().state(), "mu": self.mu.cpu()}

    def load_state(self, s):
        super().load_state(s)
        self.mu = s["mu"].to(self.env.dev)
