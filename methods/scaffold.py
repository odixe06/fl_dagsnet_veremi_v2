"""SCAFFOLD — Karimireddy et al., ICML 2020 (papers/scaffold.pdf), Algorithm 1, Option II, η_g = 1.

Local step on the corrected gradient g_i(y) − c_i + c (fed to the shared AdamW; the paper analyses SGD). After the epoch
c_i⁺ = c_i − c + (x − y_i) / (K_i η_l), K_i = optimizer steps the client actually applied, η_l = round lr.
Server (all N clients participate, |S| = N): x ← x + mean_i(y_i − x), c ← c + mean_i(c_i⁺ − c_i) — the paper's
unweighted means (FedAvg here weights by n_k; SCAFFOLD's Algorithm 1 does not). c, c_i start at 0. BN running
statistics are averaged with the same unweighted mean. Control variates of this rank's clients stay on its GPU.
"""
import torch

from core import local
from methods import fedavg


class Method(fedavg.Method):
    name = "scaffold"

    def __init__(self, env):
        super().__init__(env)
        self.c = torch.zeros_like(self.p)
        self.ci = torch.zeros(env.K, env.lay.P, device=env.dev)

    def payload_bytes(self):
        n = 4 * (2 * self.env.lay.P + self.env.lay.B)          # (x, c) down, (Δy, Δc) up
        return {"down": n, "up": n}

    def round(self, r, lr):
        env = self.env
        dc = torch.zeros_like(self.p)
        dx = torch.zeros_like(self.p)
        db = torch.zeros_like(self.b)
        loss = torch.zeros(env.K, device=env.dev)

        corr = {}

        def grad_fn(ctx, grad):                                   # c − c_i is constant within the epoch
            if corr["v"].shape[0] != ctx.Gfix:
                corr["v"] = ctx.pad(corr["v"])                     # pad once per group, not per step
            grad += corr["v"]

        def after(idx, flat, bufs, st):
            k = st["applied"].clamp_min(1)[:, None]
            new = self.ci[idx] - self.c[None] + (self.p[None] - flat) / (k * lr)
            dc.add_((new - self.ci[idx]).sum(0))
            self.ci[idx] = new
            dx.add_((flat - self.p[None]).sum(0))
            db.add_(bufs.sum(0))
            loss[idx] = st["loss"]

        for idx in env.groups_idx():
            flat, bufs = self.p.repeat(len(idx), 1), self.b.repeat(len(idx), 1)
            corr["v"] = self.c[None] - self.ci[idx]
            st = local.local_epoch(env.lay, flat, bufs, env.train, idx, env.cfg, env.hp, lr, env.gen, grad_fn=grad_fn)
            after(idx, flat, bufs, st)
        N = env.allreduce(torch.tensor(float(env.K), device=env.dev))
        self.p = self.p + env.allreduce(dx) / N
        self.c = self.c + env.allreduce(dc) / N
        self.b = env.allreduce(db) / N
        return {"train_loss": float(env.allreduce(loss.sum()) / N), "c_norm": float(self.c.norm())}

    def state(self):
        return {**super().state(), "c": self.c.cpu(), "ci": self.ci.cpu()}

    def load_state(self, s):
        super().load_state(s)
        self.c, self.ci = s["c"].to(self.env.dev), s["ci"].to(self.env.dev)
