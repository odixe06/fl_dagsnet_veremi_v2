"""SCAFFOLD — Karimireddy et al., ICML 2020 (papers/scaffold.pdf), Algorithm 1, Option II, η_g = 1.

Local step on the corrected gradient g_i(y) − c_i + c (fed to the shared AdamW; the paper analyses SGD). After the epoch
c_i⁺ = mean of the raw local gradients g_i(y_i) over the K_i applied steps: under the paper's SGD this IS Option II
(c_i − c + (x − y_i)/(K η) = mean_k g_k, since y_i = x − η Σ_k (g_k − c_i + c)); under AdamW the literal (x − y_i)/(Kη)
measures Adam's normalised steps, not gradients (smoke 29/10: F1 0.16 vs FedAvg 0.69), so the mean gradient is used.
Server (all N clients participate, |S| = N): x ← x + Σ_i (n_i/n)(y_i − x), c ← c + Σ_i (n_i/n)(c_i⁺ − c_i), BN alike.
Deviation (owner 29/10, protocol rule): sample-weighted means; Algorithm 1 averages uniformly, which on the heavy-tailed
vehicle partition pulls x back to θ⁰ (smoke: F1 0.03 at r1, when c = c_i = 0 and SCAFFOLD = unweighted FedAvg). c, c_i start at 0. Control variates of this rank's clients stay on its GPU.
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
            if corr.get("g") is None:
                corr["g"] = torch.zeros_like(corr["v"])
            ok = (ctx.mask.any(1) & torch.isfinite(grad).all(1))[:, None]
            corr["g"] += torch.where(ok, grad, 0.0)                  # raw g_i(y), before the correction
            grad += corr["v"]

        def after(idx, flat, bufs, st):
            k = st["applied"].clamp_min(1)[:, None]
            new = corr["g"][:len(idx)] / k
            w = env.n[idx][:, None]
            dc.add_((w * (new - self.ci[idx])).sum(0))
            self.ci[idx] = new
            dx.add_((w * (flat - self.p[None])).sum(0))
            db.add_((w * bufs).sum(0))
            loss[idx] = st["loss"]

        for idx in env.groups_idx():
            flat, bufs = self.p.repeat(len(idx), 1), self.b.repeat(len(idx), 1)
            corr["v"], corr["g"] = self.c[None] - self.ci[idx], None
            st = local.local_epoch(env.lay, flat, bufs, env.train, idx, env.cfg, env.hp, lr, env.gen, grad_fn=grad_fn)
            after(idx, flat, bufs, st)
        N = env.allreduce(env.n.sum().clone())
        self.p = self.p + env.allreduce(dx) / N
        self.c = self.c + env.allreduce(dc) / N
        self.b = env.allreduce(db) / N
        return {"train_loss": float(env.allreduce((loss * env.n).sum()) / N), "c_norm": float(self.c.norm())}

    def state(self):
        return {**super().state(), "c": self.c.cpu(), "ci": self.ci.cpu()}

    def load_state(self, s):
        super().load_state(s)
        self.c, self.ci = s["c"].to(self.env.dev), s["ci"].to(self.env.dev)
