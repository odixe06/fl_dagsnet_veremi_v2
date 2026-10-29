"""NILM-FL — Li et al., IEEE TSG 2025 (papers/nilm.pdf), §III-C Adaptive Federated Mutual Learning, Eq. (18)-(19), Alg. 1.

Each device trains a personalised model w_s and a unified proxy w_r on its data with mutual distillation:
    L_s = ℓ(y, y_s) + ℓ_d,   L_r = ℓ(y, y_r) + ℓ_d,   ℓ_d = ℓ(y_s, y_r) / (ℓ_s + ℓ_r)        (Eq. 18-19, adaptive weight)
Only w_r is uploaded; server w̄_r = Σ_k (n_k/n) w_r^k — deviation (owner 29/10, protocol rule): Alg. 1 averages uniformly
(1/K); evaluation uses the personalised model.
For 16-class IDS ℓ = cross-entropy and ℓ(y_s, y_r) = KL between the two softmaxes, each model distilling from the other's
detached prediction (deep mutual learning; NILM's regression used MSE). The adaptive denominator is detached.
Deviations: the personalised architecture is DAGSNet (the paper searches it with hardware-aware NAS, §III-B; the
comparison fixes DAGSNet for every client); shared protocol optimizer (AdamW, both models one optimizer, clip per model).
"""
import torch
import torch.nn.functional as F

from core import grouped, local


class Method:
    name = "nilm"
    per_client = True

    def __init__(self, env):
        self.env = env
        p0, b0 = env.lay.init(env.hp["seed"], 1, env.dev)
        self.priv_p, self.priv_b = p0.repeat(env.K, 1), b0.repeat(env.K, 1)
        self.g_p, self.g_b = p0[0].clone(), b0[0].clone()

    def payload_bytes(self):
        n = 4 * (self.env.lay.P + self.env.lay.B)
        return {"down": n, "up": n}

    def _loss(self, ctx):
        cfg = self.env.cfg
        (Ps, Pr), (Bs, Br) = ctx.P, ctx.B
        zs, nbs = grouped.TRAIN(Ps, Bs, ctx.x, cfg, mask=ctx.mask, train=True)
        zr, nbr = grouped.TRAIN(Pr, Br, ctx.x, cfg, mask=ctx.mask, train=True)
        ls_, lr_ = local.ce_loss(zs, ctx.y, ctx.mask), local.ce_loss(zr, ctx.y, ctx.mask)
        ps, pr = F.log_softmax(zs, -1), F.log_softmax(zr, -1)
        kl_s = local.masked_mean((pr.detach().exp() * (pr.detach() - ps)).sum(-1), ctx.mask)      # personalised <- proxy
        kl_r = local.masked_mean((ps.detach().exp() * (ps.detach() - pr)).sum(-1), ctx.mask)      # proxy <- personalised
        w = 1 / (ls_ + lr_).detach().clamp_min(1e-6)
        return ls_ + lr_ + w * (kl_s + kl_r), [nbs, nbr]

    def round(self, r, lr):
        env, lay = self.env, self.env.lay
        P, Bn = lay.P, lay.B
        acc_p = torch.zeros_like(self.g_p, dtype=torch.float64)
        acc_b = torch.zeros_like(self.g_b, dtype=torch.float64)
        loss = torch.zeros(env.K, device=env.dev)
        for idx in env.groups_idx():
            G = len(idx)
            flat = torch.cat([self.priv_p[idx], self.g_p.repeat(G, 1)], 1)
            bufs = torch.cat([self.priv_b[idx], self.g_b.repeat(G, 1)], 1)
            st = local.local_epoch(lay, flat, bufs, env.train, idx, env.cfg, env.hp, lr, env.gen, loss_fn=self._loss, models=2)
            self.priv_p[idx], self.priv_b[idx] = flat[:, :P], bufs[:, :Bn]
            w = env.n[idx][:, None].double()
            acc_p += (w * flat[:, P:].double()).sum(0)
            acc_b += (w * bufs[:, Bn:].double()).sum(0)
            loss[idx] = st["loss"]
        K = env.allreduce(env.n.sum().double())
        self.g_p = (env.allreduce(acc_p) / K).float()
        self.g_b = (env.allreduce(acc_b) / K).float()
        return {"train_loss": float(env.allreduce((loss * env.n).sum()) / env.allreduce(env.n.sum().clone()))}

    def evaluate(self):
        return {"client": self.env.eval_clients(self.priv_p, self.priv_b), "proxy": self.env.eval_global(self.g_p, self.g_b)}

    def state(self):
        return {k: getattr(self, k).cpu() for k in ("priv_p", "priv_b", "g_p", "g_b")}

    def load_state(self, s):
        for k in ("priv_p", "priv_b", "g_p", "g_b"):
            setattr(self, k, s[k].to(self.env.dev))
