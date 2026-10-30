"""TinyProto-FP — Lee & Choi, AAAI 2026 (papers/tinyproto.pdf), Algorithm 1: FedProto + Class-wise Prototype
Sparsification (CPS) + Adaptive Prototype Scaling (APS).

CPS: fixed binary masks m_j (s of d dims per class, spread for maximal inter-class Hamming distance), sent once; only the
s masked coordinates of a prototype travel (compressed prototype ĉ, Def. 2). APS: client i uploads n_ij·ĉ^L_ij; the
server averages over the clients holding class j, ĉ^G_j = mean_i n_ij·ĉ^L_ij (Eq. 10), without ever seeing n_ij.
Client loss (Eq. 5, 11): CE + λ·ρ(f(x), μ·c̃^G_y), c̃^G = ĉ^G scattered back into d dims (zeros off the mask), ρ = MSE per
sample as FedProto's code, λ = 1 (paper, following FedProto). Local prototypes from an eval-mode pass after training
(the authors' code base, PFLlib). Inference: nearest dense local prototype c̄^L_ij of the client over the classes it holds
("client_proto", Eq. 12; before 30/10 this used μ·c̃^G, which is zero off the mask) and the classifier head ("client").
Choices the paper leaves open here: s = ceil(d / 10) (paper default s = 50 of d = 500, 90 % compression); masks = blocks of
a seeded permutation of the d dims, cyclically (16·s > d forces overlaps; blocks spread them evenly); μ = 1 / n̄ with n̄
the mean n_ij over (client, class) pairs, computed once at start (the paper's μ-selection rule is in its appendix, not
in the PDF).
"""
import math

import torch

from core import evaluate, grouped, local


class Method:
    name = "tinyproto"
    per_client = True

    def __init__(self, env):
        self.env = env
        hp = env.hp
        self.lam = hp.get("lam", 1.0)
        d = env.cfg["head_hidden"]
        self.s = hp.get("cps_dim", math.ceil(d / 10))
        g = torch.Generator().manual_seed(hp["seed"] + 13)
        perm = torch.randperm(d, generator=g)
        self.mask = torch.zeros(16, d, dtype=torch.bool)
        for j in range(16):
            self.mask[j, perm[(j * self.s + torch.arange(self.s)) % d]] = True
        self.mask = self.mask.to(env.dev)
        cnt = torch.zeros(env.K, 16, device=env.dev)
        client = torch.repeat_interleave(torch.arange(env.K, device=env.dev), env.train.off[1:] - env.train.off[:-1])
        cnt.index_put_((client, env.train.Y.long()), torch.ones(len(env.train.Y), device=env.dev), accumulate=True)
        pairs = env.allreduce(torch.stack([(cnt > 0).sum().float(), cnt.sum()]))
        self.mu = hp.get("mu", float(pairs[0] / pairs[1]))
        p0, b0 = env.lay.init(hp["seed"], 1, env.dev)
        self.bank_p, self.bank_b = p0.repeat(env.K, 1), b0.repeat(env.K, 1)
        self.target = None                                           # μ·c̃^G (16, d)

    def payload_bytes(self):
        return {"up": 4 * 16 * self.s, "down": 4 * 16 * self.s, "masks_once": 16 * self.env.cfg["head_hidden"] // 8}

    def round(self, r, lr):
        env, cfg = self.env, self.env.cfg
        loss = torch.zeros(env.K, device=env.dev)
        for idx in env.groups_idx():
            def loss_fn(ctx):
                z, h, nb = grouped.TRAIN(ctx.P, ctx.B, ctx.x, cfg, mask=ctx.mask, train=True, hidden=True)
                ce = local.ce_loss(z, ctx.y, ctx.mask)
                if self.target is None:
                    return ce, nb
                diff = ((h - self.target[ctx.y]) ** 2).mean(-1)
                return ce + self.lam * torch.where(ctx.mask, diff, 0.0).sum(1) / ctx.mask.sum(1).clamp_min(1), nb

            flat, bufs = self.bank_p[idx], self.bank_b[idx]
            st = local.local_epoch(env.lay, flat, bufs, env.train, idx, cfg, env.hp, lr, env.gen, loss_fn=loss_fn)
            self.bank_p[idx], self.bank_b[idx] = flat, bufs
            loss[idx] = st["loss"]
        S, Cn = evaluate.class_sums(env.lay, self.bank_p, self.bank_b, env.train, cfg, torch.arange(env.K, device=env.dev))
        self.local = (S / Cn.clamp_min(1)[..., None]).half()              # dense c̄^L_ij, inference only (Eq. 12)
        self.has = Cn > 0
        # client upload: n_ij·ĉ^L_ij = masked coordinates of the class sum (n_ij · mean); server: mean over holders
        scaled = (S * self.mask[None]).sum(0)                          # Σ_i n_ij c̄_ij on the mask (zeros off it)
        holders = (Cn > 0).sum(0).float()
        scaled, holders = env.allreduce(scaled), env.allreduce(holders)
        self.target = self.mu * scaled / holders.clamp_min(1)[:, None]
        tot = env.allreduce(env.n.sum().clone())
        return {"train_loss": float(env.allreduce((loss * env.n).sum()) / tot)}

    def evaluate(self):
        env = self.env
        out = {"client": env.eval_clients(self.bank_p, self.bank_b)}
        cms = evaluate.own_cm(env.lay, self.bank_p, self.bank_b, env.test, env.cfg, torch.arange(env.K, device=env.dev),
                              max_rows=env.hp.get("eval_rows", 1 << 15),
                              protos=(self.local, self.has))
        out["client_proto"] = env.gather_cms(cms)
        return out

    def state(self):
        return {k: getattr(self, k).cpu() for k in ("bank_p", "bank_b", "target", "local", "has")}

    def load_state(self, s):
        self.bank_p, self.bank_b, self.target, self.local, self.has = (
            s[k].to(self.env.dev) for k in ("bank_p", "bank_b", "target", "local", "has"))
