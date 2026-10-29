"""FedTGP — Zhang et al., AAAI 2024 (papers/fedtgp.pdf), Algorithm 1, Eq. (2), (9)-(11).

Clients exchange prototypes only. Client i: L_i = CE + λ·ϕ(f(x), P̂^y) (Eq. 11; ϕ as in FedProto / the authors' PFLlib
code: MSE between the representation and the global prototype of its label; no term before the first P̂), then uploads
P_i^c = mean f(x) over its rows of class c (Eq. 2, collected after training in eval mode as in PFLlib).
Server: trainable global prototypes P̂^c = F(Ṕ^c), Ṕ^c ∈ R^K trainable vectors, F = FC → ReLU → FC (K → K → K), trained
S epochs per round with the adaptive-margin contrastive loss (Eq. 10): CE(−(‖P_i^c − P̂‖ + δ·onehot_c), c) over every
uploaded P_i^c, δ(t) = min(max_{c≠c'} ‖Q^c − Q^c'‖, τ), Q^c = unweighted mean of the uploaded P_i^c (Eq. 9).
Paper values: λ = 0.1, τ = 100, S = 100, SGD lr 0.01 (η). Inference: nearest global prototype ("client_proto", the
paper's rule); the client's classifier head is reported too ("client").
Deviation: server batch = ceil(#uploaded prototypes / 20) so an epoch is 20 steps, the paper's count (20 clients × ≤ 10
classes at batch 10); batch 10 over ~80k prototypes would be 800k server steps per round.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core import evaluate, grouped, local


class TGP(nn.Module):
    def __init__(self, C, K):
        super().__init__()
        self.emb = nn.Embedding(C, K)
        self.net = nn.Sequential(nn.Linear(K, K), nn.ReLU(), nn.Linear(K, K))

    def forward(self):
        return self.net(self.emb.weight)


class Method:
    name = "fedtgp"
    per_client = True

    def __init__(self, env):
        self.env = env
        hp = env.hp
        self.lam, self.tau, self.S = hp.get("lam", 0.1), hp.get("tau", 100.0), hp.get("server_epochs", 100)
        p0, b0 = env.lay.init(hp["seed"], 1, env.dev)
        self.bank_p, self.bank_b = p0.repeat(env.K, 1), b0.repeat(env.K, 1)
        K = env.cfg["head_hidden"]
        torch.manual_seed(hp["seed"] + 5)
        self.tgp = TGP(16, K).to(env.dev)
        self.opt = torch.optim.SGD(self.tgp.parameters(), lr=hp.get("server_lr", 0.01))
        self.proto = None
        self.gen_s = torch.Generator(device=env.dev).manual_seed(hp["seed"] + 9)

    def payload_bytes(self):
        d = self.env.cfg["head_hidden"]
        return {"up": 4 * 16 * d, "down": 4 * 16 * d}

    def round(self, r, lr):
        env, cfg = self.env, self.env.cfg
        loss = torch.zeros(env.K, device=env.dev)
        for idx in env.groups_idx():
            def loss_fn(ctx):
                z, h, nb = grouped.TRAIN(ctx.P, ctx.B, ctx.x, cfg, mask=ctx.mask, train=True, hidden=True)
                ce = local.ce_loss(z, ctx.y, ctx.mask)
                if self.proto is None:
                    return ce, nb
                diff = ((h - self.proto[ctx.y]) ** 2).mean(-1)
                return ce + self.lam * (diff * ctx.mask).sum(1) / ctx.mask.sum(1).clamp_min(1), nb

            flat, bufs = self.bank_p[idx], self.bank_b[idx]
            st = local.local_epoch(env.lay, flat, bufs, env.train, idx, cfg, env.hp, lr, env.gen, loss_fn=loss_fn)
            self.bank_p[idx], self.bank_b[idx] = flat, bufs
            loss[idx] = st["loss"]
        # every client uploads its class prototypes (eval-mode pass over its training rows)
        S, Cn = evaluate.class_sums(env.lay, self.bank_p, self.bank_b, env.train, cfg, torch.arange(env.K, device=env.dev))
        protos, labels = self._gather(S, Cn)
        self._server(protos, labels)
        tot = env.allreduce(env.n.sum().clone())
        return {"train_loss": float(env.allreduce((loss * env.n).sum()) / tot), "delta": self.delta,
                "n_protos": int(len(labels))}

    def _gather(self, S, Cn):
        """All uploaded (client, class) prototypes on every rank (identical server state everywhere)."""
        env = self.env
        have = Cn > 0
        P = (S / Cn.clamp_min(1)[..., None])[have]
        y = torch.nonzero(have)[:, 1]
        if env.world > 1:
            import torch.distributed as dist
            sizes = [torch.zeros(1, dtype=torch.long, device=env.dev) for _ in range(env.world)]
            dist.all_gather(sizes, torch.tensor([len(y)], device=env.dev))
            m = max(int(s) for s in sizes)
            Pp = torch.zeros(m, P.shape[1], device=env.dev); Pp[:len(y)] = P
            yp = torch.full((m,), -1, dtype=torch.long, device=env.dev); yp[:len(y)] = y
            Pa = [torch.zeros_like(Pp) for _ in range(env.world)]; ya = [torch.zeros_like(yp) for _ in range(env.world)]
            dist.all_gather(Pa, Pp); dist.all_gather(ya, yp)
            P, y = torch.cat(Pa), torch.cat(ya)
            P, y = P[y >= 0], y[y >= 0]
        return P, y

    def _server(self, P, y):
        Q = torch.stack([P[y == c].mean(0) if (y == c).any() else torch.full((P.shape[1],), float("nan"), device=P.device)
                         for c in range(16)])
        ok = ~torch.isnan(Q[:, 0])
        d = torch.cdist(Q[ok], Q[ok])
        self.delta = float(min(d.max().item(), self.tau))
        n = len(y)
        bs = math.ceil(n / 20)
        self.tgp.train()
        for _ in range(self.S):
            perm = torch.randperm(n, device=P.device, generator=self.gen_s)
            for s in range(0, n, bs):
                i = perm[s:s + bs]
                gen = self.tgp()
                dist = torch.cdist(P[i], gen)
                dist = dist + self.delta * F.one_hot(y[i], 16)
                l = F.cross_entropy(-dist, y[i])
                self.opt.zero_grad(); l.backward(); self.opt.step()
        with torch.no_grad():
            self.proto = self.tgp().detach()

    def evaluate(self):
        env = self.env
        out = {"client": env.eval_clients(self.bank_p, self.bank_b)}
        cms = evaluate.own_cm(env.lay, self.bank_p, self.bank_b, env.test, env.cfg, torch.arange(env.K, device=env.dev),
                              max_rows=env.hp.get("eval_rows", 1 << 15),
                              protos=(self.proto, torch.ones(16, dtype=torch.bool, device=env.dev)))
        out["client_proto"] = env.gather_cms(cms)
        return out

    def state(self):
        return {"bank_p": self.bank_p.cpu(), "bank_b": self.bank_b.cpu(), "tgp": {k: v.cpu() for k, v in self.tgp.state_dict().items()},
                "opt": self.opt.state_dict(), "proto": self.proto.cpu()}

    def load_state(self, s):
        self.bank_p, self.bank_b = s["bank_p"].to(self.env.dev), s["bank_b"].to(self.env.dev)
        self.tgp.load_state_dict(s["tgp"]); self.opt.load_state_dict(s["opt"]); self.proto = s["proto"].to(self.env.dev)
