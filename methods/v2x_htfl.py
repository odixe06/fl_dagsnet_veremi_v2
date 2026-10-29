"""V2X-HtFL — Liu et al., IEEE OJVT 2025, DOI 10.1109/OJVT.2025.3594030 (papers/v2x_htfl.pdf), §IV-B, IV-C, Eq. (13)-(18), (22).

Client k: feature extractor f_k = DAGSNet up to the hidden layer (h, d dims), a high-dimensional trainable projection
p_k ∈ R^{d×d} with weight normalisation (w = g · v/‖v‖ per row, the paper's "dynamic weight normalisation"), and a FIXED
simplex-ETF classifier V = sqrt(C/(C−1)) U (I − 11ᵀ/C) (U column-orthogonal, Eq. 9; same seed for every client).
Local loss: the CurricularFace-style loss of §IV-B as printed, with cos θ_j = ⟨p_k(f_k(x))/‖·‖, v_j⟩:
    L_C = −log e^{s cos(θ_y + m)} / (e^{s cos(θ_y + m)} + Σ_{j≠y} e^{s γ^t cos θ_j}),  t = round,
plus the knowledge-transfer term μ λ_k L_s (Eq. 22), λ_k = |D| / (|D| + |D_k|).
After training each client uploads its class prototypes P_k^c = mean p_k(f_k(x)) over its rows of class c (eval mode).
Server (§IV-C): trainable global prototypes P̂^c = F_G(P̄^c; u_F), P̄^c ∈ R^d trainable, F_G = multi-head self-attention over
the C prototypes (residual) → FC → ReLU → FC (FedTGP's network, which the paper builds on), trained S epochs per round
on every uploaded P_k^c with the adaptive-margin contrastive loss Eq. (16) and margin Eq. (17)-(18):
    δ(t) = min(max_{c≠c'} η(Q^c, Q^{c'}) / sqrt(ν_c + ν_{c'}), τ),  Q^c = mean_k P_k^c,  ν_c = mean_k ‖P_k^c − Q^c‖².
Inference: the client's own model with its ETF classifier ("client", headline); nearest global prototype by cosine
("client_proto", reported).

Deviations (owner 30/10: drop the generative part): §IV-D/E (feature transformer F_T, MK-MMD to StyleGAN-XL's latent
space, synthetic images) need a pretrained image generator, which tabular VeReMi does not have; L_s is therefore the
cosine distance between the client's projected feature and the global prototype of its label, 1 − η(p_k(f_k(x)), P̂^y),
instead of the loss on generated images. Eq. (16) is printed with e^{−η} although η is a cosine *similarity* (FedTGP's
form uses a distance); it is applied with the cosine distance 1 − η and scale s, i.e. CE(s(η(P_k^c, P̂^{c'}) − δ·1[c'=c])).
The paper prints no s, m, γ, τ, μ: s = 64, m = 0.5 (CurricularFace's defaults), γ = 0.99, τ = m, μ = 0.1 (FedTGP's λ);
server Adam lr 0.01 and S = 100 epochs (paper: Adam, 0.01; FedTGP: 100); server batch = ceil(#prototypes / 20) (FedTGP
deviation, methods/fedtgp.py). DAGSNet's own classifier head.5 is kept in the layout but unused (zero gradient).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core import evaluate, grouped, local
from core.dagsnet import DAGSNet
from core.flat import Layout
from methods import fedtgp

C = 16


class V2XNet(DAGSNet):
    """DAGSNet + weight-normalised projection p_k (parameters registered last: proj_v, proj_g, proj_b)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        d = cfg["head_hidden"]
        lin = nn.Linear(d, d)
        self.proj_v = nn.Parameter(lin.weight.detach().clone())
        self.proj_g = nn.Parameter(lin.weight.detach().norm(dim=1))
        self.proj_b = nn.Parameter(torch.zeros(d))


def project(P, h):
    """Grouped p_k: h (G, b, d) -> (G, b, d)."""
    v = P["proj_v"]
    W = P["proj_g"][..., None] * v / v.norm(dim=2, keepdim=True).clamp_min(1e-12)
    return torch.baddbmm(P["proj_b"][:, None].to(h.dtype), h, W.transpose(1, 2).to(h.dtype))


def etf(d, seed, device):
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(d, C, generator=g))
    return (math.sqrt(C / (C - 1)) * U @ (torch.eye(C) - torch.ones(C, C) / C)).to(device)


class GlobalProto(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(C, d) * 0.1)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self):
        e = self.emb[None]
        return self.net(self.attn(e, e, e, need_weights=False)[0][0] + self.emb)


class Method:
    name = "v2x_htfl"
    per_client = True

    def __init__(self, env):
        self.env = env
        hp = env.hp
        self.s, self.m, self.gamma = hp.get("s", 64.0), hp.get("m", 0.5), hp.get("gamma", 0.99)
        self.tau, self.mu, self.S = hp.get("tau", self.m), hp.get("mu", 0.1), hp.get("server_epochs", 100)
        env.lay = Layout(env.cfg, make=V2XNet)
        self.d = env.cfg["head_hidden"]
        p0, b0 = env.lay.init(hp["seed"], 1, env.dev)
        self.bank_p, self.bank_b = p0.repeat(env.K, 1), b0.repeat(env.K, 1)
        self.V = etf(self.d, hp["seed"] + 11, env.dev)
        torch.manual_seed(hp["seed"] + 5)
        self.gp = GlobalProto(self.d).to(env.dev)
        self.opt = torch.optim.Adam(self.gp.parameters(), lr=hp.get("server_lr", 0.01))
        self.gen_s = torch.Generator(device=env.dev).manual_seed(hp["seed"] + 9)
        self.proto, self.delta = None, 0.0
        n_all = env.allreduce(env.n.sum().clone())
        self.lam = n_all / (n_all + env.n)                            # λ_k (Eq. 22)

    def payload_bytes(self):
        return {"up": 4 * C * self.d + 4 * C, "down": 4 * C * self.d, "note": "up = prototypes of classes present + counts"}

    def _cos(self, P, h):
        g = project(P, h).float()
        return F.normalize(g, dim=-1) @ self.V

    def round(self, r, lr):
        env, cfg = self.env, self.env.cfg
        loss = torch.zeros(env.K, device=env.dev)
        gt = self.gamma ** r
        cm, sm = math.cos(self.m), math.sin(self.m)
        th, mm = math.cos(math.pi - self.m), math.sin(math.pi - self.m) * self.m
        for idx in env.groups_idx():
            lam = self.lam[idx]

            def loss_fn(ctx):
                z, h, nb = grouped.TRAIN(ctx.P, ctx.B, ctx.x, cfg, mask=ctx.mask, train=True, hidden=True)
                g = project(ctx.P, h).float()
                gn = F.normalize(g, dim=-1)
                cos = gn @ self.V                                     # (G, b, C)
                cy = cos.gather(-1, ctx.y[..., None]).squeeze(-1)
                sy = (1 - cy * cy).clamp_min(0).sqrt()
                phi = torch.where(cy > th, cy * cm - sy * sm, cy - mm)   # cos(θ_y + m), ArcFace's monotone extension
                logits = (self.s * gt * cos).scatter(-1, ctx.y[..., None], self.s * phi[..., None])
                G, b = ctx.y.shape
                lc = F.cross_entropy(logits.reshape(-1, C), ctx.y.reshape(-1), reduction="none").view(G, b)
                per = lc
                if self.proto is not None:
                    pn = F.normalize(self.proto, dim=-1)[ctx.y]
                    ls = torch.where(self.has[ctx.y], 1 - (gn * pn).sum(-1), 0.0)
                    per = per + self.mu * ctx.pad(lam)[:, None] * ls
                return local.masked_mean(per, ctx.mask) + 0.0 * z.float().sum((1, 2)), nb   # head.5: zero gradient

            flat, bufs = self.bank_p[idx], self.bank_b[idx]
            st = local.local_epoch(env.lay, flat, bufs, env.train, idx, cfg, env.hp, lr, env.gen, loss_fn=loss_fn)
            self.bank_p[idx], self.bank_b[idx] = flat, bufs
            loss[idx] = st["loss"]
        S, Cn = evaluate.class_sums(env.lay, self.bank_p, self.bank_b, env.train, cfg, torch.arange(env.K, device=env.dev),
                                    feat=project, d=self.d)
        protos, labels = fedtgp.Method._gather(self, S, Cn)
        self._server(protos, labels)
        tot = env.allreduce(env.n.sum().clone())
        return {"train_loss": float(env.allreduce((loss * env.n).sum()) / tot), "delta": self.delta,
                "n_protos": int(len(labels))}

    def _server(self, P, y):
        dev = P.device
        Q = torch.zeros(C, self.d, device=dev)
        nu = torch.zeros(C, device=dev)
        self.has = torch.zeros(C, dtype=torch.bool, device=dev)
        for c in range(C):
            if (y == c).any():
                Q[c] = P[y == c].mean(0)
                nu[c] = ((P[y == c] - Q[c]) ** 2).sum(1).mean()
                self.has[c] = True
        ok = torch.nonzero(self.has).flatten()
        if len(ok) > 1:
            eta = F.normalize(Q[ok], dim=1) @ F.normalize(Q[ok], dim=1).T
            ratio = eta / (nu[ok][:, None] + nu[ok][None]).sqrt().clamp_min(1e-12)
            ratio.fill_diagonal_(-float("inf"))
            self.delta = float(min(ratio.max().item(), self.tau))
        n = len(y)
        bs = math.ceil(n / 20)
        Pn = F.normalize(P, dim=1)
        self.gp.train()
        for _ in range(self.S):
            perm = torch.randperm(n, device=dev, generator=self.gen_s)
            for s0 in range(0, n, bs):
                i = perm[s0:s0 + bs]
                eta = Pn[i] @ F.normalize(self.gp(), dim=1).T
                l = F.cross_entropy(self.s * (eta - self.delta * F.one_hot(y[i], C)), y[i])
                self.opt.zero_grad(); l.backward(); self.opt.step()
        self.proto = self.gp().detach()          # with grad: FlopCounterMode's module hooks reject a no_grad forward

    def evaluate(self):
        env = self.env
        clients = torch.arange(env.K, device=env.dev)
        rows = env.hp.get("eval_rows", 1 << 15)
        out = {"client": env.gather_cms(evaluate.own_cm(env.lay, self.bank_p, self.bank_b, env.test, env.cfg, clients,
                                                        max_rows=rows, head=self._cos))}
        if self.proto is not None:
            pn = F.normalize(self.proto, dim=1)
            head = lambda P, h: (F.normalize(project(P, h).float(), dim=-1) @ pn.T).masked_fill(~self.has, -2.0)
            out["client_proto"] = env.gather_cms(evaluate.own_cm(env.lay, self.bank_p, self.bank_b, env.test, env.cfg,
                                                                 clients, max_rows=rows, head=head))
        return out

    def state(self):
        return {"bank_p": self.bank_p.cpu(), "bank_b": self.bank_b.cpu(),
                "gp": {k: v.cpu() for k, v in self.gp.state_dict().items()}, "opt": self.opt.state_dict(),
                "proto": None if self.proto is None else self.proto.cpu(), "has": self.has.cpu() if hasattr(self, "has") else None,
                "delta": self.delta}

    def load_state(self, s):
        dev = self.env.dev
        self.bank_p, self.bank_b = s["bank_p"].to(dev), s["bank_b"].to(dev)
        self.gp.load_state_dict(s["gp"]); self.opt.load_state_dict(s["opt"])
        self.proto = None if s["proto"] is None else s["proto"].to(dev)
        if s["has"] is not None:
            self.has = s["has"].to(dev)
        self.delta = s["delta"]
