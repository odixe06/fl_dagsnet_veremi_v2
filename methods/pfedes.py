"""pFedES — Yi et al., AAAI 2025 (papers/pfedes.pdf), §Iterative Training and Model Aggregation, Eq. (4)-(11).

Each client keeps a heterogeneous local model F_k (DAGSNet here) and shares only a small proxy homogeneous feature
extractor G(θ) that maps an input to an "enhanced" input of the same shape. Round t:
  Step 1  freeze G(θ^{t-1}); train F_k with ℓ_ω = μ·ℓ(F_k(G(x)), y) + (1 − μ)·ℓ(F_k(x), y)              Eq. (4)-(7)
  Step 2  freeze F_k; train G from θ^{t-1} with ℓ_θ = ℓ(F_k(G(x)), y) for E_fe epochs, upload θ_k^t       Eq. (8)-(10)
  Server  θ^t = Σ_k (n_k / n) θ_k^t                                                                       Eq. (11)
Inference: F_k alone (paper: "only each client's personalized heterogeneous local model F_k is used").

G follows Fig. 1(b) (Conv 3x3, 8 filters, 'same' → Conv 3x3 back to the input channels), in 1-D on DAGSNet's own patch
layout: x (77) → (7 channels × 11 positions) → Conv1d(7→8, 3) → ReLU → Conv1d(8→7, 3) → 77. Deviations: μ = 0.5 and
E_fe = 1 (paper tunes μ ∈ (0, 0.5] and E_fe per dataset, values not printed); ReLU between the two convs is assumed
(Fig. 1(b) shows the two convolutions only); F_k frozen in Step 2 means eval mode (BN running statistics, no update);
in Step 1 the two inputs share one forward (concatenated rows, one set of BN batch statistics).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import grouped, local
from core.flat import Layout


class ProxyExtractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg["patch_len"]
        self.net = nn.Sequential(nn.Conv1d(c, 8, 3, padding=1), nn.ReLU(), nn.Conv1d(8, c, 3, padding=1))
        self.k, self.c = cfg["n_features"] // c, c

    def forward(self, x):                                            # reference, (B, 77) -> (B, 77)
        fm = x.view(x.shape[0], self.k, self.c).transpose(1, 2)
        return self.net(fm).transpose(1, 2).reshape(x.shape[0], -1)


def g_forward(P, x, cfg):
    """Grouped ProxyExtractor: P (G, ...) params, x (G, b, 77) -> (G, b, 77)."""
    c = grouped._Ctx(P, {}, None, False)
    G, b, _ = x.shape
    h = x.view(G, b, cfg["n_features"] // cfg["patch_len"], cfg["patch_len"])
    h = torch.relu(grouped._conv(c, "net.0", h, 3) + P["net.0.bias"][:, None, None].to(h.dtype))
    h = grouped._conv(c, "net.2", h, 3) + P["net.2.bias"][:, None, None].to(h.dtype)
    return h.reshape(G, b, -1)


class Method:
    name = "pfedes"
    per_client = True

    def __init__(self, env):
        self.env = env
        self.mu = env.hp.get("mu", 0.5)
        self.e_fe = env.hp.get("e_fe", 1)
        self.glay = Layout(env.cfg, make=ProxyExtractor)
        p0, b0 = env.lay.init(env.hp["seed"], 1, env.dev)
        self.bank_p, self.bank_b = p0.repeat(env.K, 1), b0.repeat(env.K, 1)
        self.g, _ = self.glay.init(env.hp["seed"] + 1, 1, env.dev)
        self.g = self.g[0]

    def payload_bytes(self):
        return {"down": 4 * self.glay.P, "up": 4 * self.glay.P}

    def round(self, r, lr):
        env, cfg = self.env, self.env.cfg
        gacc = torch.zeros_like(self.g)
        loss = torch.zeros(env.K, device=env.dev)
        for idx in env.groups_idx():
            Gf = local.group_size(env.train.off, idx, env.hp)
            Pg = self.glay.params(self.g[None].repeat(Gf, 1))
            mu = self.mu

            def step1(ctx):                                           # train F_k, G frozen
                with torch.no_grad():
                    xh = g_forward(Pg, ctx.x, cfg)
                xx = torch.cat([xh, ctx.x], 1)
                mm = torch.cat([ctx.mask, ctx.mask], 1)
                z, nb = grouped.TRAIN(ctx.P, ctx.B, xx, cfg, mask=mm, train=True)
                b = ctx.x.shape[1]
                l1 = local.ce_loss(z[:, :b], ctx.y, ctx.mask)
                l2 = local.ce_loss(z[:, b:], ctx.y, ctx.mask)
                return mu * l1 + (1 - mu) * l2, nb

            flat, bufs = self.bank_p[idx], self.bank_b[idx]
            st = local.local_epoch(env.lay, flat, bufs, env.train, idx, cfg, env.hp, lr, env.gen, loss_fn=step1)
            self.bank_p[idx], self.bank_b[idx] = flat, bufs
            Fp = env.lay.params(local.pad_rows(flat, Gf))
            Fb = env.lay.bufs(local.pad_rows(bufs, Gf))

            def step2(ctx):                                           # train G, F_k frozen (eval mode)
                z, _ = grouped.TRAIN(Fp, Fb, g_forward(ctx.P, ctx.x, cfg), cfg, train=False)
                return local.ce_loss(z, ctx.y, ctx.mask), {}

            gflat = self.g.repeat(len(idx), 1)
            gb = torch.zeros(len(idx), 0, device=env.dev)
            for _ in range(self.e_fe):
                local.local_epoch(self.glay, gflat, gb, env.train, idx, cfg, env.hp, lr, env.gen, loss_fn=step2)
            gacc += (env.n[idx][:, None] * gflat).sum(0)
            loss[idx] = st["loss"]
        tot = env.allreduce(env.n.sum().clone())
        self.g = env.allreduce(gacc) / tot
        return {"train_loss": float(env.allreduce((loss * env.n).sum()) / tot)}

    def evaluate(self):
        return {"client": self.env.eval_clients(self.bank_p, self.bank_b)}

    def state(self):
        return {"bank_p": self.bank_p.cpu(), "bank_b": self.bank_b.cpu(), "g": self.g.cpu()}

    def load_state(self, s):
        self.bank_p, self.bank_b, self.g = (s[k].to(self.env.dev) for k in ("bank_p", "bank_b", "g"))
