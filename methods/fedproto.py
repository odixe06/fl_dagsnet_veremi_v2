"""FedProto — Tan et al., AAAI 2022 (papers/fedproto.pdf), Algorithm 1, Eq. (3)-(8).

Clients keep personalised models and exchange only class prototypes: the representation f(x) is DAGSNet's hidden layer
(knowledge/dagsnet.md §5). Local loss = CE(F(x), y) + λ·MSE(f(x), C̄^(y)) over rows whose class has a global prototype
(the authors' code: per-sample MSE against the global prototype of its label; zero before the first aggregation).
Local prototypes C_k^(j) = mean f(x) over the client's rows of class j, accumulated during the local epoch (authors' code).
Server: C̄^(j) = Σ_k |D_k,j| C_k^(j) / Σ_k |D_k,j| (Eq. 6 read as the sample-weighted mean; its extra 1/|N_j| is a typo).
λ = 1 (paper's best for MNIST/FEMNIST). Evaluation: the client's own classifier head ("client") and the paper's
prototype inference ŷ = argmin_j ‖f(x) − C̄^(j)‖ ("client_proto", Eq. 4).
"""
import torch

from core import evaluate, grouped, local


class Method:
    name = "fedproto"
    per_client = True

    def __init__(self, env):
        self.env = env
        self.lam = env.hp.get("lam", 1.0)
        p0, b0 = env.lay.init(env.hp["seed"], 1, env.dev)
        self.bank_p, self.bank_b = p0.repeat(env.K, 1), b0.repeat(env.K, 1)
        d = env.cfg["head_hidden"]
        self.proto = torch.zeros(16, d, device=env.dev)
        self.has = torch.zeros(16, dtype=torch.bool, device=env.dev)

    def payload_bytes(self):
        d = self.env.cfg["head_hidden"]
        return {"up": 4 * 16 * d + 4 * 16, "down": 4 * 16 * d, "note": "up = prototypes of classes present + counts"}

    def round(self, r, lr):
        env, cfg = self.env, self.env.cfg
        d = cfg["head_hidden"]
        psum = torch.zeros(16, d, device=env.dev, dtype=torch.float64)
        pcnt = torch.zeros(16, device=env.dev, dtype=torch.float64)
        loss = torch.zeros(env.K, device=env.dev)
        for idx in env.groups_idx():
            Gf = local.group_size(env.train.off, idx, env.hp)
            acc = torch.zeros(Gf, 16, d, device=env.dev)
            cnt = torch.zeros(Gf, 16, device=env.dev)

            def loss_fn(ctx):
                z, h, nb = grouped.TRAIN(ctx.P, ctx.B, ctx.x, cfg, mask=ctx.mask, train=True, hidden=True)
                ce = local.ce_loss(z, ctx.y, ctx.mask)
                m = ctx.mask & self.has[ctx.y]
                diff = ((h - self.proto[ctx.y]) ** 2).mean(-1)              # MSE per row over the d dims
                reg = (diff * m).sum(1) / ctx.mask.sum(1).clamp_min(1)       # nn.MSELoss over the batch (absent class: 0)
                with torch.no_grad():
                    w = ctx.mask.float()
                    acc.index_put_((torch.arange(Gf, device=h.device)[:, None].expand_as(ctx.y), ctx.y),
                                   h.detach() * w[..., None], accumulate=True)
                    cnt.index_put_((torch.arange(Gf, device=h.device)[:, None].expand_as(ctx.y), ctx.y), w, accumulate=True)
                return ce + self.lam * reg, nb

            flat, bufs = self.bank_p[idx], self.bank_b[idx]
            st = local.local_epoch(env.lay, flat, bufs, env.train, idx, cfg, env.hp, lr, env.gen, loss_fn=loss_fn)
            self.bank_p[idx], self.bank_b[idx] = flat, bufs
            G = len(idx)
            local_proto = acc[:G] / cnt[:G, :, None].clamp_min(1)          # C_k^(j), what client k uploads
            psum += (local_proto * cnt[:G, :, None]).sum(0).double()
            pcnt += cnt[:G].sum(0).double()
            loss[idx] = st["loss"]
        psum, pcnt = env.allreduce(psum), env.allreduce(pcnt)
        self.has = pcnt > 0
        self.proto = (psum / pcnt.clamp_min(1)[:, None]).float()
        tot = env.allreduce(env.n.sum().clone())
        return {"train_loss": float(env.allreduce((loss * env.n).sum()) / tot)}

    def evaluate(self):
        env = self.env
        out = {"client": env.eval_clients(self.bank_p, self.bank_b)}
        cms = evaluate.own_cm(env.lay, self.bank_p, self.bank_b, env.test, env.cfg, torch.arange(env.K, device=env.dev),
                              max_rows=env.hp.get("eval_rows", 1 << 15), protos=(self.proto, self.has))
        out["client_proto"] = env.gather_cms(cms)
        return out

    def state(self):
        return {"bank_p": self.bank_p.cpu(), "bank_b": self.bank_b.cpu(), "proto": self.proto.cpu(), "has": self.has.cpu()}

    def load_state(self, s):
        self.bank_p, self.bank_b, self.proto, self.has = (s[k].to(self.env.dev) for k in ("bank_p", "bank_b", "proto", "has"))
