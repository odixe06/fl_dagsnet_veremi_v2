"""PerFed-SKD — Singh et al., IEEE TCE 2024 (papers/perfedskd.pdf), Eq. (2)-(3), Algorithms 1-2.

Every device m keeps its local model ω_m and its personalised teacher V_m (its model saved after the previous round).
Round t: the server selects S = {m : a_m < A}, A = mean_m a_m (Alg. 1 lines 4-6; round 1: every device); selected devices
restart from the global ω^t, the others continue from ω_m (Alg. 2 lines 4-7). Every device trains E epochs on
    φ_m = CE(ω_m) + λ·KL(x(V_m) ‖ x(ω_m))          (Eq. 2; no teacher in round 1)
then saves V_m ← ω_m and reports its accuracy a_m; the server averages the selected devices' models,
ω^{t+1} = Σ_{m∈S} (n_m / n_S) ω_m (Alg. 1 line 11 averages uniformly over S; sample weights = protocol rule, owner 29/10). Evaluation: the personalised models.
Choices the paper leaves open: λ = 1 and L = KL at temperature 1 (λ and L are not given numerically); a_m = accuracy of
ω_m on the device's own training rows (eval mode) — the paper's "local model accuracy", test rows are never used;
no server pre-training on a "predefined dataset" (none exists here; θ⁰ is the shared random init).
Communication: only S downloads ω^t and uploads ω_m (logged per round).
"""
import numpy as np
import torch
import torch.nn.functional as F

from core import evaluate, grouped, local


class Method:
    name = "perfedskd"
    per_client = True

    def __init__(self, env):
        self.env = env
        self.lam = env.hp.get("lam", 1.0)
        p0, b0 = env.lay.init(env.hp["seed"], 1, env.dev)
        self.g_p, self.g_b = p0[0].clone(), b0[0].clone()
        self.bank_p, self.bank_b = p0.repeat(env.K, 1), b0.repeat(env.K, 1)
        self.acc = torch.full((env.K,), -1.0, device=env.dev)       # a_m; -1: round 1, every device selected
        self.has_teacher = False

    def payload_bytes(self):
        n = 4 * (self.env.lay.P + self.env.lay.B)
        return {"down": n, "up": n, "note": "selected devices only; see selected_frac per round"}

    def round(self, r, lr):
        env, cfg, lay = self.env, self.env.cfg, self.env.lay
        A = env.allreduce(self.acc.sum()) / env.allreduce(torch.tensor(float(env.K), device=env.dev))
        sel = self.acc < A if r > 1 else torch.ones(env.K, dtype=torch.bool, device=env.dev)
        acc_p = torch.zeros_like(self.g_p, dtype=torch.float64)
        acc_b = torch.zeros_like(self.g_b, dtype=torch.float64)
        loss = torch.zeros(env.K, device=env.dev)
        for idx in env.groups_idx():
            Gf = local.group_size(env.train.off, idx, env.hp)
            teacher = (lay.params(local.pad_rows(self.bank_p[idx], Gf).clone()),
                       lay.bufs(local.pad_rows(self.bank_b[idx], Gf).clone())) if self.has_teacher else None
            s = sel[idx][:, None]
            flat = torch.where(s, self.g_p[None], self.bank_p[idx])
            bufs = torch.where(s, self.g_b[None], self.bank_b[idx])

            def loss_fn(ctx):
                z, nb = grouped.TRAIN(ctx.P, ctx.B, ctx.x, cfg, mask=ctx.mask, train=True)
                ce = local.ce_loss(z, ctx.y, ctx.mask)
                if teacher is None:
                    return ce, nb
                with torch.no_grad():
                    zt, _ = grouped.TRAIN(teacher[0], teacher[1], ctx.x, cfg, train=False)
                    lt = F.log_softmax(zt.float(), -1)
                kl = local.masked_mean((lt.exp() * (lt - F.log_softmax(z, -1))).sum(-1), ctx.mask)
                return ce + self.lam * kl, nb

            st = local.local_epoch(lay, flat, bufs, env.train, idx, cfg, env.hp, lr, env.gen, loss_fn=loss_fn)
            self.bank_p[idx], self.bank_b[idx] = flat, bufs                  # ω_m, and V_m for the next round
            w = s.double() * env.n[idx][:, None].double()
            acc_p += (w * flat.double()).sum(0)
            acc_b += (w * bufs.double()).sum(0)
            loss[idx] = st["loss"]
        nsel = env.allreduce(sel.sum().double())
        wsel = env.allreduce((sel.double() * env.n.double()).sum())
        if nsel > 0:
            self.g_p = (env.allreduce(acc_p) / wsel).float()
            self.g_b = (env.allreduce(acc_b) / wsel).float()
        cms = evaluate.own_cm(lay, self.bank_p, self.bank_b, env.train, cfg, torch.arange(env.K, device=env.dev),
                              max_rows=env.hp.get("eval_rows", 1 << 15))
        self.acc = torch.as_tensor(np.trace(cms, axis1=1, axis2=2) / cms.sum((1, 2)).clip(min=1), device=env.dev,
                                   dtype=torch.float32)
        self.has_teacher = True
        Kt = env.allreduce(torch.tensor(float(env.K), device=env.dev))
        return {"train_loss": float(env.allreduce((loss * env.n).sum()) / env.allreduce(env.n.sum().clone())),
                "selected_frac": float(nsel / Kt), "local_acc_mean": float(env.allreduce(self.acc.sum()) / Kt)}

    def evaluate(self):
        return {"client": self.env.eval_clients(self.bank_p, self.bank_b)}

    def state(self):
        return {"g_p": self.g_p.cpu(), "g_b": self.g_b.cpu(), "bank_p": self.bank_p.cpu(), "bank_b": self.bank_b.cpu(),
                "acc": self.acc.cpu(), "has_teacher": self.has_teacher}

    def load_state(self, s):
        for k in ("g_p", "g_b", "bank_p", "bank_b", "acc"):
            setattr(self, k, s[k].to(self.env.dev))
        self.has_teacher = s["has_teacher"]
