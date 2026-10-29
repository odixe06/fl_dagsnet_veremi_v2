"""FedAvg — McMahan et al., AISTATS 2017 (papers/fedavg.pdf), Algorithm 1.

Server broadcasts θ^t; every client runs E local epochs from θ^t (here E = 1, all K clients, docs/research/protocol.md);
θ^{t+1} = Σ_k (n_k / n) θ_k^{t+1}. BatchNorm running statistics are averaged with the same weights (state_dict FedAvg).
"""
import torch

from core import local


class Method:
    name = "fedavg"
    per_client = False

    def __init__(self, env):
        self.env = env
        self.p, self.b = env.lay.init(env.hp["seed"], 1, env.dev)
        self.p, self.b = self.p[0], self.b[0]

    def payload_bytes(self):
        """Per participating client per round: θ down + θ up (fp32), BN buffers included."""
        n = 4 * (self.env.lay.P + self.env.lay.B)
        return {"down": n, "up": n}

    def round(self, r, lr):
        return self._round(lr)

    def _round(self, lr, loss_fn=None, grad_fn=None, after=None):
        """Broadcast θ^t, one local epoch per client, sample-weighted average. after(idx, flat, bufs) sees each group's
        trained rows before they are averaged (control variates, statistics, ...)."""
        env = self.env
        acc_p = torch.zeros_like(self.p)
        acc_b = torch.zeros_like(self.b)
        loss = torch.zeros(env.K, device=env.dev)
        steps = torch.zeros(2, device=env.dev)                       # applied, skipped (non-finite fp16 gradient)
        for idx in env.groups_idx():
            flat = self.p.repeat(len(idx), 1)
            bufs = self.b.repeat(len(idx), 1)
            self.idx = idx
            st = local.local_epoch(env.lay, flat, bufs, env.train, idx, env.cfg, env.hp, lr, env.gen,
                                   loss_fn=loss_fn, grad_fn=grad_fn)
            if after is not None:
                after(idx, flat, bufs, st)
            steps += torch.stack([st["applied"].sum(), st["skipped"].sum()])
            w = env.n[idx][:, None]
            acc_p += (w * flat).sum(0)
            acc_b += (w * bufs).sum(0)
            loss[idx] = st["loss"]
        tot = env.allreduce(env.n.sum().clone())
        self.p = env.allreduce(acc_p) / tot
        self.b = env.allreduce(acc_b) / tot
        steps = env.allreduce(steps)
        return {"train_loss": float((env.allreduce((loss * env.n).sum()) / tot).item()),
                "skipped_frac": float(steps[1] / steps.sum().clamp_min(1))}

    def evaluate(self):
        return {"global": self.env.eval_global(self.p, self.b)}

    def state(self):
        return {"p": self.p.cpu(), "b": self.b.cpu()}

    def load_state(self, s):
        self.p, self.b = s["p"].to(self.env.dev), s["b"].to(self.env.dev)
