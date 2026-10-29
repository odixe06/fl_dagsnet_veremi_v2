"""FedProx — Li et al., MLSys 2020 (papers/fedprox.pdf), Algorithm 2.

FedAvg with the local objective h_k(w; w^t) = F_k(w) + (μ/2)‖w − w^t‖²: its gradient adds μ(w − w^t), applied here to the
gradient before the shared AdamW step (protocol optimizer; the paper's analysis uses SGD). μ = 0.01 from the paper's
candidate set {0.001, 0.01, 0.1, 1} (same value as the earlier rebuild B); override with --set mu=<v>. Proximal term on
learnable parameters only (BN running statistics are not optimised). All clients every round ⇒ no straggler variant.
"""
from methods import fedavg


class Method(fedavg.Method):
    name = "fedprox"

    def __init__(self, env):
        super().__init__(env)
        self.mu = env.hp.get("mu", 0.01)

    def grad_fn(self, ctx, grad):
        grad += self.mu * (ctx.flat - self.p[None])

    def round(self, r, lr):
        return self._round(lr, grad_fn=self.grad_fn)
