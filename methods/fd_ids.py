"""FD-IDS — Peng et al., Sensors 2025 (papers/fd_ids.pdf), §3.5, Eq. (2)-(6), Algorithm 1.

FedProx + round-wise knowledge distillation with the received global model as teacher. Client objective (Eq. 6):
    λ·CE(z_s, y) + (1 − λ)·T²·KL(softmax(z_t/T) ‖ softmax(z_s/T)) + β·(μ/2)‖w − w_G^t‖²
with λ = 0.5, T = 3, β = 0.1, μ = 0.01 (paper). Teacher = w_G^t frozen (eval mode). Server: w_G^{t+1} = Σ (n_k/n) w_k (Eq. 2).
Evaluated model: the global model. The paper's DNN backbone is replaced by DAGSNet (comparison protocol).
"""
import torch
import torch.nn.functional as F

from core import grouped, local
from methods import fedavg


class Method(fedavg.Method):
    name = "fd_ids"

    def __init__(self, env):
        super().__init__(env)
        hp = env.hp
        self.lam, self.T, self.beta, self.mu = hp.get("lam", 0.5), hp.get("T", 3.0), hp.get("beta", 0.1), hp.get("mu", 0.01)
        self._teacher = None

    def _loss(self, ctx):
        cfg = self.env.cfg
        if self._teacher is None or self._teacher[0] != ctx.Gfix:
            lay = self.env.lay
            self._teacher = (ctx.Gfix, lay.params(self.p[None].repeat(ctx.Gfix, 1)), lay.bufs(self.b[None].repeat(ctx.Gfix, 1)))
        _, Pt, Bt = self._teacher
        z, nb = grouped.TRAIN(ctx.P, ctx.B, ctx.x, cfg, mask=ctx.mask, train=True)
        with torch.no_grad():
            zt, _ = grouped.TRAIN(Pt, Bt, ctx.x, cfg, train=False)
        T, m = self.T, ctx.mask.float()
        lt, ls = F.log_softmax(zt.float() / T, -1), F.log_softmax(z / T, -1)
        soft = ((lt.exp() * (lt - ls)).sum(-1) * m).sum(1) / m.sum(1).clamp_min(1) * T * T
        return self.lam * local.ce_loss(z, ctx.y, ctx.mask) + (1 - self.lam) * soft, nb

    def _grad(self, ctx, grad):
        grad += self.beta * self.mu * (ctx.flat - self.p[None])        # ∇ β·(μ/2)‖w − w_G‖²

    def round(self, r, lr):
        self._teacher = None                                          # w_G^t changes every round
        return self._round(lr, loss_fn=self._loss, grad_fn=self._grad)
