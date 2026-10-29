"""AdamW on (G, P) rows with a per-client active mask, per-client step count and per-client grad-norm clip.

Elementwise identical to torch.optim.AdamW (decoupled weight decay, bias correction) for every active row; inactive rows
(client already finished its epoch, or non-finite fp16 gradient) keep params and moments untouched.
"""
import torch


class GroupAdamW:
    def __init__(self, flat, wd=1e-4, betas=(0.9, 0.999), eps=1e-8, clip=1.0):
        self.wd, (self.b1, self.b2), self.eps, self.clip = wd, betas, eps, clip
        self.m = torch.zeros_like(flat)
        self.v = torch.zeros_like(flat)
        self.t = torch.zeros(flat.shape[0], device=flat.device)

    @torch.no_grad()
    def step(self, flat, grad, lr, active):
        """flat, grad (G, P); active (G,) bool. Returns per-client grad norm before clipping."""
        finite = torch.isfinite(grad).all(1)
        active = active & finite
        grad = torch.where(finite[:, None], grad, torch.zeros_like(grad))
        norm = grad.norm(dim=1)
        if self.clip:
            grad = grad * (self.clip / (norm + 1e-6)).clamp(max=1.0)[:, None]
        a = active[:, None]
        self.t += active.float()
        t = self.t.clamp_min(1)[:, None]
        m = torch.where(a, self.b1 * self.m + (1 - self.b1) * grad, self.m)
        v = torch.where(a, self.b2 * self.v + (1 - self.b2) * grad * grad, self.v)
        self.m.copy_(m); self.v.copy_(v)
        upd = (m / (1 - self.b1 ** t)) / ((v / (1 - self.b2 ** t)).sqrt() + self.eps)
        new = flat * (1 - lr * self.wd) - lr * upd
        flat.copy_(torch.where(a, new, flat))
        return norm, active
