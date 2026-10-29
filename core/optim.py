"""AdamW on (G, P) rows with a per-client active mask, per-client step count and per-client grad-norm clip.

Elementwise identical to torch.optim.AdamW (decoupled weight decay, bias correction) for every active row; inactive rows
(client already finished its epoch, or non-finite fp16 gradient) keep params and moments untouched. The update is one
function so torch.compile can fuse it (FUSED, set by compile_all): unfused it cost as much as the model's fwd + bwd.
"""
import torch


def _update(flat, grad, m, v, t, lr, active, wd, b1, b2, eps, clip, segments):
    finite = torch.isfinite(grad).all(1)
    active = active & finite
    grad = torch.where(finite[:, None], grad, torch.zeros_like(grad))
    G = grad.shape[0]
    norm = grad.view(G, segments, -1).norm(dim=2)                      # (G, models): clip each model separately
    if clip:
        grad = (grad.view(G, segments, -1) * (clip / (norm + 1e-6)).clamp(max=1.0)[:, :, None]).view(G, -1)
    a = active[:, None]
    t = t + active.double()
    tt = t.clamp_min(1)[:, None]
    bc1 = (1 - b1 ** tt).to(flat.dtype)                                # bias corrections in fp64, like torch's floats
    bc2s = (1 - b2 ** tt).sqrt().to(flat.dtype)
    m_new = torch.where(a, b1 * m + (1 - b1) * grad, m)
    v_new = torch.where(a, b2 * v + (1 - b2) * grad * grad, v)
    upd = (m_new / bc1) / (v_new.sqrt() / bc2s + eps)
    new = torch.where(a, flat * (1 - lr * wd) - lr * upd, flat)
    return new, m_new, v_new, t, norm, active


UPDATE = _update


def compile_all():
    global UPDATE
    UPDATE = torch.compile(_update, dynamic=False)


class GroupAdamW:
    def __init__(self, flat, wd=1e-4, betas=(0.9, 0.999), eps=1e-8, clip=1.0, segments=1):
        self.wd, (self.b1, self.b2), self.eps, self.clip, self.segments = wd, betas, eps, clip, segments
        self.m = torch.zeros_like(flat)
        self.v = torch.zeros_like(flat)
        self.t = torch.zeros(flat.shape[0], device=flat.device, dtype=torch.float64)

    @torch.no_grad()
    def step(self, flat, grad, lr, active):
        """flat, grad (G, P); active (G,) bool; lr float or 0-d tensor. Returns ((G, segments) grad norms, applied mask)."""
        lr = torch.as_tensor(lr, dtype=flat.dtype, device=flat.device)
        new, m, v, self.t, norm, active = UPDATE(flat, grad, self.m, self.v, self.t, lr, active,
                                                 self.wd, self.b1, self.b2, self.eps, self.clip, self.segments)
        flat.copy_(new); self.m.copy_(m); self.v.copy_(v)
        return norm, active
