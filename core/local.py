"""One local epoch for a group of clients at once (every client sees each of its rows exactly once).

Data is resident on the GPU, sorted by client (core/data.py): rows of client k are X[off[k]:off[k+1]]. A group of G clients
of similar size runs S = ceil(max n_k / batch) grouped steps; at step s client g takes rows [s*b, (s+1)*b) of its own
permutation, padded and masked to b. A client whose rows are exhausted is inactive: no forward contribution, no optimizer
step, no BN update. The method supplies the loss (and may edit gradients), so FedAvg, FedProx, SCAFFOLD, KD and prototype
methods share this loop and its speed.
"""
import math

import torch
import torch.nn.functional as F

from core import grouped
from core.optim import GroupAdamW


def slot(n_max, batch):
    """Rows per client per grouped step: the batch, or the whole client (rounded up to 8) when it is smaller than one batch.
    Same math either way (a client smaller than the batch takes all its rows in one step); only padding shrinks."""
    return min(batch, -(-int(n_max) // 8) * 8)


def group_schedule(sizes, clients, batch, max_rows):
    """Split `clients` (sorted by size, largest first) into groups with G * slot <= max_rows rows per grouped step."""
    order = sorted(clients, key=lambda k: -sizes[k])
    groups, i = [], 0
    while i < len(order):
        g = max(1, max_rows // slot(sizes[order[i]], batch))
        groups.append(order[i:i + g]); i += g
    return groups


def permutations(start, n, steps, batch, gen):
    """start, n (G,) -> pos (G, steps*batch) absolute row index, valid (G, steps*batch) bool."""
    dev = start.device
    width = steps * batch
    keys = torch.rand(len(n), width, device=dev, generator=gen)
    j = torch.arange(width, device=dev)[None]
    valid = j < n[:, None]
    keys = torch.where(valid, keys, 2.0)                                 # padding sorts last
    perm = keys.argsort(dim=1)                                           # (G, width): a permutation of 0..n-1, then pads
    pos = start[:, None] + torch.where(valid, perm, 0)
    return pos, valid


def ce_loss(logits, y, mask):
    """Per-client mean CE over real rows -> (G,)."""
    G, b, C = logits.shape
    l = F.cross_entropy(logits.reshape(-1, C), y.reshape(-1), reduction="none").view(G, b)
    return (l * mask).sum(1) / mask.sum(1).clamp_min(1)


class Ctx:
    """What a method's loss sees at one grouped step. Rows >= G are padding clients (never active): per-client tensors a
    method brings must be padded with ctx.pad(t) so every grouped step has one compiled shape."""
    __slots__ = ("P", "B", "x", "y", "mask", "rows", "clients", "step", "flat", "G", "Gfix")

    def pad(self, t):
        return t if t.shape[0] == self.Gfix else torch.cat([t, t[:1].expand(self.Gfix - t.shape[0], *t.shape[1:])])


def local_epoch(lay, flat, bufs, data, clients, cfg, hp, lr, gen, loss_fn=None, grad_fn=None, opt=None):
    """Train rows `flat` (G, P) / `bufs` (G, Bn) in place for one epoch of each client's data.

    loss_fn(ctx) -> (G,) per-client loss to minimise (default CE); grad_fn(ctx, grad) may edit (G, P) gradients in place
    (SCAFFOLD correction, proximal terms done analytically, ...). Returns dict of per-client stats.
    """
    X, Y, off = data.X, data.Y, data.off
    G = len(clients)
    n = (off[clients + 1] - off[clients])
    b = slot(n.max().item(), hp["batch"])
    steps = int(math.ceil(n.max().item() / b))
    ctx = Ctx()
    ctx.G, ctx.Gfix = G, max(G, hp["group_rows"] // b)
    clients_p = ctx.pad(clients)
    n_p = torch.cat([n, n.new_zeros(ctx.Gfix - G)])
    flat_p, bufs_p = ctx.pad(flat).clone(), ctx.pad(bufs).clone()
    pos, valid = permutations(off[clients_p], n_p, steps, b, gen)
    opt = opt or GroupAdamW(flat_p, wd=hp["weight_decay"], clip=hp["clip"])
    scale = hp.get("loss_scale", 2.0 ** 12)
    loss_sum = torch.zeros(ctx.Gfix, device=flat.device)
    skipped = torch.zeros(ctx.Gfix, device=flat.device)
    ctx.clients = clients_p
    for s in range(steps):
        sl = slice(s * b, (s + 1) * b)
        rows, mask = pos[:, sl], valid[:, sl]
        active = mask.any(1)
        w = flat_p.detach().requires_grad_(True)
        ctx.P, ctx.B, ctx.flat = lay.params(w), lay.bufs(bufs_p), w
        ctx.x, ctx.y, ctx.mask, ctx.rows, ctx.step = X[rows], Y[rows].long(), mask, rows, s
        with torch.autocast("cuda", dtype=torch.float16, enabled=hp.get("amp", True) and X.is_cuda):
            if loss_fn is None:
                logits, newB = grouped.TRAIN(ctx.P, ctx.B, ctx.x, cfg, mask=mask, train=True)
                loss = ce_loss(logits, ctx.y, mask)
            else:
                loss, newB = loss_fn(ctx)
        loss = torch.where(active, loss, torch.zeros_like(loss))
        (loss.sum() * scale).backward()
        grad = w.grad / scale
        if grad_fn is not None:
            grad_fn(ctx, grad)
        _, ok = opt.step(flat_p, grad, lr, active)
        nb = lay.flat_bufs(newB)
        bufs_p.copy_(torch.where(ok[:, None], nb, bufs_p))
        loss_sum += loss.detach() * ok
        skipped += (active & ~ok).float()
    flat.copy_(flat_p[:G]); bufs.copy_(bufs_p[:G])
    return {"loss": loss_sum[:G] / torch.ceil(n.float() / b), "skipped": skipped[:G], "steps": steps}
