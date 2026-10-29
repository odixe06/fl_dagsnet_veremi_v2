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


def masked_mean(v, mask):
    """(G, b) per-row values -> (G,) mean over real rows. where, not v * mask: padding clients (no real row) have
    zero-variance BN and may overflow fp16, and inf * 0 = NaN."""
    return torch.where(mask, v, 0.0).sum(1) / mask.sum(1).clamp_min(1)


def ce_loss(logits, y, mask):
    """Per-client mean CE over real rows -> (G,)."""
    G, b, C = logits.shape
    return masked_mean(F.cross_entropy(logits.reshape(-1, C), y.reshape(-1), reduction="none").view(G, b), mask)


class Ctx:
    """What a method's loss sees at one grouped step. Rows >= G are padding clients (never active): per-client tensors a
    method brings must be padded with ctx.pad(t) so every grouped step has one compiled shape."""
    __slots__ = ("P", "B", "x", "y", "mask", "rows", "clients", "step", "flat", "G", "Gfix")

    def pad(self, t):
        return pad_rows(t, self.Gfix)


def pad_rows(t, Gfix):
    """(G, ...) -> (Gfix, ...) by repeating row 0 (padding clients are never active, their rows are never used)."""
    return t if t.shape[0] == Gfix else torch.cat([t, t[:1].expand(Gfix - t.shape[0], *t.shape[1:])])


def group_size(off, clients, hp):
    """Fixed per-step client count Gfix that local_epoch pads a group to (one compiled shape)."""
    n = off[clients + 1] - off[clients]
    return max(len(clients), hp["group_rows"] // slot(n.max().item(), hp["batch"]))


def local_epoch(lay, flat, bufs, data, clients, cfg, hp, lr, gen, loss_fn=None, grad_fn=None, opt=None, models=1):
    """Train rows `flat` (G, models*P) / `bufs` (G, models*Bn) in place for one epoch of each client's data.

    models > 1: each row holds that many models of layout `lay` back to back (e.g. MobiHet private | proxy), trained by one
    AdamW with grad-norm clip per model; ctx.P / ctx.B are then lists of per-model dicts and loss_fn returns a list of new
    BN dicts. loss_fn(ctx) -> ((G,) per-client loss to minimise, new BN) (default CE); grad_fn(ctx, grad) may edit the
    (G, models*P) gradient in place (SCAFFOLD correction, proximal terms done analytically, ...).
    """
    X, Y, off = data.X, data.Y, data.off
    G = len(clients)
    n = (off[clients + 1] - off[clients])
    b = slot(n.max().item(), hp["batch"])
    steps = int(math.ceil(n.max().item() / b))
    ctx = Ctx()
    ctx.G, ctx.Gfix = G, group_size(off, clients, hp)
    clients_p = ctx.pad(clients)
    n_p = torch.cat([n, n.new_zeros(ctx.Gfix - G)])
    flat_p, bufs_p = ctx.pad(flat).clone(), ctx.pad(bufs).clone()
    pos, valid = permutations(off[clients_p], n_p, steps, b, gen)
    opt = opt or GroupAdamW(flat_p, wd=hp["weight_decay"], clip=hp["clip"], segments=models)
    scale = hp.get("loss_scale", 2.0 ** 12)
    loss_sum = torch.zeros(ctx.Gfix, device=flat.device)
    skipped = torch.zeros(ctx.Gfix, device=flat.device)
    applied = torch.zeros(ctx.Gfix, device=flat.device)
    ctx.clients = clients_p
    # One autograd leaf per parameter tensor, sharing storage with flat_p (the optimizer updates flat_p in place, the
    # leaves see it). Leaves that are views of one (G, P) tensor would make every SliceBackward allocate and add a full
    # (G, P) zero tensor: 85 % of step time at G = 32 (profiled 29/10).
    P, Bn = lay.P, lay.B
    leaves = [{k: v.detach().requires_grad_(True) for k, v in lay.params(flat_p[:, m * P:(m + 1) * P]).items()}
              for m in range(models)]
    bviews = [lay.bufs(bufs_p[:, m * Bn:(m + 1) * Bn]) for m in range(models)]
    one = models == 1
    for s in range(steps):
        sl = slice(s * b, (s + 1) * b)
        # contiguous: a slice of the (G, steps*b) schedule has a group-dependent stride, and a stride change is a
        # recompile of the static-shape graph (8 recompiles -> silent eager fallback, Kaggle probe 29/10)
        rows, mask = pos[:, sl].contiguous(), valid[:, sl].contiguous()
        active = mask.any(1)
        ctx.P, ctx.B, ctx.flat = (leaves[0], bviews[0], flat_p) if one else (leaves, bviews, flat_p)
        ctx.x, ctx.y, ctx.mask, ctx.rows, ctx.step = X[rows], Y[rows].long(), mask, rows, s
        with torch.autocast("cuda", dtype=torch.float16, enabled=hp.get("amp", True) and X.is_cuda):
            if loss_fn is None:
                logits, newB = grouped.TRAIN(ctx.P, ctx.B, ctx.x, cfg, mask=mask, train=True)
                loss = ce_loss(logits, ctx.y, mask)
            else:
                loss, newB = loss_fn(ctx)
        loss = torch.where(active, loss, torch.zeros_like(loss))
        (loss.sum() * scale).backward()
        grad = torch.cat([lv[k].grad.reshape(ctx.Gfix, -1) for lv in leaves for k, _ in lay.p], dim=1) / scale
        for lv in leaves:
            for v in lv.values():
                v.grad = None
        if grad_fn is not None:
            grad_fn(ctx, grad)
        _, ok = opt.step(flat_p, grad, lr, active)
        if Bn:
            nb = lay.flat_bufs(newB) if one else torch.cat([lay.flat_bufs(d) for d in newB], dim=1)
            bufs_p.copy_(torch.where(ok[:, None], nb, bufs_p))
        loss_sum += loss.detach() * ok
        skipped += (active & ~ok).float()
        applied += ok.float()
    flat.copy_(flat_p[:G]); bufs.copy_(bufs_p[:G])
    return {"loss": loss_sum[:G] / torch.ceil(n.float() / b), "skipped": skipped[:G], "applied": applied[:G],
            "steps": steps}


def compile_gate(lay, data, cfg, hp, gen):
    """One real grouped step (training shape) eager vs compiled, dropout off: logits, BN stats and gradients must agree
    and be finite. Returns (ok, report). Callers fall back to eager when ok is False."""
    from core import grouped as gr
    cfg0 = dict(cfg, dropout=0.0)
    b = slot(hp["batch"], hp["batch"])
    G = max(1, hp["group_rows"] // b)
    n = data.off[1:] - data.off[:-1]
    k = torch.argsort(n, descending=True)[:G]
    start, cnt = data.off[k], n[k]
    if len(k) < G:
        start, cnt = torch.cat([start, start[:1].expand(G - len(k))]), torch.cat([cnt, cnt.new_zeros(G - len(k))])
    pos, valid = permutations(start, cnt, 1, b, gen)
    flat, bufs = lay.init(hp["seed"], G, data.X.device)
    out = []
    for f in (gr.forward, gr.TRAIN):
        w = flat.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=data.X.is_cuda):
            logits, newB = f(lay.params(w), lay.bufs(bufs), data.X[pos], cfg0, mask=valid, train=True)
            loss = ce_loss(logits, data.Y[pos].long(), valid)
        (loss.sum() * 1024).backward()
        out.append((logits.detach(), lay.flat_bufs(newB), w.grad / 1024))
    (le, be, ge), (lc, bc, gc) = out
    rep = {"logit_maxdiff": float((le - lc).abs().max()), "grad_finite": bool(torch.isfinite(gc).all()),
           "grad_rel": float(((ge - gc).norm(dim=1) / ge.norm(dim=1).clamp_min(1e-12)).max()),
           "bn_maxdiff": float((be - bc).abs().max())}
    ok = rep["grad_finite"] and rep["grad_rel"] < 0.1 and rep["logit_maxdiff"] < 0.05 * float(le.abs().max())
    return ok, rep
