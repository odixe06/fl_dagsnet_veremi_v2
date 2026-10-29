"""Confusion matrices on resident test data. Global model: every test row. Per-client models: client k's own test rows."""
import torch

from core import grouped
from core.data import N_CLS


@torch.no_grad()
def global_cm(lay, p_row, b_row, test, cfg, batch=16384):
    """One model (P,), (Bn,) on all rows of `test` -> per-client (K, 16, 16) (sum = pooled matrix)."""
    P, B = lay.params(p_row[None]), lay.bufs(b_row[None])
    K = len(test.off) - 1
    client = torch.repeat_interleave(torch.arange(K, device=test.X.device), test.off[1:] - test.off[:-1])
    cm = torch.zeros(K * N_CLS * N_CLS, dtype=torch.int64, device=test.X.device)
    for s in range(0, test.X.shape[0], batch):
        x = test.X[s:s + batch][None]
        with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
            logits, _ = grouped.EVAL(P, B, x, cfg, train=False)
        pred = logits[0].argmax(1)
        y = test.Y[s:s + batch].long()
        cm += torch.bincount((client[s:s + batch] * N_CLS + y) * N_CLS + pred, minlength=cm.numel())
    return cm.view(K, N_CLS, N_CLS).cpu().numpy()


@torch.no_grad()
def own_cm(lay, flat, bufs, test, cfg, clients, max_rows=1 << 17, protos=None, head=None):
    """Rows `flat[i]` = model of client clients[i]; each scored on its own test rows -> (len(clients), 16, 16).
    protos = (C (16, d), present (16,) bool): predict argmin_j ‖h(x) − C_j‖ over present classes (FedProto Eq. 4).
    head(P, h) -> logits replaces DAGSNet's classifier (a method's own head on the hidden layer, e.g. V2X-HtFL's ETF)."""
    dev = test.X.device
    off = test.off
    n = (off[clients + 1] - off[clients])
    out = torch.zeros(len(clients), N_CLS * N_CLS, dtype=torch.int64, device=dev)
    order = torch.argsort(n, descending=True)
    i = 0
    while i < len(clients):
        b = int(n[order[i]].item())
        b = min(b, max_rows)
        g = max(1, min(len(clients) - i, max_rows // max(b, 1)))
        idx = order[i:i + g]
        P, B = lay.params(flat[idx]), lay.bufs(bufs[idx])
        start, cnt = off[clients[idx]], n[idx]
        for s in range(0, int(cnt.max().item()), b):
            j = s + torch.arange(b, device=dev)[None]
            valid = j < cnt[:, None]
            rows = start[:, None] + torch.where(valid, j, 0)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logits, h, _ = grouped.EVAL(P, B, test.X[rows], cfg, train=False, hidden=True)
            if head is not None:
                logits = head(P, h)
            if protos is not None:
                dist = torch.cdist(h.float(), protos[0][None].expand(h.shape[0], -1, -1))
                logits = -dist.masked_fill(~protos[1], float("inf"))
            code = test.Y[rows].long() * N_CLS + logits.argmax(-1)
            code = torch.arange(g, device=dev)[:, None] * N_CLS * N_CLS + code
            out[idx] += torch.bincount(code[valid], minlength=g * N_CLS * N_CLS).view(g, -1)
        i += g
    return out.view(len(clients), N_CLS, N_CLS).cpu().numpy()


@torch.no_grad()
def class_sums(lay, flat, bufs, data, cfg, clients, max_rows=1 << 15, feat=None, d=None):
    """Eval-mode hidden-layer sums and counts per (client, class) over `data` rows of each client:
    -> (len(clients), 16, d) sums, (len(clients), 16) counts. Prototype methods' "collect prototypes" pass.
    feat(P, h) -> (G, b, d) maps the hidden layer first (a method's own embedding, e.g. V2X-HtFL's projection)."""
    dev = data.X.device
    off = data.off
    n = off[clients + 1] - off[clients]
    d = d or cfg["head_hidden"]
    S = torch.zeros(len(clients), N_CLS, d, device=dev)
    Cn = torch.zeros(len(clients), N_CLS, device=dev)
    order = torch.argsort(n, descending=True)
    i = 0
    while i < len(clients):
        b = min(int(n[order[i]].item()), max_rows)
        g = max(1, min(len(clients) - i, max_rows // max(b, 1)))
        idx = order[i:i + g]
        P, B = lay.params(flat[idx]), lay.bufs(bufs[idx])
        start, cnt = off[clients[idx]], n[idx]
        for s in range(0, int(cnt.max().item()), b):
            j = s + torch.arange(b, device=dev)[None]
            valid = j < cnt[:, None]
            rows = start[:, None] + torch.where(valid, j, 0)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                _, h, _ = grouped.EVAL(P, B, data.X[rows], cfg, train=False, hidden=True)
                if feat is not None:
                    h = feat(P, h)
            y = data.Y[rows].long()
            gi = torch.arange(g, device=dev)[:, None].expand_as(y)
            w = valid.float()
            Sg = torch.zeros(g, N_CLS, d, device=dev).index_put_((gi, y), h.float() * w[..., None], accumulate=True)
            Cg = torch.zeros(g, N_CLS, device=dev).index_put_((gi, y), w, accumulate=True)
            S[idx] += Sg; Cn[idx] += Cg
        i += g
    return S, Cn
