"""GroupAdamW == torch.optim.AdamW per active row; metrics == sklearn; per-client eval == reference model."""
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from core import evaluate, local, metrics
from core.dagsnet import DAGSNet, width
from core.data import Split
from core.flat import Layout
from core.optim import GroupAdamW


def test_group_adamw_matches_torch():
    torch.manual_seed(0)
    G, P = 3, 50
    flat = torch.randn(G, P, dtype=torch.float64)
    ref = [torch.nn.Parameter(flat[i].clone()) for i in range(G)]
    opts = [torch.optim.AdamW([p], lr=1e-2, weight_decay=1e-4) for p in ref]
    go = GroupAdamW(flat, wd=1e-4, clip=0)
    for step in range(6):
        grad = torch.randn(G, P, dtype=torch.float64)
        active = torch.tensor([True, step % 2 == 0, step < 3])
        go.step(flat, grad, 1e-2, active)
        for i in range(G):
            if active[i]:
                ref[i].grad = grad[i].clone(); opts[i].step()
    for i in range(G):
        assert (flat[i] - ref[i].detach()).abs().max() < 1e-10, ((flat[i] - ref[i].detach()).abs().max(), i)


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 16, 2000); p = np.where(rng.random(2000) < 0.7, y, rng.integers(0, 16, 2000))
    y[y == 3] = 4                                         # one class absent from y_true
    cm = np.zeros((16, 16), int); np.add.at(cm, (y, p), 1)
    m = metrics.ten(cm)
    labels = np.unique(y)
    assert np.isclose(m["f1_macro"], f1_score(y, p, labels=labels, average="macro", zero_division=0))
    assert np.isclose(m["precision_macro"], precision_score(y, p, labels=labels, average="macro", zero_division=0))
    assert np.isclose(m["recall_macro"], recall_score(y, p, labels=labels, average="macro", zero_division=0))
    assert np.isclose(m["f1_weighted"], f1_score(y, p, labels=labels, average="weighted", zero_division=0))
    assert np.isclose(m["accuracy"], (y == p).mean())


def _toy(K=4, seed=0):
    rng = np.random.default_rng(seed)
    n = rng.integers(5, 40, K)
    off = np.concatenate([[0], np.cumsum(n)])
    X = torch.randn(int(off[-1]), 77)
    Y = torch.randint(0, 16, (int(off[-1]),), dtype=torch.int8)
    return Split(X, Y, torch.from_numpy(off), torch.arange(K))


def test_eval_matches_reference():
    cfg = width(0.25)
    lay = Layout(cfg)
    test = _toy()
    p, b = lay.init(1, 4)
    p = p + 0.01 * torch.randn_like(p)                   # four different models
    cms = evaluate.own_cm(lay, p, b, test, cfg, torch.arange(4), max_rows=64)
    g = evaluate.global_cm(lay, p[0], b[0], test, cfg, batch=7)
    for k in range(4):
        m = DAGSNet(cfg); m.load_state_dict(lay.state_dict(p[k], b[k]), strict=False); m.eval()
        s, e = test.off[k], test.off[k + 1]
        pred = m(test.X[s:e]).argmax(1).numpy(); y = test.Y[s:e].long().numpy()
        ref = np.zeros((16, 16), int); np.add.at(ref, (y, pred), 1)
        assert (cms[k] == ref).all(), k
    m = DAGSNet(cfg); m.load_state_dict(lay.state_dict(p[0], b[0]), strict=False); m.eval()
    pred = m(test.X).argmax(1).numpy(); ref = np.zeros((16, 16), int); np.add.at(ref, (test.Y.long().numpy(), pred), 1)
    assert (g.sum(0) == ref).all()


def test_local_epoch_visits_every_row_once():
    cfg = width(0.25)
    lay = Layout(cfg)
    data = _toy(K=5)
    seen = torch.zeros(len(data.Y), dtype=torch.int64)

    def loss_fn(ctx):
        from core import grouped
        seen.index_add_(0, ctx.rows[ctx.mask], torch.ones(int(ctx.mask.sum()), dtype=torch.int64))
        logits, newB = grouped.forward(ctx.P, ctx.B, ctx.x, cfg, mask=ctx.mask, train=True)
        return local.ce_loss(logits, ctx.y, ctx.mask), newB

    flat, bufs = lay.init(0, 5)
    hp = {"batch": 8, "group_rows": 64, "weight_decay": 1e-4, "clip": 1.0, "amp": False}
    gen = torch.Generator().manual_seed(0)
    before = flat.clone()
    local.local_epoch(lay, flat, bufs, data, torch.arange(5), cfg, hp, 1e-3, gen, loss_fn=loss_fn)
    assert (seen == 1).all()
    assert (flat != before).any(1).all()
