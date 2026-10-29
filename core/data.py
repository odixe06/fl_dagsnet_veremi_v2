"""v4 parquet -> resident arrays sorted by client: X (N, 77) fp16, Y (N,) int8, off (K+1,), ids (K,) global client_id.

Two passes over the 410 files of a split (pass 1: client_id only -> offsets; pass 2: scatter each file's rows into a
preallocated array) so peak RAM is one copy of X. Within a client, rows keep file order (not time order): training
shuffles and evaluation sums confusion matrices, so neither depends on it. Train split = fold train + val (owner 29/10).
"""
import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = json.load(open(os.path.join(HERE, "..", "knowledge", "input_order.json")))["inputs"]
N_IN, N_CLS, N_CLIENTS = 77, 16, 5343


def find_root(start="/kaggle/input"):
    """Dataset root = directory holding train/ test/ label_mapping.json (Kaggle mount paths vary)."""
    for r in [os.environ.get("V4_ROOT", ""), "/home/odixe/project/mobihet-features/data/v4-scaled"]:
        if r and os.path.exists(os.path.join(r, "label_mapping.json")):
            return r
    for p in glob.glob(os.path.join(start, "**", "label_mapping.json"), recursive=True):
        d = os.path.dirname(p)
        if os.path.isdir(os.path.join(d, "train")) and os.path.isdir(os.path.join(d, "test")):
            return d
    raise FileNotFoundError("v4 root (label_mapping.json + train/ + test/) not found")


@dataclass
class Split:
    X: object
    Y: object
    off: object
    ids: object

    @property
    def sizes(self):
        o = self.off.cpu().numpy() if torch.is_tensor(self.off) else self.off
        return np.diff(o)

    def to(self, device):
        t = lambda a: a.to(device) if torch.is_tensor(a) else torch.from_numpy(np.ascontiguousarray(a)).to(device)
        return Split(t(self.X), t(self.Y), t(self.off), t(self.ids))


def files(root, split):
    fs = sorted(glob.glob(os.path.join(root, split, "**", "*.parquet"), recursive=True))
    assert fs, (root, split)
    return fs


def load(root, split, clients=None, cache=None):
    """split in {'train', 'test'}; clients = iterable of global client_id to keep (None = all)."""
    tag = "all" if clients is None else f"sub{len(set(clients))}_{abs(hash(tuple(sorted(set(clients))))) % 10**8}"
    if cache:
        base = os.path.join(cache, f"{split}_{tag}")
        if os.path.exists(base + ".done"):
            return Split(np.load(base + "_X.npy", mmap_mode="r"), np.load(base + "_Y.npy"),
                         np.load(base + "_off.npy"), np.load(base + "_ids.npy"))
    fs = files(root, split)
    keep = np.ones(N_CLIENTS, bool) if clients is None else np.isin(np.arange(N_CLIENTS), list(clients))
    cid = [pq.read_table(f, columns=["client_id"]).column("client_id").to_numpy() for f in fs]
    counts = np.zeros(N_CLIENTS, np.int64)
    for c in cid:
        counts += np.bincount(c, minlength=N_CLIENTS)
    counts[~keep] = 0
    ids = np.nonzero(counts)[0]
    off_all = np.concatenate([[0], np.cumsum(counts)])
    N = int(off_all[-1])
    X = np.empty((N, N_IN), np.float16)
    Y = np.empty(N, np.int8)
    cur = off_all[:-1].copy()
    for f, c in zip(fs, cid):
        sel = keep[c]
        if not sel.any():
            continue
        t = pq.read_table(f, columns=ORDER + ["label"])
        x = np.stack([t.column(k).to_numpy().astype(np.float16) for k in ORDER], axis=1)[sel]
        y = t.column("label").to_numpy()[sel]
        c = c[sel]
        o = np.argsort(c, kind="stable")
        c, x, y = c[o], x[o], y[o]
        u, first, cnt = np.unique(c, return_index=True, return_counts=True)
        for k, i0, m in zip(u, first, cnt):
            X[cur[k]:cur[k] + m] = x[i0:i0 + m]
            Y[cur[k]:cur[k] + m] = y[i0:i0 + m]
            cur[k] += m
    assert (cur == off_all[1:]).all()
    off = np.concatenate([[0], np.cumsum(counts[ids])])
    assert np.isfinite(X).all() and np.abs(X).max() < 4.0
    out = Split(X, Y, off, ids)
    if cache:
        os.makedirs(cache, exist_ok=True)
        for k in ("X", "Y", "off", "ids"):
            np.save(f"{base}_{k}.npy", getattr(out, k))
        open(base + ".done", "w").write("ok")
    return out
