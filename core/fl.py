"""FL environment shared by every method: data of this rank's clients on its GPU, layout, schedule, collectives.

One process per GPU (torchrun, W = 1 or 2). Rank r owns a fixed half of the clients (alternating by train size, so both
ranks get the same work); a process group exists only to sum aggregation payloads and gather confusion matrices.
Every rank keeps an identical copy of server state (it applies the same all-reduced sums).
"""
import math
import os

import numpy as np
import torch
import torch.distributed as dist

from core import data as D, evaluate, local
from core.flat import Layout


def cosine_lr(r, R, lr_max, lr_min):
    """Round r in 1..R: lr_max at r = 1, cosine to lr_min at r = R."""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (r - 1) / max(1, R - 1)))


class Env:
    def __init__(self, cfg, hp, clients=None, cache=None, root=None):
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        if self.world > 1 and not dist.is_initialized():
            dist.init_process_group("nccl")
        self.dev = torch.device("cuda", self.rank % max(1, torch.cuda.device_count())) if torch.cuda.is_available() \
            else torch.device("cpu")
        if self.dev.type == "cuda":
            torch.cuda.set_device(self.dev)
        self.cfg, self.hp, self.lay = cfg, hp, Layout(cfg)
        root = root or D.find_root()
        all_ids = np.arange(D.N_CLIENTS) if clients is None else np.sort(np.asarray(clients))
        sizes = self._sizes(root, cache)
        order = all_ids[np.argsort(-sizes[all_ids], kind="stable")]
        self.all_ids = all_ids
        self.n_all = sizes[all_ids]                                      # train rows, global order of all_ids
        mine = np.sort(order[self.rank::self.world])
        self.train = D.load(root, "train", mine.tolist(), cache).to(self.dev)
        self.test = D.load(root, "test", mine.tolist(), cache).to(self.dev)
        assert (self.train.ids == self.test.ids).all()
        self.ids = self.train.ids.cpu().numpy()                          # global client_id of local index
        self.K = len(self.ids)
        self.n = torch.as_tensor(self.train.sizes, device=self.dev, dtype=torch.float32)
        self.groups = local.group_schedule(self.train.sizes, list(range(self.K)), hp["batch"], hp["group_rows"])
        self.gen = torch.Generator(device=self.dev).manual_seed(hp["seed"] * 1000 + self.rank)

    @staticmethod
    def _sizes(root, cache):
        npy = os.environ.get("V4_NPY")
        if npy and os.path.exists(os.path.join(npy, "done")):
            n = np.zeros(D.N_CLIENTS, np.int64)
            n[np.load(os.path.join(npy, "train_ids.npy"))] = np.diff(np.load(os.path.join(npy, "train_off.npy")))
            return n
        p = os.path.join(cache or ".", "train_sizes.npy")
        if os.path.exists(p):
            return np.load(p)
        import pyarrow.parquet as pq
        n = np.zeros(D.N_CLIENTS, np.int64)
        for f in D.files(root, "train"):
            n += np.bincount(pq.ParquetFile(f).read(columns=["client_id"]).column("client_id").to_numpy(), minlength=D.N_CLIENTS)
        if cache:
            os.makedirs(cache, exist_ok=True); np.save(p, n)
        return n

    def allreduce(self, t):
        if self.world > 1:
            dist.all_reduce(t)
        return t

    def gather_cms(self, cm):
        """Local (K_r, 16, 16) -> (len(all_ids), 16, 16) in all_ids order, on every rank."""
        full = torch.zeros(len(self.all_ids), D.N_CLS, D.N_CLS, dtype=torch.int64, device=self.dev)
        pos = np.searchsorted(self.all_ids, self.ids)
        full[torch.as_tensor(pos, device=self.dev)] = torch.as_tensor(cm, device=self.dev)
        return self.allreduce(full).cpu().numpy()

    def groups_idx(self):
        for g in self.groups:
            yield torch.as_tensor(g, device=self.dev)

    def eval_global(self, p, b):
        return self.gather_cms(evaluate.global_cm(self.lay, p, b, self.test, self.cfg))

    def eval_clients(self, bank_p, bank_b):
        """bank rows indexed by local client index."""
        cms = evaluate.own_cm(self.lay, bank_p, bank_b, self.test, self.cfg,
                              torch.arange(self.K, device=self.dev), max_rows=self.hp.get("eval_rows", 1 << 15))
        return self.gather_cms(cms)
