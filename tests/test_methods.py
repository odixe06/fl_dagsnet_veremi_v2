"""Every method runs 2 rounds on a toy CPU environment, produces finite models and per-client/global confusion matrices,
and its state round-trips through state()/load_state()."""
import importlib

import numpy as np
import pytest
import torch

from core import evaluate, local
from core.dagsnet import width
from core.data import Split
from core.flat import Layout

METHODS = ["fedavg", "fedprox", "scaffold", "mobihet", "pfedes", "fedproto", "fedtgp", "tinyproto", "nilm", "fd_ids", "afpha", "perfedskd", "lwfednids"]


class ToyEnv:
    def __init__(self, K=6, seed=0):
        rng = np.random.default_rng(seed)
        self.cfg = width(0.25)
        self.hp = {"batch": 8, "group_rows": 32, "weight_decay": 1e-4, "clip": 1.0, "amp": False, "seed": 42,
                   "lr_max": 1e-3, "lr_min": 1e-5, "rounds": 2}
        self.lay = Layout(self.cfg)
        self.dev = torch.device("cpu")
        self.world, self.rank = 1, 0

        def split(lo, hi):
            n = rng.integers(lo, hi, K)
            off = np.concatenate([[0], np.cumsum(n)])
            y = torch.from_numpy(rng.integers(0, 16, off[-1]).astype(np.int8))
            x = torch.from_numpy(rng.standard_normal((int(off[-1]), 77))).float() + 0.3 * y[:, None].float()
            return Split(x, y, torch.from_numpy(off), torch.arange(K))
        self.train, self.test = split(10, 50), split(4, 12)
        self.K = K
        self.ids = np.arange(K)
        self.all_ids = np.arange(K)
        self.n = torch.as_tensor(self.train.sizes, dtype=torch.float32)
        self.groups = local.group_schedule(self.train.sizes, list(range(K)), self.hp["batch"], self.hp["group_rows"])
        self.gen = torch.Generator().manual_seed(0)

    def allreduce(self, t):
        return t

    def gather_cms(self, cm):
        return np.asarray(cm)

    def groups_idx(self):
        for g in self.groups:
            yield torch.as_tensor(g)

    def eval_global(self, p, b):
        return evaluate.global_cm(self.lay, p, b, self.test, self.cfg)

    def eval_clients(self, bp, bb):
        return evaluate.own_cm(self.lay, bp, bb, self.test, self.cfg, torch.arange(self.K))


@pytest.mark.parametrize("name", METHODS)
def test_method_runs(name):
    env = ToyEnv()
    M = importlib.import_module(f"methods.{name}").Method(env)
    for r in (1, 2):
        info = M.round(r, 1e-3)
        assert np.isfinite(info["train_loss"])
    views = M.evaluate()
    for v, cms in views.items():
        assert cms.shape == (env.K, 16, 16)
        assert cms.sum() == len(env.test.Y), v
    s = M.state()
    M2 = importlib.import_module(f"methods.{name}").Method(ToyEnv())
    M2.load_state(s)
    for v in views:
        assert (M2.evaluate()[v] == views[v]).all()
