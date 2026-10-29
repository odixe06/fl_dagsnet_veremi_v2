"""build_npy + load_npy (V4_NPY) give the same arrays as the parquet loader, for client and fold subsets."""
import os

import numpy as np
import pytest

from core import data as D

ROOT = "/home/odixe/project/mobihet-features/data/v4-scaled"


@pytest.mark.skipif(not os.path.exists(ROOT), reason="local v4 data only")
def test_npy_matches_parquet(tmp_path):
    ids = [5, 17, 99, 1234, 4000]
    D.build_npy(ROOT, str(tmp_path), clients=ids)
    for split, folds in (("train", None), ("train", ("train",)), ("train", ("val",)), ("test", None)):
        a = D.load(ROOT, split, clients=ids[1:4], folds=folds)
        b = D.load_npy(str(tmp_path), split, clients=ids[1:4], folds=folds)
        assert (a.ids == b.ids).all() and (a.off == b.off).all()
        for k in range(len(a.ids)):                                    # same rows per client (order inside a client may differ)
            sa = np.lexsort(np.asarray(a.X[a.off[k]:a.off[k + 1]], np.float32).T)
            sb = np.lexsort(np.asarray(b.X[b.off[k]:b.off[k + 1]], np.float32).T)
            assert (np.asarray(a.X[a.off[k]:a.off[k + 1]])[sa] == np.asarray(b.X[b.off[k]:b.off[k + 1]])[sb]).all()
            assert (a.Y[a.off[k]:a.off[k + 1]][sa] == b.Y[b.off[k]:b.off[k + 1]][sb]).all()
