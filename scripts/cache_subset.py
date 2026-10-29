"""Cache a fixed random subset of clients (train + test) for local probes: python scripts/cache_subset.py N seed."""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from core import data

n, seed = int(sys.argv[1]), int(sys.argv[2])
ids = np.sort(np.random.default_rng(seed).choice(data.N_CLIENTS, n, replace=False))
root = data.find_root()
for split in ("train", "test"):
    t = time.time()
    s = data.load(root, split, clients=ids.tolist(), cache="cache")
    print(split, s.X.shape, len(s.ids), f"{time.time()-t:.0f}s", flush=True)
