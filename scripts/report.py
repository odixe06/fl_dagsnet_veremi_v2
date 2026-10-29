"""Comparison table from pulled runs: every metric recomputed from the saved confusion matrices (cms_rXXX.npz), not
copied from logs.

    python scripts/report.py runs/pulls/*/ --round 50 > docs/research/results.md

Headline view per method: the evaluated model (global model; per-client models on their own test; prototype rule for
FedProto/FedTGP/TinyProto; MobiHet private). Other views are listed under it.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core import metrics  # noqa: E402

HEADLINE = {"fedproto": "client_proto", "fedtgp": "client_proto", "tinyproto": "client_proto", "mobihet": "private"}
COLS = ["accuracy", "precision_macro", "precision_micro", "precision_weighted", "recall_macro", "recall_micro",
        "recall_weighted", "f1_macro", "f1_micro", "f1_weighted"]


def fmt(x):
    return f"{x:.4f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--round", type=int, default=0, help="0 = last round present")
    a = ap.parse_args()
    print("| method | view | round | " + " | ".join(COLS) + " | client F1 mean | p10 | min | byte/client/round |")
    print("|---|---|---:|" + "---:|" * (len(COLS) + 4))
    for d in a.runs:
        cfg = json.load(open(os.path.join(d, "config.json")))
        rounds = sorted(glob.glob(os.path.join(d, "cms_r*.npz")))
        if not rounds:
            continue
        path = os.path.join(d, f"cms_r{a.round:03d}.npz") if a.round else rounds[-1]
        r = int(path[-7:-4])
        z = np.load(path)
        rows = {json.loads(l)["round"]: json.loads(l) for l in open(os.path.join(d, "rounds.jsonl"))}
        info = rows.get(r, {})
        pb = cfg.get("payload_bytes", {})
        byte = info.get("bytes_up", pb.get("up", 0)) + info.get("bytes_down", pb.get("down", 0))
        views = [v for v in z.files if v != "ids"]
        head = HEADLINE.get(cfg["method"], views[0])
        for v in [head] + [v for v in views if v != head]:
            s = metrics.summary(z[v])
            name = cfg["method"] if v == head else ""
            print(f"| {name} | {v}{' (headline)' if v == head else ''} | {r} | " + " | ".join(fmt(s[c]) for c in COLS) +
                  f" | {fmt(s['client_f1_macro_mean'])} | {fmt(s['client_f1_macro_p10'])} | {fmt(s['client_f1_macro_min'])} | "
                  + f"{int(byte):,}".replace(",", ".") + " |")


if __name__ == "__main__":
    main()
