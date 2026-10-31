"""P1a: can the frozen 77-feature representation identify benign vs timeDelay on train -> val?"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
import torch
from torch import nn


SEED = 42
LABELS = (0, 13)


def metrics(y, pred):
    cm = np.zeros((2, 2), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(1)
    predicted = cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(2), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(2), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(2), where=precision + recall > 0)
    return {"confusion": cm.tolist(), "accuracy": float(tp.sum() / cm.sum()),
            "precision_13": float(precision[1]), "recall_13": float(recall[1]),
            "f1_13": float(f1[1]), "f1_benign": float(f1[0]), "f1_macro": float(f1.mean())}


def sample_balanced(indices, y, per_class, rng):
    out = []
    for label in (0, 1):
        rows = indices[y[indices] == label]
        if len(rows) > per_class:
            rows = rng.choice(rows, per_class, replace=False)
        out.append(rows)
    return np.concatenate(out)


def row_keys(x):
    x = np.ascontiguousarray(x)
    return x.view(np.dtype((np.void, x.dtype.itemsize * x.shape[1]))).ravel()


def collision_rate(a, b):
    ua = np.unique(row_keys(a)); ub = np.unique(row_keys(b))
    common = np.intersect1d(ua, ub, assume_unique=True)
    return {"unique_a": int(len(ua)), "unique_b": int(len(ub)), "shared_signatures": int(len(common)),
            "fraction_a": float(len(common) / max(1, len(ua))), "fraction_b": float(len(common) / max(1, len(ub)))}


def dataset_fingerprint(files, root):
    h = hashlib.sha256()
    rows = 0
    for path in files:
        pf = pq.ParquetFile(path)
        md = pf.metadata
        rows += md.num_rows
        parts = [str(path.relative_to(root)), path.stat().st_size, md.num_rows, md.num_row_groups, md.num_columns]
        for group in range(md.num_row_groups):
            rg = md.row_group(group)
            for col in range(rg.num_columns):
                c = rg.column(col)
                parts.append(c.total_compressed_size)
        h.update(json.dumps(parts, separators=(",", ":")).encode())
    return {"sha256_footer": h.hexdigest(), "files": len(files), "rows_all_labels": rows}


def load_binary(root, features):
    files = sorted((root / "train").rglob("*.parquet"))
    assert files
    xs, ys, folds, clients = [], [], [], []
    for i, path in enumerate(files, 1):
        table = pq.ParquetFile(path).read(columns=["label", "fold", "client_id", *features])
        label = table.column("label").to_numpy(zero_copy_only=False)
        keep = np.isin(label, LABELS)
        if keep.any():
            xs.append(np.stack([table.column(name).to_numpy(zero_copy_only=False).astype(np.float16)
                                for name in features], axis=1)[keep])
            ys.append((label[keep] == 13).astype(np.int8))
            folds.append((table.column("fold").to_numpy(zero_copy_only=False)[keep] == "val").astype(np.int8))
            clients.append(table.column("client_id").to_numpy(zero_copy_only=False)[keep].astype(np.int32))
        if i % 50 == 0:
            print(f"loaded {i}/{len(files)} files", flush=True)
    x = np.concatenate(xs); y = np.concatenate(ys); fold = np.concatenate(folds); client = np.concatenate(clients)
    assert x.shape[1] == 77 and np.isfinite(x).all() and np.abs(x).max() < 4
    assert set(np.unique(fold)) == {0, 1} and set(np.unique(y)) == {0, 1}
    return x, y, fold, client, files


def predict_hgb(model, x, chunk=250_000):
    return np.concatenate([(model.predict_proba(x[i:i + chunk])[:, 1] >= 0.5).astype(np.int8)
                           for i in range(0, len(x), chunk)])


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(77, 256), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp(x, y, train_rows, val_rows, device, epochs=10, batch=8192):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    xt = torch.from_numpy(x[train_rows].astype(np.float32)).to(device)
    yt = torch.from_numpy(y[train_rows].astype(np.float32)).to(device)
    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    gen = torch.Generator(device=device).manual_seed(SEED)
    for epoch in range(epochs):
        model.train(); order = torch.randperm(len(xt), generator=gen, device=device); total = 0.0
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = nn.functional.binary_cross_entropy_with_logits(model(xt[idx]), yt[idx])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += float(loss) * len(idx)
        print(f"MLP epoch {epoch + 1}/{epochs} loss={total / len(order):.6f}", flush=True)
    model.eval(); pred = []
    with torch.no_grad():
        for start in range(0, len(val_rows), 131072):
            xv = torch.from_numpy(x[val_rows[start:start + 131072]].astype(np.float32)).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                pred.append((model(xv).float().sigmoid() >= 0.5).byte().cpu().numpy())
    return model.cpu().state_dict(), np.concatenate(pred)


def client_distribution(y, pred, client):
    values = []
    order = np.argsort(client, kind="stable")
    y, pred, client = y[order], pred[order], client[order]
    edges = np.concatenate([[0], np.flatnonzero(client[1:] != client[:-1]) + 1, [len(client)]])
    for start, stop in zip(edges[:-1], edges[1:]):
        if y[start:stop].sum():
            values.append(metrics(y[start:stop], pred[start:stop])["f1_13"])
    values = np.asarray(values)
    return {"clients_with_13": int(len(values)), "mean_f1_13": float(values.mean()),
            "p10_f1_13": float(np.percentile(values, 10)), "median_f1_13": float(np.median(values)),
            "min_f1_13": float(values.min())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    features = json.loads(Path(args.features).read_text())["inputs"]
    assert len(features) == 77
    assert metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))["confusion"] == [[1, 1], [0, 2]]
    t0 = time.monotonic()
    x, y, fold, client, files = load_binary(Path(args.root), features)
    train = np.flatnonzero(fold == 0); val = np.flatnonzero(fold == 1)
    rng = np.random.default_rng(SEED)
    report = {"protocol": {"fit_fold": "train", "eval_fold": "val", "labels": [0, 13],
                            "official_test_read": False, "threshold": 0.5, "seed": SEED},
              "dataset": dataset_fingerprint(files, Path(args.root)),
              "counts": {"train_benign": int((y[train] == 0).sum()), "train_13": int((y[train] == 1).sum()),
                         "val_benign": int((y[val] == 0).sum()), "val_13": int((y[val] == 1).sum())}}

    dup_rows = sample_balanced(train, y, 200_000, rng)
    a, b = x[dup_rows[y[dup_rows] == 0]], x[dup_rows[y[dup_rows] == 1]]
    report["duplicates"] = {"exact_fp16": collision_rate(a, b),
                            "rounded_0p1": collision_rate(np.round(a.astype(np.float32), 1),
                                                          np.round(b.astype(np.float32), 1)),
                            "sample_per_class": int(min(len(a), len(b)))}
    hgb_rows = sample_balanced(train, y, 300_000, rng)
    hgb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=63,
                                         min_samples_leaf=100, l2_regularization=1e-3, random_state=SEED)
    hgb.fit(x[hgb_rows].astype(np.float32), y[hgb_rows])
    hgb_pred = predict_hgb(hgb, x[val].astype(np.float32, copy=False))
    report["hgb"] = metrics(y[val], hgb_pred)
    report["hgb"]["client"] = client_distribution(y[val], hgb_pred, client[val])
    print("HGB", json.dumps(report["hgb"]), flush=True)

    mlp_rows = sample_balanced(train, y, 1_000_000, rng)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    state, mlp_pred = train_mlp(x, y, mlp_rows, val, device)
    report["mlp"] = metrics(y[val], mlp_pred)
    report["mlp"]["client"] = client_distribution(y[val], mlp_pred, client[val])
    print("MLP", json.dumps(report["mlp"]), flush=True)

    best = max(report["hgb"]["f1_13"], report["mlp"]["f1_13"])
    if best >= 0.50:
        decision = "REPRESENTATION_SUFFICIENT_METHOD_BRANCH"
    elif best < 0.30:
        decision = "REPRESENTATION_INSUFFICIENT_RUN_RAW_P1B"
    else:
        decision = "INCONCLUSIVE_RUN_RAW_P1B"
    report["gate"] = {"best_val_f1_13": best, "decision": decision,
                      "sufficient_threshold": 0.50, "insufficient_threshold": 0.30}
    report["elapsed_seconds"] = time.monotonic() - t0
    torch.save(state, out / "mlp_state.pt")
    np.savez_compressed(out / "val_predictions.npz", row=val, y=y[val], client=client[val],
                        hgb=hgb_pred, mlp=mlp_pred)
    (out / "p1_report.json").write_text(json.dumps(report, indent=2))
    print("P1 GATE", json.dumps(report["gate"]), flush=True)


if __name__ == "__main__":
    main()
