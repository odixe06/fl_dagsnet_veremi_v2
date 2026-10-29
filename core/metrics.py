"""The 10 metrics from a 16x16 confusion matrix (rows = true, cols = predicted), plus per-class F1.

Macro averages run over classes present in y_true (sklearn's labels=present convention): a client whose test lacks a
class is not scored on it. Pooled metrics (one matrix summed over every client's own test) cover all 16 classes.
"""
import numpy as np


def ten(cm):
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    sup = cm.sum(1)
    pred = cm.sum(0)
    present = sup > 0
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    n = cm.sum()
    w = sup / n
    acc = tp.sum() / n
    return {
        "accuracy": acc,
        "precision_macro": prec[present].mean(), "precision_micro": acc, "precision_weighted": (prec * w).sum(),
        "recall_macro": rec[present].mean(), "recall_micro": acc, "recall_weighted": (rec * w).sum(),
        "f1_macro": f1[present].mean(), "f1_micro": acc, "f1_weighted": (f1 * w).sum(),
        "f1_per_class": f1.tolist(), "support": sup.astype(int).tolist(),
    }


TEN = ["accuracy", "precision_macro", "precision_micro", "precision_weighted", "recall_macro", "recall_micro",
       "recall_weighted", "f1_macro", "f1_micro", "f1_weighted"]


def summary(cms):
    """cms (K, 16, 16) per-client confusion on own test -> pooled 10 metrics + client macro-F1 distribution."""
    cms = np.asarray(cms)
    pooled = ten(cms.sum(0))
    f1c = np.array([ten(c)["f1_macro"] for c in cms])
    accc = np.array([ten(c)["accuracy"] for c in cms])
    pooled.update({"client_f1_macro_mean": f1c.mean(), "client_f1_macro_p10": np.percentile(f1c, 10),
                   "client_f1_macro_median": np.median(f1c), "client_f1_macro_min": f1c.min(),
                   "client_accuracy_mean": accc.mean()})
    return pooled
