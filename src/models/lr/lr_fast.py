import os
import json
import argparse
import time

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.utils.class_weight import compute_class_weight


def _read_pairs(jsonl_path: str):
    with open(jsonl_path, "r", encoding="utf8") as f:
        return [json.loads(line) for line in f]


def load_mmap(vec_root: str):
    """
    vec_root/
        vectors.npy   (memmap, float32, [N, D])
        ids.json      (id -> row)
    """
    X = np.load(os.path.join(vec_root, "vectors.npy"), mmap_mode="r")
    with open(os.path.join(vec_root, "ids.json"), "r", encoding="utf8") as f:
        id2row = json.load(f)
    return X, id2row


def iter_Xy_batches_mmap(
    pairs,
    X_all,
    id2row,
    batch_size=8192,
):
    idx1, idx2, yb = [], [], []

    for p in pairs:
        r1 = id2row.get(p["id1"])
        r2 = id2row.get(p["id2"])
        if r1 is None or r2 is None:
            continue

        idx1.append(r1)
        idx2.append(r2)
        yb.append(p["label"])

        if len(idx1) >= batch_size:
            v1 = X_all[idx1]
            v2 = X_all[idx2]
            Xb = np.abs(v1 - v2)
            yield Xb, np.asarray(yb, dtype=np.int32)
            idx1, idx2, yb = [], [], []

    if idx1:
        v1 = X_all[idx1]
        v2 = X_all[idx2]
        Xb = np.abs(v1 - v2)
        yield Xb, np.asarray(yb, dtype=np.int32)


def train_lr_streaming(
    method: str,
    variant: str,
    epochs: int = 2,
    batch_size: int = 8192,
):
    vec_root = f"vectors/{method}_{variant}/train_mmap"
    pairs = _read_pairs("data/train.jsonl")

    X_all, id2row = load_mmap(vec_root)

    classes = np.array([0, 1], dtype=np.int32)
    y_all = np.fromiter((p["label"] for p in pairs), dtype=np.int32)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_all,
    )
    class_weight = {int(c): float(w) for c, w in zip(classes, class_weights)}

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=5e-4,
        learning_rate="optimal",
        class_weight=class_weight,
        average=True,
        tol=1e-3,
        n_iter_no_change=2,
        random_state=42,
    )

    rng = np.random.default_rng(42)

    t0 = time.perf_counter()
    for epoch in range(epochs):
        rng.shuffle(pairs)
        for Xb, yb in iter_Xy_batches_mmap(
            pairs, X_all, id2row, batch_size
        ):
            clf.partial_fit(Xb, yb, classes=classes)

    print(f"[TRAIN] done in {time.perf_counter() - t0:.2f}s")
    return clf


import numpy as np

def _bin_metrics(labels: np.ndarray, preds: np.ndarray):
    labels = labels.astype(np.int32, copy=False)
    preds = preds.astype(np.int32, copy=False)
    tp = np.int64(((preds == 1) & (labels == 1)).sum())
    tn = np.int64(((preds == 0) & (labels == 0)).sum())
    fp = np.int64(((preds == 1) & (labels == 0)).sum())
    fn = np.int64(((preds == 0) & (labels == 1)).sum())

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0

    prec_denom = tp + fp
    rec_denom = tp + fn
    precision = tp / prec_denom if prec_denom else 0.0
    recall = tp / rec_denom if rec_denom else 0.0

    f1_denom = precision + recall
    f1 = 2 * precision * recall / f1_denom if f1_denom else 0.0

    mcc_denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if mcc_denom:
        mcc = ((tp * tn) - (fp * fn)) / np.sqrt(mcc_denom)
    else:
        mcc = 0.0

    return acc, precision, recall, f1, mcc


def sweep_thresholds(labels: np.ndarray, sims: np.ndarray, thresholds: np.ndarray | None = None):
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.01, dtype=np.float32)

    best_f1, best_thr_f1, best_pack_f1 = -1.0, 0.5, None
    best_mcc, best_thr_mcc, best_pack_mcc = -1.0, 0.5, None

    for t in thresholds:
        preds = (sims >= t).astype(np.int32, copy=False)
        acc, prec, rec, f1, mcc = _bin_metrics(labels, preds)

        if f1 > best_f1:
            best_f1, best_thr_f1 = f1, float(t)
            best_pack_f1 = (acc, prec, rec, f1, mcc)

        if mcc > best_mcc:
            best_mcc, best_thr_mcc = mcc, float(t)
            best_pack_mcc = (acc, prec, rec, f1, mcc)

    return {
        "best_threshold_f1": best_thr_f1,
        "best_f1": best_pack_f1[3],
        "best_precision_at_best_f1": best_pack_f1[1],
        "best_recall_at_best_f1": best_pack_f1[2],
        "best_accuracy_at_best_f1": best_pack_f1[0],
        "best_mcc_at_best_f1": best_pack_f1[4],
        "best_threshold_mcc": best_thr_mcc,
        "best_mcc": best_pack_mcc[4],
        "best_precision_at_best_mcc": best_pack_mcc[1],
        "best_recall_at_best_mcc": best_pack_mcc[2],
        "best_accuracy_at_best_mcc": best_pack_mcc[0],
    }

def evaluate_streaming(
    clf,
    method: str,
    variant: str,
    split: str,
    batch_size: int = 8192,
):
    vec_root = f"vectors/{method}_{variant}/{split}_mmap"
    pairs = _read_pairs(f"data/{split}.jsonl")

    X_all, id2row = load_mmap(vec_root)

    y_true_all, y_proba_all = [], []

    for Xb, yb in iter_Xy_batches_mmap(
        pairs, X_all, id2row, batch_size
    ):
        proba = clf.predict_proba(Xb)[:, 1]
        y_true_all.append(yb)
        y_proba_all.append(proba.astype(np.float32, copy=False))

    return (
        np.concatenate(y_true_all),
        np.concatenate(y_proba_all),
    )


def find_best_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    best_t, best_f1 = 0.5, 0.0

    for t in thresholds:
        f1 = f1_score(
            y_true,
            (y_proba >= t).astype(np.int32),
            zero_division=0,
        )
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


def print_report(y_true, y_pred, y_proba, header: str):
    print(header)
    print(f"ROC-AUC: {roc_auc_score(y_true, y_proba):.4f}")
    print(f"F1-score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Not Clone", "Clone"],
        )
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="c2v")
    p.add_argument("--variant", default="normalized")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument(
        "--calibrate_on",
        choices=["val", "test", "none"],
        default="val",
    )
    args = p.parse_args()

    clf = train_lr_streaming(
        args.method,
        args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    threshold = 0.5
    if args.calibrate_on != "none":
        y_true_cal, y_proba_cal = evaluate_streaming(
            clf, args.method, args.variant, split=args.calibrate_on, batch_size=args.batch_size
        )
        print("[EVAL] done")
        sweep = sweep_thresholds(y_true_cal, y_proba_cal, thresholds=np.arange(0.0, 1.01, 0.01))
        threshold = float(sweep["best_threshold_f1"])  
        print(
            f"[CALIBRATION-SWEEP] split={args.calibrate_on} "
            f"t_f1={sweep['best_threshold_f1']:.2f} F1={sweep['best_f1']:.4f} "
            f"(t_mcc={sweep['best_threshold_mcc']:.2f} MCC={sweep['best_mcc']:.4f})"
        )

    y_true, y_proba = evaluate_streaming(
        clf,
        args.method,
        args.variant,
        split="test",
        batch_size=args.batch_size,
    )
    y_pred = (y_proba >= threshold).astype(np.int32)

    print_report(
        y_true,
        y_pred,
        y_proba,
        header=f"[LR-MMAP] {args.method.upper()} ({args.variant}) @t={threshold:.4f}",
    )


if __name__ == "__main__":
    main()
