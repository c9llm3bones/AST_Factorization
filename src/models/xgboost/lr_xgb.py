import os
import json
import argparse
import time
from typing import Tuple

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, f1_score

from scipy.sparse import csr_matrix
import xgboost as xgb

def iter_diff_batches(vec_root: str, jsonl_path: str, batch_size: int = 8192):
    X_all, id2row = load_mmap(vec_root)
    dim = X_all.shape[1]
    rows, cols, data, y_batch = [], [], [], []
    rptr = [0]

    cnt = 0
    for p in _read_pairs(jsonl_path):
        r1 = id2row.get(p["id1"]); r2 = id2row.get(p["id2"])
        if r1 is None or r2 is None:
            continue
        v1 = X_all[r1]; v2 = X_all[r2]
        if v1.shape != v2.shape:
            continue
        diff = np.abs(v1 - v2)
        nz = np.nonzero(diff)[0]
        cols.extend(nz.tolist())
        data.extend(diff[nz].astype(np.float32, copy=False).tolist())
        y_batch.append(int(p["label"]))
        cnt += 1
        rptr.append(len(cols))
        if cnt >= batch_size:
            X_csr = csr_matrix((np.array(data, dtype=np.float32), np.array(cols, dtype=np.int32), np.array(rptr, dtype=np.int32)), shape=(cnt, dim))
            y = np.asarray(y_batch, dtype=np.int32)
            yield X_csr, y
            # reset
            rows, cols, data, y_batch = [], [], [], []
            rptr = [0]
            cnt = 0

    if cnt > 0:
        X_csr = csr_matrix((np.array(data, dtype=np.float32), np.array(cols, dtype=np.int32), np.array(rptr, dtype=np.int32)), shape=(cnt, dim))
        y = np.asarray(y_batch, dtype=np.int32)
        yield X_csr, y

class MMapDataIter(xgb.core.DataIter):
    def __init__(self, vec_root: str, jsonl_path: str, batch_size: int = 8192):
        super().__init__()
        self.vec_root = vec_root
        self.jsonl_path = jsonl_path
        self.batch_size = batch_size
        self._gen = None

    def reset(self):
        self._gen = iter_diff_batches(self.vec_root, self.jsonl_path, self.batch_size)

    def next(self, input_data):
        try:
            X_csr, y = next(self._gen)
            input_data(data=X_csr, label=y)
            return 1
        except StopIteration:
            return 0

def _read_pairs(jsonl_path: str):
    with open(jsonl_path, "r", encoding="utf8") as f:
        for line in f:
            yield json.loads(line)


def load_mmap(vec_root: str):
    X = np.load(os.path.join(vec_root, "vectors.npy"), mmap_mode="r")
    with open(os.path.join(vec_root, "ids.json"), "r", encoding="utf8") as f:
        id2row = json.load(f)
    return X, id2row


def write_libsvm(vec_root: str, jsonl_path: str, out_path: str, batch_flush: int = 8192) -> Tuple[int, int]:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    X_all, id2row = load_mmap(vec_root)

    total, kept = 0, 0
    buf = []

    def flush():
        if not buf:
            return
        with open(out_path, "a", encoding="utf8") as fout:
            fout.write("\n".join(buf))
            fout.write("\n")
        buf.clear()

    if os.path.exists(out_path):
        os.remove(out_path)

    for p in _read_pairs(jsonl_path):
        total += 1
        r1 = id2row.get(p["id1"])
        r2 = id2row.get(p["id2"])
        if r1 is None or r2 is None:
            continue
        v1 = X_all[r1]
        v2 = X_all[r2]
        if v1.shape != v2.shape:
            continue

        diff = np.abs(v1 - v2)
        y = int(p["label"])

        nz = np.nonzero(diff)[0]
        if nz.size == 0:
            buf.append(f"{y}")
        else:
            parts = [f"{y}"] + [f"{i+1}:{float(diff[i])}" for i in nz]
            buf.append(" ".join(parts))

        kept += 1
        if len(buf) >= batch_flush:
            flush()

    flush()
    return kept, total


def count_pos_neg(jsonl_path: str) -> Tuple[int, int]:
    pos = neg = 0
    for p in _read_pairs(jsonl_path):
        if int(p["label"]) == 1:
            pos += 1
        else:
            neg += 1
    return pos, neg


def sweep_thresholds(labels: np.ndarray, sims: np.ndarray, thresholds: np.ndarray | None = None):
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.01, dtype=np.float32)

    labels = labels.astype(np.int32, copy=False)
    best_f1, best_thr_f1 = -1.0, 0.5
    best_pack_f1 = None
    best_mcc, best_thr_mcc = -1.0, 0.5
    best_pack_mcc = None

    for t in thresholds:
        preds = (sims >= t).astype(np.int32, copy=False)

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

        if f1 > best_f1:
            best_f1, best_thr_f1 = float(f1), float(t)
            best_pack_f1 = (acc, precision, recall, f1, mcc)
        if mcc > best_mcc:
            best_mcc, best_thr_mcc = float(mcc), float(t)
            best_pack_mcc = (acc, precision, recall, f1, mcc)

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

def train_xgb(method: str, variant: str, num_boost_round: int = 1000, early_stopping_rounds: int = 50):
    os.makedirs("models/lr", exist_ok=True)
    train_root = f"vectors/{method}_{variant}/train_mmap"
    val_root   = f"vectors/{method}_{variant}/val_mmap"

    pos, neg = count_pos_neg("data/train.jsonl")
    scale_pos_weight = neg / max(1, pos)

    params = dict(
        objective="binary:logistic",
        tree_method="hist",
        max_depth=6,
        eta=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        nthread=0,
    )

    train_iter = MMapDataIter(train_root, "data/train.jsonl", batch_size=8192)
    val_iter   = MMapDataIter(val_root,   "data/val.jsonl",   batch_size=8192)

    dtrain = xgb.QuantileDMatrix(train_iter)
    dval   = xgb.QuantileDMatrix(val_iter, ref=dtrain)  

    t0 = time.perf_counter()
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50,
    )
    print(f"[TRAIN] done in {time.perf_counter() - t0:.2f}s")
    booster.save_model(f"models/lr/xgb_{method}_{variant}.json")
    return booster

def predict_split(booster: xgb.Booster, method: str, variant: str, split: str, batch_size: int = 8192):
    vec_root = f"vectors/{method}_{variant}/{split}_mmap"
    y_true_chunks, y_proba_chunks = [], []

    for X_csr, y in iter_diff_batches(vec_root, f"data/{split}.jsonl", batch_size=batch_size):
        dmat = xgb.DMatrix(X_csr)  
        proba = booster.predict(dmat, iteration_range=(0, booster.best_iteration + 1))
        y_true_chunks.append(y)
        y_proba_chunks.append(proba.astype(np.float32, copy=False))

    y_true = np.concatenate(y_true_chunks) if y_true_chunks else np.empty((0,), dtype=np.int32)
    y_proba = np.concatenate(y_proba_chunks) if y_proba_chunks else np.empty((0,), dtype=np.float32)
    return y_true, y_proba


def print_report(y_true, y_pred, y_proba, header: str):
    print(header)
    print(f"ROC-AUC: {roc_auc_score(y_true, y_proba):.4f}")
    print(f"F1-score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(classification_report(y_true, y_pred, target_names=["Not Clone", "Clone"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="tk")
    ap.add_argument("--variant", default="normalized")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--early", type=int, default=50)
    ap.add_argument("--calibrate_on", choices=["val", "test", "none"], default="val")
    args = ap.parse_args()

    booster = train_xgb(args.method, args.variant, num_boost_round=args.rounds, early_stopping_rounds=args.early)

    threshold = 0.5
    if args.calibrate_on != "none":
        y_true_cal, y_proba_cal = predict_split(booster, args.method, args.variant, split=args.calibrate_on)
        sweep = sweep_thresholds(y_true_cal, y_proba_cal, thresholds=np.arange(0.0, 1.01, 0.01))
        threshold = float(sweep["best_threshold_f1"])
        print(
            f"[CALIBRATION-SWEEP] split={args.calibrate_on} "
            f"t_f1={sweep['best_threshold_f1']:.2f} F1={sweep['best_f1']:.4f} "
            f"(t_mcc={sweep['best_threshold_mcc']:.2f} MCC={sweep['best_mcc']:.4f})"
        )

    y_true_test, y_proba_test = predict_split(booster, args.method, args.variant, split="test")
    y_pred_test = (y_proba_test >= threshold).astype(np.int32)
    print_report(y_true_test, y_pred_test, y_proba_test, header=f"[XGB] {args.method.upper()} ({args.variant}) @t={threshold:.4f}")


if __name__ == "__main__":
    main()