import os
import json
import argparse
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# --- Bag-of-Nodes imports ---
from src.ast_factorizators.BON.bag_of_nodes import bag_of_nodes_factorization as bon_factorize

# --- WL imports ---
from src.ast_factorizators.WL.wl_hash import (
    wl_hash_factorization,
    load_wl_vocab
)

# --- Bag-of-Paths imports ---
from src.ast_factorizators.BOP.bag_of_paths import (
    bag_of_paths_factorization,
    load_path_vocab
)

# --- Tree Kernel imports ---
from src.ast_factorizators.TK.tree_kernel import (
    tree_kernel_factorization_with_vocab,
    load_tk_vocab
)

from src.ast_factorizators.C2V.code2vec import (
    c2v_factorization,
    load_c2v_vocab,
    load_tfidf_vectorizer,
    ast_path_tfidf_factorization,
)

from time import time
def _bin_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    return acc, prec, rec, f1, mcc

def sweep_thresholds(labels: np.ndarray, sims: np.ndarray, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.01)

    best_f1, best_thr_f1, best_pack_f1 = -1.0, 0.5, None
    best_mcc, best_thr_mcc, best_pack_mcc = -1.0, 0.5, None

    for t in thresholds:
        preds = (sims >= t).astype(int)
        acc, prec, rec, f1, mcc = _bin_metrics(labels, preds)
        if f1 > best_f1:
            best_f1, best_thr_f1 = f1, t
            best_pack_f1 = (acc, prec, rec, f1, mcc)
        if mcc > best_mcc:
            best_mcc, best_thr_mcc = mcc, t
            best_pack_mcc = (acc, prec, rec, f1, mcc)

    out = {
        "best_threshold_f1": round(best_thr_f1, 2),
        "best_f1": best_pack_f1[3],
        "best_precision_at_best_f1": best_pack_f1[1],
        "best_recall_at_best_f1": best_pack_f1[2],
        "best_accuracy_at_best_f1": best_pack_f1[0],
        "best_mcc_at_best_f1": best_pack_f1[4],

        "best_threshold_mcc": round(best_thr_mcc, 2),
        "best_mcc": best_pack_mcc[4],
        "best_precision_at_best_mcc": best_pack_mcc[1],
        "best_recall_at_best_mcc": best_pack_mcc[2],
        "best_accuracy_at_best_mcc": best_pack_mcc[0],
        "best_f1_at_best_mcc": best_pack_mcc[3],
    }
    return out

def load_factorizer(method: str, variant: str):
    if method == "bon":
        return bon_factorize

    elif method == "wl":
        path = "src/ast_factorizators/WL/wl_vocab.json" if variant == "raw" else "src/ast_factorizators/WL/wl_vocab_normalized.json"
        load_wl_vocab(path)
        return wl_hash_factorization

    elif method == "bop":
        path = f"src/ast_factorizators/BOP/bop_vocab_{'raw' if variant == 'raw' else 'normalized'}.json"
        load_path_vocab(path)
        return bag_of_paths_factorization

    elif method == "tk":  
        path = f"src/ast_factorizators/TK/tk_vocab_{'raw' if variant == 'raw' else 'normalized'}.json"
        load_tk_vocab(path)
        return tree_kernel_factorization_with_vocab
    elif method == "c2v":
        path = f"src/ast_factorizators/C2V/c2v_vocab_{'raw' if variant=='raw' else 'normalized'}.json"
        load_c2v_vocab(path)
        return c2v_factorization
    elif method == "c2v_tfidf":
        load_tfidf_vectorizer("src/ast_factorizators/C2V/c2v_tfidf.pkl")
        return ast_path_tfidf_factorization
    else:
        raise ValueError(f"Unknown method: {method}. Supported: 'bon', 'wl', 'bop', 'tk', 'c2v'")
        
def compute_similarity(factorizer, path1: str, path2: str) -> float:
    try:
        with open(path1, "r", encoding="utf8") as f1, open(path2, "r", encoding="utf8") as f2:
            ast1 = json.load(f1)["ast"]
            ast2 = json.load(f2)["ast"]
        if ast1 is None or ast2 is None:
            return 0.0
        v1 = np.array(factorizer(ast1))
        v2 = np.array(factorizer(ast2))
        if np.all(v1 == 0) and np.all(v2 == 0):
            return 0.0
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        return cosine_similarity([v1], [v2])[0, 0]
    except Exception:
        return 0.0

def evaluate(method: str, variant: str, split: str = "test", top_k: int = 50):
    ast_dir = "output_ast" if variant == "raw" else "output_ast_normalized"
    jsonl_path = f"data/{split}.jsonl"
    ast_full_dir = os.path.join(ast_dir, split)

    factorizer = load_factorizer(method, variant)
    labels, sims = [], []

    with open(jsonl_path, "r", encoding="utf8") as f:
        i = 0
        for line in f:
            obj = json.loads(line)
            id1, id2 = obj["id1"], obj["id2"]
            label = obj["label"]
            p1 = os.path.join(ast_full_dir, f"{id1}.json")
            p2 = os.path.join(ast_full_dir, f"{id2}.json")
            if os.path.isfile(p1) and os.path.isfile(p2):
                sim = compute_similarity(factorizer, p1, p2)
                labels.append(label)
                sims.append(sim)
            i += 1
            if i % 1000 == 0:
                print(f"Processed {i} pairs")
    
    labels = np.array(labels)
    sims = np.array(sims)

    if sims.size == 0:
        print("No computed vectors. Check vectors/{method}_{variant}/{split}")
        return sims, labels, {
            "roc_auc": 0.0, "pr_auc": 0.0, "accuracy": 0.0, "precision": 0.0,
            "mcc": 0.0, "recall": 0.0, "f1": 0.0,
            "best_threshold_f1": None, "best_f1": 0.0,
            "best_precision_at_best_f1": 0.0, "best_recall_at_best_f1": 0.0,
            "best_accuracy_at_best_f1": 0.0, "best_mcc_at_best_f1": 0.0,
            "best_threshold_mcc": None, "best_mcc": 0.0,
            "best_precision_at_best_mcc": 0.0, "best_recall_at_best_mcc": 0.0,
            "best_accuracy_at_best_mcc": 0.0, "best_f1_at_best_mcc": 0.0,
        }

    print("Средняя схожесть:", np.mean(sims))
    print("Мин/Макс:", sims.min(), sims.max())
    print("Доля клонов (label=1):", np.mean(labels))
    threshold = 0.5
    preds = (sims >= threshold).astype(int)

    roc_auc = roc_auc_score(labels, sims)
    pr_auc = average_precision_score(labels, sims)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)

    name = f"{method.upper()} ({variant})"
    print(f"\n[{split.upper()}] {name}")
    print(f"  ROC-AUC      : {roc_auc:.4f}")
    print(f"  PR-AUC       : {pr_auc:.4f}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Precision    : {prec:.4f}")
    print(f"  Recall       : {rec:.4f}")
    print(f"  F1-score     : {f1:.4f}")
    print(f"  MCC          : {mcc:.4f}")
    
    sweep = sweep_thresholds(labels, sims)
    print(f"Лучший F1: {sweep['best_f1']:.4f} при пороге {sweep['best_threshold_f1']}")
    print(f"Лучший MCC: {sweep['best_mcc']:.4f} при пороге {sweep['best_threshold_mcc']}")

    return sims, labels, {
        "roc_auc": roc_auc, "pr_auc": pr_auc, "accuracy": acc,
        'precision': prec, "mcc": mcc, "recall": rec, 'f1': f1,
        **sweep,
    }

def evaluate_fast(method: str, variant: str, split: str = "test", top_k: int = 50):
    vec_dir = f"vectors/{method}_{variant}/{split}"
    jsonl_path = f"data/{split}.jsonl"

    labels, sims = [], []

    with open(jsonl_path, "r", encoding="utf8") as f:
        i = 0
        for line in f:
            obj = json.loads(line)
            id1, id2 = obj["id1"], obj["id2"]
            label = obj["label"]

            try:
                v1 = np.load(os.path.join(vec_dir, f"{id1}.npy"))
                v2 = np.load(os.path.join(vec_dir, f"{id2}.npy"))
                if v1.size == 0 or v2.size == 0:
                    sim = 0.0
                else:
                    sim = cosine_similarity([v1], [v2])[0, 0]
                labels.append(label)
                sims.append(sim)
            except Exception:
                continue
            i += 1
            if i % 1000 == 0:
                print(f"Processed {i} pairs")
                
    labels = np.array(labels)
    sims = np.array(sims)
    
    if sims.size == 0:
        print(f"No computed vectors. Check vectors/{method}_{variant}/{split}")
        return sims, labels, {
            "roc_auc": 0.0, "pr_auc": 0.0, "accuracy": 0.0, "precision": 0.0,
            "mcc": 0.0, "recall": 0.0, "f1": 0.0,
            "best_threshold_f1": None, "best_f1": 0.0,
            "best_precision_at_best_f1": 0.0, "best_recall_at_best_f1": 0.0,
            "best_accuracy_at_best_f1": 0.0, "best_mcc_at_best_f1": 0.0,
            "best_threshold_mcc": None, "best_mcc": 0.0,
            "best_precision_at_best_mcc": 0.0, "best_recall_at_best_mcc": 0.0,
            "best_accuracy_at_best_mcc": 0.0, "best_f1_at_best_mcc": 0.0,
        }
        
    name = f"{method.upper()} ({variant})"
    print(f"\n[{split.upper()}] {name}")
    print("Средняя схожесть:", np.mean(sims))
    print("Мин/Макс:", sims.min(), sims.max())
    print("Доля клонов (label=1):", np.mean(labels))

    # threshold = 0.5
    # preds = (sims >= threshold).astype(int)

    # roc_auc = roc_auc_score(labels, sims)
    # pr_auc = average_precision_score(labels, sims)
    # acc = accuracy_score(labels, preds)
    # prec = precision_score(labels, preds, zero_division=0)
    # rec = recall_score(labels, preds, zero_division=0)
    # f1 = f1_score(labels, preds, zero_division=0)
    # mcc = matthews_corrcoef(labels, preds)

    # name = f"{method.upper()} ({variant})"
    # print(f"\n[{split.upper()}] {name}")
    # print(f"  ROC-AUC      : {roc_auc:.4f}")
    # print(f"  PR-AUC       : {pr_auc:.4f}")
    # print(f"  Accuracy     : {acc:.4f}")
    # print(f"  Precision    : {prec:.4f}")
    # print(f"  Recall       : {rec:.4f}")
    # print(f"  F1-score     : {f1:.4f}")
    # print(f"  MCC          : {mcc:.4f}")

    sweep = sweep_thresholds(labels, sims)
    threshold = sweep['best_threshold_f1']
    preds = (sims >= threshold).astype(int)

    roc_auc = roc_auc_score(labels, sims)
    pr_auc = average_precision_score(labels, sims)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    print(f"Лучший F1: {sweep['best_f1']:.4f} при пороге {sweep['best_threshold_f1']}")
    print(f"Лучший MCC: {sweep['best_mcc']:.4f} при пороге {sweep['best_threshold_mcc']}")

    return sims, labels, {
        "roc_auc": roc_auc, "pr_auc": pr_auc, "accuracy": acc,
        'precision': prec, "mcc": mcc, "recall": rec, 'f1': f1,
        **sweep,
    }
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bon", "wl", "bop", "tk", "c2v", "c2v_tfidf"], required=True)  
    parser.add_argument("--variant", choices=["raw", "normalized"], required=True)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    args = parser.parse_args()
    start = time()
    sims, labels, metrics = evaluate_fast(args.method, args.variant, args.split)
    duration = time() - start

    os.makedirs("logs", exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": args.method,
        "variant": args.variant,
        "split": args.split,
        "duration_sec": round(duration, 2),
        **metrics
    }
    with open("logs/eval_metrics.jsonl", "a", encoding="utf8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Метрики записаны в logs/eval_metrics.jsonl")