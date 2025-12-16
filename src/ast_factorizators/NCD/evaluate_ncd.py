import os
import json
import argparse
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

from src.ast_factorizators.NCD.ncd_similarity import ncd_similarity

def evaluate_ncd(variant: str = "normalized", split: str = "test"):
    ast_dir = "output_ast" if variant == "raw" else "output_ast_normalized"
    jsonl_path = f"data/{split}.jsonl"
    ast_full_dir = os.path.join(ast_dir, split)

    labels, sims = [], []

    with open(jsonl_path, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            id1, id2 = obj["id1"], obj["id2"]
            label = obj["label"]
            p1 = os.path.join(ast_full_dir, f"{id1}.json")
            p2 = os.path.join(ast_full_dir, f"{id2}.json")

            if os.path.isfile(p1) and os.path.isfile(p2):
                try:
                    with open(p1, "r", encoding="utf8") as f1, open(p2, "r", encoding="utf8") as f2:
                        ast1 = json.load(f1)["ast"]
                        ast2 = json.load(f2)["ast"]
                    if ast1 is not None and ast2 is not None:
                        sim = ncd_similarity(ast1, ast2)
                        labels.append(label)
                        sims.append(sim)
                except Exception:
                    pass

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} pairs")

    labels = np.array(labels)
    sims = np.array(sims)

    print(f"\nДоля клонов: {labels.mean():.4f}")
    print(f"Средняя схожесть: {sims.mean():.4f}")

    # Metrics at threshold 0.5
    preds = (sims >= 0.5).astype(int)
    print(f"\n[TEST] NCD ({variant})")
    print(f"  ROC-AUC      : {roc_auc_score(labels, sims):.4f}")
    print(f"  PR-AUC       : {average_precision_score(labels, sims):.4f}")
    print(f"  Precision    : {precision_score(labels, preds, zero_division=0):.4f}")
    print(f"  Recall       : {recall_score(labels, preds, zero_division=0):.4f}")
    print(f"  F1-score     : {f1_score(labels, preds, zero_division=0):.4f}")
    print(f"  MCC          : {matthews_corrcoef(labels, preds):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["raw", "normalized"], default="normalized")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    args = parser.parse_args()
    evaluate_ncd(args.variant, args.split)