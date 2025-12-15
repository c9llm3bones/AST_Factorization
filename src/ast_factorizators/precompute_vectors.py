import os
import json
import numpy as np
from typing import Callable

from src.ast_factorizators.BON.bag_of_nodes import bag_of_nodes_factorization
#from src.ast_factorizators.BON.node_types import load_known_types as load_bon_types
from src.ast_factorizators.WL.wl_hash import wl_hash_factorization, load_wl_vocab
from src.ast_factorizators.BOP.bag_of_paths import (
    bag_of_paths_factorization,
    load_path_vocab
)

def get_factorizer(method: str, variant: str, max_depth: int = 8) -> Callable:
    if method == "bon":
        #path = "node_types_raw.txt" if variant == "raw" else "node_types_normalized.txt"
        #load_bon_types(path)
        return bag_of_nodes_factorization
    elif method == "wl":
        path = "src/ast_factorizators/WL/wl_vocab_raw.json" if variant == "raw" else "src/ast_factorizators/WL/wl_vocab_normalized.json"
        load_wl_vocab(path)
        return wl_hash_factorization
    elif method == "bop":
        path = f"src/ast_factorizators/BOP/bop_vocab_{'raw' if variant == 'raw' else 'normalized'}.json"
        load_path_vocab(path)
        # Замыкание, чтобы передать max_depth
        return lambda ast: bag_of_paths_factorization(ast, max_depth=max_depth)
    else:
        raise ValueError(f"Unknown method: {method}")

def precompute_vectors(
    method: str,
    variant: str,
    split: str,
    input_ast_dir: str,
    output_vec_dir: str,
    max_depth: int = 8  
):
    factorizer = get_factorizer(method, variant, max_depth=max_depth)
    os.makedirs(output_vec_dir, exist_ok=True)

    total = 0
    for filename in os.listdir(input_ast_dir):
        if not filename.endswith(".json"):
            continue

        ast_path = os.path.join(input_ast_dir, filename)
        id_ = filename[:-5]

        try:
            with open(ast_path, "r", encoding="utf8") as f:
                ast = json.load(f)["ast"]
            if ast is None:
                vec = np.array([])
            else:
                vec = np.array(factorizer(ast))
            
            vec_path = os.path.join(output_vec_dir, f"{id_}.npy")
            np.save(vec_path, vec)
            total += 1

        except Exception as e:
            print(f"Ошибка при обработке {ast_path}: {e}")

        if total % 1000 == 0:
            print(f"[{method}/{variant}/{split}] Обработано: {total}")

    print(f"Завершено: {method}/{variant}/{split} : {total} векторов")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bon", "wl", "bop"], required=True)
    parser.add_argument("--variant", choices=["raw", "normalized"], required=True)
    parser.add_argument("--max-depth", type=int, default=8, help="Max path depth for BoP")
    args = parser.parse_args()

    splits = ["train", "val", "test"]
    for split in splits:
        input_dir = f"output_ast{'' if args.variant == 'raw' else '_normalized'}/{split}"
        output_dir = f"vectors/{args.method}_{args.variant}/{split}"
        precompute_vectors(
            args.method,
            args.variant,
            split,
            input_dir,
            output_dir,
            max_depth=args.max_depth
        )