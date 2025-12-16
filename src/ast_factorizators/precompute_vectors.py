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
from src.ast_factorizators.TK.tree_kernel import (
    tree_kernel_factorization_with_vocab,
    load_tk_vocab
)
from src.ast_factorizators.C2V.code2vec import c2v_factorization, load_c2v_vocab

from src.ast_factorizators.C2V.code2vec import load_tfidf_vectorizer, ast_path_tfidf_factorization

def get_factorizer(method: str, variant: str, max_depth: int = 8) -> Callable:
    if method == "bon":
        return bag_of_nodes_factorization
    elif method == "wl":
        path = "src/ast_factorizators/WL/wl_vocab.json" if variant == "raw" else "src/ast_factorizators/WL/wl_vocab_normalized.json"
        load_wl_vocab(path)
        return wl_hash_factorization
    elif method == "bop":
        path = f"src/ast_factorizators/BOP/bop_vocab_{'raw' if variant == 'raw' else 'normalized'}.json"
        load_path_vocab(path)
        return lambda ast: bag_of_paths_factorization(ast, max_depth=max_depth)
    elif method == "tk":
        path = f"src/ast_factorizators/TK/tk_vocab_{'raw' if variant == 'raw' else 'normalized'}.json"
        load_tk_vocab(path)
        return tree_kernel_factorization_with_vocab
    elif method == "c2v":
        path = f"src/ast_factorizators/C2V/c2v_vocab_{'raw' if variant=='raw' else 'normalized'}.json"
        load_c2v_vocab(path)
        return lambda ast: c2v_factorization(ast, max_path_length=max_depth)
    elif method == "c2v_tfidf":
        load_tfidf_vectorizer("src/ast_factorizators/C2V/c2v_tfidf.pkl")
        return ast_path_tfidf_factorization
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
            print(f"Error processing {ast_path}: {e}")

        if total % 1000 == 0:
            print(f"[{method}/{variant}/{split}] Processed: {total}")

    print(f"Completed: {method}/{variant}/{split} : {total} vectors")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bon", "wl", "bop", "tk", "c2v", "c2v_tfidf"], required=True)
    parser.add_argument("--variant", choices=["raw", "normalized"], required=True)
    parser.add_argument("--max-depth", type=int, default=10, help="Max path depth for BoP")
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