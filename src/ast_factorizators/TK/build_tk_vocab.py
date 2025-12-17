import os
import json
import argparse
from collections import Counter
from typing import Iterable, Tuple, Dict, List

from src.ast_factorizators.TK.tree_kernel import ast_to_tree_node, extract_all_subtrees


def iter_ast_files(ast_dir: str) -> Iterable[str]:
    if not os.path.isdir(ast_dir):
        return
    for fn in os.listdir(ast_dir):
        if fn.endswith(".json"):
            yield os.path.join(ast_dir, fn)


def collect_hashes_from_dir(ast_dir: str, max_depth: int) -> Counter:
    counts = Counter()
    for path in iter_ast_files(ast_dir):
        try:
            with open(path, "r", encoding="utf8") as f:
                obj = json.load(f)
            ast = obj.get("ast")
            if not ast:
                continue
            tree = ast_to_tree_node(ast)
            if not tree:
                continue
            hashes = extract_all_subtrees(tree, max_depth=max_depth)
            if hashes:
                counts.update(hashes)
        except Exception:
            continue
    return counts


def apply_filters(hashes: Counter, min_count: int, top_k: int | None) -> List[str]:
    if min_count > 1:
        hashes = Counter({h: c for h, c in hashes.items() if c >= min_count})
    if top_k is not None and top_k > 0:
        hashes = Counter(dict(hashes.most_common(top_k)))
    keys_sorted = sorted((str(h) for h in hashes.keys()))
    return keys_sorted


def build_tk_vocab(
    variant: str,
    splits: List[str],
    ast_root: str | None,
    max_depth: int,
    min_count: int,
    top_k: int | None,
) -> Dict[str, int]:
    total = Counter()
    for split in splits:
        if ast_root:
            ast_dir = os.path.join(ast_root, split)
        else:
            suffix = "" if variant == "raw" else "_normalized"
            ast_dir = f"output_ast{suffix}/{split}"

        if not os.path.isdir(ast_dir):
            print(f"[SKIP] split not found: {ast_dir}")
            continue

        part = collect_hashes_from_dir(ast_dir, max_depth=max_depth)
        print(f"[INFO] {split}: collected {len(part)} unique subtree hashes")
        total.update(part)

    print(f"[INFO] total unique subtree hashes before filtering: {len(total)}")

    keys_sorted = apply_filters(total, min_count=min_count, top_k=top_k)
    vocab = {k: i for i, k in enumerate(keys_sorted)}
    print(f"[INFO] vocab size after filtering: {len(vocab)}")
    return vocab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=["raw", "normalized"], help="Which AST variant to use")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to include")
    p.add_argument("--ast-root", default=None, help="Root dir with AST subdirs per split; if omitted, uses output_ast[_normalized]/<split>")
    p.add_argument("--max-depth", type=int, default=5, help="Max subtree depth")
    p.add_argument("--min-count", type=int, default=1, help="Keep hashes with frequency >= min_count")
    p.add_argument("--top-k", type=int, default=None, help="Keep only top-K most frequent hashes")
    p.add_argument("--out", default=None, help="Output path for vocab JSON")

    args = p.parse_args()

    vocab = build_tk_vocab(
        variant=args.variant,
        splits=args.splits,
        ast_root=args.ast_root,
        max_depth=args.max_depth,
        min_count=args.min_count,
        top_k=args.top_k,
    )

    if args.out:
        out_path = args.out
    else:
        os.makedirs("src/ast_factorizators/TK", exist_ok=True)
        out_path = f"src/ast_factorizators/TK/tk_vocab_{args.variant}.json"

    with open(out_path, "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    print(f"vocab Tree Kernel saved: {len(vocab)} subtrees")
    print(f"path: {out_path}")


if __name__ == "__main__":
    main()