import os
import json
from collections import Counter
from src.ast_factorizators.TK.tree_kernel import ast_to_tree_node, extract_all_subtrees

def build_tk_vocab(ast_dir: str, vocab_path: str, max_depth: int = 5):
    all_hashes = Counter()
    
    for filename in os.listdir(ast_dir):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(ast_dir, filename), "r", encoding="utf8") as f:
            ast = json.load(f)["ast"]
            if ast:
                tree = ast_to_tree_node(ast)
                if tree:
                    hashes = extract_all_subtrees(tree, max_depth=max_depth)
                    all_hashes.update(hashes)
    
    vocab = {str(h): i for i, h in enumerate(sorted(all_hashes.keys()))}
    
    with open(vocab_path, "w", encoding="utf8") as f:
        json.dump(vocab, f)
    
    print(f"vocab Tree Kernel saved: {len(vocab)} subtrees")

if __name__ == "__main__":
    import sys
    variant = sys.argv[1]  
    build_tk_vocab(
        f"output_ast{'_normalized' if variant == 'normalized' else ''}/train",
        f"src/ast_factorizators/TK/tk_vocab_{variant}.json",
        max_depth=5
    )