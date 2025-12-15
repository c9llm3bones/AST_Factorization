import os
from collections import Counter
from typing import Dict, Any, List
import json
def extract_paths(node: Dict[str, Any], current_path: List[str], max_depth: int = 8) -> List[tuple]:
    """
    Extract root-to-leaf paths, enriching leaf nodes with their semantic value (text).
    Handles your AST format: {"type": "...", "value": {"type": "...", "text": "..."}}.
    """
    if len(current_path) >= max_depth:
        return []

    # Determine the label for the current node
    node_type = node.get("type", "Unknown")
    label = node_type  # default

    # If it's a leaf (no children), try to enrich with text value
    children = node.get("children", [])
    if not children:
        value = node.get("value")
        text = ""
        if isinstance(value, dict) and "text" in value:
            text = str(value["text"])
        elif isinstance(value, str):
            text = value
        # Only append text if it's non-empty and meaningful
        if text and not text.isspace():
            # Avoid explosion: truncate very long literals
            if len(text) > 50:
                text = text[:50] + "..."
            label = f"{node_type}:{text}"
        # else: keep just node_type (e.g., for 'block' with no text)

    paths = []
    if not children:
        # Leaf: complete the path
        paths.append(tuple(current_path + [label]))
    else:
        # Internal node: recurse
        for child in children:
            paths.extend(extract_paths(child, current_path + [label], max_depth))
    
    return paths

def bag_of_paths(ast: Dict[str, Any], max_depth: int = 8) -> Counter:
    """
    Returns Counter: {path_tuple: frequency}
    """
    paths = extract_paths(ast, [], max_depth)
    return Counter(paths)

TOP_PATHS = []
PATH_TO_IDX = {}

def build_path_vocab(ast_dir: str, top_k: int = 5000, max_depth: int = 8):
    """Build vocabulary of most frequent paths from a directory of ASTs."""
    global TOP_PATHS, PATH_TO_IDX
    total_counter = Counter()
    
    print(f"Collecting paths from {ast_dir} (depth={max_depth})...")
    for filename in os.listdir(ast_dir):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(ast_dir, filename), "r", encoding="utf8") as f:
            ast = json.load(f)["ast"]
            if ast:
                counter = bag_of_paths(ast, max_depth=max_depth)
                total_counter.update(counter)
    
    TOP_PATHS = [p for p, _ in total_counter.most_common(top_k)]
    PATH_TO_IDX = {p: i for i, p in enumerate(TOP_PATHS)}
    print(f"Path vocab built: {len(TOP_PATHS)} paths")

def load_path_vocab(path: str):
    """Load precomputed path vocab (as list of lists, since tuple not JSON-serializable)."""
    global TOP_PATHS, PATH_TO_IDX
    with open(path, "r", encoding="utf8") as f:
        path_list = json.load(f)
    TOP_PATHS = [tuple(p) for p in path_list]
    PATH_TO_IDX = {p: i for i, p in enumerate(TOP_PATHS)}

def save_path_vocab(path: str):
    """Save path vocab as list of lists (JSON-compatible)."""
    with open(path, "w", encoding="utf8") as f:
        json.dump([list(p) for p in TOP_PATHS], f)

def bag_of_paths_factorization(ast: Dict[str, Any], max_depth: int = 8) -> List[int]:
    """Convert AST to fixed-length vector based on path vocab."""
    if not PATH_TO_IDX:
        raise RuntimeError("Path vocab not loaded. Call build_path_vocab or load_path_vocab first.")
    
    counter = bag_of_paths(ast, max_depth=max_depth)
    vec = [0] * len(TOP_PATHS)
    for path, cnt in counter.items():
        if path in PATH_TO_IDX:
            vec[PATH_TO_IDX[path]] = cnt
    return vec

#build_path_vocab("output_ast_normalized/train", top_k=5000, max_depth=6)
#save_path_vocab("src/ast_factorizators/BOP/bop_vocab_normalized.json")