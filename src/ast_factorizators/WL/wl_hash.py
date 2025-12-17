import hashlib
from collections import Counter
from typing import Dict, Any, Optional
import os
import json
from collections import Counter

def wl_hash(node: Dict[str, Any], sort_children: bool = False, _cache: Optional[dict] = None) -> str:
    if _cache is None:
        _cache = {}

    node_id = id(node)
    if node_id in _cache:
        return _cache[node_id]

    children = node.get("children", [])
    if not children:
        # Лист: тип + значение (если есть)
        base = node.get("type", "")
        if "value" in node:
            base += ":" + str(node["value"])
        h = hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()
        _cache[node_id] = h
        return h

    # Рекурсивно хэшируем детей
    child_hashes = [wl_hash(ch, sort_children=sort_children, _cache=_cache) for ch in children]
    
    if sort_children:
        child_hashes.sort()

    # Собираем строку: тип(хэш1,хэш2,...)
    s = node.get("type", "") + "(" + ",".join(child_hashes) + ")"
    h = hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()
    _cache[node_id] = h
    return h

def bag_of_wl_hashes(ast: Dict[str, Any], sort_children: bool = False) -> Counter:
    counter = Counter()
    cache = {}

    def dfs(n: Dict[str, Any]):
        for ch in n.get("children", []):
            dfs(ch)
        h = wl_hash(n, sort_children=sort_children, _cache=cache)
        counter[h] += 1

    dfs(ast)
    return counter

TOP_WL_HASHES = []
HASH_TO_IDX = {}

def build_wl_vocab(ast_dir: str, top_k: int = 3000, save_path: str = None):
    global TOP_WL_HASHES, HASH_TO_IDX
    total_counter = Counter()
    
    print(f"Сбор WL-хэшей из {ast_dir}...")
    for filename in os.listdir(ast_dir):
        if filename.endswith(".json"):
            with open(os.path.join(ast_dir, filename), "r", encoding="utf8") as f:
                ast = json.load(f)["ast"]
                if ast:
                    counter = bag_of_wl_hashes(ast, sort_children=False)
                    total_counter.update(counter)
    TOP_WL_HASHES = [h for h, _ in total_counter.most_common(top_k)]
    HASH_TO_IDX = {h: i for i, h in enumerate(TOP_WL_HASHES)}

    if save_path:
        with open(save_path, "w", encoding="utf8") as f:
            json.dump(TOP_WL_HASHES, f)

    print(f"Словарь WL: {len(TOP_WL_HASHES)} хэшей")
    
def load_wl_vocab(load_path: str):
    global TOP_WL_HASHES, HASH_TO_IDX
    with open(load_path, "r", encoding="utf8") as f:
        TOP_WL_HASHES = json.load(f)
    HASH_TO_IDX = {h: i for i, h in enumerate(TOP_WL_HASHES)}
    print(f"Загружено {len(TOP_WL_HASHES)} WL хэшей из {load_path}")

def wl_hash_factorization(ast: Dict[str, Any]) -> list:
    """Преобразует AST в вектор длины top_k."""
    if not HASH_TO_IDX:
        raise RuntimeError("Сначала вызовите build_wl_vocab()")
    
    counter = bag_of_wl_hashes(ast, sort_children=False)
    vec = [0] * len(TOP_WL_HASHES)
    for h, cnt in counter.items():
        if h in HASH_TO_IDX:
            vec[HASH_TO_IDX[h]] = cnt
    return vec

#build_wl_vocab("output_ast/train", top_k=3000, save_path="src/ast_factorizators/WL/wl_vocab.json")
#build_wl_vocab("output_ast_normalized/train", top_k=3000, save_path="src/ast_factorizators/WL/wl_vocab_normalized.json")
