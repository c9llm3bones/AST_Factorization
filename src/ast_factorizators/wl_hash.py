"""
Weisfeiler–Lehman / subtree hashing для AST.

Экспортируемые функции:
- wl_hash(node, sort_children=True, _cache=None) -> str
- bag_of_wl_hashes(root, sort_children=True) -> collections.Counter

Параметр sort_children:
    True  — сортируем child-hashes перед объединением (делает код нечувствительным к порядку детей)
    False — сохраняем порядок детей (порядок важен)
"""
from collections import Counter
from typing import Dict, Any, Optional
import hashlib
import json
import os


def read_ast_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def wl_hash(node: Dict[str, Any], sort_children: bool = True, _cache: Optional[dict] = None) -> str:
    """
    Recursively calculate SHA1 hash of subtree.
    - leaf: hash 'type:text'
    - non-leaf: hash 'type(' + join(child_hashes) + ')'
    Caching by id(node) (within the same structure) speeds up traversal.
    """
    if _cache is None:
        _cache = {}

    key = id(node)
    if key in _cache:
        return _cache[key]

    children = node.get("children", [])
    if not children:
        base = (node.get("type") or "") + ":" + (node.get("text") or "")
        h = hashlib.sha1(base.encode("utf8")).hexdigest()
        _cache[key] = h
        return h

    child_hashes = [wl_hash(ch, sort_children=sort_children, _cache=_cache) for ch in children]
    if sort_children:
        child_hashes = sorted(child_hashes)
    s = (node.get("type") or "") + "(" + ",".join(child_hashes) + ")"
    h = hashlib.sha1(s.encode("utf8")).hexdigest()
    _cache[key] = h
    return h


def bag_of_wl_hashes(root: Dict[str, Any], sort_children: bool = True) -> Counter:
    """
    Returns Counter: {hash_hex: count}
    """
    c = Counter()
    cache = {}

    def dfs(n: Dict[str, Any]):
        for ch in n.get("children", []):
            dfs(ch)
        h = wl_hash(n, sort_children=sort_children, _cache=cache)
        c[h] += 1

    dfs(root)
    return c


__all__ = ["read_ast_json", "wl_hash", "bag_of_wl_hashes"]
