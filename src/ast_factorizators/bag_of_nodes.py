"""
bag_of_nodes.py

bag-of-nodes — счётчик типов узлов AST.

Ожидаемый вход: AST в виде словаря:
    { "type": "...", "text": "...", "children": [ ... ] }

Экспортируемые функции:
- bag_of_nodes(root) -> collections.Counter
- read_ast_json(path) -> dict
"""
from collections import Counter
from typing import Dict, Any
import json
import os


def read_ast_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def iter_nodes(node: Dict[str, Any]):
    yield node
    for ch in node.get("children", []):
        yield from iter_nodes(ch)


def bag_of_nodes(root: Dict[str, Any]) -> Counter:
    """
    (строка типа -> частота).

    Пример:
        {'IfStmt': 3, 'MethodDeclaration': 2, 'NameExpr': 10, ...}
    """
    c = Counter()
    for n in iter_nodes(root):
        typ = n.get("type") or "Unknown"
        c[typ] += 1
    return c

__all__ = ["read_ast_json", "bag_of_nodes"]
