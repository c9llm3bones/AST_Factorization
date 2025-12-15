from collections import Counter, defaultdict
from typing import Dict, Any, List
import json

with open("src/ast_factorizators/BON/meaningful_node_types.txt", encoding="utf8") as f:
    KNOWN_TYPES = [line.strip() for line in f if line.strip()]
TYPE_TO_IDX = {t: i for i, t in enumerate(KNOWN_TYPES)}

def iter_nodes(node: Dict[str, Any]):
    if isinstance(node, dict):
        yield node
        for child in node.get("children", []):
            yield from iter_nodes(child)
    elif isinstance(node, list):
        for item in node:
            yield from iter_nodes(item)

def bag_of_nodes_factorization(ast: Dict[str, Any]) -> List[int]:
    counter = Counter()
    for node in iter_nodes(ast):
        typ = node.get("type")
        if typ in TYPE_TO_IDX:
            counter[typ] += 1

    vec = [0] * len(KNOWN_TYPES)
    for typ, cnt in counter.items():
        vec[TYPE_TO_IDX[typ]] = cnt
    return vec

if __name__ == "__main__":
    with open("output_ast/train/74.json", encoding="utf8") as f:
        data = json.load(f)
    vec = bag_of_nodes_factorization(data["ast"])
    print(f"Вектор длиной {len(vec)}, ненулевых: {sum(1 for x in vec if x)}")