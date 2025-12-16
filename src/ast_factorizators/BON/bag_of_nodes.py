from collections import Counter
from typing import Dict, Any, List
import json

with open("src/ast_factorizators/BON/node_types_in_dataset.txt", encoding="utf8") as f:
    KNOWN_TYPES = [line.strip() for line in f if line.strip()]

MAX_VARS = 100
MAX_STRS = 50
MAX_NUMS = 50

TYPE_TO_IDX = {}
idx = 0

for t in KNOWN_TYPES:
    if t not in ["identifier", "string_literal", "decimal_integer_literal"]:
        TYPE_TO_IDX[t] = idx
        idx += 1

for i in range(MAX_VARS):
    TYPE_TO_IDX[f"identifier:VAR_{i}"] = idx
    idx += 1

for i in range(MAX_STRS):
    TYPE_TO_IDX[f"string_literal:STR_{i}"] = idx
    idx += 1

for i in range(MAX_NUMS):
    TYPE_TO_IDX[f"decimal_integer_literal:NUM_{i}"] = idx
    idx += 1

VOCAB_SIZE = idx


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
        
        if typ in ["identifier", "string_literal", "decimal_integer_literal", 
                   "hex_integer_literal", "octal_integer_literal", 
                   "binary_integer_literal", "decimal_floating_point_literal"]:
            
            value = node.get("value")
            if isinstance(value, dict):
                text = value.get("text", "")
            elif isinstance(value, str):
                text = value
            else:
                text = ""
            
            key = f"{typ}:{text}"
            
            if key in TYPE_TO_IDX:
                counter[key] += 1
            elif typ in TYPE_TO_IDX:
                counter[typ] += 1
        else:
            if typ in TYPE_TO_IDX:
                counter[typ] += 1
    
    vec = [0] * VOCAB_SIZE
    for key, cnt in counter.items():
        if key in TYPE_TO_IDX:
            vec[TYPE_TO_IDX[key]] = cnt
    
    return vec

