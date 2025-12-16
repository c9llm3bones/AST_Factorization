import gzip
import json
import os

def serialize_ast(ast_dict: dict) -> bytes:
    """Serialize AST dict to compact, deterministic bytes."""
    return json.dumps(
        ast_dict,
        sort_keys=True,
        separators=(',', ':'),  # no whitespace
        ensure_ascii=False
    ).encode('utf-8')

def compressed_len(data: bytes) -> int:
    """Return length of gzipped data."""
    return len(gzip.compress(data))

def ncd_similarity(ast1: dict, ast2: dict) -> float:
    """Compute 1 - NCD as similarity score."""
    s1 = serialize_ast(ast1)
    s2 = serialize_ast(ast2)
    s12 = s1 + s2

    c1 = compressed_len(s1)
    c2 = compressed_len(s2)
    c12 = compressed_len(s12)

    if max(c1, c2) == 0:
        return 1.0

    ncd = (c12 - min(c1, c2)) / max(c1, c2)
    similarity = 1.0 - ncd
    # Clamp to [0, 1] for safety
    return max(0.0, min(1.0, similarity))