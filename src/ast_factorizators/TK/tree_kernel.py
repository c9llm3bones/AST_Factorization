from typing import Dict, Any, List
from collections import defaultdict
import hashlib


class TreeNode:
    """Wrapper for AST node."""
    
    def __init__(self, node_type: str, children: List['TreeNode'] = None, value: str = None):
        self.node_type = node_type
        self.children = children or []
        self.value = value
        self._hash = None
    
    def __hash__(self):
        if self._hash is None:
            h = hashlib.md5()
            h.update(self.node_type.encode())
            if self.value:
                h.update(self.value.encode())
            for child in self.children:
                h.update(str(hash(child)).encode())
            self._hash = int(h.hexdigest()[:16], 16)
        return self._hash
    
    def __eq__(self, other):
        if not isinstance(other, TreeNode):
            return False
        return (self.node_type == other.node_type and 
                self.value == other.value and
                len(self.children) == len(other.children))


def ast_to_tree_node(ast: Dict[str, Any]) -> TreeNode:
    """Convert AST dict to TreeNode."""
    if not isinstance(ast, dict):
        return None
    
    node_type = ast.get("type", "unknown")
    
    value = None
    if "value" in ast:
        val = ast["value"]
        if isinstance(val, dict) and "text" in val:
            value = val["text"]
        elif isinstance(val, str):
            value = val
    
    children = []
    for child in ast.get("children", []):
        child_node = ast_to_tree_node(child)
        if child_node:
            children.append(child_node)
    
    return TreeNode(node_type, children, value)


def extract_all_subtrees(node: TreeNode, max_depth: int = 5) -> List[int]:
    """Extract all subtree hashes up to max_depth."""
    subtrees = []
    
    def extract(n: TreeNode, depth: int):
        if depth > max_depth or n is None:
            return
        
        subtrees.append(hash(n))
        
        for child in n.children:
            extract(child, depth + 1)
    
    extract(node, 0)
    return subtrees


def tree_kernel_factorization(ast: Dict[str, Any], max_depth: int = 5) -> List[int]:
    """
    Tree Kernel factorization: extract subtree patterns.
    
    Returns:
        List of subtree hashes (can be converted to BON-like vector)
    """
    tree = ast_to_tree_node(ast)
    if tree is None:
        return []
    
    subtrees = extract_all_subtrees(tree, max_depth=max_depth)
    
    from collections import Counter
    subtree_counts = Counter(subtrees)
    
    vec = []
    for hash_val in sorted(subtree_counts.keys()):
        vec.append(subtree_counts[hash_val])
    
    return vec


def tree_kernel_similarity(ast1: Dict[str, Any], ast2: Dict[str, Any], 
                          max_depth: int = 5) -> float:
    """
    Compute tree kernel similarity using subtree overlap.
    
    Args:
        ast1, ast2: AST dictionaries
        max_depth: Maximum depth for subtrees
    
    Returns:
        Normalized similarity [0, 1]
    """
    tree1 = ast_to_tree_node(ast1)
    tree2 = ast_to_tree_node(ast2)
    
    if tree1 is None or tree2 is None:
        return 0.0
    
    subtrees1 = set(extract_all_subtrees(tree1, max_depth))
    subtrees2 = set(extract_all_subtrees(tree2, max_depth))
    
    if len(subtrees1) == 0 or len(subtrees2) == 0:
        return 0.0
    
    intersection = len(subtrees1 & subtrees2)
    union = len(subtrees1 | subtrees2)
    
    return intersection / union if union > 0 else 0.0


_TK_VOCAB = None

def load_tk_vocab(vocab_path: str):
    """Load tree kernel vocabulary."""
    global _TK_VOCAB
    import json
    with open(vocab_path, 'r', encoding='utf8') as f:
        _TK_VOCAB = json.load(f)
    print(f"Loaded Tree Kernel vocab: {len(_TK_VOCAB)} subtrees")


def tree_kernel_factorization_with_vocab(ast: Dict[str, Any], max_depth: int = 5) -> List[int]:
    """
    Tree kernel factorization using vocabulary.
    
    Returns:
        Fixed-size vector based on vocabulary
    """
    if _TK_VOCAB is None:
        raise ValueError("Tree Kernel vocabulary not loaded. Call load_tk_vocab() first.")
    
    tree = ast_to_tree_node(ast)
    if tree is None:
        return [0] * len(_TK_VOCAB)
    
    subtrees = extract_all_subtrees(tree, max_depth=max_depth)
    
    from collections import Counter
    subtree_counts = Counter(subtrees)
    
    vec = [0] * len(_TK_VOCAB)
    for subtree_hash, count in subtree_counts.items():
        hash_str = str(subtree_hash)
        if hash_str in _TK_VOCAB:
            idx = _TK_VOCAB[hash_str]
            vec[idx] = count
    
    return vec
