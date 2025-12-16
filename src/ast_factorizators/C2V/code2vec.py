from typing import Dict, Any, List, Tuple, Set
from collections import Counter
import json
import pickle
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer

class ASTPathNode:
    """Node in AST path representation."""
    
    def __init__(self, node_type: str, value: str = None):
        self.node_type = node_type
        self.value = value
    
    def __repr__(self):
        if self.value:
            return f"{self.node_type}:{self.value}"
        return self.node_type
    
    def __eq__(self, other):
        return self.node_type == other.node_type and self.value == other.value
    
    def __hash__(self):
        return hash((self.node_type, self.value))


class ASTPath:
    """Path between two terminal nodes in AST."""
    
    def __init__(self, start_terminal: ASTPathNode, 
                 path_nodes: List[Tuple[str, ASTPathNode]],
                 end_terminal: ASTPathNode):
        self.start = start_terminal
        self.path = path_nodes  
        self.end = end_terminal
    
    def to_string(self) -> str:
        """Convert path to string representation."""
        path_str = ""
        for direction, node in self.path:
            path_str += f"{direction}{node}|"
        return f"{self.start}|{path_str}{self.end}"
    
    def __repr__(self):
        return self.to_string()
    
    def __hash__(self):
        return hash(self.to_string())
    
    def __eq__(self, other):
        return self.to_string() == other.to_string()


def is_terminal(node: Dict[str, Any]) -> bool:
    """Check if node is a terminal (leaf with meaningful value)."""
    if not isinstance(node, dict):
        return False
    
    if node.get("children"):
        return False
    
    value = node.get("value")
    if value:
        if isinstance(value, dict) and "text" in value:
            return True
        if isinstance(value, str):
            return True
    
    return False


def get_node_representation(node: Dict[str, Any]) -> ASTPathNode:
    """Convert AST node to ASTPathNode."""
    node_type = node.get("type", "unknown")
    
    value = None
    if "value" in node:
        val = node["value"]
        if isinstance(val, dict) and "text" in val:
            value = val["text"]
        elif isinstance(val, str):
            value = val
    
    return ASTPathNode(node_type, value)


def extract_terminals(ast: Dict[str, Any], path: List[Dict] = None) -> List[Tuple[ASTPathNode, List[Dict]]]:
    """
    Extract all terminal nodes with their paths from root.
    
    Returns:
        List of (terminal_node, path_from_root)
    """
    if path is None:
        path = []
    
    terminals = []
    
    if is_terminal(ast):
        terminal = get_node_representation(ast)
        terminals.append((terminal, path + [ast]))
        return terminals
    
    for child in ast.get("children", []):
        if isinstance(child, dict):
            child_terminals = extract_terminals(child, path + [ast])
            terminals.extend(child_terminals)
    
    return terminals


def find_lca_path(path1: List[Dict], path2: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Find Lowest Common Ancestor and split paths.
    
    Returns:
        (path_up_from_terminal1, path_lca, path_down_to_terminal2)
    """
    lca_idx = 0
    for i in range(min(len(path1), len(path2))):
        if id(path1[i]) == id(path2[i]):
            lca_idx = i
        else:
            break

    path_up = path1[lca_idx+1:][::-1]  
    path_lca = [path1[lca_idx]] if lca_idx < len(path1) else []
    path_down = path2[lca_idx+1:]
    
    return path_up, path_lca, path_down


def extract_ast_paths(ast: Dict[str, Any], max_path_length: int = 8, 
                     max_paths: int = 1000) -> List[ASTPath]:
    """
    Extract all paths between terminal nodes.
    
    Args:
        ast: AST dictionary
        max_path_length: Maximum path length to consider
        max_paths: Maximum number of paths to extract (for efficiency)
    
    Returns:
        List of AST paths
    """
    terminals = extract_terminals(ast)
    
    if len(terminals) < 2:
        return []
    
    paths = []
    
    terminal_pairs = list(combinations(range(len(terminals)), 2))
    if len(terminal_pairs) > max_paths:
        import random
        random.seed(42)
        terminal_pairs = random.sample(terminal_pairs, max_paths)
    
    for i, j in terminal_pairs:
        terminal1, path1 = terminals[i]
        terminal2, path2 = terminals[j]
        
        path_up, path_lca, path_down = find_lca_path(path1, path2)
        
        path_nodes = []
        
        for node in path_up:
            path_nodes.append(("(up)", get_node_representation(node)))
        
        for node in path_lca:
            path_nodes.append(("(dot)", get_node_representation(node)))
        
        for node in path_down:
            path_nodes.append(("(down)", get_node_representation(node)))
        
        if len(path_nodes) > max_path_length:
            continue
        
        ast_path = ASTPath(terminal1, path_nodes, terminal2)
        paths.append(ast_path)
    
    return paths


def ast_path_bag_factorization(ast: Dict[str, Any], path_vocab: Dict[str, int] = None,
                               max_path_length: int = 10) -> List[int]:
    """
    Create bag-of-paths vector for AST using path vocab.
    
    Args:
        ast: AST dictionary
        path_vocab: Dictionary mapping path strings to indices
        max_path_length: Maximum path length
    
    Returns:
        Bag-of-paths vector
    """
    if ast is None:
        return []
    
    paths = extract_ast_paths(ast, max_path_length=max_path_length)
    
    if path_vocab is None:
        path_counter = Counter(path.to_string() for path in paths)
        return list(path_counter.values())
    
    path_counter = Counter(path.to_string() for path in paths)
    vec = [0] * len(path_vocab)
    
    for path_str, count in path_counter.items():
        if path_str in path_vocab:
            vec[path_vocab[path_str]] = 1 # count
    
    return vec


def build_path_vocab(ast_files: List[str], max_vocab_size: int = 10000,
                    min_frequency: int = 3, max_path_length: int = 8) -> Dict[str, int]:
    """
    Build vocabulary of most frequent paths from training data.
    
    Args:
        ast_files: List of paths to AST JSON files
        max_vocab_size: Maximum vocabulary size
        min_frequency: Minimum frequency to include in vocab
        max_path_length: Maximum path length
    
    Returns:
        Dictionary mapping path strings to indices
    """
    path_counter = Counter()
    
    print(f"Building path vocabulary from {len(ast_files)} files...")
    
    for i, file_path in enumerate(ast_files):
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                data = json.load(f)
                ast = data.get("ast")
                
                if ast is None:
                    continue
                
                paths = extract_ast_paths(ast, max_path_length=max_path_length)
                for path in paths:
                    path_counter[path.to_string()] += 1
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(ast_files)} files, "
                      f"found {len(path_counter)} unique paths")
        
        except Exception as e:
            continue
    
    frequent_paths = {path: cnt for path, cnt in path_counter.items() 
                     if cnt >= min_frequency}
    
    sorted_paths = sorted(frequent_paths.items(), key=lambda x: x[1], reverse=True)
    top_paths = sorted_paths[:max_vocab_size]
    
    vocab = {path: idx for idx, (path, _) in enumerate(top_paths)}
    
    print(f"\nVocabulary statistics:")
    print(f"  Total unique paths: {len(path_counter)}")
    print(f"  Paths with freq >= {min_frequency}: {len(frequent_paths)}")
    print(f"  Vocabulary size: {len(vocab)}")
    
    return vocab

_C2V_VOCAB = None
_C2V_TFIDF = None 

def save_c2v_vocab(path: str, vocab: Dict[str, int]) -> None:
    with open(path, "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False)

def load_c2v_vocab(path: str) -> None:
    global _C2V_VOCAB
    with open(path, "r", encoding="utf8") as f:
        _C2V_VOCAB = json.load(f)

def c2v_factorization(ast: Dict[str, Any], max_path_length: int = 8) -> List[int]:
    return ast_path_bag_factorization(ast, path_vocab=_C2V_VOCAB, max_path_length=max_path_length)

def build_tfidf_vectorizer(ast_files: List[str], max_path_length: int = 8, 
                          max_df: float = 0.99, min_df: int = 2) -> TfidfVectorizer:
    """
    Build TfidfVectorizer on AST paths from training files.
    
    Args:
        ast_files: List of paths to AST JSON files
        max_path_length: Maximum path length
        max_df: Ignore terms that appear in more than max_df fraction of documents
        min_df: Ignore terms that appear in fewer than min_df documents
    
    Returns:
        Fitted TfidfVectorizer
    """
    print(f"Building TF-IDF vectorizer from {len(ast_files)} files...")
    
    path_documents = [] 
    
    for i, file_path in enumerate(ast_files):
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                data = json.load(f)
                ast = data.get("ast")
                if ast is None:
                    path_documents.append("")  
                    continue
                
                paths = extract_ast_paths(ast, max_path_length=max_path_length)
                doc = " ".join(path.to_string() for path in paths)
                path_documents.append(doc)
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(ast_files)} files")
        
        except Exception as e:
            path_documents.append("")  
    
    vectorizer = TfidfVectorizer(
        binary=True, 
        max_df=max_df,
        min_df=min_df,
        token_pattern=r'.+', 
        lowercase=False
    )
    vectorizer.fit(path_documents)
    
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    return vectorizer


def save_tfidf_vectorizer(vectorizer: TfidfVectorizer, path: str) -> None:
    """Save TfidfVectorizer to disk (pickle)."""
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)


def load_tfidf_vectorizer(path: str) -> None:
    """Load TfidfVectorizer from disk."""
    global _C2V_TFIDF
    with open(path, "rb") as f:
        _C2V_TFIDF = pickle.load(f)


def ast_path_tfidf_factorization(ast: Dict[str, Any], max_path_length: int = 10) -> List[float]:
    """
    TF-IDF factorization of AST paths.
    
    Returns:
        Dense TF-IDF vector (list of floats)
    """
    global _C2V_TFIDF
    if _C2V_TFIDF is None:
        raise RuntimeError("TF-IDF vectorizer not loaded. Call load_tfidf_vectorizer() first.")
    
    paths = extract_ast_paths(ast, max_path_length=max_path_length)
    doc = " ".join(path.to_string() for path in paths) if paths else ""
    
    sparse_vec = _C2V_TFIDF.transform([doc])
    dense_vec = sparse_vec.toarray()[0].tolist()
    return dense_vec
