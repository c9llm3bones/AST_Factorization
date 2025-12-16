import os
import json
from typing import Dict, Any, Set

def normalize(ast: dict, normalize_literals: bool = True) -> dict:

    # Whitelist of identifiers that should NOT be normalized
    SAFE_IDENTIFIERS = {
        "Object", "String", "System", "out", "println", "print", "Integer",
        "Exception", "IOException", "RuntimeException", "Error",
        "Math", "Collections", "Arrays", "List", "Map", "Set",
        "MessageDigest", "getInstance", "digest", "update", "SHA", "MD5",
        "BASE64Encoder", "encode", "Cipher",
        "Level", "SEVERE", "BootSecurityManager", "securityLogger", "log",
        "File", "Scanner", "PrintWriter", "BufferedReader", "InputStreamReader",
        "FileReader", "FileWriter", "InputStream", "OutputStream",
        "URL", "openStream", "openConnection", "getContentLength", "getInputStream",
        "toString", "getBytes", "length", "append", "close", "readLine", 
        "renameTo", "equals", "hashCode", "compareTo", "clone",
        "add", "remove", "get", "put", "contains", "size", "isEmpty",
        "this", "super", "class",
        "true", "false", "null"
    }

    local_vars: Set[str] = set()
    
    def collect_locals(node):
        if isinstance(node, dict):
            node_type = node.get("type")
            
            if node_type == "formal_parameters":
                for ch in node.get("children", []):
                    if isinstance(ch, dict) and ch.get("type") == "formal_parameter":
                        for name in _extract_identifier_names(ch):
                            local_vars.add(name)
            
            elif node_type == "local_variable_declaration":
                for ch in node.get("children", []):
                    if isinstance(ch, dict) and ch.get("type") == "variable_declarator":
                        for name in _extract_identifier_names(ch):
                            local_vars.add(name)
            
            elif node_type == "enhanced_for_statement":
                for ch in node.get("children", []):
                    if isinstance(ch, dict):
                        if ch.get("type") in ["local_variable_declaration", "variable_declarator"]:
                            for name in _extract_identifier_names(ch):
                                local_vars.add(name)
            
            elif node_type == "for_statement":
                for ch in node.get("children", []):
                    if isinstance(ch, dict) and ch.get("type") == "local_variable_declaration":
                        for name in _extract_identifier_names(ch):
                            local_vars.add(name)
            
            elif node_type == "catch_clause":
                for ch in node.get("children", []):
                    if isinstance(ch, dict) and ch.get("type") == "catch_formal_parameter":
                        for name in _extract_identifier_names(ch):
                            local_vars.add(name)
            
            elif node_type == "resource_specification":
                for ch in node.get("children", []):
                    if isinstance(ch, dict) and ch.get("type") == "resource":
                        for name in _extract_identifier_names(ch):
                            local_vars.add(name)
            
            for value in node.values():
                if isinstance(value, (dict, list)):
                    collect_locals(value)
                    
        elif isinstance(node, list):
            for item in node:
                collect_locals(item)

    def _extract_identifier_names(node):
        if isinstance(node, dict):
            if node.get("type") == "identifier" and "value" in node:
                val = node["value"]
                if isinstance(val, dict) and "text" in val:
                    name = val["text"]
                    if isinstance(name, str):
                        yield name
                elif isinstance(val, str):
                    yield val
            
            for value in node.values():
                if isinstance(value, (dict, list)):
                    yield from _extract_identifier_names(value)
                    
        elif isinstance(node, list):
            for item in node:
                yield from _extract_identifier_names(item)

    collect_locals(ast)
    
    var_map: Dict[str, str] = {}
    var_counter = 0
    
    str_map: Dict[str, str] = {}
    str_counter = 0
    
    num_map: Dict[str, str] = {}
    num_counter = 0

    def replace_locals(node):
        nonlocal var_counter, str_counter, num_counter
        
        if isinstance(node, dict):
            node_type = node.get("type")
            
            if node_type == "identifier" and "value" in node:
                val = node["value"]
                
                if isinstance(val, dict) and "text" in val and "type" in val:
                    name = val["text"]
                    if isinstance(name, str) and name in local_vars and name not in SAFE_IDENTIFIERS:
                        if name not in var_map:
                            var_map[name] = f"VAR_{var_counter}"
                            var_counter += 1
                        node["value"] = {"type": val["type"], "text": var_map[name]}
                        
                elif isinstance(val, str):
                    name = val
                    if name in local_vars and name not in SAFE_IDENTIFIERS:
                        if name not in var_map:
                            var_map[name] = f"VAR_{var_counter}"
                            var_counter += 1
                        node["value"] = {"type": "identifier", "text": var_map[name]}
            
            elif normalize_literals and node_type == "string_literal" and "value" in node:
                val = node["value"]
                if isinstance(val, dict) and "text" in val:
                    original_text = val["text"]
                    if original_text and len(original_text) > 2:
                        if original_text not in str_map:
                            str_map[original_text] = f"STR_{str_counter}"
                            str_counter += 1
                        node["value"] = {"type": val.get("type", "string_literal"), 
                                        "text": f'"{str_map[original_text]}"'}
                elif isinstance(val, str) and val and len(val) > 2:
                    if val not in str_map:
                        str_map[val] = f"STR_{str_counter}"
                        str_counter += 1
                    node["value"] = f'"{str_map[val]}"'
            
            elif normalize_literals and node_type in ["decimal_integer_literal", "hex_integer_literal", 
                                                       "octal_integer_literal", "binary_integer_literal",
                                                       "decimal_floating_point_literal"] and "value" in node:
                val = node["value"]
                if isinstance(val, dict) and "text" in val:
                    original_text = val["text"]
                    if original_text not in ["0", "1", "-1", "0.0", "1.0"]:
                        if original_text not in num_map:
                            num_map[original_text] = f"NUM_{num_counter}"
                            num_counter += 1
                        node["value"] = {"type": val.get("type", node_type), 
                                        "text": num_map[original_text]}
                elif isinstance(val, str) and val not in ["0", "1", "-1", "0.0", "1.0"]:
                    if val not in num_map:
                        num_map[val] = f"NUM_{num_counter}"
                        num_counter += 1
                    node["value"] = num_map[val]
            
            for value in node.values():
                if isinstance(value, (dict, list)):
                    replace_locals(value)
                    
        elif isinstance(node, list):
            for item in node:
                replace_locals(item)

    replace_locals(ast)
    return ast


def normalize_and_save(input_dir: str, output_dir: str, normalize_literals: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    
    total = 0
    failed = 0
    skipped = 0

    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    total_files = len(files)
    
    print(f"Found {total_files} JSON files to process")

    for i, filename in enumerate(files, 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            with open(input_path, "r", encoding="utf8") as f:
                data = json.load(f)

            ast = data.get("ast")
            
            if ast is None:
                skipped += 1
                with open(output_path, "w", encoding="utf8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                normalized_ast = normalize(ast, normalize_literals=normalize_literals)
                data["ast"] = normalized_ast
                
                with open(output_path, "w", encoding="utf8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                total += 1

        except Exception as e:
            print(f"Error {input_path}: {e}")
            failed += 1

        if i % 1000 == 0:
            print(f"Progress: {i}/{total_files} | Processed: {total} | Skipped: {skipped} | Failed: {failed}")

    print(f"Normalization done")
    print(f"Succeed: {total}")
    print(f"Skipped (no AST): {skipped}")
    print(f"Errors: {failed}")


if __name__ == "__main__":
    splits = ["train", "val", "test"]
    
    NORMALIZE_LITERALS = True
    
    for split in splits:
        input_dir = f"output_ast/{split}"
        output_dir = f"output_ast_normalized/{split}"
        
        if not os.path.exists(input_dir):
            print(f"Warning: Directory {input_dir} not found, skipping...")
            continue
        
        print(f"Normalization: {split}")
        
        normalize_and_save(input_dir, output_dir, normalize_literals=NORMALIZE_LITERALS)