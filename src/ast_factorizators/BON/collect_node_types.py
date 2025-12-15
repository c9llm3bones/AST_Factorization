import os
import json
from typing import Set

def collect_types_from_ast(ast: dict, types_set: Set[str]):
    if isinstance(ast, dict):
        if "type" in ast:
            types_set.add(ast["type"])
        for key, value in ast.items():
            if key == "children" and isinstance(value, list):
                for child in value:
                    collect_types_from_ast(child, types_set)
            elif isinstance(value, (dict, list)):
                collect_types_from_ast(value, types_set)
    elif isinstance(ast, list):
        for item in ast:
            collect_types_from_ast(item, types_set)

def collect_all_node_types(data_dirs):
    all_types = set()
    
    for data_dir in data_dirs:
        print(f"Scanning: {data_dir}")
        for filename in os.listdir(data_dir):
            if filename.endswith(".json"):
                path = os.path.join(data_dir, filename)
                try:
                    with open(path, "r", encoding="utf8") as f:
                        obj = json.load(f)
                        ast = obj.get("ast")
                        if ast:
                            collect_types_from_ast(ast, all_types)
                except Exception as e:
                    continue  
    
    return sorted(all_types)

data_dirs = ["output_ast/train", "output_ast/test", "output_ast/val"]
node_types = collect_all_node_types(data_dirs)

with open("src/ast_factorizators/BON/node_types_in_dataset.txt", "w", encoding="utf8") as f:
    for t in node_types:
        f.write(t + "\n")

print(f"Found {len(node_types)} unique node types.")
print("List saved to: node_types_in_dataset.txt")
print("\nExamples:")
for t in sorted(node_types)[:20]:
    print(f" {t}")