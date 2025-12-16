import os
import json
from typing import Set, Dict
from collections import Counter

def collect_types_from_ast(ast: dict, types_set: Set[str], type_counts: Dict[str, int] = None):
    if isinstance(ast, dict):
        if "type" in ast:
            node_type = ast["type"]
            types_set.add(node_type)
            if type_counts is not None:
                type_counts[node_type] += 1
        
        for key, value in ast.items():
            if isinstance(value, (dict, list)):
                collect_types_from_ast(value, types_set, type_counts)
                
    elif isinstance(ast, list):
        for item in ast:
            collect_types_from_ast(item, types_set, type_counts)

def collect_all_node_types(data_dirs, collect_statistics=True):
    all_types = set()
    type_counts = Counter() if collect_statistics else None
    
    total_files = 0
    failed_files = 0
    empty_ast_files = 0
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"Warning: Directory not found: {data_dir}")
            continue
            
        print(f"Scanning: {data_dir}")
        files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
        
        for i, filename in enumerate(files, 1):
            path = os.path.join(data_dir, filename)
            try:
                with open(path, "r", encoding="utf8") as f:
                    obj = json.load(f)
                    ast = obj.get("ast")
                    
                    if ast:
                        collect_types_from_ast(ast, all_types, type_counts)
                        total_files += 1
                    else:
                        empty_ast_files += 1
                        
            except json.JSONDecodeError as e:
                print(f"  JSON error in {filename}: {e}")
                failed_files += 1
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                failed_files += 1
            
            if i % 1000 == 0:
                print(f"  Progress: {i}/{len(files)} files")
    
    file_stats = {
        "total_processed": total_files,
        "empty_ast": empty_ast_files,
        "failed": failed_files
    }
    
    return sorted(all_types), type_counts, file_stats

def save_node_types(node_types, type_counts, output_path, top_n=50):
    with open(output_path, "w", encoding="utf8") as f:       
        for t in node_types:
            if type_counts:
                count = type_counts[t]
                f.write(f"{t}\n")
            else:
                f.write(t + "\n")
    
    print(f"Node types list saved to: {output_path}")
    
    if type_counts:
        stats_path = output_path.replace(".txt", "_stats.txt")
        with open(stats_path, "w", encoding="utf8") as f:
            f.write(f"Top {top_n} most frequent node types:\n")
            f.write("="*60 + "\n\n")
            
            for node_type, count in type_counts.most_common(top_n):
                f.write(f"{count:8d}  {node_type}\n")
        
        print(f"Statistics saved to: {stats_path}")

def main():
    data_dirs = [
        "output_ast_normalized/train",
        "output_ast_normalized/test", 
        "output_ast_normalized/val"
    ]
    
    print("Collecting node types from normalized AST files...")

    
    node_types, type_counts, file_stats = collect_all_node_types(
        data_dirs, 
        collect_statistics=True
    )
    
    output_dir = "src/ast_factorizators/BON"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "node_types_in_dataset.txt")
    save_node_types(node_types, type_counts, output_path, top_n=50)
    
    print("SUMMARY")

    print(f"Unique node types found: {len(node_types)}")
    print(f"Files processed successfully: {file_stats['total_processed']}")
    print(f"Files with empty AST: {file_stats['empty_ast']}")
    print(f"Files failed to process: {file_stats['failed']}")
    
    if type_counts:
        total_nodes = sum(type_counts.values())
        print(f"Total nodes counted: {total_nodes:,}")
        print(f"\nTop 10 most frequent node types:")
        for i, (node_type, count) in enumerate(type_counts.most_common(10), 1):
            percentage = (count / total_nodes) * 100
            print(f"{i:2d}. {node_type:30s} {count:8,} ({percentage:5.2f}%)")
    
    print("Sample node types (first 20 alphabetically):")
    for t in node_types[:20]:
        if type_counts:
            print(f"  {t:30s} (count: {type_counts[t]:,})")
        else:
            print(f"  {t}")

if __name__ == "__main__":
    main()