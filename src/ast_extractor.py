import json
import os
import csv
from tree_sitter import Language, Parser
import tree_sitter_java

JAVA_LANGUAGE = Language(tree_sitter_java.language())
parser = Parser()
parser.language = JAVA_LANGUAGE

def node_to_dict(node, source_bytes, max_depth=100, depth=0):
    if depth > max_depth:
        return {"type": "MaxDepthExceeded"}

    text = None
    if node.child_count == 0 and node.type not in {"{", "}", "(", ")", ";", "."}:
        try:
            text = source_bytes[node.start_byte:node.end_byte].decode('utf8')
        except:
            text = "<decode_error>"

    children = []
    for child in node.children:
        child_dict = node_to_dict(child, source_bytes, max_depth, depth + 1)
        if child_dict is not None and child_dict != {}:
            children.append(child_dict)

    result = {"type": node.type}
    if children:
        result["children"] = children
    if text is not None:
        result["value"] = {"type": node.type, "text": text}

    return result

def parse_method_full(code: str):
    try:
        wrapped = f"class Dummy {{ {code} }}"
        source_bytes = wrapped.encode('utf8')
        tree = parser.parse(source_bytes)
        root = tree.root_node

        class_decl = None
        for child in root.children:
            if child.type == 'class_declaration':
                class_decl = child
                break
        if not class_decl:
            return None

        class_body = None
        for child in class_decl.children:
            if child.type == 'class_body':
                class_body = child
                break
        if not class_body or not class_body.children:
            return None

        method_node = None
        for child in class_body.children:
            if child.type not in ['{', '}']:
                method_node = child
                break
        if method_node is None:
            return None

        return node_to_dict(method_node, source_bytes)

    except Exception as e:
        return None

def jsonl_to_ast(input_jsonl, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    processed, failed = 0, 0
    failed_rows = []

    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            id1, id2 = obj['id1'], obj['id2']
            func1, func2 = obj['func1'], obj['func2']

            for func, id_ in [(func1, id1), (func2, id2)]:
                out_path = os.path.join(output_dir, f"{id_}.json")
                if os.path.exists(out_path):
                    continue

                ast = parse_method_full(func)
                try:
                    with open(out_path, 'w', encoding='utf-8') as out_f:
                        json.dump({"id": id_, "ast": ast}, out_f, ensure_ascii=False, indent=2)
                    if ast is not None:
                        processed += 1
                    else:
                        failed += 1
                        failed_rows.append({"id": id_, "func": func})
                except Exception as e:
                    failed += 1
                    failed_rows.append({"id": id_, "func": func})

            if (processed + failed) % 1000 == 0:
                print(f"Processed: {processed} Failed: {failed}")

    if failed_rows:
        csv_path = os.path.join(output_dir, "failed_methods.csv")
        with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["id", "func"])
            writer.writeheader()
            writer.writerows(failed_rows)

    print(f"DONE. Total processed: {processed} Failed: {failed}")


inputs = ["test", "train", "val"]


for input in inputs:
    output_dir = "output_ast" + '/' + input
    file = 'data' + '/' + input + '.jsonl'
    jsonl_to_ast(file, output_dir)
    print('succesfully parsed all files for ', input)