import json
from ast_factorizators.bag_of_nodes import bag_of_nodes
from ast_factorizators.wl_hash import wl_hash


def load_ast(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ast = load_ast("src/raw_ast/HelloWorld.ast.json")

    bon = bag_of_nodes(ast)
    print("Bag-of-Nodes:")
    print(bon)

    wl = wl_hash(ast)
    print("\nWL Hash:")
    print(wl)


if __name__ == "__main__":
    main()
