import os, json, argparse, glob
from src.ast_factorizators.C2V.code2vec import build_path_vocab, save_c2v_vocab

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["raw", "normalized"], required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--max-vocab-size", type=int, default=3000)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--max-path-length", type=int, default=10)
    args = p.parse_args()

    ast_dir = f"output_ast{'' if args.variant=='raw' else '_normalized'}/{args.split}"
    files = glob.glob(os.path.join(ast_dir, "*.json"))
    vocab = build_path_vocab(files, max_vocab_size=args.max_vocab_size,
                             min_frequency=args.min_frequency,
                             max_path_length=args.max_path_length)
    os.makedirs("src/ast_factorizators/C2V", exist_ok=True)
    out_path = f"src/ast_factorizators/C2V/c2v_vocab_{args.variant}.json"
    save_c2v_vocab(out_path, vocab)
    print(f"Saved vocab to {out_path}, size={len(vocab)}")