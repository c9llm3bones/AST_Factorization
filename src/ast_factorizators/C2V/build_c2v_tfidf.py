from pathlib import Path
from src.ast_factorizators.C2V.code2vec import build_tfidf_vectorizer, save_tfidf_vectorizer

train_files = list(Path("output_ast_normalized/train").glob("*.json"))
vectorizer = build_tfidf_vectorizer(train_files, max_path_length=10)
save_tfidf_vectorizer(vectorizer, "src/ast_factorizators/C2V/c2v_tfidf.pkl")