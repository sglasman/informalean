from pathlib import Path


def processed_statements_dir():
    path = Path("data/statements/processed")
    path.mkdir(parents=True, exist_ok=True)
    return path


def tfidf_svd_statements_path():
    return processed_statements_dir() / "tfidf_svd.npy"


def faiss_index_path():
    return processed_statements_dir() / "faiss_index.faiss"


def preprocessed_statements_path():
    return processed_statements_dir() / "preprocessed_statements.jsonl"

def processed_statements_path():
    return processed_statements_dir() / "processed_statements.jsonl"
