from pathlib import Path


def processed_statements_dir():
    path = Path("data/statements/processed")
    path.mkdir(parents=True, exist_ok=True)
    return path


def nearest_neighbors_path(threshold: float):
    return processed_statements_dir() / f"nearest_neighbors_{threshold}.json"

def preprocessed_statements_path():
    return processed_statements_dir() / "preprocessed_statements.jsonl"

def processed_statements_path():
    return processed_statements_dir() / "processed_statements"
