from pathlib import Path


def processed_statements_dir():
    path = Path("data/statements/processed")
    path.mkdir(parents=True, exist_ok=True)
    return path


def nearest_neighbors_path(threshold: float):
    return processed_statements_dir() / f"nearest_neighbors_{threshold}.json"

def processed_statements_step_1_path():
    return processed_statements_dir() / "processed_statements_step_1"

def processed_statements_step_2_path():
    return processed_statements_dir() / "processed_statements_step_2"

def processed_statements_final_path():
    return processed_statements_dir() / "processed_statements_final"

statements_models_path = Path("artifacts/statements")