from collections import Counter
from datasets import load_dataset
from informalean.data.statements.schemas import raw_herald_statements_features
from informalean.datasets import HERALD_STATEMENTS
from pathlib import Path
import numpy as np
import informalean.data.vectorize as vectorize


def load_raw_statements():
    return load_dataset(
        path=HERALD_STATEMENTS, features=raw_herald_statements_features
    )["train"]


def process_statements():
    raw = load_raw_statements()
    return raw.map(_strip_statement)

def save_tfidf_embeddings():
    output_dir = Path("data/statements/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "tfidf_svd", vectorize.tfidf(process_statements()["formal_statement"]))
    
# Unused


def _unpack_opens(formal_statement: str) -> tuple[list[str], list[str]]:
    open_namespaces = []
    open_scoped_namespaces = []
    for line in formal_statement.splitlines():
        if line.startswith("open scoped"):
            open_scoped_namespaces.extend(line.removeprefix("open scoped").split())
        elif line.startswith("open"):
            open_namespaces.extend(line.removeprefix("open").split())
    return (open_namespaces, open_scoped_namespaces)


def _add_opens(raw):
    (open_namespaces, open_scoped_namespaces) = _unpack_opens(raw["formal_statement"])
    raw["open_namespaces"] = open_namespaces
    raw["open_scoped_namespaces"] = open_scoped_namespaces
    return raw


def _strip_statement(raw):
    raw["formal_statement"] = (
        raw["formal_statement"]
        .strip()
        .removeprefix("import Mathlib\n")
        .removesuffix(":= sorry")
        .strip()
    )
    return


# Inspection helpers


def _most_frequent_namespaces(data, limit: int):
    counter = Counter()
    i = 0
    for row in data:
        i += 1
        if i % 1000 == 0:
            print(f"Counted {i} rows")
        counter[
            (
                frozenset(row["open_namespaces"]),
                frozenset(row["open_scoped_namespaces"]),
            )
        ] += 1
    return counter.most_common(limit)
