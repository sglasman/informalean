from collections import Counter
import logging
from datasets import load_dataset, Dataset
from informalean.config import DataConfig
from informalean.data.files import (
    faiss_index_path,
    tfidf_svd_statements_path,
    preprocessed_statements_path,
)
from informalean.data.statements.schemas import raw_herald_statements_features
from informalean.data.datasets import HERALD_STATEMENTS
import numpy as np
import informalean.data.vectorize as vectorize
import faiss
from informalean.data.ann import ann

logger = logging.getLogger(__name__)


def _load_raw_statements() -> Dataset:
    return load_dataset(
        path=HERALD_STATEMENTS, features=raw_herald_statements_features
    )["train"]


def load_preprocessed_statements() -> Dataset:
    if not preprocessed_statements_path().exists():
        return _preprocess_statements()
    else:
        return load_dataset("json", data_files=str(preprocessed_statements_path()))[
            "train"
        ]


def _preprocess_statements() -> Dataset:
    raw = _load_raw_statements()
    result = _agg_formal_statements(
        raw.map(_normalize_statement).remove_columns("formal_statement").map(_add_hash)
    )
    result.to_json(preprocessed_statements_path())
    return result


def _save_tfidf_embeddings() -> None:
    np.save(
        tfidf_svd_statements_path(),
        vectorize.tfidf(load_preprocessed_statements()["normalized_formal_statement"]),
    )


def _load_tfidf_embeddings() -> np.array:
    if not tfidf_svd_statements_path().exists():
        logger.info("Saved TF-IDF embeddings didn't exist; loading them")
        _save_tfidf_embeddings()
    return np.load(tfidf_svd_statements_path())


def _save_faiss_index(data_config: DataConfig) -> None:
    faiss.write_index(
        ann(_load_tfidf_embeddings(), data_config), str(faiss_index_path())
    )


def _load_faiss_index(data_config: DataConfig):
    if not faiss_index_path().exists():
        logger.info("Saved FAISS index didn't exist; loading it")
        _save_faiss_index(data_config)
    return faiss.read_index(str(faiss_index_path()))


# def nearest_neighbors(faiss_index, )


def _normalize_statement(data):
    data["normalized_formal_statement"] = (
        data["formal_statement"]
        .strip()
        .removeprefix("import Mathlib\n")
        .removesuffix(":= sorry")
        .strip()
    )
    return data


def _add_hash(data):
    data["hash"] = hash(data["normalized_formal_statement"])
    return data


# There are some examples in the dataset with identical formal statement. Aggregate these.


def _agg_formal_statements(dataset: Dataset):
    return Dataset(
        dataset.with_format("arrow")[:]
        .group_by("hash")
        .aggregate(
            [
                ("informal_statement", "list"),
                ("id", "list"),
                ("normalized_formal_statement", "one"),
            ]
        )
    )


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
