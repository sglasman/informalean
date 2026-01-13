from collections import Counter
import duckdb
import hashlib
from itertools import takewhile
import logging
import random
from datasets import load_dataset, Dataset
from informalean.config import DataConfig
from informalean.data.files import (
    faiss_index_path,
    processed_statements_path,
    tfidf_svd_statements_path,
    preprocessed_statements_path,
)
from informalean.data.statements.schemas import (
    raw_herald_statements_features,
    preprocessed_herald_statements_features,
)
from informalean.data.datasets import HERALD_STATEMENTS
import numpy as np
import informalean.data.vectorize as vectorize
import faiss
from informalean.data.ann import ann
from informalean.util.string_helpers import (
    open_brackets,
    close_brackets,
    declaration_words,
)
from informalean.util.union_find import UnionFind

logger = logging.getLogger(__name__)


def load_raw_statements() -> Dataset:
    return load_dataset(
        path=HERALD_STATEMENTS, features=raw_herald_statements_features
    )["train"]


def load_preprocessed_statements() -> Dataset:
    if not preprocessed_statements_path().exists():
        return _preprocess_statements()
    else:
        return load_dataset(
            "json",
            data_files=str(preprocessed_statements_path()),
            features=preprocessed_herald_statements_features,
        )["train"]


def load_processed_statements(data_config: DataConfig) -> Dataset:
    if not processed_statements_path().exists():
        return _process_preprocessed_statements(data_config)
    else:
        return load_dataset("json", data_files=str(processed_statements_path()))[
            "train"
        ]


# Main processing


def _preprocess_statements() -> Dataset:
    raw = load_raw_statements()
    result = _agg_formal_statements(
        raw.map(_normalize_statement).remove_columns("formal_statement").map(_add_hash)
    )
    result.to_json(preprocessed_statements_path())
    return result


def _process_preprocessed_statements(data_config: DataConfig) -> Dataset:
    with_nearest_neighbors = _with_nearest_neighbors(
        load_preprocessed_statements(), data_config
    )
    return _with_group_ids(with_nearest_neighbors, data_config)


def _with_nearest_neighbors(preprocessed, data_config: DataConfig) -> Dataset:
    distances, indices = _load_faiss_index(data_config).search(
        _load_tfidf_embeddings(), data_config.n_nearest_neighbors
    )
    # The row indices of the index are the same as those of the preprocessed data and the embeddings
    return preprocessed.map(
        lambda row, index: {
            "nearest_neighbors": [
                {
                    "original_index": neighbor_index,
                    "hash": preprocessed[neighbor_index]["hash"],
                    "similarity": float(similarity),
                }
                for neighbor_index, similarity in zip(
                    indices[index, 1:], distances[index, 1:]
                )
            ],
            **row,
        },
        with_indices=True,
    )


def _with_group_ids(with_nearest_neighbors: Dataset, data_config: DataConfig):
    union_find = _run_union_find(with_nearest_neighbors, data_config.group_distance_threshold)
    logger.info(f"Number of components: {union_find.n_components()}")
    result = with_nearest_neighbors.map(
        lambda row, i: {"group_id": union_find.find(i), **row}, with_indices=True
    )
    result.to_json(processed_statements_path())
    return result


def _run_union_find(with_nearest_neighbors: Dataset, threshold: float):
    union_find = UnionFind(len(with_nearest_neighbors))
    uncompressed_tfidf = vectorize.tfidf(load_preprocessed_statements()["normalized_formal_statement"])
    union_find.run(
        lambda i: [
            neighbors["original_index"]
            for neighbors in with_nearest_neighbors[i]["nearest_neighbors"]
            if sparse_similarity(i, neighbors["original_index"], uncompressed_tfidf) > threshold
        ]
    )
    return union_find

def sparse_similarity(i: int, j: int, uncompressed_tfidf) -> float:
    return (uncompressed_tfidf[i] @ uncompressed_tfidf[j].T)[0, 0]

# Auxiliary functions


def _save_tfidf_embeddings() -> None:
    np.save(
        tfidf_svd_statements_path(),
        vectorize.svd_tfidf(load_preprocessed_statements()["normalized_formal_statement"]),
    )


def _load_tfidf_embeddings() -> np.array:
    if not tfidf_svd_statements_path().exists():
        logger.info("Saved TF-IDF embeddings didn't exist; creating them")
        _save_tfidf_embeddings()
    return np.load(tfidf_svd_statements_path())


def _save_faiss_index(data_config: DataConfig) -> None:
    faiss.write_index(
        ann(_load_tfidf_embeddings(), data_config), str(faiss_index_path())
    )


def _load_faiss_index(data_config: DataConfig):
    if not faiss_index_path().exists():
        logger.info("Saved FAISS index didn't exist; creating it")
        _save_faiss_index(data_config)
    index = faiss.read_index(str(faiss_index_path()))
    index.nprobe = data_config.faiss_statement_nprobe
    return index


def _normalize_statement(data):
    filtered_lines = [
        line
        for line in data["formal_statement"].splitlines()
        if not (
            line.startswith("import ")
            or line.startswith("open ")
            or line.startswith("--")
        )
    ]
    in_multiline_comment = False
    removed_multiline_comments = []
    log_after = False
    for line in filtered_lines:
        if in_multiline_comment and line.endswith("-/"):
            in_multiline_comment = False
        elif line.startswith("/--") and line.endswith("-/"):
            continue
        elif line.startswith("/--") and not line.endswith("-/"):
            in_multiline_comment = True
            log_after = True
            logger.info(f"In multiline comment. Id: {data['id']}")
        elif not in_multiline_comment:
            removed_multiline_comments.append(line)

    stripped_statement = (
        " ".join(removed_multiline_comments)
        .removesuffix(":= by sorry")
        .removesuffix(":= sorry")
    )
    if log_after:
        logger.info(stripped_statement)
    split = stripped_statement.split()
    declaration_index = 0
    data["keep"] = True
    try:
        declaration_index = next(
            i for i, word in enumerate(split) if word in declaration_words
        )
    except StopIteration:
        # 6 examples in the dataset have formatting issues preventing this parsing approach from working. We throw them out.
        data["keep"] = False

    theorem_name, offset_theorem_name_index = _find_theorem_name(
        split[declaration_index + 1 :]
    )
    data["theorem_name"] = theorem_name
    # Remove theorem name from statement - we want to learn to interpret statements without getting a clue from the theorem name
    if offset_theorem_name_index is not None:
        del split[declaration_index + 1 + offset_theorem_name_index]
    data["normalized_formal_statement"] = " ".join(split)
    return data


# Find the theorem name - this deserves its own function
def _find_theorem_name(words: list[str]) -> tuple[str, int]:
    # Assumptions:
    # - Brackets of any kind are never nested
    # - A open bracket of one kind is never closed with a close bracket of another kind
    in_brackets = False
    for i, word in enumerate(words):
        if word[0] in open_brackets and word[-1] in close_brackets:
            continue
        elif word[0] in open_brackets:
            in_brackets = True
        elif word[-1] in close_brackets and in_brackets:
            in_brackets = False
        elif not in_brackets and word == ":":
            return "", None
        elif not in_brackets:
            return word, i
        elif in_brackets:
            continue
    return "", None


def _add_hash(data):
    data["hash"] = hashlib.md5(data["normalized_formal_statement"].encode()).hexdigest()
    return data


# There are some examples in the dataset with identical formal statement. Aggregate these.


def _agg_formal_statements(dataset: Dataset):
    arrow_table = dataset.with_format("arrow").data.table
    result = (
        duckdb.sql("""
        SELECT
            hash,
            list(informal_statement) as informal_statements,
            list(id) as ids,
            list(theorem_name) as theorem_names,
            first(normalized_formal_statement) as normalized_formal_statement
        FROM arrow_table
        GROUP BY hash
    """)
        .arrow()
        .read_all()
    )
    return Dataset.from_dict(result.to_pydict())


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
