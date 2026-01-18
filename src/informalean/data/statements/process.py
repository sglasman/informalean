from collections import Counter
import duckdb
import hashlib
from itertools import takewhile
import logging
import random
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from informalean.config import DataConfig
from informalean.data.minhash_lsh import minhash_lsh
from informalean.data.files import (
    preprocessed_statements_path,
    processed_statements_path,
    nearest_neighbors_path,
)
from informalean.data.statements.schemas import (
    raw_herald_statements_features,
    preprocessed_herald_statements_features,
)
from informalean.data.datasets import HERALD_STATEMENTS
import numpy as np
from informalean.util.language_helpers import chinese_char_fraction
from informalean.util.string_helpers import (
    open_brackets,
    close_brackets,
    declaration_words,
)
from informalean.util.union_find import UnionFind
import json

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


def load_processed_statements(data_config: DataConfig) -> DatasetDict:
    if not processed_statements_path().exists():
        return _process_preprocessed_statements(data_config)
    else:
        return load_from_disk(processed_statements_path())


# Main processing


def _preprocess_statements() -> Dataset:
    raw = load_raw_statements()
    # Some statements are in Chinese
    english_statements = raw.filter(
        lambda r: chinese_char_fraction(r["informal_statement"]) < 0.2
    )
    result = _agg_formal_statements(
        english_statements.map(_normalize_statement)
        .remove_columns("formal_statement")
        .map(_add_hash)
    )
    result.to_json(preprocessed_statements_path())
    return result


def _process_preprocessed_statements(
    preprocessed_statements: Dataset, data_config: DataConfig
) -> DatasetDict:
    nearest_neighbors = _load_nearest_neighbors(preprocessed_statements, data_config)
    union_find = _union_find(nearest_neighbors)
    with_groups = preprocessed_statements.map(
        lambda row, i: {"group_id": union_find.find(i), **row}, with_indices=True
    )
    splits = _generate_splits(with_groups, union_find, train_ratio = 0.8, val_ratio = 0.1)
    splits.save_to_disk(processed_statements_path())
    return splits


# Auxiliary functions


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
    for line in filtered_lines:
        if in_multiline_comment and line.endswith("-/"):
            in_multiline_comment = False
        elif line.startswith("/--") and line.endswith("-/"):
            continue
        elif line.startswith("/--") and not line.endswith("-/"):
            in_multiline_comment = True
        elif not in_multiline_comment:
            removed_multiline_comments.append(line)

    stripped_statement = (
        " ".join(removed_multiline_comments)
        .strip()
        .removesuffix(":= by sorry")
        .removesuffix(":=  by sorry")
        .removesuffix(":=by sorry")
        .removesuffix(":= sorry")
    )
    if "by sorry" in stripped_statement:
        print(f"Ending bytes: {stripped_statement[-20:].encode()}")
        print(f"Ends with target: {stripped_statement.endswith(':= by sorry')}")
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


def _load_nearest_neighbors(
    preprocessed: Dataset, data_config: DataConfig
) -> list[list[int]]:
    threshold = data_config.minhash_lsh_threshold
    if nearest_neighbors_path(threshold).exists():
        return json.loads(nearest_neighbors_path(threshold).read_text())
    else:
        nearest_neighbors = minhash_lsh(
            preprocessed["normalized_formal_statement"], data_config
        )
        nearest_neighbors_path(threshold).write_text(json.dumps(nearest_neighbors))
        return nearest_neighbors


def _union_find(neighbors: list[list[int]]) -> UnionFind:
    union_find = UnionFind(len(neighbors))
    union_find.run(lambda i: [j for j in neighbors[i] if j != i])
    return union_find


def _generate_splits(
    with_groups: Dataset, uf: UnionFind, train_ratio: float, val_ratio: float
) -> DatasetDict:
    n = uf.n
    group_ids_to_sizes = {group_id: size for size, group_id in uf.top_components(uf.n)}
    group_ids = list(group_ids_to_sizes.keys())
    random.shuffle(group_ids)
    target_train_count = int(n * train_ratio)
    target_val_count = int(n * val_ratio)
    current_train_count = 0
    current_val_count = 0

    group_to_split = {}
    for group_id in group_ids:
        if current_train_count < target_train_count:
            group_to_split[group_id] = "train"
            current_train_count += group_ids_to_sizes[group_id]
        elif current_val_count < target_val_count:
            group_to_split[group_id] = "val"
            current_val_count += group_ids_to_sizes[group_id]
        else:
            group_to_split[group_id] = "test"

    train_indices, val_indices, test_indices = [], [], []
    for i, group_id in enumerate(with_groups["group_id"]):
        split = group_to_split[group_id]
        if split == "train":
            train_indices.append(i)
        elif split == "val":
            val_indices.append(i)
        else:
            test_indices.append(i)

    return DatasetDict({
        "train": with_groups.select(train_indices),
        "val": with_groups.select(val_indices),
        "test": with_groups.select(test_indices),
    })


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
