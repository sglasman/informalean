import hashlib
import json
import logging
import random
from collections import Counter

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tokenizers import Tokenizer

from informalean.common import dependencies
from informalean.config import DataConfig, TrainConfig
from informalean.data.datasets import HERALD_STATEMENTS
from informalean.files import (
    nearest_neighbors_path,
    processed_statements_final_path,
    processed_statements_step_1_path,
    processed_statements_step_2_path,
)
from informalean.data.minhash_lsh import minhash_lsh
from informalean.data.statements.schemas import (
    raw_herald_statements_features,
)
from informalean.util.language_helpers import chinese_char_fraction
from informalean.util.string_helpers import (
    close_brackets,
    declaration_words,
    open_brackets,
)
from informalean.util.union_find import UnionFind

logger = logging.getLogger(__name__)


# Main processing


def load_processed_statements(data_config: DataConfig) -> DatasetDict:
    if processed_statements_final_path().exists():
        return load_from_disk(processed_statements_final_path())
    else:
        return process_statements(data_config, from_step=2)


def process_statements(
    data_config: DataConfig, train_config: TrainConfig, from_step: int = 0
) -> DatasetDict:
    if from_step <= 0:
        raw = _load_raw_statements()
        english_statements = _filter_to_english(raw)
        normalized_statements = _normalize_statements(english_statements)
        with_weights = _add_weight(normalized_statements)

        # Grouping
        nearest_neighbors = _generate_nearest_neigbors(
            with_weights, data_config
        )
        union_find = _union_find(nearest_neighbors)
        with_groups = _group_statements(with_weights, union_find)
        with_groups.save_to_disk(processed_statements_step_1_path())
    else:
        if (
            not processed_statements_step_1_path().exists()
            or not nearest_neighbors_path(data_config.minhash_lsh_threshold).exists()
        ):
            logger.warning(
                "Failed to find saved step 1 processed statements or saved nearest neighbors. Running from step 0."
            )
            return process_statements(data_config, train_config, from_step=0)
        else:
            with_groups = load_from_disk(processed_statements_step_1_path())
            nearest_neighbors = _load_nearest_neighbors(data_config)
            union_find = _union_find(nearest_neighbors)
    if from_step <= 1:
        conversational = _convert_to_conversational(with_groups)
        filtered_for_length = _filter_for_length(
            conversational,
            tokenizer=dependencies.tokenizer(train_config.model_name),
            data_config=data_config,
        )
        logger.info(
            f"Filtered out {len(conversational) - len(filtered_for_length)} statements exceeding max token length."
        )
        filtered_for_length.save_to_disk(processed_statements_step_2_path())
    else:
        if not processed_statements_step_2_path().exists():
            logger.warning(
                "Failed to find saved step 2 processed statements. Running from step 1."
            )
            return process_statements(data_config, train_config, from_step=1)
        else:
            filtered_for_length = load_from_disk(processed_statements_step_2_path())
    if from_step <= 2:
        splits = _generate_splits(
            filtered_for_length, union_find, train_ratio=0.8, val_ratio=0.1
        )
        splits.save_to_disk(processed_statements_final_path())
        return splits


def _load_raw_statements() -> Dataset:
    return load_dataset(
        path=HERALD_STATEMENTS, features=raw_herald_statements_features
    )["train"]


def _filter_to_english(raw: Dataset) -> Dataset:
    # Some statements are in Chinese
    return raw.filter(
        lambda r: chinese_char_fraction(r["informal_statement"]) < 0.2,
        desc="Filtering out non-English statements",
    )


def _normalize_statements(english_statements: Dataset) -> Dataset:
    return (
        english_statements.map(_normalize_statement, desc="Normalizing statements")
        .filter(lambda r: r["keep"], desc="Filtering out a few bad statements")
        .remove_columns(["formal_statement", "keep"])
        .map(_add_hash, desc="Adding hashes")
    )


def _normalize_statement(example):
    filtered_lines = [
        line
        for line in example["formal_statement"].splitlines()
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
    split = stripped_statement.split()
    declaration_index = 0
    example["keep"] = True
    try:
        declaration_index = next(
            i for i, word in enumerate(split) if word in declaration_words
        )
    except StopIteration:
        # 6 examples in the dataset have formatting issues preventing this parsing approach from working. We throw them out.
        example["keep"] = False

    theorem_name, offset_theorem_name_index = _find_theorem_name(
        split[declaration_index + 1 :]
    )
    example["theorem_name"] = theorem_name
    # Remove theorem name from statement - we want to learn to interpret statements without getting a clue from the theorem name
    if offset_theorem_name_index is not None:
        del split[declaration_index + 1 + offset_theorem_name_index]
    example["normalized_formal_statement"] = " ".join(split)
    return example


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


def _load_nearest_neighbors(data_config: DataConfig) -> list[list[int]]:
    threshold = data_config.minhash_lsh_threshold
    if nearest_neighbors_path(threshold).exists():
        return json.loads(nearest_neighbors_path(threshold).read_text())
    else:
        raise Exception("Nearest neighbors file not found")


def _generate_nearest_neigbors(
    dataset: Dataset, data_config: DataConfig
) -> list[list[int]]:
    nearest_neighbors = minhash_lsh(dataset["normalized_formal_statement"], data_config)
    nearest_neighbors_path(data_config.minhash_lsh_threshold).write_text(
        json.dumps(nearest_neighbors)
    )
    return nearest_neighbors


def _union_find(neighbors: list[list[int]]) -> UnionFind:
    union_find = UnionFind(len(neighbors))
    union_find.run(lambda i: [j for j in neighbors[i] if j != i])
    return union_find


def _group_statements(dataset: Dataset, union_find: UnionFind):
    return dataset.map(
        lambda row, i: {"group_id": union_find.find(i), **row},
        with_indices=True,
        desc="Adding group IDs",
    )


def _convert_to_conversational(dataset: Dataset) -> Dataset:
    return dataset.map(
        _add_conversational,
        remove_columns=["normalized_formal_statement", "informal_statement"],
        desc="Converting to conversational prompt completion for training",
    )


def _add_conversational(example):
    example["prompt"] = [
        {
            "role": "system",
            "content": "You are translating Lean statements to precise mathematical English. Do not add assumptions or drop hypotheses.",
        },
        {
            "role": "user",
            "content": f"FORMAL:\n{example['normalized_formal_statement']}",
        },
    ]
    example["completion"] = [
        {
            "role": "assistant",
            "content": f"INFORMAL:\n{example['informal_statement']}",
        }
    ]
    return example


def _filter_for_length(
    conversational: Dataset, tokenizer: Tokenizer, data_config
) -> Dataset:
    return (
        conversational.map(
            lambda b: _len_tokenize_batch(b, tokenizer),
            batched=True,
            desc="Adding tokenized lengths",
        )
        .filter(
            lambda e: e["tokenized_length"] <= data_config.max_tokenized_length,
            desc="Filtering out overlong examples",
        )
        .remove_columns("tokenized_length")
    )


def _len_tokenize_batch(batch, tokenizer: Tokenizer):
    chat_formatted = [
        tokenizer.apply_chat_template(
            p + c, add_generation_prompt=False, truncation=False, tokenize=False
        )
        for p, c in zip(batch["prompt"], batch["completion"])
    ]
    return {
        "tokenized_length": [
            len(t) for t in tokenizer(chat_formatted, padding=False)["input_ids"]
        ]
    }


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

    return DatasetDict(
        {
            "train": with_groups.select(train_indices),
            "val": with_groups.select(val_indices),
            "test": with_groups.select(test_indices),
        }
    )


# There are many examples in the dataset with identical formal statement. Add duplicate count.


def _add_weight(dataset: Dataset) -> Dataset:
    hash_counts = Counter(dataset["hash"])
    return dataset.map(
        lambda row: {"weight": 1.0 / hash_counts[row["hash"]], **row},
        desc="Adding weights due to duplicates",
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
