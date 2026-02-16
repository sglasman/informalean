from informalean.config import Config
from informalean.data.statements.process import load_processed_statements


def val_sorted_by_prompt_length(config: Config):
    val = load_processed_statements(config)["val"]
    val_with_prompt_len = val.map(lambda e: {"prompt_len": len(e["prompt"][1]["content"])})
    return val_with_prompt_len.sort("prompt_len", reverse=True)